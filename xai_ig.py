"""M4: IG of a fixed selected action's Q-value; importable with any 8->5 model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients

FEATURES = ['overload', 'removed', 'sbp', 'dbp', 'hr', 'time', 'age', 'albumin']
SCALE = np.array([4, 4, 200, 120, 150, 1, 100, 5], dtype=np.float32)
ACTIONS = [0, 5, 8, 10, 13]


def baseline_for(state):
    """Zero overload/removal, reference vitals; preserve time, age and albumin.

    This is a reference contrast, not a clinically validated healthy patient.
    Age/albumin attributions are zero by construction for this baseline.
    """
    b = np.array(state, dtype=np.float32, copy=True)
    b[:5] = np.array([0, 0, 120, 80, 75]) / SCALE[:5]
    return b


def explain(model, state, baseline=None, n_steps=128, atol=1e-4, rtol=1e-3):
    """Hold argmax action fixed along the IG path; fail on poor completeness."""
    device = next(model.parameters()).device
    x = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)
    b = torch.as_tensor(baseline_for(state) if baseline is None else baseline,
                        dtype=torch.float32, device=device).reshape(1, -1)
    if x.shape != (1, 8) or b.shape != x.shape or not torch.isfinite(x).all() or not torch.isfinite(b).all():
        raise ValueError('Expected finite normalized state and baseline of shape (8,)')
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            q = model(x)
            if q.shape != (1, 5) or not torch.isfinite(q).all():
                raise ValueError('Model must output five finite Q-values')
            action = int(q.argmax(1).item())
            difference = float((q[0, action] - model(b)[0, action]).item())
        ig = IntegratedGradients(model)
        for steps in (n_steps, n_steps * 2, n_steps * 4):
            attrs, delta = ig.attribute(x, baselines=b, target=action, n_steps=steps,
                                       return_convergence_delta=True)
            residual = float(attrs.sum().item() - difference)
            if np.isfinite(residual) and abs(residual) <= atol + rtol * abs(difference):
                break
        else:
            raise ValueError(f'IG completeness failed: residual={residual:.6g}')
        return dict(action=action, q=float(q[0, action]), q_difference=difference,
                    attributions=attrs.detach().cpu().numpy()[0], residual=residual,
                    captum_delta=float(delta.item()), integration_steps=steps)
    finally:
        model.train(was_training)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(model, env, seeds, output, demo=False, reference=None):
    """Each seed is one simulated session, not necessarily a distinct UCI patient."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, all_attrs, session_means = [], [], []
    label = 'DEMO - untrained network' if demo else 'Trained policy - simulated sessions'
    first = None
    for seed in seeds:
        state = env.reset(seed=seed)
        session_attrs = []
        for step in range(16):
            result = explain(model, state, baseline=reference)
            attrs = result.pop('attributions')
            b = baseline_for(state) if reference is None else reference
            row = dict(seed=seed, step=step, **result)
            row.update({f'state_{k}': float(v) for k, v in zip(FEATURES, state)})
            row.update({f'baseline_{k}': float(v) for k, v in zip(FEATURES, b)})
            row.update({f'ig_{k}': float(v) for k, v in zip(FEATURES, attrs)})
            state, reward, done, info = env.step(result['action'])
            row.update(reward=float(reward), hypotension=bool(info['hypotension']))
            rows.append(row)
            session_attrs.append(attrs)
            if done:
                break
        session_attrs = np.stack(session_attrs)
        all_attrs.extend(session_attrs)
        session_means.append(np.abs(session_attrs).mean(0))
        if first is None:
            first = session_attrs
    values = np.stack(all_attrs)
    means = np.stack(session_means)
    summary = [dict(feature=k, mean_abs_ig=float(means[:, i].mean()),
                    session_std=float(means[:, i].std()), mean_signed_ig=float(values[:, i].mean()))
               for i, k in enumerate(FEATURES)]
    write_csv(output / 'attributions.csv', rows)
    write_csv(output / 'feature_summary.csv', summary)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(FEATURES, first[0], color=['#16847c' if x >= 0 else '#c95e51' for x in first[0]])
    ax.axvline(0, color='black', linewidth=.6)
    ax.set(xlabel='Contribution to selected-action Q-value', title=f'{label}\nFirst decision: {ACTIONS[rows[0]["action"]]} mL/kg/h')
    fig.tight_layout(); fig.savefig(output / 'decision.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    limit = max(float(np.abs(first).max()), 1e-9)
    im = ax.imshow(first.T, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
    ax.set(yticks=range(8), yticklabels=FEATURES, xlabel='Step (0 = session start)', title=f'{label}\nOne session; selected action may change across steps')
    fig.colorbar(im, ax=ax, label='Signed IG'); fig.tight_layout()
    fig.savefig(output / 'session.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(FEATURES, means.mean(0), xerr=means.std(0), color='#16847c')
    ax.set(xlabel='Mean absolute IG (error bars: session standard deviation)', title=f'{label}\n{len(seeds)} simulated sessions')
    fig.tight_layout(); fig.savefig(output / 'cross_session.png', dpi=160); plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--demo', action='store_true')
    mode.add_argument('--model', type=Path, help='TorchScript 8-input/5-output Q-network')
    parser.add_argument('--evidence', type=Path, help='M3 JSON approval bound to model SHA256')
    parser.add_argument('--reference', type=Path, help='JSON list of eight normalized reference values; enables age/albumin contrast')
    parser.add_argument('--sessions', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--output', type=Path, default=Path('results/xai'))
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error('--sessions must be positive')
    digest = None
    if args.demo:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5))
    else:
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if not args.evidence:
            parser.error('Trained explanations require --evidence from M3')
        evidence = json.loads(args.evidence.read_text())
        if (evidence.get('model_sha256') != digest or evidence.get('convergence_passed') is not True
                or evidence.get('stability_passed') is not True
                or sorted(evidence.get('training_seeds', [])) != [0, 1, 2, 3, 4]
                or not evidence.get('report')):
            parser.error('Evidence must match checkpoint and approve convergence/stability over seeds 0..4 with report path')
        report = args.evidence.parent / evidence['report']
        if not report.is_file():
            parser.error('M3 report file does not exist')
        model = torch.jit.load(str(args.model), map_location='cpu')
    from env import DialysisEnv
    reference = json.loads(args.reference.read_text()) if args.reference else None
    seeds = list(range(args.seed, args.seed + args.sessions))
    rows = audit(model, DialysisEnv(), seeds, args.output, args.demo, reference)
    manifest = dict(demo=args.demo, model_sha256=digest, seeds=seeds,
                    baseline=reference if reference is not None else 'matched demographics/time, reference vitals, zero overload/removal',
                    max_abs_residual=max(abs(r['residual']) for r in rows), torch=torch.__version__)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Wrote {len(rows)} explanations to {args.output}; demo={args.demo}')


if __name__ == '__main__':
    main()
