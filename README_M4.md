# M4: Integrated Gradients and presentation handoff

Owner per repository: Mohanad Mostfa Ali. Based on main commit 9058b42a07227ff8ca365cacc5ab79a54ec7bc59.

## Install and run

Copy `xai_ig.py` into the repository root next to `env.py`. Install `requirements-m4.txt` in your Python environment.

```sh
python -m pip install -r requirements-m4.txt
python xai_ig.py --demo --sessions 10 --output results/xai_demo
```

Demo uses a fixed random untrained network. Its pictures demonstrate the software and support no conclusions about learned dialysis behavior.

For a trained policy, M2 exports its online Q-network to TorchScript:

```python
online_q.eval()
traced = torch.jit.trace(online_q.cpu(), torch.zeros(1, 8))
traced.save('models/ddqn_scripted.pt')
```

M3 supplies `results/xai_approval.json` after reviewing the convergence plots and five-seed stability report:

```json
{
  "model_sha256": "REPLACE_WITH_SHA256_OF_EXPORTED_MODEL",
  "convergence_passed": true,
  "stability_passed": true,
  "training_seeds": [0, 1, 2, 3, 4],
  "report": "validation_report.md"
}
```

The report path is relative to the evidence JSON. This checks M3's declaration, not the scientific validity of that report. The hash ties approval to the exact export.

```sh
python xai_ig.py --model models/ddqn_scripted.pt --evidence results/xai_approval.json --sessions 100 --seed 10000 --output results/xai
```

M2 must preserve the existing normalized state order, action index order, and preprocessing. The CLI uses the existing simulator. To explain another PyTorch network directly, import `explain(model, normalized_state, baseline)`.

## Outputs and interpretation

- `decision.png`: signed feature contributions to the chosen action's Q-value at the first step.
- `session.png`: signed attribution heatmap for the first session.
- `cross_session.png`: session-weighted mean absolute contributions with session standard deviations.
- `attributions.csv`: states, actual baselines, selected actions, contributions, completeness residuals, reward and hypotension flag.
- `feature_summary.csv`: aggregate signed and absolute contributions.
- `manifest.json`: mode, model hash, session seeds, baseline and maximum residual.

The action target stays fixed while integrating from the reference to the observed state. A positive attribution increases that Q-value relative to the reference. It does not establish causation or explain why that action beats every alternative. Q-value scale depends on the learned reward.

The default baseline preserves age, albumin and time, with reference vitals and zero overload/removal. Those preserved features have zero IG by construction. Therefore **this run cannot measure reliance on age or albumin**. Run a second audit with an explicit cohort reference to inspect those features:

```sh
python xai_ig.py --model models/ddqn_scripted.pt --evidence results/xai_approval.json --reference reference.json --output results/xai_cohort
```

`reference.json` contains eight normalized numbers. Construct it from the training cohort only, record its provenance, and inspect whether its straight-line paths leave the plausible state region. Compare rankings across both references. Elevated age/albumin attribution is a review prompt, not automatically an error.

Numerical completeness requires `abs(sum(IG) - (Q(x,a)-Q(b,a))) <= 1e-4 + 1e-3*abs(Q difference)`. Integration retries up to four times the requested resolution and fails instead of silently accepting a poor result.

Session seeds 10000 onward avoid the five training RNG seeds, but this does **not** guarantee distinct held-out patient rows. M1/M3 must supply patient-level splits before claiming generalization to unseen patients.

## Known project limitations to discuss

The existing environment synthesizes weight, systolic pressure and serum albumin. It maps urine albumin grade to synthetic serum albumin using a hand-chosen formula. Do not describe these values as measured serum albumin. Physiology coefficients are hand-tuned, so explanations describe this simulator and reward. The eight inputs also omit previous action and baseline parameters used internally. Discuss partial observability with M1/M2 before training.

## Team handoff

M1: freeze normalization/action order and document data assumptions. M2: provide the trained online network export. M3: provide validation report, model-bound approval and held-out patient split. M4: run both baseline audits, inspect clinically relevant patterns, insert actual figures into the presentation, and merge teammates' completed slides.

The final trained results and merged team deck depend on artifacts absent from this repository snapshot. `slides_m4.md` supplies ready-to-present M4 wording and the merge order without inventing results.
