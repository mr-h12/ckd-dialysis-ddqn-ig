"""
Module 2 - Full pipeline, one command (owner: Gomana Hossam Eldine).

Runs every step in the order the project requires:
  1. Train Double DQN (our model) and standard DQN (baseline) on seeds 0-4, save models + logs.
  2. Module 3 (evaluate.py): tabular Q-learning + rule-based baselines, comparison table on held-out
     patients, convergence/stability curves, and the sign-off (xai_approval.json) for step 3.
  3. Only if step 2 approves (and the approval matches the model's SHA-256): Integrated Gradients on
     the seed-0 DDQN model (Module 4 code), twice: default per-state baseline and training-cohort mean.

Outputs:
  models/ddqn_seed<k>.pt, models/dqn_seed<k>.pt, models/ddqn_scripted_seed<k>.pt, models/ddqn_scripted.pt
  results/train_log_<name>_seed<k>.csv     per-episode training log
  results/metrics_table.csv, reward_curves.png, loss_curves.png, validation_report.md, xai_approval.json
                                           (see evaluate.py)
  results/xai/                             IG figures and CSVs (decision, session, cross-session)
  results/xai_cohort/                      same audit against the mean training patient (+ reference.json)

Run:   python main.py            # full run (~2 min on CPU)
       python main.py --quick    # smoke test: 1 seed, few episodes, 2 IG sessions -> results/quick/
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import evaluate                       # Module 3: run()
import train                          # Module 2: train(), save_log()
import xai_ig                         # Module 4: audit()
from env import DialysisEnv

# ---------------------------------------------------------------------------
# Settings (change here)
# ---------------------------------------------------------------------------
FULL = dict(seeds=train.SEEDS, episodes=train.CONFIG["episodes"], ig_sessions=100,
            models=Path("models"), results=Path("results"))
QUICK = dict(seeds=[0], episodes=200, ig_sessions=2,                     # never overwrites the real models
             models=Path("results/quick/models"), results=Path("results/quick"))
AGENTS = {"ddqn": True, "dqn": False}  # name -> double flag (DQN = same code with double=False)
IG_SEED = 10000                        # IG sessions start here, away from training and test seeds


# ---------------------------------------------------------------------------
# Step 1: train both agents on every seed, save exactly like train.py does
# ---------------------------------------------------------------------------
def train_all(cfg):
    for name, double in AGENTS.items():
        for seed in cfg["seeds"]:
            print(f"Training {name.upper()} seed {seed} for {cfg['episodes']} episodes")
            agent, log = train.train(seed, double=double, episodes=cfg["episodes"])
            train.save_log(cfg["results"] / f"train_log_{name}_seed{seed}.csv", log)
            agent.save(cfg["models"] / f"{name}_seed{seed}.pt")
            if double:
                agent.export_scripted(cfg["models"] / f"ddqn_scripted_seed{seed}.pt")
    shutil.copy(cfg["models"] / f"ddqn_scripted_seed{cfg['seeds'][0]}.pt", cfg["models"] / "ddqn_scripted.pt")


# ---------------------------------------------------------------------------
# Step 3: Integrated Gradients on the seed-0 DDQN (only after step 2 approves)
# ---------------------------------------------------------------------------
def cohort_reference():
    """Mean normalized start state over the 320 TRAINING patients (train.py's split, no test patients).
    One env.reset per patient; seed = patient index so the sampled SBP/weight/overload noise is reproducible."""
    train_profiles, _ = train.split_profiles()
    starts = [DialysisEnv(profiles=train_profiles[[i]]).reset(seed=i) for i in range(len(train_profiles))]
    return np.mean(starts, axis=0).astype(np.float32), len(train_profiles)


def explain(cfg):
    model_path = cfg["models"] / "ddqn_scripted.pt"
    # same SHA-256 evidence check as the xai_ig.py CLI: the approval must belong to THIS checkpoint
    digest = xai_ig.check_evidence(model_path, cfg["results"] / "xai_approval.json", cfg["seeds"])
    model = torch.jit.load(str(model_path), map_location="cpu")
    seeds = list(range(IG_SEED, IG_SEED + cfg["ig_sessions"]))

    # 3a. default baseline: zero overload/removal, reference vitals, the state's own time/age/albumin
    out = cfg["results"] / "xai"
    rows = xai_ig.audit(model, DialysisEnv(), seeds, out)
    manifest = dict(note="M3 approved (xai_approval.json)", model=str(model_path), model_sha256=digest,
                    seeds=seeds, max_abs_residual=max(abs(r["residual"]) for r in rows), torch=torch.__version__)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nIG: {len(rows)} explained decisions -> {out} (max completeness residual {manifest['max_abs_residual']:.1e})")

    # 3b. cohort baseline: the mean training patient. Unlike 3a, age and albumin differ from the
    #     reference, so they can receive non-zero attribution (3a gives them 0 by construction).
    ref, n_patients = cohort_reference()
    out = cfg["results"] / "xai_cohort"
    rows = xai_ig.audit(model, DialysisEnv(), seeds, out, reference=ref)
    (out / "reference.json").write_text(json.dumps(dict(
        reference=dict(zip(xai_ig.FEATURES, ref.tolist())), units="normalized state (raw / env.SCALE)",
        source=f"mean env.reset() start state over the {n_patients} training patients of train.split_profiles()",
        reset_seeds=f"0..{n_patients - 1} (one per patient)", model_sha256=digest), indent=2))
    with open(out / "feature_summary.csv") as f:          # top features, as saved by xai_ig.audit
        summary = sorted(csv.DictReader(f), key=lambda r: -float(r["mean_abs_ig"]))
    print(f"IG vs training-cohort mean: {len(rows)} decisions -> {out}; mean |IG| by feature:")
    for r in summary:
        print(f"  {r['feature']:9s} {float(r['mean_abs_ig']):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the whole project: train -> check -> explain")
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 200 episodes, 2 IG sessions")
    cfg = QUICK if parser.parse_args().quick else FULL
    cfg["models"].mkdir(parents=True, exist_ok=True); cfg["results"].mkdir(parents=True, exist_ok=True)

    train_all(cfg)
    # gate: no Integrated Gradients unless Module 3 approved convergence + stability + safety
    if not evaluate.run(cfg):
        sys.exit(f"Stopped before Integrated Gradients: Module 3 did NOT approve the DDQN. "
                 f"See {cfg['results'] / 'validation_report.md'}")
    explain(cfg)
    print("\nDone.")
