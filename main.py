"""
Module 2 - Full pipeline, one command (owner: Gomana Hossam Eldine).

Runs every step in the order the project requires:
  1. Train Double DQN (our model) and standard DQN (baseline) on seeds 0-4, save models + logs.
  2. Module 3 (evaluate.py): tabular Q-learning + rule-based baselines, comparison table on held-out
     patients, convergence/stability curves, and the sign-off (xai_approval.json) for step 3.
  3. Only after step 2: Integrated Gradients on the seed-0 DDQN model (Module 4 code).

Outputs:
  models/ddqn_seed<k>.pt, models/dqn_seed<k>.pt, models/ddqn_scripted_seed<k>.pt, models/ddqn_scripted.pt
  results/train_log_<name>_seed<k>.csv     per-episode training log
  results/metrics_table.csv, reward_curves.png, loss_curves.png, validation_report.md, xai_approval.json
                                           (see evaluate.py)
  results/xai/                             IG figures and CSVs (decision, session, cross-session)

Run:   python main.py            # full run (~2 min on CPU)
       python main.py --quick    # smoke test: 1 seed, few episodes, 2 IG sessions -> results/quick/
"""
import argparse
import json
import shutil
from pathlib import Path

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
# Step 3: Integrated Gradients on the seed-0 DDQN (only after step 2)
# ---------------------------------------------------------------------------
def explain(cfg, approved):
    model_path = cfg["models"] / "ddqn_scripted.pt"
    model = torch.jit.load(str(model_path), map_location="cpu")
    seeds = list(range(IG_SEED, IG_SEED + cfg["ig_sessions"]))
    out = cfg["results"] / "xai"
    rows = xai_ig.audit(model, DialysisEnv(), seeds, out)
    manifest = dict(note="M3 approved (results/xai_approval.json)" if approved
                    else "NOT approved by M3: see results/validation_report.md",
                    model=str(model_path), seeds=seeds,
                    max_abs_residual=max(abs(r["residual"]) for r in rows), torch=torch.__version__)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nIG: {len(rows)} explained decisions -> {out} (max completeness residual {manifest['max_abs_residual']:.1e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the whole project: train -> check -> explain")
    parser.add_argument("--quick", action="store_true", help="smoke test: 1 seed, 200 episodes, 2 IG sessions")
    cfg = QUICK if parser.parse_args().quick else FULL
    cfg["models"].mkdir(parents=True, exist_ok=True); cfg["results"].mkdir(parents=True, exist_ok=True)

    train_all(cfg)
    approved = evaluate.run(cfg)
    explain(cfg, approved)
    print("\nDone.")
