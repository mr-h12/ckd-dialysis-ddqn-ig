"""
Module 2 - Full pipeline, one command (owner: Gomana Hossam Eldine).

Runs every step in the order the project requires:
  1. Train Double DQN (our model) and standard DQN (baseline) on seeds 0-4, save models + logs.
  2. Convergence + stability check: smoothed reward curves and held-out reward per seed (mean +/- std).
  3. Only after step 2: Integrated Gradients on the seed-0 DDQN model (Module 4 code).

Outputs:
  models/ddqn_seed<k>.pt, models/dqn_seed<k>.pt, models/ddqn_scripted_seed<k>.pt, models/ddqn_scripted.pt
  results/train_log_<name>_seed<k>.csv     per-episode training log
  results/stability.csv                    held-out reward + hypotension rate per agent and seed
  results/reward_curves.png                smoothed training reward, mean over seeds
  results/xai/                             IG figures and CSVs (decision, session, cross-session)

Module 3 (Zyad) adds tabular Q-learning, the official 100-patient metrics table and the
xai_approval.json that signs off on step 2. Until then the IG run is labelled "M3 approval pending".

Run:   python main.py            # full run (~3 min on CPU)
       python main.py --quick    # smoke test: 1 seed, few episodes, 2 IG sessions -> results/quick/
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                 # save figures to files, no window
import matplotlib.pyplot as plt
import numpy as np
import torch

import train                          # Module 2: train(), save_log(), quick_eval(), split_profiles()
import xai_ig                         # Module 4: audit()
from env import DialysisEnv, run_constant

# ---------------------------------------------------------------------------
# Settings (change here)
# ---------------------------------------------------------------------------
FULL = dict(seeds=train.SEEDS, episodes=train.CONFIG["episodes"], ig_sessions=100,
            models=Path("models"), results=Path("results"))
QUICK = dict(seeds=[0], episodes=200, ig_sessions=2,                     # never overwrites the real models
             models=Path("results/quick/models"), results=Path("results/quick"))
AGENTS = {"ddqn": True, "dqn": False}  # name -> double flag (DQN = same code with double=False)
IG_SEED = 10000                        # IG sessions start here, away from training and test seeds
SMOOTH = 50                            # moving-average window for the reward curves
RULE_ACTION = 4                        # rule-based reference: constant 13 mL/kg/h, the clinical ceiling


# ---------------------------------------------------------------------------
# Step 1: train both agents on every seed, save exactly like train.py does
# ---------------------------------------------------------------------------
def train_all(cfg):
    logs, agents = {}, {}
    for name, double in AGENTS.items():
        for seed in cfg["seeds"]:
            print(f"Training {name.upper()} seed {seed} for {cfg['episodes']} episodes")
            agent, log = train.train(seed, double=double, episodes=cfg["episodes"])
            train.save_log(cfg["results"] / f"train_log_{name}_seed{seed}.csv", log)
            agent.save(cfg["models"] / f"{name}_seed{seed}.pt")
            if double:
                agent.export_scripted(cfg["models"] / f"ddqn_scripted_seed{seed}.pt")
            logs[name, seed], agents[name, seed] = log, agent
    shutil.copy(cfg["models"] / f"ddqn_scripted_seed{cfg['seeds'][0]}.pt", cfg["models"] / "ddqn_scripted.pt")
    return logs, agents


# ---------------------------------------------------------------------------
# Step 2: convergence (reward curves) + stability (held-out reward over seeds)
# ---------------------------------------------------------------------------
def check_stability(cfg, logs, agents):
    # held-out score of every trained agent on the 80 unseen patients (100 sessions, greedy)
    rows = []
    for (name, seed), agent in agents.items():
        reward, hypo = train.quick_eval(agent)
        rows.append(dict(agent=name, seed=seed, heldout_reward=reward, hypotension_sessions=hypo))
    train.save_log(cfg["results"] / "stability.csv", rows)

    print("\nHeld-out reward (100 sessions, 80 unseen patients):")
    for name in AGENTS:
        r = [row["heldout_reward"] for row in rows if row["agent"] == name]
        h = [row["hypotension_sessions"] for row in rows if row["agent"] == name]
        print(f"  {name.upper():4s} per seed {np.round(r, 2).tolist()} -> mean {np.mean(r):.2f} +/- {np.std(r):.2f}"
              f" | hypotension sessions {np.mean(h):.1%}")

    # hook for Module 3: rule-based reference on the same held-out sessions
    _, test_profiles = train.split_profiles()
    env = DialysisEnv(profiles=test_profiles)
    rule = [run_constant(env, RULE_ACTION, seed) for seed in range(100)]
    print(f"  RULE constant 13 mL/kg/h -> mean {np.mean([x[2] for x in rule]):.2f}"
          f" | hypotension sessions {np.mean([x[3] for x in rule]):.1%}")

    # smoothed training reward: mean over seeds, shaded band = std over seeds
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in AGENTS:
        curves = np.array([[row["reward"] for row in logs[name, s]] for s in cfg["seeds"]])
        window = min(SMOOTH, curves.shape[1])
        smooth = np.array([np.convolve(c, np.ones(window) / window, mode="valid") for c in curves])
        x = np.arange(window - 1, curves.shape[1])
        ax.plot(x, smooth.mean(0), label=f"{name.upper()} (mean of {len(cfg['seeds'])} seeds)")
        ax.fill_between(x, smooth.mean(0) - smooth.std(0), smooth.mean(0) + smooth.std(0), alpha=0.2)
    ax.set(xlabel="Episode", ylabel=f"Reward ({window}-episode moving average)", title="Training convergence")
    ax.legend(); fig.tight_layout(); fig.savefig(cfg["results"] / "reward_curves.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# Step 3: Integrated Gradients on the seed-0 DDQN (only after step 2)
# ---------------------------------------------------------------------------
def explain(cfg):
    model_path = cfg["models"] / "ddqn_scripted.pt"
    model = torch.jit.load(str(model_path), map_location="cpu")
    seeds = list(range(IG_SEED, IG_SEED + cfg["ig_sessions"]))
    out = cfg["results"] / "xai"
    rows = xai_ig.audit(model, DialysisEnv(), seeds, out)
    manifest = dict(note="pipeline run from main.py; M3 approval (xai_approval.json) pending",
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

    logs, agents = train_all(cfg)
    check_stability(cfg, logs, agents)
    explain(cfg)
    print("\nDone. Pending from Module 3: tabular Q-learning, metrics table, xai_approval.json.")
