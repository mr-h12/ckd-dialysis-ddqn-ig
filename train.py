"""
Module 2 - Training script (owner: Gomana Hossam Eldine).

Plays thousands of pretend dialysis sessions, learns from them, and saves:
  results/train_log_<name>_seed<k>.csv   per-episode reward, loss, epsilon, hypotension
  models/<name>_seed<k>.pt               trained weights
  models/ddqn_scripted_seed<k>.pt        TorchScript export for Module 4 (Integrated Gradients)
  models/ddqn_scripted.pt                the same export for seed 0 (default file Module 4 reads)

Run:   python train.py                       # DDQN, seeds 0-4
       python train.py --dqn                 # standard DQN baseline (for Module 3)
       python train.py --episodes 300 --seeds 0   # quick test
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from agents import DDQNAgent
from env import DialysisEnv, load_profiles, N_STEPS

# ---------------------------------------------------------------------------
# Hyperparameters (change here)
# ---------------------------------------------------------------------------
CONFIG = dict(
    episodes=1500,          # number of practice sessions per seed
    gamma=0.99,             # importance of future points
    lr=1e-3,                # learning speed
    batch_size=64,          # memories per learning step
    buffer_size=50_000,     # memory box size
    target_update=200,      # refresh the slow copy every N learning steps
    reward_scale=0.1,       # rewards are multiplied by this inside the agent
    eps_start=1.0,          # at first: 100% random exploration
    eps_end=0.05,           # at the end: 5% random
    eps_decay_frac=0.6,     # reach eps_end after 60% of the episodes
)
SEEDS = [0, 1, 2, 3, 4]               # shared contract
TRAIN_SEED_OFFSET = 500_000           # keeps training patients away from M3 (0-99) and M4 (10000+) seeds
MODELS, RESULTS = Path("models"), Path("results")


# ---------------------------------------------------------------------------
# Patient split: train on 320 UCI profiles, keep 80 never-seen profiles for testing
# ---------------------------------------------------------------------------
def split_profiles():
    profiles = load_profiles()
    idx = np.random.default_rng(0).permutation(len(profiles))   # fixed shuffle -> same split for everyone
    return profiles[idx[:320]], profiles[idx[320:]]


def epsilon_at(episode, total):
    """Linear decay of exploration from eps_start to eps_end."""
    frac = min(1.0, episode / (CONFIG["eps_decay_frac"] * total))
    return CONFIG["eps_start"] + frac * (CONFIG["eps_end"] - CONFIG["eps_start"])


# ---------------------------------------------------------------------------
# Training one agent for one seed
# ---------------------------------------------------------------------------
def train(seed, double=True, episodes=None):
    episodes = episodes or CONFIG["episodes"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_profiles, _ = split_profiles()
    env = DialysisEnv(profiles=train_profiles)
    agent = DDQNAgent(double=double, gamma=CONFIG["gamma"], lr=CONFIG["lr"],
                      batch_size=CONFIG["batch_size"], buffer_size=CONFIG["buffer_size"],
                      target_update=CONFIG["target_update"], reward_scale=CONFIG["reward_scale"])
    log = []
    for ep in range(episodes):
        # seed the environment once, then let it keep drawing new patients
        state = env.reset(seed=TRAIN_SEED_OFFSET + seed) if ep == 0 else env.reset()
        eps = epsilon_at(ep, episodes)
        total_r, losses, hypo, done = 0.0, [], 0, False
        while not done:
            action = agent.act(state, eps)                      # 1. pick a UF rate
            next_state, r, done, info = env.step(action)        # 2. the pretend patient reacts
            agent.store(state, action, r, next_state, done)     # 3. remember it
            loss = agent.learn()                                # 4. learn from random memories
            if loss is not None:
                losses.append(loss)
            state, total_r, hypo = next_state, total_r + r, hypo + info["hypotension"]
        log.append(dict(episode=ep, reward=total_r, loss=float(np.mean(losses)) if losses else float("nan"),
                        epsilon=eps, hypotension_steps=hypo, final_overload=info["overload"]))
        if (ep + 1) % 100 == 0:
            recent = np.mean([row["reward"] for row in log[-100:]])
            print(f"  seed {seed} | episode {ep + 1:5d} | avg reward (last 100) {recent:7.2f} | epsilon {eps:.2f}")
    return agent, log


# ---------------------------------------------------------------------------
# Quick check on the 80 unseen patients (greedy, no exploration). Module 3 does the official evaluation.
# ---------------------------------------------------------------------------
def quick_eval(agent, n=100):
    _, test_profiles = split_profiles()
    env = DialysisEnv(profiles=test_profiles)
    rewards, hypo_sessions = [], 0
    for seed in range(n):
        s, done, total, hypo = env.reset(seed=seed), False, 0.0, False
        while not done:
            s, r, done, info = env.step(agent.act(s, epsilon=0.0))
            total, hypo = total + r, hypo or info["hypotension"]
        rewards.append(total); hypo_sessions += hypo
    return float(np.mean(rewards)), hypo_sessions / n


def save_log(path, log):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0]))
        writer.writeheader(); writer.writerows(log)


# ---------------------------------------------------------------------------
# Main: train all seeds, save everything
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dqn", action="store_true", help="standard DQN baseline instead of Double DQN")
    parser.add_argument("--episodes", type=int, default=CONFIG["episodes"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    name = "dqn" if args.dqn else "ddqn"
    MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    for seed in args.seeds:
        print(f"Training {name.upper()} seed {seed} for {args.episodes} episodes")
        agent, log = train(seed, double=not args.dqn, episodes=args.episodes)
        save_log(RESULTS / f"train_log_{name}_seed{seed}.csv", log)
        agent.save(MODELS / f"{name}_seed{seed}.pt")
        if not args.dqn:
            agent.export_scripted(MODELS / f"ddqn_scripted_seed{seed}.pt")
            if seed == 0:
                shutil.copy(MODELS / "ddqn_scripted_seed0.pt", MODELS / "ddqn_scripted.pt")
        r, h = quick_eval(agent)
        print(f"  seed {seed} on 100 unseen-patient sessions: mean reward {r:.2f}, hypotension sessions {h:.0%}")
