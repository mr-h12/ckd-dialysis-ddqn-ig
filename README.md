# CKD Dialysis Fluid Volume Regulation — DDQN + Integrated Gradients

Group 2 class project (Selected Topics: RL & XAI). A Double DQN picks the ultrafiltration (UF) rate every 15 min of a hemodialysis session; Integrated Gradients audits which patient signals drove each decision.

## Team & modules

| Module | Owner | ID |
|---|---|---|
| M1 Data & Dialysis Simulator (MDP env) | Yousef Mohamed Hasabo | 231011574 |
| M2 DDQN Agent | Gomana Hossam Eldine | 231013248 |
| M3 Baselines, Evaluation & Validation | Zyad Gbriel | 231002617 |
| M4 Integrated Gradients & Literature | Mohanad Mostfa Ali | 231012398 |
| Presentation (external job, one person) | Yousef Mohamed Hasabo | 231011574 |

Full module breakdown, shared contracts (state / action / env API) and timeline: [TEAM_PLAN.md](TEAM_PLAN.md).

## Trained models

`python train.py` (M2, about 1 min on CPU, fully deterministic) regenerates everything in `models/`:

| File | What |
|---|---|
| `models/ddqn_scripted_seed{0..4}.pt` | TorchScript DDQN online Q-network (8 → 64 → 64 → 5) per training seed. Load with `torch.jit.load(path)`. |
| `models/ddqn_scripted.pt` | Copy of seed 0, the default model for `xai_ig.py`. |
| `models/ddqn_seed{0..4}.pt` | Raw `state_dict` weights. |

Quick check on 100 sessions of the 80 held-out UCI patients (greedy policy):

| Seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Mean reward | 1.00 | 0.97 | 0.88 | 1.49 | 0.76 |
| Hypotension sessions | 4% | 0% | 0% | 0% | 1% |

## M4 explanations

See [M4 setup and handoff](README_M4.md) for Integrated Gradients, demo plots, model validation requirements, and presentation content.
