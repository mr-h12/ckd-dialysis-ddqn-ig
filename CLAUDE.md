# CKD Dialysis Fluid Regulation — DDQN + Integrated Gradients

Group 2 university project (Selected Topics: RL & XAI). Spec PDFs live in `project details /`.
Team: Gomana Hossam Eldine (231013248), Yousef Mohamed Hasabo (231011574), Mohanad Mostfa Ali (231012398), Zyad Gbriel (231002617).
Deadline: 9 Nov 2026 11:00 PM (Classroom). Discussion: 9 Nov 2026.

## Task
Model dialysis fluid removal / ultrafiltration (UF) rate with a Double DQN; audit feature attribution across sessions with Integrated Gradients.

## Mandatory (missing any = ZERO grade)
- Explicit MDP: State S, Action A, Reward R, Transition S' — documented in code and slides.
- Baselines vs DDQN: standard DQN, tabular Q-learning, rule-based controller (constant UF, capped 13 mL/kg/h).
- Convergence + stability checks (reward curves, multiple seeds) BEFORE running Integrated Gradients.
- Deliverables in one folder: dataset (or direct link + source/reliability evidence + size/features/target), complete runnable commented Python, presentation.
- Presentation sections: title+members/IDs, idea, problem statement, literature review, proposed approach, proposed solution, results & discussion, future work, references (prefer 2025–2026 high-impact papers, one consistent style).

## Scope
- Class project: keep it small. UCI CKD for patient profiles only; simple 3–4 equation simulator; actions `{0,5,8,10,13} mL/kg/h`; 5 references.

## Code conventions
- Python; organize into clear sections/functions with comments explaining each step (graded).
- Code must run end-to-end and reproduce reported results — fix random seeds.

## Gotchas
- Paths contain trailing spaces: `selected topics /`, `project details /` — always quote them.

## Team plan
- `TEAM_PLAN.md` — module split (owners TBD), shared contracts (state/action/env API), timeline. Follow its contracts.
