# Plan: Split the Group 2 project into 4 modules (CKD dialysis UF control, DDQN + Integrated Gradients)

## Context
The 4 team members need the project split into modules, with each module broken into concrete points and owned by one student. The split must cover the whole idea and the whole pipeline, including every requirement that scores a ZERO if missed (see CLAUDE.md). There is no code yet, only the spec PDFs. Due 9 Nov 2026 at 11 PM (discussion also 9 Nov); that gives 12 days.

## Project idea (one line)
During hemodialysis, an agent picks the ultrafiltration (UF) rate every 15 min. The goal is to reach the patient's dry weight without crashing blood pressure (intradialytic hypotension). A Double DQN learns that policy. Integrated Gradients then shows which patient signals drove each UF decision, and how that changes over the session and across sessions.

## End-to-end pipeline
```
[1] Dataset (real CKD patient profiles)
      │ clean + sample patient parameters
      ▼
[2] Dialysis simulator = MDP environment (S, A, R, S')
      │ reset(patient) / step(uf_rate)
      ▼
[3] Agents: DDQN (proposed) | DQN | Tabular Q | Rule-based (constant UF, capped 13 mL/kg/h)
      │ train with fixed seeds, save models + logs
      ▼
[4] Evaluation: convergence curves, multi-seed stability, metrics table vs baselines
      │ only after the agent is shown to be stable ↓
      ▼
[5] XAI: Integrated Gradients on DDQN Q-network → per-step, per-session, cross-session attributions
      ▼
[6] Deliverables: dataset folder + README, runnable code, presentation (8 sections + refs)
```

## Shared contracts (agree on Day 1 so everyone can work in parallel)
- **State vector** (normalized, 8 features, fixed order):
  `fluid_overload_kg, removed_so_far_kg, sbp, dbp, heart_rate, time_elapsed_frac, age, albumin`
- **Actions**: 5 discrete UF rates per body weight: `[0, 5, 8, 10, 13] mL/kg/h` (13 = common clinical ceiling)
- **Step** = 15 min. **Episode** = one 4-hour session = 16 steps.
- **Env API** (Gymnasium style): `env.reset(seed) -> state`, `env.step(a) -> state, reward, done, info`
- **Agent API**: `agent.act(state, greedy) -> a`, `agent.train(env, episodes) -> log`, `agent.save(path)`
- Seeds: `[0, 1, 2, 3, 4]`. Results are saved as CSV in `results/` and models as `.pt` in `models/`.
- Repo layout (flat, few files):
  `data/`, `env.py`, `agents.py`, `train.py`, `evaluate.py`, `xai_ig.py`, `main.py`, `requirements.txt`, `README.md`, `slides/`

---

## Module 1: Data & Dialysis Simulator (MDP environment). Owner: **Yousef Mohamed Hasabo**
1. Dataset: UCI CKD (400 patients, 24 features), used only to generate realistic patients (age, BP, albumin). The simulator does the rest.
   - Mention the larger Lin et al. 2019 hemodialysis dataset (Scientific Data) as future work only.
2. Write the dataset justification: why this dataset, its source and reliability with citations, and its size, features and target. This goes into the README and the slides.
3. Preprocessing: handle missing values, normalize, and build a patient-profile sampler (age, BP, albumin, starting weight gain).
4. Simple simulator (3–4 equations, cite one paper for the idea):
   - fluid overload goes down by the UF rate
   - BP drops when the UF rate is high
   - HR rises when BP drops
   - small random noise
5. Formal MDP definition: write S, A, R and S' as equations, for the code docstring and a slide.
   - **Reward:** `+ fluid removed toward dry weight − big penalty if SBP < 90 − penalty for |ΔUF| jumps − terminal penalty for remaining overload`
6. Sanity tests: a fixed-UF run makes BP drop when UF is high, and an episode ends at 16 steps.
7. Slides owned: **Problem Statement**, **Dataset**, **MDP formulation**.

## Module 2: DDQN Agent (proposed model). Owner: **Gomana Hossam Eldine**
1. Q-network (PyTorch MLP 8→64→64→5), replay buffer, ε-greedy schedule.
2. Double DQN update: the online net selects the argmax, the target net evaluates it, and the target net is soft- or hard-updated.
3. Training loop with logging: episode reward, loss, ε and hypotension events per episode.
4. Hyperparameter config at the top of the file (γ, lr, batch, buffer size, target update), plus a short tuning sweep.
5. Train over 5 seeds, then save models and logs for Modules 3 and 4.
6. Write `main.py` to run the full pipeline end to end, and `requirements.txt`.
7. Slides owned: **Project Idea**, **Proposed Approach** (DDQN algorithm + architecture), **Proposed Solution**.

## Module 3: Baselines, Evaluation & Validation. Owner: **Zyad Gbriel**
1. **Standard DQN** baseline: reuse the Module 2 network, but the target net both selects and evaluates (1 flag).
2. **Tabular Q-learning**: discretize the state into bins (overload × SBP × time).
3. **Rule-based controller**: constant UF rate, never above 13 mL/kg/h (one line of code).
4. Convergence checks: smoothed reward curves and loss curves, compared across agents.
5. Stability checks: mean ± std over 5 seeds and the variance of the final policy.
6. Evaluate on 100 held-out simulated patients. Metrics:
   - final fluid overload
   - % of sessions with hypotension
   - number of hypotension steps
   - mean reward
   - UF smoothness

   Output: the comparison table and the plots.
7. Slides owned: **Results & Discussion** (table, curves, interpretation).

## Module 4: Explainability (Integrated Gradients), Literature & Presentation. Owner: **Mohanad Mostfa Ali**
1. Integrated Gradients on the trained DDQN Q-network, targeting the Q-value of the chosen action. Use Captum, with a baseline equal to the "healthy / zero-overload" state. Justify it in one sentence: "We use a healthy patient with no extra fluid as the baseline."
2. Per-step explanation: an attribution bar chart for single decisions, e.g. "why UF was cut to 5 mL/kg/h".
3. Within-session view: an attribution heatmap of feature × time step across one session.
4. Cross-session audit: compare attributions across many patients and sessions. Check that the physiologically sensible features dominate (overload, SBP), and flag cases that rely on age/albumin.
5. Sanity check of IG completeness: the sum of attributions ≈ Q(x) − Q(baseline).
6. Literature review: 5 papers, IEEE style:
   - anchor: RL (dueling double DQN) for dialysis dry weight, IEEE JBHI 2022
   - 2–3 recent (2025–2026) papers on ML for dialysis or IG in healthcare
   - van Hasselt et al. 2016 (Double DQN) and Sundararajan et al. 2017 (Integrated Gradients)
7. Presentation assembly: Title + members and IDs, Literature Review, XAI results, **Future Work**, **References**. Also merge everyone's slides into one deck.

---

## Dependencies & timeline (Oct 28 → Nov 9)
| Day | M1 Yousef | M2 Gomana | M3 Zyad | M4 Mohanad |
|---|---|---|---|---|
| Oct 28–30 | dataset + env v1 | DDQN on a dummy env | rule-based + tabular | literature review, IG on a dummy net |
| Oct 31–Nov 2 | env final + MDP doc | train DDQN on the real env | DQN + eval script | IG pipeline ready |
| Nov 3–5 | dataset README | 5-seed runs, save models | convergence/stability + table | IG on the trained models |
| Nov 6–8 | slides | main.py end to end, slides | plots, slides | merge deck |
| Nov 9 | **all: full run from a clean env, rehearse, submit before 11 PM** | | | |

Critical path: env (M1) → DDQN training (M2) → evaluation (M3) + IG (M4). Until the real env lands, M2, M3 and M4 work against a dummy env that follows the shared contract.

## Verification (before submitting)
- `python main.py` runs from a fresh venv and regenerates every results CSV, plot and IG figure.
- Each ZERO-grade item is ticked against CLAUDE.md: MDP defined, 3 baselines compared, convergence/stability shown before IG, dataset justified, presentation has all 8 sections + references.
