# M4 presentation content and merge plan

## 1. Title and team
CKD Dialysis Fluid Volume Regulation
Double DQN with Integrated Gradients

Group 2, Selected Topics: Reinforcement Learning and Explainable AI

- Yousef Mohamed Hasabo, 231011574: M1
- Gomana Hossam Eldine, 231013248: M2
- Zyad Gbriel, 231002617: M3
- Mohanad Mostfa Ali, 231012398: M4

Speaker note: These assignments follow the repository, which differs from the attached overview.

## 2. Related work connects sequential control and explanation
Yang et al. (2022) use Duel-DDQN for dialysis dry-weight assessment [1].
Double Q-learning separates selection and evaluation [2].
Integrated Gradients attributes a model output relative to a reference [3].
Recent dialysis studies investigate interpretable BP warnings and hypotension prediction [4, 5].

Speaker note: Our task selects UF during one session. Dry-weight assessment and risk prediction are related but different tasks. See literature_review.md for the comparison and full sources.

## 3. IG explains one chosen action's Q-value
We choose the action with the largest Q-value at the observed state.
We hold that action fixed while integrating gradients from a reference state.
Each feature receives a signed contribution to the Q-value difference.

IG_i = (x_i - b_i) * integral from 0 to 1 of dQ(b + alpha*(x-b), a)/dx_i d alpha

Speaker note: Positive contributions raise the selected action's Q-value relative to the reference. This does not establish a medical cause or prove superiority over another action.

## 4. The reference determines the question
Default reference: zero excess fluid and reference vitals, preserving time and demographics.
A second cohort reference allows age and albumin to vary.
We compare conclusions across references before interpreting feature rankings.

Speaker note: The matched reference forces age and albumin contributions to zero. It cannot establish that the model ignores them. Synthetic serum albumin in this environment is an assumption.

## 5. One decision, one session, many sessions
Insert decision.png, session.png and cross_session.png from a validated trained run.
The heatmap tracks changing signed contributions as actions change.
Across sessions, we report mean absolute IG with session standard deviations.

Current status: only software demonstration is available. Trained-policy findings remain pending M2 and M3.

Speaker note: Demo images carry an untrained-network label. Keep that label if demonstrating the implementation before trained results exist.

## 6. Verification precedes interpretation
M3 reviews convergence and stability across five training seeds.
The approval file identifies the exact exported model by SHA256.
Every explained decision passes numerical completeness within tolerance.
We inspect reference sensitivity and patient-level evaluation splits.

Speaker note: Completeness checks arithmetic. It does not validate the simulator, policy safety, or generalization.

## 7. Future work addresses the simulator's assumptions
Use measured dialysis time series and serum albumin.
Validate fluid and pressure dynamics against observations.
Evaluate on patient-level held-out cohorts.
Compare multiple explanation references and alternative-action contrasts.
Resolve hidden simulator state with M1 and M2.

## 8. References
Use the five IEEE references in literature_review.md, split over two slides if needed.

## Final deck order

1. Title and team (M4)
2. Project idea (M2)
3. Problem statement (M1)
4. Dataset and MDP (M1)
5. Related work (M4)
6. Proposed approach and solution (M2)
7. Results and discussion, including convergence and stability (M3)
8. Explanation method and actual figures (M4)
9. Future work (M4)
10. References (M4)

Before final assembly, request editable slides from each owner. Replace pending content with actual results. Keep units and state/action definitions consistent with the repository, and verify all IDs, plots and references.
