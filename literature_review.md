# Literature review (five references, IEEE style)

Verified against publisher pages and original-paper records on 28 October 2026. Foundational algorithm references retain their original years. Two application papers are from 2025.

## Comparison and project relevance

| Work | Contribution | How we use it | Boundary |
|---|---|---|---|
| Yang et al. [1] | Formulates dry-weight assessment as sequential decision making with Duel-DDQN | Motivation for an MDP and value-based RL in dialysis | Its dry-weight assessment policy differs from our within-session UF controller. We do not reproduce its validation |
| van Hasselt et al. [2] | Separates action selection from action evaluation to reduce Q-value overestimation | DDQN algorithm basis | Demonstrated on games, not clinical safety evidence |
| Sundararajan et al. [3] | Introduces IG using sensitivity and implementation invariance | Chosen-action attribution and completeness checks | Attribution depends on the reference and model |
| Li et al. [4] | Explainable ML for warning of blood-pressure anomalies in maintenance hemodialysis | Motivation for interpretable patient-signal analysis | Prediction and SHAP differ from control and IG |
| Wang et al. [5] | Develops ML models for intradialytic hypotension prediction | Context for BP-risk modeling and evaluation | Predictive performance does not validate a treatment policy |

Our contribution is a classroom combination of a simplified dialysis environment, DDQN comparisons, and an IG audit across simulated sessions. None of these papers validates our hand-tuned simulator or the synthetic albumin mapping.

## References

[1] Z. Yang, Y. Tian, T. Zhou, Y. Zhu, P. Zhang, J. Chen, and J. Li, “Optimization of Dry Weight Assessment in Hemodialysis Patients via Reinforcement Learning,” *IEEE Journal of Biomedical and Health Informatics*, vol. 26, no. 10, pp. 4880–4891, 2022, doi: 10.1109/JBHI.2022.3192021. https://doi.org/10.1109/JBHI.2022.3192021

[2] H. van Hasselt, A. Guez, and D. Silver, “Deep Reinforcement Learning with Double Q-learning,” in *Proceedings of AAAI*, 2016. Original paper: https://arxiv.org/abs/1509.06461

[3] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic Attribution for Deep Networks,” in *Proceedings of ICML*, PMLR, vol. 70, pp. 3319–3328, 2017. https://proceedings.mlr.press/v70/sundararajan17a.html

[4] Z. Li, S. Hao, S. Shi, L. Li, and Z. Tao, “An explainable machine learning model for early warning of hypertensive and hypotensive anomalies in maintenance hemodialysis patients,” *BMC Nephrology*, vol. 26, art. 318, 2025, doi: 10.1186/s12882-025-04270-5. https://link.springer.com/article/10.1186/s12882-025-04270-5 (Publisher notes a September 2025 correction.)

[5] P. Wang et al., “Construction and application of machine learning models for predicting intradialytic hypotension,” *PLOS One*, vol. 20, no. 10, e0333357, 2025, doi: 10.1371/journal.pone.0333357. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0333357

Implementation reference: Captum IntegratedGradients API, https://captum.ai/api/integrated_gradients.html. The API returns an approximation delta based on completeness. A small delta verifies numerical integration, not clinical reliability.
