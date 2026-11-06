# Literature review (ten references, IEEE style)

Verified against publisher pages and original-paper records on 28 October 2026. Foundational algorithm references retain their original years. Two application papers are from 2025 [4, 5] and two reviews are from 2026 [9, 10]; [9] and [10] were checked against Crossref on 6 November 2026.

## Comparison and project relevance

| Work | Contribution | How we use it | Boundary |
|---|---|---|---|
| Yang et al. [1] | Formulates dry-weight assessment as sequential decision making with Duel-DDQN | Motivation for an MDP and value-based RL in dialysis | Its dry-weight assessment policy differs from our within-session UF controller. We do not reproduce its validation |
| van Hasselt et al. [2] | Separates action selection from action evaluation to reduce Q-value overestimation | DDQN algorithm basis | Demonstrated on games, not clinical safety evidence |
| Sundararajan et al. [3] | Introduces IG using sensitivity and implementation invariance | Chosen-action attribution and completeness checks | Attribution depends on the reference and model |
| Li et al. [4] | Explainable ML for warning of blood-pressure anomalies in maintenance hemodialysis | Motivation for interpretable patient-signal analysis | Prediction and SHAP differ from control and IG |
| Wang et al. [5] | Develops ML models for intradialytic hypotension prediction | Context for BP-risk modeling and evaluation | Predictive performance does not validate a treatment policy |
| Kanbay et al. [6] | Reviews intradialytic hypotension, plasma refill, and associated risk factors | Motivates the simulator pressure response and choice of rate and pressure thresholds | The simulator uses simplified dynamics and project-specific thresholds |
| Cheungpasitporn et al. [9] | Roadmap for clinical AI in nephrology, including AI-assisted UF management in dialysis, IDH prediction, and RL as adaptive decision support | Places our UF controller within current nephrology AI and its interpretability and validation barriers | A narrative review; it does not evaluate a specific UF policy |
| Tang et al. [10] | Scoping review of 72 RL studies for sepsis treatment (mostly fluids and vasopressors on MIMIC) | Supports our explicit S/A/R design, baselines, and caution about off-policy claims; motivates interpretability | Sepsis ICU care, not dialysis; all reviewed studies are retrospective |

Our contribution is a classroom combination of a simplified dialysis environment, DDQN comparisons, and an IG audit across simulated sessions. None of these papers validates our hand-tuned simulator or the synthetic albumin mapping.

## References

[1] Z. Yang, Y. Tian, T. Zhou, Y. Zhu, P. Zhang, J. Chen, and J. Li, “Optimization of Dry Weight Assessment in Hemodialysis Patients via Reinforcement Learning,” *IEEE Journal of Biomedical and Health Informatics*, vol. 26, no. 10, pp. 4880–4891, 2022, doi: 10.1109/JBHI.2022.3192021. https://doi.org/10.1109/JBHI.2022.3192021

[2] H. van Hasselt, A. Guez, and D. Silver, “Deep Reinforcement Learning with Double Q-learning,” in *Proceedings of AAAI*, 2016. Original paper: https://arxiv.org/abs/1509.06461

[3] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic Attribution for Deep Networks,” in *Proceedings of ICML*, PMLR, vol. 70, pp. 3319–3328, 2017. https://proceedings.mlr.press/v70/sundararajan17a.html

[4] Z. Li, S. Hao, S. Shi, L. Li, and Z. Tao, “An explainable machine learning model for early warning of hypertensive and hypotensive anomalies in maintenance hemodialysis patients,” *BMC Nephrology*, vol. 26, art. 318, 2025, doi: 10.1186/s12882-025-04270-5. https://link.springer.com/article/10.1186/s12882-025-04270-5 (Publisher notes a September 2025 correction.)

[5] P. Wang et al., “Construction and application of machine learning models for predicting intradialytic hypotension,” *PLOS One*, vol. 20, no. 10, e0333357, 2025, doi: 10.1371/journal.pone.0333357. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0333357

[6] M. Kanbay, L. A. Ertuglu, B. Afsar, E. Ozdogan, D. Siriopol, A. Covic, C. Basile, and A. Ortiz, “An update review of intradialytic hypotension: concept, risk factors, clinical implications and management,” *Clin. Kidney J.*, vol. 13, no. 6, pp. 981–993, 2020, doi: 10.1093/ckj/sfaa078. https://doi.org/10.1093/ckj/sfaa078

[7] L. Rubini, P. Soundarapandian, and P. Eswaran, “Chronic Kidney Disease,” UCI Machine Learning Repository, 2015, doi: 10.24432/C5G020. https://doi.org/10.24432/C5G020

[8] H. Polat, H. Danaei Mehr, and A. Cetin, “Diagnosis of chronic kidney disease based on support vector machine by feature selection methods,” *J. Med. Syst.*, vol. 41, no. 4, art. 55, 2017, doi: 10.1007/s10916-017-0703-x. https://doi.org/10.1007/s10916-017-0703-x

[9] W. Cheungpasitporn, A. Athavale, L. Ghazi, K. B. Kashani, T. Colicchio, J. L. Koyner, J. Chen, J. H. Ix, G. Nadkarni, and J. A. Neyra, “Transforming nephrology through artificial intelligence: a state-of-the-art roadmap for clinical integration,” *Clin. Kidney J.*, vol. 19, no. 2, art. sfag004, 2026, doi: 10.1093/ckj/sfag004. https://doi.org/10.1093/ckj/sfag004

[10] Z. Tang, P. Sun, R. Liang, S. Tong, Y. Yang, M. Yang, and H. Chen, “Reinforcement learning for treatment decision-making in sepsis: a scoping review,” *npj Digit. Med.*, 2026, doi: 10.1038/s41746-026-03034-1. https://doi.org/10.1038/s41746-026-03034-1 (Published online 18 July 2026; Crossref lists no volume or article number yet.)

Implementation reference: Captum IntegratedGradients API, https://captum.ai/api/integrated_gradients.html. The API returns an approximation delta based on completeness. A small delta verifies numerical integration, not clinical reliability.
