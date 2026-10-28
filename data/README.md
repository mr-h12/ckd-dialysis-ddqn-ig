# Dataset: UCI Chronic Kidney Disease

- **Source:** UCI Machine Learning Repository, dataset id 336, https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease. License CC BY 4.0.
- **Origin / reliability:** collected over ~2 months at Apollo Hospitals, Karaikudi, India (donated 2015 by L. Jerlin Rubini, Dr. P. Soundarapandian, Dr. P. Eswaran). One of the most-used public CKD datasets and the standard benchmark for CKD classification papers.
- **Size:** 400 patients × 24 features + 1 target. 11 numeric, 13 nominal; has missing values.
- **Target:** `class` = `ckd` / `notckd` (250 / 150).
- **File:** `ckd.csv` (regenerate with `python data/fetch_data.py`).

## How we use it
We do **not** train on the target. The dataset only supplies realistic patient profiles for the dialysis simulator (`env.py`):

| Simulator parameter | From UCI | How |
|---|---|---|
| age | `age` | as is |
| diastolic BP | `bp` | as is (mmHg) |
| systolic BP | — | `bp` + pulse pressure ~ N(50, 8) |
| serum albumin | `al` (urine albumin grade 0–5) | `4.2 − 0.25·al` + noise, clipped to 2.5–5 g/dL. Assumption: more protein lost in urine → lower serum albumin. |
| weight, fluid overload, HR | — | sampled: N(70,12) kg, U(1.5, 4) kg, N(75, 8) bpm |

Missing values are filled with the column median.

**Limitation / future work:** UCI CKD has no intradialytic time series. The Lin et al. 2019 hemodialysis dataset (Scientific Data; 1,075 patients, 165,986 sessions) would allow learning the dynamics from real sessions instead of a hand-built simulator.
