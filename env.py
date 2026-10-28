"""
Module 1 — Hemodialysis simulator as an MDP (owner: Yousef Mohamed Hasabo).

An agent picks the ultrafiltration (UF) rate every 15 min of a 4-hour session.
Goal: remove the patient's extra fluid without crashing blood pressure
(intradialytic hypotension, SBP < 90 mmHg).

MDP
---
State S (8 features, normalized to ~[0, 1], fixed order — shared contract):
    overload  fluid still above dry weight (kg)         / 4
    removed   fluid removed so far this session (kg)    / 4
    sbp       systolic blood pressure (mmHg)            / 200
    dbp       diastolic blood pressure (mmHg)           / 120
    hr        heart rate (bpm)                          / 150
    time      elapsed fraction of the session (t / 16)
    age       patient age (years)                       / 100
    albumin   serum albumin (g/dL)                      / 5

Action A: index into ACTIONS = [0, 5, 8, 10, 13] mL/kg/h (13 = clinical ceiling).

Transition S -> S' (dt = 0.25 h, W = body weight in kg):
    (1) dv = min(u * W * dt / 1000, overload)       fluid removed this step (kg)
        overload' = overload - dv,  removed' = removed + dv
    (2) SBP' = SBP - a * (1 + age_f) * (u - refill) + b * (SBP0 - SBP) + noise
        refill = plasma refilling rate (mL/kg/h), higher with higher albumin
        age_f  = extra BP sensitivity of older patients
    (3) DBP' = DBP + 0.5 * (SBP' - SBP) + noise
    (4) HR'  = HR0 + c * (SBP0 - SBP') + noise      (heart compensates for BP drop)

Reward R:
    r = dv  - 10 * [SBP' < 90]  - 0.1 * |u - u_prev| / 13
    at the last step also: - 2 * overload'          (fluid left behind)

Episode: 16 steps (4 h). Patient profiles (age, diastolic BP, urine albumin) are
sampled from the UCI CKD dataset; see data/README.md for derived quantities.
"""
import csv
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (shared contract — do not change without telling the team)
# ---------------------------------------------------------------------------
ACTIONS = [0, 5, 8, 10, 13]        # UF rates, mL/kg/h
N_STEPS = 16                       # 16 x 15 min = 4 h session
DT = 0.25                          # hours per step
STATE_DIM = 8
N_ACTIONS = len(ACTIONS)
HYPOTENSION_SBP = 90               # mmHg

# Simulator coefficients (hand-tuned so a constant 13 mL/kg/h often causes
# hypotension while 5-8 mL/kg/h usually does not)
A_DROP = 0.6        # mmHg drop per (mL/kg/h) of net fluid loss, per step
B_RECOVER = 0.1     # fraction of BP gap to baseline recovered per step
C_HR = 0.3          # bpm rise per mmHg of SBP drop
SCALE = np.array([4, 4, 200, 120, 150, 1, 100, 5], dtype=np.float32)

DATA_PATH = Path(__file__).parent / "data" / "ckd.csv"


# ---------------------------------------------------------------------------
# Patient profiles from UCI CKD
# ---------------------------------------------------------------------------
def load_profiles(path=DATA_PATH):
    """Return an (N, 3) array of [age, diastolic bp, urine albumin 0-5].
    Missing values are filled with the column median."""
    with open(path) as f:
        rows = [[r["age"], r["bp"], r["al"]] for r in csv.DictReader(f)]
    x = np.array([[float(v) if v not in ("", "nan") else np.nan for v in r] for r in rows])
    med = np.nanmedian(x, axis=0)
    return np.where(np.isnan(x), med, x)


def sample_patient(profiles, rng):
    """Turn one UCI row into simulator parameters.
    UCI has no SBP, serum albumin or weight, so these are derived/sampled:
      SBP0    = diastolic bp + pulse pressure ~ N(50, 8)
      albumin = 4.2 - 0.25 * urine albumin grade (more proteinuria -> lower serum albumin)
      weight ~ N(70, 12) kg, starting fluid overload ~ U(1.5, 4) kg, HR0 ~ N(75, 8)
    """
    age, dbp, al = profiles[rng.integers(len(profiles))]
    return dict(
        age=age,
        dbp0=dbp,
        sbp0=dbp + rng.normal(50, 8),
        albumin=float(np.clip(4.2 - 0.25 * al + rng.normal(0, 0.2), 2.5, 5.0)),
        weight=float(np.clip(rng.normal(70, 12), 45, 110)),
        overload0=rng.uniform(1.5, 4.0),
        hr0=rng.normal(75, 8),
    )


# ---------------------------------------------------------------------------
# Environment (Gymnasium-style API: reset(seed) -> state, step(a) -> s, r, done, info)
# ---------------------------------------------------------------------------
class DialysisEnv:
    def __init__(self, profiles=None):
        self.profiles = load_profiles() if profiles is None else profiles
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        """Start a new session with a freshly sampled patient."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p = sample_patient(self.profiles, self.rng)
        self.overload = self.p["overload0"]
        self.removed = 0.0
        self.sbp, self.dbp, self.hr = self.p["sbp0"], self.p["dbp0"], self.p["hr0"]
        self.t = 0
        self.u_prev = 0
        return self._state()

    def step(self, a):
        u = ACTIONS[a]
        p, rng = self.p, self.rng

        # (1) fluid balance
        dv = min(u * p["weight"] * DT / 1000, self.overload)
        self.overload -= dv
        self.removed += dv

        # (2) SBP: drops with net fluid loss (UF minus plasma refill), recovers toward baseline
        refill = 4.0 + 2.0 * (p["albumin"] - 3.5)          # ~2-7 mL/kg/h
        age_f = max(0.0, (p["age"] - 50) / 50)
        sbp_new = (self.sbp - A_DROP * (1 + age_f) * (u - refill)
                   + B_RECOVER * (p["sbp0"] - self.sbp) + rng.normal(0, 2))
        sbp_new = min(sbp_new, p["sbp0"] + 10)               # refill can't push BP far above baseline

        # (3) DBP follows SBP at half the change; (4) HR compensates for the drop
        self.dbp += 0.5 * (sbp_new - self.sbp) + rng.normal(0, 1)
        self.sbp = sbp_new
        self.hr = p["hr0"] + C_HR * (p["sbp0"] - self.sbp) + rng.normal(0, 1)
        self.t += 1

        # reward
        hypo = self.sbp < HYPOTENSION_SBP
        r = dv - 10.0 * hypo - 0.1 * abs(u - self.u_prev) / 13
        self.u_prev = u
        done = self.t >= N_STEPS
        if done:
            r -= 2.0 * self.overload

        info = dict(sbp=self.sbp, uf=u, removed=self.removed,
                    overload=self.overload, hypotension=bool(hypo))
        return self._state(), r, done, info

    def _state(self):
        raw = [self.overload, self.removed, self.sbp, self.dbp, self.hr,
               self.t / N_STEPS, self.p["age"], self.p["albumin"]]
        return np.array(raw, dtype=np.float32) / SCALE


# ---------------------------------------------------------------------------
# Sanity check: python env.py
# ---------------------------------------------------------------------------
def run_constant(env, a, seed):
    """Run one episode with a fixed action; return (steps, min SBP, total reward, any hypotension)."""
    s, done, steps, total, sbps, hypo = env.reset(seed), False, 0, 0.0, [], False
    while not done:
        s, r, done, info = env.step(a)
        steps, total = steps + 1, total + r
        sbps.append(info["sbp"])
        hypo |= info["hypotension"]
    assert s.shape == (STATE_DIM,)
    return steps, min(sbps), total, hypo


if __name__ == "__main__":
    env = DialysisEnv()
    print(f"loaded {len(env.profiles)} UCI CKD patient profiles")

    # same patient (seed), no UF vs max UF
    steps0, min0, _, _ = run_constant(env, 0, seed=0)
    steps13, min13, _, _ = run_constant(env, 4, seed=0)
    assert steps0 == steps13 == N_STEPS, "episode must last 16 steps"
    assert min13 < min0, "high UF must lower blood pressure"
    assert run_constant(env, 2, 7) == run_constant(env, 2, 7), "same seed must reproduce"

    # constant-UF policies over 100 patients (rule-based baseline preview)
    for a, u in enumerate(ACTIONS):
        res = [run_constant(env, a, seed) for seed in range(100)]
        print(f"UF {u:>2} mL/kg/h: mean reward {np.mean([x[2] for x in res]):6.2f} | "
              f"hypotension sessions {np.mean([x[3] for x in res]):4.0%} | "
              f"mean min SBP {np.mean([x[1] for x in res]):5.1f}")
    print("sanity checks passed")
