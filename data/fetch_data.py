"""Download UCI CKD (id 336) once and save it as data/ckd.csv. Needs: pip install ucimlrepo"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=336)
df = ds.data.features.join(ds.data.targets)
df.to_csv(Path(__file__).parent / "ckd.csv", index=False)
print(df.shape)
