import subprocess
import sys

import torch
import evaluate
from main import FULL, train_all

torch.set_num_threads(1)
cfg = FULL
cfg["models"].mkdir(exist_ok=True)
cfg["results"].mkdir(exist_ok=True)
train_all(cfg)
if not evaluate.run(cfg):
    sys.exit("Validation did not pass. See results/validation_report.md.")
subprocess.run(
    [sys.executable, "xai_ig.py", "--model", "models/ddqn_scripted.pt",
     "--evidence", "results/xai_approval.json", "--sessions", "100",
     "--output", "results/xai"],
    check=True,
)
