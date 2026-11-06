# M4: Integrated Gradients and presentation handoff

Owner per repository: Mohanad Mostfa Ali.

## Install and run

Run these commands from the repository root. `python main.py` already trains the models, runs M3's sign-off and writes the trained seed-0 DDQN audit to `results/xai/`, then the cohort-reference audit (age/albumin contrast) to `results/xai_cohort/`. The commands below run `xai_ig.py` on its own.

```sh
python -m pip install -r requirements.txt
python xai_ig.py --demo --sessions 10 --output results/xai_demo
```

Demo uses a fixed random untrained network. Its pictures demonstrate the software and support no conclusions about learned dialysis behavior.

For a trained policy, M2 exports its online Q-network to TorchScript:

```python
online_q.eval()
traced = torch.jit.trace(online_q.cpu(), torch.zeros(1, 8))
traced.save('models/ddqn_scripted_seed0.pt')
```

M3 supplies `results/xai_approval.json` after reviewing the convergence plots and five-seed stability report:

```json
{
  "model_sha256": "REPLACE_WITH_SHA256_OF_EXPORTED_MODEL",
  "convergence_passed": true,
  "stability_passed": true,
  "training_seeds": [0, 1, 2, 3, 4],
  "report": "validation_report.md"
}
```

The report path is relative to the evidence JSON. This checks M3's declaration, not the scientific validity of that report. The hash ties approval to the exact export.

```sh
python xai_ig.py --model models/ddqn_scripted_seed0.pt --evidence results/xai_approval.json --sessions 100 --seed 10000 --output results/xai
```

M2 must preserve the existing normalized state order, action index order, and preprocessing. The CLI uses the existing simulator. To explain another PyTorch network directly, import `explain(model, normalized_state, baseline)`.

## Outputs and interpretation

- `decision.png`: signed feature contributions to the chosen action's Q-value at the first step.
- `session.png`: signed attribution heatmap for the first session.
- `cross_session.png`: session-weighted mean absolute contributions with session standard deviations.
- `attributions.csv`: states, actual baselines, selected actions, contributions, completeness residuals, reward and hypotension flag.
- `feature_summary.csv`: aggregate signed and absolute contributions.
- `manifest.json`: mode, model hash, session seeds, baseline and maximum residual.

The action target stays fixed while integrating from the reference to the observed state. A positive attribution increases that Q-value relative to the reference. It does not establish causation or explain why that action beats every alternative. Q-value scale depends on the learned reward.

The default baseline preserves age, albumin and time, with reference vitals and zero overload/removal. Those preserved features have zero IG by construction. Therefore **this run cannot measure reliance on age or albumin**. Run a second audit with an explicit cohort reference to inspect those features:

```sh
python xai_ig.py --model models/ddqn_scripted_seed0.pt --evidence results/xai_approval.json --reference reference.json --output results/xai_cohort
```

`reference.json` contains eight normalized numbers. Construct it from the training cohort only, record its provenance, and inspect whether its straight-line paths leave the plausible state region. Compare rankings across both references. Elevated age/albumin attribution is a review prompt, not automatically an error.

Numerical completeness requires `abs(sum(IG) - (Q(x,a)-Q(b,a))) <= 1e-4 + 1e-3*abs(Q difference)`. Integration retries up to 32 times the requested resolution and fails instead of silently accepting a poor result.

Session seeds 10000 onward avoid the five training RNG seeds, but IG sessions in `main.py` draw from all 400 UCI profiles, including the 320 training profiles. The patient-level split in `train.py` (80 held-out profiles) applies to evaluation, not to IG, so do not describe IG results as unseen patients.

## Known project limitations to discuss

The existing environment synthesizes weight, systolic pressure and serum albumin. It maps urine albumin grade to synthetic serum albumin using a hand-chosen formula. Do not describe these values as measured serum albumin. Physiology coefficients are hand-tuned, so explanations describe this simulator and reward. The eight inputs also omit previous action and baseline parameters used internally. Discuss partial observability with M1/M2 before training.

## Team handoff

Trained models, M3's validation report and both IG audits are produced by `python main.py`. M4 inspects both audits for clinically relevant patterns and hands slide content and figures to Yousef, who builds the deck (see `TEAM_PLAN.md`). `slides_m4.md` holds that M4 content.
