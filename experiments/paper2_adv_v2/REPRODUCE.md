# Paper 2: reproducing the results

"What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model"

## Environment

Every result was computed with the versions in `requirements_paper2.txt`:
- Python 3.10.11 on Windows 11.
- An NVIDIA GTX 1650 (4 GB) with driver 610.60, used for the networks.

Retraining on another GPU, driver or TensorFlow version will not reproduce the networks bit for bit.

The scikit-learn version matters for the gradient-boosted trees. `HistGradientBoostingClassifier` stops early on an internal 10% validation split, so a different scikit-learn version can stop at a different iteration.

## Saved models

All 220 trained models are listed with their SHA-256 hashes in `MODELS_SHA256.txt`. The `.keras` files are the networks and the `.joblib` files are the trees. Together they take 1.5 GB, too much for this repository; they are available from the authors on request.

Every result file can be checked against the model it came from. `run_adv_r2.py` asserts that a loaded model reproduces the detected-flow set stored in the matching `results_v3` file.

## Run order

Run all commands from `Papers pipeline/` with `python -m experiments.paper2_adv_v2.<script> ...`. Every step skips work that has already finished, so an interrupted queue can simply be restarted.

**1. Train and evaluate (`run_adv_v3.py --dataset D --group G`)**

This step writes `models_v3/` and `results_v3/`. Run the groups in this order:
1. `base`
2. `defend_nets`
3. `defend_trees`
4. `nottl` (UNSW-NB15 only)
5. `splits`
6. `more_seeds`
7. `r2_nets`
8. `r2_trees`
9. `r3_sweep` (TON-IoT only)

**2. Further evaluation of the saved models (`run_adv_r2.py --dataset D --tags T --parts P`)**

This step writes `results_r2/`. It needs no retraining. The parts are:
- `union`, `fs`, `op`, `feas` for every model;
- `sat` for the trees;
- `opu` for the undefended models: the undefended model under the threshold rule at its own FPR;
- `bpa` for the networks: BPDA action-space PGD;
- `relax` for the undefended models: the consistency-relaxed realistic attack.

The trees use `--seeds 0,1,2`. Add `--nottl` for the detectors without TTL features.

**3. Tables, figures and checks (in the manuscript folder)**

Run these in order:
1. `compute_values.py`
2. `make_figures.py`
3. `build.py`, then `build.py --supp`
4. `verify.py`

`verify.py` re-checks every quantitative statement in the text against the result files. Its exit code is the number of failures.

The queue scripts that ran these steps are `logs/p2v4_*.sh`, `logs/p2r2_*.sh` and `logs/p2r3*.sh`.

## Compute time (one laptop, sum of the recorded per-run times in `results_v3`)

| Dataset | Family | Runs | Training (h) | Attack evaluation (h) |
|---|---|---|---|---|
| TON-IoT | networks | 51 | 30.2 | 9.5 |
| TON-IoT | trees | 46 | 3.0 | 18.7 |
| UNSW-NB15 | networks | 57 | 13.9 | 16.6 |
| UNSW-NB15 | trees | 52 | 5.3 | 81.2 |

The evaluations in `run_adv_r2.py` add about 15 h on top of this.

## Decision rule (operating point)

The models predict the class with the highest probability (argmax). The matched-operating-point analysis (`op`, `opu`) uses a different rule: a flow is benign if its benign probability is at least a threshold set on the validation benign flows.

The undefended reference is evaluated under the same threshold rule, at its own validation FPR (`opu`). It is not evaluated with argmax.
