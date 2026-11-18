"""Reproduce the evidence for Paper 7's four headline findings from the original files, in seconds, on CPU.

    python reproduce_headline.py          (from Revision_v2/Paper7/checks, with the project's Python env)

F4  silent CNN failure: min-max stream-1 CNN vs RF on the same 7 features (Study 6, first re-run, TON_IoT)
F5  tf.keras EarlyStopping(restore_best_weights=True) keeps last-epoch weights when early stopping never fires
F7  original DP-FedAvg: loss reduced outside GradientTape -> all gradients None -> no update; frozen history
F17 original integrated framework: headline 0.7455 vs its own 3-seed rerun; population vs sample s.d.
Exit code = number of findings NOT reproduced.
"""
import glob
import inspect
import json
import os
import statistics as st
import subprocess
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')     # CPU only
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

HERE = Path(__file__).resolve().parent                    # .../paper7/checks
if HERE.parents[1].name == 'papers':                      # public repository: <repo>/papers/paper7/checks
    PIPE = HERE.parents[2]
    V1 = HERE.parent / 'original_v1' / 'experiments'
else:                                                     # authors' working tree
    ROOT = HERE.parents[2]                                # IDS_IOT_Research
    PIPE = ROOT / 'Papers pipeline'
    V1 = ROOT / '_v1_backup_2026-10-24' / 'experiments'
bad = 0


def jl(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def report(name, ok, lines):
    global bad
    print(f'\n=== {name}: {"REPRODUCED" if ok else "NOT REPRODUCED"}')
    for l in lines:
        print('   ', l)
    bad += not ok


# ---------------------------------------------------------------- F4
R = PIPE / 'experiments/paper6_multimodal/results_v2/TON_IoT'
cnn = [jl(p) for p in sorted(glob.glob(str(R / 's1/seed*.json')))]
rf = [jl(p) for p in sorted(glob.glob(str(R / 'rf_s1/seed*.json')))]
cf, rff = [r['macro_f1'] for r in cnn], [r['macro_f1'] for r in rf]
be = [r['best_epoch'] for r in cnn]
fpr = st.mean(r['benign_fpr'] for r in cnn)
report('F4 silent CNN failure on min-max inputs', st.mean(cf) < 0.1 and st.mean(rff) > 0.9, [
    f'CNN stream 1, per seed macro-F1: {[round(x, 4) for x in cf]} (mean {st.mean(cf):.3f}); best epochs {be}; benign FPR {fpr:.3f}',
    f'RF on the same features, per seed: {[round(x, 4) for x in rff]} (mean {st.mean(rff):.3f})',
    f'files: {R / "s1"}, {R / "rf_s1"}'])

# ---------------------------------------------------------------- F5
import numpy as np
import tensorflow as tf

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
m = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
es.set_model(m)
es.on_train_begin()
for epoch, (w, vloss) in enumerate([(1.0, 1.0), (2.0, 0.5), (3.0, 0.9)]):   # best epoch = 2 (w = 2.0)
    m.layers[0].set_weights([np.array([[w]]), np.array([0.0])])
    es.on_epoch_end(epoch, {'val_loss': vloss})
es.on_train_end()
final_w = float(m.layers[0].get_weights()[0][0, 0])
src_end = inspect.getsource(type(es).on_train_end)
report(f'F5 Keras {tf.keras.__version__ if hasattr(tf.keras, "__version__") else tf.__version__} EarlyStopping default',
       final_w == 3.0 and 'set_weights' not in src_end, [
           'three epochs, patience 10 (early stopping never fires), best val_loss at epoch 2 (weight 2.0)',
           f'weight after on_train_end: {final_w} (last epoch = 3.0, best = 2.0)',
           f'set_weights in EarlyStopping.on_train_end: {"set_weights" in src_end}',
           f'TensorFlow {tf.__version__}; source: {inspect.getsourcefile(type(es))}'])

# ---------------------------------------------------------------- F7
v = tf.Variable(1.0)
with tf.GradientTape() as t:
    per_sample = v * tf.constant([2.0, 3.0])
g = t.gradient(tf.reduce_mean(per_sample), v)          # reduce_mean outside the tape, as in the original code
try:
    try:   # authors' tree: the original code at its snapshot commit
        code = subprocess.run(['git', '-C', str(PIPE), 'show', '7183c0e:experiments/paper4_federated/run_federated.py'],
                              capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:   # public repository: the file is unchanged since the original study (same text as 7183c0e)
        code = (PIPE / 'experiments/paper4_federated/run_federated.py').read_text(encoding='utf-8').splitlines()
    excerpt = [f'{i + 1}: {code[i]}' for i in (186, 187, 188, 200, 201, 202, 207, 208)]
except Exception as e:                                    # pragma: no cover
    excerpt = [f'(git show failed: {e})']
h = [x['f1'] for x in jl(V1 / 'paper4_federated/results/dp_fedavg_eps13.json')['history']]
report('F7 original DP-FedAvg never trained', g is None and max(h[1:]) - min(h[1:]) < 0.002, [
    f'toy: gradient of reduce_mean computed outside the tape = {g}',
    'original code (commit 7183c0e, run_federated.py):', *['  ' + l for l in excerpt],
    f'original DP history (sigma 1.0), macro-F1 per checkpoint: {[round(x, 4) for x in h]}'])

# ---------------------------------------------------------------- F17
base = V1 / 'paper7_integrated/results/TON_IoT'
head = jl(base / 'integrated_results.json')['conditions']['C4']['full_after_adapt']['f1']
ms = jl(base / 'multiseed_c4.json')
seeds = {s: ms[s]['C4']['full_after_adapt']['f1'] for s in ('42', '123', '456')}
vals = list(seeds.values())
report('F17 headline not among own seeds', all(abs(head - x) > 0.002 for x in vals), [
    f'headline (integrated_results.json, C4 full_after_adapt f1): {head:.4f}',
    f'three-seed rerun (multiseed_c4.json): {seeds}',
    f'mean {st.mean(vals):.4f}; population s.d. {st.pstdev(vals):.4f} (printed: 0.2559); sample s.d. {st.stdev(vals):.4f}',
    f'single-class check, seed 42 clean metrics: {ms["42"]["C4"]["clean_metrics"]}'])

print(f'\n{4 - bad} of 4 findings reproduced')
raise SystemExit(bad)
