"""Check Paper 1: documents, methods text against the code, and every quantitative claim whose truth is not guaranteed
by construction. Exit code = number of failures.

Final-review round: the values that involve a GBDT, the tree grid, size budgets, model selection or latency come from
results_p1final (run_p1_final.py / bench_p1_final.py, queue logs/p1final_queue.sh) and are withheld by
compute_values.py until every final group is complete (values.json: p1final_complete). Until then the documents
contain ⟦…⟧ placeholders, which are reported as FAILURES ('pending p1final'), and the claims that depend on the final
runs are listed as PEND. Order: compute_values.py -> make_figures.py -> build.py [--anon] -> scrub.py -> verify.py"""
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1]
BL = PIPE / 'experiments' / 'baseline'
RES = BL / 'results_v2'
CODE = (BL / 'run_p1_v2.py').read_text(encoding='utf-8')
FCODE = (BL / 'run_p1_final.py').read_text(encoding='utf-8')
RBCODE = (BL / 'gbdt_rollback.py').read_text(encoding='utf-8')
RBTEST = (BL / 'test_gbdt_rollback.py').read_text(encoding='utf-8')
BENCH = (BL / 'bench_p1_final.py').read_text(encoding='utf-8')
CFG = (BL / 'config.py').read_text(encoding='utf-8')
V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
src = (HERE / 'Manuscript.md').read_text(encoding='utf-8')
filled = HERE / 'Manuscript_filled.md'
text = filled.read_text(encoding='utf-8') if filled.exists() else src
DSS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
HARD = ('TON_IoT', 'UNSW_NB15')
FINAL = bool(V.get('p1final_complete'))
fails, pending = [], []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def pend(msg):
    print('  PEND ' + msg)
    pending.append(msg)


def num(k):
    return float(str(V[k]).replace('−', '-').replace(',', '').rstrip('%'))


print(f"p1final status: {V.get('p1final_status')}  (complete: {FINAL})")
print('Documents')
for name in ('Manuscript_v2.docx', 'Manuscript_v2_anon.docx'):
    p = HERE / name
    if not p.exists():
        check(False, f'{name} not built'); continue
    d = Document(p)
    body = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
    n_ph = body.count('⟦')
    check(n_ph == 0 and '{{' not in body, f'{name}: no unfilled placeholders' + ('' if FINAL else f' ({n_ph} pending p1final)'))
    if 'anon' in name:
        check(not any(n in body for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol'))
              and not d.core_properties.author, f'{name}: anonymous')
todo = re.findall(r'⟦([^⟧]{0,60})', src)
check(not todo, 'Manuscript.md: no ⟦…⟧ to-do markers')
check('companion stud' not in src, 'no uncitable references to companion studies')
out = subprocess.run([sys.executable, str(HERE / 'check_citations.py')], capture_output=True, text=True).stdout
check('not in references: []' in out and 'but not cited: []' in out, 'every citation has a reference and vice versa')
refs = src.split('## References')[1]
nodoi = [r[:60] for r in refs.strip().split('\n\n') if r.strip() and 'doi.org' not in r
         and not re.search(r'USENIX|PMLR|NeurIPS|Neural Information Processing|ICLR|arXiv|Proceedings|http', r)]
check(not nodoi, f'every reference has a DOI or a recognised venue/URL ({nodoi[:3]})')
import build  # noqa: E402  (captions, highlights, abstract limit)
hl = [build.fill(h, V)[0] for h in build.HIGHLIGHTS]
check(3 <= len(hl) <= 5 and all(len(h) <= 85 for h in hl) and not any('⟦' in h for h in hl),
      'highlights: 3–5, filled, each ≤ 85 characters' + ('' if FINAL else ' (pending p1final)'))
abstract = text.split('## Abstract')[1].split('**Keywords')[0]
check(len(abstract.split()) <= build.ABSTRACT_MAX_WORDS, f'abstract ≤ {build.ABSTRACT_MAX_WORDS} words ({len(abstract.split())})')
check('earlier version' not in abstract and 'earlier draft' not in abstract and 'earlier' not in text.split('## 7. Conclusion')[1].split('## Declarations')[0],
      'no earlier-version framing in the abstract and conclusion (referee M7)')
fig = HERE / 'figures' / 'fig1_accuracy_size.png'
P = json.loads((HERE / 'points.json').read_text())
check(fig.exists() and FINAL and P.get('_final'), 'Figure 1 regenerated from the final (rolled-back) points' + ('' if FINAL else ' (pending p1final)'))
check('Figure 1' in src and 'Figure 2' not in src, 'Figure 1 referenced, no stray Figure 2')
check('a few hundred small trees' not in src, 'no "a few hundred small trees" (numbers review W2)')
check('the attack detection rate' in src and "'Attack detection'" in (HERE / 'compute_values.py').read_text(encoding='utf-8'),
      'attack detection rate announced and reported in Table A.1 (numbers review W1)')
check('Up to ~100 KB' in build.HIGHLIGHTS[0] or 'hl_budget' in build.HIGHLIGHTS[0], 'highlight 1 computed per dataset outcome (numbers review O5)')
check('include a full-integer int8 MLP on TON-IoT' in src and 'among the ensembles, a quantised LightGBM model of 74 KB' in src,
      'De Bernardis et al. described as in their abstract (refs review R1, R2)')

print('§3–4 methods text against the code')
cfg_src = CODE.split('TREE_CFGS = {')[1].split('\n}')[0]
check("for n in (25, 100) for d in (8, 12, 16)" in cfg_src and 'n_estimators=200, max_depth=None' in cfg_src
      and 'n_estimators=200, max_depth=20, min_samples_leaf=2' in cfg_src,
      'RF grid in code = 25/100 trees × depth 8/12/16, 200 unlimited, earlier 200/20/leaf 2')
check('25 or 100 trees of maximum depth 8, 12 or 16; 200 trees of unlimited depth; and the configuration of the earlier version (200 trees, depth 20, minimum leaf size 2)' in src,
      'RF grid stated as in the code')
check("f'dt_d{d}': ('dt', dict(max_depth=d)) for d in (4, 6, 8, 12)" in FCODE and 'for n in (10, 25) for d in (4, 6)' in FCODE
      and 'single decision trees** of depth 4, 6, 8 or 12 and shallow forests of 10 or 25 trees of depth 4 or 6' in src,
      'decision trees and shallow forests: code and text agree')
check('for l in (7, 15, 31, 63)' in cfg_src and 'learning_rate=0.1' in cfg_src and 'max_iter=1000' in CODE
      and 'n_iter_no_change=20' in CODE and 'validation_fraction=0.1' in CODE, 'GBDT settings in code')
check('at most 7, 15, 31 or 63 leaves per tree, at learning rates 0.1 (the library default) and 0.05, up to 1,000 iterations with early stopping after 20 iterations without improvement of the log-loss on an internal 10% split of the oversampled training set' in src,
      'GBDT settings stated as in the code')
check('model._predictors = model._predictors[:k]' in RBCODE and 'int(np.argmax(vs))' in RBCODE
      and "GB_CFGS = {k: v for k, v in R.TREE_CFGS.items() if v[0] == 'gb'}" in FCODE and 'rollback=True' in FCODE,
      'rollback to argmax(validation_score_) applied to every GBDT of the grid (code)')
check('refitted with max_iter=best_iter' in RBTEST and 'gives the same model as refitting with that number of iterations' in src,
      '"same model as refitting" claim backed by test_gbdt_rollback.py')
check('the learning rate of 0.05 was added after we had seen the instability' in src.lower(), 'post-hoc lr 0.05 grid disclosed (referee M3)')
check('CAP_ITERS, CAP_LEAVES, CAP_LR = (50, 100, 200, 400), (7, 15, 31), (0.1,)' in FCODE
      and 'cap the iterations at 50, 100, 200 or 400 with 7, 15 or 31 leaves (learning rate 0.1' in src,
      'capped GBDT grid: code and text agree')
check("validation_fraction=None" in FCODE and "X_val=Xv[:, cols]" in FCODE and 'Early stopping on the provided validation split' in src,
      'early stopping on the validation split (sensitivity): code and text')
check('6 * n_int + 2 * n_leaf + 4 * edges' in FCODE and '(6 bytes per internal node), a 16-bit float per leaf' in src,
      'quantised GBDT encoding: code and text agree')
check("'lr': ('lr', dict(C=1.0))" in CODE and "D['Qfit'], D['Qte']) if kind == 'lr'" in CODE, 'LR: C = 1 on quantile features (code)')
check('C = 1) on the quantile-transformed features' in src, 'LR stated as in the code')
tree = ast.parse(CODE)
const = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
         if isinstance(t, ast.Name) and isinstance(n.value, (ast.Constant, ast.Tuple))
         and all(isinstance(e, ast.Constant) for e in getattr(n.value, 'elts', [n.value]))}
for n in tree.body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and isinstance(n.value, ast.Tuple):
        for t, v in zip(n.targets[0].elts, n.value.elts):
            if isinstance(v, ast.Constant):
                const[t.id] = v.value
check((const['EPOCHS'], const['PATIENCE'], const['BATCH'], const['LR']) == (100, 15, 256, 1e-3),
      'network budget in code = 100 epochs, patience 15, batch 256, lr 1e-3')
check('batch size 256, for at most 100 epochs with early stopping on the validation loss (patience 15)' in src
      and 'learning rate 10⁻³, halved after 5 epochs' in src, 'network budget stated as in the code')
check('choice(len(Xf), 500, replace=False)' in CODE and 'calibrated on 500 records of the oversampled training set' in src,
      'int8 calibration: 500 records of the oversampled set')
check('int8 weights and activations with float input and output' in src and 'supported_ops' not in CODE,
      'int8 conversion described as float-fallback with float I/O (refs review m2)')
check('TFLITE_BUILTINS_INT8' in BENCH and 'inference_input_type = tf.int8' in BENCH and 'full-integer quantisation (int8 operations only, int8 input and output' in src,
      'full-integer control: code and text')
check('Conv1D(r(32), 3' in CODE and 'Conv1D(r(64), 3' in CODE and 'Dense(r(128)' in CODE and 'Dropout(0.3)' in CODE,
      'CNN layers in code = 32/64 filters, kernel 3, dense 128/64, dropout 0.3')
check("('rf_t100_d16', 'rf_v1', 'gb_l31', 'gb05_l31', 'cnn_w1', 'mlp_w1')" in CODE and
      'medium random forest (100 trees, depth 16), the earlier forest configuration, GBDTs with 31 leaves at both learning rates and both networks at width 1' in src,
      'port ablation models as in the code')
check("yield 'cnn_w1', 'cnn_w1_minmax', seed, allc" in CODE and 'CNN of width 1 is also trained on the min–max scaled features' in src,
      'min-max control: CNN w1 on min-max inputs (code and text)')
check('JIT = True' in CODE and 'XLA does not use the seed for dropout masks' in src, 'XLA training and its non-reproducibility stated')
check('range(200)' in BENCH and 'warm=20' in BENCH and 'repeats=5' in BENCH and "'OMP_NUM_THREADS'" in BENCH
      and 'threadpool_limits(1)' in BENCH and 'set_intra_op_parallelism_threads(1)' in BENCH and 'num_threads=1' in BENCH,
      'latency protocol in code: all libraries on one thread, 200 calls, 20 warm-up, 5 repeats')
check('median over five repeats of 200 calls after 20 warm-up calls' in src and 'thread counts are set to one before the libraries are loaded' in src,
      'latency protocol stated as in the code')
check('lgb.LGBMClassifier(n_estimators=n_it, num_leaves=p' in BENCH and 'LightGBM model, a tree library compiled to C++' in src,
      'compiled tree runtime control: code and text')
check('internal node = int16 feature + float32 threshold + 2 x uint16 child = 10 bytes' in CODE
      and '10 bytes per internal node (a 16-bit feature index, a 32-bit threshold and two 16-bit child indices)' in src,
      'compact tree encoding stated as in the code')
check('rfecv_subsample     = 50_000' in CFG and 'random subsample of 50,000 training records' in src, 'RFECV subsample stated (refs review m5)')
check('not the data split or the oversampled training set' in src, 'seeds described correctly (SMOTE is fixed)')
check("vk = 'vf1' if basis == 'mean' else 'vf1_0'" in (HERE / 'compute_values.py').read_text(encoding='utf-8')
      and 'we select, on the validation split' in src, 'model selection on validation macro-F1 (code and text; referee M3)')

print('§5 results: independent of the final runs')
f = num
check(f('TON_IoT_gb01_min') < 0.5 and f('TON_IoT_gb05_min') < 0.85 and V['TON_IoT_gb05_collapsed'].startswith('1 '),
      'v2 GBDTs before rollback: lr 0.1 down to < 0.5 on TON, one lr 0.05 seed collapsed')
check(f('TON_gbl7_s0_last_loss') > 1000 * f('TON_gbl7_s0_best_loss') and f('TON_gbl7_s0_f1') < 0.6,
      'diverged v2 model: validation loss exploded after the best iteration (C1)')
for ds in DSS:
    check(f(f'{ds}_cnn_int8_drop_max') <= 0.01, f'{ds}: CNN int8 cost <= 0.01 (mean over seeds)')
check(f('UNSW_NB15_mlp_int8_drop_max') > f('UNSW_NB15_cnn_int8_drop_max') and f('N_BaIoT_mlp_int8_drop_max') > f('N_BaIoT_cnn_int8_drop_max'),
      'MLPs lose more than CNNs under int8 (UNSW, N-BaIoT)')
lo, hi = (float(x) for x in V['int8_size_ratio'].split('–'))
check(hi < 4, 'int8 shrink factor below the nominal four')
check(85 <= f('UNSW_NB15_int8_agree_min') <= 95, '"about one record in ten" differs (UNSW agreement 85–95%)')
check(V['UNSW_NB15_best_net'].startswith('int8') and f('UNSW_NB15_cnn_w2_int8_gain') > 0, 'UNSW: best network is int8 and above its float32 version')
check(90 <= f('TON_IoT_rf_v1_pickle_kb') / 1024 <= 100, 'earlier forest pickled ~95 MB')
lo, hi = (float(x) for x in V['rf_pickle_ratio'].split('–'))
check(9 <= lo and hi <= 11, 'pickled forest about ten times its compact size')
for c in ('rf_t100_d16', 'rf_v1', 'cnn_w1', 'mlp_w1'):
    d = f(f'TON_noports_{c}')
    check(-0.02 <= d <= -0.01 and V[f'TON_noports_{c}_worst_cls'] == 'ransomware', f'ports: {c} falls by about 0.015 ({d:+.3f}), ransomware worst')
for ds in HARD:
    check(f(f'{ds}_cnn_q_minus_minmax') > 0, f'{ds}: quantile inputs better than min-max for the CNN')

print('§5 results: final runs')
if FINAL:
    import statistics as st
    pts = {ds: P[ds] for ds in DSS}
    is_net = lambda p: p['cfg'].startswith(('cnn', 'mlp'))
    is_tree = lambda p: not is_net(p) and p['cfg'] != 'lr'
    for ds in HARD:
        check(f(f'{ds}_tree_net_gap') > 0 and f(f'{ds}_clean_tree_minus_net') > 0, f'{ds}: best tree above best network (all and clean records)')
        check(V[f'{ds}_best_tree_cfg'].startswith('gb'), f'{ds}: the best tree model is a GBDT (§5.4 "the best GBDT")')
        bt = max((p for p in pts[ds] if is_tree(p)), key=lambda p: p['f1'])
        ub = next(p for p in pts[ds] if p['cfg'] == 'rf_t200_dNone')
        check(ub['f1'] <= max(p['f1'] for p in pts[ds] if is_tree(p) and p['size'] < ub['size']) + 0.0005,
              f'{ds}: unbounded forest not more accurate than the best smaller tree ("never more accurate")')
        check(f'{ds}_tree_over_net_kb' in V, f'{ds}: a tree model above the best network exists')
        check(V[f'{ds}_budget_winner'] in ('network', 'tree', 'tie') and 'abs_budget' in V, f'{ds}: budget outcome computed ({V.get(f"{ds}_budget_winner")})')
        _pb = float(V[f'{ds}_budget_welch_p'].replace('< ', '').replace('<', ''))   # 'tie' exactly when Welch p >= 0.05
        check((V[f'{ds}_budget_winner'] == 'tie') == (_pb >= 0.05 and '<' not in V[f'{ds}_budget_welch_p']),
              f'{ds}: budget tie <=> Welch p >= 0.05 (p {V[f"{ds}_budget_welch_p"]})')
        if f'{ds}_tree_over_net_sig_welch_p' in V:
            check(V[f'{ds}_tree_over_net_sig_welch_p'].startswith('<') or float(V[f'{ds}_tree_over_net_sig_welch_p']) < 0.05,
                  f'{ds}: "significantly more accurate than every network" from the stated size (p {V[f"{ds}_tree_over_net_sig_welch_p"]})')
    L = json.loads((Path(os.environ.get('P1_FIN') or BL / 'results_p1final') / 'latency.json').read_text())
    check(L.get('threads') == 1 and L.get('env', {}).get('OMP_NUM_THREADS') == '1', 'latency: one thread for every library')
    for ds in HARD:
        others = [e['single_ms'] for c, e in L[ds].items() if isinstance(e, dict) and c.startswith(('cnn', 'mlp', 'lr'))]
        check(f(f'{ds}_lat_best_tree_ms') >= max(others), f'{ds}: best GBDT slower than every network and LR ("the most accurate tree models are the slowest")')
    nets = [e for ds in DSS for c, e in L[ds].items() if c.startswith(('cnn', 'mlp')) and isinstance(e, dict)]
    check(all(e['int8_single_ms'] > e['tflite_f32_single_ms'] for e in nets), '"int8 saves memory, not time": int8 slower than float32 TFLite everywhere')
    check(f('lat_int8_delegate_maxdiff') < 15, 'int8 times hardly change without XNNPACK (supports the kernel explanation)')
    check(int(V['N_BaIoT_n_998']) > int(V['n_grid_configs']) / 2, 'N-BaIoT saturated (more than half the main-grid configurations >= 0.998)')
    check(int(V['gb_diverged_n']) > 0, 'some GBDT runs diverged in the discarded iterations')
    check(float(V['lgbm_tree_speedup'].split('–')[0]) > 1, 'compiled tree library cheaper per tree than scikit-learn')
    check(all(len(h) <= 85 for h in hl), 'highlights ≤ 85 characters with final values')
else:
    for k in ('tree-over-network ranking, budget outcome, overtaking point, selection on validation', 'rolled-back GBDT collapse counts and cap counts',
              'single-thread latency of all families, int8/XNNPACK/full-int8 diagnostics, LightGBM proxy', 'GBDT port ablation after rollback',
              'quantised GBDT encoding', 'early stopping on the validation split', 'Figure 1 with rolled-back and capped GBDTs'):
        pend(k)

print(f'\n{len(fails)} failure(s), {len(pending)} pending')
for x in fails:
    print('  FAIL', x)
sys.exit(len(fails))
