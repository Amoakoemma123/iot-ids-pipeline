"""Check Paper 2 v4 (after review round 2): documents, methods text against run_adv_v3.py, and (once results exist) every quantitative claim.
Exit code = number of failures; checks needing results not yet produced are PEND.
Order: compute_values.py -> make_figures.py -> build.py [--anon] -> verify.py"""
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1]
RES = PIPE / 'experiments' / 'paper2_adv_v2' / 'results_v3'
CODE = (PIPE / 'experiments' / 'paper2_adv_v2' / 'run_adv_v3.py').read_text(encoding='utf-8')
R2CODE = (PIPE / 'experiments' / 'paper2_adv_v2' / 'run_adv_r2.py').read_text(encoding='utf-8')
V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
src = (HERE / 'Manuscript.md').read_text(encoding='utf-8')
src += '\n' + (HERE / 'Supplement.md').read_text(encoding='utf-8')   # final review round 2: method details moved to S18
filled = HERE / 'Manuscript_filled.md'
text = filled.read_text(encoding='utf-8') if filled.exists() else src
DSS = ('TON_IoT', 'UNSW_NB15')
BASE = ('cnn_none', 'mlp_none', 'rf_none', 'gb_none')
DEF = ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades', 'mlp_at_threat', 'rf_augment', 'gb_augment',
       'rf_augment2', 'gb_augment2', 'rf_augment_nodelay', 'gb_augment_nodelay')
FIVE = {'cnn_at_threat', 'rf_augment', 'gb_augment', 'cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_trades'}
fails, pending = [], []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def pend(msg):
    print('  PEND ' + msg)
    pending.append(msg)


def const(name):
    for node in ast.parse(CODE).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(t, ast.Tuple) and name in [e.id for e in t.elts]:
                    return ast.literal_eval(node.value)[[e.id for e in t.elts].index(name)]
    raise KeyError(name)


def done(models):
    return all((RES / ds / m / f'seed{s}.json').exists() for ds in DSS for m in models for s in (range(5) if m.endswith('_none') or m in FIVE else range(3)))


BASE_DONE, DEF_DONE = done(BASE), done(DEF)
NOTTL_DONE = all((RES / 'UNSW_NB15' / f'{m}_nottl' / f'seed{s}.json').exists() for m in ('cnn_none', 'rf_none', 'gb_none') for s in range(3))
SPLITS_DONE = all((RES / ds / f'{m}_split{k}' / 'seed0.json').exists() for ds in DSS for m in BASE for k in (1, 2))
ALL_DONE = BASE_DONE and DEF_DONE and NOTTL_DONE and SPLITS_DONE

print('Documents')
for name in ('Manuscript_v2.docx', 'Manuscript_v2_anon.docx'):
    p = HERE / name
    if not p.exists():
        pend(f'{name} not built yet'); continue
    d = Document(p)
    body = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
    if ALL_DONE:
        check('⟦' not in body and '{{' not in body, f'{name}: no unfilled placeholders')
    else:
        pend(f'{name}: {body.count(chr(10214))} unfilled placeholders until all runs are done')
    if 'anon' in name:
        check(not any(n in body for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol'))
              and not d.core_properties.author, f'{name}: anonymous')
todo = re.findall(r'⟦([^⟧]{0,50})', src)
(check(not todo, 'Manuscript.md: no ⟦…⟧ to-do markers') if ALL_DONE else pend(f'{len(todo)} ⟦…⟧ to-do markers in Manuscript.md'))
check('companion stud' not in src, 'no uncitable references to companion studies')
out = subprocess.run([sys.executable, str(HERE / 'check_citations.py')], capture_output=True, text=True).stdout
check('not in references: []' in out and 'but not cited: []' in out, f'every citation has a reference and vice versa {out.strip()[:200]}')
refs = (HERE / 'Manuscript.md').read_text(encoding='utf-8').split('## References')[1]
nodoi = [r[:60] for r in refs.strip().split('\n\n') if r.strip() and 'doi.org' not in r
         and not re.search(r'USENIX|PMLR|NeurIPS|Neural Information Processing|ICLR|arXiv|Proceedings|http', r)]
check(not nodoi, f'every reference has a DOI or a recognised venue/URL ({nodoi[:3]})')
import build  # noqa: E402
(check(3 <= len(build.HIGHLIGHTS) <= 5 and all(len(build.fill(h, V)[0]) <= 85 for h in build.HIGHLIGHTS),
       'highlights: 3–5, each ≤ 85 characters') if build.HIGHLIGHTS else pend('highlights not written yet'))
abstract = text.split('## Abstract')[1].split('**Keywords')[0]
(pend('abstract not written yet') if '⟦' in abstract else
 check(len(abstract.split()) <= build.ABSTRACT_MAX_WORDS, f'abstract ≤ {build.ABSTRACT_MAX_WORDS} words ({len(abstract.split())})'))
for n, f in ((1, 'fig1_asr_budget.png'), (2, 'fig2_defences.png'), (3, 'fig3_tree_defences.png'), (4, 'fig4_ttl_sweep.png')):
    check(f'Figure {n}' in src or f'Figures {n} ' in src or f'and {n} ' in src, f'Figure {n} referenced')
    (check((HERE / 'figures' / f).exists(), f'Figure {n} file exists') if ALL_DONE else pend(f'Figure {n} needs results'))

print('§3–4 methods text against the code')
check(const('BETAS') == (0.1, 0.25, 0.5, 1.0, 2.0) and 'β ∈ {0.1, 0.25, 0.5, 1, 2}' in src, 'budgets β as in the code')
check('np.array([1000.0, 10.0, 1.0])' in CODE and 'np.array([1000.0, 10.0, 1.0, 0.1])' in CODE
      and 'β(b + 1,000) payload bytes, β(p + 10) packets, β(d + 1 s) seconds of delay and β(k + 0.1 s) of ACK delay' in src,
      'budget floors as in the code')
check(const('N_EVAL') == 5000 and const('N_TTL') == 4, 'attack set 5,000 flows, 4 TTL candidates (code)')
check("120 if sp.ds == 'TON_IoT' else 48" in CODE and '120 (TON-IoT) or 48 (UNSW-NB15) random points' in src
      and '128 queries per flow and budget on TON-IoT and 320 on UNSW-NB15' in src, 'black-box candidate counts stated as in the code')
check('u + 0.05 * tf.sign(action(' in CODE and 'steps=50' in CODE and '50 steps of 5% of the box' in src, 'white-box action PGD: 50 steps of 0.05')
check(const('EPS_UNC') == (0.01, 0.05, 0.1) and 'ε ∈ {0.01, 0.05, 0.1}, 50 steps' in src, 'basic unconstrained ε grid and steps')
check("for beta in (0.5, 1.0)" in CODE and '(β = 0.5 and 1)' in src, 'transfer budgets')
check(const('BETA_TRAIN') == 0.5 and const('EPS_TRAIN') == 0.05 and 'ε = 0.05, 7 steps of ε/4' in src and '(β = 0.5, 7 sign steps of 0.05 of the box from a uniform random start)' in src,
      'adversarial-training budgets')
check("ROBUST = dict(epochs=30, steps=200, patience=5, bs=256)" in CODE and 'at most 30 epochs of 200 steps' in src
      and 'patience 5' in src and 'clean oversampled batch of 256 records and 128 further records' in src, 'robust training loop as in the code')
check(const('RARE_SHARE') == 0.02 and 'fewer than 2% of the training records' in src, 'CCAT rare threshold 2%')
check('6.0 * tf.reduce_mean' in CODE and '6 × the KL divergence' in src, 'TRADES weight 6')
check(const('AUG_N') == 20000 and 'unit_grid(sp.k, 24, grid_seed, drop)' in CODE and 'up to 20,000 real training attack flows' in src
      and 'box corners and 24 random points' in src, 'tree augmentation as in the code')
check('rng.random(n) >= 0.5' in CODE and 'each flow keeps its TTL with probability 0.5' in src, 'augmentation TTL per flow (fixed 2026-11-02)')
check("n_estimators=100, max_depth=16" in CODE and "max_leaf_nodes=31, learning_rate=0.05" in CODE
      and '100 trees of maximum depth 16' in src and 'at most 31 leaves per tree, learning rate 0.05' in src, 'tree models as in the code')
check("('mlp', 'at_threat', DEF_SEEDS)" in CODE and 'applied to both the CNN and the MLP' in src, 'MLP gets AT-threat only')

print('§4 round-2 methods text against run_adv_r2.py')
check(R2CODE.count('EPS_R2 = (0.005, 0.01, 0.02, 0.05, 0.1)') == 1 and 'ε ∈ {0.005, 0.01, 0.02, 0.05, 0.1}' in src, 'feature-space reference ε grid (r2)')
check('for _ in range(128):' in R2CODE and '128 random corners of the full ball' in src, 'r2: 128 corners of the full ball')
check('itertools.product((-1.0, 0.0, 1.0), repeat=len(mut))' in R2CODE and '{−ε, 0, +ε}' in src and '{0, +ε}' in src, 'r2: exhaustive mutable patterns, increase-only subset')
check('a = eps * 0.25 * 0.1 ** (s / (steps - 1))' in CODE and 'step decaying from 0.25ε to 0.025ε' in src, 'BPDA PGD step schedule')
check('iters=150, pop=32' in R2CODE and '150 rounds of 32 proposals' in src, 'tree saturation: long adaptive search')
check('(5, 9, 17) if sp.k == 3 else (3, 5, 7)' in R2CODE and '17³ points on TON-IoT and 7⁴ on UNSW-NB15' in src, 'tree saturation grids')
check('min(500, n)' in R2CODE and '500 detected flows per seed' in src, 'tree saturation subsample 500')
check('1460.0' in R2CODE and '1500.0' in R2CODE and '1,460 payload bytes per packet' in src and '1,500 bytes per packet including headers' in src, 'MTU bound as in the code')
check('np.round(A[:, 1])' in R2CODE and 'rounded to an integer' in src, 'integer packets')
check("ops['fpr1'] = Thresh(model, normal, tau_for(model, D, 0.01))" in R2CODE and "val_fpr(undef_model, D)" in R2CODE
      and 'vf = val_fpr(model, D)' in R2CODE and 'mdl = Thresh(model, normal, tau_for(model, D, vf))' in R2CODE
      and "it gives the undefended model's FPR (and, separately, 1%)" in src and 'For the undefended model the threshold gives its own validation FPR' in src, 'matched operating point as in the code (undefended under the threshold rule)')
check("ada=kind in ('cnn', 'mlp')" in R2CODE and 'with the grid search and, for the networks, the adaptive search' in src, 'op attack: adaptive for networks only')
check("mn_(mx(b / mx(p, 1.0), 28.0 + zero), 40.0 + zero)" in CODE and "adds 40 bytes, or the flow's mean packet size if that is smaller (but at least 28 bytes)" in src,
      'UNSW header-byte rule described as in the code (W7)')
P6CODE = (PIPE / 'experiments' / 'paper6_multimodal' / 'run_p6_v2.py').read_text(encoding='utf-8')
check('early_stopping=True, validation_fraction=0.1' in P6CODE and "'gb': ('gb'," in CODE, 'GBDT internal early stopping in code (shared make_tree)')
# early-stopping collapse (keeps 20 iterations past the best; seen in Papers 1 and 6): no P2 GBDT seed may fall far below its group
import glob, statistics as _st
bad = []
for ds in DSS:
    for d in sorted((RES / ds).glob('gb_*')):
        f1 = [json.loads(Path(x).read_text())['macro_f1'] for x in sorted(d.glob('seed*.json'))]
        if len(f1) > 1 and min(f1) < _st.median(f1) - 0.03:
            bad.append((ds, d.name, round(min(f1), 3), round(_st.median(f1), 3)))
check(not bad, f'no GBDT seed collapsed (every seed within 0.03 macro-F1 of its median) {bad}')
check("internal early stopping on 10% of the training data" in src, 'GBDT early stopping stated (minor 7)')

print('§5 results')
P = lambda k: float(V[k].rstrip('%'))                       # noqa: E731  percent value
F = lambda k: float(V[k].replace('−', '-').split()[0])      # noqa: E731  signed value (pp or F1)
pv = lambda k: 0.0 if V[k].startswith('<') else float(V[k])  # noqa: E731  p-value
T, U = 'TON_IoT', 'UNSW_NB15'
# 5.1
check(min(float(V[f'{T}_{m}_f1']) for m in ('rf_none', 'gb_none')) > max(float(V[f'{T}_{m}_f1']) for m in ('cnn_none', 'mlp_none')), 'TON: trees most accurate')
check(V[f'{U}_mlp_none_f1'] == min(V[f'{U}_{m}_f1'] for m in BASE) and V[f'{U}_gb_none_f1'] == max(V[f'{U}_{m}_f1'] for m in BASE), 'UNSW F1 range MLP..GBDT')
check(V[f'{U}_gb_none_fpr'] == min(V[f'{U}_{m}_fpr'] for m in BASE) and V[f'{U}_cnn_none_fpr'] == max(V[f'{U}_{m}_fpr'] for m in BASE), 'UNSW FPR range GBDT..CNN')
check(all(P(f'{U}_{m}_miss') < 1 for m in ('rf_none', 'gb_none')), 'UNSW: trees miss < 1%')
check(all(P(f'{T}_{m}_fs_strongest_e0.01') >= 99.95 and P(f'{T}_{m}_bb_corners_e0.01') >= 99.95 for m in BASE), 'TON: corners of the full ball evade every detected flow at 0.01')
check(P(f'{T}_cnn_none_wb_unconstrained_e0.01') < 99, 'TON CNN: white-box eps-PGD below the corner search')
for ds in DSS:
    for m in BASE:
        exp = '≤ 0.005' if not (ds == U and m in ('rf_none', 'gb_none')) else None
        if exp:
            check(V[f'{ds}_{m}_cross_mut'] == exp and V[f'{ds}_{m}_cross_inc'] == exp, f'{ds} {m}: mutable and increase-only eps attack above realistic at 0.005')
check(V[f'{U}_rf_none_cross_mut'] not in ('≤ 0.005',) and V[f'{U}_gb_none_cross_mut'] not in ('≤ 0.005',), 'UNSW trees need a larger ball')
check(V[f'{U}_rf_none_cross_inc'] == '> 0.1', 'UNSW RF increases only: never up to 0.1')
check(float(V['union_plus_gain_max'].split()[0]) < 1.0, 'omitted attacks add < 1 pp to the union')
check(all(F(f'{ds}_{m}_wbs_minus_bb_b0.5') < 0 for ds in DSS for m in ('cnn_none', 'mlp_none')) and F(f'{T}_cnn_none_wbs_minus_wb_b0.5') > 0,
      'strong PGD below grid for both nets on both datasets, above basic PGD')
check(float(V[f'{T}_sat_gain_max'].split()[0]) < 5 and float(V[f'{U}_sat_gain_max'].split()[0]) > 8, 'tree saturation: small on TON, large on UNSW')
check(max(((F(f'{U}_{m}_sat_gain'), m) for m in ('rf_none', 'rf_augment', 'gb_none', 'gb_augment')))[1] == 'gb_augment', 'UNSW: largest saturation gap is the augmented GBDT')
check(float(V['feas_loss_max'].split()[0]) <= 1.5, 'feasibility: at most ~1 pp lost')
for ds in DSS:
    for m in BASE:
        up = P(f'{ds}_{m}_bb_volume_b2.0') > P(f'{ds}_{m}_bb_volume_b0.1')
        check(up if (ds, m) != (U, 'rf_none') else True, f'{ds} {m}: ASR grows with budget')
rf = [P(f'{U}_rf_none_bb_volume_b{b}') for b in (0.1, 0.25, 0.5, 1.0, 2.0)]
check(max(rf) == rf[1] and rf[-1] < rf[1], 'UNSW RF peaks at beta 0.25 (W1 sentence)')
check(P(f'{T}_cnn_none_bb_volume_b0.5') > P(f'{T}_cnn_none_bb_only_bytes_pkts_b0.5') > max(P(f'{T}_cnn_none_bb_only_{a}_b0.5') for a in ('bytes', 'pkts', 'delay')),
      'TON CNN: no single action suffices')
check(P(f'{U}_cnn_none_bb_only_delay_b0.5') > 20 and all(P(f'{U}_{m}_bb_only_delay_b0.5') < 3 for m in ('rf_none', 'gb_none')), 'UNSW: delays alone evade nets, not trees')
check(all(P(f'{T}_{m}_bb_volume_b0.5_floor0.0') < 10 for m in BASE), 'TON: floors x0 -> under 10%')
check(V[f'{U}_rf_none_queries_median'] == V[f'{U}_gb_none_queries_median'] == '65', 'UNSW trees: 65th query (first TTL candidate)')
check(int(V[f'{U}_cnn_none_queries_median']) < 65, 'UNSW CNN needs fewer queries than trees')
check(P(f'{U}_cnn_none_strongest_volume_b0.5_overall') > P(f'{U}_cnn_none_strongest_volume_b0.5'), 'UNSW CNN overall > ASR')
# 5.2
check(all(V[f'{T}_nets_gt_trees_all_{a}'] == 'True' for a in ('strongest_b0.5', 'strongest_volume_b0.5', 'bb_volume_b0.5', 'bb_both_b0.5', 'bb_volume_b2.0')),
      'TON: each net above each tree, every attack in Table 6, significant')
d = {(n, t): F(f'{T}_{n}_vs_{t}_strongest_b0.5') for n in ('cnn_none', 'mlp_none') for t in ('rf_none', 'gb_none')}
check(min(d, key=d.get) == ('cnn_none', 'gb_none') and max(d, key=d.get) == ('mlp_none', 'rf_none'), 'TON strongest gap range CNN-GBDT .. MLP-RF')
check(pv(f'{T}_nettree_strongest_maxp') < 0.01, 'TON nets-trees p < 0.01')
check(V[f'{U}_nets_gt_trees_all_strongest_volume_b0.5'] == 'True', 'UNSW: nets above trees volume-only, significant')
d = {(n, t): F(f'{U}_{n}_vs_{t}_strongest_volume_b0.5') for n in ('cnn_none', 'mlp_none') for t in ('rf_none', 'gb_none')}
check(min(d, key=d.get) == ('cnn_none', 'gb_none') and max(d, key=d.get) == ('mlp_none', 'rf_none'), 'UNSW volume gap range CNN-GBDT .. MLP-RF')
check(F(f'{U}_cnn_none_vs_rf_none_strongest_b0.5') < F(f'{U}_cnn_none_vs_rf_none_strongest_volume_b0.5') and pv(f'{U}_mlp_none_vs_rf_none_strongest_b0.5_p') > 0.05
      and all(F(f'{U}_{n}_vs_gb_none_strongest_b0.5') > 0 and pv(f'{U}_{n}_vs_gb_none_strongest_b0.5_p') < 0.05 for n in ('cnn_none', 'mlp_none')),
      'UNSW with TTL: gap narrows, MLP-RF n.s., GBDT below both nets')
check(all(float(V[f'{U}_{m}_strongest_volume_b0.5_sdpp']) > 10 for m in ('cnn_none', 'mlp_none')) and float(V[f'{U}_rf_none_strongest_volume_b0.5_sdpp']) < 3,
      'UNSW: nets SD > 10 pp, RF < 3 pp')
for ds in DSS:
    for m in ('mlp_none', 'rf_none', 'gb_none'):
        check(P(f'{ds}_{m}_transfer_action_b0.5') < P(f'{ds}_{m}_bb_both_b0.5'), f'{ds} {m}: transfer below query attack')
    check(V.get(f'{ds}_splits_nets_above_trees', '').startswith(('3 of 3', '2 of 3')), f'{ds}: nets above trees on most splits')
# 5.3 (Table S6)
pc = lambda m, c, ds=T: P(f'{ds}_{m}_pc_{c}')  # noqa: E731
check(all(pc(m, c) > 70 for m in BASE for c in ('ransomware', 'scanning')), 'TON: ransomware and scanning evade every model')
four = ('backdoor', 'injection', 'password', 'xss')
check(max(pc('rf_none', c) for c in four) == pc('rf_none', 'password') and min(pc('rf_none', c) for c in four) == pc('rf_none', 'backdoor')
      and pc('rf_none', 'password') < 2, 'TON: RF range backdoor..password, < 2%')
check(min(pc('gb_none', c) for c in four) == pc('gb_none', 'injection') and max(pc('gb_none', c) for c in four) == pc('gb_none', 'password'), 'TON: GBDT range injection..password')
check(all(pc(n, c) > pc('gb_none', c) for n in ('cnn_none', 'mlp_none') for c in four), 'TON: nets evaded more than GBDT on those classes')
cls = [k.split('_pc_')[1] for k in V if k.startswith(f'{U}_rf_none_pc_')]
lo = min(pc(m, c, U) for m in ('cnn_none', 'mlp_none', 'rf_none') for c in cls)
check(f'{lo:.1f}%' == V[f'{U}_rf_none_pc_exploits'], 'UNSW: min over nets+RF is RF exploits')
check({'reconnaissance', 'worms'} <= {c for c in cls if pc('gb_none', c, U) < lo}, 'UNSW GBDT below that on reconnaissance and worms')
# 5.4 defences
print('§5.4 defences')
NETS = ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades')
for ds in DSS:
    check(all(V[f'{ds}_{m}_nseeds'] == '5' for m in NETS + ('rf_augment', 'gb_augment')), f'{ds}: CNN defences and augmented trees have 5 seeds')
    check(all(float(V[f'{ds}_{m}_op_own_test_fpr']) > float(V[f'{ds}_cnn_none_op_own_test_fpr']) for m in NETS), f'{ds}: every CNN defence raises the FPR')
# review round 3: reference = undefended model under the threshold rule at its own FPR; endpoint = missed or evaded
check(all(P(f'{T}_{m}_op_matched_e2e') > P(f'{T}_cnn_none_op_thr_e2e') for m in ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_trades')),
      'TON matched: loop, AT-all, CCAT, TRADES missed/evaded more than undefended')
check(pv(f'{T}_cnn_trades_prim_p') < 0.05 and F(f'{T}_cnn_trades_prim') > 0, 'TON: TRADES significantly worse at matched FPR')
check(all(pv(f'{T}_{m}_prim_p') > 0.05 for m in ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'mlp_at_threat')),
      'TON: no other network defence significant vs undefended')
check(P(f'{T}_cnn_at_threat_op_matched_e2e') < P(f'{T}_cnn_none_op_thr_e2e'), 'TON: AT-threat missed/evaded lower (n.s.)')
check(P(f'{T}_mlp_at_threat_op_matched_e2e') < P(f'{T}_mlp_none_op_thr_e2e'), 'TON: MLP AT-threat lower (n.s.)')
check(all(pv(f'{T}_cnn_at_threat_prim_vs_{o}_p') < 0.05 and F(f'{T}_cnn_at_threat_prim_vs_{o}') < 0 for o in ('cnn_trades', 'cnn_clean_loop'))
      and all(pv(f'{T}_cnn_at_threat_prim_vs_{o}_p') > 0.05 for o in ('cnn_at_all', 'cnn_ccat')),
      'TON: AT-threat significantly better than TRADES and loop; not AT-all, CCAT')
check(P(f'{T}_cnn_trades_op_matched_detected') < 70, 'TON: TRADES detects few at matched threshold')
check(all(F(f'{T}_{m}_none_rule_d_e2e') < 0 for m in ('cnn', 'rf', 'gb')) and F(f'{T}_rf_none_rule_d_e2e') < F(f'{T}_cnn_none_rule_d_e2e')
      and F(f'{T}_gb_none_rule_d_e2e') < F(f'{T}_cnn_none_rule_d_e2e'), 'TON: threshold rule lowers undefended evasion, trees most')
check(all(F(f'{U}_{m}_none_rule_d_e2e_volume') < 0 for m in ('cnn', 'mlp', 'rf')), 'UNSW: rule lowers volume evasion too (lesson 2)')
check((P(f'{T}_cnn_none_op_thr_asr') - P(f'{T}_rf_none_op_thr_asr')) > (P(f'{T}_cnn_none_op_own_asr') - P(f'{T}_rf_none_op_own_asr'))
      and (P(f'{T}_cnn_none_op_thr_asr') - P(f'{T}_gb_none_op_thr_asr')) > (P(f'{T}_cnn_none_op_own_asr') - P(f'{T}_gb_none_op_own_asr'))
      and (P(f'{T}_mlp_none_op_thr_asr') - P(f'{T}_rf_none_op_thr_asr')) > (P(f'{T}_mlp_none_op_own_asr') - P(f'{T}_rf_none_op_own_asr'))
      and (P(f'{T}_mlp_none_op_thr_asr') - P(f'{T}_gb_none_op_thr_asr')) > (P(f'{T}_mlp_none_op_own_asr') - P(f'{T}_gb_none_op_own_asr')),
      'TON: net-tree gap widens under the threshold rule')
check(all(pv(f'{U}_{m}_prim_p') < 0.05 and F(f'{U}_{m}_prim') < 0 for m in ('cnn_at_threat', 'cnn_trades', 'cnn_clean_loop', 'mlp_at_threat')),
      'UNSW: AT-threat, TRADES, loop, MLP AT-threat significant at matched FPR')
check(pv(f'{U}_cnn_at_threat_prim_vs_cnn_clean_loop_p') < 0.05 and all(pv(f'{U}_cnn_at_threat_prim_vs_{o}_p') > 0.05 for o in ('cnn_at_all', 'cnn_ccat', 'cnn_trades')),
      'UNSW: AT-threat better than loop only')
check(P(f'{U}_cnn_trades_op_matched_e2e_volume') <= P(f'{U}_cnn_at_threat_op_matched_e2e_volume'), 'UNSW abstract range TRADES..AT-threat')
check(all(P(f'{U}_{m}_op_matched_e2e') >= P(f'{U}_cnn_none_op_thr_e2e') - 5 for m in ('cnn_at_all', 'cnn_ccat'))
      and all(P(f'{U}_{m}_op_matched_e2e') < P(f'{U}_cnn_none_op_thr_e2e') - 20 for m in ('cnn_at_threat', 'cnn_trades')),
      'UNSW all actions: AT-all/CCAT lose advantage, AT-threat/TRADES keep it (S11)')
check(int(V[f'{T}_n_primary_sig']) == 5 and int(V[f'{U}_n_primary_sig']) == 8, 'primary family counts as worded (TON 5 incl. TRADES worse)')
check(P(f'{U}_net_fpr1_asr_min') > 90, 'UNSW at 1% FPR every network missed/evaded > 90%')
check(P('bpda_new_max') <= 0.5, 'BPDA action PGD adds at most a fraction of a percent')
check(F(f'{T}_cnn_none_relax_gain') > 5 and abs(F(f'{T}_rf_none_relax_gain')) < 2 and abs(F(f'{U}_rf_none_relax_gain')) < 2
      and all(F(f'{ds}_{m}_relax_gain') <= F('relax_gain_max') + 1e-9 for ds in DSS for m in ('cnn_none', 'mlp_none') if f'{ds}_{m}_relax_gain' in V),
      'relax: networks gain much, RF little')
mx = max((F(f'{ds}_{m}_relax_gain'), m) for ds in DSS for m in BASE if f'{ds}_{m}_relax_gain' in V)
check(mx[1] in ('cnn_none', 'mlp_none'), 'relax: largest gain is a network ("up to ... for the networks")')
check(float(V[f'{T}_cnn_none_relax_width_max']) < 0.005, 'relax: TON realistic width below smallest ball')
if f'{T}_cnn_at_all_e0.1_opd_e2e' in V:
    check(V.get(f'{T}_cnn_at_all_e0.1_n5') == '5', 'sweep: AT-all eps 0.1 on five TON seeds')
else:
    pend('sweep: AT-all eps 0.1 on five TON seeds')
check(P(f'{T}_cnn_clean_loop_strongest_b0.5') < P(f'{T}_cnn_none_strongest_b0.5') / 2 + 1, 'TON: loop about halves ASR at own threshold')
check(P(f'{T}_cnn_at_all_wb_unconstrained_e0.05_r2seeds') < 35 and P(f'{T}_cnn_ccat_wb_unconstrained_e0.05_r2seeds') < 35
      and P(f'{T}_cnn_at_all_wb_bpda_e0.05') > 80 and P(f'{T}_cnn_ccat_wb_bpda_e0.05') > 80, 'TON: eps-ball defences robust only to the weak PGD')
check(V[f'{T}_cnn_corners_e0.05_min'] == '100.0%', 'TON: corners evade every detected flow of every CNN')
check(float(V[f'{T}_cnn_at_all_bpda_f1']) < 0.7 and float(V[f'{T}_cnn_at_all_bpda_fpr']) > 0.2, 'TON: BPDA AT-all collapsed')
check(P(f'{U}_cnn_at_all_bb_corners_e0.05') < P(f'{U}_cnn_at_threat_bb_corners_e0.05') and P(f'{U}_cnn_ccat_bb_corners_e0.05') < P(f'{U}_cnn_at_threat_bb_corners_e0.05')
      and max(P(f'{U}_cnn_at_all_bb_corners_e0.05'), P(f'{U}_cnn_ccat_bb_corners_e0.05')) < min(P(f'{U}_{m}_bb_corners_e0.05') for m in ('cnn_none', 'cnn_clean_loop', 'cnn_trades', 'cnn_at_threat', 'cnn_at_all_bpda')),
      'UNSW: eps-ball defences least evaded under corners')
check(max(P(f'{U}_{m}_s0_op_matched_asr') for m in ('cnn_at_threat_b0.25', 'cnn_at_threat', 'cnn_at_threat_b1.0'))
      < min(P(f'{U}_{m}_s0_op_matched_asr') for m in ('cnn_at_all_e0.02', 'cnn_at_all', 'cnn_at_all_e0.1')), 'UNSW sweep: every AT-threat config below every AT-all config')
check(F(f'{T}_cnn_at_all_d_strongest_b0.5') < -20 and F(f'{T}_cnn_at_all_dloop_strongest_b0.5') > -10, 'TON: AT-all gain mostly the loop')
check(all(pv(f'{T}_{m}_prim_p') < 0.05 and abs(F(f'{T}_{m}_dclean')) < 0.005 and abs(F(f'{T}_{m}_dfpr')) < 0.005 for m in ('rf_augment', 'gb_augment')),
      'TON: tree augmentation significant, no clean or FPR cost')
check(P(f'{T}_gb_augment_op_matched_asr') < P(f'{T}_gb_none_op_own_asr') / 5, 'TON: GBDT augmentation holds at matched FPR')
check(abs(F(f'{T}_gb_augment2_minus_gb_augment_strongest_b0.5')) < 3 and abs(F(f'{T}_rf_augment2_minus_rf_augment_strongest_b0.5')) < 3, 'TON: second round changes little')
for t in ('rf', 'gb'):
    check(P(f'{T}_{t}_augment_strongest_b0.5') < P(f'{T}_{t}_augment_nobytes_strongest_b0.5') < P(f'{T}_{t}_none_strongest_b0.5')
          and P(f'{T}_{t}_augment_nodelay_strongest_b0.5') > P(f'{T}_{t}_augment_strongest_b0.5'), f'TON {t}: held-out actions between aug and none')
check(P(f'{T}_gb_augment_ad_none_b2.0') < P(f'{T}_gb_none_ad_none_b2.0') and P(f'{T}_rf_augment_ad_none_b2.0') < P(f'{T}_rf_none_ad_none_b2.0'), 'TON: aug holds at beta 2 (adaptive)')
check(float(V[f'{T}_gb_augment_f1']) >= max(float(V[f'{T}_{m}_f1']) for m in BASE) - 0.002, 'TON: augmented GBDT as accurate as best')
for t in ('rf', 'gb'):
    check(P(f'{U}_{t}_augment_bb_both_b0.5') < 2 and P(f'{U}_{t}_augment_strongest_volume_b0.5') < 3, f'UNSW {t} aug: candidates almost removed')
    check(P(f'{U}_{t}_augment_bb_ttl_only_all') > 10 and P(f'{U}_{t}_augment_ad_free_b0.5') > 25, f'UNSW {t} aug: unseen TTLs break it')
    check(P(f'{U}_{t}_augment_strongest_b0.5') < P(f'{U}_{t}_none_strongest_b0.5') - 30 and pv(f'{U}_{t}_augment_prim_p') < 0.05
          and abs(F(f'{U}_{t}_augment_dclean')) < 0.005 and abs(F(f'{U}_{t}_augment_dfpr')) < 0.01, f'UNSW {t} aug: well below undefended, significant, no cost')
    check(P(f'{U}_{t}_augment_nodelay_strongest_volume_b0.5') > P(f'{U}_{t}_augment_strongest_volume_b0.5'), f'UNSW {t}: no-delay variant evaded more by volume')
check(P(f'{U}_rf_augment2_strongest_b0.5') < P(f'{U}_rf_augment_strongest_b0.5') and P(f'{U}_gb_augment2_strongest_b0.5') < P(f'{U}_gb_augment_strongest_b0.5'),
      'UNSW: second round lowers both')
for ds in DSS:
    check(float(V[f'{ds}_cnn_ccat_rare_f1']) > float(V[f'{ds}_cnn_at_all_rare_f1'])
          and float(V[f'{ds}_cnn_ccat_rare_f1']) < min(float(V[f'{ds}_cnn_none_rare_f1']), float(V[f'{ds}_cnn_at_threat_rare_f1'])),
          f'{ds}: CCAT rare F1 above AT-all, below undefended and AT-threat')
    check(all(F(f'{ds}_{m}_drare') < 0 and F(f'{ds}_{m}_dfpr') > 0 for m in NETS + ('mlp_at_threat', 'cnn_at_all_bpda')), f'{ds}: every network defence incl. loop lowers rare F1, raises FPR')
check(F(f'{T}_cnn_at_all_bpda_drare') == min(F(f'{T}_{m}_drare') for m in NETS + ('mlp_at_threat', 'cnn_at_all_bpda'))
      and F(f'{T}_cnn_at_all_bpda_dfpr') == max(F(f'{T}_{m}_dfpr') for m in NETS + ('mlp_at_threat', 'cnn_at_all_bpda')), 'TON: extremes are the BPDA AT-all')
check(P(f'{T}_cnn_none_strongest_b0.5') > P(f'{T}_rf_none_strongest_b0.5'), 'abstract: TON CNN vs RF')
# 5.5
print('§5.5 TTL')
check(all(abs(P(f'{U}_{m}_bb_ttl_only_unseen') - P(f'{U}_{m}_bb_ttl_only')) < 5 for m in ('cnn_none', 'mlp_none', 'gb_none'))
      and P(f'{U}_rf_none_bb_ttl_only_unseen') < P(f'{U}_rf_none_bb_ttl_only') - 10, 'UNSW: unseen TTLs about as effective, except RF')
check(all(P(f'{U}_{m}_nottl_strongest_b0.5') < P(f'{U}_{m}_strongest_volume_b0.5_s3') for m in ('cnn_none', 'mlp_none'))
      and P(f'{U}_rf_none_nottl_strongest_b0.5') > P(f'{U}_rf_none_strongest_volume_b0.5_s3') + 5
      and abs(P(f'{U}_gb_none_nottl_strongest_b0.5') - P(f'{U}_gb_none_strongest_volume_b0.5_s3')) < 5, 'UNSW no-TTL: nets harder, RF easier, GBDT about as easy (same seeds)')
s3 = [P(f'{U}_{m}_nottl_strongest_b0.5') for m in ('cnn_none', 'mlp_none', 'rf_none', 'gb_none')]
check(max(s3) - min(s3) < 15, 'UNSW no-TTL: nets-trees gap largely closed (spread < 15 pp)')
check(P(f'{U}_cnn_none_nottl_det') > P(f'{U}_cnn_none_det'), 'UNSW no-TTL CNN detects more')

# final review round 2 (referee major 1): AT-threat with a corner-start inner attack
def _civ(k):
    return [float(x.replace('−', '-').replace(' pp', '')) for x in V[k].strip('[]').split(', ')]
check(V.get('atcs_done_TON_IoT') == 'yes' and V.get('atcs_done_UNSW_NB15') == 'yes', 'corner-start AT-threat: five seeds on both datasets')
lo, hi = _civ('TON_IoT_cnn_at_threat_cs_d_cnn_none_e2e_ci')
check(lo < 0 < hi and 'did not change the network result' in src, 'TON corner-start vs undefended CI spans 0; text says it did not change the result')
lo, hi = _civ('TON_IoT_cnn_at_threat_cs_d_cnn_at_threat_e2e_ci')
check(lo < 0 < hi, 'TON corner-start vs random-start AT-threat CI spans 0')
lo, hi = _civ('UNSW_NB15_cnn_at_threat_cs_d_cnn_none_e2e_volume_ci')
check(hi < 0, 'UNSW corner-start AT-threat below undefended (volume, CI < 0)')
lo, hi = _civ('TON_IoT_cnn_at_threat_d_cnn_none_e2e_ci')
check(lo < -25 and hi > 10 and 'not evidence that the gain is zero' in src and 'no detectable gain on TON-IoT' in src, 'TON AT-threat CI wide; null not worded as evidence of absence')
check('the only defence in our study that did so' not in src, 'lesson 3: no implied network/tree comparison under unequal inner searches')
# referee major 3: destination replies
check(V.get('reply_done') == 'yes', 'reply sensitivity: all 16 model/dataset rows')
check(all(not V[f'{ds}_reply_def_range'].startswith('−') for ds in ('TON_IoT', 'UNSW_NB15')) and 'were evaded more' in src, 'replies raised evasion of every defence (text: evaded more)')
check(V['TON_IoT_reply_undef_range'].startswith('−') and 'direction depends on the model' in src, 'undefended: replies change ASR in both directions')
check('These restrictions are conservative' not in src, 'no "conservative" claim for unmodelled replies')
# referee major 2: attack-aware ROC
check(V.get('roc_done') == 'yes', 'ROC: all 13 models on both datasets')
lo, hi = _civ('TON_IoT_cnn_at_threat_roc_s1_dpauc_ci')
check(hi < 0 and 'did help there under one decision rule but not under another' in src, 'TON AT-threat ROC gain (CI < 0) stated as rule-dependent')
lo, hi = _civ('TON_IoT_cnn_at_threat_roc_s2_dpauc_ci')
check(lo < 0 < hi, 'TON AT-threat gain vanishes under the second rule')
check(_civ('TON_IoT_cnn_trades_roc_s1_dpauc_ci')[0] > 0 and _civ('TON_IoT_cnn_trades_roc_s2_dpauc_ci')[0] > 0, 'TRADES worse under both rules')
lo, hi = _civ('UNSW_NB15_cnn_at_threat_roc_s1_dpauc_ci')
check(lo < 0 < hi and 'helped only at the undefended model' in src, 'UNSW: no significant network gain at FPR <= 5%; text limits the gain to the high-FPR point')

print(f'\n{len(fails)} failure(s), {len(pending)} pending')
for f in fails:
    print('  FAIL', f)
sys.exit(len(fails))
