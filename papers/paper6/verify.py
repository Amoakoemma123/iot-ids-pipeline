"""Check Manuscript_v2.docx (Paper 6, round 2: results_v4): no unfilled placeholders, and every qualitative claim in the text
holds in the result files. Exit code = number of failures. Checks whose results are not yet available are
reported as PENDING (not failures) so that a partial build can be checked."""
import json
import re
import statistics as st
import sys
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / 'experiments' / 'paper6_multimodal' / 'results_v4'
V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
DATA = json.loads((HERE / 'data_stats.json').read_text())
DSS = ('TON_IoT', 'UNSW_NB15', 'N_BaIoT')
CNN = ('s1', 's2', 'early', 'late', 'attn')
fails, pending = [], []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def fv_s(x):
    return float(x.replace('−', '-'))


def pend(msg):
    print('  PEND ' + msg)
    pending.append(msg)


def L(ds, exp, s):
    p = RES / ds / exp / f'seed{s}.json'
    return json.loads(p.read_text()) if p.exists() else None


def done(ds, exp, n=5, pat='seed{}'):
    return all((RES / ds / exp / (pat.format(s) + '.json')).exists() for s in range(n))


RANDOM_DONE = all(done(ds, f'{a}_rand{r}', 1, pat=f'seed{r % 5}') for ds in DSS for a in ('s1', 's2', 'late', 'attn')
                  for r in range(10))
MLP_DONE = all(done(ds, f'm_{a}') for ds in ('TON_IoT', 'UNSW_NB15') for a in CNN)   # MLP only on the two non-saturated datasets
PERM_DONE = all((RES / ds / f'early_perm{r}' / f'seed{r}.json').exists() for ds in ('TON_IoT', 'UNSW_NB15') for r in range(5))
SPLIT_DONE = all((RES / ds / f'late_split{k}' / f'seed{k}.json').exists() for ds in ('TON_IoT', 'UNSW_NB15') for k in range(1, 6))
LAT_DONE = (RES / 'latency.json').exists()
ALL_DONE = RANDOM_DONE and MLP_DONE and PERM_DONE and SPLIT_DONE and LAT_DONE


def runs(ds, exp):
    R = [L(ds, exp, s) for s in range(5)]
    return R if all(R) else None


def m(ds, exp, k='macro_f1'):
    return st.mean(r[k] for r in runs(ds, exp))


def rnd(ds, a):
    R = [L(ds, f'{a}_rand{r}', r % 5) for r in range(10)]
    return R if all(R) else None


text = (HERE / 'Manuscript_filled.md').read_text(encoding='utf-8')
src = (HERE / 'Manuscript.md').read_text(encoding='utf-8')

print('Placeholders, markers and anonymity')
for name in ('Manuscript_v2.docx', 'Manuscript_v2_anon.docx'):
    p = HERE / name
    if p.exists():
        d = Document(p)
        body = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
        if ALL_DONE:
            check('⟦' not in body and '{{' not in body, f'{name}: no unfilled placeholders')
        else:
            pend(f'{name}: {body.count(chr(10214))} unfilled placeholders until all runs are done')
        if 'anon' in name:
            check(not any(n in body for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol')) and not d.core_properties.author,
                  f'{name}: anonymous')
if ALL_DONE:
    check('⟦' not in src, 'Manuscript.md: no ⟦…⟧ to-do markers')
    check('PENDING' not in src, 'Manuscript.md: no PENDING comments')
else:
    import re as _re
    pend('to-do markers remain until all runs are done: ' + ', '.join(_re.findall(r'⟦([^⟧]*)⟧', src)))
for n in range(1, 4):
    check(f'Figure {n}' in text, f'Figure {n} cited in the text')
for n in range(1, 8):
    check(f'Table {n}' in text, f'Table {n} cited in the text')
if LAT_DONE:
    check('Table 8' in text, 'Table 8 cited in the text')
else:
    pend('Table 8 (latency) cited once latency.json exists')

print('§3 data')
check(DATA['TON_IoT']['test']['mitm'] < 200 and DATA['UNSW_NB15']['test']['worms'] < 50, 'MITM and worms are small test classes')
check(max(DATA['TON_IoT']['s1_median']) < 0.01, 'TON stream-1 features concentrated near zero after min–max')

print('§5.1 overview')
check(m('TON_IoT', 's1') > m('TON_IoT', 's2') and m('TON_IoT', 'rf_s1') > m('TON_IoT', 'rf_s2'), 'TON: stream 1 stronger than stream 2 (CNN and RF)')
nbm = {a: m('N_BaIoT', a) for a in CNN + ('rf', 'rf_s1', 'rf_s2', 'gb', 'gb_s1', 'gb_s2')}
check(min(nbm.values()) >= 0.99 and min(nbm, key=nbm.get) == 's2', 'N-BaIoT: every model ≥ 0.99, minimum is stream 2 (quoted range)')
NETS = CNN + ('early_wide', 'early_head', 'late_all')
for ds in ('TON_IoT', 'UNSW_NB15'):
    sds = {a: st.stdev(r['macro_f1'] for r in runs(ds, a)) for a in NETS}
    check(abs(max(sds.values()) - float(V[f'{ds}_sd_max'])) < 6e-4, f'{ds}: quoted largest network SD is the maximum')
check(float(V['TON_IoT_sd_max']) <= 0.002 < float(V['UNSW_NB15_sd_max']), 'SD small on TON, larger on UNSW')
lo = min(m('UNSW_NB15', a, 'benign_fpr') for a in NETS + ('rf', 'gb', 'rf_s1', 'rf_s2', 'gb_s1', 'gb_s2'))
check(lo >= 0.12, f'UNSW: every model misclassifies ≥12% of benign records (min {lo:.3f})')
check(max(m('TON_IoT', 'rf'), m('TON_IoT', 'gb')) >= max(m('TON_IoT', a) for a in NETS), 'TON: tree ensembles on all features score highest')
check(abs(m('TON_IoT', 'early') - min(m('TON_IoT', a) for a in ('early', 'late', 'attn'))) < 1e-12
      and abs(m('TON_IoT', 'late') - max(m('TON_IoT', a) for a in ('early', 'late', 'attn'))) < 1e-12,
      'TON fusion range is early (min) to late (max)')

print('§5.2 fusion vs one network')
for ds in ('TON_IoT', 'UNSW_NB15'):
    d = [L(ds, 'early', s)['macro_f1'] - max(L(ds, 's1', s)['macro_f1'], L(ds, 's2', s)['macro_f1']) for s in range(5)]
    check(all(x > 0 for x in d), f'{ds}: early fusion beats the better stream in every seed')
    check(m(ds, 's1') > m(ds, 's2'), f'{ds}: better single stream is stream 1')
check(m('N_BaIoT', 's1') > m('N_BaIoT', 's2'), 'N_BaIoT: better single stream is stream 1')
for ds in DSS:
    d = [L(ds, 'attn', s)['macro_f1'] - L(ds, 'early', s)['macro_f1'] for s in range(5)]
    ci = V[f'{ds}_d_attn_early_ci'].replace('−', '-').strip('[]').split(',')
    check(float(ci[0]) <= 0 <= float(ci[1]), f'{ds}: attention − early CI includes zero')
    check(m(ds, 'late') >= m(ds, 'attn'), f'{ds}: attention fusion below late fusion')
    check(all(m(ds, a) <= m(ds, 'rf') + 1e-4 for a in CNN), f'{ds}: no CNN beats the RF on all features (within 1e-4)')
check(all(L('TON_IoT', 'rf', s)['macro_f1'] > L('TON_IoT', 'late', s)['macro_f1'] for s in range(5)), 'TON: RF > late fusion in all five seeds')
check(m('TON_IoT', 'rf', 'benign_fpr') < m('TON_IoT', 'late', 'benign_fpr'), 'TON: RF lower benign FPR than late fusion')
pc = {c: st.mean(r['per_class_f1'][c] for r in runs('UNSW_NB15', 'rf')) - st.mean(r['per_class_f1'][c] for r in runs('UNSW_NB15', 'late'))
      for c in DATA['UNSW_NB15']['classes']}
top3 = sorted(pc, key=pc.get, reverse=True)[:3]
check(set(top3) == {'worms', 'shellcode', 'exploits'}, f'UNSW: largest RF − late per-class margins = worms, shellcode, exploits (got {top3})')
for c in ('mitm',):
    vals = [st.mean(r['per_class_f1'][c] for r in runs('TON_IoT', a)) for a in ('early', 'late', 'attn')]
    others = [max(st.mean(r['per_class_f1'][k] for r in runs('TON_IoT', a)) for a in ('early', 'late', 'attn')) -
              min(st.mean(r['per_class_f1'][k] for r in runs('TON_IoT', a)) for a in ('early', 'late', 'attn'))
              for k in DATA['TON_IoT']['classes'] if k != c]
    check(max(vals) - min(vals) > max(others), 'TON: fusion models differ most on MITM')
if 'TON_IoT_d_late_early_pholm' in V:
    check(float(V['TON_IoT_d_late_early_pholm']) < 0.05, 'TON late − early significant after Holm')
    check(float(V['UNSW_NB15_d_late_early_pholm']) >= 0.05, 'UNSW late − early not significant after Holm')

print('§5.2 capacity control')
ph = lambda k: float(V[k + '_pholm'])
fv = lambda k: float(V[k].replace('−', '-'))
check(ph('TON_IoT_d_late_early_wide') < 0.05 and fv('TON_IoT_d_late_early_wide') > 0, 'TON: late fusion above wide early fusion (Holm)')
check(ph('TON_IoT_d_late_late_all') >= 0.05 and abs(fv('TON_IoT_d_late_late_all')) <= 0.002, 'TON: late on all features within ~0.001 of late (n.s.)')
check(fv('TON_IoT_d_early_head_early') >= 0.6 * fv('TON_IoT_d_late_early'), 'TON: early fusion with head recovers most of the late advantage')
check(fv('TON_IoT_d_early_wide_early') < fv('TON_IoT_d_late_early'), 'TON: widening adds less than late fusion')
check(all(ph(f'UNSW_NB15_d_{k}') >= 0.05 for k in ('late_early', 'attn_early', 'late_early_wide', 'late_late_all')), 'UNSW: no family test significant')
check(V['UNSW_NB15_d_late_late_all_pos'] == '5', 'UNSW: late above late-all in 5/5 seeds')
check(V['holm_sig_list'] == 'TON_IoT_d_late_early, TON_IoT_d_late_early_wide', 'Holm-significant set as described')
for k in ('late_early', 'late_early_wide', 'late_late_all', 'attn_early'):
    check(abs(fv(f'TON_IoT_d_{k}_clean') - fv(f'TON_IoT_d_{k}')) <= 0.003, f'TON clean-subset {k} within a few thousandths')

print('§5.3 random splits')
if all(rnd(ds, a) for ds in DSS for a in ('s1', 's2', 'late', 'attn', 'rf_s1', 'rf_s2')):
    rf1 = lambda ds, a: [r['macro_f1'] for r in rnd(ds, a)]
    check(m('TON_IoT', 's2') < min(rf1('TON_IoT', 's2')) and m('TON_IoT', 'rf_s2') < min(rf1('TON_IoT', 'rf_s2')),
          'TON: semantic stream 2 below every random stream of its size (CNN and RF)')
    for a in ('s1', 's2', 'rf_s1', 'rf_s2'):
        check(sum(v > m('UNSW_NB15', a) for v in rf1('UNSW_NB15', a)) >= 9, f'UNSW: semantic {a} below nearly all (≥9/10) random streams')
    for ds in ('TON_IoT', 'UNSW_NB15'):
        check(sum(m(ds, 'late') > v for v in rf1(ds, 'late')) == int(V[f'{ds}_rand_late_below'])
              and '{{%s_rand_late_below}} of the ten' % ds in src, f'{ds}: count of random splits below semantic late fusion stated from data')
    check(fv('UNSW_NB15_semrand_late_p') < 0.05 and 'a small positive effect whose interval reaches just past the margin' in src, 'UNSW: within-seed semantic − random late small positive, p < 0.05 (as stated, descriptive)')
    for ds in ('TON_IoT', 'UNSW_NB15'):
        check(abs(fv(f'{ds}_semrand_late')) < 0.01 and abs(fv(f'{ds}_semrand_attn')) < 0.01, f'{ds}: within-seed semantic − random below 0.01')
    for a in ('late', 'attn'):
        check(abs(m('N_BaIoT', a) - st.mean(rf1('N_BaIoT', a))) < 0.0015 and abs(fv(f'N_BaIoT_semrand_{a}')) < 0.0015, f'N-BaIoT {a}: semantic vs random about 0.001')
    for ds in ('TON_IoT', 'UNSW_NB15'):
        d = m(ds, 'late') - st.mean(rf1(ds, 'late'))
        check(0 <= d < 0.01, f'{ds}: semantic late advantage is at most about 0.01 ({d:+.4f})')
        check(float(V[f'{ds}_gain_sem'].replace('−', '-')) > float(V[f'{ds}_gain_rand'].replace('−', '-')), f'{ds}: larger fusion gain on semantic split')
        check(m(ds, 's1') < st.mean(rf1(ds, 's1')) or m(ds, 's2') < st.mean(rf1(ds, 's2')),
              f'{ds}: a semantic stream is weaker than its random counterparts')
    check(int(V['TON_IoT_rand_late_above_early']) == 10, 'TON: late fusion on every random split above early fusion')
else:
    pend('random-split runs not complete')

print('§5.4 attention')
check(float(V['TON_IoT_attn_rho_adv']) < 0.3, 'TON: attention–advantage correlation weak (< 0.3)')
check(0.4 <= float(V['UNSW_NB15_attn_rho_adv']) < 0.7, 'UNSW: attention–advantage correlation moderate')
check(V['TON_IoT_attn_class_min'] == 'dos', 'TON: DoS has the lowest stream-1 weight')
check(m('TON_IoT', 's1') and st.mean(r['per_class_f1']['dos'] for r in runs('TON_IoT', 's1')) >
      st.mean(r['per_class_f1']['dos'] for r in runs('TON_IoT', 's2')), 'TON: stream 1 classifies DoS better than stream 2')
nb = [st.mean(r['per_class_f1'][c] for r in runs('N_BaIoT', a)) for a in ('s1', 's2') for c in DATA['N_BaIoT']['classes']]
check(min(nb) >= 0.97, f'N-BaIoT: both streams classify every class almost perfectly (min per-class F1 {min(nb):.3f})')
check(float(V['UNSW_NB15_attn_rho_seeds']) > max(float(V['TON_IoT_attn_rho_seeds']), float(V['N_BaIoT_attn_rho_seeds'])),
      'cross-seed stability highest on UNSW (the exception in the text)')

for ds in DSS:
    check(float(V[f'{ds}_attn_rho_rel']) >= 0.8 and float(V[f'{ds}_attn_rho_rel_p']) < 0.01, f'{ds}: weights track in-model reliance (ρ ≥ 0.8, p < 0.01)')
for ds in ('TON_IoT', 'UNSW_NB15'):
    check(float(V[f'{ds}_rel_rho_adv_p']) >= 0.05, f'{ds}: reliance not significantly tied to stream advantage')
    check(float(V[f'{ds}_abl_drop_s1']) > float(V[f'{ds}_abl_drop_s2']), f'{ds}: removing stream 1 costs more')

print('§4.1 budget')
check(int(V['n_capped']) <= 2, f"almost no network run used the full budget ({V['n_capped']}/{V['n_cnn_runs_main']})")

print('pending experiments')
for flag, msg in ((MLP_DONE, 'MLP encoder runs (§5.3 MLP sentence)'), (PERM_DONE, 'column-order runs (§4.2 control; text to add)'),
                  (SPLIT_DONE, 'data re-splits (text to add)'), (RANDOM_DONE, 'random splits (§5.3)')):
    if not flag:
        pend(msg + ' not complete')

print('§5.2 data splits, §5.3 column order + MLP, §4.2 reuse')
if SPLIT_DONE:
    check(all(V[f'{ds}_splitd_rf_late_pos'] == '6' for ds in ('TON_IoT', 'UNSW_NB15')), 'RF above late fusion on all six splits (both)')
    check(V['TON_IoT_splitd_late_early_pos'] == '6', 'TON: late above early on all six splits')
    check(all(abs(fv(f'{ds}_splitd_attn_early')) < 0.01 for ds in ('TON_IoT', 'UNSW_NB15')), 'attention within 0.01 of early over splits')
if PERM_DONE:
    lo, hi = (float(x) for x in V['TON_IoT_perm_range'].split('–'))
    check(lo <= fv('TON_IoT_early_f1') + 0.002 and hi >= fv('TON_IoT_early_f1') - 0.002, 'TON: original order inside the column-order range')
if MLP_DONE:
    check(all(abs(fv(f'{ds}_m_late_f1') - fv(f'{ds}_late_f1')) < 0.006 for ds in ('TON_IoT', 'UNSW_NB15')), 'MLP late within a few thousandths of CNN late')
    check(V['TON_IoT_d_m_late_m_early_pos'] == '5' and fv('UNSW_NB15_d_m_late_m_early_p') >= 0.05, 'MLP: late > early on TON 5/5, n.s. on UNSW')
    lo, hi = (fv_s(x) for x in V['TON_IoT_semrand_m_late_ci'].strip('[]').split(', '))
    lo2, hi2 = (fv_s(x) for x in V['UNSW_NB15_semrand_m_late_ci'].strip('[]').split(', '))
    check(lo < 0 < hi and lo2 < 0 < hi2, 'MLP semantic − random late: both CIs span zero')
check(V['n_reused_main'] == '15' and all(not (RES / 'N_BaIoT' / a / 'seed0.json').exists() or
      ('reused_from' in json.loads((RES / 'N_BaIoT' / a / 'seed0.json').read_text())) == (a in ('s1', 'early', 'late'))
      for a in CNN), 'reuse disclosure: 15 N-BaIoT runs = stream 1, early, late')

print('§5.5 scaling')
check(float(V['TON_IoT_s1_mm']) < 0.1 and float(V['TON_IoT_s1_q']) > 0.85, 'TON stream-1 CNN: failure with min–max, works with quantile')
check(float(V['TON_IoT_s1_mm_fpr']) > 0.5, 'TON stream-1 min–max CNN flags most benign records')
check(float(V['TON_IoT_s2_mm']) > float(V['TON_IoT_s1_mm']), 'min–max run would rank stream 2 above stream 1 (reversal)')
check(float(V['TON_IoT_s1_q']) > float(V['TON_IoT_s2_q']), 'quantile run ranks stream 1 above stream 2')
check(int(V['TON_IoT_s1_mm_best_ep_max']) <= 5, 'min–max stream-1 CNN best epoch ≤ 5')

print('§5.6 latency')
if (RES / 'latency.json').exists():
    Lj = json.loads((RES / 'latency.json').read_text())
    T_ = Lj['TON_IoT']
    check(max(T_[a]['batch1000_ms'] for a in ('early', 'late', 'attn')) / min(T_[a]['batch1000_ms'] for a in ('early', 'late', 'attn')) < 1.1,
          'TON: early/late/attention equally fast for 1,000 records (within 10%)')
    check(all(1.7 <= T_[a]['batch1000_ms'] / T_['early']['batch1000_ms'] <= 2.3 for a in ('early_wide', 'late_all')), 'TON: wide early and late-all about twice as long')
    check(T_['late']['single_ms'] > T_['early']['single_ms'] and T_['attn']['single_ms'] > T_['early']['single_ms'], 'TON: late/attn slower than early per record')
    check(all(Lj[d]['rf']['single_ms'] > 10 * Lj[d]['late']['single_ms'] for d in DSS), 'RF per record > 10x late fusion on every dataset')
    check(all(Lj[d]['gb']['size_mb'] < Lj[d]['rf']['size_mb'] for d in ('TON_IoT', 'UNSW_NB15')), 'GBDT smaller than RF (TON, UNSW)')
    check(Lj['UNSW_NB15']['gb']['single_ms'] > Lj['UNSW_NB15']['rf']['single_ms'], 'GBDT slower per record than RF on UNSW')
    check('Table 8' in src, 'Table 8 cited')
else:
    pend('latency.json not available')

print('round-2 review fixes')
# TON within-seed semantic - random late fusion: interval excludes zero at full precision (text says so)
d = []
for s_ in range(5):
    sem = L('TON_IoT', 'late', s_)['macro_f1']
    rr = [L('TON_IoT', f'late_rand{r}', r % 5)['macro_f1'] for r in range(10) if r % 5 == s_]
    d.append(sem - st.mean(rr))
import numpy as _np
from scipy import stats as _stats
h = _stats.t.ppf(0.975, 4) * _np.std(d, ddof=1) / _np.sqrt(5)
check(_np.mean(d) - h <= 0 and fv('TON_IoT_semrand_late_p') >= 0.05 and 'negligible by our margin, although the semantic mean lies above all ten' in src
      and V['TON_IoT_rand_late_below'] == '10', 'TON semantic − random late: CI includes zero, negligible, semantic above all ten')
check(abs(_np.mean(d)) + h < 0.01, 'TON semantic − random late: interval inside the ±0.01 margin ("negligible")')
check(abs(_np.mean(d)) + h <= 0.0025, 'TON semantic − random late: "within 0.002" (upper CI rounds to 0.002)')
lo, hi = (fv_s(x) for x in V['UNSW_NB15_semrand_late_ci'].strip('[]').split(', '))
check(0 < lo and 0.01 < hi <= 0.013, 'UNSW semantic − random late: CI above zero, reaching just past the margin')
check(hi <= 0.015, 'UNSW: grouping effect "by about 0.01 at most" (upper CI <= 0.015)')
lo, hi = (fv_s(x) for x in V['UNSW_NB15_d_late_late_all_ci'].strip('[]').split(', '))
check(lo > 0 and fv('UNSW_NB15_d_late_late_all_pholm') >= 0.05, 'UNSW late − late_all: positive CI, not significant after Holm')
check(fv('UNSW_NB15_d_rf_late_clean') > 0 and fv('TON_IoT_d_rf_late_clean') > 0, 'RF above late fusion on duplicate-free test records')
check(fv('UNSW_NB15_ci_halfwidth') >= 5 * fv('TON_IoT_ci_halfwidth'), 'UNSW intervals much wider than TON')
check(int(V['rerun_same_epoch']) < int(V['rerun_n']) / 2 and fv('rerun_dF_max') > 0.01, 'reruns: most stop at another epoch, max |dF1| > 0.01')
check(V['UNSW_NB15_late_fp_fuzzers'] == V['UNSW_NB15_rf_fp_fuzzers'] and int(V['UNSW_NB15_late_fp_fuzzers'][:-1]) > 50,
      'UNSW benign false positives mostly fuzzers for late fusion and RF')
check(fv('TON_IoT_d_early_head_early') >= 0.5 * fv('TON_IoT_d_late_early'), 'TON: early + head recovers most of late − early ("mostly the head")')
for ds in ('TON_IoT', 'UNSW_NB15'):
    for a in ('s1', 's2', 'early', 'late', 'attn'):
        check(V[f'{ds}_{a}_fin'] == V[f'{ds}_{a}_f1'], f'{ds} {a}: Table 7 final row = Table 2')
check(abs(fv('TON_IoT_rf_t25_d12_f1') - fv('TON_IoT_late_f1')) < 0.01 and abs(fv('UNSW_NB15_rf_t25_d12_f1') - fv('UNSW_NB15_late_f1')) < 0.01,
      'compact forest about at late-fusion level')
check(int(V['TON_IoT_attn_class_min_seeds']) >= 3, 'DoS lowest weight in most seeds')
check(fv('UNSW_NB15_semrand_late') > 0 and fv('TON_IoT_semrand_late') > 0, 'abstract: random streams LOWERED late fusion')
check('p = 0.000' not in text and 'p = < ' not in text, 'no "p = 0.000" / "p = <" in the filled text')

print('round-3 review fixes')
ORDER_ALL = NETS + tuple('m_' + a for a in CNN) + ('rf', 'rf_s1', 'rf_s2', 'gb', 'gb_s1', 'gb_s2')
for ds in ('TON_IoT', 'UNSW_NB15'):
    dmin = min(m(ds, a, 'attack_detection') for a in ORDER_ALL if runs(ds, a))
    check(float(V[f'{ds}_det_min']) <= dmin < float(V[f'{ds}_det_min']) + 0.001, f'{ds}: "at least {V[ds + "_det_min"]}" detection rate holds literally ({dmin:.5f})')
    check(float(V[f'{ds}_det_min_cnn']) <= min(m(ds, a, 'attack_detection') for a in CNN), f'{ds}: main-CNN detection floor holds')
capped = [(ds, a, s) for ds in DSS for a in NETS for s in range(5)
          if L(ds, a, s) and 'reused_from' not in L(ds, a, s) and L(ds, a, s)['epochs'] >= 120]
check(capped == [('N_BaIoT', 's2', 4)] and 'N-BaIoT stream 2, seed 4' in src, f'the one capped run is N-BaIoT stream 2 seed 4 ({capped})')
check(m('UNSW_NB15', 'gb', 'benign_fpr') < m('UNSW_NB15', 'rf', 'benign_fpr') and m('UNSW_NB15', 'gb') < m('UNSW_NB15', 'rf'), 'UNSW: GBDT lower FPR but lower F1 than RF')
check(m('UNSW_NB15', 'attn', 'benign_fpr') > m('UNSW_NB15', 'early', 'benign_fpr') and m('UNSW_NB15', 'attn') > m('UNSW_NB15', 'early'), 'UNSW: attention higher FPR but higher F1 than early')
pcf = lambda ds, a, c: st.mean(r['per_class_f1'][c] for r in runs(ds, a))
check(all(pcf('UNSW_NB15', a, c) > pcf('UNSW_NB15', 'rf', c) for a in ('early', 'late', 'attn') for c in ('analysis', 'backdoor')),
      'UNSW: fusion CNNs ahead of the RF on analysis and backdoor')
g = m('TON_IoT', 'late') - m('TON_IoT', 'early')
check(0.6 <= (m('TON_IoT', 'late_all') - m('TON_IoT', 'early')) / g < 0.9 and 'largely reproduced' in src, 'TON: late-on-all reproduces most (not all) of the gain ("largely")')
check((m('TON_IoT', 'early_wide') - m('TON_IoT', 'early')) / g > 0.25 and 'not from the number of parameters' not in src, 'TON: parameters account for a measurable share; "not from parameters" removed')
lo, hi = (fv_s(x) for x in V['TON_IoT_d_late_early_head_ci'].strip('[]').split(', '))
check(lo < 0 < hi, 'TON: late − early with head interval spans zero')
lo, hi = (fv_s(x) for x in V['UNSW_NB15_d_late_early_head_ci'].strip('[]').split(', '))
check(lo > 0 and V['UNSW_NB15_d_late_early_head_pos'] == '5', 'UNSW: late above early with head (CI > 0, 5/5)')
check('wider head' not in src and 'extra 128-unit dense layer' in src, '"wider head" replaced by the extra dense layer')
lo, hi = (fv_s(x) for x in V['TON_IoT_d_m_attn_m_early_ci'].strip('[]').split(', '))
check(lo > 0 and 'gained nothing' not in src, 'MLP attention − early on TON excludes zero; no "gained nothing" claim')
lo, hi = (fv_s(x) for x in V['UNSW_NB15_d_late_early_ci'].strip('[]').split(', '))
check(lo > 0 and fv('UNSW_NB15_d_late_early_pholm') >= 0.05 and 'no difference between fusion architectures was significant after Holm correction' in src.split('**Keywords')[0],
      'UNSW late − early: unadjusted CI > 0, n.s. only after Holm (abstract says so)')
check('at most about 0.01 macro-F1' not in src and '0.02' not in src.split('**Keywords')[0] and fv_s(V['UNSW_NB15_semrand_late_ci'].strip('[]').split(', ')[1]) < 0.015, 'abstract: no 0.02 claim; UNSW semantic−random CI upper < 0.015')
check(all(V[f'n_reused_rand_{a}_conf'] == '0' for a in ('s1', 's2', 'late', 'attn')) and V['n_reused_rand_late_all'] == '10' and V['late_main_new_conf'] == 'True',
      'no reused random-split run on TON/UNSW (only the 10 N-BaIoT late runs); semantic late runs new')
# final review round 2: bootstrap, per-class concentration, timed GBDT = scored GBDT, label conflicts
check(fv_s(V['TON_IoT_d_late_early_boot_lo']) > 0 and V['TON_IoT_d_late_early_topcls'] == 'mitm' and fv_s(V['TON_IoT_d_late_early_topcls_share'].rstrip('%')) >= 50,
      'TON late − early: bootstrap CI > 0; mostly MITM (as stated)')
check(V['UNSW_NB15_d_late_early_topcls'] == 'worms', 'UNSW late − early: largest contribution from worms')
check(V.get('gb_lat_source') == 'rolled back', 'Table 8 GBDT = the rolled-back GBDT that is scored (compute_values asserts it reproduces seed 0)')
check(fv_s(V['UNSW_NB15_dup_majority_other_label_pct'].rstrip('%')) > 20 and V['TON_IoT_dup_any_other_label_pct'] == '0%', 'label conflicts: UNSW majority-other share large, none on TON')
check(fv('TON_IoT_rerun_dF_max') >= fv('TON_IoT_semrand_late_4') and fv('UNSW_NB15_rerun_dF_mean') > fv('TON_IoT_rerun_dF_mean'),
      'per-dataset reruns: TON max ≥ TON semantic − random; UNSW drives the pooled variation')
for ds in ('TON_IoT', 'UNSW_NB15'):
    a_ = [float(x) for x in V[f'{ds}_splitd_attn_early_minmax'].replace('−', '-').split(' to ')]
    check(a_[0] <= fv(f'{ds}_splitd_attn_early_4') <= a_[1], f'{ds}: per-split attention − early range brackets the mean')
check(fv('TON_IoT_rf_t25_d12_f1') < fv('TON_IoT_late_f1') and fv('TON_IoT_late_f1_4') - fv('TON_IoT_rf_t25_d12_f1') > fv('TON_IoT_d_late_early_4'),
      'TON: 4.4 MB forest below late fusion by more than late − early')
check(fv('UNSW_NB15_rf_t25_d12_f1') <= fv('UNSW_NB15_late_f1'), 'UNSW: compact forest at or below late fusion')
check(fv('TON_IoT_rf_t100_d16_mb') < 100, 'a ~41 MB forest exists, so the text must not say forests need hundreds of MB')
check('needed hundreds of megabytes' not in src, 'no "needed hundreds of megabytes" claim')
du = V['UNSW_NB15_dup_acc_range'].split('–'); cl = V['UNSW_NB15_clean_acc_range'].split('–')
check(float(du[1]) < float(cl[0]), 'UNSW: every model less accurate on matched than on unmatched records')
check(fv('UNSW_NB15_d_rf_late_dup_acc') <= 0.002 and fv('UNSW_NB15_d_rf_late_clean_acc') > 0.02, 'UNSW: RF no better than late fusion on matched records, ahead on the rest')
for c in ('backdoor', 'generic', 'analysis'):
    check(int(V[f'UNSW_NB15_dup_rate_{c}'][:-1]) > 3 * int(V['UNSW_NB15_dup_rate_normal'][:-1]), f'UNSW: {c} matched far more often than benign')
check(V['tables']['holm'].count('\n') == 9, 'Table B.1 has all eight family tests')
check(' 1 ties' not in text, 'no "1 ties"')
check('Params (TON-IoT)' in V['tables']['main'], 'Table 2 Params column labelled TON-IoT')
check('MQTT' not in src and 'bi-flow' not in src, 'Attota described only as far as its abstract supports')
check('flow-versus-behaviour split is used' not in src, 'R1: no claim that Ayantayo/Dinh use this split')
check('2403.15511v1' in src, 'Dinh 2024b cites the 2024 version (v1)')
abstract = text.split('## Abstract')[1].split('**Keywords')[0]
check(len(abstract.split()) <= 250, f'abstract ≤ 250 words ({len(abstract.split())})')
check('only non-saturated IoT benchmark' in abstract, 'abstract states TON-IoT is the only non-saturated IoT benchmark')
check('beat the tuned networks' in abstract and 'untuned networks' not in src, 'tree claim made against the tuned networks; no leftover "untuned networks"')
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('build', HERE / 'build.py'); _b = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_b)
_hl = [_b.fill(h, {**V, **_b.EXTRA})[0] for h in _b.HIGHLIGHTS]
check(all(len(h) <= 85 for h in _hl), f'highlights ≤ 85 characters (longest {max(map(len, _hl))})')
check(any('tuned fusion networks' in h for h in _hl) and not any('gained nothing' in h or 'untuned' in h for h in _hl), 'highlights: trees vs tuned networks; no "untuned"')
# final review round 2 (M3): network grid
check(V.get('nettune_done') == 'yes', 'network grid complete (tune_net.json + all *_tuned runs)')
_ci = lambda k: tuple(fv_s(x) for x in V[k].strip('[]').split(', '))
check(_ci('TON_IoT_d_late_tuned_early_tuned_ci')[0] < 0 < _ci('TON_IoT_d_late_tuned_early_tuned_ci')[1] and 'tuned early fusion' in src or 'tuning early fusion' in src, 'TON: tuned late − tuned early CI spans 0; text says tuning early fusion matched it')
check(_ci('TON_IoT_d_early_tuned_early_ci')[0] > 0 and V['TON_IoT_d_early_tuned_early_pos'] == '5', 'TON: tuning raised early fusion (CI > 0, 5/5)')
check(all(fv(f'{ds}_d_{t}_bestnet') > 0 and V[f'{ds}_{t}_bestnet_welch_p'] == '< 0.001' for ds in ('TON_IoT', 'UNSW_NB15') for t in ('rf', 'gb')), 'RF and GBDT above the best (tuned) network on TON and UNSW, Welch p < 0.001')
check(V['UNSW_NB15_late_tuned_same'] == 'True' and _ci('UNSW_NB15_d_late_tuned_late_ci')[1] < 0 and 'run-to-run variation' in src, 'UNSW late: grid kept fixed config, rerun lower (CI < 0) and text calls it run-to-run variation')
check(V['TON_IoT_early_tuned_cfg'].endswith('width ×2') and V['TON_IoT_late_tuned_cfg'].endswith('width ×2') and 'twice the width for both fusion models on TON-IoT' in src, 'TON: both chosen configurations are width ×2 as stated')
check('n_reused_rand}} of the' not in src, '6.3 no longer claims reused late-fusion random-split runs')
check('{{table:nettune}}' in src and 'Table B.6' in src, 'network grid table in Appendix B')

print(f'\n{len(fails)} failure(s), {len(pending)} pending')
for f in fails:
    print('  FAIL', f)
sys.exit(len(fails))
