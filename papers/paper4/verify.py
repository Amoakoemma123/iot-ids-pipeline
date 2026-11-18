"""Check the Paper 4 manuscript: no unfilled placeholders, figures cited, and every qualitative claim
in the text holds in the result files. Exit code = number of failures."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from docx import Document
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / 'Papers pipeline' / 'experiments' / 'paper4_federated' / 'results_v2'
V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
text = (HERE / 'Manuscript_filled.md').read_text(encoding='utf-8')
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def runs(ds, e):
    return {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted((RES / ds / e).glob('seed*.json'))}


def sel(ds, e):
    return {k: r.get('test_f1_selected', r.get('test_f1_selected_mean')) for k, r in runs(ds, e).items()}


def fin(ds, e):
    return {k: r.get('test_f1_final', r.get('test_f1_final_mean')) for k, r in runs(ds, e).items()}


def mean(d):
    return float(np.mean(list(d.values())))


def paired(a, b):
    k = sorted(set(a) & set(b)); x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k])
    return float((x - y).mean()), float(stats.ttest_rel(x, y).pvalue)


def sig_pos(a, b):
    d, p = paired(a, b); return d > 0 and p < 0.05


def sig_neg(a, b):
    d, p = paired(a, b); return d < 0 and p < 0.05


def pc(ds, e, c, full='test_selected_full'):
    return float(np.mean([r[full]['per_class_f1'][c] for r in runs(ds, e).values()]))


PENDING = set()


def no_unfilled(body):
    """final review round: P4F_* placeholders wait for the diagnostic runs (run_p4final.py); reported, not failed."""
    keys = re.findall(r'⟦([^⟧]+)⟧', body)
    PENDING.update(k for k in keys if k.startswith('P4F_'))
    return '{{' not in body and all(k.startswith('P4F_') for k in keys)


print('Placeholders, figures, anonymity')
for name in ('Supplement_v2.docx', 'Supplement_v2_anon.docx'):
    if (HERE / name).exists():
        d = Document(HERE / name)
        body = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
        check(no_unfilled(body), f'{name}: no unfilled placeholders (other than pending P4F_*)')
        if 'anon' in name:
            check(not any(n in body for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol')) and not d.core_properties.author, f'{name}: anonymous')
sup_order = re.findall(r'\{\{table:(\w+)\}\}', (HERE / 'Supplement.md').read_text(encoding='utf-8'))
check(sup_order.index('paired') == 0, f'supplement table number quoted in the Fig. 1 caption (S1 = paired): {sup_order}')
for name in ('Manuscript_v2.docx', 'Manuscript_v2_anon.docx', 'Manuscript_v2_ieee.docx', 'Manuscript_IEEE_template.docx'):
    p = HERE / name
    if p.exists():
        d = Document(p)
        body = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
        check(no_unfilled(body), f'{name}: no unfilled placeholders (other than pending P4F_*)')
        check(len(d.inline_shapes) == 3, f'{name}: three figures embedded')
        if 'anon' in name:
            check(not any(n in body for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol')) and not d.core_properties.author, f'{name}: anonymous')
check(no_unfilled(text) and '<!--' not in (HERE / 'Manuscript.md').read_text(encoding='utf-8').split('## References')[0].replace('<!--TABLE', ''),
      'no unfilled placeholders or to-do comments')
for n in (1, 2, 3):
    check(f'Fig. {n}' in text, f'Fig. {n} cited')

fl = lambda k: float(V[k].split()[0])
holm = lambda k: V[k + '_sig_holm'] == 'True'

for ds, tag in (('TON_IoT', 'TON'), ('UNSW_NB15', 'UNSW')):
    print(f'{ds}: federation vs references (label skew)')
    C, L5, L1 = sel(ds, 'central'), sel(ds, 'local_a0.5'), sel(ds, 'local_a0.1')
    F = {a: sel(ds, f'fedavg_{a}') for a in ('iid', 'a1', 'a0.5', 'a0.1')}
    check(holm(f'{tag}_d_fedavg_iid_vs_central') and paired(F['iid'], C)[0] < 0, f'{tag}: IID gap to centralised significant after Holm')
    check(float(V[f'{tag}_trend_slope']) < 0 and V[f'{tag}_trend_p'] in ('p < 0.001',) or float(V[f'{tag}_trend_p'].split('= ')[-1].replace('p < ', '')) < 0.05,
          f'{tag}: gap grows with skew (trend negative, p < 0.05)')
    check(paired(F['a0.1'], L1)[0] > paired(F['a0.5'], L5)[0] > 0, f'{tag}: gain over local larger at alpha 0.1 than 0.5')
    check(paired(F['a0.1'], C)[0] < paired(F['a0.5'], C)[0], f'{tag}: cost larger at alpha 0.1 than 0.5')
    if tag == 'TON':
        check(holm('TON_d_fedavg_a01_vs_local_a01') and not holm('TON_d_fedavg_a05_vs_central') and not holm('TON_d_fedavg_a01_vs_central'),
              'TON: only IID gap and alpha-0.1 local gain significant after Holm (as stated)')
        check(mean(L5) < mean(F['a0.5']) - 0.05, 'TON: local-only much worse at alpha 0.5')
    else:
        check(not sig_pos(F['a0.5'], L5) and abs(paired(F['a0.5'], L5)[0]) < 0.05, 'UNSW: gain over local-only small and not significant at alpha 0.5')
        check(float(V['UNSW_local_a01_lo']) < 0.1, 'UNSW: weakest alpha-0.1 client near useless')
    print(f'{ds}: algorithms')
    for a in ('a0.5', 'a0.1'):
        ak = a.replace('.', '')
        for alg in ('fedprox', 'fedadam', 'fedlc'):
            d = paired(sel(ds, f'{alg}_{a}'), F[a])[0]
            check(not (d > 0 and holm(f'{tag}_d_{alg}_{ak}_vs_fedavg_{ak}')), f'{tag}: {alg} not significantly better than FedAvg after Holm ({a})')
        check(not holm(f'{tag}_d_scaffold_{ak}_vs_fedavgsgd_{ak}'), f'{tag}: SCAFFOLD vs FedAvg-SGD not significant after Holm ({a})')
        check(abs(paired(sel(ds, f'fedprox_{a}'), F[a])[0]) < 0.05 and not holm(f'{tag}_d_fedprox_{ak}_vs_fedavg_{ak}'), f'{tag}: FedProx differs little from FedAvg ({a})')
    gap = [s - f for s, f in zip(sel(ds, 'fedavg_a0.1').values(), fin(ds, 'fedavg_a0.1').values())]
    check(np.mean(gap) < 0.05, f'{tag}: selected and final round close for FedAvg alpha 0.1 ({np.mean(gap):.3f})')
    print(f'{ds}: scaling and communication')
    S = [mean(sel(ds, e)) for e in ('scale_N2', 'fedavg_a0.5', 'scale_N8', 'scale_N16')]
    check(S[0] > S[3] and (tag == 'UNSW' or all(S[i] > S[i + 1] for i in range(3))), f'{tag}: macro-F1 falls with N ({[round(x, 3) for x in S]})')
    mb = [runs(ds, e)[0]['bytes_per_round'] for e in ('scale_N2', 'fedavg_a0.5', 'scale_N8', 'scale_N16')]
    check(all(abs(mb[i] / mb[0] - [1, 2, 4, 8][i]) < 1e-6 for i in range(4)), f'{tag}: traffic per round linear in N')
    r0 = runs(ds, 'fedavg_a0.5')[0]
    check(r0['total_mb'] > r0['raw_train_mb'], f'{tag}: 50-round FL traffic exceeds the raw training matrix')
    check(V[f'{tag}_tunec_best'] == '0.003', f'{tag}: centralised lr check selects 0.003')

print('TON-specific claims')
F5 = sel('TON_IoT', 'fedavg_a0.5')
check(holm('TON_d_fedadam_a05_vs_fedavg_a05') and paired(sel('TON_IoT', 'fedadam_a0.5'), F5)[0] < 0, 'TON: FedAdam deficit at alpha 0.5 significant after Holm')
others = [k for k in V if k.endswith('_sig_holm') and V[k] == 'True' and any(f'_d_{a}_' in k for a in ('fedprox', 'fedadam', 'fedlc', 'scaffold'))]
check(others == ['TON_d_fedadam_a05_vs_fedavg_a05_sig_holm'], f'only algorithm difference significant after Holm is FedAdam TON alpha 0.5 ({others})')
check(sig_neg(sel('TON_IoT', 'scaffold_a0.5'), F5), 'TON: SCAFFOLD trails Adam FedAvg at alpha 0.5')
for e in ('fedavg_iid', 'fedavg_a1', 'fedavg_a0.5', 'fedavg_a0.1'):
    check(np.mean([r['test_selected_full']['benign_fpr'] for r in runs('TON_IoT', e).values()]) < 0.04, f'TON: {e} benign FPR below 4%')
for e in ('scaffold_a0.5', 'scaffold_a0.1', 'fedavgsgd_a0.5', 'fedavgsgd_a0.1'):
    check(np.mean([r['test_selected_full']['benign_fpr'] for r in runs('TON_IoT', e).values()]) < 0.07, f'TON: {e} benign FPR below 7%')
for e, d in (('fedprox', 0), ('fedlc', 0), ('scaffold', 0)):
    ref = 'fedavgsgd_a0.5_noports' if e == 'scaffold' else 'fedavg_a0.5_noports'
    check(not sig_pos(sel('TON_IoT', f'{e}_a0.5_noports'), sel('TON_IoT', ref)) and not sig_neg(sel('TON_IoT', f'{e}_a0.5_noports'), sel('TON_IoT', ref)),
          f'TON no ports: {e} matches its reference')
check(sig_neg(sel('TON_IoT', 'fedadam_a0.5_noports'), sel('TON_IoT', 'fedavg_a0.5_noports')), 'TON no ports: FedAdam trails FedAvg')
check(paired(sel('TON_IoT', 'fedavg_a0.5_noports'), sel('TON_IoT', 'central_noports'))[0] < paired(F5, sel('TON_IoT', 'central'))[0],
      'TON: federation costs more without ports (alpha 0.5)')

print('UNSW-specific claims')
cent_fpr = np.mean([r['test_selected_full']['benign_fpr'] for r in runs('UNSW_NB15', 'central').values()])
check(cent_fpr > 0.10, f'UNSW: centralised benign FPR above 10% ({cent_fpr:.3f})')
check(paired(sel('UNSW_NB15', 'fedprox_a0.1'), sel('UNSW_NB15', 'fedavg_a0.1'))[1] < 0.05 and paired(sel('UNSW_NB15', 'fedlc_a0.5'), sel('UNSW_NB15', 'fedavg_a0.5'))[1] < 0.05
      and paired(sel('UNSW_NB15', 'fedlc_a0.5'), sel('UNSW_NB15', 'fedavg_a0.5'))[0] > 0, 'UNSW: FedProx (0.1) and FedLC (0.5) favoured before correction')
sc = sel('UNSW_NB15', 'scaffold_a0.1'); scf = fin('UNSW_NB15', 'scaffold_a0.1')
check(np.std(list(sc.values()), ddof=1) > 0.06 and mean(scf) < mean(sc) - 0.05, 'UNSW: SCAFFOLD erratic at alpha 0.1 (wide spread, final round lower)')
check(min(sel('UNSW_NB15', 'scaffold_a0.1_lr0.1').values()) < 0.15, 'UNSW: SCAFFOLD lr 0.1 collapses in at least one repetition')
weak = [min(sel('UNSW_NB15', e), key=sel('UNSW_NB15', e).get) for e in ('scaffold_a0.1', 'scaffold_a0.1_lr0.01', 'scaffold_a0.1_lr0.1')]
check(len(set(weak)) == 1, f'UNSW SCAFFOLD alpha 0.1: same weakest repetition at every lr ({weak})')
check(mean(sel('UNSW_NB15', 'scaffold_a0.1_lr0.01')) < mean(sel('UNSW_NB15', 'fedavgsgd_a0.1')), 'UNSW SCAFFOLD lr 0.01 does not help')
check(all(float(V[f'UNSW_cli_{a}_owngain'].split()[0]) <= 0.01 and int(V[f'UNSW_cli_{a}_ownneg']) >= 10 for a in ('a05', 'a01')), 'UNSW: no gain on own classes; most pairs better alone')

print('Tuning')
for ds in ('TON_IoT', 'UNSW_NB15', 'N_BaIoT'):
    T3 = json.loads((RES / ds / 'tuned_v3.json').read_text())
    edges = [k for k in ('fedavg', 'fedprox', 'fedadam', 'fedlc', 'scaffold', 'fedavgsgd') if T3.get(f'{k}_edge')]
    check(edges == (['fedprox'] if ds == 'TON_IoT' else []), f'{ds}: grid-edge optima as stated ({edges})')
check(json.loads((RES / 'TON_IoT' / 'tuned_v3.json').read_text())['fedprox'] == 1e-5, 'TON: FedProx edge is the smallest mu')

print('Device-level partitions')
pval = lambda k: float(V.get(k, k).replace('p < ', '').replace('p = ', '').split()[0])
check(holm('TON_d_fedavg_natr_vs_local_natr') and holm('NB_d_fedavg_dev_vs_local_dev'), 'device partitions: FedAvg above local-only, significant after Holm (both)')
gn, g1 = paired(sel('TON_IoT', 'fedavg_natr'), sel('TON_IoT', 'central'))[0], paired(sel('TON_IoT', 'fedavg_a0.1'), sel('TON_IoT', 'central'))[0]
check(abs(gn - g1) < 0.03, f'TON: host partition costs about as much as alpha 0.1 ({gn:.3f} vs {g1:.3f})')
check(abs(float(V['TON_skew_natr_top2_mean'][:-1]) - float(V['TON_skew_a05_top2_mean'][:-1])) < 10 and abs(float(V['TON_skew_natr_cls_mean']) - float(V['TON_skew_a05_cls_mean'])) < 1,
      'TON: host partition label skew similar to alpha 0.5')
check(float(V['NB_skew_dev_top2_mean'][:-1]) < float(V['NB_skew_a05_N9_top2_mean'][:-1]), 'N-BaIoT: device label skew milder than alpha 0.5')
check(float(V['TON_w_fedavg_natr_vs_fedavg_a05_mean']) < 0 and float(V['NB_w_fedavg_dev_vs_fedavg_a05_N9_mean']) < 0,
      'device partitions tend to cost more than alpha 0.5 (both datasets, point estimates)')
check(float(V['TON_w_fedavg_natr_vs_fedavg_a01_mean']) > -0.03 and float(V['NB_w_fedavg_dev_vs_fedavg_a01_N9_mean']) > 0,
      'device partitions do not cost more than alpha 0.1 (as stated)')
check(int(V['n_alg_equiv']) < int(V['n_alg_cmp']), 'only some algorithm comparisons are equivalent at +-0.05 (as stated)')
check(float(V['UNSW_fedlc_a05_fpr_m']) > float(V['UNSW_fedavg_a05_fpr_m']), 'UNSW: FedLC gain comes with a higher benign FPR')
check(float(V['NB_d_fedadam_dev_vs_fedavg_dev_mean']) > 0, 'N-BaIoT: FedAdam has the best point estimate over FedAvg')
check(float(V['TON_dpfull_ddos_loss']) < 10, 'TON full DP: DDoS loses just under ten points')
check(float(V['TON_mitm_s02']) < float(V['TON_mitm_s06']), 'TON DP: MITM not monotone in sigma (as stated)')
check(float(V['TON_fedavg_natr_sel_sd']) > float(V['TON_fedavg_a05_sel_sd']), 'TON: host partition varies more between repetitions than alpha 0.5')
check(float(V['TON_fedavg_natr_sel'].split('± ')[1]) > 2 * float(V['TON_fedavg_nat_sel'].split('± ')[1]), 'TON: fixed host order gave a several times narrower CI')
check(float(V['TON_cli_anatr_owngain'].split()[0]) > 0, 'TON hosts: FedAvg better on own classes (common test set)')
check(int(V['TON_own_pairs_neg']) >= 15 and fl('TON_own_rep_gain') < 0 and pval('TON_own_rep_p') < 0.05, 'TON hosts: local model better on own traffic in most pairs')
check(fl('TON_own_loc_f1') > fl('TON_own_fed_f1'), 'TON hosts: local own-traffic mean above FedAvg')
check(float(V['TON_own_loc_attack_detection']) - float(V['TON_own_fed_attack_detection']) < 0.5 * 100 * (fl('TON_own_loc_f1') - fl('TON_own_fed_f1')),
      'TON hosts: most of the own-traffic gap is confusion between attack types, not missed attacks')
check(int(V['TON_np_own_neg']) >= 13 and fl('TON_np_own_gain') < 0 and pval('TON_np_own_p') < 0.05, 'TON hosts without ports: reversal persists')
check(fl('TON_np_common_gain') > 0 and pval(V['TON_np_common_gain'].replace(', uncorrected', '').rsplit(', ', 1)[1]) < 0.05, 'TON hosts without ports: FedAvg keeps the common-test advantage')
check(int(V['NB_own_pairs_neg']) > int(V['NB_own_pairs_n']) / 2 and fl('NB_own_rep_gain') < 0, 'N-BaIoT: local model better on own traffic in most pairs')
check(int(V['NB_dev_neg']) > int(V['NB_dev_n']) / 2, 'N-BaIoT: FedAvg worse on own traffic for most devices')
for alg, ref in (('fedprox', 'fedavg'), ('fedadam', 'fedavg'), ('fedlc', 'fedavg'), ('scaffold', 'fedavgsgd')):
    check(not sig_pos(sel('TON_IoT', f'{alg}_natr'), sel('TON_IoT', f'{ref}_natr')), f'TON hosts: {alg} not significantly better than {ref}')
    check(not (holm(f'NB_d_{alg}_dev_vs_{ref}_dev') and paired(sel('N_BaIoT', f'{alg}_dev'), sel('N_BaIoT', f'{ref}_dev'))[0] > 0), f'N-BaIoT: {alg} not significantly better than {ref} after Holm')

check(fl('NB_own_fed_attack_detection') >= 99 and fl('NB_own_loc_attack_detection') >= 99, 'NB own traffic: both models detect almost every attack (gap = confusion between attack types)')
check(fl('TON_own_loc_attack_detection') - fl('TON_own_fed_attack_detection') > 1 and fl('TON_own_loc_attack_detection') - fl('TON_pers_fed_site_attack_detection') > 1, 'TON own traffic: FedAvg misses more own attacks than the local model in both run sets')
check(fl('RERUN_MAXDIFF') >= 0.05, 'rerun noise quoted as comparable to several effects')
print('Personalisation')
for t in ('TON', 'NB'):
    if t == 'TON':
        check(abs(fl(f'{t}_pers_full1_gain')) < 0.03, f'{t}: one epoch of fine-tuning closes the own-traffic gap')
    else:
        check(0.35 < fl(f'{t}_pers_full1_gain') / fl(f'{t}_pers_fed_gain') < 0.65, f'{t}: one epoch of fine-tuning closes about half of the own-traffic gap')
    check(fl(f'{t}_pers_loc_common') < fl(f'{t}_pers_full1_common') < fl(f'{t}_pers_fed_common'), f'{t}: fine-tuned common-test score between local and federated')
    if t == 'TON':
        check(fl(f'{t}_pers_low1_common') > fl(f'{t}_pers_full1_common') and fl(f'{t}_pers_prox1_common') > fl(f'{t}_pers_full1_common')
              and abs(fl(f'{t}_pers_low1_gain')) < 0.03 and abs(fl(f'{t}_pers_prox1_gain')) < 0.03, f'{t}: low-lr and proximal fine-tuning keep more breadth at the same own-traffic accuracy')
    else:
        check(fl(f'{t}_pers_low1_common') > fl(f'{t}_pers_full1_common') + 0.01 and fl(f'{t}_pers_low1_gain') > fl(f'{t}_pers_full1_gain'),
              f'{t}: low-lr fine-tuning keeps more breadth and comes closer on own traffic')
        check(abs(fl(f'{t}_pers_prox1_common') - fl(f'{t}_pers_full1_common')) < 0.01 and fl(f'{t}_pers_prox1_gain') <= fl(f'{t}_pers_full1_gain'),
              f'{t}: proximal fine-tuning neither keeps more breadth nor comes closer on own traffic')
    for E in ('2', '5'):
        check(abs(fl(f'{t}_pers_full{E}_gain') - fl(f'{t}_pers_full1_gain')) < 0.03, f'{t}: more epochs change little ({E})')
check(int(V['TON_pers_full1_giveup']) > int(V['NB_pers_full1_giveup']) + 20, 'fine-tuning cost in breadth depends on the dataset (TON much larger than N-BaIoT)')
check(float(V['TON_pers_full1_either_site_attack_detection']) > float(V['TON_pers_fed_site_attack_detection']) and
      float(V['TON_pers_full1_either_test_benign_fpr']) > 2 * float(V['TON_pers_fed_test_benign_fpr']),
      'TON: two-model alerting raises own-traffic detection but many false alerts on unfamiliar traffic')

print('Normalisation')
for k in ('central_bn', 'fedavg_bn_a05'):
    check(float(V[f'TON_{k}_recal_swing']) < 0.5 * float(V[f'TON_{k}_swing']), f'TON {k}: pooled recalibration more than halves the swing')
check(float(V['TON_central_swing']) < 0.5 * float(V['TON_central_bn_swing']), 'TON: GN swing much smaller than BN swing')
check(float(V['TON_bnfr_a05_fr_swing']) < 0.3 * float(V['TON_bnfr_a05_swing']), 'TON: federated re-estimation removes most of the swing')
check(float(V['TON_bnfr_a05_fin_sd']) < 2 * float(V['TON_fedavg_a05_fin_sd']) and float(V['TON_bnfr_a05_run_fin_sd']) > 4 * float(V['TON_bnfr_a05_fin_sd']),
      'TON: final round as predictable as GN after re-estimation, much less with server-averaged statistics')
check(pval(V['TON_d_bnfr_a05_run_vs_gn'].split('], ')[1].replace(', uncorrected', '')) > 0.05, 'TON: BN (server-averaged) vs GN at selected round not significant')
check(fl('TON_d_bnfr_a05_fr_vs_gn') > 0 and fl('TON_d_bnfr_a01_fr_vs_gn') > 0, 'TON: re-estimated BN exceeds GN at both alpha')
fs = lambda e: float(np.std(list(fin('TON_IoT', e).values()), ddof=1))
check(fs('central_bn') > 5 * fs('central'), 'TON: centralised BN final-round spread much larger')
check(float(V['UNSW_bnfr_a05_swing']) < 0.03 and float(V['UNSW_bnfr_a05_fr_swing']) < 0.03, 'UNSW: swing small with either statistics')

print('DP tuning and per-class wording')
check(float(V['TON_dpt_C2_sel']) > max(float(V['TON_dpt_C05_sel']), float(V['TON_dpt_C4_sel'])), 'TON DP: best clipping norm inside the grid (C = 2)')
check(float(V['TON_dpt_C2_sel']) - float(V['TON_dp_sigma1_sel_mean']) < 0.05, 'TON DP: tuning the clip norm recovers only a few points')
check(float(V['TON_dpt_r40_nonoise_sel']) - float(V['TON_dpt_r40_sigma1_sel']) > 0.08, 'TON DP: 40-round schedule keeps a gap of > 8 points')
fn, f18 = mean(sel('TON_IoT', 'dp_full_nonoise')), mean(sel('TON_IoT', 'dp_full_sigma1.8'))
e18 = max(r['dp']['epsilon_max'] for r in runs('TON_IoT', 'dp_full_sigma1.8').values())
check(fn - f18 > float(V['TON_dpt_r40_nonoise_sel']) - float(V['TON_dpt_r40_sigma1_sel']), f'TON DP: full budget gap ({fn - f18:.3f}) larger than the 40-round gap')
check(2.5 < e18 < 3.5, f'TON DP: sigma 1.8 at full budget keeps eps near 3 ({e18:.2f})')
for ds in ('TON_IoT', 'UNSW_NB15'):
    for p in ('iid', 'a0.5', 'a0.1'):
        g100 = mean(sel(ds, f'fedavg_{p}_r100')) - mean(sel(ds, 'central_r100'))
        check(g100 < -0.02, f'{ds}: FedAvg {p} still trails centralised after 100 rounds ({g100:.3f})')
att = [c for c in runs('TON_IoT', 'dp_noclip_nonoise')[0]['test_selected_full']['per_class_f1'] if c != 'normal']
lost = [c for c in att if pc('TON_IoT', 'dp_noclip_nonoise', c) - pc('TON_IoT', 'dp_sigma0.6', c) >= 0.10]
check(len(lost) == int(V['TON_dp_nlost_06_code']), f'TON sigma 0.6: {len(lost)} attack classes lose at least ten points ({lost})')
lostf = [c for c in att if pc('TON_IoT', 'dp_full_nonoise', c) - pc('TON_IoT', 'dp_full_sigma1.8', c) >= 0.10]
check(len(lostf) == int(V['TON_dpfull_nlost']) and len(lostf) > len(att) / 2, f'TON full budget: most attack classes lose at least ten points ({lostf})')
check(sorted(c for c in att if c not in lostf) == ['ddos', 'dos'], 'TON full budget: only DDoS and DoS spared (and benign)')
c3, c6, c25 = (float(V[f'TON_dp_cost_short_{k}']) for k in ('03', '06', '25'))
check(c3 >= 0.6 * c25 and max(c3, c6, c25) - min(c3, c6, c25) <= 4, f'TON DP: most of the cost appears with the first noise, nearly flat after ({c3}, {c6}, {c25})')
check(pc('TON_IoT', 'dp_sigma1.5', 'mitm') < 0.01 and pc('TON_IoT', 'dp_sigma2.5', 'mitm') < 0.01, 'TON: MITM not detected from sigma 1.5')
gd = pc('UNSW_NB15', 'dp_noclip_nonoise', 'generic') - pc('UNSW_NB15', 'dp_sigma1', 'generic')
check(0.06 <= gd <= 0.14, f'UNSW: generic loses about ten points ({gd:.2f})')

print('Differential privacy')
for ds, tag in (('TON_IoT', 'TON'), ('UNSW_NB15', 'UNSW')):
    nn, cn, ref = mean(sel(ds, 'dp_noclip_nonoise')), mean(sel(ds, 'dp_clip_nonoise')), mean(sel(ds, 'fedavg_gn_short'))
    check(abs(nn - ref) < 0.04, f'{tag}: DP code without noise is close to non-private FedAvg ({nn:.3f} vs {ref:.3f})')
    check(abs(cn - nn) < 0.03, f'{tag}: clipping alone costs little ({cn:.3f} vs {nn:.3f})')
    sg = [mean(sel(ds, f'dp_sigma{s}')) for s in ('0.6', '1', '1.5', '2.5')]
    check(nn - max(sg) > 0.04 and max(sg) - min(sg) < 0.04, f'{tag}: noise costs points and cost nearly flat across sigma ({[round(x, 3) for x in sg]})')
    eps = [r['dp']['epsilon_max'] for r in runs(ds, 'dp_sigma2.5').values()]
    check(all(e < 2 for e in eps), f'{tag}: strongest noise gives eps < 2 ({max(eps):.2f})')
for c in ('normal', 'dos', 'ddos'):
    check(pc('TON_IoT', 'dp_noclip_nonoise', c) - pc('TON_IoT', 'dp_sigma0.6', c) < 0.05, f'TON: {c} almost unaffected by noise')
check(pc('TON_IoT', 'dp_sigma0.6', 'password') < 0.4 * pc('TON_IoT', 'dp_noclip_nonoise', 'password'), 'TON: password largely lost at weakest noise')
for c in ('backdoor', 'ransomware', 'scanning'):
    dlt = pc('TON_IoT', 'dp_noclip_nonoise', c) - pc('TON_IoT', 'dp_sigma0.6', c)
    check(0.08 <= dlt <= 0.22, f'TON: {c} loses 10-20 points ({dlt:.2f})')
for c in ('backdoor', 'dos', 'shellcode'):
    check(pc('UNSW_NB15', 'dp_sigma0.6', c) < 0.05, f'UNSW: {c} disappears under noise')
for c, lim in (('normal', 0.055), ('exploits', 0.055), ('fuzzers', 0.055), ('reconnaissance', 0.11), ('generic', 0.12)):
    check(pc('UNSW_NB15', 'dp_noclip_nonoise', c) - min(pc('UNSW_NB15', f'dp_sigma{x}', c) for x in ('0.6', '1', '1.5', '2.5')) < lim, f'UNSW: {c} loses less than {lim:.2f} under noise')
for c in ('analysis', 'worms'):
    check(max(pc('UNSW_NB15', e, c) for e in ('dp_noclip_nonoise', 'dp_clip_nonoise', 'dp_sigma0.6', 'dp_sigma1', 'dp_sigma1.5', 'dp_sigma2.5')) < 0.1, f'UNSW: {c} near zero in every DP configuration')
e06 = np.mean([r['dp']['epsilon_max'] for r in runs('UNSW_NB15', 'dp_sigma0.6').values()])
e06t = np.mean([r['dp']['epsilon_max'] for r in runs('TON_IoT', 'dp_sigma0.6').values()])
check(e06 > e06t, 'UNSW: larger eps than TON for the same noise (smaller clients)')

print('Settings quoted in the text')
check(all(len(runs(ds, e)) == 5 for ds in ('TON_IoT', 'UNSW_NB15') for e in ('central', 'fedavg_a0.5', 'fedprox_a0.1', 'scaffold_a0.1', 'fedavg_bn_a0.5_fedrecal')), 'core and fed-recal experiments have five repetitions')
check(all(len(runs('TON_IoT', e)) == 5 for e in ('fedavg_natr', 'local_natr', 'fedlc_natr', 'fedavg_natr_pers', 'local_natr_noports', 'fedavg_natr_noports_ft')), 'TON host partition: five repetitions')
check(all(len(runs('TON_IoT', e)) == 3 for e in ('dp_sigma0.2', 'dp_sigma0.3', 'dp_full_sigma1.8')), 'small-noise and full-budget DP: three repetitions')
check(all(len(runs('N_BaIoT', e)) == 5 for e in ('central', 'local_dev', 'fedavg_dev', 'fedprox_dev', 'fedadam_dev', 'fedlc_dev', 'scaffold_dev', 'fedavgsgd_dev',
                                                   'fedavg_iid9', 'fedavg_a0.5_N9', 'fedavg_a0.1_N9', 'fedavg_dev_pers')), 'N-BaIoT: five repetitions everywhere')
check(all(json.loads(p.read_text())['cfg'].get('partition') == 'device' for p in (RES / 'N_BaIoT').glob('tune3_*/seed*.json')), 'N-BaIoT tuned on the device partition (as stated)')
NBI = json.loads((HERE.parents[1] / 'Papers pipeline' / 'dataset4' / 'nbaiot_prepared.json').read_text())
check(len(NBI['classes']) == 9 and set(NBI['dropped_classes']) == {'gafgyt_tcp', 'gafgyt_udp'} and max(NBI['dropped_classes'].values()) < 100, 'N-BaIoT: BASHLITE floods dropped (fewer than 100 distinct records)')
check(all(len(runs(ds, e)) == 3 for ds in ('TON_IoT', 'UNSW_NB15') for e in ('scale_N16', 'central_bn', 'dp_sigma1')), 'scaling, normalisation and DP have three repetitions')
check('repository DOI to be added' not in text, 'repository link is final')

words = len(re.findall(r'\w+', text.split('## References')[0]))
print(f'pending placeholders (diagnostic runs not yet done): {len(PENDING)}: ' + ', '.join(sorted(PENDING)))
print(f'\n{len(fails)} failures; ~{words} words before references')
sys.exit(len(fails))
