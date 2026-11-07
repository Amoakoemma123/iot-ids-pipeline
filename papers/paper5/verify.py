"""Check Manuscript_v2.docx (v3 text): no unfilled placeholders, and every qualitative claim in the
text holds in the result files. Exit code = number of failures."""
import json
import re
import sys
from pathlib import Path

from docx import Document

HERE = Path(__file__).resolve().parent
PIPE = HERE.parents[1]   # repository root
R3 = PIPE / 'experiments' / 'paper5_xai' / 'results_v3'
R2 = PIPE / 'experiments' / 'paper5_xai' / 'results_v2'
V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
SEEDS = (42, 43, 44)
DSS = ['TON_IoT', 'UNSW_NB15']
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def R(ds, stage, seed=None):
    p = R3 / ds / (f'{stage}.json' if seed is None else f'seed{seed}/{stage}.json')
    return json.loads(p.read_text())


def R2_(ds, stage):
    return json.loads((R2 / ds / f'{stage}.json').read_text())


text = (HERE / 'Manuscript_filled.md').read_text(encoding='utf-8')
_sp = HERE / 'Supplement_filled.md'   # detail moved to the Supplementary Material (12k-word cap)
supp = _sp.read_text(encoding='utf-8') if _sp.exists() else ''
text_all = text + chr(10) + supp

print('Placeholders and anonymity')
for name in ('Manuscript_v2.docx', 'Manuscript_v2_anon.docx'):
    p = HERE / name
    if p.exists():
        d = Document(p)
        body = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
        check('⟦' not in body and '{{' not in body, f'{name}: no unfilled placeholders')
        if 'anon' in name:
            check(not any(n in body for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol')) and not d.core_properties.author,
                  f'{name}: anonymous')
check('⟦' not in text, 'Manuscript_filled.md: no unfilled placeholders')
check(bool(supp) and '⟦' not in supp, 'Supplement_filled.md built, no unfilled placeholders')
for n in range(1, 5):
    check(re.search(rf'Tables? (?:S\d+ and )?S{n}(?!\d)', text_all) is not None, f'Table S{n} cited')
_words = len(text[text.index('## Abstract'):].split())
check(_words <= 12000, f'main text incl. abstract and references within the 12,000-word cap ({_words})')
check('<!-- UNSW_' not in (HERE / 'Manuscript.md').read_text(encoding='utf-8'), 'no UNSW to-do markers left in Manuscript.md')
for n in range(1, 5):
    check(f'Figure {n}' in text, f'Figure {n} cited in the text')

print('§3.3 audit and decomposition')
D = R('UNSW_NB15', 'leakage_decomposition')
A, B, C, Dd = D['A_original'], D['B_id_removed_duplicates_kept'], D['C_id_kept_duplicates_removed'], D['D_corrected']
check(A['seen_test']['accuracy'] < A['unseen_test']['accuracy'], 'test records with a copy in training classified LESS accurately (original model)')
check(V['unsw_dupes_with_id'] == '0', 'with id present no row is a duplicate')
check(V['unsw_sttl31_attack_pct'] == '0.0', 'sttl=31 occurs only in benign traffic')
Df = R('UNSW_NB15', 'leakage_decomposition_fixed')
check(all(r['id_effect']['lo'] > 0 for r in Df['dups_kept']['seeds'].values()), 'fixed features, duplicates kept: id effect significant in every seed')
check(all(r['id_effect']['lo'] <= 0 for r in Df['dups_removed']['seeds'].values()), 'fixed features, duplicates removed: id effect CI includes zero in every seed')
check(abs(Df['dups_removed']['seeds']['42']['without_id']['macro_f1'] - float(V['UNSW_rf_f1'])) < 1e-3, 'fixed-feature corrected condition reproduces the baseline RF')
check(A['full_test']['macro_f1'] > B['full_test']['macro_f1'] and C['full_test']['macro_f1'] > Dd['full_test']['macro_f1'], 're-selection variant: id raises macro-F1 in both pairs')
for ds, tag in (('TON_IoT', 'TON'), ('UNSW_NB15', 'UNSW')):
    Cp = R(ds, 'feature_space_copies')
    check(Cp.get('definition', '').startswith('exact'), f'{ds}: feature-space copies use exact equality')
    r = Cp['seeds']['42']['rf']
    if ds == 'UNSW_NB15':
        check(r['macro_f1_without_copies'] > r['macro_f1_all'], 'UNSW: copies are harder (excluding them raises RF macro-F1)')
    else:
        check(Cp['share_copy_with_label'] < 0.01, 'TON: exact feature-space coincidences are rare (< 1%)')

print('§3.4 models')
for ds, tag in (('TON_IoT', 'TON'), ('UNSW_NB15', 'UNSW')):
    p = R(ds, 'performance', 42)
    check(p['cnn']['file_mb'] < 1 and p['rf']['pickle_mb'] > 50, f'{ds}: CNN < 1 MB, RF > 50 MB')
    check(all(R(ds, 'performance', s)[m]['ms_per_1000'] < 100 for s in SEEDS for m in ('rf', 'cnn')), f'{ds}: 1,000 records in < 0.1 s')
check(R('UNSW_NB15', 'performance', 42)['rf']['benign_fpr'] > 0.1, 'UNSW: large share of benign records misclassified (>10%)')
check(float(V['UNSW_v2_aerf_f1']) > float(V['UNSW_v2_cnn_f1']) and float(V['UNSW_v2_rf_f1']) > float(V['UNSW_v2_aerf_f1']) and
      float(V['TON_v2_rf_f1']) == max(float(V[f'TON_v2_{m}_f1']) for m in ('rf', 'cnn', 'lr', 'lstm', 'aerf')), 'RF best on both; AE-RF > CNN on UNSW (as stated)')

print('§5.1 global importance')
g = R2_('TON_IoT', 'global_shap')
check(all(r['rank_lo'] == r['rank_hi'] == i + 1 for i, r in enumerate(g['rf']['ranking'][:3])), 'TON RF top-3 have single-rank 95% intervals')
check(g['cross_model']['jaccard@5'] < 1, 'TON RF and CNN share only part of their top-5')
check(sum('port' in r['feature'] for r in g['rf']['ranking'][:5]) >= 1 and sum('port' in r['feature'] for r in g['cnn']['ranking'][:5]) >= 1,
      'TON: port features in the top-5 of both models')
g = R2_('UNSW_NB15', 'global_shap')
check([r['feature'] for r in g['rf']['ranking'][:2]] == ['sttl', 'ct_state_ttl'] and {r['feature'] for r in g['cnn']['ranking'][1:3]} == {'sttl', 'ct_state_ttl'},
      'UNSW: sttl/ct_state_ttl are RF ranks 1-2 and CNN ranks 2-3')

print('§5.2 agreement, stability, faithfulness')
for ds in DSS:
    for m in ('rf', 'cnn'):
        S = R(ds, 'local_c', 42)[m]['summary']
        check(1.8 <= S['shared5']['mean'] <= 3.2, f'{ds} {m}: typically two or three shared top-5 features ({S["shared5"]["mean"]:.2f})')
        check(S['j5_chance_corrected']['lo'] > 0, f'{ds} {m}: agreement above chance')
        check(S['j5']['hi'] < 0.75, f'{ds} {m}: agreement far from complete (J5 CI < 0.75)')
        Dl = {s: R(ds, 'deletion_c', s)[m] for s in SEEDS}
        _c = [Dl[s]['paired'][f'{st}_shap_minus_lime_{k}'] for s in SEEDS for st in ('mean', 'record', 'gauss', 'condx') for k in (1, 3, 5)]
        if m == 'rf':
            check(all(d['mean'] > 0 for d in _c), f'{ds} rf: SHAP > LIME (mean) in every seed, strategy and k')
        else:
            check(sum(d['mean'] > 0 for d in _c) >= 34 and not any(d['hi'] < 0 for d in _c), f'{ds} cnn: SHAP > LIME in >=34/36 cells, none significantly negative')
        for st in ('mean', 'record', 'gauss', 'condx'):
            check(sum(Dl[s]['paired'][f'{st}_shap_minus_lime_{k}']['lo'] > 0 for s in SEEDS for k in (1, 3, 5)) >= 5,
                  f'{ds} {m} {st}: SHAP significantly above LIME in most cells ("under every replacement strategy")')
        r = Dl[42]['summary']['condx_shap_3']['mean'] / Dl[42]['summary']['mean_shap_3']['mean']
        check(r < 0.6, f'{ds} {m}: conditional drop much smaller than mean-replacement drop ({r:.2f})')
    loc = R(ds, 'local_c', 42)
    _dd = loc['rf']['summary_correct_only']['j5']['mean'] - loc['rf']['summary']['j5']['mean']
    check(0 <= _dd < 0.03 or ds == 'TON_IoT' and abs(_dd) < 0.03, f'{ds}: excluding misclassified records changes RF agreement little (slightly raises it on UNSW: {_dd:+.3f})')
check(min(int(V[f'{t}_{m}_d_lime_rand_sig']) for t in ('TON', 'UNSW') for m in ('rf', 'cnn')) == int(V['UNSW_rf_d_lime_rand_sig']),
      'LIME weakest (fewest LIME>random) for the UNSW RF')
_u = R('UNSW_NB15', 'deletion_c', 42)['rf']['summary']
check(abs(_u['mean_lime_3']['mean'] - _u['mean_rand_3']['mean']) < 0.02, 'UNSW RF: LIME top-3 no better than random under mean replacement')
_u = R('UNSW_NB15', 'local_c', 42)
check(_u['rf']['summary']['j5_chance_corrected']['mean'] < 0.1 < _u['cnn']['summary']['j5_chance_corrected']['mean'],
      'UNSW: RF (not CNN) agreement barely above chance')
L = R('TON_IoT', 'local_c', 42)
check(L['cnn']['summary']['fj5']['mean'] - L['cnn']['summary']['j5']['mean'] > L['rf']['summary']['fj5']['mean'] - L['rf']['summary']['j5']['mean'] > 0,
      'TON: field-level merging raises agreement, more for the CNN than the RF')
check(L['rf']['summary']['rho']['mean'] < 0.2 < L['cnn']['summary']['rho']['mean'], 'TON: RF Spearman rho low, CNN higher')
for ds in DSS:
    for m in ('rf', 'cnn'):
        for st in ('deletion', 'deletion_c'):
            check(not any(R(ds, st, s_)[m]['paired'][k]['hi'] < 0 for s_ in SEEDS for k in R(ds, st, s_)[m]['paired'] if '_shap_minus_lime_' in k),
                  f'{ds} {m} {st}: LIME never significantly above SHAP')
        raw, con = R(ds, 'local', 42)[m]['summary']['j5']['mean'], R(ds, 'local_c', 42)[m]['summary']['j5']['mean']
        check((raw > con) == (m == 'rf'), f'{ds} {m}: raw-weight ranking raises RF agreement, lowers CNN agreement')
check(L['cnn']['summary']['lime_r2']['mean'] < L['rf']['summary']['lime_r2']['mean'], 'TON: LIME surrogate fits CNN worse than RF')
for ds in DSS:
    St = R(ds, 'stability_c', 42)['rf'], R(ds, 'stability_c', 42)['cnn']
    rf, cnn = St
    check(cnn['summary']['stab_5000']['mean'] - cnn['summary']['stab_1000']['mean'] < rf['summary']['stab_5000']['mean'] - rf['summary']['stab_1000']['mean'] + 0.02 or ds == 'UNSW_NB15',
          f'{ds}: 5,000 samples help the CNN less than the RF')
    check(rf['summary']['shap_self']['mean'] > 0.8 and cnn['summary']['shap_self']['mean'] < 0.7, f'{ds}: TreeSHAP stable (>0.8), CNN SHAP at default unstable (<0.7)')
    check(cnn['summary']['shap_self']['mean'] < cnn['summary']['stab_1000']['mean'], f'{ds}: CNN SHAP less stable than LIME at default')
    if ds == 'TON_IoT':
        check(all(St_[m]['paired_consensus_minus_single']['lo'] <= 0 for St_ in [R(ds, 'stability_c', 42)] for m in ('rf', 'cnn')),
              f'{ds}: consensus LIME not significantly closer to SHAP (RF and CNN), as stated')
    Bb = R(ds, 'shap_budget', 42)
    c = {k: v['self_agreement']['mean'] for k, v in Bb['cnn'].items()}
    check(c['fixed_bg_ns1000'] > c['fixed_bg_ns200'] + 0.08, f'{ds}: part of CNN SHAP instability is sampling noise (1,000 vs 200 samples)')
    if ds == 'TON_IoT':
        check(0.5 <= c['fixed_bg_ns1000'] <= 0.65, 'TON: CNN SHAP at 1,000 samples has top-5 Jaccard of only about 0.6')
    check(c['fixed_bg_ns1000'] - c['new_bg_ns1000'] < 0.06, f'{ds}: new backgrounds lower CNN SHAP self-agreement only a little')
    check(abs(Bb['rf']['bg1000']['self_agreement']['mean'] - Bb['rf']['bg200']['self_agreement']['mean']) < 0.03, f'{ds}: RF background size does not matter')
St = R('TON_IoT', 'stability_c', 42)
check(St['rf']['summary']['stab_5000']['mean'] > St['rf']['summary']['stab_1000']['mean'] + 0.1, 'TON RF: 5,000 samples stabilise LIME')
L42 = R('TON_IoT', 'local_c', 42)['cnn']['summary']; S42 = R('TON_IoT', 'stability_c', 42)['cnn']['summary']
check(L42['j5']['mean'] > S42['shap_self']['mean'] - 0.2,
      'TON: CNN cross-method agreement not far below SHAP self-agreement at defaults (within 0.2)')

print('§5.3 counterfactuals')
for ds in DSS:
    for s in SEEDS:
        C = R(ds, 'counterfactual', s)
        for m in ('rf', 'cnn'):
            check(C[m]['random']['success'] >= 0.99, f'{ds} {m} seed {s}: every detected attack evadable with any group')
            check(C[m]['attacker']['success'] < C[m]['random']['success'] - 0.3, f'{ds} {m} seed {s}: attacker-only far less successful')
CU = {s: R('UNSW_NB15', 'counterfactual', s) for s in SEEDS}; CT = {s: R('TON_IoT', 'counterfactual', s) for s in SEEDS}
ur = [CU[s]['rf']['attacker']['success'] for s in SEEDS]; tr = [CT[s]['rf']['attacker']['success'] for s in SEEDS]
check(min(tr) - 0.05 <= sum(ur) / 3 <= max(tr) + 0.05, 'UNSW RF attacker success similar to TON')
check(ur[0] > tr[0] and all(u < t for u, t in zip(ur[1:], tr[1:])), 'UNSW RF attacker success above TON in seed 42 only (as stated)')
check(min(CU[s]['cnn']['attacker']['success'] for s in SEEDS) > max(CT[s]['cnn']['attacker']['success'] for s in SEEDS),
      'UNSW CNN attacker success higher than TON in every seed')
for s in SEEDS:
    g_ = CU[s]['rf']['attacker']['group_counts']
    check(max(g_, key=g_.get) == 'src_volume' and g_.get('sttl', 0) < g_['src_volume'], f'UNSW RF seed {s}: source volume main route, sttl weaker')
    check(set(CU[s]['attacker_groups']) == {'sttl', 'src_volume'} and CU[s]['attacker_groups']['sttl'] == ['sttl'], f'UNSW seed {s}: strict attacker groups')
    gs = [f for g_, fs in CU[s]['groups'].items() for f in fs]
    check(len(gs) == len(set(gs)), f'UNSW seed {s}: dependency groups do not overlap')
C = R('TON_IoT', 'counterfactual', 42)
check(C['rf']['shap']['success'] <= C['rf']['random']['success'], 'TON RF: SHAP guidance does not raise success')
check(C['cnn']['shap']['queries_median'] < C['cnn']['random']['queries_median'] / 2, 'TON CNN: SHAP guidance cuts median queries')
check(C['rf']['random']['group_counts'].get('protocol_state', 0) > 0.5 * C['rf']['random']['k']['n'], 'TON RF: protocol state in most CFs')
check(CU[42]['cnn']['shap']['queries_median'] > CU[42]['cnn']['random']['queries_median'], 'UNSW CNN: SHAP guidance raises median queries')

print('§5.4 hardening')
ARMS = ('original', 'counterfactual', 'control', 'control_copy')
for ds, tag in (('TON_IoT', 'TON'), ('UNSW_NB15', 'UNSW')):
    H = [R(ds, 'hardening', s) for s in SEEDS]
    Ha = [R(ds, 'hardening_attacker', s) for s in SEEDS]
    check(all(h[a]['evasion'] >= 0.99 for h in H for a in ARMS), f'{ds} all-groups: evasion ~100% for every arm and seed')
    check(all(abs(h[a]['macro_f1'] - h['original']['macro_f1']) < 0.01 for h in Ha for a in ARMS), f'{ds} attacker: macro-F1 within 0.01 for every arm')
    check(all(abs(h[a]['benign_fpr'] - h['original']['benign_fpr']) < 0.005 for h in Ha for a in ARMS), f'{ds} attacker: benign FPR within 0.5 points')
    check(all(h['counterfactual']['evasion'] < h['control']['evasion'] - 0.05 for h in Ha), f'{ds} attacker: counterfactuals beat interpolation control by > 5 points')
    if ds == 'TON_IoT':
        check(all(h['control_copy']['evasion'] - h['counterfactual']['evasion'] < 0.07 and h['control_copy']['evasion'] < h['original']['evasion'] - 0.15 for h in Ha),
              'TON attacker: benign-donor control almost as effective (within 7 points, > 15 points below original)')
    else:
        check(all(h['control_copy']['evasion'] > h['counterfactual']['evasion'] + 0.05 for h in Ha), 'UNSW attacker: benign-donor control does not match counterfactuals')
    red = 1 - float(V[f'{tag}_hard_att_counterfactual_ev']) / float(V[f'{tag}_hard_att_original_ev'])
    if ds == 'TON_IoT':
        check(red > 0.5, f'TON attacker: evasion more than halved ({red:.2f})')
        pc = {c: sum(h['counterfactual']['evasion_per_class'].get(c, 0) for h in Ha) / 3 for c in Ha[0]['original']['evasion_per_class']}
        check(pc['injection'] < 0.1 and pc['password'] < 0.1 and pc['ransomware'] > 0.5, 'TON attacker: injection/password almost gone, ransomware mostly evadable')
    else:
        check(0.18 <= red <= 0.35, f'UNSW attacker: evasion cut by about a quarter ({red:.2f})')
        check(all(abs(h[a]['evasion'] - h['original']['evasion']) < 0.06 for h in Ha for a in ('control', 'control_copy')), 'UNSW: both controls stay near the original')

print('§5.5 shortcuts')
P = R2_('TON_IoT', 'ports')
check(P['macro_f1_with_ports'] - P['macro_f1_without_ports'] < 0.05, 'TON: removing ports costs < 5 F1 points')
T = R('UNSW_NB15', 'ttl_reselect', 42)
check(T['sttl_rule_test']['binary_accuracy'] > 0.8, 'UNSW: sttl rule > 80% binary accuracy on test')
check(not any('ttl' in f for f in T['features_selected']), 'UNSW TTL removal: no TTL feature re-selected')
check(0 < float(V['UNSW_rf_f1']) - T['rf_macro_f1'] < 0.05, 'UNSW TTL removal costs the RF a little (0-5 points)')

print('Consistency with the run settings')
for k, want in (('N_LOCAL', 'up to {} test records per class'), ('N_RUNS', 'LIME {} times'), ('N_STAB', 'For {} test records per class')):
    check(want.format(V[k]) in text_all, f'text quotes {k}={V[k]}')
for tag in ('TON', 'UNSW'):
    check(V[f'{tag}_hard_att_nseeds'] == '3' and V[f'{tag}_hard_all_nseeds'] == '3', f'{tag} hardening has three seeds')
check('v2 release to be added' not in text, 'data-availability link is final')

print('Final review round (re-analysis of saved outputs)')
fv = lambda k: float(V[k].split()[0])
for t in ('TON', 'UNSW'):
    for m in ('rf', 'cnn'):
        above = fv(f'{t}_{m}_j5_shap_global') > fv(f'{t}_{m}_j5')
        check(above == (not (t == 'TON' and m == 'cnn')), f'{t} {m}: local SHAP top-5 overlaps global top-5 more than LIME (except TON CNN), as stated')
        check((float(V[f'{t}_{m}_j5_norm']) > 0.5) == (m == 'cnn'), f'{t} {m}: normalised agreement > 0.5 for the CNN only, as stated')
    check(int(V[f'{t}_rf_d_shap_rand_sig']) == int(V[f'{t}_cnn_d_shap_rand_sig']) == int(V[f'{t}_rf_d_shap_lime_cells']), f'{t}: SHAP beats random in all combinations')
for m in ('rf', 'cnn'):
    check(abs(fv(f'UNSW_{m}_j5_notwin') - fv(f'UNSW_{m}_j5')) < 0.03, f'UNSW {m}: agreement essentially unchanged without feature-space twins')
check(float(V['UNSW_majority_bin']) < float(V['UNSW_sttl_rule_bin']), 'UNSW: sttl rule beats the binary majority baseline')
check(V['UNSW_nfields_local'] == V['UNSW_nfeat'], 'UNSW: no one-hot features (fields = features), as stated')
check(V['UNSW_rfecv_at_floor'] == 'True' and V['TON_rfecv_at_floor'] == 'False', 'RFECV at its floor on UNSW only, as stated')
check(int(V['TON_cf_att_min'] if 'TON_cf_att_min' in V else V['cf_att_min']) < int(V['cf_att_max']), 'attacker-controllable range well defined')
check(V['TON_cnn_cf_attacker_succ_3s_mean'] == V['cf_att_min'], 'min of attacker-controllable range is the TON CNN')
for t in ('TON', 'UNSW'):
    if f'{t}_hp_reproduces' in V:
        check(V[f'{t}_hp_reproduces'] == 'True' or float(V[f'{t}_hp_maxdiff']) <= 1.0, f'{t}: hardening_paired reproduces Table 9 (exactly, or within 1 point as stated in the text)')
        if V[f'{t}_hp_reproduces'] != 'True':
            check('second run of the hardening experiment' in text_all, f'{t}: small rerun difference disclosed')
    else:
        print(f'  PENDING {t}: hardening_paired (queue p5final) not finished')
for k in ('ls_lead_cells', 'decf_interaction_ci', 'TON_rf_cfmono_mono'):
    if k not in V:
        print(f'  PENDING {k} (queue p5final) not finished')

words = len(re.findall(r'\w+', text.split('## References')[0]))
print(f'\n{len(fails)} failures; ~{words} words before references')
sys.exit(len(fails))
