"""Paper 7: check Manuscript_filled.md. Exit code = number of failures.

1. no unfilled placeholders (except the ones that need author input, reported as PENDING);
2. every qualitative statement in the text holds in values.json (which is computed from result files);
3. the catalogue has as many entries as the text claims;
4. in-text citations [Author Year] and the reference list match.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
TXT = (HERE / 'Manuscript_filled.md').read_text(encoding='utf-8')
AUTHOR_INPUT = set()   # the audit is reported anonymously (17 Nov 2026): no right-of-reply sentence
fails, pending = [], []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def f(k):
    """numeric value of a key: first number in the string, unicode minus handled."""
    s = str(V[k]).replace('−', '-').replace('+', '')
    return float(re.search(r'-?\d+(?:\.\d+)?', s.replace(',', '')).group())


def rng(k):
    a, b = re.findall(r'\d+(?:\.\d+)?', str(V[k]))[:2]
    return float(a), float(b)


print('1. placeholders')
body = TXT.split('\n', 1)[1]          # line 1 is the file comment
left = set(re.findall(r'⟦([^⟧]+)⟧', body))
for k in sorted(left & AUTHOR_INPUT):
    print('  PEND ' + k + ' (author input)'); pending.append(k)
check(not (left - AUTHOR_INPUT), f'no unfilled data placeholders {sorted(left - AUTHOR_INPUT)}')

print('2. claims')
# abstract / F5
check(f('P6:TON_IoT_s1_mm') < f('P6:TON_IoT_majority_f1'), 'min-max stream-1 CNN below majority-class predictor')
check(f('P6:TON_IoT_s1_q') > f('P6:TON_IoT_s2_f1'), 'with quantile inputs the volume stream beats stream 2 ("ranking reversed")')
check(f('P7:p6v2_s1_best_epoch_max') <= 4, 'fusion failure peaked within the first few epochs')
check(f('P7:p3v2_joint_epochs_min') > 4 and f('P7:p1_minmax_best_epoch_min_ton') > 4,
      'early-peak rule would not catch P3 / P1 cases')
check(abs(f('P7:p1v1_rf_cnn_gap')) > 0.05 and abs(f('P1:TON_IoT_rf_v1_f1') - f('P1:TON_IoT_cnn_w1_f1')) < 0.015,
      'P1 RF-CNN gap 0.076 shrank to about 0.01')
check(f('P3:TON_fact_map_effect_v2budget') > f('P3:TON_fact_budget_effect_minmax') > f('P3:TON_fact_budget_effect_asinh'),
      'map effect > budget effect (min-max) > budget effect (asinh)')
# F6
check(f('P7:p6v3_capped_diff') <= f('P7:p6v3_capped'), 'capped runs with different best epoch <= capped runs')
check(f('P6:TON_IoT_d_late_early_pholm') < 0.05, 'TON late-early survives (Holm p < 0.05)')
# F7
r00 = {s: f(f'P7:p3v1_r00_{s}') for s in ('naive', 'ewc', 'replay', 'lwf', 'gem')}
check(min(r00, key=r00.get) == 'replay' and r00['ewc'] < r00['naive'], 'v1 replay Task-0 worst, EWC below naive')
check(f('P4:TON_bnfr_a05_swing') > 3 * f('P4:TON_fedavg_a05_swing') and f('P4:TON_bnfr_a05_fr_swing') < 2 * f('P4:TON_fedavg_a05_swing'),
      'BN swing >> GN; re-estimation brings it near GN')
check(V['P3:TON_task0_identical'] == V['P3:UNSW_task0_identical'] == V['P3:NB_task0_identical'] == 'yes',
      'corrected Task-0 identical on all three datasets')
# F8
check(abs(f('P3:TON_er2000fb_d')) < 0.02, 'full-budget ER-2000 change small')
# F9
check(f('P7:p4v1_dp_later') < 0.01 < f('P7:p4v1_dp_first'), 'v1 DP frozen after first checkpoint')
check(f('P4:TON_dp_gap_full') > 10, 'DP cost large but not a collapse')
check(f('P4:TON_dp_full_sigma18_ci') > 0.5, 'DP model trains (macro-F1 > 0.5)')
# F10
check(f('P4:RERUN_MAXDIFF') > f('P6:rerun_dF_max') > f('P7:p1_rerun_float_max'), 'rerun spreads ordered as reported')
check(f('P7:p1_rerun_int8_max') > f('P7:p1_rerun_float_max'), 'int8 rerun difference larger than float')
# F11
check(f('P7:p4v1_iid_best42') >= f('P7:p4v1_iid_best_mean'), 'IID: seed 42 best checkpoint at/above mean (best of three)')
check(f('P7:p4v1_a01_best42') < f('P7:p4v1_a01_best_mean'), 'alpha 0.1: seed 42 not the best')
for t in ('iid', 'a05', 'a01'):
    check(f(f'P7:p4v1_{t}_final_mean') < f(f'P7:p4v1_{t}_best_mean'), f'{t}: final-round mean below best-checkpoint mean')
# F12
check(V['P3:TON_ewc_edge'] == 'no' and V['P3:TON_lwf_edge'] == 'no', 'retuned EWC/LwF interior')
check(f('P3:TON_lwf_tune_4') > f('P3:TON_lwf_tune_0p5'), 'new LwF pick scores higher under new criterion')
# F13
check(f('P4:TON_d_fedavg_natr_vs_central') < f('P4:TON_d_fedavg_a05_vs_central'), 'host partition costs more than alpha 0.5')
check(f('P4:TON_skew_natr_top2_mean') <= f('P4:TON_skew_a05_top2_mean') + 5, 'host skew similar to alpha 0.5')
check(f('P4:TON_fedavg_natr_sel') < f('P4:TON_fedavg_nat_sel'), 'redrawn partitions lower FedAvg')
check(f('P4:TON_own_pairs_neg') > f('P4:TON_own_pairs_n') / 2 and V['P4:NB_dev_neg'] == V['P4:NB_dev_n'], 'own-traffic reversal')
check(f('P4:TON_d_fedavg_natr_vs_local_natr') > 0, 'federation wins on common test set')
# F14 / F4
check(f('P2:TON_IoT_cnn_none_strongest_b0.5') < f('P2:TON_IoT_cnn_none_wb_unconstrained_e0.05'), 'TON: constraints lower ASR vs weak eps-PGD')
check(f('P2:UNSW_NB15_cnn_none_strongest_b0.5') > f('P2:UNSW_NB15_cnn_none_wb_unconstrained_e0.05'), 'UNSW: constraints raise ASR vs weak eps-PGD')
check(float(V['P2:wbs_minus_bb_range'].split('–')[0]) > 0, 'strong white-box action PGD below the grid search for both networks, both datasets')
pc = lambda k: float(V[k].rstrip('%'))  # noqa: E731
check(pc('P2:TON_IoT_gb_augment_strongest_b0.5') < pc('P2:TON_IoT_gb_none_strongest_b0.5') / 5
      and min(pc(f'P2:UNSW_NB15_{m}_augment_strongest_b0.5') for m in ('rf', 'gb')) > 25,
      'P2: tree augmentation works on TON_IoT GBDT, unseen TTLs still evade UNSW trees')
_DEF = ('cnn_clean_loop', 'cnn_at_all', 'cnn_ccat', 'cnn_at_threat', 'cnn_trades')
check(all(f(f'P2:{d}_{m}_fpr') > f(f'P2:{d}_cnn_none_fpr') for d in ('TON_IoT', 'UNSW_NB15') for m in _DEF),
      'P2: every network defence raises the benign FPR, both datasets')
check(all(f(f'P2:{d}_cnn_ccat_rare_f1') < f(f'P2:{d}_cnn_none_rare_f1') for d in ('TON_IoT', 'UNSW_NB15')),
      'P2: CCAT lowers rare-class F1 on both datasets')
check(pc('P2:TON_IoT_cnn_at_all_wb_unconstrained_e0.05') + 20 < pc('P2:TON_IoT_cnn_at_threat_wb_unconstrained_e0.05')
      and pc('P2:TON_IoT_cnn_at_all_wb_bpda_e0.05') > 80, 'P2: AT-all looks robust to eps-PGD until BPDA')
check(pc('P2:TON_IoT_cnn_at_threat_strongest_volume_b0.5') < pc('P2:TON_IoT_cnn_clean_loop_strongest_volume_b0.5')
      < pc('P2:TON_IoT_cnn_none_strongest_volume_b0.5')
      and all(pc(f'P2:TON_IoT_{m}_strongest_volume_b0.5') <= pc('P2:TON_IoT_cnn_clean_loop_strongest_volume_b0.5') for m in _DEF),
      'P2: own operating point, defences span at_threat .. clean loop, all below undefended')
check(float(V['P2:TON_IoT_cnn_at_threat_prim_p']) >= 0.05 and pc('P2:TON_IoT_cnn_trades_op_matched_e2e') > pc('P2:TON_IoT_cnn_none_op_thr_e2e'),
      'P2: matched FPR - AT-threat n.s. vs undefended, TRADES worse (TON_IoT)')
check(f('P2:UNSW_NB15_gb_none_bb_ttl_only') < f('P2:UNSW_NB15_cnn_none_bb_ttl_only'), 'TTL-only: GBDT below CNN')
check(f('P5:unsw_sttl31_attack_pct') == 0, 'sttl 31 has no attacks')
check(f('P5:UNSW_sttl_rule_bin') > f('P5:UNSW_majority_test'), 'TTL rule beats majority share')
check(-0.02 <= f('P1:TON_noports_rf_v1') <= -0.01, 'ports cost about 0.015')
# F1 / F2
check(V['P7:p5v1_top_rf'] == V['P7:p5v1_top_cnn'] == 'id', 'v1 SHAP top feature = id for RF and CNN')
check(f('P5:decf_removed_acc_effect_mean') > 0.03, 'id still affects accuracy without duplicates (metric-dependent)')
check(f('P5:UNSW_rf_f1_nocopy') > f('P5:UNSW_rf_f1'), 'RF higher on clean subset (copies did not inflate)')
check(f('P6:UNSW_NB15_d_rf_late_clean') > f('P6:UNSW_NB15_d_rf_late') > 0, 'RF margin over late fusion grows on clean subset')
check(f('P5:UNSW_fcopy_feat_pct') > f('P5:UNSW_fcopy_pct'), 'features-only copies > copies with label')
# F3
check(f('P3:oracle_max_abs') < 0.05, 'oracle preprocessing changes small')
# F16
check(f('P5:TON_hard_att_control_copy_ev') - f('P5:TON_hard_att_counterfactual_ev') <= 5, 'copy control close to counterfactual hardening')
check(all(V[f'P5:{d}_tr_union_copy_cf_better'] is True or V[f'P5:{d}_tr_union_copy_cf_better'] == 'True' for d in ('TON', 'UNSW'))
      and all(V[f'P5:{d}_tr_union_copy_nsig_cf'] in (3, '3') for d in ('TON', 'UNSW')), 'union attacker: counterfactuals beat copy control in every seed, both datasets')
check(f('P5:TON_hard_all_original_ev') == f('P5:TON_hard_all_counterfactual_ev') == 100, 'all-field hardening 100 -> 100')
check(f('P5:TON_fcopy_8dp_pct') > 10 * f('P5:TON_fcopy_pct'), 'rounding inflated copies')
check(f('P5:TON_cnn_sb_fixed_bg_ns1000') > f('P5:TON_cnn_sb_fixed_bg_ns200'), 'more SHAP samples -> more stable')
# F17
check(f('P7:p1v1_size_ratio_unsw') > 1.5 * f('P7:p1v1_size_ratio_ton'), 'v1 size ratio not constant across datasets')
check(f('P1:TON_IoT_lat_best_tree_ms') > 10 * f('P1:TON_IoT_lat_best_net_ms'), 'best trees much slower per record')
check(f('P7:p1v1_cnn_kb_ton') > 3 * f('P1:TON_IoT_cnn_w1_kb_w'), 'Keras file >> weights')
# F18 / Table 1 / Section 5.1
check(f('P6:TON_IoT_d_rf_late') > 0 and f('P6:UNSW_NB15_d_rf_late') > 0, 'RF beats late fusion on both')
check(abs(f('P6:TON_IoT_semrand_late')) <= 0.01 and abs(f('P6:UNSW_NB15_semrand_late')) <= 0.01, 'semantic vs random at most about 0.01')
check(f('P1:TON_IoT_tree_minus_net') > 0 and V['P1:TON_IoT_budget_winner'] in ('tie', 'network', 'tree')
      and float(V['P1:TON_IoT_tree_over_net_sig_welch_p'].lstrip('< ')) < 0.05, 'P1 budget clause computed; tree significantly better from the stated size')
check(f('P7:p1v1_rf_f1') > f('P7:p7v1_headline') and f('P7:p1v1_cnn_f1') > f('P7:p7v1_headline'), 'P1 RF and CNN beat the v1 framework')
check(f('P3:TON_ewc_vs_naive_d') > -0.02 and f('P3:TON_ewc_vs_naive_d') < 0.02, 'EWC ~ fine-tuning on TON')
check(f('P4:TON_d_fedavg_iid_vs_central') < 0, 'IID FedAvg below central')
check(f('P2:TON_IoT_cnn_ccat_dclean') < 0, 'CCAT costs clean F1')
check(f('P2:TON_IoT_cnn_ccat_rare_f1') < f('P2:TON_IoT_cnn_none_rare_f1'), 'CCAT lowers rare-class F1')
# F19
check(f('P1:TON_IoT_gb05_min') > f('P1:TON_IoT_gb01_min'), 'lr 0.05 worst seed better than lr 0.1 worst seed')
# F20
seeds = [f(f'P7:p7v1_seed{s}') for s in (42, 123, 456)]
check(all(abs(f('P7:p7v1_headline') - x) > 0.002 for x in seeds), 'v1 P7 headline not among its seeds')
check(abs(f('P7:p7v1_sd_pop') - 0.2559) < 0.001, 'printed SD = population SD')
check(f('P2:TON_IoT_cnn_none_strongest_volume_b0.5_sdpp') > 1, 'ASR SD over model seeds >> 0.002 of the original restarts')
# F22
check(abs(f('P7:p2v1_ccat_f1_main') - 0.531) < 0.001 and abs(f('P7:p2v1_ccat_f1_sens') - 0.579) < 0.001, 'v1 CCAT 0.531 vs 0.579')

# round-1 revision claims
check(f('P2:TON_IoT_cnn_at_threat_fpr') > 2 * f('P2:TON_IoT_cnn_none_fpr'), 'F15: threat-matched defence raised benign FPR (text: from 0.015 to 0.044)')
check(f('P3:TON_er2000_fpr_final') > 3 * f('P3:TON_joint_fpr_final'), 'F15: replay FPR far above joint')
check(f('P4:TON_d_fedavg_natr_vs_central_pholm') > 0.05, 'F10: host-partition cost not significant after Holm')
check(abs(f('P6:N_BaIoT_d_rf_late')) < 0.001, 'F16: RF ties late fusion on N-BaIoT')
check(f('P5:UNSW_hard_att_counterfactual_ev') < f('P5:UNSW_hard_att_control_copy_ev'), 'F13: on UNSW counterfactuals beat copy control')
check(0.6 < f('P6:TON_IoT_d_rf_rf_best1') / f('P6:TON_IoT_d_early_best1') < 1, 'F16: forest gained "nearly as much"')
check(f('P3:TON_joint_oracle_p') < 0.05 and f('P3:oracle_max_abs') < 0.05, 'Sec 4.6: nominally significant but small')
check(f('P7:p3v2_joint_epochs_min') - 5 <= 4, 'F4: some P3 runs would be flagged by an early-peak rule (epochs - patience <= 4)')
check(int(V['P7:p1v1_cnn_kb_ton']) == 349, 'F14: Keras file 349 KB (KiB, same unit as weights)')
check(f('P2:TON_IoT_cnn_at_threat_fpr') > f('P2:TON_IoT_cnn_none_fpr') and f('P2:TON_IoT_cnn_at_threat_dclean') > -0.05,
      'F15: small macro-F1 cost, large FPR rise')
# external audit wording ("none deduplicated, all kept TTL, none had a tuned tree baseline")
def kofn(k):
    a, b = re.findall(r'\d+', V[k])[:2]
    return int(a), int(b)
a, b = kofn('P7:ext_F2'); check(a == b, f'ext: no repo deduplicated ({V["P7:ext_F2"]} without)')
a, b = kofn('P7:ext_F3'); check(a == b, f'ext: all determinable kept TTL ({V["P7:ext_F3"]})')
check(int(V['P7:ext_F16_untuned']) + int(V['P7:ext_F16_notree']) + int(V['P7:ext_F16_nobase']) == int(V['P7:ext_n']),
      'ext: F16 categories cover all nine (so none has a tuned tree ensemble)')
a, b = kofn('P7:ext_F1'); check(a > 0 and a < b / 2, 'ext: identifiers "less common" (minority, nonzero)')
check(int(V['P7:ext_n']) == 9 and 'stopping at nine' in TXT, 'ext: nine papers as declared')
check(int(V['P7:ext_ttl_ablated']) == 0, 'ext: no repository ablated TTL')
for col in ('F4', 'F5', 'F6', 'F14'):
    a, b = kofn(f'P7:ext_{col}')
    check(b > 0, f'ext: {col} assessable in at least one repository ({V[f"P7:ext_{col}"]})')
found = sum(kofn(f'P7:ext_{c}')[0] > 0 for c in ('F4', 'F5', 'F6', 'F14'))
check(found == 4 and 'instances of two software faults and risk configurations for two more' in TXT
      and 'demonstrated instances of two software faults (an F5 analogue and F14) and risk configurations for two (F4 and F6)' in TXT,
      f'ext: all four software-fault columns non-empty ({found}); text: 2 demonstrated (F5 analogue, F14) + 2 risk (F4, F6)')
check(int(V['P7:ext_own_F2_n']) > 0 and int(V['P7:ext_own_n']) == 5 and 'five of the nine the account name' in TXT,
      'ext: confirmed author releases (5) with assessable F2')
check(f"re-checked {V['P7:ext_hand_checked']} findings" in TXT and 'coded by one agent only' in TXT,
      'ext: re-check count stated, single coding of software-fault columns disclosed')
check('external-audit notes' not in TXT and 'not released publicly' in TXT.split('**Data and code availability.**')[1],
      'data statement: audit not public')
check(int(V['P7:ext_F4_bn']) <= kofn('P7:ext_F4')[0], 'ext: F4 with BN subset')
a, b = re.findall(r'\d+', V['P7:ext_agree_mapped'])[:2]
check(int(a) / int(b) > 0.9, f'ext: recode agreement {V["P7:ext_agree_mapped"]}')
check(int(V['P7:rev_fixedany']) + int(V['P7:rev_rejected']) + int(V['P7:rev_unknown']) <= int(V['P7:rev_raised']),
      'review table: outcome counts do not exceed raised')
check('six routes' in TXT and TXT.count('**Reference models**') == 1, 'Sec 5.1: six routes listed')
check('Study 4 five' in TXT and ' 279 ' not in TXT.split('## References')[0], 'review rounds and reference count corrected')
# all flagged rows of Table 2 marked New = the four named in the abstract/introduction
new_rows = [r for r in re.findall(r'^\| (F\d+) \|([^\n]*)$', TXT, re.M) if '**New**' in r[1]]
new_rows = [r[0] for r in new_rows]
f14 = [r for r in re.findall(r'^\| (F\d+) \|([^\n]*)$', TXT, re.M) if r[0] == 'F14'][0][1]
check(new_rows == ['F4', 'F5', 'F6'] and 'partly new' in f14 and 'Three are software faults' in TXT and 'only partly covered' in TXT
      and 'Three software faults do not, and a fourth is only partly covered' in TXT
      and 'F4, F5 and F6 are new, and F14 is partly covered' in TXT,
      f'three new modes + one partly covered (F14), same in abstract, introduction and Section 4: {new_rows}')
# Table 3: each study named as changed in the attribution sentence has a "yes: Study k" row
for k in (1, 2, 4, 6):
    check(re.search(rf'yes: (?:Study|Studies) [^|]*\b{k}\b', TXT) is not None, f'Table 3 has a mode that changed Study {k} on its own')
check(re.search(r'yes: (?:Study|Studies) [^|]*\b5\b', TXT) is None and "Study 5's change cannot be traced to a single mode" in TXT
      and 'four of the five changed conclusions' in TXT, 'Study 5: no single-mode attribution, stated')
check('| Review | A reviewer with access to the code compares' in TXT and '| F2, F5, F7, F12, F13, F14 |' in TXT,
      'Table 5 has the code-versus-methods row with the six modes of Sec 5.1')
check(f('P4:TON_d_fedavg_iid_vs_central_mean') > -0.05, 'Study 4 IID gap essentially confirmed (|gap| < 5 points)')
check(kofn('P7:ext_F4') == (8, 8) and V['P7:ext_F4_bn'] == '3', 'ext F4 recode (linear scaling = risk): 8 of 8, 3 with BN')

print('3. catalogue count')
rows = re.findall(r'^\| F(\d+) \|', TXT, re.M)
half = len(rows) // 2
check(rows[:half] == rows[half:], 'Tables 2 and 3 list the same modes in the same order')
rows = rows[:half]
paras = re.findall(r'^\*\*F(\d+)\.', TXT, re.M)
n_claim = int(re.search(r'document (\d+) failure modes', TXT).group(1))
check(len(rows) == len(paras) == n_claim and rows == [str(i) for i in range(1, n_claim + 1)] == paras,
      f'table rows {len(rows)}, paragraphs {len(paras)}, claimed {n_claim}')

print('4. citations')
main, refs = TXT.split('## References')
main = main.split('## Declarations')[0]
cited = set()
for grp in re.findall(r'\[([^\]]*\d{4}[^\]]*)\]', main):
    for c in grp.split(';'):
        m = re.match(r"\s*([A-Z][\w'’\-]+|van de Ven|D'hooge)(?: (?:et al\.|and [A-Z][\w\-]+))? (\d{4})", c.strip())
        if m:
            cited.add((m.group(1).split()[-1], m.group(2)))
for a, y in re.findall(r"([A-Z][\w'\-]+)(?: et al\.| and [A-Z][\w\-]+)? \[(\d{4})\]", main):
    cited.add((a, y))
listed = set()
for r in refs.strip().splitlines():
    if r.strip():
        m = re.match(r'(.+?)\. ((?:19|20)\d\d)\. ', r)
        first = m.group(1).split(',')[0].split(' and ')[0].strip()
        listed.add((first.split()[-1], m.group(2)))
cited = {('Ven' if a == 'van de Ven' else a, y) for a, y in cited}
check(not (cited - listed), f'cited but not listed: {sorted(cited - listed)}')
check(not (listed - cited), f'listed but not cited: {sorted(listed - cited)}')

print(f'\n{len(fails)} failure(s), {len(pending)} pending')
sys.exit(len(fails))
