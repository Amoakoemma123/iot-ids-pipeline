"""Check Paper 3 (v2 manuscript, results_v3 + results_p3final): no unfilled placeholders, anonymity, table/citation
coverage, and every qualitative or hard-coded claim in the text against the run files (via values.json where
compute_values already derives the number from the run files, directly from the JSONs otherwise).
Exit code = number of failures. Checks whose runs are not yet available are reported as PEND, not failures:
the final review round (2026-11-07) leaves ⟦P3F_...⟧ placeholders for the corrected A-GEM, the replay-weighting
ablation, the random forest on the replay memory and the range-share diagnostic (queue logs/p3final_queue.sh).
Run compute_values.py, make_figures.py and build.py (both variants) first."""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / 'experiments' / 'paper3_continual'   # repository root (authors' tree: <project>/Papers pipeline)
sys.path.insert(0, str(EXP.parents[1]))
sys.path.insert(0, str(HERE))
from experiments.paper3_continual.analyze_cl_v2 import metrics  # noqa: E402

RES = EXP / 'results_v3'
import os  # noqa: E402
RESF = Path(os.environ.get('P3F_RESULTS', EXP / 'results_p3final'))   # overrides only for plumbing tests
V = json.loads(Path(os.environ.get('P3_VALUES_OUT', HERE / 'values.json')).read_text(encoding='utf-8'))
DSETS = {'TON': 'TON_IoT', 'UNSW': 'UNSW_NB15', 'NB': 'N_BaIoT'}
AG_DONE = V.get('P3F_agem_done') == 'yes'
STRATS = ('naive', 'ewc', 'lwf', 'agem', 'er500', 'er2000', 'joint') if AG_DONE else ('naive', 'ewc', 'lwf', 'er500', 'er2000', 'joint')
NONREPLAY = ('naive', 'ewc', 'lwf', 'agem') if AG_DONE else ('naive', 'ewc', 'lwf')
fails, pending = [], []
PENDING_KEYS = set()


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def pend(msg):
    print('  PEND ' + msg)
    pending.append(msg)


def f(key):
    """values.json entry as float (handles +, − and %)."""
    return float(V[key].replace('−', '-').replace('+', '').replace('%', '').replace(',', ''))


def pv(key):
    """p-value entry as float ('< 0.001' -> 0.0005)."""
    s = V[key].strip()
    return 0.0005 if s.startswith('<') else float(s)


def no_unfilled(body):
    """⟦P3F_*⟧ placeholders wait for the final-round runs; reported as pending, not failed."""
    keys = re.findall(r'⟦([^⟧]+)⟧', body)
    PENDING_KEYS.update(k for k in keys if k.startswith('P3F_'))
    return '{{' not in body and all(k.startswith('P3F_') for k in keys)


def load(ds, group, res=RES):
    out = {}
    for p in sorted((res / ds / group).glob('*.json')):
        r = json.loads(p.read_text())
        out.setdefault(r['strategy'], {})[(r['order'], r['seed'])] = (r, metrics(r))
    return out


G = {d: load(ds, 'main') for d, ds in DSETS.items()}
for d in G:   # the flawed results_v3 A-GEM is never checked against the text; the corrected one replaces it
    G[d].pop('agem', None)
    if AG_DONE:
        G[d]['agem'] = load(DSETS[d], 'main', RESF)['agem2']
src = (HERE / 'Manuscript.md').read_text(encoding='utf-8')
text = (HERE / 'Manuscript_filled.md').read_text(encoding='utf-8')
body = text[:text.index('## References')]
refs = text[text.index('## References'):]
abstract = text.split('## Abstract')[1].split('**Keywords')[0]

# ---------------------------------------------------------------------------------------------------------------
print('Placeholders, markers and anonymity')
from docx import Document  # noqa: E402
for name in ('Manuscript_v2.docx', 'Manuscript_v2_anon.docx', 'Highlights.docx'):
    p = HERE / name
    if not p.exists():
        check(False, f'{name} exists'); continue
    d = Document(p)
    allt = '\n'.join(x.text for x in d.paragraphs) + '\n'.join(c.text for t in d.tables for r in t.rows for c in r.cells)
    check(no_unfilled(allt), f'{name}: no unfilled placeholders (other than pending P3F_*)')
    if 'anon' in name:
        check(not any(n in allt for n in ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku', 'amoakopuyol', 'Amoakoemma'))
              and not d.core_properties.author, f'{name}: anonymous (names, repository user, document author)')
check(no_unfilled(text), 'Manuscript_filled.md: no unfilled placeholders (other than pending P3F_*)')
check('⟦' not in src and 'PENDING' not in src and 'TODO' not in src, 'Manuscript.md: no to-do markers')
for n in range(1, 8):
    check(f'Table {n}' in body, f'Table {n} cited in the text')
check('Figure 1' in body, 'Figure 1 cited in the text')
check('Table 8' not in body and 'Figure 2' not in body, 'no citation of a table or figure that does not exist')
check((HERE / 'figures' / 'fig1_task0_retention.png').exists(), 'Figure 1 file exists (run make_figures.py)')
meta = json.loads((HERE / 'figures' / 'fig1_meta.json').read_text())
check(meta['fig1_agem_plotted'] == AG_DONE, 'Figure 1 plots A-GEM only in its corrected version (else omitted)')

print('References')
entries = [e for e in refs.split('\n') if re.match(r'^(?:[A-Z]|van )[^\n]*\(\d{4}[a-z]?\)\.', e)]
NODOI = ('USENIX', 'PMLR', 'Neural Information Processing', 'ICLR', 'arXiv')
check(all('https://doi.org/' in e or any(n in e for n in NODOI) for e in entries),
      f'every reference has a DOI, or is from a venue without DOIs ({len(entries)} entries)')
keys = []
for e in entries:
    first = re.split(r',|\(', e)[0].strip()
    year = re.search(r'\((\d{4}[a-z]?)\)', e).group(1)
    keys.append((first.split()[-1] if ' ' in first and first.split()[0] in ('van', 'de', 'De') else first, year))
for sur, yr in keys:
    sur_ = sur.replace('van de Ven', 'Ven')
    check(re.search(re.escape(sur_) + r'[^()]{0,60}?\(?' + yr, body) is not None, f'reference {sur} ({yr}) cited in the text')
for m in re.finditer(r'([A-Z][A-Za-z\-]+)(?: et al\.| & [A-Z][A-Za-z\-]+)?,? \(?(\d{4}[a-z]?)', body):
    sur, yr = m.group(1), m.group(2)
    if sur in ('Section', 'Table', 'Task', 'Figure') or not 1980 < int(yr[:4]) < 2027:
        continue
    check(any(k[1] == yr and (k[0] == sur or k[0].endswith(sur)) for k in keys), f'citation {sur} {yr} has a reference entry')
check('only the exemplar-based iCaRL approached joint retraining' in src and 'such as EWC, LwF and iCaRL largely fail' not in src,
      'R1: Rahman et al. (2022) described correctly (iCaRL was the exception)')
check('(Moustafa, 2021; see also Alsaedi et al., 2020)' in src, 'R2: Moustafa (2021) cited for the TON-IoT network flows')
check('implementation tricks and evaluation settings change the ranking' in src, 'R3: Mai et al. (2022) paraphrased as in the paper')

# ---------------------------------------------------------------------------------------------------------------
print('§3 data and tasks')
for d, n_att in (('TON', 9), ('UNSW', 9), ('NB', 8)):
    r = next(iter(G[d]['naive'].values()))[0]
    check(len([c for c in r['classes'] if c != 'normal']) == n_att, f'{d}: {n_att} attack classes in the run files')
    check(len(r['tasks']) == 3 and all('normal' in t for t in r['tasks']), f'{d}: three tasks, benign in every task')
check(f('TON_clip_share') < 0.01, 'TON: well under 0.01% of compressed test values beyond the clip window')
check((V['TON_n_features'], V['UNSW_n_features'], V['NB_n_features']) == ('13', '39', '115'),
      'feature counts as stated by the run files (13 / 39 / all 115)')
nb_info = json.loads((EXP.parents[1] / 'dataset4' / 'nbaiot_prepared.json').read_text())
check(nb_info['sample_frac'] == 0.03 and nb_info['n_conflict'] == 17 and nb_info['dropped_classes'] == {'gafgyt_tcp': 40, 'gafgyt_udp': 32},
      'N-BaIoT preparation: 3% sample, 17 conflicting vectors, floods dropped with 40 and 32 distinct vectors')
check('Later-task values outside the Task-0 range are clipped' not in src and 'clipped to [−1, 2], which corresponds to about −1 to 2,000' in src,
      'W2/C4: clipping described after the compression map, in compressed units')
check(abs(np.sinh(2 * np.arcsinh(1000.0)) / 1000 - 2000) < 1, 'clip limit 2 in compressed units is about 2,000 in min–max units')
check('N-BaIoT uses all' in src and 'Dataset-level cleaning uses the whole training split' in src,
      'C5/C6: N-BaIoT has no feature selection; dataset-level cleaning on the whole training split stated')

print('§4 protocol')
for d in DSETS:
    check(all(len(G[d][s]) == 5 for s in STRATS), f'{d}: five seeds for every strategy (semantic order)')
    check(V[f'{d}_task0_identical'] == 'yes', f'{d}: Task-0 predictions identical across strategies')
    tn = json.loads((RES / DSETS[d] / 'tuned.json').read_text())
    check(sorted(map(float, tn['ewc']['scores'])) == [10.0 ** k for k in range(8)], f'{d}: EWC grid 10^0–10^7 in factors of 10')
    check(sorted(map(float, tn['lwf']['scores'])) == [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32], f'{d}: LwF grid as listed')
    r = next(iter(G[d]['er2000'].values()))[0]
    check(r['stages'][-1]['buffer'] <= 2000, f'{d}: ER-2000 memory holds at most 2,000 records')
check(V['UNSW_er2000_mem_final'] == '1,889', 'C11: UNSW ER-2000 memory ends with 1,889 records')
check('counts only after five epochs' not in src, 'C8: unimplemented "counts only after five epochs" removed')
check('validation records of the current task' in src, 'C3: early stopping on current-task validation records stated')
check('γ = 1' in src and 'benign included' in src, 'C10/C14: EWC without decay (γ = 1); LwF distils benign too')
check('bit-identical' not in src and 'Task 0 is trained once per seed' in src, 'C12: Task-0 sharing described by construction')
check('seed equal to the order index' in src and 'training randomness only' in src and 'six comparisons' in src,
      'C13/M3: seeds, fixed split and Holm family size stated')
check('1:1' in src and 'drawn with replacement' in src and 'up-weighted twice' in src, 'C2/M2: replay mixing and double weighting described')
check('inference mode' in src and 'zero weight' in src, 'C1/C9: corrected A-GEM described (memory weights, inference mode)')
for d in DSETS:
    r = json.loads(next((RES / DSETS[d] / 'main').glob('naive_o0_s0.json')).read_text())
    check(f"{r['params']:,}" == V[f'{d}_params'], f'{d}: parameter count from the run files')

# ---------------------------------------------------------------------------------------------------------------
print('§5.1 overview')
for d in DSETS:
    check(f(f'{d}_joint_f1') < f(f'{d}_rf_f1'), f'{d}: joint retraining below the random forest')
    check(f(f'{d}_naive_Rlast_t0') < 0.005, f'{d}: fine-tuning recognises none of the Task-0 classes at the end')
check(f('UNSW_joint_f1') == min(f(f'{d}_joint_f1') for d in DSETS), 'UNSW has the lowest upper bound')
check('over benign traffic and the Task-0 attack classes' in src, 'A3: Task-0 macro-F1 includes benign')

print('§5.2 replay against regularisation')
if V.get('holm_provisional') == 'yes':
    pend('Holm-adjusted p of the confirmatory family is provisional (flawed A-GEM p stands in until the corrected runs exist)')
for d in DSETS:
    check(pv(f'{d}_er2000_vs_naive_pholm') < 0.05, f'{d}: ER-2000 > fine-tuning, Holm-significant')
    check(pv(f'{d}_ewc_vs_naive_pholm') >= 0.05, f'{d}: EWC not significant vs fine-tuning (as stated)')
    check(min(f(f'{d}_er500_FM'), f(f'{d}_er2000_FM')) < min(f(f'{d}_{s}_FM') for s in NONREPLAY),
          f'{d}: replay has the least forgetting (only family that substantially reduced it)')
    check(f(f'{d}_er2000_vs_er500') > 0, f'{d}: 2,000-record memory better than 500')
    check(V[f'{d}_ewc_edge'] == 'no', f'{d}: EWC weight chosen inside the grid')
    t2 = f(f'{d}_joint_Rlast_t2') - f(f'{d}_er2000_Rlast_t2')
    t01 = max(f(f'{d}_joint_Rlast_t{j}') - f(f'{d}_er2000_Rlast_t{j}') for j in (0, 1))
    check(t2 > t01 and t01 < 0.03, f'{d}: ER-2000 attack-class shortfall on the newest task (Δt2 {t2:.3f}); earlier tasks at or near joint (≤ {t01:.3f})')
check(f('UNSW_ewc_vs_naive_d') > 0 and f('NB_ewc_vs_naive_d') > 0 and f('TON_ewc_vs_naive_d') <= 0,
      'EWC: small gains on UNSW and NB, none on TON')
check(all(f(f'{d}_ewc_vs_naive_d') < 0.2 * f(f'{d}_er2000_vs_naive_d') for d in DSETS), 'EWC gains far below replay')
check(f('TON_er2000_rec_normal') < f('TON_joint_rec_normal') - 0.05, 'O3: TON ER-2000 benign recall below joint (benign part of the gap)')
check(sorted(pv(f'{d}_er2000_vs_er500_p') < 0.05 for d in DSETS) == [False, False, True] and pv('UNSW_er2000_vs_er500_p') < 0.05,
      '500 -> 2,000: clear only on UNSW')
check(f('UNSW_er2000_Rlast_t1') > f('UNSW_joint_Rlast_t1'), 'UNSW: ER-2000 above joint on Task 1')
for d in ('TON', 'UNSW'):
    check(pv(f'{d}_lwf_vs_naive_pholm') < 0.05 and f(f'{d}_lwf_vs_naive_d') > 0, f'{d}: LwF significant gain')
    check(f(f'{d}_lwf_gap_closed') < 25, f'{d}: LwF gain small (closes < 25% of the gap)')
    check(V[f'{d}_lwf_alpha'] == V[f'{d}_lwf_alpha_pre_ext'], f'{d}: extended grid left the LwF weight unchanged')
check(f('NB_lwf_vs_naive_d') <= 0 or pv('NB_lwf_vs_naive_pholm') >= 0.05, 'NB: LwF no improvement')
nbt = json.loads((RES / 'N_BaIoT' / 'tuned.json').read_text())['lwf']
check(V['NB_lwf_alpha'] == V['NB_lwf_alpha_pre_ext'] == '0.5', 'NB: extended grid left the LwF weight at 0.5 (smallest of the original grid)')
check(nbt['value'] > min(map(float, nbt['scores'])) and V['NB_lwf_edge'] == 'no', 'NB: LwF weight now inside the extended grid')
check(max(f('NB_lwf_tune_0p1'), f('NB_lwf_tune_0p25')) < f('NB_lwf_tune_0p5'), 'NB: lower LwF weights scored worse on the dev seed')
check('edge of the grid' not in src, '§6.2 grid-edge limitation removed')
check(f('UNSW_er2000_rec_backdoor') < f('UNSW_joint_rec_backdoor'), 'UNSW: ER-2000 backdoor recall below joint')
check(max(f('NB_er2000_rec_mirai_udp'), f('NB_er2000_rec_mirai_udpplain')) <
      min(f('NB_joint_rec_mirai_udp'), f('NB_joint_rec_mirai_udpplain')), 'NB: ER-2000 confuses the two Mirai UDP floods')
kb = [f(f'{d}_buffer_kb_2000') for d in DSETS]
check(min(kb) == f('TON_buffer_kb_2000') and max(kb) == f('NB_buffer_kb_2000'), f'memory size range TON–NB ({min(kb):.0f}–{max(kb):.0f} KiB) — §6.1')
check(' kB' not in src and 'KiB' in src, 'C11: memory sizes in KiB')
st = {d: f(f'{d}_store_all_ratio') for d in DSETS}
check(min(st, key=st.get) == 'UNSW' and max(st, key=st.get) == 'TON', 'storage ratio range quoted as UNSW–TON')
gc = sorted(round(f(f'{d}_er2000_gap_closed')) for d in DSETS)
check((gc[0], gc[-1]) == (81, 91), f'ER-2000 closes 81–91% of the gap (got {gc[0]}–{gc[-1]}) — §6.1, §7')
check(min(f(f'{d}_er2000_gap_closed') for d in DSETS) == f('NB_er2000_gap_closed') and
      max(f(f'{d}_er2000_gap_closed') for d in DSETS) == f('TON_er2000_gap_closed'), 'abstract range NB–TON is min–max')
allgc = {**{('s', d): f(f'{d}_er2000_gap_closed') for d in DSETS}, **{('o', d): f(f'{d}_er2000_orders_gap_closed') for d in DSETS}}
check(min(allgc.values()) == f('UNSW_er2000_orders_gap_closed') and max(allgc.values()) == f('TON_er2000_gap_closed'),
      'O1: highlight range UNSW (random orders) – TON covers both semantic and random orders')
check('no better than' not in src, 'O2: "no better than fine-tuning" removed')
check('and the results for malware classification (Rahman' not in src and 'plain experience replay also performed poorly' in src,
      'R1 (§5.2): agreement with Rahman et al. limited to regularisation')

if AG_DONE:
    pos = [d for d in DSETS if V[f'{d}_agem_vs_naive_sig'] == 'yes' and f(f'{d}_agem_vs_naive_d') > 0]
    neg = [d for d in DSETS if V[f'{d}_agem_vs_naive_sig'] == 'yes' and f(f'{d}_agem_vs_naive_d') < 0]
    vs = V['P3F_agem_verdict_short']
    check(('did not significantly' in vs) == (not pos and not neg), f'A-GEM verdict matches the Holm tests ("{vs}")')
    check(len(G['TON']['agem']) == 5, 'corrected A-GEM: five seeds')
    # generated phrases re-read by hand 2026-11-09 (abstract, §5.2, §5.3, §5.5, §6.1, conclusion); now guarded by checks
    check(all(f(f'{d}_agem_f1') >= f(f'{d}_naive_f1') for d in pos), 'A-GEM above fine-tuning where reported significant')
    check(all(f(f'{d}_agem_f1') < f(f'{d}_er2000_f1') for d in DSETS), '§6.1 "replaying them was the better use": A-GEM below ER-2000 on all three')
else:
    pend('corrected A-GEM runs (results_p3final, group agemfix) not complete: A-GEM rows, §5.2/§5.3 A-GEM text, abstract/conclusion verdict')

print('§5.2 final-round checks (replay weighting, random forest on the memory)')
if V.get('P3F_replayabl_done') == 'yes':
    for d in DSETS:
        check(f'P3F_{d}_er2000p_Rlast_t2' in V, f'{d}: ER-2000-pool values present')
    # conclusion: "shrank by more than half on all three", "part of the gap moved to the earlier tasks"
    check(all(f(f'P3F_{d}_er2000p_shortfall_t2') < 0.5 * f(f'{d}_er2000_shortfall_t2') for d in DSETS),
          'conclusion: newest-task shortfall more than halved on all three datasets')
    check(all(f(f'P3F_{d}_er2000p_old_cls') < f(f'{d}_er2000_old_cls') for d in DSETS),
          'conclusion: earlier-task retention lower with ER-2000-pool on all three (gap moved)')
    # §5.3 / abstract / highlight: pooled FPR "about" the joint level and significantly below ER-2000
    check(abs(f('P3F_TON_er2000p_fpr_final') - f('TON_joint_fpr_final')) <= 0.01, '§5.3: TON pooled FPR about the joint level')
    check(f('P3F_TON_er2000p_fpr_d') < 0 and V['P3F_TON_er2000p_fpr_p'].startswith(('0.00', '<')),
          '§5.3: TON pooled FPR below ER-2000 (p < 0.01)')
    check((f('TON_er2000_fpr_final') - f('P3F_TON_er2000p_fpr_final')) > (f('TON_er2000_fpr_final') - f('TON_er2000b_fpr_final')),
          '§5.3 "most of the cost from double weighting": pooling removes more than the half-benign memory')
else:
    pend('replay-weighting ablation (group replayabl) not complete: §5.2 "weighting effect" paragraph, §5.3 ER-2000-pool FPR')
if V.get('P3F_rfmem_done') == 'yes':
    for d in DSETS:
        check(V[f'P3F_{d}_rfjoint_last_equals_rf'] == 'yes', f'{d}: all-data forest after the last task equals the RF reference')
    # pre-submission review: RF + memory against BOTH CNN replay variants
    check(V['P3F_rfmem_hi_er2000'] == 'TON-IoT, UNSW-NB15 and N-BaIoT', 'RF + memory above ER-2000 on all three (p < 0.05)')
    check(V['P3F_rfmem_hi_pool'] == 'TON-IoT and N-BaIoT' and V['P3F_rfmem_lo_pool'] == 'UNSW-NB15',
          'highlight "beat CNN replay on two of three datasets": above ER-2000-pool on TON/NB, below on UNSW (p < 0.05)')
    check(V['P3F_rfmem_above_joint'] == 'TON-IoT and N-BaIoT', '§5.2/§6.1: RF + memory above CNN joint retraining on TON/NB only')
    check(f('P3F_UNSW_rfmem_old_det') < 0.8 < min(f('UNSW_er2000_old_det'), f('P3F_UNSW_er2000p_old_det')),
          '§5.2: on UNSW the forest misses earlier attacks that CNN replay flags')
    check('not' not in V['P3F_rfmem_verdict'].split('but')[0], 'RF verdict: no hedge in the ER-2000 part')
    for d in DSETS:
        check(f(f'P3F_{d}_rfmem_fpr_final') < f(f'{d}_er2000_fpr_final'), f'{d}: §6.1 RF + memory FPR below ER-2000')
    # pooled replay: "rose clearly only on UNSW-NB15", "marginal on TON-IoT"
    pv = lambda k: 0.0 if V[k].startswith('<') else float(V[k])
    check(pv('P3F_UNSW_er2000p_p') < 0.01 and 0.01 <= pv('P3F_TON_er2000p_p') < 0.05 and pv('P3F_NB_er2000p_p') > 0.05,
          'ER-2000-pool macro-F1: clear only on UNSW, marginal on TON, none on NB')
else:
    pend('random forest on the replay memory (group rfmem) not complete: abstract, §5.2 tree-model paragraph, §6.1 advice')
if V.get('P3F_size_done') == 'yes':
    for d in DSETS:
        check(f(f'P3F_{d}_sz_cnn_kib') < 100, f'{d}: network under 100 KiB')
        check(float(V[f'P3F_{d}_sz_rf_er2000_kib'].replace(',', '')) > 300, f'{d}: "few hundred kilobytes": RF + memory > 300 KiB')
    check(float(V['P3F_sz_ratio_min'].replace(',', '')) >= 3, '§6.1 "forests are far larger": at least 3x the network')
else:
    pend('size benchmark (bench_p3_size.py, logs/p3size_gate.sh) not run: §6.1 RF size / prediction-time sentence')

print('§5.3 class vs alert retention')
for d in DSETS:
    check(f(f'{d}_naive_old_det') > 0.5 and f(f'{d}_naive_old_det') > f(f'{d}_naive_old_cls') + 0.5,
          f'{d}: fine-tuning still flags most earlier-task attack records (excluding the newest task)')
    check(f(f'{d}_lwf_old_cls') < 0.2 < f(f'{d}_lwf_old_det'), f'{d}: LwF low class, high detection retention on earlier tasks')
    check(f(f'{d}_lwf_A') < min(f(f'{d}_er500_A'), f(f'{d}_er2000_A')), f'{d}: LwF class retention below replay')
od = {d: [f(f'{d}_naive_old_det')] for d in DSETS}
check(min(od, key=lambda d: od[d][0]) == 'UNSW' and f('TON_naive_old_det') >= f('NB_naive_old_det'),
      'abstract range UNSW–TON is min–max of fine-tuning earlier-task detection retention')
reps = {(d, s): f(f'{d}_{s}_old_det') for d in DSETS for s in ('er500', 'er2000')}
check(min(reps.values()) == f('TON_er500_old_det'), 'replay earlier-task detection minimum is TON ER-500 (the quoted floor)')
check(f('NB_fpr_all_max') <= 0.005, 'NB: benign FPR tiny at every stage (seed means)')
others = [s for s in STRATS if s != 'agem']
check(min(f(f'UNSW_{s}_fpr_t1') for s in others) > 0.15, 'UNSW: fine-tuning, EWC, LwF, replay and joint near 0.2 FPR after Task 1')
check(f('UNSW_naive_fpr_final') < 0.1 < f('UNSW_naive_fpr_t1'), 'UNSW: fine-tuning FPR falls back after Task 2')
check(f('UNSW_naive_Rlast_t1') < 0.05, 'UNSW: fine-tuning forgets the Task-1 classes (explains its FPR drop)')
check(min(f(f'UNSW_{s}_fpr_final') for s in ('ewc', 'lwf', 'er2000', 'joint')) > 0.15, 'UNSW: EWC/LwF/ER/joint stay near 0.2')
check(0.11 <= f('TON_er500_fpr_final') and f('TON_er2000_fpr_final') <= 0.14, 'TON: replay FPR about one in eight')
check(min(f('TON_er500_fpr_final'), f('TON_er2000_fpr_final')) > 3 * f('TON_joint_fpr_final'), 'TON: replay FPR well above joint')
check(min(f('TON_er500_fpr_final'), f('TON_er2000_fpr_final')) > 0.1, '§6.1: replay FPR on TON above 0.1')
if 'TON_er2000b_fpr_final' in V:
    check(pv('TON_er2000b_fpr_p') < 0.05 and f('TON_er2000b_fpr_d') < 0, 'TON: benign-share memory lowers FPR significantly (uncorrected)')
    check(f('TON_er2000b_fpr_final') > f('TON_joint_fpr_final'), 'TON: benign-share memory still above joint (excess remains)')
    check(0 < f('TON_er2000b_f1_d') < 0.03, 'TON: benign-share memory F1 rose slightly')
    check(abs(f('TON_er2000b_det_A') - f('TON_er2000_det_A')) < 0.005, 'TON: benign-share memory detection retention unchanged')
else:
    pend('benign-share memory runs not complete')

R0 = {(d, s): np.mean([m['R'][1][0] for (r, m) in G[d][s].values()]) for d in DSETS for s in ('naive', 'ewc', 'lwf')}
D0 = {(d, s): np.mean([m['Rd'][t][0] for (r, m) in G[d][s].values() for t in (1, 2)]) for d in DSETS for s in ('naive', 'lwf')}
check(max(R0.values()) < 0.05, f'Fig. 1: under fine-tuning, EWC and LwF, Task-0 class recall near zero after Task 1 (max {max(R0.values()):.3f})')
check('A-GEM delays the drop' not in src, 'A-GEM-specific Figure 1 sentence (flawed runs) removed')
check(min(D0.values()) > 0.75, f'Fig. 1: most Task-0 attack records still flagged (fine-tuning/LwF min {min(D0.values()):.2f})')
check(max(abs(np.mean([m['R'][t][0] for (r, m) in G[d]['ewc'].values()]) - np.mean([m['R'][t][0] for (r, m) in G[d]['naive'].values()]))
          for d in DSETS for t in range(3)) < 0.1, 'Fig. 1 caption: EWC follows fine-tuning closely (Task-0 class retention)')
det_min = min(np.min([m['Rd'][t][0] for (r, m) in G[d][s].values()]) for d in DSETS for s in ('naive', 'lwf', 'er2000', 'joint') +
              (('agem',) if AG_DONE else ()) for t in range(3))
check(float(meta['fig1_det_lo']) < det_min, f"Figure 1 detection axis ({meta['fig1_det_lo']}) starts below every plotted value ({det_min:.3f})")

print('§5.4 not-yet-learned attacks')
for d in DSETS:
    check(f(f'{d}_fpr_t0') < 0.05, f'{d}: FPR after Task 0 low, so not-yet-learned flagging is not flag-everything')
    check(f(f'{d}_zd_stage1_min') > f(f'{d}_zd_stage0_t2'), f'{d}: Task-2 flagging after Task 1 higher than after Task 0 for every strategy (W4)')
check(max(f('NB_zd_stage0_t1'), f('NB_zd_stage0_t2')) < 0.5, 'NB: BASHLITE-trained model misses more than half of Mirai')
check(f('NB_zd_stage1_min') > 0.95, 'NB: Mirai UDP flagged almost completely after Task 1')
check(f('NB_zd_stage0_t2') < f('NB_zd_stage0_t1'), 'NB range quoted as stage0_t2–stage0_t1 is ascending')
check('rises to' not in src and 'A-GEM gives the lowest rate' not in src, 'W4: no "rises"; flawed A-GEM lowest-rate sentence removed')
check('zero-day' not in src.lower() and 'Zero-day' not in V['table:oracle'], 'terminology: "not-yet-learned" throughout')

print('§5.5 task order and pre-processing scope')
for d in DSETS:
    for suf in ('_noagem', ''):
        w = V[f'{d}_orders_replay_wins{suf}']
        a, b = map(int, re.findall(r'\d+', w)[:2])
        check(a == b == int(V[f'{d}_n_orders']), f'{d}: replay beats {"fine-tuning, EWC and LwF" if suf else "every active non-replay strategy"} in every random order ({w})')
    check(f(f'{d}_orders_replay_margin_min_noagem') > 0, f'{d}: smallest replay margin positive')
if not AG_DONE:
    pend('A-GEM over the random orders (P3F_orders_agem_sentence, P3F_orders_nonreplay) waits for the corrected runs')
check(f('UNSW_naive_orders_f1') > f('UNSW_naive_f1'), 'UNSW: fine-tuning higher over random orders than semantic')
check(f('UNSW_er500_f1') < f('UNSW_er500_orders_f1'), 'UNSW: semantic order harder for ER-500 than the random orders')
check(f('UNSW_joint_orders_f1_min') <= f('UNSW_joint_f1') <= f('UNSW_joint_orders_f1_max'), 'UNSW: joint semantic within its random-order range')
check(f('UNSW_er2000_f1') > f('UNSW_er2000_orders_f1'), 'UNSW: ER-2000 higher on the semantic order than over random orders')
check(abs(f('UNSW_er500_orders_f1') - f('UNSW_er2000_orders_f1')) < 0.02, 'UNSW: ER-500 ≈ ER-2000 over random orders')
check('five seeds and five task orders' not in src, 'O6: seeds/orders wording')
zd = {d: [f(k) for k in V if k.startswith(d + '_') and k.endswith('_oracle_zd_d')] for d in DSETS}
check(f('oracle_max_abs') <= 0.03, 'oracle shortcut changed mean final macro-F1 by at most 0.03')
big = max((abs(f(k)), k) for k in V if k.endswith('_oracle_d'))[1]
check(big == 'NB_naive_oracle_d', 'largest oracle change is fine-tuning on N-BaIoT')
check(abs(f('NB_naive_oracle_d')) < f('NB_naive_f1_sd') and pv('NB_naive_oracle_p') >= 0.05, 'NB fine-tuning oracle change below seed SD, n.s.')
sig = sorted(k[:-2] for k in V if k.endswith('_oracle_p') and pv(k) < 0.05)
check(sig == ['TON_joint_oracle'] and f('TON_joint_oracle_d') < 0, f'only TON joint oracle change with uncorrected p < 0.05, a loss ({sig})')
check(pv('TON_joint_oracle_pholm3') < 0.05, 'A4: TON joint oracle loss survives Holm over the three TON tests')
check(np.mean(zd['TON']) < 0 and np.mean(zd['NB']) < 0 < np.mean(zd['UNSW']), 'oracle not-yet-learned flagging: down on TON/NB, up on UNSW')
up = max(float(b) for d in ('TON', 'UNSW') for s in ('naive', 'er2000', 'joint') for b in V[f'{d}_{s}_oracle_ci'].strip('[]').split(', '))
check(up <= 0.025 and all(f(k) <= 0.02 for k in V if k.endswith('_oracle_d')),
      f'O5: shortcut did not inflate TON/UNSW results by more than about 0.02 (largest upper CI bound {up:+.3f})')
check('inconclusive on N-BaIoT' in abstract, 'O5: abstract carries the N-BaIoT caveat')
nb = [float(b) for b in V['NB_naive_oracle_ci'].strip('[]').split(', ')]
check(nb[0] < 0 < nb[1] and nb[1] - nb[0] > 0.05, 'NB oracle CI wide and spans zero (inconclusive)')
check(f('oracle_ci_bound_TU') <= 0.03, 'TON/UNSW oracle CIs within ±0.03')
check('fall outside the Task-0 range' not in src, 'W1: 0.002% no longer described as the share outside the Task-0 range')
if V.get('P3F_rangeshare_done') == 'yes':
    check(f('P3F_TON_out01_te') >= f('TON_clip_share'), 'range share outside [0, 1] at least the share beyond the clip')
else:
    pend('range-share diagnostic (group rangeshare) not complete: §5.5 share outside [0, 1] and feature overlap')

print('review round 2026-11-02')
og = {d: f(f'{d}_er2000_orders_gap_closed') for d in DSETS}
check(min(og, key=og.get) == 'UNSW' and max(og, key=og.get) == 'TON', 'random-order gap range quoted as UNSW–TON')
check(og['UNSW'] < f('UNSW_er2000_gap_closed'), 'UNSW: less gap closed over random orders than semantic')
check(f('UNSW_joint_FM') > 0.1, 'UNSW: joint retraining shows class-level forgetting')
bwt = [abs(f(f'{d}_{s}_BWT') + f(f'{d}_{s}_FM')) < 0.003 for d in DSETS for s in STRATS]
check(sum(bwt) >= len(bwt) - 3, f'BWT = -FM for almost all rows ({sum(bwt)}/{len(bwt)}), so Table 2 shows plasticity instead')
for bad in ('bought with false alarms', 'raising false alarms', 'plasticity, not forgetting', 'companion stud',
            'runs TON-IoT on the CPU', 'is needed for families', 'not reported as benchmarks', 'in every task order'):
    check(bad not in src, f'reviewer-flagged phrase removed: "{bad}"')
if 'TON_er2000fb_f1' in V and 'UNSW_er2000fb_f1' in V and 'NB_er2000fb_f1' in V:
    check('{{TON_er2000fb_d}}' in src, 'budget check (ER-2000 with the full budget) reported in §5.2')
    check(all(f(f'{d}_er2000fb_Rlast_t2') > f(f'{d}_er2000_Rlast_t2') for d in DSETS), 'budget: newest-task retention rose on all three')
    check(all(f(f'{d}_er2000fb_old_cls') < f(f'{d}_er2000_old_cls') for d in ('UNSW', 'NB')), 'budget: earlier-task retention fell on UNSW and NB')
    check(f('TON_er2000fb_old_cls') >= f('TON_er2000_old_cls') and 'fell on UNSW-NB15' in src, 'budget: TON earlier tasks did not fall (text names UNSW/NB only)')
    check(all(float(V[f'{d}_er2000fb_p']) > 0.05 for d in DSETS), 'budget: no significant change in final macro-F1')
else:
    pend('budget check (ER-2000 with the joint training budget) not run yet')

print('§5.6 input scaling (first run of the protocol)')
check('earlier version' not in src and 'previous version' not in src and 'v2 factorial' not in src,
      'A1/M7: no ambiguous "earlier version"; the results_v2 runs are "a first run of the present protocol"')
check('silent' not in src.lower(), 'M7: "silent failure" framing removed')
check('batch-normalised CNN failed' not in src, 'M7: unmeasured batch-normalisation mechanism not claimed')
check(abs(f('TON_v2_naive_f1') - f('TON_naive_f1')) < 0.05, 'TON: fine-tuning hardly changed between runs')
check(f('TON_joint_f1') - f('TON_v2_joint_f1') > 0.1 and f('TON_er2000_f1') - f('TON_v2_er2000_f1') > 0.2,
      'TON: joint and replay rose much more')
check(f('UNSW_joint_f1') > f('UNSW_v2_joint_f1') and f('UNSW_er2000_f1') > f('UNSW_v2_er2000_f1'), 'UNSW: joint and ER-2000 rose')
check(abs(f('NB_joint_f1') - f('NB_v2_joint_f1')) < 0.01, 'NB: joint retraining hardly changed')
check(f('NB_v2_er2000_f1') > f('NB_er2000_f1') and 'On TON-IoT and UNSW-NB15, the badly scaled inputs lowered' in src,
      'O4: NB ER-2000 was higher in the first run; the "lowered replay" sentence is limited to TON/UNSW')
import build  # noqa: E402
check('at every stage' in build.TABLE_CAPTIONS['5.6'][0] and '100 and 50' not in build.TABLE_CAPTIONS['5.6'][0],
      'W3: Table 7 caption gives the joint-retraining budget (40/patience 5 vs 100/patience 15, every stage)')
check('over five seeds' in build.TABLE_CAPTIONS['5.1'][0] and 'standard deviation over all' not in build.TABLE_CAPTIONS['5.1'][0],
      'A2: Table 2 caption: SD over seeds, macro average over classes')
if 'TON_fact_minmax_v2' in V:
    check(f('TON_fact_map_effect_avg') > f('TON_fact_budget_effect_avg') > 0, 'O7: averaged over budgets, the map contributed more')
    check(abs(f('TON_fact_map_effect_v3budget') - f('TON_fact_budget_effect_minmax')) < 0.03, 'O7: under the present budget the two changes are of similar size')
    check('The input map was the larger factor' not in src, 'O7: unqualified "larger factor" removed')
    check(f('TON_fact_budget_effect_minmax') > f('TON_fact_budget_effect_asinh') > 0, 'TON: budget helps more with min–max inputs')
    check(max(f('TON_fact_asinh_v2'), f('TON_fact_minmax_v3')) < f('TON_fact_asinh_v3'), 'TON: neither change alone reaches the combined result')
    check(f('TON_fact_asinh_v3') < f('TON_rf_f1'), 'TON: combined result below the random forest')
    check(pv('TON_fact_vs_v2_p') >= 0.05, 'TON: reconstruction not significantly different from the first run')
    check(abs(f('TON_fact_asinh_v3') - f('TON_joint_f1')) < 1e-9, 'factorial combined cell is the main joint result')
else:
    pend('factorial runs not complete')

print('§6 discussion')
jt = [m['train_s'] for d in DSETS for (r, m) in G[d]['joint'].values()]
check(240 <= min(jt) and max(jt) <= 660, f'joint retraining sequence took about five to ten minutes ({min(jt):.0f}–{max(jt):.0f} s)')
for d in DSETS:
    check(max(f(f'{d}_{s}_fpr_final') for s in STRATS) < 0.25, f'{d}: no strategy flags a quarter of benign traffic')
check(f('NB_zd_stage1_max') > 0.95 and min(f('NB_zd_stage0_t1'), f('NB_zd_stage0_t2')) < 0.5,
      'NB: from under half to almost all of Mirai traffic')
check(all(23000 < int(V[f'{d}_params'].replace(',', '')) < 25000 for d in DSETS), 'about 24,000 parameters (under 100 KiB at 32 bits)')

print('Abstract and highlights')
aw = len(abstract.split())
# longest phrase compute_values can generate for each pending abstract placeholder (words)
WORST = {'P3F_orders_nonreplay': 5, 'P3F_agem_verdict_short': 15, 'P3F_rfmem_verdict': 17}
pend_abs = re.findall(r'⟦(P3F_[^⟧]+)⟧', abstract)
worst = aw + sum(WORST.get(k, 15) - 1 for k in pend_abs)
check(worst <= 250, f'abstract within 250 words, also with the longest pending phrases ({aw} words now, worst case {worst})')
check('earlier version' not in abstract, 'M7: no earlier-version sentence in the abstract')
z0 = {k: f(k) for k in V if re.fullmatch(r'(TON|UNSW|NB)_zd_stage0_t[12]', k)}
check(min(z0, key=z0.get) == 'NB_zd_stage0_t2' and max(z0, key=z0.get) == 'UNSW_zd_stage0_t2',
      'abstract not-yet-learned range NB t2 – UNSW t2 is min–max over datasets and tasks')
hd = Document(HERE / 'Highlights.docx')
hls = [p.text for p in hd.paragraphs if p.text.strip()]
check(all(len(h) <= 85 for h in hls), f'every highlight ≤ 85 characters ({max(map(len, hls))})')
check(3 <= len([h for h in hls if not h.lower().startswith('highlights')]) <= 5, 'three to five highlights')
check(not any('A-GEM' in h or 'earlier version' in h for h in hls), 'highlights: no A-GEM claim before the rerun, no earlier version')

print(f'\n{len(fails)} failure(s), {len(pending)} pending')
for x in fails:
    print('  FAIL', x)
for x in pending:
    print('  PEND', x)
if PENDING_KEYS:
    print(f'pending placeholders ({len(PENDING_KEYS)}): ' + ', '.join(sorted(PENDING_KEYS)))
sys.exit(len(fails))
