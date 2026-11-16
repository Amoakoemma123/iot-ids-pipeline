"""Fill Manuscript.md from values.json and build Manuscript_v2.docx.

Order: compute_values.py -> make_figures.py -> build.py [--anon] -> verify.py
Unfilled placeholders are listed and left visible as ⟦key⟧, so a partial build still works.
"""
import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from docx_helpers import add_table, bullet, caption, numbered, para, setup

HERE = Path(__file__).resolve().parent
MD = HERE / 'Manuscript.md'
FIG = HERE / 'figures'

# Authors as on the Telco paper (user, 2026-10-24); details copied from ../../../NEw REsearch/manuscript/build_manuscript.py
AUTHOR = 'Emmanuel Kwame Amoako'
EMAIL = 'amoakopuyol123@gmail.com'
AUTHOR_LINES = (
    'Emmanuel Kwame Amoakoᵃ, Mahamudu Amaduᵇ, James Quaysonᵇ, Kingsford Donkorᶜ, Daniel Asamoah Aprakuᵈ',
    'ᵃ Independent researcher, P.O. Box 121, Dormaa Ahenkro, Ghana',
    'ᵇ Department of Information Technology and Decision Sciences, University of Energy and Natural Resources, Sunyani, Ghana',
    'ᶜ Department of Computer Engineering, University of Energy and Natural Resources, Sunyani, Ghana',
    'ᵈ Department of Computer Science, University of Energy and Natural Resources, Sunyani, Ghana',
)
AUTHOR_NAMES = ('Amoako', 'Amadu', 'Quayson', 'Donkor', 'Apraku')
EXTRA = {
    'REPO': 'https://github.com/Amoakoemma123/iot-ids-pipeline',
    # Elsevier wording; user 2026-11-13 (review r3, small edit 9): AI was used for readability
    'AI_DECLARATION': ('During the preparation of this work the authors used Claude (Anthropic) to improve the readability and language of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The released scripts compute every reported number from the result files, and an automated check compares each quantitative statement in the text with those files.'),
}

# Elsevier: 3–5 highlights, each ≤ 85 characters including spaces (separate file). Written once results exist
# (verify.py reports them as pending while this list is empty).
HIGHLIGHTS = [
    'Attacker may only pad, add packets, delay and set TTL; derived features recomputed',
    'Feature-space attacks overstated evasion, partly by breaking feature dependencies',
    'Network defence gains depended on the operating point and the decision rule',
    'Tree augmentation cut TON-IoT evasion at the same false-positive rate',
    'Unseen TTL values and destination replies still evaded the augmented trees',
]
ABSTRACT_MAX_WORDS = 250

FIGURES = {
    '5.1': [(FIG / 'fig1_asr_budget.png', 16.0,
             'Black-box grid-search attack success rate (share of detected attack flows classified as benign) against the budget β, '
             'volume and timing actions with the TTL unchanged, for the four undefended models. Lines: mean over five seeds; '
             'bands: range over seeds. On UNSW-NB15 the dotted line marks the rate reached by choosing the TTL alone.')],
    '5.3': [(FIG / 'fig2_defences.png', 16.0,
             'Black-box grid-search attack success rate (volume, timing and, on UNSW-NB15, TTL) against the budget β for the CNN '
             'without defence, with the clean-loop control and with each adversarial-training defence, each at its own operating '
             'point (class of highest probability); Table 7 compares them at a matched false-positive rate. Lines: mean over '
             'seeds; bands: range over seeds. Each curve is at the model’s own threshold, which Section 5.3 shows to favour the defences.'),
            (FIG / 'fig3_tree_defences.png', 16.0,
             'Tree defences at β = 0.5: attack success rate of the grid search and of the strongest attack (grid, denser grid, '
             'adaptive search and, on UNSW-NB15, any TTL) against the undefended trees and each augmentation variant (mean over '
             'seeds; error bars: range over seeds). The gap '
             'between the two bars shows how much of the augmentation gain holds only against the candidates it was built with.'),
            (FIG / 'fig5_roc.png', 16.0,
             'Attack-aware ROC: share of the sampled attack flows missed or evaded by the β = 0.5 grid search when the benign '
             'threshold is set on validation benign flows to each false-positive rate (rule of Table 7; UNSW-NB15 volume and '
             'timing actions). Top: CNN without defence and with each adversarial-training defence; bottom: undefended (dashed) '
             'and augmented (solid) trees. Lines: mean over seeds (five for the CNNs, three for the trees); bands: range over '
             'seeds. Grid search only. The second decision rule is in Table S20.')],
    '5.4': [(FIG / 'fig4_ttl_sweep.png', 16.0,
             'UNSW-NB15: attack success rate of the TTL choice alone (no other action) for every initial TTL from 1 to 255, '
             'undefended models, mean over five seeds. Dotted lines: the four TTL candidates used by the grid search and in training.')],
}
_ASR = 'attack success rate (ASR), the share of detected attack flows of the attack set classified as benign after the attack'
TABLE_CAPTIONS = {
    '2.3': ['Closest prior studies of realistic evasion attacks on flow-based intrusion detectors and this study. '
            '"–": not reported in the parts of the paper we checked.'],
    '3.': ['Attacker actions, their budgets and the selected features recomputed from them. All other selected features '
           '(destination bytes and packets, protocol, service and state fields, ports, connection counts) are left unchanged.'],
    '5.1': ['Undefended models, clean test set: macro-F1 (mean ± standard deviation over five seeds), mean F1 of the rare classes, '
            'benign false-positive rate and share of the sampled attack flows detected before any modification.',
            f'Undefended models: {_ASR}, at β = 0.5 (mean ± standard deviation over five seeds, in percentage points). Grid: '
            'black-box search over the budget box; adaptive: score-based search on the flows the grid did not evade; transfer: '
            'actions computed on the undefended CNN of the same seed; strongest: a flow counts as evaded if the grid, the grid '
            'with 1,000 further points, the adaptive searches or (networks) the strong white-box PGD evades it, with a 95% '
            'two-level bootstrap confidence interval (seeds, then flows); with five seeds this interval understates between-seed '
            'uncertainty. The TTL-only attack on UNSW-NB15 is in Table 8. n/a: not applicable to the model.',
            'Realistic attack against the strengthened feature-space reference, undefended models (mean over seeds: five for '
            'the CNN and MLP trained with all features, three for the trees and for the detectors without TTL features). '
            'Realistic: strongest attack with volume and timing actions only, β = 0.5. Realistic range: largest, over the '
            'mutable features, of the median width (in min–max units) of the interval each feature can reach under the '
            'realistic actions. Relaxed realistic: every mutable feature set independently within that interval (all '
            'combinations of its two ends and its value, 1,000 random points, and the realistic evasions), i.e. realistic '
            'magnitudes with the dependencies between features broken. Mutable features: exhaustive search over the patterns '
            '{−ε, 0, +ε} (±) or {0, +ε} (increases only) on the volume and timing features in min–max space, with BPDA PGD on '
            'the same features for the networks (±); on UNSW-NB15 the realistic actions also lower sload, smean and dload, so '
            'increases only is not a relaxation of them there. Full ball: black-box corners of the whole ε-ball and, for the '
            'networks, BPDA PGD. Crossover: smallest ε in {0.005, 0.01, 0.02, 0.05, 0.1} at which the feature-space attack '
            'evades at least as many detected flows as the realistic one.'],
    '5.2': ['Networks minus trees: difference in strongest-attack ASR at β = 0.5 (percentage points) with Welch’s t-test over '
            'five seeds each, Holm-adjusted within each dataset and attack (four tests). On TON-IoT, which has no TTL action, '
            'all attacks use volume and timing only. Grid-search rows: Table S14.'],
    '5.3': ['Defences at a matched operating point. Own: class of highest probability. Matched: a flow is benign if its '
            'benign probability is at least a threshold set on validation benign flows to the undefended model’s FPR (same '
            'family and seed); the undefended model is shown under the same threshold rule at its own validation FPR. '
            'Missed: share of the sampled attack flows classified benign before modification. Evaded: share of all sampled '
            'attack flows detected before and classified benign after the attack (grid search and, for the networks, adaptive '
            'search at β = 0.5; on UNSW-NB15 volume and timing actions only; all actions: Table S11). ASR, own: evaded share of '
            'the detected flows at the own operating point. Missed or evaded: the sum of the two. Δ: paired '
            'difference in missed or evaded from the undefended model at the matched rule, with the Holm-adjusted p-value of '
            'the primary family for the networks and an unadjusted p-value for the trees (whose primary test uses the '
            'strongest attack at the own operating point, Table S19). Seeds: number of seeds of the operating-point runs.'],
    '5.4': ['UNSW-NB15: the TTL artefact. TTL-only ASR for the four candidates, for the other values of 1–255 and for any value '
            '1–255 (undefended models, five seeds), and detectors retrained without sttl and ct_state_ttl (three seeds): clean '
            'macro-F1, grid-search ASR with volume and timing actions for both sets of detectors (same seeds) and '
            'strongest-attack ASR with volume and timing actions at β = 0.5.'],
}

SUPP_CAPTIONS = {
    'S1': ['Saturation of the black-box search, volume and timing actions at β = 0.5, undefended models (ASR, mean over five '
           'seeds). Adaptive (any TTL): the TTL is drawn from 1–255 during the adaptive search. Strongest: all actions.'],
    'S2': ['Per-action ablation and budget floors, grid search with volume and timing actions at β = 0.5, undefended models '
           '(ASR, mean over five seeds). Floors × k: the additive floors of the budget (1,000 bytes, 10 packets, 1 s, 0.1 s) '
           'multiplied by k.'],
    'S3': ['Undefended models: share of the sampled attack flows missed before any modification, share of all sampled attack '
           'flows classified benign after the strongest attack (missed or evaded), and the strongest-attack ASR re-weighted by '
           'the test-set prevalence of each attack class (mean over five seeds).'],
    'S4': ['Cheapest successful volume-and-timing action of the grid search at β = 0.5, per attack class, undefended CNN and '
           'GBDT (medians over the evaded flows, means over the seeds with at least one evaded flow of the class; Seeds).'],
    'S5': ['Further data splits, seed 0 only: clean macro-F1 and ASR at β = 0.5 on the main split and two further stratified '
           'splits. The main-split strongest attack includes the 1,000-point grid; the split runs do not, so their strongest '
           'attack is somewhat weaker. Compare the grid columns.'],
    'S6': ['Strongest-attack ASR per attack class at β = 0.5 (all actions), undefended models, mean over five seeds, with each '
           'class’s share of the attack flows in the test set.'],
    'S7': ['Defended CNNs under the feature-space reference attacks at ε = 0.05 (min–max space) and under the strongest '
           'realistic attack at β = 0.5 (all actions). White-box ε-PGD is the 50-step attack through the differentiable quantile '
           'transform used to train AT-all and CCAT; BPDA PGD is the strong version (margin loss, restarts, straight-through '
           'quantile transform). Means over the seeds of each model.'],
    'S8': ['Saturation of the attacks on the trees: 500 detected flows per seed (seeds 0–2), β = 0.5. Regular grids: every '
           'point of a grid with r levels per action (r³ or r⁴ candidates, each with the TTL candidates on UNSW-NB15); long '
           'adaptive search: 150 rounds of 32 proposals with any TTL, on the flows nothing else evaded. Strongest: the union '
           'of the main evaluation on the same flows; the union of all column starts from it, so the columns do not add up.'],
    'S9': ['Physical feasibility: grid search at β = 0.5 (all actions) as in the main evaluation, and the same candidates with '
           'the number of added packets rounded to an integer and enough packets added for the payload to fit 1,460 payload '
           'bytes per packet (TON-IoT) or 1,500 bytes per packet including headers (UNSW-NB15); candidates that would need more '
           'packets than the budget allows are discarded. Evasions lost: share of '
           'detected flows evaded only by an infeasible candidate. Means over the round-2 seeds.'],
    'S10': ['One further training configuration per defence, seed 0 only: AT-all with ε = 0.02 and 0.1 and AT-threat trained '
            'at β = 0.25 and 1, next to the main configuration (seed 0). ASR at matched FPR: benign threshold set to the '
            'undefended CNN’s validation FPR. AT-all with ε = 0.1 on all five TON-IoT seeds: Section 5.3.'],
    'S11': ['Operating point as in Table 7 for both datasets, with all actions (TTL included) on UNSW-NB15; on TON-IoT, which has no TTL action, the rows repeat Table 7. Δ with unadjusted p-values, which therefore differ from the Holm-adjusted values of Table 7.'],
    'S12': ['Gradient masking check for the action-space attack: strong white-box action PGD (margin loss, four restarts, 100 '
            'steps, each TTL choice, β = 0.5) with the gradient taken through the straight-through (BPDA) quantile transform, '
            'against every network at its own operating point. New evasions: share of detected flows evaded by it but not by '
            'the strongest attack. Means over the seeds of each model.'],
    'S13': ['Further defence variants, as in Table S19: second augmentation round, augmentation without delays or without byte '
            'padding (held-out action), and AT-all trained against the BPDA attack.'],
    'S14': ['Networks minus trees for the grid search (as Table 6).'],
    'S15': ['The features selected by the shared pipeline (25 on TON-IoT, 10 on UNSW-NB15) and their treatment by the '
            'realistic attack. Destination-side features and the connection state are not modelled as reacting to the '
            'attack; Section 6.2 discusses the consequences.'],
    'S16': ['AT-threat (CNN) with the inner PGD of each training flow started at the most benign-scoring of all corners of the β = 0.5 box and eight random points (five seeds). Missed or evaded at the matched benign FPR as in Table 7 (UNSW-NB15: volume and timing actions). Δ: paired difference over seeds with t-based 95% CI, paired t-test p and Welch p as a check; unadjusted, outside the primary family. The undefended CNN is under the same threshold rule at its own FPR.'],
    'S17': ['Sensitivity to destination replies: β = 0.5 grid (all actions on TON-IoT; volume and timing on UNSW-NB15) with and without one destination reply of 40 header bytes per added packet (TON-IoT: dst_pkts, dst_ip_bytes; UNSW-NB15: dbytes, dload; connection state unchanged), same candidates and seeds as Table 4. Lost: evaded without replies but not with them; gained: the reverse (shares of detected flows).'],
    'S19': ['Defences: seeds, clean macro-F1 and its paired difference from the undefended model of the same seeds, mean F1 of '
            'the rare classes, benign false-positive rate, and strongest-attack ASR at β = 0.5 at the model’s own operating '
            'point, with the undefended model’s ASR on the same seeds and paired changes against the undefended model and, for '
            'CNN defences, the clean-loop control (secondary family). On TON-IoT, which has no TTL action, every attack uses '
            'volume and timing only. Further variants (second augmentation round, held-out actions, BPDA training): Table S13.'],
    'S20': ['Attack-aware ROC: share of the sampled attack flows missed or evaded by the β = 0.5 grid search (TON-IoT all actions; UNSW-NB15 volume and timing) when the benign threshold is set on validation benign flows to each FPR, for the rule of Table 7 (benign if p(benign) exceeds the threshold) and for a second rule (benign if 1 − the highest attack-class probability exceeds it). Grid search only (the adaptive search depends on the threshold), all attack-set flows. Partial area: mean over 0.5–5% FPR (trapezoid). Δ: paired difference from the undefended model of the same family over seeds, t-based 95% CI; unadjusted, secondary. Test FPRs run slightly above the validation FPRs.'],
    'S21': ['The primary family as specified before the operating-point experiments: ASR at the matched benign FPR (UNSW-NB15: volume and timing) with the undefended model under the class of highest probability as reference, and the tree augmentations on the strongest-attack ASR. Paired differences over seeds, Holm-adjusted within each dataset. The family reported in the main text (Section 4.5, Table 7) replaces the endpoint by the share missed or evaded and the reference by the undefended model under the same threshold rule.'],
}


def fill(text, V):
    missing = set()

    def tab(m):
        k = m.group(1)
        if k in V['tables']:
            return V['tables'][k]
        missing.add('table:' + k)
        return f'⟦table:{k}⟧'

    def val(m):
        k = m.group(1)
        if k in V:
            return str(V[k])
        missing.add(k)
        return f'⟦{k}⟧'

    text = re.sub(r'\{\{table:(\w+)\}\}', tab, text)
    text = re.sub(r'\{\{([\w.]+)\}\}', val, text)
    text = text.replace('p = < ', 'p < ')   # p-values formatted as '< 0.001'
    return text, sorted(missing)


def line_numbers(doc):
    sectPr = doc.sections[0]._sectPr
    ln = OxmlElement('w:lnNumType')
    ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); ln.set(qn('w:distance'), '283')
    sectPr.find(qn('w:pgMar')).addnext(ln)


def section_key(h):
    m = re.match(r'(S\d+)\. ', h)
    if m:                      # supplementary section: one table per section, labelled S1, S2, ...
        return m.group(1)
    if h.startswith('Appendix '):
        return h.split()[1].rstrip('.')
    m = re.match(r'(\d+\.\d+|\d+\.)', h)
    return m.group(1) if m else None


def build(md_text, out, authors=True):
    lines = md_text.splitlines()
    doc = Document(); setup(doc)
    doc.styles['Normal'].paragraph_format.line_spacing = 1.5
    line_numbers(doc)
    title_at = lines.index('## Title') + 2
    title = lines[title_at].strip('*')
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph(style='Title'); p.add_run(title); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(24)
    if authors:
        for line in (*AUTHOR_LINES, '', f'Corresponding author: {AUTHOR}, {EMAIL}'):
            para(doc, line, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    else:
        doc.core_properties.author = doc.core_properties.last_modified_by = ''
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    current, pending, in_refs, used = None, [], False, {}

    def flush():
        nonlocal pending
        for path, width, cap in pending:
            q = doc.add_paragraph(); q.alignment = WD_ALIGN_PARAGRAPH.CENTER
            q.paragraph_format.keep_with_next = True
            if Path(path).exists():
                q.add_run().add_picture(str(path), width=Cm(width))
            else:   # partial build before the results exist; verify.py reports the marker
                q.add_run(f'⟦figure {Path(path).name} not yet generated⟧')
            caption(doc, 'Figure', cap)
        pending = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if i in (0, title_at) or line.startswith('## Title'):
            i += 1; continue
        if line.startswith('## ') or line.startswith('### '):
            flush()
            level = 1 if line.startswith('## ') else 2
            text = line.lstrip('# ').strip()
            current = section_key(text)
            pending = list(FIGURES.get(current, []))
            if text in ('1. Introduction', 'References'):
                doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            doc.add_paragraph(text, style=f'Heading {level}')
            in_refs = text == 'References'
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i]); i += 1
            rows = [[c for c in re.split(r'(?<!\\)\|', r.strip())[1:-1]] for r in block]
            rows = [r for r in rows if not all(re.fullmatch(r'\s*:?-{3,}:?\s*', c) for c in r)]
            caps = TABLE_CAPTIONS.get(current)
            assert caps and used.get(current, 0) < len(caps), f'table without caption in section {current}'
            caption(doc, 'Table', caps[used.get(current, 0)], above=True,
                    label=f'{current}.{used.get(current, 0) + 1}' if current == 'A' else (current if current and current.startswith('S') else None))
            used[current] = used.get(current, 0) + 1
            add_table(doc, rows)
            continue
        if not line.strip() or line.startswith('<!--'):
            i += 1; continue
        if re.match(r'^\s*- ', line):
            bullet(doc, line.strip()[2:], 1 if line.startswith('  ') else 0); i += 1; continue
        m = re.match(r'^(\d+)\. (.*)', line)
        if m:
            numbered(doc, m.group(1), m.group(2)); i += 1; continue
        q = para(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT if in_refs else WD_ALIGN_PARAGRAPH.JUSTIFY)
        if in_refs:
            q.paragraph_format.left_indent = Cm(1.27); q.paragraph_format.first_line_indent = Cm(-1.27)
        i += 1
    flush()
    doc.save(out)
    return used


def main_supp():
    """Supplement.md -> Supplement_v2.docx (and _anon); tables labelled S1, S2, ..."""
    anon = '--anon' in sys.argv
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(EXTRA)
    if anon:
        V['REPO'] = '[repository link removed for anonymous review]'
    text, missing = fill((HERE / 'Supplement.md').read_text(encoding='utf-8'), V)
    TABLE_CAPTIONS.update(SUPP_CAPTIONS)
    out = HERE / ('Supplement_v2_anon.docx' if anon else 'Supplement_v2.docx')
    used = build(text, out, authors=not anon)
    (HERE / ('Supplement_filled_anon.md' if anon else 'Supplement_filled.md')).write_text(text, encoding='utf-8')
    print(f'saved {out.name}; tables:', sum(used.values()))
    unused = {k: len(c) - used.get(k, 0) for k, c in SUPP_CAPTIONS.items() if used.get(k, 0) != len(c)}
    if unused:
        print('WARNING unused supplement captions:', unused)
    if missing:
        print(f'WARNING {len(missing)} unfilled placeholders:', ', '.join(missing))


def main():
    if '--supp' in sys.argv:
        return main_supp()
    anon = '--anon' in sys.argv
    V = json.loads((HERE / 'values.json').read_text(encoding='utf-8'))
    V.update(EXTRA)
    md = MD.read_text(encoding='utf-8')
    if anon:   # double-anonymised copy: no CRediT names, no author-owned repository link
        md = re.sub(r'\*\*CRediT authorship contribution statement\.\*\*[^\n]*\n\n', '', md)
        V['REPO'] = '[repository link removed for anonymous review]'
    text, missing = fill(md, V)
    (HERE / ('Manuscript_filled_anon.md' if anon else 'Manuscript_filled.md')).write_text(text, encoding='utf-8')
    out = HERE / ('Manuscript_v2_anon.docx' if anon else 'Manuscript_v2.docx')
    used = build(text, out, authors=not anon)
    unused = {k: len(c) - used.get(k, 0) for k, c in TABLE_CAPTIONS.items() if used.get(k, 0) != len(c)}
    hl = [fill(h, V)[0] for h in HIGHLIGHTS]
    hd = Document(); setup(hd)
    hd.add_paragraph('Highlights', style='Heading 1')
    for h in hl:
        bullet(hd, h)
    hd.save(HERE / 'Highlights.docx')
    for h in hl:
        if len(h) > 85:
            print(f'WARNING highlight > 85 chars ({len(h)}): {h}')
    abstract = text.split('## Abstract')[1].split('**Keywords')[0]
    aw = len(abstract.split())
    print(f'abstract: {aw} words (limit {ABSTRACT_MAX_WORDS})' + ('  <-- TOO LONG' if aw > ABSTRACT_MAX_WORDS else ''))
    words = len(re.findall(r'\w+', text.split('## References')[0]))
    print(f'saved {out.name}  (~{words} words before references)')
    if unused:
        print('WARNING unused table captions:', unused)
    if missing:
        print(f'WARNING {len(missing)} unfilled placeholders:', ', '.join(missing))


if __name__ == '__main__':
    main()
