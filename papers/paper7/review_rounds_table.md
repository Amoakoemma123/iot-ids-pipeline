# AI-instance review rounds: counts per paper, round and role (for Paper 7, Section 5.2)

Built 2026-11-06 from the review reports on disk (`Revision_v2/Paper4..Paper7/review_*.md`), `STATUS.md`, `Paper7/response_r1.md`, `Paper7/evidence_P5_P6_pipeline.md`, the git logs of `Revision_v2` and `Papers pipeline`, and spot greps of the current manuscripts. No existing file was changed.

## Counting rules

- **Raised** = items that the report itself labels as problems: Critical/Major/Minor headings (C/M/m, MC/m, N, H/M/L, MAJ/min, B/L, E/W/R), numbered minor lists, the numbers audits' W/O, P, C, B/C or N/C/U/R rows, and non-OK verdict rows in the references tables. Each labelled item counts once and sub-bullets are not counted. Not counted: "verified/checked correct" sections, strengths, re-run plans, "requested changes" lists that repeat earlier items, "still open from round N" carry-overs (they are counted in the round that first raised them), and items the report itself marks as fine (for example P4 r1 methods minor 10, P4 r4 numbers C18). For P7 r1 referee the 5 prioritised requests in §4 are the majors, because the response letter follows them.
- **Fixed** = recorded as fixed, resolved, mostly resolved or disclosed by (in order of preference): the next round's status table, a STATUS line, a commit message, `response_r1.md`, or a grep of the current manuscript/build file. Disclosure counts as fixed when the reviewer asked for disclosure.
- **Partly** = the next round or STATUS says "partly", or a request was answered only in part (declined components are named).
- **Rejected / false alarm / no action** = a later instance or the main session judged the item wrong, or the item was explicitly not acted on because its source could not be verified. Items the reviewer itself said need no action are also listed here, marked separately.
- **Unknown / open** = no item-level evidence of the outcome, or still open in the last record. Aggregate STATUS phrases such as "all items fixed" were **not** turned into item counts.
- **Superseded** = moot because the manuscript was rebuilt from new runs or the item's text was removed.
- **Changed** = the fix altered a printed number, or removed, reversed or rescoped a stated finding. Pure wording, format, citation or disclosure fixes do not count. Only items with a known outcome are counted, so this is a lower bound.

## Table 1. Per report

| Paper | Round (date) | Role | Raised (labels used) | Fixed | Partly | Rejected / FA / no action | Unknown / open | Superseded | Changed no./concl. | Outcome evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| P4 | r1 (10-25) | referee | 24 (M1–M10, minor 1–14) | 19 | 4 (M5, M8, M10; minor 14: commit hash but no DOI) | 1 FA (minor 4: "random F1 ≈ 0.10 is incorrect"; the r3 referee showed 0.10 = 1/K is exact) | 0 | 0 | 11 (M1–M7, M9; minor 2, 11, 13) | `review_r2_referee.md` Part 1; `review_r3_referee.md` m9; STATUS l.195 |
| P4 | r1 | numbers | 57 (N1–N17, C1–C27, U1–U9, R1–R4; 10 rated H) | 0 itemised | 0 | 0 | 0 | 57 (v1 text replaced by a full rebuild) | n/a | `review_r2_methods_code.md` "m4 Superseded (new manuscript)"; STATUS l.111 "FULL REBUILD" |
| P4 | r1 | methods-code | 20 (C1–C4, M1–M7, minor 1–9) | 18 | 0 | 0 | 1 (m7 `pipeline.pkl` assertion, still open in r4) | 1 (m4) | 10 (C1–C4, M1–M4, M6, M7) | `review_r2_methods_code.md` status table; `review_r4_methods_code.md` m9; bbbfca9 |
| P4 | r1 | references | 16 (6 misattributed + 5 metadata per STATUS; 5 structural) | 16 | 0 | 0 | 0 | 0 | 0 | `review_r3_references.md` §2 (all fixed or removed) |
| P4 | r2 (10-25) | referee | 26 (MC1–MC10, m1–m16) | 25 (resolved/mostly by r4) | 1 (m13 archived DOI) | 0 | 0 | 0 | 5 (MC1, MC3, MC4, MC6, MC8) | `review_r3_referee.md` Part 1; `review_r4_referee.md` §5; ae29b4c |
| P4 | r2 | numbers (+refs) | 28 (P1–P28: 5 H, 13 M, 10 L; 0 wrong numbers of ≈330) | 21 | 4 (P10, P14, P22, P27) | 0 | 3 (P9, P26, P28 not fixed at r3; P26 evidence conflicts: STATUS l.159 vs r4 methods m9) | 0 | 5 (P1, P2, P5, P10, P14) | `review_r3_numbers.md` "Round-2 items" |
| P4 | r2 | methods-code | 15 (C1, M1–M2, minor 1–12) | 15 (12 at r3, M2/m8/m11 by r4) | 0 | 0 | 0 | 0 | 5 (C1, m2, m3, m4, m6) | `review_r3_methods_code.md` and `review_r4_methods_code.md` status tables; a848bed, cd758f6 |
| P4 | r3 (10-25 night) | referee | 10 (N1–N10) | 9 | 1 (N3) | 0 | 0 | 0 | 6 (N1–N6; N2 reversed "every site gains" into "local better 17/20") | `review_r4_referee.md` §5; STATUS l.167, l.184 |
| P4 | r3 | numbers | 14 (3 H, 6 M, 5 L; 0 wrong of ≈610) | 0 | 0 | 0 | 14 (r4 audit has no status table) | 0 | unknown | — |
| P4 | r3 | methods-code | 12 (M1–M4, minor 1–8) | 10 | 2 (m4, m8) | 0 | 0 | 0 | 3 (M1, M2, M3) | `review_r4_methods_code.md` status table |
| P4 | r3 | references | 6 (3 metadata, 1 misattribution, 2 uncited) | 5 | 0 | 0 | 1 (uncited companion studies, still open at r4) | 0 | 0 | `review_r4_references.md` §4 |
| P4 | r4 (10-26) | referee | 31 (M1–M5, minor 1–26) | 4 (M1, M3, M5, minor 23) | 1 (minor 22) | 0 | 26 | 0 | 1 (M3: personalisation finding became dataset-specific) | STATUS l.183–185, l.194; f0d4e13, 235920c, 0000edc |
| P4 | r4 | numbers | 17 (C1–C17; 0 wrong of ≈660) | 1 (C1) | 0 | 0 | 16 | 0 | 1 (C1) | STATUS l.185 "full-budget per-class (7/9 classes lose ≥10 pts)" |
| P4 | r4 | methods-code | 15 (M1–M4, minor 1–8, 10–12) | 3 (M1, M2, m8) | 1 (M4: no-port check only) | 0 | 11 | 0 | 2 (M1: all N-BaIoT numbers; own traffic 7/9 n.s. became 9/9, p = 0.006; M2) | f0d4e13 message; STATUS l.183, l.192 |
| P4 | r4 | references | 5 (1 support problem, 4 optional) | 0 | 0 | 0 | 5 | 0 | 0 | — |
| P5 | r1 (10-24) | numbers | 6 (M1–M3, m1–m3) | 5 | 0 | 0 | 1 (m3) | 0 | 4 (M1, M2: 4,845 became 8,472; M3; m1: 67% became 66.8%) | `review_round2_referee.md` Part A 1, 3, 15; Manuscript.md l.58 |
| P5 | r1 | methods-code | 19 (H1–H3, M1–M6, L1–L10) | 8 (H1, H2, H3, M1, M2, L8; M4, M5 later) | 2 (M3, M6) | 1 no action (L7, judged harmless by the r2 instance) | 8 (L1–L6, L9, L10) | 0 | 6 (H1, H3: v2 "100%→64%" withdrawn; H2, M1, M2, M3) | `review_round2_methods_code.md` §3; STATUS l.81, l.99; Manuscript.md l.134, l.184 |
| P5 | r2 (10-24 eve) | referee | 24 (M1–M8, m1–m16) | 6 (M1, M2, M3, M5, M7, m8) | 0 | 0 | 18 | 0 | 4 (M2, M3, M5, M7) | STATUS l.99–103; 5c819da, a199dcd |
| P5 | r2 | numbers | 15 (4 wrong, 11 overstated; 0 table errors of ~620) | 6 (#2, #5, #7, #8, #9, #10; checked in the current Manuscript.md) | 0 | 0 | 9 | 0 | 4 (#5, #8, #9, #10) | Manuscript.md l.9, l.58, l.160, l.188 |
| P5 | r2 | methods-code | 26 (MAJ-1–9, min-1–17) | 8 (MAJ-1–6, 8, 9) | 0 | 0 | 18 (MAJ-7, all min) | 0 | 4 (MAJ-1, MAJ-3, MAJ-6, MAJ-8) | STATUS l.99–103; 5c819da |
| P6 | r1 (11-01) | referee | 23 (M1–M8, minor 1–15) | 13 (M2, M3, M7; minors 1–5, 7, 9, 12, 14, 15) | 9 (M1, M4, M5, M6, M8; minor 6, 8, 10, 13). Declined parts: second non-saturated IoT dataset, CNN tuning | 0 | 1 (minor 11) | 0 | 6 (M2, M3, M4, M5, minor 1, minor 3) | `review_r2_referee.md` status tables; STATUS l.344–345 |
| P6 | r1 | numbers | 23 (W1–W4, C1–C13, M1–M6) | 7 (W2, W3, W4, C1, M1, M2, M6) | 0 | 0 | 16 | 0 | 4 (W2/C1 Holm family, W3 5/5 became 2/5, M1) | 98724a4, 01397ef; `review_r2_referee.md` minor 1 |
| P6 | r1 | methods-code | 24 (B1–B4, mismatch 1–10, L1–L10) | 10 (B1, B2, B4, mm4, mm5, mm8, mm9, mm10, L7, L8) | 1 (L1) | 2 no action (L4, L10: the reviewer itself said no action needed) | 11 | 0 | 2 (B1: 26 capped runs re-run in v4; B4) | `review_r2_methods_code.md` §1, m1/m3/m5/m10–m12; d2e74ca, b37e9c3; Manuscript.md l.77 |
| P6 | r1 | references | 7 (priority fixes 1–7; 40/40 refs real) | 3 (1, 6 after r2, 7) | 0 | 0 | 4 | 0 | 0 | 98724a4; STATUS l.344 |
| P6 | r2 (11-03) | referee | 22 (N1–N5, n1–n17) | 5 (N2, N3, N4, n1, n16) | 2 (N1, N5; second dataset and CNN tuning declined) | 0 | 15 | 0 | 3 (N1, N2, N3) | STATUS l.344–345; 5dd6165 |
| P6 | r2 | numbers + refs | 21 (E1–E5, W1–W10, R1–R6) | 10 (E1–E5, R1–R5) | 0 | 0 | 11 (W1–W10, R6) | 0 | 3 (E1, E2, E4) | STATUS l.344; build.py l.71; Manuscript.md l.95 |
| P6 | r2 | methods-code | 17 (M1–M3, m1–m14) | 8 (M1–M3, m1, m3, m4, m5, m7) | 0 | 0 | 9 | 0 | 2 (M1, M3) | STATUS l.344; Manuscript.md l.77; build.py l.77 |
| P7 | r1 (11-05) | referee | 27 (5 major requests, minor 1–22) | 25 | 2 (major 5 and minor 21: a human-verification statement was written, but no independent human re-derivation was done) | 0 (the isolating run for F7 was declined and the mode reframed) | 0 | 0 | 4 (major 1 four modes "New"; major 2 "typical standard" removed; major 3 22→18 modes; minor 13 seeds) | `response_r1.md`; c86f352, 99d1bc6, c6949f2 |
| P7 | r1 | numbers | 32 (W1–W12, O1–O20) | 28 | 3 (O3, O14, O16) | 0 | 0 | 1 (O19) | 10 (W1, W2, W4–W9, W12, O12) | `review_r2_numbers.md` §A |
| P7 | r1 | references | 18 (3 claim, 4 metadata, 11 citation-needed) | 15 | 0 | 1 rejected (cite the TF XLA page: the page could not be confirmed, so the claim rests on the project's own logs) | 2 (Kaufman article no.; "deterministic where allowed" row) | 0 | 2 (Flood novelty claim, Kus claim) | `response_r1.md` |
| P7 | r2 (11-05/06) | referee | 23 (M1–M8, minor 1–15) | 0 | 0 | 0 | 23 (fixes pending) | 0 | unknown | STATUS l.390 |
| P7 | r2 | numbers | 23 (B1–B4, C1–C19) | 0 | 0 | 0 | 23 (fixes pending) | 0 | unknown | STATUS l.390 |
| **Total (32 reports on disk)** | | | **646** | **303** | **33** | **5** (2 FA/rejected by another instance or the author, 3 marked no-action by the reviewer itself) | **246** | **59** | **≥107** | |

Not in the totals (no report on disk; points reconstructed): **P5 r1 referee** (10-24, 5/10 major). Its 14 points are listed in `review_round2_referee.md` Part A: fixed 9, partly 5 (points 3, 5, 7, 8, 13), changed 6 (points 1, 3, 6, 7, 10, 11, including "Lightweight" dropped from the title).

### Table 2. By role (same 32 reports)

| Role | Reports | Raised | Fixed | Partly | Rejected / FA / no action | Unknown / open | Superseded | Changed |
|---|---|---|---|---|---|---|---|---|
| Referee | 9 | 210 | 106 | 20 | 1 | 83 | 0 | ≥40 |
| Numbers (incl. P4 r2 and P6 r2 refs parts) | 10 | 236 | 78 | 7 | 0 | 93 | 58 | ≥31 |
| Methods-code | 8 | 148 | 80 | 6 | 3 | 58 | 1 | ≥34 |
| References | 5 | 52 | 39 | 0 | 1 | 12 | 0 | ≥2 |

Without P7 r2 (outcomes pending) and P4 r1 numbers (superseded): 543 raised, 303 fixed, 33 partly, 5 rejected/no action, 200 unknown, 2 superseded. Where the outcome is known (343 items), 88% were fixed, 10% partly fixed, and under 2% were false alarms or no-action items.

Caveats: (1) The numbers audits mostly flag prose, not table values. Every audit after P4 r1 found 0–5 wrong printed numbers among ≈330–660 checked. (2) "Fixed" for the last round of each paper (P4 r4, P5 r2, P6 r2) rests on STATUS, commit messages and greps of the current text. Most minor items in those rounds have no item-level record, which is why they are counted as unknown. (3) Only two items were shown to be wrong by another route. Reviewer instances almost never produced a false alarm that anyone recorded, but nothing systematically looked for false alarms either.

## (a) Failures the reviewer instances missed that another route caught

| # | Failure | Missed by | Caught by (route) | Evidence | Changed a number/conclusion? |
|---|---|---|---|---|---|
| 1 | P5 §3 said 13.7% of UNSW test records are copies (features + label), but the clean-subset F1 (0.6089 → 0.6472) excluded the 16.3% features-only set | P5 r2 numbers and methods-code (MAJ-2's table printed both 13.7% and 16.3%), P5 r2 referee, P5 fresh r3 review | Main session, during the P7 evidence pass | `evidence_P5_P6_pipeline.md` l.128 ("not fixed; P5 is marked ready"); STATUS l.380; a6212d7 | Definition only (both numbers correct); P5 PDF stale |
| 2 | The `id` effect depends on the metric: accuracy +0.05 with and without duplicates, so "leak = id + duplicates together" holds only for macro-F1 | P5 r1 (all 3 roles) and P5 r2 (all 3 roles; MAJ-3/M7 attacked the decomposition design, not the metric) | A later review round (P5 fresh review 2026-10-31, no report) | STATUS l.199; 25ede5b | Yes (Table 1 and text) |
| 3 | LIME features ranked by raw coefficient (slope per SD), not by contribution | P5 r1 and r2 methods-code (both read the XAI code), both referees | A later review round (P5 fresh review 2026-10-31) | STATUS l.199, l.208–209; b743315, 25ede5b | Yes (TON RF J5 0.39 → 0.24; UNSW RF near chance) |
| 4 | §6.1 contradiction in P5 | P5 r1, r2 | P5 fresh review 2026-10-31 | STATUS l.199 | Wording |
| 5 | P6 v3: 30 runs hit the 60-epoch cap at last-epoch weights (26 had an earlier best epoch) | P6 r1 methods-code B1 found the bug but listed only 13 main runs, ignoring random splits | Main session (recount from results_v3) | `evidence_P5_P6_pipeline.md` §4 item 3 | Count corrected (all re-run in v4 anyway) |
| 6 | P4 BatchNorm running statistics made evaluation swing (central val F1 0.51–0.82) | P4 r1 referee, methods-code, numbers (the referee asked only for a normalisation ablation) | Main session, while watching v2 runs | STATUS l.118; 401f938 (GroupNorm default) | Yes (model changed; BN-era runs superseded) |
| 7 | P6 v4 analysis reused 104 random-split + 15 N-BaIoT v3 runs | Arose between rounds; no reviewer had seen it | Script (compute_values crashed on the missing clean score) | STATUS l.325; f618ede | Yes ("above all 10 random splits" became 8/10; UNSW +0.009, p = .06) |
| 8 | P7 §4.6 oracle sentence: the sign of −0.016 is reversed relative to its framing | P7 r1 numbers (O10 flagged only the significance) | A later review round (P7 r2 numbers B1) | `review_r2_numbers.md` B1 | Pending |
| 9 | Errors introduced while fixing round-1 items: "Study 4 six review rounds" (from the fix for r1 minor 14; P4 had five review rounds), the stale "279 references" (now 314), and "every number from result files" (1.9% is hand-typed) | Fixes made in the main session | A later review round (P7 r2 numbers B4, B3, C2) | `review_r2_numbers.md` | Pending |
| 10 | P5 TON copy check at 8 decimals gave 17.1% instead of 0.1% (never printed) | — | Route not documented (inferred: r2 methods-code measured 0.1%) | `evidence_P5_P6_pipeline.md` item 15d | No (internal near miss) |

Related but outside the review rounds: P3 `verify.py` caught 4 text errors before P3's only review (STATUS l.307, 0836a6c). The P7 v1 headline (0.7455 not among its own seeds; population SD) was found by the main session (STATUS l.379).

## (b) Rounds recorded in STATUS or commits with no report on disk

| Paper | Round | Record | Report on disk |
|---|---|---|---|
| P3 | Only review round, 2026-11-02 09:39, 6/10 major | STATUS l.311; ceb1070 (earlier-task retention, UNSW FPR, random-order gap, oracle CIs, limitations, refs); pipeline 96b2fbc ("budget group – reviewer check") → full-budget check, STATUS l.324 | None; issue count unknown |
| P4 | Fresh review 2026-10-31, 7.5 minor (= review round 5) | STATUS l.198 "all items fixed"; 8ffe0d2 "Paper 4 review round 5 fixes" | None |
| P4 | STATUS "round 5" and "round 6" | These are **revision** rounds (fix/experiment cycles answering reviews r3 and r4: ec9a6de/52bbca3 and f0d4e13/235920c/21678ac), not review rounds | n/a. P4 had 5 review rounds, 4 with reports |
| P5 | Round-1 referee, 2026-10-24, 5/10 major | STATUS l.60 (names only the numbers and methods reports) | None; 14 points recoverable from `review_round2_referee.md` Part A |
| P5 | Fresh review (round 3), 2026-10-31, 6/10 major | STATUS l.199; 25ede5b | None |
| P4 r2, P5 r1–r2, P6 r2, P7 r2 | Roles not run separately: no references reviewer for P4 r2 (inside numbers), P5 (Crossref literature agent instead), P6 r2 (merged with numbers), P7 r2 (referee + numbers only) | — | — |
| P1, P2 | No review round recorded in STATUS | — | — |

The P7 r2 numbers audit (C17) independently flags the same three missing reports (P3, P5 round 3, P4 fresh review). The paper's statement that the review reports are released should be qualified accordingly.
