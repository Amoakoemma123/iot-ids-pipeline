# Paper 2: response to the final-round-2 reviews (2026-11-16)

Referee report: review_final2_referee.md (6/10, major revision). All numbers are filled from values.json; verify.py: 0 failures.

## Major issues

| Item | What changed |
|---|---|
| 1. Unequal inner searches (networks vs trees) | New runs: **AT-threat, corner start** (§4.4): the inner PGD of each training flow starts at the most benign-scoring of all box corners plus eight random points, as strong a candidate set as the tree augmentation's; five seeds, both datasets (Table S16, §5.3). TON-IoT: 40.9% missed or evaded at the matched FPR, against 38.3% for AT-threat with a random start (+2.6 pp [−19.3 pp, +24.4 pp]) and 46.9% undefended (−6.0 pp [−19.9 pp, +7.8 pp]). UNSW-NB15: −44.9 pp [−56.1 pp, −33.7 pp] against undefended, as with the random start. The network result therefore does not come from a weaker inner search. "The only defence in our study that did so" is removed; §6.1, §6.2 and §7 state the comparison. Fix (b), AT-threat at β = 1 on all seeds, was not run; the corner start addresses the same reach problem more directly. |
| 2a. Single tail threshold | New attack-aware ROC (run_adv_r2 part 'roc'; Table S20; §4.5, §5.3): share missed or evaded by the β = 0.5 grid as the validation FPR runs 0.5–5%, partial area as summary. TON-IoT: AT-threat −20.5 pp [−37.2 pp, −3.8 pp], corner start −21.5 pp [−32.1 pp, −11.0 pp], TRADES +14.1 pp [+3.2 pp, +25.0 pp]. UNSW-NB15: no network defence lowers it at FPR ≤ 5% (AT-threat −4.5 pp [−19.6 pp, +10.6 pp]); the matched point there sits at the undefended CNN's FPR of about 0.198. The paper now says the network-defence result depends on the operating point and the decision rule (abstract, highlight 3, §5.3, §6.1, §7). Grid search only: the adaptive search depends on the threshold. Secondary, added in revision. |
| 2b. Null worded as finding | t-based CI given (AT-threat −8.6 pp, [−34.4 pp, +17.2 pp]); heading "no detectable gain"; §5.3 says the absence of a significant gain is not evidence of zero gain; Welch tests added as a check (Table S16); §4.5 says tests are conditional on one training set. |
| 2c. Decompose missed or evaded | Table 7 now has separate "Missed" and "Evaded" columns (ASR, own kept). |
| 2d. Second rule for TRADES | Second rule (1 − max attack-class probability) in Table S20: TRADES +44.0 pp [+28.5 pp, +59.5 pp], so the TRADES result does not hinge on the rule; the AT-threat gain does (+1.0 pp [−19.6 pp, +21.6 pp]). |
| 3. Destination replies called conservative | "Conservative" removed (§3). Table S15 lists the selected features and their role. New sensitivity run (run_adv_r2 part 'reply', Table S17): one 40-byte reply per added packet. Undefended models: −3.1 pp to +4.6 pp (TON-IoT), −0.8 pp to +12.2 pp (UNSW-NB15), both directions. Every defence was evaded more (+3.6 pp to +9.9 pp; +3.4 pp to +25.0 pp): e.g. augmented GBDT on TON-IoT 1.6% → 10.0%, UNSW-NB15 AT-threat 8.8% → 28.8%. Stated in §5.3, §6.1, §6.2, §7 and the abstract. Connection-state changes were not modelled. |
| 4. Mixed rules/metrics; lower bound; MILP | Abstract and cover letter now quote the tree result under the same rule and FPR as the networks (16.9% → 1.6% missed or evaded) and call it a lower bound from our searches; §7 says "at least". Held-out-action results stay in §5.3 (abstract limit). Exact MILP not run; §6.2 gives the concrete reason (about 142,400–436,439 leaf variables per flow for the RF, up to 1,000 boosted trees for the GBDT, for hundreds of flows and several seeds). |
| 5. Length | Main text 14.3k → about 12.6k words (build count incl. tables). Moved to the supplement: method details (S18.1–3), the full AT-all comparison (S18.4), the defence-cost table (old Table 8, now S19). Table 4 lost two columns; §2 contributions, §5.1, §5.2, §5.3, §5.4, §6.1 and §6.2 shortened (§6.2 is now a list). **Not yet at the 9.5–10k target**; the new analyses (ROC, corner start, replies) added text. |
| 6. Lesson 1 | §6.1 first lesson now leads with the ranking failure (ii), and says the strong ε-ball is usually a safe upper bound but not a guide to defence choice. |

## Minor issues

1. Tree p-values in Table 7 labelled unadjusted.
2. §5.4 uses the volume-only values for the networks without TTL features, as Table 8.
3. Table 5 caption states its seed sets.
4. Original primary family reported in Table S21; §4.5 states that it leads to the same TON-IoT conclusion.
5. §4.5 says the bootstrap understates between-seed uncertainty; defence differences use t-based CIs.
6. S11 retitled ("both datasets, unadjusted p") and its caption explains why its p-values differ from Table 7.
7. §4.5 and §6.2 say tests are conditional on one training set.
8. §3 gives an operational reading of the floors for a SYN probe.
9. Abstract no longer claims a TTL-dependent tree advantage; §5.4 says the RF without TTL features is still below the networks but on three seeds, with no claim of a difference.
10. Abstract rewritten with fewer numbers and one recommendation.
11. §4.2 gives GBDT iterations after early stopping (111–480 on TON-IoT; the cap on 4 of five UNSW-NB15 seeds) and RF leaves per tree (tree_capacity.py). The RF was not trained with out-of-bag scoring.
12. Figure 2 caption states the curves are at each model's own threshold.
13. Highlights rewritten; all ≤ 85 characters.
14. Zenodo deposit of the models: **decision for the authors** (data statement unchanged: on request).
15. §5.1 says the strongest-attack query count is an upper bound. The median queries to first success of the adaptive search was not stored and is not reported.
