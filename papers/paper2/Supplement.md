# Paper 2 supplementary material (double-brace placeholders are filled from values.json by build.py --supp)

## Title

**Supplementary material for: What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model**

## Contents

This supplement gives the tables referred to in the main text as Tables S1–S17 and S19–S21; Section S18 gives method details moved from the main text. Every number is computed from the released result files by the same script as the main text ({{REPO}}). ASR is the share of detected attack flows of the attack set classified as benign after the attack; all values are means over the seeds stated in each caption.

## S1. Saturation of the black-box search

{{table:saturation}}

## S2. Per-action ablation and budget floors

The ablation and floor grids are scaled copies of one unit grid, so the candidates of a larger budget do not contain those of a smaller one; this is why a few entries are not monotone (for example bytes only above bytes and packets for the UNSW-NB15 random forest).

{{table:ablation}}

## S3. End-to-end evasion and prevalence-weighted ASR

{{table:overall}}

## S4. Cheapest successful action

For each flow that the grid search evades without a TTL change, the smallest share of the budget used by any successful candidate, and the absolute amounts of that candidate (medians over flows, means over seeds).

{{table:minaction}}

## S5. Further data splits

{{table:splits}}

## S6. Strongest-attack ASR per attack class

{{table:perclass}}

## S7. Defended CNNs under the feature-space reference attacks

{{table:fsdef}}

## S8. Saturation of the attacks on the trees

{{table:treesat}}

## S9. Physical feasibility of the grid search

{{table:feas}}

## S10. One further configuration per defence

{{table:sweep}}

## S11. Operating point with all actions (both datasets, unadjusted p)

{{table:opall}}

## S12. BPDA action-space PGD against the networks

{{table:bpda}}

## S13. Further defence variants

{{table:defvar}}

## S14. Networks against trees, grid search

{{table:nettree_grid}}

## S15. Selected features and how the attack treats them

| Dataset | Feature | Role in the attack |
|---|---|---|
| TON-IoT | src_bytes, src_pkts, src_ip_bytes | set by the attacker (padding, extra packets; header bytes added per packet) |
| TON-IoT | duration | set by the attacker (delays) |
| TON-IoT | dst_bytes, dst_pkts, dst_ip_bytes | destination side; unchanged in the attack, but may react to extra packets in practice |
| TON-IoT | conn_state_rej, conn_state_s0, conn_state_s1, conn_state_sf, conn_state_sh | connection state; unchanged in the attack, may react to destination replies |
| TON-IoT | src_port | held fixed (the attacker could choose it; Section 6.2) |
| TON-IoT | dst_port, proto_tcp, proto_udp, service_dns, service_http | fixed by the attack's target and protocol |
| TON-IoT | dns_qclass, dns_qtype, dns_rcode, dns_RD_t, dns_RA_f, dns_rejected_f, dns_rejected_t | fixed (content of the attack's DNS exchange) |
| UNSW-NB15 | sbytes | set by the attacker (padding, extra packets) |
| UNSW-NB15 | ackdat | set by the attacker (delayed handshake acknowledgement) |
| UNSW-NB15 | sttl | set by the attacker (initial TTL) |
| UNSW-NB15 | sload, smean | recomputed from the modified bytes, packets and duration |
| UNSW-NB15 | dload | recomputed from the unchanged destination bytes and the modified duration |
| UNSW-NB15 | ct_state_ttl | recomputed from the new TTL with the testbed's lookup |
| UNSW-NB15 | dbytes | destination side; unchanged in the attack, but may react to extra packets in practice |
| UNSW-NB15 | ct_dst_src_ltm, ct_srv_dst | counts over recent connections; fixed (would need further flows) |

## S16. AT-threat with a corner-start inner attack

{{table:atcs}}

## S17. Destination replies to extra packets

{{table:reply}}

## S18. Method details

### S18.1 Feature recomputation

**Feature consistency.** Every selected feature that depends on the actions is recomputed (Table 2). On TON-IoT, whose byte counts are payload bytes, the source bytes, packets and duration change directly, and the source IP bytes grow by the added payload plus, for each added packet, the flow's header bytes per packet, estimated as (source IP bytes − source bytes)/packets and clipped to 20–60 bytes (40 bytes for flows without packets). On UNSW-NB15, whose source byte count includes headers (at least 28 bytes per packet in the data), each added packet also adds 40 bytes, or the flow's mean packet size if that is smaller (but at least 28 bytes); the mean source packet size is recomputed from bytes and packets; the source load is rescaled by the change in bytes and duration and the destination load by the change in duration, because the destination sends the same bytes over a longer flow; the ACK delay changes directly; and the TTL-derived state feature (ct_state_ttl) is looked up from the training data for the new TTL, the destination TTL and the connection state, using the nearest training TTL for values not seen in training. The recomputed raw features then pass through the deployed min–max scaling and, for the networks, the quantile transform. With all actions zero, this mapping reproduces the pipeline's features to within {{map_err_max}}, a rounding effect. The main evaluation allows a fractional number of added packets and does not bound the payload per packet; Section 5.1 reports a check with integer packets and a maximum transmission unit (MTU).

### S18.2 Attacks

- **Black-box grid search** (decision-based). For each flow and budget, the search evaluates all corners of the budget box (each action at zero or at its maximum; 8 on TON-IoT, 16 on UNSW-NB15) and 120 (TON-IoT) or 48 (UNSW-NB15) random points inside it, scaled to the flow's budget. On UNSW-NB15 each candidate is combined with the unchanged TTL and with each of the {{N_TTL}} most frequent non-zero TTLs of benign training flows ({{UNSW_ttl_cands}}), giving 128 queries per flow and budget on TON-IoT and 320 on UNSW-NB15. The flow evades if any candidate is classified benign. We report volume-and-timing actions with the TTL unchanged (volume only), the TTL choice with all other actions at zero (TTL only), and both together; per-action variants (bytes only, packets only, delays only); and, at β = 0.5, the search with 1,000 further random points per flow. The grid of each budget is the same unit grid scaled to that budget, so the candidates of a larger budget do not contain those of a smaller one.
- **TTL sweep** (UNSW-NB15): the TTL-only attack with each value from 1 to 255.
- **Adaptive search** (score-based): from a random start in the budget box, 40 rounds of 16 proposals per flow (Gaussian steps with a step size decaying from 0.3 to 0.02 of the box, random snaps of one action to zero or its maximum), keeping the proposal with the highest benign probability; on UNSW-NB15 the TTL is either kept, drawn from the candidates or drawn from 1–255. It is run, at β = 0.5 and 2, on the flows that the grid search did not evade, and reported together with the grid (grid + adaptive). It is a score-based random search in the spirit of the Square Attack (Andriushchenko et al., 2020), used in AutoAttack (Croce & Hein, 2020), adapted to a box of three or four actions.
- **White-box action PGD** (networks): sign-gradient ascent on the actions, scaled to the budget box, through the feature map, the min–max scaling and a differentiable implementation of the quantile transform, for each TTL choice. A basic version (log-probability of the benign class, one random start, 50 steps of 5% of the box) is run at every budget; a strong version (margin between the benign class and the best other class, restarts from the zero and full corners and two random points, 100 steps with a step decaying from 10% to 1% of the box, success checked every 25 steps) at β = 0.5 and 2.
- **Transfer**: actions computed by the basic white-box PGD on the undefended CNN of the same seed (β = 0.5 and 1), applied to the undefended MLP, RF and GBDT.

### S18.3 Saturation, feasibility and the feature-space reference

**Saturation and feasibility checks.** On the trees, for which no white-box attack exists, we attack 500 detected flows per seed (seeds 0–2) with regular grids of increasing resolution (up to 17³ points on TON-IoT and 7⁴ on UNSW-NB15, each with every TTL candidate) and a longer adaptive search (150 rounds of 32 proposals, any TTL) on the flows nothing else evaded. We did not run an exact attack on the trees, so their ASRs remain lower bounds. To check physical feasibility, we repeat the grid search at β = 0.5 with the number of added packets rounded to an integer and with enough packets added for the payload to fit within 1,460 payload bytes per packet (TON-IoT) or 1,500 bytes per packet including headers (UNSW-NB15), discarding candidates that would need more packets than the budget allows.

**Conventional reference.** We compare with attacks in min–max feature space, which are not realisable and serve only for comparison. The basic reference is targeted L∞ PGD (ε ∈ {0.01, 0.05, 0.1}, 50 steps, through the differentiable quantile transform) on all selected features and on the mutable features only (the action-dependent volume and timing features). Because a black-box search of the same ball beat this attack by a wide margin (Section 5.1), we strengthened the reference for ε ∈ {0.005, 0.01, 0.02, 0.05, 0.1}: a black-box search over 128 random corners of the full ball; an exhaustive search over the patterns {−ε, 0, +ε} on the mutable features and over the increase-only patterns {0, +ε}; and, for the networks, a strong PGD (margin loss, restarts from the record, a random point and a random corner, step decaying from 0.25ε to 0.025ε) through a straight-through approximation of the quantile transform (backward-pass differentiable approximation, BPDA; Athalye et al., 2018), on the full ball and on the mutable features. The strongest feature-space attack on the mutable features counts a flow as evaded if the exhaustive search or the BPDA PGD on those features evades it.

### S18.4 Is AT-all a fair baseline? (full comparison)

AT-all and CCAT were trained with the white-box ε-PGD through the differentiable quantile transform. That attack evaded them on TON-IoT on only {{TON_IoT_cnn_at_all_wb_unconstrained_e0.05_r2seeds}} and {{TON_IoT_cnn_ccat_wb_unconstrained_e0.05_r2seeds}} of detected flows at ε = 0.05, but the BPDA PGD on {{TON_IoT_cnn_at_all_wb_bpda_e0.05}} and {{TON_IoT_cnn_ccat_wb_bpda_e0.05}}, and the black-box corner search evaded every detected flow of every TON-IoT CNN, defended or not (Table S7). The apparent robustness of the ε-ball defences in their own ball was therefore an artefact of a weak attack, and on TON-IoT a ranking of the defences by white-box ε-PGD would have been misleading. Training AT-all against the BPDA attack instead did not give a usable baseline: on TON-IoT the network lost most of its clean accuracy (macro-F1 {{TON_IoT_cnn_at_all_bpda_f1}}, benign FPR {{TON_IoT_cnn_at_all_bpda_fpr}}), it was still evaded on every detected flow by the corner search, and at the matched FPR the realistic attack evaded it on {{TON_IoT_cnn_at_all_bpda_op_matched_asr}} of detected flows. We therefore cannot exclude that a better-trained ε-ball defence would do better, but we found none. On UNSW-NB15 the ranking does differ between threat models: under the corner search the ε-ball defences were the least evaded CNNs (AT-all {{UNSW_NB15_cnn_at_all_bb_corners_e0.05}}, CCAT {{UNSW_NB15_cnn_ccat_bb_corners_e0.05}}, against {{UNSW_NB15_cnn_at_threat_bb_corners_e0.05}} for AT-threat), and under the realistic attack the most evaded. One further configuration of each defence (seed 0, Table S10) showed that the configuration matters: at the matched FPR on TON-IoT, AT-threat trained at β = 0.25 or 1 was evaded on {{TON_IoT_cnn_at_threat_b0.25_s0_op_matched_asr}} and {{TON_IoT_cnn_at_threat_b1.0_s0_op_matched_asr}} of detected flows, against {{TON_IoT_cnn_at_threat_s0_op_matched_asr}} for the main configuration on the same seed, and AT-all with ε = 0.02, 0.05 and 0.1 on {{TON_IoT_cnn_at_all_e0.02_s0_op_matched_asr}}, {{TON_IoT_cnn_at_all_s0_op_matched_asr}} and {{TON_IoT_cnn_at_all_e0.1_s0_op_matched_asr}}; on UNSW-NB15, with the TTL action included, every AT-threat configuration was evaded less than every AT-all configuration. Because ε = 0.1 looked best for AT-all on seed 0, we retrained it on all five TON-IoT seeds: at the matched FPR it left {{TON_IoT_cnn_at_all_e0.1_op_matched_e2e}} of attack flows missed or evaded ({{TON_IoT_cnn_at_all_e0.1_opd_e2e}} against the undefended CNN, p = {{TON_IoT_cnn_at_all_e0.1_opd_e2e_p}}; AT-threat minus AT-all at ε = 0.1: {{TON_IoT_cnn_at_threat_opd_vs_cnn_at_all_e0.1_e2e}}, p = {{TON_IoT_cnn_at_threat_opd_vs_cnn_at_all_e0.1_e2e_p}}; unadjusted, outside the primary family), at a clean macro-F1 of {{TON_IoT_cnn_at_all_e0.1_f1_5}} and a benign FPR of {{TON_IoT_cnn_at_all_e0.1_fpr_5}}. Its seed-0 advantage therefore did not hold across seeds. A wider sweep of both defences over all seeds may still find better settings.

## S19. Clean performance and strongest attack of every defence

{{table:defences}}

## S20. Attack-aware ROC

{{table:roc}}

## S21. The primary family as first specified

{{table:origfam}}
