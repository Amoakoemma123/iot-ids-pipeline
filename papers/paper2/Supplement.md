# Paper 2 supplementary material (double-brace placeholders are filled from values.json by build.py --supp)

## Title

**Supplementary material for: What Can an Attacker Actually Change? Evasion and Hardening of Compact Flow-Based Intrusion Detectors Under a Realistic Threat Model**

## Contents

This supplement gives the tables referred to in the main text as Tables S1–S14. Every number is computed from the released result files by the same script as the main text ({{REPO}}). ASR is the share of detected attack flows of the attack set classified as benign after the attack; all values are means over the seeds stated in each caption.

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

## S11. Operating point with all actions on UNSW-NB15

{{table:opall}}

## S12. BPDA action-space PGD against the networks

{{table:bpda}}

## S13. Further defence variants

{{table:defvar}}

## S14. Networks against trees, grid search

{{table:nettree_grid}}
