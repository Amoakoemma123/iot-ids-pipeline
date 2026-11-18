# Paper 4 v2 supplementary material (double-brace placeholders are filled from values.json by build.py --supp)

## Title

**Supplementary Material for "Federated Intrusion Detection for IoT Networks Revisited: Device-Level Partitions, Tuned Baselines and Record-Level Differential Privacy"**

## S1. Data, Partitions and Protocol Details

### A. N-BaIoT preparation

N-BaIoT contains the traffic of nine commercial IoT devices (two doorbells, a thermostat, a baby monitor, four security cameras and a webcam), recorded while benign and while infected with the BASHLITE (gafgyt) and Mirai botnets, and described by 115 statistics of the packet streams computed by the Kitsune feature extractor. The full data have 7,062,606 records. We draw a uniform 3% sample of every device–class file (seed 42), which keeps every device's class mix, and remove duplicates at the 32-bit floating-point precision the model sees (54,091 records; at 64-bit precision most of them look distinct), and 17 records whose feature vector carries two labels. The BASHLITE TCP and UDP floods saturate the stream statistics: their sampled records collapse to BASHLITE tcp (40 distinct records) and BASHLITE udp (32 distinct records), so no split can keep their test records unseen in training, and we drop both classes. The remaining 9 classes are benign traffic, three BASHLITE and five Mirai attacks; 2 devices (Ennio Doorbell and Samsung SNH 1011 N Webcam) were not infected with Mirai. The records are split 70/15/15 stratified by device and class, zero-variance features are dropped and min–max scaling is fitted on the training split, without feature selection, and a guard removes the 1,117 validation and 1,168 test records that are still identical to a training record (features and label) after scaling. This gives 110,227 training, 22,503 validation and 22,453 test records. Because the stream statistics of neighbouring packets overlap, a random split still makes absolute N-BaIoT scores optimistic; the comparisons in the paper are between methods trained and tested on the same split.

### B. Partitions

We describe a client's label skew by the share of its two largest classes and the number of classes that reach 1% of its records. Over the five repetitions on TON-IoT, the two largest classes make up 38–75% of a client's data at α = 0.5 (mean 56%, 7.8 classes on average) and 53–98% at α = 0.1 (mean 77%, 4.7 classes). Under the host partition, which assigns each of the 682 destination hosts of the training split as a whole to the currently smallest of four clients, in a fresh random order per repetition, the two largest classes make up 42–67% (mean 52%; 7.9 classes). The largest host receives 26% of the training records and the ten largest 81%, so client sizes vary from 23,741 to 52,783 records. On N-BaIoT the devices hold 3,116 to 18,992 training records; the two largest classes make up 40–67% of a device's data (7.9 classes on average), against 64% and 88% on average for Dirichlet partitions over nine clients with α = 0.5 and 0.1.

### C. Training details

Round selection uses the pooled validation split; in a deployment each client would report the confusion counts on its own validation records, which the server sums to the same pooled score. Local-only models are selected on the pooled validation set, which favours models that generalise beyond the client's own traffic and therefore, if anything, disadvantages local models on their own traffic. In every epoch the last incomplete batch is dropped (training is compiled with a fixed batch size); the dropped records differ between epochs. GPU training with XLA is not bit-wise deterministic, so a repetition reproduces its partition and initialisation but not every training step; re-running identical configurations (15 pairs) changed the validation-selected test macro-F1 by 0.036 on average (SD of the signed differences 0.049) and by up to 0.109. We did not try TensorFlow's deterministic mode (TF_DETERMINISTIC_OPS without XLA), which would slow training. For reference, a constant predictor of the majority class scores a macro-F1 of 0.037 on TON-IoT and 0.066 on UNSW-NB15. The centralised reference keeps its Adam state across epochs; it and the local-only models are evaluated and selected every two epochs (50 checkpoints), so a selected "round" of the centralised model counts two epochs.

### D. Hyperparameter grids and selected values

Tuning uses three tuning repetitions on TON-IoT and UNSW-NB15 (at α = 0.5) and two on N-BaIoT (on the device partition), with partitions (on N-BaIoT, where the device partition is fixed, only initialisations and batch orders), initialisations and batch orders different from the reported repetitions; the criterion is the mean validation macro-F1 of the selected round. Selected values are given for TON-IoT, UNSW-NB15 and N-BaIoT in that order. FedAvg's Adam learning rate ({10⁻³, 3·10⁻³, 6·10⁻³, 10⁻²}; on N-BaIoT also 3·10⁻⁴, which scored lower): 0.003, 0.003, 10⁻³. FedProx's μ ({10⁻⁵, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹}): 10⁻⁵, 10⁻³, 10⁻⁴. FedAdam's server learning rate ({0.003, 0.01, 0.03, 0.1}): 0.03, 0.03, 0.03. FedLC's τ ({0.1, 0.3, 1, 3, 10, 30, 100}): 3, 10, 1. SCAFFOLD's SGD learning rate ({0.01, 0.03, 0.1, 0.3, 1}): 0.1, 0.03, 0.1. FedAvg with SGD ({0.01, 0.03, 0.1, 0.3, 1}): 0.3, 0.03, 0.1. On N-BaIoT, values that diverged on both other datasets (SGD learning rate 1) or scored lowest in their grids there (μ = 10⁻¹, server learning rate 0.003) are left out and FedLC's grid starts at τ = 0.3. FedLC's grid was extended to τ = 30 and 100 because its best τ on UNSW-NB15 lay at the upper edge of narrower grids. Only FedProx's μ on TON-IoT (smallest value) lies at an edge of its grid; because μ → 0 recovers FedAvg, this does not affect the FedProx comparison. The centralised reference's learning rate was checked over the FedAvg grid: 0.003 on TON-IoT and 0.003 on UNSW-NB15 (mean validation macro-F1 0.001: 0.896, 0.003: 0.902, 0.006: 0.901, 0.01: 0.870 on TON-IoT).

### E. DP-SGD details and communication

A DP step whose Poisson draw is empty is skipped (the accounting treats every step as released), and the noise comes from TensorFlow's standard, not cryptographically secure, generator. The guarantee assumes that the preprocessing statistics, the client sizes and the hyperparameters, all derived from the training data without privacy, are public; it does not hide which client participated. A model update for TON-IoT occupies 95.0 kB, so a round with four clients transfers 0.76 MB and 50 rounds 38.0 MB (SCAFFOLD twice as much), more than the 12.8 MB of the preprocessed TON-IoT training matrix: FL does not save bandwidth over a one-off upload; its benefit is that the traffic never leaves the site.

## S2. Primary Comparisons

For each dataset we declared a family of primary comparisons after the first analysis of TON-IoT and UNSW-NB15 but before the N-BaIoT, personalisation and no-port experiments: FedAvg against the centralised reference (IID, α = 0.5, α = 0.1, device-level partition) and, at α = 0.5, α = 0.1 and the device-level partition, FedAvg against the local-only reference, FedProx, FedAdam and FedLC against FedAvg, and SCAFFOLD against FedAvg with SGD (on N-BaIoT, comparisons with the local-only reference and between algorithms use the device partition only; 19, 13 and 9 comparisons). Their p-values are Holm-corrected within each dataset; † marks significance after correction. All other comparisons are exploratory, with uncorrected p-values. With five repetitions, the minimum difference detectable with 80% power in the algorithm comparisons is about 0.05 on TON-IoT and 0.04 on UNSW-NB15. Exact non-parametric tests cannot reach p < 0.05 with n = 5; the t-tests assume approximately normal paired differences, which we cannot check. Table S1 lists every primary comparison with its uncorrected and Holm-adjusted p-value.

<!--TABLE:paired-->
| Dataset | Comparison (paired over repetitions) | Mean difference [95% CI] | p | Holm-adjusted p |
|---|---|---|---|---|
| TON-IoT | FedAvg, IID − Centralised |−0.042 [−0.059, −0.025] | 0.002 | 0.042 † |
| TON-IoT | FedAvg, α = 0.5 − Centralised |−0.059 [−0.101, −0.016] | 0.018 | 0.202 |
| TON-IoT | FedAvg, α = 0.1 − Centralised |−0.185 [−0.286, −0.084] | 0.007 | 0.098 |
| TON-IoT | FedAvg, α = 0.5 − Local only |+0.080 [+0.043, +0.117] | 0.004 | 0.057 |
| TON-IoT | FedAvg, α = 0.1 − Local only |+0.286 [+0.194, +0.379] | 0.001 | 0.019 † |
| TON-IoT | FedProx, α = 0.5 − FedAvg |+0.005 [−0.037, +0.048] | 0.745 | 1.000 |
| TON-IoT | FedProx, α = 0.1 − FedAvg |−0.010 [−0.066, +0.046] | 0.645 | 1.000 |
| TON-IoT | FedAdam, α = 0.5 − FedAvg |−0.083 [−0.118, −0.049] | 0.003 | 0.042 † |
| TON-IoT | FedAdam, α = 0.1 − FedAvg |−0.046 [−0.073, −0.018] | 0.010 | 0.119 |
| TON-IoT | FedLC, α = 0.5 − FedAvg |+0.011 [−0.040, +0.062] | 0.570 | 1.000 |
| TON-IoT | FedLC, α = 0.1 − FedAvg |−0.017 [−0.055, +0.022] | 0.297 | 1.000 |
| TON-IoT | SCAFFOLD, α = 0.5 − FedAvg (SGD) |+0.020 [−0.001, +0.042] | 0.058 | 0.521 |
| TON-IoT | SCAFFOLD, α = 0.1 − FedAvg (SGD) |+0.018 [−0.043, +0.078] | 0.468 | 1.000 |
| TON-IoT | FedAvg, device partition − Centralised |−0.179 [−0.284, −0.074] | 0.009 | 0.119 |
| TON-IoT | FedAvg, device partition − Local only |+0.190 [+0.117, +0.262] | 0.002 | 0.034 † |
| TON-IoT | FedProx, device partition − FedAvg |−0.024 [−0.082, +0.034] | 0.318 | 1.000 |
| TON-IoT | FedAdam, device partition − FedAvg |−0.076 [−0.132, −0.020] | 0.020 | 0.202 |
| TON-IoT | FedLC, device partition − FedAvg |+0.014 [−0.041, +0.069] | 0.520 | 1.000 |
| TON-IoT | SCAFFOLD, device partition − FedAvg (SGD) |−0.004 [−0.024, +0.016] | 0.610 | 1.000 |
| UNSW-NB15 | FedAvg, IID − Centralised |−0.065 [−0.084, −0.047] | < 0.001 | 0.007 † |
| UNSW-NB15 | FedAvg, α = 0.5 − Centralised |−0.099 [−0.139, −0.060] | 0.002 | 0.027 † |
| UNSW-NB15 | FedAvg, α = 0.1 − Centralised |−0.171 [−0.268, −0.074] | 0.008 | 0.082 |
| UNSW-NB15 | FedAvg, α = 0.5 − Local only |+0.032 [−0.037, +0.100] | 0.271 | 1.000 |
| UNSW-NB15 | FedAvg, α = 0.1 − Local only |+0.136 [+0.066, +0.206] | 0.006 | 0.064 |
| UNSW-NB15 | FedProx, α = 0.5 − FedAvg |−0.004 [−0.024, +0.016] | 0.582 | 1.000 |
| UNSW-NB15 | FedProx, α = 0.1 − FedAvg |+0.045 [+0.011, +0.080] | 0.021 | 0.187 |
| UNSW-NB15 | FedAdam, α = 0.5 − FedAvg |−0.009 [−0.025, +0.007] | 0.212 | 1.000 |
| UNSW-NB15 | FedAdam, α = 0.1 − FedAvg |−0.018 [−0.046, +0.009] | 0.140 | 0.983 |
| UNSW-NB15 | FedLC, α = 0.5 − FedAvg |+0.063 [+0.015, +0.110] | 0.021 | 0.187 |
| UNSW-NB15 | FedLC, α = 0.1 − FedAvg |−0.004 [−0.043, +0.034] | 0.770 | 1.000 |
| UNSW-NB15 | SCAFFOLD, α = 0.5 − FedAvg (SGD) |−0.003 [−0.036, +0.031] | 0.841 | 1.000 |
| UNSW-NB15 | SCAFFOLD, α = 0.1 − FedAvg (SGD) |−0.070 [−0.185, +0.045] | 0.165 | 0.990 |
| N-BaIoT | FedAvg, device partition − Centralised |−0.105 [−0.124, −0.085] | < 0.001 | < 0.001 † |
| N-BaIoT | FedAvg, device partition − Local only |+0.187 [+0.171, +0.204] | < 0.001 | < 0.001 † |
| N-BaIoT | FedProx, device partition − FedAvg |+0.013 [−0.008, +0.033] | 0.159 | 0.325 |
| N-BaIoT | FedAdam, device partition − FedAvg |+0.028 [−0.012, +0.068] | 0.125 | 0.325 |
| N-BaIoT | FedLC, device partition − FedAvg |+0.011 [−0.004, +0.027] | 0.108 | 0.325 |
| N-BaIoT | SCAFFOLD, device partition − FedAvg (SGD) |+0.017 [−0.002, +0.037] | 0.067 | 0.267 |
| N-BaIoT | FedAvg, IID − Centralised |−0.032 [−0.048, −0.015] | 0.006 | 0.030 † |
| N-BaIoT | FedAvg, α = 0.5 − Centralised |−0.057 [−0.081, −0.033] | 0.003 | 0.020 † |
| N-BaIoT | FedAvg, α = 0.1 − Centralised |−0.214 [−0.316, −0.111] | 0.004 | 0.026 † |

## S3. UNSW-NB15

UNSW-NB15 is harder for every model (Table S2): the centralised model reaches 0.532 ± 0.005, and its benign false-positive rate is above 10%. FedAvg differs from it by −0.065 [−0.084, −0.047]† (IID), −0.099 [−0.139, −0.060]† (α = 0.5) and −0.171 [−0.268, −0.074] (α = 0.1), with a negative trend over skew levels (−0.036 per level, p = 0.030). Against local training, the difference is small at α = 0.5 (+0.032 [−0.037, +0.100]) and large at α = 0.1 (+0.136 [+0.066, +0.206]), where a client alone sees too few classes to learn a useful detector (the weakest client reaches 0.047). On its own classes a UNSW-NB15 client does not gain from joining (−0.015 ± 0.057 at α = 0.5, −0.002 ± 0.078 at α = 0.1; 12 and 12 of 20 pairs do better alone). FedProx differs from FedAvg by −0.004 [−0.024, +0.016] and +0.045 [+0.011, +0.080], FedLC by +0.063 [+0.015, +0.110] and −0.004 [−0.043, +0.034], FedAdam by −0.009 [−0.025, +0.007] and −0.018 [−0.046, +0.009] (α = 0.5 and 0.1), and SCAFFOLD differs from FedAvg with SGD by −0.003 [−0.036, +0.031] and −0.070 [−0.185, +0.045]. FedLC's gain at α = 0.5 comes with a higher benign FPR (17.39 ± 3.81% against 11.78 ± 7.91% for FedAvg). After feature selection, 16% of UNSW-NB15 test records share their feature vector with a training record (412 of them with a different label), which affects all methods alike.

<!--TABLE:UNSW_main-->
| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Selected round | Benign FPR (%) |
|---|---|---|---|---|---|
| Centralised (same recipe) | 0.532 ± 0.005 | 0.004 | 0.518 ± 0.015 | 36 | 12.84 ± 2.41 |
| Local only, α = 0.5 | 0.401 ± 0.046 | 0.037 | 0.387 ± 0.053 | – | – |
| Local only, α = 0.1 | 0.226 ± 0.050 | 0.040 | 0.217 ± 0.051 | – | – |
| FedAvg, IID | 0.467 ± 0.018 | 0.015 | 0.462 ± 0.028 | 39 | 13.57 ± 2.33 |
| FedAvg, α = 1.0 | 0.479 ± 0.015 | 0.012 | 0.471 ± 0.012 | 45 | 10.18 ± 3.73 |
| FedAvg, α = 0.5 | 0.433 ± 0.040 | 0.032 | 0.414 ± 0.046 | 37 | 11.78 ± 7.91 |
| FedAvg, α = 0.1 | 0.361 ± 0.101 | 0.081 | 0.334 ± 0.127 | 47 | 6.31 ± 7.03 |
| FedProx, α = 0.5 | 0.428 ± 0.038 | 0.030 | 0.422 ± 0.045 | 41 | 9.02 ± 5.69 |
| FedProx, α = 0.1 | 0.407 ± 0.090 | 0.072 | 0.352 ± 0.134 | 34 | 5.88 ± 6.74 |
| FedAdam, α = 0.5 | 0.424 ± 0.052 | 0.042 | 0.410 ± 0.061 | 40 | 11.79 ± 8.28 |
| FedAdam, α = 0.1 | 0.343 ± 0.117 | 0.094 | 0.285 ± 0.109 | 38 | 5.50 ± 5.43 |
| SCAFFOLD (SGD), α = 0.5 | 0.394 ± 0.027 | 0.021 | 0.367 ± 0.045 | 45 | 17.75 ± 3.05 |
| SCAFFOLD (SGD), α = 0.1 | 0.291 ± 0.102 | 0.082 | 0.209 ± 0.132 | 30 | 19.64 ± 5.85 |
| FedAvg (SGD), α = 0.5 | 0.396 ± 0.029 | 0.024 | 0.384 ± 0.019 | 44 | 13.07 ± 5.57 |
| FedAvg (SGD), α = 0.1 | 0.361 ± 0.049 | 0.039 | 0.320 ± 0.095 | 41 | 14.19 ± 3.16 |
| FedLC, α = 0.5 | 0.495 ± 0.016 | 0.013 | 0.485 ± 0.005 | 39 | 17.39 ± 3.81 |
| FedLC, α = 0.1 | 0.357 ± 0.120 | 0.096 | 0.350 ± 0.118 | 40 | 10.53 ± 9.37 |

Under severe skew on UNSW-NB15, SCAFFOLD is erratic: its validation-selected score varies widely across repetitions (0.291 ± 0.102) and falls to 0.209 ± 0.132 at the final round. At learning rate 0.1 it collapses in one repetition, with non-finite weights, to a score close to the majority-class predictor, and a smaller learning rate (0.01) does not help (0.279 ± 0.087; final round 0.183 ± 0.081). We observed that the same repetition is the weakest at every learning rate, which suggests that this partition is hard for SCAFFOLD rather than that the learning rate is wrong.

## S4. Robustness Checks

**Device-level against Dirichlet partitions.** Exploratory Welch differences of FedAvg between partitions: TON-IoT host partition minus α = 0.5, −0.120 [−0.217, −0.022], p = 0.024 (Welch, uncorrected); N-BaIoT device partition minus α = 0.5, −0.048 [−0.074, −0.022], p = 0.003 (Welch, uncorrected). Against the stronger skew of α = 0.1 the device-level partitions do not cost more: TON-IoT +0.006 [−0.113, +0.125], p = 0.912 (Welch, uncorrected), N-BaIoT +0.109 [+0.007, +0.211], p = 0.041 (Welch, uncorrected).

**Per-client view under label skew.** On the common test set a TON-IoT client's local model reaches FedAvg's score in only 1 of 20 client–repetition pairs at α = 0.5 and 0 of 20 at α = 0.1. Restricted to the classes a client holds (at least 1% of its data), FedAvg is better by 0.049 ± 0.028 at α = 0.5 and 0.068 ± 0.071 at α = 0.1 (repetition means), although 3 and 4 of the 20 pairs do better alone. Local-only clients reach 0.760 ± 0.028 at α = 0.5 (size-weighted 0.765 ± 0.032; weakest and strongest client 0.661 and 0.865) and 0.428 ± 0.070 at α = 0.1 (from 0.155 to 0.616).

**100 rounds.** Selected rounds cluster near the end of the 50-round budget on TON-IoT, so we repeated the comparisons with the centralised model with 100 rounds (three repetitions; exploratory). On TON-IoT FedAvg's differences from the centralised model are −0.036 [−0.061, −0.011], p = 0.026, uncorrected (IID), −0.042 [−0.060, −0.025], p = 0.009, uncorrected (α = 0.5) and −0.227 [−0.498, +0.045], p = 0.070, uncorrected (α = 0.1) after 100 rounds, with the selected round still late (78 of 100 at α = 0.5); on UNSW-NB15 the gaps are −0.045 [−0.129, +0.039], p = 0.146, uncorrected, −0.119 [−0.215, −0.024], p = 0.033, uncorrected and −0.197 [−0.487, +0.094], p = 0.100, uncorrected. A longer budget narrows the TON-IoT gap at α = 0.5 by about a third (an unpaired contrast of three with five repetitions) and leaves the others within their wide intervals; at α = 0.1 and on UNSW-NB15 some intervals include zero.

**Port features.** Removing the two port features from TON-IoT changes the centralised macro-F1 by −0.143 [−0.189, −0.096], p = 0.001, uncorrected and FedAvg's by −0.194 [−0.243, −0.146], p < 0.001, uncorrected at α = 0.5 and −0.215 [−0.288, −0.142], p = 0.001, uncorrected at α = 0.1; without ports the federation gaps are −0.110 [−0.162, −0.059], p = 0.004, uncorrected (α = 0.5) and −0.257 [−0.372, −0.143], p = 0.003, uncorrected (α = 0.1). Without ports, at α = 0.5, FedProx (−0.028 [−0.096, +0.040], p = 0.318, uncorrected) and FedLC (+0.004 [−0.048, +0.056], p = 0.833, uncorrected) match FedAvg, FedAdam trails it (−0.124 [−0.143, −0.105], p < 0.001, uncorrected), and SCAFFOLD matches FedAvg with SGD (+0.012 [−0.012, +0.036], p = 0.239, uncorrected). On the host partition without ports, FedAvg reaches 0.565 ± 0.059 on the common test set against 0.444 ± 0.039 for local models; on a site's own traffic the local model reaches 0.628 and FedAvg 0.519 (attack detection 99.5% and 92.7%), and one epoch of local fine-tuning gives an own-traffic difference from the local model of 0.005 ± 0.098 at a common-test macro-F1 of 0.432 ± 0.034.

**Fixed versus redrawn host partitions.** With the host order fixed (large hosts first), five repetitions gave FedAvg 0.770 ± 0.031; with redrawn host partitions, 0.720 ± 0.099.

## S5. N-BaIoT per Device and Personalisation

Table S3 gives every N-BaIoT setting, including the algorithms (the main text shows the references and partitions only), and Table S4 each device's own-traffic scores. On the device partition, FedProx, FedAdam and FedLC differ from FedAvg by +0.013 [−0.008, +0.033], +0.028 [−0.012, +0.068] and +0.011 [−0.004, +0.027], SCAFFOLD differs from FedAvg with SGD by +0.017 [−0.002, +0.037], and FedAvg with SGD differs from Adam-based FedAvg by −0.052 [−0.076, −0.028], p = 0.004, uncorrected.

<!--TABLE:NB_full-->
| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Benign FPR (%) |
|---|---|---|---|---|
| Centralised | 0.997 ± 0.001 | 0.001 | 0.981 ± 0.038 | 0.17 ± 0.14 |
| Local only (mean over devices) | 0.705 ± 0.010 | 0.008 | 0.691 ± 0.007 | – |
| FedAvg, one client per device | 0.893 ± 0.019 | 0.015 | 0.881 ± 0.028 | 1.03 ± 0.47 |
| FedProx, per device | 0.905 ± 0.009 | 0.007 | 0.904 ± 0.008 | 1.21 ± 0.53 |
| FedAdam, per device | 0.920 ± 0.024 | 0.019 | 0.907 ± 0.020 | 1.56 ± 1.18 |
| FedLC, per device | 0.904 ± 0.019 | 0.016 | 0.890 ± 0.021 | 1.17 ± 0.24 |
| FedAvg (SGD), per device | 0.840 ± 0.007 | 0.006 | 0.832 ± 0.010 | 1.59 ± 0.88 |
| SCAFFOLD (SGD), per device | 0.858 ± 0.019 | 0.015 | 0.849 ± 0.020 | 1.23 ± 0.60 |
| FedAvg, IID, N = 9 | 0.965 ± 0.016 | 0.013 | 0.965 ± 0.016 | 0.39 ± 0.07 |
| FedAvg, α = 0.5, N = 9 | 0.940 ± 0.025 | 0.020 | 0.924 ± 0.034 | 0.40 ± 0.11 |
| FedAvg, α = 0.1, N = 9 | 0.784 ± 0.103 | 0.083 | 0.755 ± 0.084 | 0.29 ± 0.16 |

<!--TABLE:NB_dev-->
| Device | Training records | Classes | Local model, own traffic | FedAvg, own traffic | Difference | Local model, all test traffic |
|---|---|---|---|---|---|---|
| Danmini Doorbell | 17,235 | 9 | 0.971 | 0.951 | −0.020 | 0.827 |
| Ecobee Thermostat | 13,342 | 9 | 0.972 | 0.905 | −0.067 | 0.900 |
| Ennio Doorbell | 3,116 | 4 | 0.938 | 0.928 | −0.010 | 0.234 |
| Philips B120N10 Baby Monitor | 18,992 | 9 | 0.985 | 0.953 | −0.033 | 0.849 |
| Provision PT 737E Security Camera | 13,007 | 9 | 0.976 | 0.801 | −0.174 | 0.756 |
| Provision PT 838 Security Camera | 13,468 | 9 | 0.979 | 0.781 | −0.198 | 0.759 |
| Samsung SNH 1011 N Webcam | 3,517 | 4 | 0.984 | 0.907 | −0.077 | 0.229 |
| SimpleHome XCS7 1002 WHT Security Camera | 13,931 | 9 | 0.980 | 0.896 | −0.084 | 0.902 |
| SimpleHome XCS7 1003 WHT Security Camera | 13,619 | 9 | 0.978 | 0.910 | −0.068 | 0.893 |

Table S5 compares personalisation baselines on the device-level partitions. Each site adapts the validation-selected federated model of the same run to its own training data for one, two or five local epochs, with the same Adam learning rate, a ten times smaller one, or the same one plus a proximal term (μ/2)‖w − w_g‖² towards the federated model (μ = 0.1). All settings were fixed in advance, and no adapted model is selected on test data. The own-traffic difference is taken against the site's local-only model (mean over repetitions on TON-IoT, over devices on N-BaIoT); the common-test-set score is the mean over sites. The alert columns score the two-model combination: the federated and the adapted model run side by side, and an alert is raised when either predicts an attack; for the fine-tuning rows the own-traffic difference and the common-test macro-F1 are those of the adapted model alone. The table comes from the personalisation re-run, so its federated-model detection rates (e.g., N-BaIoT 100.0%, TON-IoT 91.1%) differ slightly from those of the main runs quoted in the paper (99.9% and 96.5%).

<!--TABLE:pers-->
| Dataset | Model | Local epochs | Own-traffic macro-F1 minus local model | Common test macro-F1 | Own-traffic attack detection (%) | Own-traffic benign FPR (%) | Common test benign FPR (%) |
|---|---|---|---|---|---|---|---|
| TON-IoT | Federated model alone | – | −0.130 ± 0.052 | 0.710 ± 0.096 | 91.1 | 2.9 | 1.9 |
| TON-IoT | Local model alone | – | 0 | 0.530 ± 0.040 | 99.7 | 2.6 | – |
| TON-IoT | Fine-tuning, same lr (alert columns: with the federated model) | 1 | 0.008 ± 0.059 | 0.551 ± 0.027 | 99.9 | 4.1 | 16.4 |
| TON-IoT | Fine-tuning, same lr (alert columns: with the federated model) | 2 | 0.014 ± 0.060 | 0.541 ± 0.048 | 99.9 | 4.0 | 16.5 |
| TON-IoT | Fine-tuning, same lr (alert columns: with the federated model) | 5 | 0.013 ± 0.031 | 0.530 ± 0.038 | 99.9 | 4.2 | 17.3 |
| TON-IoT | Fine-tuning, lr / 10 (alert columns: with the federated model) | 1 | 0.006 ± 0.040 | 0.613 ± 0.050 | 99.9 | 3.9 | 12.1 |
| TON-IoT | Fine-tuning, lr / 10 (alert columns: with the federated model) | 2 | 0.018 ± 0.034 | 0.598 ± 0.049 | 99.9 | 3.8 | 14.0 |
| TON-IoT | Fine-tuning, lr / 10 (alert columns: with the federated model) | 5 | 0.026 ± 0.056 | 0.584 ± 0.053 | 99.9 | 3.9 | 15.4 |
| TON-IoT | Proximal fine-tuning (μ = 0.1) (alert columns: with the federated model) | 1 | 0.008 ± 0.039 | 0.590 ± 0.049 | 99.9 | 4.0 | 15.5 |
| TON-IoT | Proximal fine-tuning (μ = 0.1) (alert columns: with the federated model) | 2 | 0.009 ± 0.043 | 0.586 ± 0.048 | 99.9 | 3.9 | 15.6 |
| TON-IoT | Proximal fine-tuning (μ = 0.1) (alert columns: with the federated model) | 5 | 0.011 ± 0.034 | 0.585 ± 0.052 | 99.9 | 4.0 | 16.5 |
| N-BaIoT | Federated model alone | – | −0.073 ± 0.045 | 0.900 ± 0.015 | 100.0 | 1.5 | 0.9 |
| N-BaIoT | Local model alone | – | 0 | 0.705 ± 0.010 | 99.9 | 0.8 | – |
| N-BaIoT | Fine-tuning, same lr (alert columns: with the federated model) | 1 | −0.034 ± 0.020 | 0.844 ± 0.038 | 100.0 | 1.6 | 2.8 |
| N-BaIoT | Fine-tuning, same lr (alert columns: with the federated model) | 2 | −0.032 ± 0.015 | 0.840 ± 0.027 | 100.0 | 1.7 | 3.2 |
| N-BaIoT | Fine-tuning, same lr (alert columns: with the federated model) | 5 | −0.032 ± 0.014 | 0.832 ± 0.024 | 100.0 | 1.7 | 3.6 |
| N-BaIoT | Fine-tuning, lr / 10 (alert columns: with the federated model) | 1 | −0.023 ± 0.017 | 0.878 ± 0.021 | 100.0 | 1.5 | 1.6 |
| N-BaIoT | Fine-tuning, lr / 10 (alert columns: with the federated model) | 2 | −0.019 ± 0.014 | 0.880 ± 0.020 | 100.0 | 1.6 | 1.7 |
| N-BaIoT | Fine-tuning, lr / 10 (alert columns: with the federated model) | 5 | −0.015 ± 0.015 | 0.880 ± 0.020 | 100.0 | 1.6 | 1.8 |
| N-BaIoT | Proximal fine-tuning (μ = 0.1) (alert columns: with the federated model) | 1 | −0.041 ± 0.028 | 0.845 ± 0.026 | 100.0 | 1.6 | 2.0 |
| N-BaIoT | Proximal fine-tuning (μ = 0.1) (alert columns: with the federated model) | 2 | −0.036 ± 0.015 | 0.841 ± 0.022 | 100.0 | 1.6 | 2.8 |
| N-BaIoT | Proximal fine-tuning (μ = 0.1) (alert columns: with the federated model) | 5 | −0.025 ± 0.012 | 0.850 ± 0.020 | 100.0 | 1.7 | 2.1 |

**Diagnostics of the own-traffic reversal (three repetitions).** The blocked split pools each dataset's records (TON-IoT: the training, validation and test records of the paper's pipeline, with its feature selection and scaling; N-BaIoT: the same 3% sample, deduplicated as in S1-A and re-scaled on the new training split) and cuts every (host or device, class) group in file order into a first 65% for training, a 5% gap, 12.5% for validation, a 5% gap and the last 12.5% for testing; groups too small for a block stay in training, and a guard removes validation and test records identical (features and label) to a training record. On N-BaIoT the file order is the capture order; the TON-IoT dataset does not document the order of its network file, so there the split is blocked but not necessarily temporal. The random control cuts the same groups with the same fractions after a random permutation. FedAvg and local-only models are trained as in the paper (host partition redrawn in each repetition; one client per device). Under the blocked split, FedAvg − local on own traffic is −0.049 [−0.087, −0.011] on TON-IoT (FedAvg worse in 7 of 12 site–repetition pairs) and −0.029 [−0.111, +0.053] on N-BaIoT (19 of 27); under the random control it is −0.126 [−0.354, +0.102] (11 of 12) and −0.055 [−0.079, −0.031] (24 of 27). On the common test set of the blocked split, FedAvg − local is 0.221 ± 0.174 (TON-IoT) and 0.148 ± 0.048 (N-BaIoT). The label-prior correction adds log π_site,c − log π_pooled,c (training class shares) to the federated model's log-probabilities on a site's traffic and masks the classes absent from the site's training data; it needs no training. On the original random split, masking alone moves the own-traffic difference from the local model to −0.124 [−0.237, −0.011] (TON-IoT) and −0.074 [−0.087, −0.061] (N-BaIoT) and the full correction to −0.140 [−0.310, +0.030] and −0.073 [−0.086, −0.060], against −0.125 [−0.238, −0.012] and −0.074 [−0.087, −0.061] without correction (FedAvg re-runs of the first three repetitions, scored against the local models of the main runs, which share their partitions); under the blocked split the corrected difference is −0.044 [−0.092, +0.004] and −0.030 [−0.119, +0.059]. The correction is designed to improve accuracy rather than macro-F1. Paired over repetitions, the blocked-split difference minus the random-control difference is, on TON-IoT, +0.076 [−0.123, +0.276], p = 0.240, uncorrected, and on N-BaIoT, +0.026 [−0.035, +0.086], p = 0.211, uncorrected; with three repetitions neither reduction is significant. On the paper's TON-IoT host partition, own-traffic attack detection is 99.7% for the local model and 96.5% for FedAvg, at benign FPRs of 2.6% and 3.4%; a re-run of the same FedAvg configuration detected 91.1%, so we do not interpret the detection gap.

## S6. Normalisation and Number of Clients

We measure the swing as the mean absolute change in validation macro-F1 between consecutive rounds over rounds 11–50. Two run sets are reported. The first (three repetitions, first rows of Tables S6 and S7) trains with server-averaged statistics and additionally scores every round after replacing the running statistics by those of a batch of pooled training records (a diagnostic that needs pooled data): this reduces the swing from 0.196 to 0.009 for the centralised TON-IoT model and from 0.151 to 0.017 for FedAvg at α = 0.5 (group normalisation: 0.012 and 0.016); the batch-normalised centralised TON-IoT model reaches 0.568 ± 0.380 at the final round (standard deviation 0.153, against 0.890 ± 0.011 and 0.009 with group normalisation). The second set (five repetitions, used in the paper and in the last rows of the tables) scores the same models both with server-averaged statistics and after federated re-estimation; with server-averaged statistics the swing is 0.110 and the final-round score 0.634 ± 0.339 (standard deviation 0.273), after re-estimation 0.018 and 0.855 ± 0.037 (0.030). The two run sets differ in their swing because individual runs differ in how often the running statistics drift. On UNSW-NB15 the swing is small with either kind of statistics (0.010 and 0.007), and batch normalisation with federated re-estimation differs from group normalisation by −0.002 [−0.013, +0.009], p = 0.657, uncorrected (α = 0.5; with server-averaged statistics +0.003 [−0.006, +0.011], p = 0.457, uncorrected).

Tables S6 and S7 also report FedAvg with N = 2, 8 and 16 clients at α = 0.5. With the same data split over more clients, each client holds less data and a narrower label mix, and macro-F1 falls on TON-IoT from 0.863 ± 0.030 (N = 2) to 0.840 ± 0.043 (N = 4), 0.802 ± 0.050 (N = 8) and 0.757 ± 0.045 (N = 16); on UNSW-NB15 the trend is in the same direction but within the wide intervals (0.474 ± 0.124 at N = 2, 0.391 ± 0.042 at N = 8, 0.372 ± 0.101 at N = 16). The N = 4 row has five repetitions and the others three, so these contrasts are unpaired; the same holds for Fig. 2, which shows the centralised batch-normalised model of the three-repetition set. The traffic per round grows linearly with N (3.04 MB per round at N = 16).

<!--TABLE:TON_extra-->
| Setting | Test macro-F1, validation-selected round | Test macro-F1, final round | MB per round |
|---|---|---|---|
| FedAvg, N = 4, α = 0.5 | 0.840 ± 0.043 | 0.829 ± 0.040 | 0.76 |
| FedAvg, N = 2 | 0.863 ± 0.030 | 0.814 ± 0.108 | 0.38 |
| FedAvg, N = 8 | 0.802 ± 0.050 | 0.800 ± 0.042 | 1.52 |
| FedAvg, N = 16 | 0.757 ± 0.045 | 0.758 ± 0.045 | 3.04 |
| FedAvg, BatchNorm, α = 0.5, server-averaged statistics (three repetitions) | 0.865 ± 0.010 | 0.739 ± 0.087 | 0.77 |
| FedAvg, BatchNorm, α = 0.1, server-averaged statistics (three repetitions) | 0.667 ± 0.242 | 0.507 ± 0.165 | 0.77 |
| Centralised, BatchNorm | 0.865 ± 0.049 | 0.568 ± 0.380 | – |
| FedAvg, BatchNorm, α = 0.5, server-averaged statistics (five repetitions) | 0.862 ± 0.020 | 0.634 ± 0.339 | – |
| FedAvg, BatchNorm, α = 0.1, server-averaged statistics (five repetitions) | 0.733 ± 0.122 | 0.543 ± 0.087 | – |
| FedAvg, BatchNorm, α = 0.5, federated re-estimation (five repetitions) | 0.872 ± 0.037 | 0.855 ± 0.037 | – |
| FedAvg, BatchNorm, α = 0.1, federated re-estimation (five repetitions) | 0.753 ± 0.114 | 0.707 ± 0.122 | – |

<!--TABLE:UNSW_extra-->
| Setting | Test macro-F1, validation-selected round | Test macro-F1, final round | MB per round |
|---|---|---|---|
| FedAvg, N = 4, α = 0.5 | 0.433 ± 0.040 | 0.414 ± 0.046 | 0.76 |
| FedAvg, N = 2 | 0.474 ± 0.124 | 0.445 ± 0.153 | 0.38 |
| FedAvg, N = 8 | 0.391 ± 0.042 | 0.388 ± 0.039 | 1.52 |
| FedAvg, N = 16 | 0.372 ± 0.101 | 0.364 ± 0.099 | 3.04 |
| FedAvg, BatchNorm, α = 0.5, server-averaged statistics (three repetitions) | 0.423 ± 0.084 | 0.400 ± 0.104 | 0.77 |
| FedAvg, BatchNorm, α = 0.1, server-averaged statistics (three repetitions) | 0.305 ± 0.250 | 0.259 ± 0.293 | 0.77 |
| Centralised, BatchNorm | 0.519 ± 0.044 | 0.470 ± 0.118 | – |
| FedAvg, BatchNorm, α = 0.5, server-averaged statistics (five repetitions) | 0.435 ± 0.048 | 0.424 ± 0.059 | – |
| FedAvg, BatchNorm, α = 0.1, server-averaged statistics (five repetitions) | 0.347 ± 0.120 | 0.299 ± 0.143 | – |
| FedAvg, BatchNorm, α = 0.5, federated re-estimation (five repetitions) | 0.431 ± 0.047 | 0.430 ± 0.060 | – |
| FedAvg, BatchNorm, α = 0.1, federated re-estimation (five repetitions) | 0.349 ± 0.133 | 0.324 ± 0.153 | – |

## S7. Differential Privacy

The DP-SGD code without clipping or noise is close to non-private FedAvg on the same schedule on TON-IoT (0.760 ± 0.064 against 0.735 ± 0.051; the two differ in dropout and in Poisson versus shuffled batches), and clipping alone costs little in macro-F1 (0.743 ± 0.054), although Password already loses (Fig. 3). On UNSW-NB15 (Table S8, Fig. S1) noise costs 5 to 7 points against same-schedule non-private training for ε between 1.3 and 18.5 (the smaller clients give larger ε for the same noise); the backdoor and DoS classes, weak even without noise, and shellcode (0.39 without noise) disappear, generic and reconnaissance traffic lose up to about ten points, and benign, exploit and fuzzer traffic at most five; the analysis and worms classes are near zero in every DP configuration. On TON-IoT the rare MITM class (727 training records) is detected poorly and erratically under noise: its F1 is 0.24 without noise, 0.05 and 0.05 at σ = 0.2 and 0.3, 0.13 at σ = 0.6 and 0.07 at σ = 1, and zero from σ = 1.5. Against non-private FedAvg on the same schedule rather than the DP code without clipping or noise, 4 attack classes lose at least ten points at σ = 0.6 and 5 at σ = 2.5. At σ = 0.2 (ε ≈ 402) macro-F1 is 0.681 ± 0.056, 5 points below non-private training, and at σ = 0.3 0.656 ± 0.063. The full budget has no clipping-only run, so the effect of clipping alone at 50 × 2 (relevant for Password) is not known.

DP tuning on TON-IoT at σ = 1 (two repetitions each, on seeds of the reported repetitions, so these test scores are not independent of the tuning; the privacy cost of tuning is not accounted): clipping norms of 0.5, 2 and 4 instead of 1 give 0.649, 0.667 and 0.648, so the best clipping norm lies inside the grid, but C = 2 scores slightly above the C = 1 of the main runs, so the reported DP cost may be overstated by one to two points; learning rates of 0.001 and 0.006 give 0.604 and 0.652; and doubling the schedule to 40 rounds gives 0.659 at ε = 4.4, still 11 points below the same code without noise or clipping (0.772). The DP configuration was tuned on TON-IoT at α = 0.5 only; under severe skew, small clients would have a sampling rate close to one and a much larger ε for the same noise.

Optimiser and batch size at matched ε (TON-IoT, α = 0.5, 20 × 1, three repetitions on the seeds of the σ = 1 runs, so ε equals theirs). Because most of the cost appears at negligible noise, we checked whether it comes from Adam, which rescales every coordinate of the noisy gradient by its running magnitude, or from the batch size, the main lever of DP-SGD. DP-SGD with SGD and momentum 0.9 (state reset every round, as for Adam; learning rate 0.03 or 0.1, chosen on validation: 0.1) reaches 0.652 ± 0.040 (learning rate 0.03: 0.646 ± 0.036; 0.1: 0.652 ± 0.040), a difference from the Adam runs of +0.004 [−0.019, +0.027], p = 0.562, uncorrected. Adam with a four times larger expected batch (B = 1024), with σ recalibrated in each repetition to the ε of the B = 256 run (mean σ = 1.64, ε = 3.1), reaches 0.643 ± 0.026, a difference of −0.006 [−0.025, +0.014], p = 0.335, uncorrected. Against non-private training on the same schedule (0.735 ± 0.051), the costs are 8 points (SGD with momentum), 9 (B = 1024) and 9 (Adam, B = 256). The privacy cost of choosing between these configurations is not accounted. These checks use the short schedule only; the selected SGD learning rate is the larger of the two tried, and the larger batch keeps the number of epochs and therefore takes four times fewer steps.

<!--TABLE:UNSW_dp-->
| Setting | Noise σ | ε (max over clients, δ = 10⁻⁵) | Test macro-F1, validation-selected | Test macro-F1, final |
|---|---|---|---|---|
| Non-private FedAvg, same schedule (minibatch Adam, dropout) | – | – | 0.408 ± 0.056 | 0.392 ± 0.080 |
| DP-SGD code, no clipping, no noise | 0 | ∞ | 0.437 ± 0.052 | 0.438 ± 0.054 |
| DP-SGD code, clipping C = 1, no noise | 0 | ∞ | 0.444 ± 0.024 | 0.443 ± 0.026 |
| DP-SGD, C = 1 | 0.6 | 18.52 (17.04–20.33) | 0.363 ± 0.028 | 0.360 ± 0.039 |
| DP-SGD, C = 1 | 1 | 5.27 (4.75–5.91) | 0.348 ± 0.014 | 0.343 ± 0.022 |
| DP-SGD, C = 1 | 1.5 | 2.62 (2.37–2.94) | 0.343 ± 0.006 | 0.334 ± 0.025 |
| DP-SGD, C = 1 | 2.5 | 1.33 (1.21–1.49) | 0.336 ± 0.006 | 0.326 ± 0.028 |
