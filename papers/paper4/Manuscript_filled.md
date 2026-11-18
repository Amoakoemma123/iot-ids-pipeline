# Paper 4 v2 manuscript (double-brace placeholders are filled from values.json by build.py)

## Title

**Federated Intrusion Detection for IoT Networks Revisited: Device-Level Partitions, Tuned Baselines and Record-Level Differential Privacy**

## Abstract

Federated learning (FL) is widely proposed for training intrusion detection systems (IDS) across IoT sites without sharing traffic, but its measured benefit depends on round selection, client partitions, baseline tuning and whether privacy budgets match the training. We re-evaluate FL for a compact 1D-CNN IDS on TON-IoT, UNSW-NB15 and N-BaIoT with redrawn partitions, validation-based round selection, tuned baselines, same-recipe centralised and local-only references, and record-level DP-SGD with a matching accountant, partitioning TON-IoT by destination host and N-BaIoT by physical device besides Dirichlet label skew. On the common test set, FedAvg beats local-only training by +0.190 macro-F1 (TON-IoT hosts) and +0.187 (N-BaIoT devices). On each site's own test traffic, however, the local model is better in 17 of 20 site–repetition pairs on TON-IoT and 43 of 45 on N-BaIoT under random record splits (exploratory); under a blocked split it is smaller (by 61% and 47%, not significantly) but persists on TON-IoT. Local fine-tuning brings TON-IoT sites to parity within our resolution and halves the N-BaIoT gap, at a cost in detecting other sites' attacks. FedProx, FedAdam, SCAFFOLD and FedLC do not improve on tuned FedAvg after Holm correction. On TON-IoT, batch normalisation with server-averaged statistics makes the model swing between rounds; federated re-estimation removes the swing. At the full training budget, record-level DP-SGD at ε ≈ 2.9 costs 17 macro-F1 points on TON-IoT, and 7 of 9 attack classes lose ten points or more.

**Index Terms:** convolutional neural networks, differential privacy, federated learning, Internet of Things, intrusion detection, non-IID data

## I. Introduction

Intrusion detection systems (IDS) based on machine learning need traffic from many sites to learn the variety of attacks an IoT deployment faces [1], but operators are often unable or unwilling to pool raw traffic [2]. Federated learning (FL) trains a shared model by exchanging model updates instead of data [3], and a growing literature applies it to intrusion detection [2], [4]–[6]. IoT sites observe different devices and attack mixes, so client data are highly heterogeneous [7], [8], and the shared updates should themselves leak little about individual records, which calls for differential privacy (DP) [9], [10].

How much FL helps an IDS depends on evaluation choices: selecting the reported round on the test set inflates scores; one random client partition hides how much a result depends on the split; label skew alone may not represent sites that observe different devices; untuned baselines or a centralised reference with a different recipe bias comparisons; and a privacy budget means little unless the implemented mechanism satisfies it. Rather than audit published studies for these choices, we show how much each matters in a controlled re-evaluation of a compact one-dimensional convolutional network (CNN) on TON-IoT [11], [12], UNSW-NB15 [13] and N-BaIoT [14]. The contributions are:

1. An evaluation protocol for FL-IDS: partitions redrawn in each of five repetitions, validation-based round selection, every algorithm tuned, and paired comparisons with a family of primary comparisons corrected for multiple testing.
2. Matched centralised and local-only references, with every site scored on the common test set and on its own test traffic: under random record splits, federation improves detection of other sites' attacks, but a site's own model is better on its own traffic (exploratory). We quantify how local adaptation trades the two and test the disagreement with a blocked split and a label-prior correction.
3. Two device-level partitions, by destination host on TON-IoT and by physical device on N-BaIoT, compared with Dirichlet partitions of similar label skew.
4. An algorithm comparison on three datasets, every algorithm tuned at a matched client learning rate: none of FedProx [15], FedAdam [16], SCAFFOLD [17] and FedLC [18] improves significantly on FedAvg after correction.
5. Batch normalisation with server-averaged running statistics makes the evaluated model swing between rounds; group normalisation [19], recommended by Hsieh et al. [20], avoids this, and federated re-estimation of the statistics removes the swing.
6. Record-level DP-SGD with an RDP accountant [9], [21], [22], sanity runs and a per-class account of the privacy cost at a short and the full training budget.

## II. Related Work

Surveys of FL-based intrusion detection identify heterogeneous client data, communication cost and privacy–utility trade-offs as open problems [2], [5]. DÏoT trains device-type-specific models federatedly [4]; Rey et al. compare federated, local and centralised malware detection on N-BaIoT [23]; Mármol Campos et al. evaluate FL-IDS on ToN_IoT under non-IID splits [6]; others compare aggregation algorithms [24] or study consumer IoT [25], unsupervised models [26] and knowledge distillation [27]. Poisoning [28] is outside our scope.

FedAvg degrades when client label distributions differ [29]; FedProx [15], SCAFFOLD [17], FedAdam [16] and FedLC [18] counter this with a proximal term, control variates, an adaptive server optimiser and logit calibration, but no algorithm dominates under Dirichlet label skew [7], [8]. Batch normalisation fails under skewed data [20] because of mismatched statistics [30]. Personalised FL keeps normalisation statistics local (FedBN [31]) or regularises client models towards the shared one (Ditto [32]), also for IoT intrusion detection [33], [34]; its premise, that a client's own model fits its own distribution best, is what we measure for IDS under device-level partitions.

DP-SGD clips per-example gradients and adds Gaussian noise [9], with Rényi DP accounting [21], [22]. Client-level DP [10], [35] needs many clients, which a small consortium lacks, so each client applies record-level DP. DP has been applied to federated industrial IoT IDS [36], [37] and hurts under-represented classes most [38].

## III. Data and Evaluation Protocol

### A. Datasets

TON-IoT contains network flows [12] from an IoT/IIoT testbed [11] with nine attack classes and benign traffic; UNSW-NB15 contains enterprise-testbed flows with nine attack categories [13]. Both pass through our corrected pipeline, released with this paper: identifier columns are removed, exact duplicates are removed after all column drops, the data are split 70/15/15 (stratified), and feature selection and min–max scaling are fitted on the training split, leaving 25 features for TON-IoT (128,484, 27,533 and 27,533 training, validation and test records) and 10 for UNSW-NB15 (74,167, 15,894, 15,894). On TON-IoT the features include the source and destination ports, which can act as testbed shortcuts. N-BaIoT [14] contains benign, BASHLITE and Mirai traffic of nine commercial IoT devices, described by 115 Kitsune stream statistics [39]. We use a uniform 3% sample of every device–class file, deduplicated at the model's 32-bit precision and without the two degenerate BASHLITE flood classes, split 70/15/15 by device and class with a cross-split duplicate guard (9 classes; 110,227, 22,503 and 22,453 records; Supplement S1). Preprocessing is fitted centrally, a simplification a deployment would replace with federated statistics.

### B. Client partitions

**Dirichlet label skew.** On TON-IoT and UNSW-NB15 the training split is distributed over N = 4 clients (2, 8 and 16 in a scaling study), at random (IID) or with per-class Dirichlet proportions, α ∈ {1.0, 0.5, 0.1} [7], [8]. At α = 0.5 the two largest classes make up 56% of a TON-IoT client's data on average, at α = 0.1 77%.

**Hosts on TON-IoT.** Each destination host is assigned, as a whole, to one of four clients, so that a client sees all traffic to the devices it monitors; the assignment is redrawn in every repetition. Its label skew is similar to α = 0.5 (two largest classes 52% on average), but the clients differ in the devices behind the traffic and, because traffic is concentrated on few hosts, in size (23,741 to 52,783 records). A client's own test traffic is the test traffic to its hosts (5,175 to 11,147 records with 7 to 10 classes per site). UNSW-NB15 has no usable addresses.

**Devices on N-BaIoT.** Every physical device is one client (N = 9), and its own test traffic is its test split (584 to 3,949 records, 4 to 9 classes). The label skew is milder than at α = 0.5 (two largest classes 47% on average). As references we distribute N-BaIoT over nine clients at random and with α = 0.5 and 0.1. Each repetition draws a new partition (except the fixed device partition), initialisation and batch order.

### C. Model, training and selection

The CNN has two convolutional blocks (32 and 64 filters, kernel 3, max-pooling) with group normalisation [19], global average pooling and two dense layers (128 and 64 units, dropout 0.3), 23,754 parameters (95.0 kB). Federated training runs for 50 rounds of two local epochs (batch size 256) with size-weighted aggregation. We report the test macro-F1 at the round with the highest validation macro-F1 (pooled validation split; Supplement S1) and at the final round; the test set never influences any choice. The centralised reference trains the same model with the same optimiser and learning rate on the pooled data for 100 epochs, checkpointed and selected every two epochs; local-only models are trained in the same way on each client's data alone. Every model is scored on the common test set and, under device-level partitions, on each client's own test traffic: macro-F1 over the classes present in it, attack detection rate and benign false-positive rate (FPR).

### D. Tuning and statistics

Hyperparameters are tuned on the validation set in separate tuning repetitions (α = 0.5 on TON-IoT and UNSW-NB15, device partition on N-BaIoT): FedAvg's Adam learning rate first, then FedProx's μ, FedAdam's server learning rate and FedLC's τ at that client learning rate, and the SGD learning rate of SCAFFOLD and of FedAvg with SGD (Supplement S1). Only FedProx's μ on TON-IoT (smallest value) lies at an edge of its grid; because μ → 0 recovers FedAvg, this does not affect the FedProx comparison. Core results use five repetitions; the scaling study, the batch-normalisation runs with server-averaged statistics only, the 100-round check and the DP study use three, and contrasts between run sets of different sizes are unpaired. We report means with t-based 95% confidence intervals (± half-width). Differences between methods on the same partition are paired over repetitions and reported as mean difference [95% CI]; differences between partitions use Welch's test. For each dataset a family of primary comparisons, declared before the N-BaIoT, personalisation and no-port experiments, is Holm-corrected [40] (Supplement S2, Table S1); † marks significance after correction. All other comparisons are exploratory, with uncorrected p-values.

## IV. Algorithms and Differential Privacy

FedAvg [3] trains locally with Adam (reset each round). FedProx [15] adds (μ/2)‖w − w_g‖² to the local loss. FedAdam [16] applies a server Adam step (β₁ = 0.9, β₂ = 0.99, τ = 10⁻³) to the averaged update; only its server learning rate is tuned. SCAFFOLD [17] uses local SGD with control variates (option II; model updates size-weighted, control variates averaged uniformly) and doubles communication; FedAvg with SGD is its reference. FedLC [18] trains on calibrated logits z_c − τ·n_{k,c}^{−1/4}, with n_{k,c} the number of class-c records of client k (set to 1 for classes the client does not hold); evaluation uses the uncalibrated output.

Each client trains with DP-SGD [9]: batches are Poisson-sampled with rate q = B/n_k, per-example gradients are clipped to L2 norm C = 1, Gaussian noise with standard deviation σC is added to their sum, and Adam takes the step. The model uses group normalisation and no dropout. The privacy loss of each client's records is accounted with RDP for the subsampled Gaussian mechanism [21], [22] over all local steps and converted to (ε, δ) with δ = 10⁻⁵; we report the maximum over clients. The unit is one training record: protecting a device's traffic as a whole would need group privacy or user-level DP, and the validation records used for round selection are not protected. DP runs use α = 0.5, mostly with 20 rounds of one local epoch (per-example gradients are costly) and once with the full 50 rounds of two epochs; sanity runs use the same code without noise, with and without clipping.

## V. Results

### A. Federation under label skew

Table I reports TON-IoT (UNSW-NB15 in Supplement S3), and Fig. 1 shows the paired comparisons. The centralised TON-IoT model reaches 0.899 ± 0.012. FedAvg differs from it by −0.042 [−0.059, −0.025]† with IID clients, −0.059 [−0.101, −0.016] at α = 0.5 and −0.185 [−0.286, −0.084] at α = 0.1; only the IID gap is significant after correction, but the gap grows with skew (exploratory slope over the four skew levels coded 0–3, tested over repetitions: −0.046 per level, p = 0.019; UNSW-NB15: −0.036, p = 0.030). On the common test set FedAvg improves on local training by +0.080 [+0.043, +0.117] at α = 0.5 and +0.286 [+0.194, +0.379]† at α = 0.1, largely because a local model cannot detect classes its client never saw (Supplement S4). For FedAvg, selected and final rounds are close, but the selected rounds cluster near the end of the budget (round 47 of 50 at α = 0.5); 100 rounds narrow the α = 0.5 gap and leave all gaps within their intervals (Supplement S4). Without the port features TON-IoT scores fall (centralised: −0.143) and federation costs more.

<!--TABLE:TON_main-->
| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Selected round | Benign FPR (%) |
|---|---|---|---|---|---|
| Centralised (same recipe) | 0.899 ± 0.012 | 0.010 | 0.890 ± 0.011 | 41 | 1.94 ± 0.74 |
| Local only, α = 0.5 | 0.760 ± 0.028 | 0.022 | 0.743 ± 0.031 | – | – |
| Local only, α = 0.1 | 0.428 ± 0.070 | 0.056 | 0.407 ± 0.062 | – | – |
| FedAvg, IID | 0.857 ± 0.009 | 0.007 | 0.845 ± 0.030 | 44 | 2.61 ± 0.39 |
| FedAvg, α = 1.0 | 0.872 ± 0.011 | 0.009 | 0.868 ± 0.014 | 48 | 2.81 ± 0.34 |
| FedAvg, α = 0.5 | 0.840 ± 0.043 | 0.034 | 0.829 ± 0.040 | 47 | 2.40 ± 0.30 |
| FedAvg, α = 0.1 | 0.714 ± 0.104 | 0.084 | 0.680 ± 0.144 | 41 | 3.12 ± 1.14 |
| FedProx, α = 0.5 | 0.845 ± 0.031 | 0.025 | 0.833 ± 0.034 | 49 | 2.41 ± 0.99 |
| FedProx, α = 0.1 | 0.704 ± 0.154 | 0.124 | 0.652 ± 0.162 | 42 | 2.71 ± 1.01 |
| FedAdam, α = 0.5 | 0.757 ± 0.025 | 0.020 | 0.742 ± 0.025 | 44 | 3.24 ± 0.39 |
| FedAdam, α = 0.1 | 0.668 ± 0.117 | 0.094 | 0.630 ± 0.109 | 40 | 4.22 ± 1.21 |
| SCAFFOLD (SGD), α = 0.5 | 0.715 ± 0.017 | 0.013 | 0.691 ± 0.044 | 36 | 4.55 ± 0.52 |
| SCAFFOLD (SGD), α = 0.1 | 0.657 ± 0.019 | 0.015 | 0.597 ± 0.086 | 40 | 6.29 ± 3.60 |
| FedAvg (SGD), α = 0.5 | 0.694 ± 0.025 | 0.020 | 0.688 ± 0.031 | 46 | 4.19 ± 0.39 |
| FedAvg (SGD), α = 0.1 | 0.640 ± 0.077 | 0.062 | 0.615 ± 0.081 | 40 | 4.60 ± 1.31 |
| FedLC, α = 0.5 | 0.851 ± 0.024 | 0.019 | 0.829 ± 0.037 | 48 | 2.94 ± 0.49 |
| FedLC, α = 0.1 | 0.697 ± 0.109 | 0.088 | 0.669 ± 0.094 | 29 | 2.96 ± 1.22 |

### B. Algorithms

No drift-reducing algorithm improves on FedAvg on any dataset after correction (Fig. 1). On TON-IoT FedProx differs from FedAvg by +0.005 [−0.037, +0.048] and FedLC by +0.011 [−0.040, +0.062] at α = 0.5; FedAdam trails it (−0.083 [−0.118, −0.049]†, the only algorithm difference significant after correction; at α = 0.1 and on the host partition, −0.046 [−0.073, −0.018] and −0.076 [−0.132, −0.020], not after correction). As only its server learning rate was tuned, at FedAvg's client learning rate, this deficit may partly reflect the tuning design; on N-BaIoT FedAdam has the best point estimate (+0.028 [−0.012, +0.068]). SCAFFOLD does not improve on FedAvg with SGD (+0.020 [−0.001, +0.042]) and trails Adam-based FedAvg (−0.125 [−0.165, −0.085], p < 0.001, uncorrected). On UNSW-NB15 FedProx (α = 0.1) and FedLC (α = 0.5) are better before correction only; FedLC's gain (+0.063 [+0.015, +0.110]) comes with a benign FPR of 17.4% against 11.8%, and SCAFFOLD is erratic under severe skew (Supplement S3). No gain survives correction, but only in 12 of the 24 algorithm comparisons does the 90% CI lie within ±0.05 (equivalence at that margin); the others, including FedLC on UNSW-NB15, are inconclusive.

### C. Device-level partitions

**Cost against pooled training.** Under the TON-IoT host partition (Table II) FedAvg reaches 0.720 ± 0.099, −0.179 [−0.284, −0.074] against the centralised model, with a larger spread over repetitions than at α = 0.5 (SD 0.080 against 0.034). Although its label skew is similar to α = 0.5, it costs about as much as α = 0.1 (exploratory Welch tests in Supplement S4). On N-BaIoT (Table III) FedAvg differs from the centralised model (0.997 ± 0.001) by −0.105 [−0.124, −0.085]† with one client per device, −0.032 [−0.048, −0.015]† with IID clients, and −0.057 [−0.081, −0.033]† and −0.214 [−0.316, −0.111]† at α = 0.5 and 0.1. Device-level partitions thus tended to cost more than Dirichlet partitions of similar label skew (α = 0.5), but not more than α = 0.1. The comparison is exploratory and confounded with client sizes; as the N-BaIoT device partition has milder label skew than α = 0.5 yet costs more, the likely mechanism is feature skew between devices, the setting FedBN [31] addresses.

<!--TABLE:TON_nat-->
| Setting | Common test macro-F1 | SD | Own-traffic macro-F1 | Own-traffic attack detection (%) | Own-traffic benign FPR (%) |
|---|---|---|---|---|---|
| Centralised | 0.899 ± 0.012 | 0.010 | – | – | – |
| Local only (mean over sites) | 0.530 ± 0.040 | 0.032 | 0.791 ± 0.068 | 99.7 ± 0.4 | 2.6 ± 0.9 |
| FedAvg | 0.720 ± 0.099 | 0.080 | 0.670 ± 0.113 | 96.5 ± 1.4 | 3.4 ± 0.9 |
| FedProx | 0.696 ± 0.073 | 0.058 | 0.646 ± 0.096 | 92.8 ± 3.1 | 2.6 ± 0.7 |
| FedAdam | 0.644 ± 0.045 | 0.036 | 0.616 ± 0.055 | 95.4 ± 3.1 | 3.7 ± 1.9 |
| FedLC | 0.734 ± 0.049 | 0.040 | 0.671 ± 0.082 | 92.4 ± 2.2 | 4.2 ± 2.5 |
| FedAvg (SGD) | 0.674 ± 0.020 | 0.016 | 0.579 ± 0.075 | 95.7 ± 1.3 | 5.2 ± 0.8 |
| SCAFFOLD (SGD) | 0.670 ± 0.027 | 0.021 | 0.572 ± 0.069 | 97.3 ± 1.6 | 6.7 ± 2.6 |

<!--TABLE:NB_main-->
| Setting | Test macro-F1, validation-selected round | SD | Test macro-F1, final round | Benign FPR (%) |
|---|---|---|---|---|
| Centralised | 0.997 ± 0.001 | 0.001 | 0.981 ± 0.038 | 0.17 ± 0.14 |
| Local only (mean over devices) | 0.705 ± 0.010 | 0.008 | 0.691 ± 0.007 | – |
| FedAvg, one client per device | 0.893 ± 0.019 | 0.015 | 0.881 ± 0.028 | 1.03 ± 0.47 |
| FedAvg, IID, N = 9 | 0.965 ± 0.016 | 0.013 | 0.965 ± 0.016 | 0.39 ± 0.07 |
| FedAvg, α = 0.5, N = 9 | 0.940 ± 0.025 | 0.020 | 0.924 ± 0.034 | 0.40 ± 0.11 |
| FedAvg, α = 0.1, N = 9 | 0.784 ± 0.103 | 0.083 | 0.755 ± 0.084 | 0.29 ± 0.16 |

**Common test set versus own traffic.** On the common test set federation improves on local training by +0.190 [+0.117, +0.262]† on TON-IoT and +0.187 [+0.171, +0.204]† on N-BaIoT. On a site's own test traffic, with records split at random, the result reverses (exploratory). On TON-IoT FedAvg − local is −0.121 [−0.204, −0.038] over repetitions (p = 0.015; FedAvg worse in 17 of 20 site–repetition pairs). The reversal is not a port shortcut: without the port features FedAvg is still worse in 16 of 20 pairs (−0.110 [−0.186, −0.034], p = 0.016) and keeps its common-test-set advantage (+0.121 [+0.044, +0.199], p = 0.012, uncorrected). On N-BaIoT FedAvg is worse in 43 of 45 device–repetition pairs (−0.081 [−0.095, −0.067], p < 0.001; with the device as the unit −0.081 [−0.131, −0.031], p = 0.006, worse on 9 of 9 devices; Supplement S5); both models detect 99.9% of a device's own attacks, at benign FPRs of 1.6% (FedAvg) and 0.8% (local), so the gap is confusion between attack types. Two observations point to site-specific fitting, which a random split rewards: on other sites' traffic of the classes a TON-IoT site holds, FedAvg is better (0.138 ± 0.072), and N-BaIoT local models reach 0.94 to 0.99 on their own traffic. Two diagnostics test this (three repetitions; Supplement S5). Under a blocked split, which cuts each site's records into contiguous training, validation and test blocks, the local advantage shrinks by 61% on TON-IoT and 47% on N-BaIoT (not significantly) but persists on TON-IoT (−0.049 [−0.087, −0.011]; N-BaIoT −0.029 [−0.111, +0.053]), so part of it comes from the similarity of training and test records under random splits. Correcting the federated output for each site's label prior leaves the gap essentially unchanged (−0.125 [−0.238, −0.012] before and −0.140 [−0.310, +0.030] after on TON-IoT), which points to fitting to each site's feature distribution rather than to its label prior.

**Personalisation.** This experiment re-trains FedAvg on the same partitions. As GPU training is not bit-wise deterministic, identical configurations differed in test macro-F1 by 0.036 on average over 15 re-runs (maximum 0.109; Supplement S1), as much as several effects above. After one local epoch on the site's own data, the selected federated model's own-traffic difference from the local model moves from −0.130 ± 0.052 to 0.008 ± 0.059 on TON-IoT (parity within our resolution) and from −0.073 ± 0.045 to −0.034 ± 0.020 on N-BaIoT (± over devices). The cost in breadth depends on the dataset: common-test macro-F1 falls from 0.710 ± 0.096 to 0.551 ± 0.027 on TON-IoT, giving up 89% of the federated model's gain over local training, but only from 0.900 ± 0.015 to 0.844 ± 0.038 on N-BaIoT (28%); Supplement S5 also compares a smaller learning rate and a proximal term. Running the federated and the adapted model side by side and alerting when either predicts an attack detects 99.9% of a TON-IoT site's own attacks at a benign FPR of 4.1% (federated model alone: 91.1% and 2.9%), but on the common test set the benign FPR of the pair rises to 16.4%, because the adapted model raises false alerts on unfamiliar benign traffic.

### D. Batch versus group normalisation

With batch normalisation whose running statistics are averaged by the server, the evaluated TON-IoT model swings between rounds, and a centralised model between two-epoch checkpoints (Fig. 2): in five repetitions at α = 0.5, the mean absolute change in validation macro-F1 between consecutive rounds is 0.110 for FedAvg, against 0.016 with group normalisation. The cause is the running statistics (Supplement S6). At the validation-selected round the score does not suffer (+0.022 against group normalisation, exploratory), so the swing matters for deploying an arbitrary round. Federated re-estimation fixes it: after aggregation every client sends the per-channel mean and mean square of each normalisation layer's input over up to 8,192 records, layer by layer, and the server averages them by client size; we score each round's model with these statistics, while training continues with the server-averaged ones. This adds one exchange per normalisation layer and round (192 values), whose privacy we do not analyse. It reduces the swing to 0.018 and makes the final round as predictable as with group normalisation (0.855 ± 0.037, against 0.634 ± 0.339 with server-averaged statistics). At the round selected on the re-estimated validation score, batch normalisation then exceeds group normalisation by +0.032 [+0.009, +0.055], p = 0.018, uncorrected at α = 0.5 and +0.039 [+0.018, +0.060], p = 0.007, uncorrected at α = 0.1 (exploratory), so our group-normalised results are, if anything, slightly conservative. On UNSW-NB15 the swing is small either way.

### E. Record-level differential privacy

Table IV reports DP on TON-IoT (UNSW-NB15 in Supplement S7). The DP code without clipping or noise matches non-private training on the same schedule, and clipping alone costs little in macro-F1 (but see Password in Fig. 3). On the short schedule, noise costs 5 points against non-private training at σ = 0.2 (ε ≈ 402), 8 at σ = 0.3, 8 at ε = 12.15 (11.72–12.50) and 10 at ε = 0.81 (0.78–0.84): most of the cost appears at noise levels that give no meaningful privacy, and the curve is nearly flat from σ = 0.3 (ε ≈ 117) to ε = 0.81 (0.78–0.84). At the ε of σ = 1 (about 3), neither SGD with momentum nor a four times larger batch with recalibrated noise reduces the cost (8 and 9 points against 9; Supplement S7), so it is not an artefact of Adam or of the batch size; why most of it already appears at σ = 0.2 remains open. On this schedule the cost falls on a subset of attack classes (Fig. 3): benign, DoS and DDoS traffic are detected almost as well as without noise, whereas 5 of the nine attack classes lose at least ten points at σ = 0.6 against the same code without clipping or noise, including password, backdoor, ransomware and scanning with 9,986 to 13,893 training records each, so the disparate impact of DP [38] is not limited to small classes. Tuning the clipping norm or learning rate, or doubling the schedule, does not remove the cost (Supplement S7). At the full budget with σ = 1.8 (ε = 2.9), DP-SGD reaches 0.683 ± 0.057 against 0.856 ± 0.046 for the same code without clipping or noise (three repetitions), a paired gap of 0.173 [0.144, 0.202] (16 points below non-private FedAvg, unpaired); here the cost is spread: 7 of 9 attack classes lose ten points or more, the largest loss being Password (0.85 to 0.31), and only DDoS, DoS and benign traffic lose less (DDoS 9.8 points). Part of this cost falls on classes that only the longer non-private schedule learns.

<!--TABLE:TON_dp-->
| Setting | Rounds × epochs | Noise σ | ε (δ = 10⁻⁵) | Test macro-F1, selected round | Test macro-F1, final round |
|---|---|---|---|---|---|
| Non-private FedAvg | 20 × 1 | – | – | 0.735 ± 0.051 | 0.721 ± 0.018 |
| DP code, no clipping or noise | 20 × 1 | 0 | ∞ | 0.760 ± 0.064 | 0.754 ± 0.059 |
| DP code, clipping only | 20 × 1 | 0 | ∞ | 0.743 ± 0.054 | 0.741 ± 0.059 |
| DP-SGD, C = 1 | 20 × 1 | 0.2 | 402 (398–404) | 0.681 ± 0.056 | 0.663 ± 0.051 |
| DP-SGD, C = 1 | 20 × 1 | 0.3 | 117 (115–119) | 0.656 ± 0.063 | 0.645 ± 0.054 |
| DP-SGD, C = 1 | 20 × 1 | 0.6 | 12.15 (11.72–12.50) | 0.658 ± 0.053 | 0.629 ± 0.010 |
| DP-SGD, C = 1 | 20 × 1 | 1.0 | 3.12 (2.99–3.24) | 0.648 ± 0.040 | 0.633 ± 0.002 |
| DP-SGD, C = 1 | 20 × 1 | 1.5 | 1.58 (1.51–1.63) | 0.660 ± 0.059 | 0.626 ± 0.006 |
| DP-SGD, C = 1 | 20 × 1 | 2.5 | 0.81 (0.78–0.84) | 0.633 ± 0.022 | 0.623 ± 0.021 |
| Non-private FedAvg | 50 × 2 | – | – | 0.840 ± 0.043 | 0.829 ± 0.040 |
| DP code, no clipping or noise | 50 × 2 | 0 | ∞ | 0.856 ± 0.046 | 0.854 ± 0.033 |
| DP-SGD, C = 1 | 50 × 2 | 1.8 | 2.93 (2.81–3.03) | 0.683 ± 0.057 | 0.651 ± 0.055 |

## VI. Discussion

**Evaluation.** With the host order fixed, FedAvg's interval on the host partition was several times narrower than with redrawn partitions (Supplement S4), and with correction only 10 of the 17 primary comparisons of FedAvg with a reference remain significant. We recommend that FL-IDS studies report validation-selected and final-round scores, redraw partitions, use device-level partitions where possible, tune baselines and widen grids with edge optima, include same-recipe centralised and local-only references, score sites on blocked or temporal own-traffic splits, and check DP code with no-noise sanity runs.

**Deployment.** Tuned FedAvg with group normalisation, or batch normalisation with federated re-estimation, is a strong default for a small consortium; the 23,754-parameter model is small enough for an IoT gateway, although we did not measure device-side cost. Joining buys detection of attacks a site has not seen; on traffic like its training traffic its own model can be better, and local adaptation recovers that accuracy at a data-dependent cost in breadth. Record-level DP is affordable for benign, DoS and DDoS traffic on a short schedule but degrades most attack classes at the full budget, so per-class detection should be monitored.

**Limitations.** All datasets come from testbeds or laboratory captures, and UNSW-NB15 is not IoT traffic; the TON-IoT host partition assigns testbed hosts to synthetic sites, and N-BaIoT's devices were recorded in one laboratory. Datasets with more devices, such as CICIoT2023 [41], or designed for federated evaluation, such as Edge-IIoTset [42], would allow richer device-level partitions. The main splits are random by record, which favours a site's local model on its own traffic and makes absolute scores optimistic (on UNSW-NB15 many test records share their feature vector with a training record; Supplement S3). Only differences of about five points are detectable. Preprocessing, including feature selection, is fitted centrally; all clients participate in every round; hyperparameters are tuned at one setting per dataset; algorithms are compared with group normalisation only; and TON-IoT's port features carry much of its signal. DP runs cover only α = 0.5, and the record-level guarantee does not hide participation. We do not study poisoning or personalised FL such as FedBN and Ditto.

## VII. Conclusion

Under a protocol that separates the effect of federation from the partition, the choice of round and the tuning of baselines, federation improves a site's common-test-set macro-F1 by up to +0.286 over training alone by detecting attacks it has not seen. Under device-level partitions and random record splits, a site's own model is usually better on its own traffic (exploratory); a blocked split reduces but does not remove this advantage on TON-IoT, and local adaptation recovers all or part of that accuracy at a dataset-dependent cost in breadth. Device-level partitions tended to cost more than Dirichlet partitions with a similar label mix. Tuned FedAvg is a strong default: FedProx, FedAdam, SCAFFOLD and FedLC did not improve significantly on it after Holm correction. On TON-IoT, a batch-normalised model with server-averaged statistics should not be deployed at an arbitrary round; group normalisation or federated re-estimation avoids the swing. Record-level DP-SGD at single-digit ε sharply degrades several attack classes, so private federated IDS should be evaluated per class. Future work should use datasets with separate sites and temporal splits, federated preprocessing and larger federations.

## Declarations

**Data and code availability.** TON-IoT, UNSW-NB15 and N-BaIoT are publicly available from their providers. The code, the corrected pipeline, the N-BaIoT sampling script and all result files are available at https://github.com/Amoakoemma123/iot-ids-pipeline (commit 2a1c16a). Supplementary material accompanies this paper. The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors.

**Declaration of competing interest.** The authors declare no competing interests.



## Acknowledgment

During the preparation of this work the authors used Claude (Anthropic) to write and edit the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] A. A. Diro and N. Chilamkurti, "Distributed attack detection scheme using deep learning approach for Internet of Things," *Future Gener. Comput. Syst.*, vol. 82, pp. 761–768, 2018, doi: 10.1016/j.future.2017.08.043.

[2] S. Agrawal et al., "Federated learning for intrusion detection system: Concepts, challenges and future directions," *Comput. Commun.*, vol. 195, pp. 346–361, 2022, doi: 10.1016/j.comcom.2022.09.012.

[3] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. AISTATS*, PMLR vol. 54, 2017, pp. 1273–1282.

[4] T. D. Nguyen, S. Marchal, M. Miettinen, H. Fereidooni, N. Asokan, and A.-R. Sadeghi, "DÏoT: A federated self-learning anomaly detection system for IoT," in *Proc. IEEE ICDCS*, 2019, pp. 756–767, doi: 10.1109/ICDCS.2019.00080.

[5] J. L. Hernandez-Ramos et al., "Intrusion detection based on federated learning: A systematic review," *ACM Comput. Surv.*, vol. 57, no. 12, pp. 1–65, 2025, doi: 10.1145/3731596.

[6] E. Mármol Campos et al., "Evaluating federated learning for intrusion detection in Internet of Things: Review and challenges," *Comput. Netw.*, vol. 203, Art. no. 108661, 2022, doi: 10.1016/j.comnet.2021.108661.

[7] T.-M. H. Hsu, H. Qi, and M. Brown, "Measuring the effects of non-identical data distribution for federated visual classification," arXiv:1909.06335, 2019.

[8] Q. Li, Y. Diao, Q. Chen, and B. He, "Federated learning on non-IID data silos: An experimental study," in *Proc. IEEE ICDE*, 2022, pp. 965–978, doi: 10.1109/ICDE53745.2022.00077.

[9] M. Abadi et al., "Deep learning with differential privacy," in *Proc. ACM CCS*, 2016, pp. 308–318, doi: 10.1145/2976749.2978318.

[10] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang, "Learning differentially private recurrent language models," in *Proc. ICLR*, 2018.

[11] A. Alsaedi, N. Moustafa, Z. Tari, A. Mahmood, and A. Anwar, "TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems," *IEEE Access*, vol. 8, pp. 165130–165150, 2020, doi: 10.1109/ACCESS.2020.3022862.

[12] N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustain. Cities Soc.*, vol. 72, Art. no. 102994, 2021, doi: 10.1016/j.scs.2021.102994.

[13] N. Moustafa and J. Slay, "UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)," in *Proc. MilCIS*, 2015, pp. 1–6, doi: 10.1109/MilCIS.2015.7348942.

[14] Y. Meidan et al., "N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders," *IEEE Pervasive Comput.*, vol. 17, no. 3, pp. 12–22, 2018, doi: 10.1109/MPRV.2018.03367731.

[15] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in *Proc. MLSys*, vol. 2, 2020, pp. 429–450.

[16] S. J. Reddi et al., "Adaptive federated optimization," in *Proc. ICLR*, 2021.

[17] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh, "SCAFFOLD: Stochastic controlled averaging for federated learning," in *Proc. ICML*, PMLR vol. 119, 2020, pp. 5132–5143.

[18] J. Zhang et al., "Federated learning with label distribution skew via logits calibration," in *Proc. ICML*, PMLR vol. 162, 2022, pp. 26311–26329.

[19] Y. Wu and K. He, "Group normalization," in *Proc. ECCV*, LNCS vol. 11217, 2018, pp. 3–19, doi: 10.1007/978-3-030-01261-8_1.

[20] K. Hsieh, A. Phanishayee, O. Mutlu, and P. B. Gibbons, "The non-IID data quagmire of decentralized machine learning," in *Proc. ICML*, PMLR vol. 119, 2020, pp. 4387–4398.

[21] I. Mironov, "Rényi differential privacy," in *Proc. IEEE CSF*, 2017, pp. 263–275, doi: 10.1109/CSF.2017.11.

[22] I. Mironov, K. Talwar, and L. Zhang, "Rényi differential privacy of the sampled Gaussian mechanism," arXiv:1908.10530, 2019.

[23] V. Rey, P. M. Sánchez Sánchez, A. Huertas Celdrán, and G. Bovet, "Federated learning for malware detection in IoT devices," *Comput. Netw.*, vol. 204, Art. no. 108693, 2022, doi: 10.1016/j.comnet.2021.108693.

[24] R. Lazzarini, H. Tianfield, and V. Charissis, "Federated learning for IoT intrusion detection," *AI*, vol. 4, no. 3, pp. 509–530, 2023, doi: 10.3390/ai4030028.

[25] S. I. Popoola, A. L. Imoize, M. Hammoudeh, B. Adebisi, O. Jogunola, and A. M. Aibinu, "Federated deep learning for intrusion detection in consumer-centric Internet of Things," *IEEE Trans. Consum. Electron.*, vol. 70, no. 1, pp. 1610–1622, 2024, doi: 10.1109/TCE.2023.3347170.

[26] B. Olanrewaju-George and B. Pranggono, "Federated learning-based intrusion detection system for the Internet of Things using unsupervised and supervised deep learning models," *Cyber Secur. Appl.*, vol. 3, Art. no. 100068, 2025, doi: 10.1016/j.csa.2024.100068.

[27] H. Peng, C. Wu, and Y. Xiao, "FD-IDS: Federated learning with knowledge distillation for intrusion detection in non-IID IoT environments," *Sensors*, vol. 25, no. 14, Art. no. 4309, 2025, doi: 10.3390/s25144309.

[28] T. D. Nguyen, P. Rieger, M. Miettinen, and A.-R. Sadeghi, "Poisoning attacks on federated learning-based IoT intrusion detection system," in *Proc. Workshop Decentralized IoT Syst. Secur. (DISS)*, 2020, doi: 10.14722/diss.2020.23003.

[29] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, "Federated learning with non-IID data," arXiv:1806.00582, 2018.

[30] Y. Wang, Q. Shi, and T.-H. Chang, "Why batch normalization damage federated learning on non-IID data?" *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 36, no. 1, pp. 1692–1706, 2025, doi: 10.1109/TNNLS.2023.3323302.

[31] X. Li, M. Jiang, X. Zhang, M. Kamp, and Q. Dou, "FedBN: Federated learning on non-IID features via local batch normalization," in *Proc. ICLR*, 2021.

[32] T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and robust federated learning through personalization," in *Proc. ICML*, PMLR vol. 139, 2021, pp. 6357–6368.

[33] S. Sun, L. Zhou, Z. Wang, and L. Han, "Robust intrusion detection based on personalized federated learning for IoT environment," *Comput. Secur.*, vol. 154, Art. no. 104442, 2025, doi: 10.1016/j.cose.2025.104442.

[34] H. Yan, X. Lin, S. Li, H. Peng, and B. Zhang, "Global or local adaptation? Client-sampled federated meta-learning for personalized IoT intrusion detection," *IEEE Trans. Inf. Forensics Security*, vol. 20, pp. 279–293, 2025, doi: 10.1109/TIFS.2024.3516548.

[35] J. Fu et al., "Differentially private federated learning: A systematic review," *ACM Comput. Surv.*, vol. 58, no. 11, pp. 1–38, 2026, doi: 10.1145/3801079.

[36] P. Ruzafa-Alcazar et al., "Intrusion detection based on privacy-preserving federated learning for the Industrial IoT," *IEEE Trans. Ind. Informat.*, vol. 19, no. 2, pp. 1145–1154, 2023, doi: 10.1109/TII.2021.3126728.

[37] O. Friha, M. A. Ferrag, M. Benbouzid, T. Berghout, B. Kantarci, and K.-K. R. Choo, "2DF-IDS: Decentralized and differentially private federated learning-based intrusion detection system for industrial IoT," *Comput. Secur.*, vol. 127, Art. no. 103097, 2023, doi: 10.1016/j.cose.2023.103097.

[38] E. Bagdasaryan, O. Poursaeed, and V. Shmatikov, "Differential privacy has disparate impact on model accuracy," in *Proc. NeurIPS*, vol. 32, 2019.

[39] Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, "Kitsune: An ensemble of autoencoders for online network intrusion detection," in *Proc. NDSS*, 2018, doi: 10.14722/ndss.2018.23204.

[40] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.

[41] E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, Art. no. 5941, 2023, doi: 10.3390/s23135941.

[42] M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, "Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning," *IEEE Access*, vol. 10, pp. 40281–40306, 2022, doi: 10.1109/ACCESS.2022.3165809.
