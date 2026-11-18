# Paper 4 v2 manuscript (double-brace placeholders are filled from values.json by build.py)

## Title

**Federated Intrusion Detection for IoT Networks Revisited: Device-Level Partitions, Tuned Baselines and Record-Level Differential Privacy**

## Abstract

Federated learning (FL) is widely proposed for training intrusion detection systems (IDS) across IoT sites without sharing traffic, but its measured benefit depends on round selection, client partitions, baseline tuning and whether privacy budgets match the training. We re-evaluate FL for a compact 1D-CNN IDS on TON-IoT, UNSW-NB15 and N-BaIoT with redrawn partitions, validation-based round selection, tuned baselines, same-recipe centralised and local-only references, and record-level DP-SGD with a matching accountant, partitioning TON-IoT by destination host and N-BaIoT by physical device besides Dirichlet label skew. On the common test set, FedAvg beats local-only training by {{TON_d_fedavg_natr_vs_local_natr_mean}} macro-F1 (TON-IoT hosts) and {{NB_d_fedavg_dev_vs_local_dev_mean}} (N-BaIoT devices). On each site's own test traffic, however, the local model is better in {{TON_own_pairs_neg}} of {{TON_own_pairs_n}} site–repetition pairs on TON-IoT and {{NB_own_pairs_neg}} of {{NB_own_pairs_n}} on N-BaIoT under random record splits (exploratory); under a blocked split it is smaller (by {{P4F_TON_blk_shrink}}% and {{P4F_NB_blk_shrink}}%, not significantly) but persists on TON-IoT. Local fine-tuning brings TON-IoT sites to parity within our resolution and halves the N-BaIoT gap, at a cost in detecting other sites' attacks. FedProx, FedAdam, SCAFFOLD and FedLC do not improve on tuned FedAvg after Holm correction. On TON-IoT, batch normalisation with server-averaged statistics makes the model swing between rounds; federated re-estimation removes the swing. At the full training budget, record-level DP-SGD at ε ≈ {{TON_dp_full_sigma18_eps}} costs {{TON_dp_gap_full}} macro-F1 points on TON-IoT, and {{TON_dpfull_nlost}} of {{TON_dpfull_natt}} attack classes lose ten points or more.

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

TON-IoT contains network flows [12] from an IoT/IIoT testbed [11] with nine attack classes and benign traffic; UNSW-NB15 contains enterprise-testbed flows with nine attack categories [13]. Both pass through our corrected pipeline, released with this paper: identifier columns are removed, exact duplicates are removed after all column drops, the data are split 70/15/15 (stratified), and feature selection and min–max scaling are fitted on the training split, leaving {{TON_nfeat}} features for TON-IoT ({{TON_ntrain}}, {{TON_nval}} and {{TON_ntest}} training, validation and test records) and {{UNSW_nfeat}} for UNSW-NB15 ({{UNSW_ntrain}}, {{UNSW_nval}}, {{UNSW_ntest}}). On TON-IoT the features include the source and destination ports, which can act as testbed shortcuts. N-BaIoT [14] contains benign, BASHLITE and Mirai traffic of nine commercial IoT devices, described by {{NB_nfeat}} Kitsune stream statistics [39]. We use a uniform {{NB_frac}}% sample of every device–class file, deduplicated at the model's 32-bit precision and without the two degenerate BASHLITE flood classes, split 70/15/15 by device and class with a cross-split duplicate guard ({{NB_nclasses}} classes; {{NB_ntrain}}, {{NB_nval}} and {{NB_ntest}} records; Supplement S1). Preprocessing is fitted centrally, a simplification a deployment would replace with federated statistics.

### B. Client partitions

**Dirichlet label skew.** On TON-IoT and UNSW-NB15 the training split is distributed over N = 4 clients (2, 8 and 16 in a scaling study), at random (IID) or with per-class Dirichlet proportions, α ∈ {1.0, 0.5, 0.1} [7], [8]. At α = 0.5 the two largest classes make up {{TON_skew_a05_top2_mean}} of a TON-IoT client's data on average, at α = 0.1 {{TON_skew_a01_top2_mean}}.

**Hosts on TON-IoT.** Each destination host is assigned, as a whole, to one of four clients, so that a client sees all traffic to the devices it monitors; the assignment is redrawn in every repetition. Its label skew is similar to α = 0.5 (two largest classes {{TON_skew_natr_top2_mean}} on average), but the clients differ in the devices behind the traffic and, because traffic is concentrated on few hosts, in size ({{TON_natr_size_min}} to {{TON_natr_size_max}} records). A client's own test traffic is the test traffic to its hosts ({{TON_own_n_min}} to {{TON_own_n_max}} records with {{TON_own_cls_min}} to {{TON_own_cls_max}} classes per site). UNSW-NB15 has no usable addresses.

**Devices on N-BaIoT.** Every physical device is one client (N = 9), and its own test traffic is its test split ({{NB_own_n_min}} to {{NB_own_n_max}} records, {{NB_own_cls_min}} to {{NB_own_cls_max}} classes). The label skew is milder than at α = 0.5 (two largest classes {{NB_skew_dev_top2_mean}} on average). As references we distribute N-BaIoT over nine clients at random and with α = 0.5 and 0.1. Each repetition draws a new partition (except the fixed device partition), initialisation and batch order.

### C. Model, training and selection

The CNN has two convolutional blocks (32 and 64 filters, kernel 3, max-pooling) with group normalisation [19], global average pooling and two dense layers (128 and 64 units, dropout 0.3), {{TON_nparams}} parameters ({{TON_model_kb}} kB). Federated training runs for 50 rounds of two local epochs (batch size 256) with size-weighted aggregation. We report the test macro-F1 at the round with the highest validation macro-F1 (pooled validation split; Supplement S1) and at the final round; the test set never influences any choice. The centralised reference trains the same model with the same optimiser and learning rate on the pooled data for 100 epochs, checkpointed and selected every two epochs; local-only models are trained in the same way on each client's data alone. Every model is scored on the common test set and, under device-level partitions, on each client's own test traffic: macro-F1 over the classes present in it, attack detection rate and benign false-positive rate (FPR).

### D. Tuning and statistics

Hyperparameters are tuned on the validation set in separate tuning repetitions (α = 0.5 on TON-IoT and UNSW-NB15, device partition on N-BaIoT): FedAvg's Adam learning rate first, then FedProx's μ, FedAdam's server learning rate and FedLC's τ at that client learning rate, and the SGD learning rate of SCAFFOLD and of FedAvg with SGD (Supplement S1). {{EDGE_SENTENCE}} Core results use five repetitions; the scaling study, the batch-normalisation runs with server-averaged statistics only, the 100-round check and the DP study use three, and contrasts between run sets of different sizes are unpaired. We report means with t-based 95% confidence intervals (± half-width). Differences between methods on the same partition are paired over repetitions and reported as mean difference [95% CI]; differences between partitions use Welch's test. For each dataset a family of primary comparisons, declared before the N-BaIoT, personalisation and no-port experiments, is Holm-corrected [40] (Supplement S2, Table {{stabref:paired}}); † marks significance after correction. All other comparisons are exploratory, with uncorrected p-values.

## IV. Algorithms and Differential Privacy

FedAvg [3] trains locally with Adam (reset each round). FedProx [15] adds (μ/2)‖w − w_g‖² to the local loss. FedAdam [16] applies a server Adam step (β₁ = 0.9, β₂ = 0.99, τ = 10⁻³) to the averaged update; only its server learning rate is tuned. SCAFFOLD [17] uses local SGD with control variates (option II; model updates size-weighted, control variates averaged uniformly) and doubles communication; FedAvg with SGD is its reference. FedLC [18] trains on calibrated logits z_c − τ·n_{k,c}^{−1/4}, with n_{k,c} the number of class-c records of client k (set to 1 for classes the client does not hold); evaluation uses the uncalibrated output.

Each client trains with DP-SGD [9]: batches are Poisson-sampled with rate q = B/n_k, per-example gradients are clipped to L2 norm C = 1, Gaussian noise with standard deviation σC is added to their sum, and Adam takes the step. The model uses group normalisation and no dropout. The privacy loss of each client's records is accounted with RDP for the subsampled Gaussian mechanism [21], [22] over all local steps and converted to (ε, δ) with δ = 10⁻⁵; we report the maximum over clients. The unit is one training record: protecting a device's traffic as a whole would need group privacy or user-level DP, and the validation records used for round selection are not protected. DP runs use α = 0.5, mostly with 20 rounds of one local epoch (per-example gradients are costly) and once with the full 50 rounds of two epochs; sanity runs use the same code without noise, with and without clipping.

## V. Results

### A. Federation under label skew

Table {{tabref:TON_main}} reports TON-IoT (UNSW-NB15 in Supplement S3), and Fig. 1 shows the paired comparisons. The centralised TON-IoT model reaches {{TON_central_sel}}. FedAvg differs from it by {{TON_d_fedavg_iid_vs_central}} with IID clients, {{TON_d_fedavg_a05_vs_central}} at α = 0.5 and {{TON_d_fedavg_a01_vs_central}} at α = 0.1; only the IID gap is significant after correction, but the gap grows with skew (exploratory slope over the four skew levels coded 0–3, tested over repetitions: {{TON_trend_slope}} per level, {{TON_trend_p}}; UNSW-NB15: {{UNSW_trend_slope}}, {{UNSW_trend_p}}). On the common test set FedAvg improves on local training by {{TON_d_fedavg_a05_vs_local_a05}} at α = 0.5 and {{TON_d_fedavg_a01_vs_local_a01}} at α = 0.1, largely because a local model cannot detect classes its client never saw (Supplement S4). For FedAvg, selected and final rounds are close, but the selected rounds cluster near the end of the budget (round {{TON_fedavg_a05_round}} of 50 at α = 0.5); 100 rounds narrow the α = 0.5 gap and leave all gaps within their intervals (Supplement S4). Without the port features TON-IoT scores fall (centralised: {{TON_d_central_noports_vs_central_mean}}) and federation costs more.

{{table:TON_main}}

### B. Algorithms

No drift-reducing algorithm improves on FedAvg on any dataset after correction (Fig. 1). On TON-IoT FedProx differs from FedAvg by {{TON_d_fedprox_a05_vs_fedavg_a05}} and FedLC by {{TON_d_fedlc_a05_vs_fedavg_a05}} at α = 0.5; FedAdam trails it ({{TON_d_fedadam_a05_vs_fedavg_a05}}, the only algorithm difference significant after correction; at α = 0.1 and on the host partition, {{TON_d_fedadam_a01_vs_fedavg_a01}} and {{TON_d_fedadam_natr_vs_fedavg_natr}}, not after correction). As only its server learning rate was tuned, at FedAvg's client learning rate, this deficit may partly reflect the tuning design; on N-BaIoT FedAdam has the best point estimate ({{NB_d_fedadam_dev_vs_fedavg_dev}}). SCAFFOLD does not improve on FedAvg with SGD ({{TON_d_scaffold_a05_vs_fedavgsgd_a05}}) and trails Adam-based FedAvg ({{TON_d_scaffold_a05_vs_fedavg_a05}}). On UNSW-NB15 FedProx (α = 0.1) and FedLC (α = 0.5) are better before correction only; FedLC's gain ({{UNSW_d_fedlc_a05_vs_fedavg_a05}}) comes with a benign FPR of {{UNSW_fedlc_a05_fpr_m}}% against {{UNSW_fedavg_a05_fpr_m}}%, and SCAFFOLD is erratic under severe skew (Supplement S3). No gain survives correction, but only in {{n_alg_equiv}} of the {{n_alg_cmp}} algorithm comparisons does the 90% CI lie within ±0.05 (equivalence at that margin); the others, including FedLC on UNSW-NB15, are inconclusive.

### C. Device-level partitions

**Cost against pooled training.** Under the TON-IoT host partition (Table {{tabref:TON_nat}}) FedAvg reaches {{TON_fedavg_natr_sel}}, {{TON_d_fedavg_natr_vs_central}} against the centralised model, with a larger spread over repetitions than at α = 0.5 (SD {{TON_fedavg_natr_sel_sd}} against {{TON_fedavg_a05_sel_sd}}). Although its label skew is similar to α = 0.5, it costs about as much as α = 0.1 (exploratory Welch tests in Supplement S4). On N-BaIoT (Table {{tabref:NB_main}}) FedAvg differs from the centralised model ({{NB_central_sel}}) by {{NB_d_fedavg_dev_vs_central}} with one client per device, {{NB_d_fedavg_iid9_vs_central}} with IID clients, and {{NB_d_fedavg_a05_N9_vs_central}} and {{NB_d_fedavg_a01_N9_vs_central}} at α = 0.5 and 0.1. Device-level partitions thus tended to cost more than Dirichlet partitions of similar label skew (α = 0.5), but not more than α = 0.1. The comparison is exploratory and confounded with client sizes; as the N-BaIoT device partition has milder label skew than α = 0.5 yet costs more, the likely mechanism is feature skew between devices, the setting FedBN [31] addresses.

{{table:TON_nat}}

{{table:NB_main}}

**Common test set versus own traffic.** On the common test set federation improves on local training by {{TON_d_fedavg_natr_vs_local_natr}} on TON-IoT and {{NB_d_fedavg_dev_vs_local_dev}} on N-BaIoT. On a site's own test traffic, with records split at random, the result reverses (exploratory). On TON-IoT FedAvg − local is {{TON_own_rep_gain}} over repetitions ({{TON_own_rep_p}}; FedAvg worse in {{TON_own_pairs_neg}} of {{TON_own_pairs_n}} site–repetition pairs). The reversal is not a port shortcut: without the port features FedAvg is still worse in {{TON_np_own_neg}} of {{TON_np_own_n}} pairs ({{TON_np_own_gain}}, {{TON_np_own_p}}) and keeps its common-test-set advantage ({{TON_np_common_gain}}). On N-BaIoT FedAvg is worse in {{NB_own_pairs_neg}} of {{NB_own_pairs_n}} device–repetition pairs ({{NB_own_rep_gain}}, {{NB_own_rep_p}}; with the device as the unit {{NB_dev_gain}}, {{NB_dev_gain_p}}, worse on {{NB_dev_neg}} of {{NB_dev_n}} devices; Supplement S5); both models detect {{NB_own_fed_attack_detection}}% of a device's own attacks, at benign FPRs of {{NB_own_fed_benign_fpr}}% (FedAvg) and {{NB_own_loc_benign_fpr}}% (local), so the gap is confusion between attack types. Two observations point to site-specific fitting, which a random split rewards: on other sites' traffic of the classes a TON-IoT site holds, FedAvg is better ({{TON_cli_anatr_owngain}}), and N-BaIoT local models reach {{NB_loc_own_min}} to {{NB_loc_own_max}} on their own traffic. Two diagnostics test this (three repetitions; Supplement S5). Under a blocked split, which cuts each site's records into contiguous training, validation and test blocks, the local advantage shrinks by {{P4F_TON_blk_shrink}}% on TON-IoT and {{P4F_NB_blk_shrink}}% on N-BaIoT (not significantly) but persists on TON-IoT ({{P4F_TON_blk_own_gain}}; N-BaIoT {{P4F_NB_blk_own_gain}}), so part of it comes from the similarity of training and test records under random splits. Correcting the federated output for each site's label prior leaves the gap essentially unchanged ({{P4F_TON_orig_fed_gain}} before and {{P4F_TON_orig_prior_gain}} after on TON-IoT), which points to fitting to each site's feature distribution rather than to its label prior.

**Personalisation.** This experiment re-trains FedAvg on the same partitions. As GPU training is not bit-wise deterministic, identical configurations differed in test macro-F1 by {{RERUN_MEANABS}} on average over {{RERUN_N}} re-runs (maximum {{RERUN_MAXDIFF}}; Supplement S1), as much as several effects above. After one local epoch on the site's own data, the selected federated model's own-traffic difference from the local model moves from {{TON_pers_fed_gain}} to {{TON_pers_full1_gain}} on TON-IoT (parity within our resolution) and from {{NB_pers_fed_gain}} to {{NB_pers_full1_gain}} on N-BaIoT (± over devices). The cost in breadth depends on the dataset: common-test macro-F1 falls from {{TON_pers_fed_common}} to {{TON_pers_full1_common}} on TON-IoT, giving up {{TON_pers_full1_giveup}}% of the federated model's gain over local training, but only from {{NB_pers_fed_common}} to {{NB_pers_full1_common}} on N-BaIoT ({{NB_pers_full1_giveup}}%); Supplement S5 also compares a smaller learning rate and a proximal term. Running the federated and the adapted model side by side and alerting when either predicts an attack detects {{TON_pers_full1_either_site_attack_detection}}% of a TON-IoT site's own attacks at a benign FPR of {{TON_pers_full1_either_site_benign_fpr}}% (federated model alone: {{TON_pers_fed_site_attack_detection}}% and {{TON_pers_fed_site_benign_fpr}}%), but on the common test set the benign FPR of the pair rises to {{TON_pers_full1_either_test_benign_fpr}}%, because the adapted model raises false alerts on unfamiliar benign traffic.

### D. Batch versus group normalisation

With batch normalisation whose running statistics are averaged by the server, the evaluated TON-IoT model swings between rounds, and a centralised model between two-epoch checkpoints (Fig. 2): in five repetitions at α = 0.5, the mean absolute change in validation macro-F1 between consecutive rounds is {{TON_bnfr_a05_swing}} for FedAvg, against {{TON_fedavg_a05_swing}} with group normalisation. The cause is the running statistics (Supplement S6). At the validation-selected round the score does not suffer ({{TON_d_bnfr_a05_run_vs_gn_mean}} against group normalisation, exploratory), so the swing matters for deploying an arbitrary round. Federated re-estimation fixes it: after aggregation every client sends the per-channel mean and mean square of each normalisation layer's input over up to 8,192 records, layer by layer, and the server averages them by client size; we score each round's model with these statistics, while training continues with the server-averaged ones. This adds one exchange per normalisation layer and round (192 values), whose privacy we do not analyse. It reduces the swing to {{TON_bnfr_a05_fr_swing}} and makes the final round as predictable as with group normalisation ({{TON_bnfr_a05_fin}}, against {{TON_bnfr_a05_run_fin}} with server-averaged statistics). At the round selected on the re-estimated validation score, batch normalisation then exceeds group normalisation by {{TON_d_bnfr_a05_fr_vs_gn}} at α = 0.5 and {{TON_d_bnfr_a01_fr_vs_gn}} at α = 0.1 (exploratory), so our group-normalised results are, if anything, slightly conservative. On UNSW-NB15 the swing is small either way.

### E. Record-level differential privacy

Table {{tabref:TON_dp}} reports DP on TON-IoT (UNSW-NB15 in Supplement S7). The DP code without clipping or noise matches non-private training on the same schedule, and clipping alone costs little in macro-F1 (but see Password in Fig. 3). On the short schedule, noise costs {{TON_dp_cost_short_02}} points against non-private training at σ = 0.2 (ε ≈ {{TON_dp_sigma02_eps}}), {{TON_dp_cost_short_03}} at σ = 0.3, {{TON_dp_cost_short_06}} at ε = {{TON_dp_sigma06_eps}} and {{TON_dp_cost_short_25}} at ε = {{TON_dp_sigma25_eps}}: most of the cost appears at noise levels that give no meaningful privacy, and the curve is nearly flat from σ = 0.3 (ε ≈ {{TON_dp_sigma03_eps}}) to ε = {{TON_dp_sigma25_eps}}. At the ε of σ = 1 (about 3), neither SGD with momentum nor a four times larger batch with recalibrated noise reduces the cost ({{P4F_dp_sgdm_cost}} and {{P4F_dp_b1024_cost}} points against {{TON_dp_cost_short_1}}; Supplement S7), so it is not an artefact of Adam or of the batch size; why most of it already appears at σ = 0.2 remains open. On this schedule the cost falls on a subset of attack classes (Fig. 3): benign, DoS and DDoS traffic are detected almost as well as without noise, whereas {{TON_dp_nlost_06_code}} of the nine attack classes lose at least ten points at σ = 0.6 against the same code without clipping or noise, including password, backdoor, ransomware and scanning with {{TON_ntrain_ransomware}} to {{TON_ntrain_password}} training records each, so the disparate impact of DP [38] is not limited to small classes. Tuning the clipping norm or learning rate, or doubling the schedule, does not remove the cost (Supplement S7). At the full budget with σ = 1.8 (ε = {{TON_dp_full_sigma18_eps}}), DP-SGD reaches {{TON_dp_full_sigma18_ci}} against {{TON_dp_full_nonoise_ci}} for the same code without clipping or noise ({{TON_dp_full_sigma18_n_word}} repetitions), a paired gap of {{TON_dp_full_gap_ci}} ({{TON_dp_cost_full_vs_fedavg}} points below non-private FedAvg, unpaired); here the cost is spread: {{TON_dpfull_nlost}} of {{TON_dpfull_natt}} attack classes lose ten points or more, the largest loss being {{TON_dpfull_worst}}, and only {{TON_dpfull_spared}} lose less (DDoS {{TON_dpfull_ddos_loss}} points). Part of this cost falls on classes that only the longer non-private schedule learns.

{{table:TON_dp}}

## VI. Discussion

**Evaluation.** With the host order fixed, FedAvg's interval on the host partition was several times narrower than with redrawn partitions (Supplement S4), and with correction only {{n_fedref_sig}} of the {{n_fedref}} primary comparisons of FedAvg with a reference remain significant. We recommend that FL-IDS studies report validation-selected and final-round scores, redraw partitions, use device-level partitions where possible, tune baselines and widen grids with edge optima, include same-recipe centralised and local-only references, score sites on blocked or temporal own-traffic splits, and check DP code with no-noise sanity runs.

**Deployment.** Tuned FedAvg with group normalisation, or batch normalisation with federated re-estimation, is a strong default for a small consortium; the 23,754-parameter model is small enough for an IoT gateway, although we did not measure device-side cost. Joining buys detection of attacks a site has not seen; on traffic like its training traffic its own model can be better, and local adaptation recovers that accuracy at a data-dependent cost in breadth. Record-level DP is affordable for benign, DoS and DDoS traffic on a short schedule but degrades most attack classes at the full budget, so per-class detection should be monitored.

**Limitations.** All datasets come from testbeds or laboratory captures, and UNSW-NB15 is not IoT traffic; the TON-IoT host partition assigns testbed hosts to synthetic sites, and N-BaIoT's devices were recorded in one laboratory. Datasets with more devices, such as CICIoT2023 [41], or designed for federated evaluation, such as Edge-IIoTset [42], would allow richer device-level partitions. The main splits are random by record, which favours a site's local model on its own traffic and makes absolute scores optimistic (on UNSW-NB15 many test records share their feature vector with a training record; Supplement S3). Only differences of about five points are detectable. Preprocessing, including feature selection, is fitted centrally; all clients participate in every round; hyperparameters are tuned at one setting per dataset; algorithms are compared with group normalisation only; and TON-IoT's port features carry much of its signal. DP runs cover only α = 0.5, and the record-level guarantee does not hide participation. We do not study poisoning or personalised FL such as FedBN and Ditto.

## VII. Conclusion

Under a protocol that separates the effect of federation from the partition, the choice of round and the tuning of baselines, federation improves a site's common-test-set macro-F1 by up to {{TON_d_fedavg_a01_vs_local_a01_mean}} over training alone by detecting attacks it has not seen. Under device-level partitions and random record splits, a site's own model is usually better on its own traffic (exploratory); a blocked split reduces but does not remove this advantage on TON-IoT, and local adaptation recovers all or part of that accuracy at a dataset-dependent cost in breadth. Device-level partitions tended to cost more than Dirichlet partitions with a similar label mix. Tuned FedAvg is a strong default: FedProx, FedAdam, SCAFFOLD and FedLC did not improve significantly on it after Holm correction. On TON-IoT, a batch-normalised model with server-averaged statistics should not be deployed at an arbitrary round; group normalisation or federated re-estimation avoids the swing. Record-level DP-SGD at single-digit ε sharply degrades several attack classes, so private federated IDS should be evaluated per class. Future work should use datasets with separate sites and temporal splits, federated preprocessing and larger federations.

## Declarations

**Data and code availability.** TON-IoT, UNSW-NB15 and N-BaIoT are publicly available from their providers. The code, the corrected pipeline, the N-BaIoT sampling script and all result files are available at {{REPO}}{{CODE_COMMIT}}. Supplementary material accompanies this paper. The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors.

**Declaration of competing interest.** The authors declare no competing interests.



## Acknowledgment

{{AI_DECLARATION}}

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
