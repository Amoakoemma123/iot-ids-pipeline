# Paper 1 v2 manuscript (double-brace placeholders are filled from values.json by build.py)

## Title

**How Light Is Lightweight? Detection Accuracy Against Model Footprint in IoT Intrusion Detection**

## Abstract

Lightweight intrusion detectors for the Internet of Things are usually compact neural networks, justified against tree ensembles of unbounded size and often timed on GPUs or workstations. We measure macro-F1, model size and single-thread CPU latency of logistic regression, decision trees, random forests and gradient-boosted trees (GBDTs) at several size budgets, and of a compact 1D-CNN and an MLP at three widths in float32 and int8, on TON-IoT, UNSW-NB15 and N-BaIoT with five seeds, counting the networks' input transform in their size. At the size of the largest int8 network with its input table (204 KB on TON-IoT and 145 KB on UNSW-NB15), the best tree model was more accurate than the best network, by 0.004 and 0.012. From 201 KB (TON-IoT) and 97 KB (UNSW-NB15), a tree model significantly beat every network, the best tree by 0.010 and 0.046; an unbounded forest was no more accurate than the best smaller tree model. Int8 cost the CNNs at most 0.010 macro-F1 and the MLPs up to 0.021. On one thread, the TON-IoT tree models tied for best accuracy took 9.8–82.1 ms per record in scikit-learn, the best network 0.42 ms, and a compiled tree library was 59–108 times cheaper per tree. scikit-learn's early-stopped GBDT keeps the iterations after its best validation score; boosting diverged in them on some seeds, and rolling back to the best iteration removed every collapse. Lightweight claims need size-budgeted tree baselines, deployable bytes and single-thread latency with a named runtime, over several seeds.

**Keywords:** intrusion detection; Internet of Things; lightweight models; model size; random forest; gradient boosting; quantisation; evaluation methodology

## 1. Introduction

Intrusion detection for the Internet of Things (IoT) is usually framed as a trade-off between detection quality and the resources of the devices that must run the detector. Gateways, routers and embedded controllers have little memory, few cores and no accelerator, so a large share of recent work proposes "lightweight" detectors: compact convolutional or recurrent networks, often a few tens of thousands of parameters, that are reported to approach the accuracy of larger models at a fraction of their size (e.g. Zhao et al., 2022; Lahasan & Samma, 2022; Mazinani et al., 2026).

Such claims depend on what the compact network is compared with. Tree ensembles remain among the strongest models on tabular data (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022; McElfresh et al., 2023), and network-flow records are tabular. A random forest grown without limits can indeed occupy tens or hundreds of megabytes, but its size is a choice: the number and depth of its trees can be bounded, and gradient-boosted trees build their accuracy from many small trees whose number can be capped. A comparison between a compact network and an unbounded forest therefore says little about which family is better at a given size. The same holds for speed: latencies measured on a desktop GPU, or with all cores of a workstation, do not describe a single-core gateway.

This study began as a re-examination of our own earlier, unpublished comparison of this kind; Section 5.5 uses it only as a worked example of two evaluation pitfalls.

We ask three questions. (i) At a given model size, which model family detects attacks best on IoT flow data? (ii) How much does post-training int8 quantisation, the usual route to deploying networks on small devices, cost in detection quality? (iii) What does each model cost per decision on one CPU core, and how much of that cost belongs to the runtime rather than the model? Our contributions are:

1. An accuracy–footprint comparison of logistic regression, decision trees, random forests and gradient-boosted trees at several size budgets, including GBDTs with capped iterations, and a compact 1D-CNN and an MLP at three widths, in float32 and int8, on TON-IoT, UNSW-NB15 and N-BaIoT with five seeds each, with configurations selected on validation data.
2. Single-thread CPU latency, per record and per batch, and model size in one deployable unit for every configuration, with a compiled tree runtime and full-integer network quantisation as runtime controls.
3. A port-number ablation on TON-IoT, which tests whether the models rely on testbed-specific port assignments.
4. Three evaluation pitfalls, each shown with a controlled experiment: scikit-learn's early-stopped histogram GBDT keeps the iterations after its best validation score and can be evaluated after boosting has diverged; min–max scaling of heavy-tailed counts makes a batch-normalised network look weaker than it is; and the pickled size of a forest is about ten times its deployable size.

Section 2 reviews related work, Section 3 describes the data, Section 4 the models and protocol, Section 5 the results, and Section 6 discusses implications and limitations.

## 2. Related Work

### 2.1 Lightweight intrusion detection for IoT

Many studies propose compact neural detectors for IoT traffic and report their size or speed. Zhao et al. (2022) combine principal component analysis with a lightweight network built from inverted residual blocks and channel shuffling, emphasising low complexity and small model size; Lahasan and Samma (2022) evolve a shallow autoencoder with two hidden neurons for N-BaIoT; Zakariyya et al. (2023) reduce the memory and training time of a regularised deep network for IoT security monitoring; and Musthafa et al. (2025) prune and quantise a stacked LSTM ensemble to run on a Raspberry Pi. Hardware studies go further down: Ngo et al. (2023) run a small neural network on an AI microcontroller and an FPGA, and Mazinani et al. (2026) profile six deep architectures on an STM32 microcontroller with CICIoT2023, finding that post-training int8 quantisation removes more than 90% of the model size at negligible loss and that an MLP and a temporal convolutional network give the best trade-off. Quantisation is the usual route to such devices: Sharmila and Nagapadma (2023) quantise an autoencoder to 8-bit integers, reducing its size by 92%, and Ullah et al. (2026) compare float32, float16 and int8 versions of a 1D-CNN, reporting a reduction from 58 to 26 KB with an estimated, not measured, latency on a Cortex-M4 target. The int8 scheme behind TensorFlow Lite is described by Jacob et al. (2018).

A smaller group puts tree models on constrained hardware. Soe et al. (2020) show that feature reduction lets classical classifiers run on a Raspberry Pi; Tekin et al. (2023) compare the energy and latency of several learners on cloud, edge and IoT devices and find a decision tree cheapest on-device; and Javed et al. (2024) embed a tree-based detector in a smart thermostat, with inference times below 0.3 ms. Closest to our study, De Bernardis et al. (2026) compare logistic regression, decision trees, random forests, gradient boosting and an MLP on TON-IoT and deploy them on two microcontrollers; among the ensembles, a quantised LightGBM model of 74 KB gave the best accuracy–size trade-off, and a depth-5 decision tree had the fastest inference (5.6 µs on an ESP32-C3).

Two gaps remain. Most studies that propose a compact network show that it is small, not that it is better than a tree ensemble held to the same size: none of the neural studies above reports a comparison with a size-limited tree model trained on the same features (we checked the full texts where we could obtain them and the abstracts otherwise). The studies that include trees train on one dataset and include no compact CNN of the kind proposed in the neural studies; De Bernardis et al. (2026) include a full-integer int8 MLP on TON-IoT, but not at matched size budgets or over several seeds. Footprint is also reported in incompatible units (parameters, kilobytes, relative reductions, estimated microcontroller latency), and seed variance is rarely given. We compare both families at several size budgets, in the same units, on three datasets and five seeds.

### 2.2 Trees and neural networks on tabular data

On tabular benchmarks, tree ensembles match or beat deep networks in most settings, particularly on medium-sized data with uninformative features (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) and with skewed or heavy-tailed feature distributions (McElfresh et al., 2023), and surveys conclude that gradient-boosted trees remain strong baselines (Borisov et al., 2024). Network-flow records have these properties: counts of bytes and packets span several orders of magnitude, and many fields are categorical. Random forests (Breiman, 2001) and gradient-boosted trees are therefore natural references for any neural detector on flow features, and their size can be bounded by the number and depth of trees.

### 2.3 Evaluation pitfalls in learning-based intrusion detection

Learning-based IDS are prone to evaluation pitfalls, including inappropriate baselines, spurious correlations and unrealistic performance measurements (Sommer & Paxson, 2010; Arp et al., 2022). Apruzzese et al. (2023) argue that assessments must account for deployment factors, hardware among them, and re-evaluate detectors on several platforms. Benchmark datasets contain construction and labelling errors (Engelen et al., 2021) and design flaws that bias experiments (Flood et al., 2024), and models trained on one dataset rarely transfer to another, because they learn dataset-specific feature–label associations (Booij et al., 2022; Layeghy & Portmann, 2023). Such associations are a form of shortcut learning (Geirhos et al., 2020). Metadata such as addresses and ports is a known example: including it contaminates the learned models (D'hooge et al., 2022). In a testbed, attack tools and victim services often use fixed ports, so a model can learn the testbed's port assignment instead of the attack's behaviour; as far as their texts or abstracts show, none of the lightweight-IDS studies in Section 2.1 tests whether its accuracy survives without such fields.

## 3. Data

### 3.1 Datasets and pre-processing

We use three public datasets with the pre-processing described below, which is released with the code. TON-IoT contains network flows from an IoT/IIoT testbed with nine attack classes (backdoor, DDoS, DoS, injection, MITM, password, ransomware, scanning, XSS) and benign traffic (Alsaedi et al., 2020; Moustafa, 2021); we use its Train_Test_Network file. UNSW-NB15 contains flows from an enterprise-style testbed with nine attack categories (Moustafa & Slay, 2015); we use the official training partition (175,341 records), which we split ourselves. It is not IoT traffic, and we keep it as a second, independent benchmark. N-BaIoT contains benign, BASHLITE and Mirai traffic of nine commercial IoT devices, described by 115 stream statistics computed by the Kitsune feature extractor (Meidan et al., 2018; Mirsky et al., 2018).

TON-IoT and UNSW-NB15 pass through the same pipeline, in this order. Identifier and free-text columns are removed; categorical fields with at most 30 distinct values are one-hot encoded and those with more are dropped; exact duplicates are then removed. The data are split into stratified training, validation and test sets (70/15/15). Numeric columns with more than 40% missing values and zero-variance columns are dropped, missing values are imputed with the median and infinite values are replaced by the 99th percentile, all estimated on the training split; validation and test records identical to a training record after these steps are removed (duplicates within the training split can reappear and are kept). Feature selection is fitted on the training split: a mutual-information filter (threshold 0.05) followed by recursive feature elimination with three-fold cross-validation (RFECV) on a random subsample of 50,000 training records, using a 50-tree random forest scored by macro-F1, with a floor of ten features. It keeps 25 features on TON-IoT and 10 on UNSW-NB15. The selected features are min–max scaled, and the training split is oversampled with SMOTE (Chawla et al., 2002) until every class has as many records as the largest; validation and test sets are never resampled. For N-BaIoT we use a uniform 3% sample of every device–class file, remove duplicates at 32-bit precision, drop the two BASHLITE flood classes (their records collapse to fewer than 100 distinct vectors), split 70/15/15 by device and class with the same cross-split guard, keep all features and do not oversample, because the classes are nearly balanced. Table 1 summarises the data.

| Dataset | Classes | Selected features | Training (after oversampling) | Validation | Test |
|---|---|---|---|---|---|
| TON-IoT | 10 | 25 | 128,484 (293,550) | 27,533 | 27,533 |
| UNSW-NB15 | 10 | 10 | 74,167 (363,220) | 15,894 | 15,894 |
| N-BaIoT | 9 | 115 | 110,227 (none) | 22,503 | 22,453 |

Because the cross-split guard compares all columns before feature selection, some test records coincide with a training record on the selected features (0.1% of the TON-IoT and 16.3% of the UNSW-NB15 test records); we repeat the main comparison on the test records without such a match. Two classes are small: TON-IoT's MITM class has 156 test records and UNSW-NB15's worms class 19, so their F1 scores are noisy. Kitsune features are statistics over sliding time windows, so neighbouring windows of one N-BaIoT device can fall into both training and test sets; N-BaIoT is therefore an easy case, and we use it to check that conclusions do not depend on a hard dataset.

## 4. Models and Protocol

### 4.1 Models and size budgets

Each model family is trained at several sizes, so that every configuration is a point on the accuracy–footprint plane (Table 2).

- **Logistic regression** (LR; multinomial, L2 penalty, C = 1) on the quantile-transformed features.
- **Random forests** (RF; scikit-learn, Pedregosa et al., 2011; √d features per split) with 25 or 100 trees of maximum depth 8, 12 or 16; 200 trees of unlimited depth; and the configuration of the earlier version (200 trees, depth 20, minimum leaf size 2). For the smallest budgets we add **single decision trees** of depth 4, 6, 8 or 12 and shallow forests of 10 or 25 trees of depth 4 or 6.
- **Gradient-boosted trees** (GBDT; scikit-learn's histogram-based implementation) with at most 7, 15, 31 or 63 leaves per tree, at learning rates 0.1 (the library default) and 0.05, up to 1,000 iterations with early stopping after 20 iterations without improvement of the log-loss on an internal 10% split of the oversampled training set. scikit-learn stops after those 20 iterations but keeps them in the model, and in some runs boosting diverged inside them: on TON-IoT, with 7 leaves, learning rate 0.1 and seed 0, the internal validation log-loss rose from 0.042 after 248 iterations to 361,323 after 268, and the stored model reached a test macro-F1 of 0.496. Every GBDT is therefore rolled back to the iteration with the best internal validation score, which gives the same model as refitting with that number of iterations. The learning rate of 0.05 was added after we had seen the instability at 0.1 and before we had found its cause; we report both (Table A.2). **Size-budgeted GBDTs** cap the iterations at 50, 100, 200 or 400 with 7, 15 or 31 leaves (learning rate 0.1, same early stopping and rollback), so that GBDTs also exist below 100 KB.
- **1D-CNN**: the compact network of the earlier version, which treats the feature vector as a one-channel sequence: two convolutional blocks (32 and 64 filters, kernel 3, batch normalisation, max-pooling), global average pooling, dense layers of 128 and 64 units with dropout 0.3, and a softmax layer (23,946 parameters on TON-IoT). Widths 0.5 and 2 scale every layer.
- **MLP**: dense layers of 128 and 64 units with batch normalisation, a 64-unit layer with dropout 0.3, and a softmax layer (17,162 parameters on TON-IoT), at the same three widths.

The networks are trained with Adam (learning rate 10⁻³, halved after 5 epochs without improvement of the validation loss, minimum 10⁻⁵), batch size 256, for at most 100 epochs with early stopping on the validation loss (patience 15); the weights of the epoch with the lowest validation loss are restored in every run. Their inputs are the selected features after a quantile transform to the uniform distribution (1,000 quantiles, fitted on a 200,000-record subsample of the oversampled training set; on all training records for N-BaIoT). The transform is monotone, so it does not change the tree models, which use the min–max scaled features. Section 5.5 shows why it matters for the networks.

**Quantisation.** Every trained network is also converted to TensorFlow Lite with post-training integer quantisation (int8 weights and activations with float input and output; the converter may keep unsupported operations in float), calibrated on 500 records of the oversampled training set, and evaluated on the same test set. As a runtime control, the seed-0 networks are also converted with full-integer quantisation (int8 operations only, int8 input and output; Section 5.4).

**Input-scaling control.** To measure the effect of the earlier version's input scaling, the CNN of width 1 is also trained on the min–max scaled features, with everything else unchanged, on TON-IoT and UNSW-NB15 (five seeds each).

The networks were trained on the GPU with XLA compilation. GPU kernels are not bit-reproducible and XLA does not use the seed for dropout masks, so a rerun with the same seed can stop at a different epoch and differ slightly in macro-F1; the five seeds absorb this variation together with the variation from initialisation.

### 4.2 Footprint and latency

Model size is measured in two ways. The **compact size**, used for all comparisons, is the size of a plain deployable encoding: for tree models, 10 bytes per internal node (a 16-bit feature index, a 32-bit threshold and two 16-bit child indices) plus, per leaf, the class probabilities as 16-bit floats (random forests and decision trees) or one 32-bit value (gradient boosting); for LR, its 32-bit weights; for networks, four bytes per parameter in float32 and the TensorFlow Lite file in int8. LR and the networks read quantile-transformed inputs (Section 4.1), and a deployed model must carry the transform, so their size also includes its table of 1,000 quantiles per feature as 32-bit values: 98 KB on TON-IoT, 39 KB on UNSW-NB15 and 449 KB on N-BaIoT. The tree models need no transform, since their splits can be expressed on the raw feature values. The table is the largest part of every int8 network; a smaller table (fewer quantiles, or a parametric transform) would shrink it, which we did not evaluate. This tree encoding is nominal: it counts the bytes a plain array layout would need, ignores the routing of missing values (there are none after pre-processing), and was not implemented as a runtime. The trees of the unlimited-depth forest on UNSW-NB15 have more than 65,535 nodes and would need wider child indices, so its compact size is slightly understated. The float32 TensorFlow Lite files of the networks differ from four bytes per parameter by −2.9 to +7.6 KB (Table 4). Tree models can be quantised as well: for the GBDTs we also report a **quantised encoding** with an 8-bit feature index, an 8-bit index into the histogram bin edges that scikit-learn already uses for training, two 16-bit child indices (6 bytes per internal node), a 16-bit float per leaf, and the 32-bit bin edges of the features used, stored once. The **serialised size** (the pickled scikit-learn object) is also reported, because it is what a practitioner who saves the model directly will see; it is 9.6–9.9 times the compact size for the random forests of the main grid and 8–16 times for the GBDTs, since scikit-learn stores 64-bit values and bookkeeping fields for every node.

Latency is measured on one thread of the laptop CPU (Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz) with the GPU hidden: the OpenMP, BLAS and TensorFlow thread counts are set to one before the libraries are loaded, and every timed call runs under a one-thread limit. Per record (batch of one), we take the median over five repeats of 200 calls after 20 warm-up calls (for GBDT seeds 1–4, three repeats of 100 calls after 10); per 1,000 records, the median of ten calls in one batch. Tree models are timed through scikit-learn's predict (the GBDTs for all five seeds, the other models for seed 0), float32 networks through TensorFlow and through the float32 TensorFlow Lite interpreter, and int8 networks through the TensorFlow Lite interpreter, with and without its default XNNPACK delegate. To separate the model from the runtime, we also time a LightGBM model, a tree library compiled to C++, with the same leaves, learning rate and number of iterations as the most accurate GBDT, trained on the same data; it is a different model, so we use it only for its cost per tree. One thread approximates a single-core gateway; absolute times on embedded processors will be larger, and Section 5.4 shows that the ranking of models by cost per decision depends on the runtime.

### 4.3 Port ablation

On TON-IoT the selected features include the source and destination port numbers. We retrain a medium random forest (100 trees, depth 16), the earlier forest configuration, GBDTs with 31 leaves at both learning rates and both networks at width 1 without these two features. The other selected features of both datasets are listed in Appendix A.

### 4.4 Evaluation

The primary metric is macro-F1 on the test set; we also report the benign false-positive rate (FPR, the share of benign test records classified as an attack), the attack detection rate (the share of attack records not classified as benign) and the macro-averaged average precision (the area under the one-vs-rest precision–recall curve). Every configuration is trained with five seeds (0–4) on the same data split. For each dataset we select, on the validation split, the configuration with the highest validation macro-F1 and the smallest configuration whose validation macro-F1 is within 0.01 of it, and report their test macro-F1 (mean over seeds); the validation score is the validation macro-F1 of the seed-0 model of every configuration (validation scores of the other network seeds were not recorded). Where the text names the configuration with the highest mean test macro-F1, this is descriptive, not a selection. Selected comparisons are tested with Welch's t-test over seeds (deterministic models such as LR and single decision trees have zero spread, and the test then compares the other side's seeds with that fixed value); the seeds do not change SMOTE or the quantile transform's subsample; the seeds share one data split, so these tests reflect training randomness only (Section 6.3), and they are not corrected for multiplicity. Differences are computed before rounding. Experiments ran on one laptop with an NVIDIA GTX 1650 GPU (4 GB), used only to train the networks, with scikit-learn 1.7.2 and TensorFlow 2.10.0.

## 5. Results

### 5.1 Overview

Table 2 shows selected configurations; Appendix A lists all 49 points per dataset (37 tree and LR configurations, and six networks in float32 and int8). Tree ensembles tie at the top on both hard datasets. The highest mean test macro-F1 is that of GBDT with 63 leaves at lr 0.1 on TON-IoT (0.962) and of GBDT with 31 leaves at lr 0.05 on UNSW-NB15 (0.608), and 5 and 9 tree configurations, respectively, including the best, are statistically tied (Welch p ≥ 0.05). The best network is 0.010 below on TON-IoT (CNN of width 2: 0.952; Welch p < 0.001) and 0.046 below on UNSW-NB15 (int8 CNN of width 2: 0.563; p < 0.001). On UNSW-NB15 the best network is an int8 model, +0.002 relative to its own float32 version, which shows that differences of this size are noise. On N-BaIoT 22 of the 29 configurations of the main grid (without the capped GBDTs and the small trees) have a mean of 0.998 or more (to three decimals), and the ranking carries little information. Logistic regression is far behind on the two hard datasets (0.874 and 0.405). On UNSW-NB15 every model classifies 11–23% of the benign test records as attacks (LR 23%) (Appendix A), far too many for operational use, and its macro-F1 of about 0.6 is pulled down by a few attack classes (lowest per-class F1 of the unbounded forest: backdoor 0.18, analysis 0.22, dos 0.36). The CNN of width 1 reaches 0.947 on TON-IoT, against 0.880 in our earlier draft (Section 5.5).

Three observations qualify the table. First, before rollback (Section 4.1) the GBDT at the default learning rate looked unstable: in 6 of 20 seed and leaf-count combinations on TON-IoT and 2 of 20 on UNSW-NB15, macro-F1 fell more than 0.05 below the mean at learning rate 0.05, down to 0.334, and at learning rate 0.05 one seed of the 7-leaf GBDT on TON-IoT fell to 0.798. These were models evaluated after boosting had diverged: in 17 of the 120 GBDT runs the internal validation loss at the stopping point was more than ten times its best value. After rollback the lowest seed on TON-IoT is 0.942 at learning rate 0.1 and 0.951 at 0.05 (0.574 and 0.597 on UNSW-NB15), and 0 of 20 combinations on TON-IoT and 0 of 20 on UNSW-NB15 remain more than 0.05 below the learning-rate-0.05 mean (Table A.2). Early stopping on the provided validation split, instead of an internal split of the oversampled training set, changes the best GBDT by −0.001 on TON-IoT and +0.003 on UNSW-NB15 (Welch p 0.118 and 0.062; lowest seed over the eight GBDT configurations 0.942 and 0.565). Second, the 1,000-iteration cap was reached in 14 of 20 learning-rate-0.05 runs on UNSW-NB15 and 4 of 20 on TON-IoT, so the size of these models is set by the cap, not by convergence. Third, the tree-over-network ranking holds on the test records that have no exact match in the training set on the selected features (16.3% of the UNSW-NB15 test set has such a match, 0.1% of TON-IoT's): on those records the tree model that is best on the full test set leads the best network by 0.010 on TON-IoT and 0.086 on UNSW-NB15 (network side in float32, because clean-record scores were not stored for the int8 models). The ranking within the trees can change on those records: the most accurate tree model there is GBDT with 63 leaves at lr 0.1 (0.962) on TON-IoT and RF with 200 trees of unlimited depth (0.667) on UNSW-NB15.

| Model | Encoding | TON-IoT macro-F1 | TON-IoT size (KB) | UNSW-NB15 macro-F1 | UNSW-NB15 size (KB) | N-BaIoT macro-F1 | N-BaIoT size (KB) |
|---|---|---|---|---|---|---|---|
| LR | compact | 0.874 ± 0.000 | 99 | 0.405 ± 0.000 | 39 | 0.979 ± 0.000 | 453 |
| RF, earlier configuration (200 trees, depth 20) | compact | 0.957 ± 0.000 | 10,171 | 0.605 ± 0.002 | 54,254 | 1.000 ± 0.000 | 726 |
| RF, 200 trees, depth unlimited | compact | 0.960 ± 0.001 | 22,486 | 0.606 ± 0.003 | 197,983 | 1.000 ± 0.000 | 770 |
| RF, 25 trees, depth 12 | compact | 0.946 ± 0.001 | 450 | 0.572 ± 0.004 | 744 | 0.999 ± 0.000 | 78 |
| RF, 100 trees, depth 16 | compact | 0.955 ± 0.000 | 4,173 | 0.603 ± 0.003 | 12,787 | 1.000 ± 0.000 | 372 |
| GBDT, 15 leaves, lr 0.1 | compact | 0.951 ± 0.004 | 110 | 0.594 ± 0.009 | 290 | 1.000 ± 0.000 | 154 |
| GBDT, 15 leaves, lr 0.05 | compact | 0.962 ± 0.001 | 1,628 | 0.606 ± 0.004 | 1,580 | 1.000 ± 0.000 | 276 |
| GBDT, 63 leaves, lr 0.1 | compact | 0.962 ± 0.001 | 891 | 0.606 ± 0.007 | 2,238 | 1.000 ± 0.000 | 473 |
| GBDT, 63 leaves, lr 0.05 | compact | 0.962 ± 0.001 | 1,918 | 0.601 ± 0.003 | 8,514 | 1.000 ± 0.000 | 952 |
| GBDT, 31 leaves, ≤ 50 iterations | compact | 0.955 ± 0.001 | 201 | 0.595 ± 0.003 | 207 | 1.000 ± 0.000 | 183 |
| GBDT, 7 leaves, ≤ 200 iterations | compact | 0.946 ± 0.005 | 96 | 0.580 ± 0.004 | 158 | 1.000 ± 0.000 | 100 |
| GBDT, 15 leaves, ≤ 50 iterations | compact | 0.949 ± 0.003 | 82 | 0.574 ± 0.003 | 97 | 1.000 ± 0.000 | 88 |
| GBDT, 31 leaves, lr 0.05 | compact | 0.960 ± 0.003 | 1,491 | 0.608 ± 0.002 | 4,112 | 1.000 ± 0.000 | 432 |
| GBDT, 7 leaves, lr 0.1 | compact | 0.947 ± 0.006 | 105 | 0.586 ± 0.010 | 198 | 1.000 ± 0.000 | 100 |
| Decision tree, depth 12 | compact | 0.942 ± 0.000 | 12 | 0.548 ± 0.000 | 27 | 0.996 ± 0.000 | 1.9 |
| CNN width 1 | float32 | 0.947 ± 0.001 | 191 | 0.550 ± 0.002 | 133 | 0.999 ± 0.000 | 543 |
| CNN width 1 (int8) | int8 | 0.947 ± 0.001 | 133 | 0.547 ± 0.003 | 75 | 0.998 ± 0.001 | 485 |
| MLP width 1 | float32 | 0.948 ± 0.001 | 165 | 0.553 ± 0.002 | 99 | 0.998 ± 0.001 | 561 |
| MLP width 1 (int8) | int8 | 0.946 ± 0.001 | 118 | 0.532 ± 0.025 | 58 | 0.989 ± 0.014 | 481 |

### 5.2 Accuracy against size

Figure 1 places every configuration on the accuracy–footprint plane. We compare the families at the size of the largest int8 network with its input table, 204 KB on TON-IoT and 145 KB on UNSW-NB15; this budget was set after the sizes were known, so that every int8 network competes. At that size the most accurate model on TON-IoT is GBDT with 31 leaves and at most 50 iterations (201 KB, 0.955), 0.004 above the best of the other family, int8 CNN of width 2 (204 KB, 0.951) (Welch p < 0.001); on UNSW-NB15 it is GBDT with 15 leaves and at most 50 iterations (97 KB, 0.574), 0.012 above int8 CNN of width 2 (145 KB, 0.563) (p 0.014). The TON-IoT outcome depends on this accounting. Without the table, the largest int8 network is 106 KB, and at that size the best network has the higher mean on TON-IoT (0.951 against 0.949 for the best tree model), while on UNSW-NB15 the best tree model stays ahead (0.574 against 0.563). The tree side of this comparison includes the capped GBDTs, single decision trees and shallow forests. These budget comparisons pick each family's representative by its test mean among many configurations, so they are descriptive and favour whichever family has more configurations near the budget; the validation-selected results follow below. The smallest tree model above the best network of any size is GBDT with 31 leaves and at most 50 iterations (201 KB, 0.955) on TON-IoT (p < 0.001) and GBDT with 7 leaves and at most 100 iterations (86 KB, 0.567) on UNSW-NB15 (p 0.282); the smallest significantly above it is GBDT with 31 leaves and at most 50 iterations (201 KB, 0.955) on TON-IoT (p < 0.001) and GBDT with 15 leaves and at most 50 iterations (97 KB, 0.574) on UNSW-NB15 (p 0.014). In the quantised encoding of Section 4.2 the GBDTs shrink by a factor of 1.2–1.7 on TON-IoT and UNSW-NB15 (on N-BaIoT, where the bin edges of 115 features must be stored, 12 of 20 GBDTs grow, by up to 2.5 times), and rounding their leaf values to 16 bits changes test macro-F1 by at most 0.000; at the same budget in that encoding, the best GBDT is GBDT with 31 leaves and at most 100 iterations (177 KB quantised, 0.958) on TON-IoT and GBDT with 31 leaves and at most 50 iterations (128 KB quantised, 0.595) on UNSW-NB15.

The Welch tests above vary only the training seed. Resampling the test records as well (a hierarchical bootstrap over seeds and test records, 2,000 replicates), the 95% interval for the lead of the best tree model over the best network is [+0.006, +0.013] on TON-IoT and [+0.033, +0.063] on UNSW-NB15, and for the smallest tree model significantly above every network [+0.000, +0.006] and [+0.003, +0.025]. On TON-IoT the lower end of the second interval is at zero, so that lead is small relative to test-sample variation. The ranking also held when the data changed. On five further stratified 70/15/15 splits (selected features kept from the original split; scaling, oversampling and the quantile transform refitted; one seed per split, so these replicates are descriptive), the best tree model led the best network on every split, by +0.009 on average on TON-IoT (at least +0.0070) and +0.042 on UNSW-NB15 (at least +0.0280), and so did the smallest tree model significantly above every network (+0.004, at least +0.0026; +0.022, at least +0.0089). Feature selection used labels of records that fall into the new test sets, which makes the replicates slightly optimistic for every model alike.

Selected on validation macro-F1, the best configuration is GBDT with 63 leaves at lr 0.1 (891 KB; validation 0.961, test 0.962) on TON-IoT and GBDT with 31 leaves at lr 0.05 (4,112 KB; validation 0.620, test 0.608) on UNSW-NB15. The smallest configuration within 0.01 of it on validation is GBDT with 7 leaves and at most 200 iterations (96 KB; validation 0.953, test 0.946) on TON-IoT, GBDT with 15 leaves at lr 0.05 (1,580 KB; validation 0.611, test 0.606) on UNSW-NB15 and a decision tree of depth 12 (1.9 KB; validation 0.996, test 0.996) on N-BaIoT. On TON-IoT this small configuration clears the validation threshold by 0.002, and on the test set it is within 0.01 of the selected best in 1 of 5 seeds; on UNSW-NB15 the margin is 0.001 and the share 5 of 5. Unbounded forests are accurate but large: 200 trees of unlimited depth reach 0.960 on TON-IoT at 22,486 KB, against 0.962 for the best tree model at 891 KB.

### 5.3 Quantisation

Post-training int8 quantisation shrinks the networks by a factor of 1.4–3.8, less than the nominal four because the TensorFlow Lite file also stores the graph and the quantisation parameters, which dominate the smallest models. For the CNNs the cost is small on every dataset: at most 0.002 macro-F1 on TON-IoT, 0.010 on UNSW-NB15 and 0.001 on N-BaIoT (mean over seeds), and at most 0.020 for a single seed. The MLPs lose more and less predictably: up to 0.021 on UNSW-NB15 and 0.016 on N-BaIoT on average, and up to 0.051 and 0.073 for single seeds. The int8 models agree with their float32 versions on at least 99.4% of TON-IoT test records and 89.7% of UNSW-NB15's (means over seeds; single seeds down to 98.9% and 85.3%). Agreement matters beyond macro-F1: at 89.7% agreement, about one UNSW-NB15 record in ten receives a different label from the int8 and the float32 version of the same MLP, so the alerts an operator sees change although macro-F1 changes little. The full-integer variant of Section 5.4 changes seed-0 macro-F1 by at most 0.000 relative to these models. We used one calibration set of 500 records and did not tune the quantisation; an int8 model should be evaluated per seed, not assumed equivalent to its float32 version.

### 5.4 Cost per decision

Table 3 gives the cost per record and per 1,000 records on one thread, and Table 4 the TensorFlow Lite variants of the networks. Through TensorFlow the networks take 0.35–0.47 ms per record in float32, with little dependence on width, because a single call is dominated by the framework's fixed overhead; through the float32 TensorFlow Lite interpreter the same models take 0.004–0.076 ms. The int8 models take 0.05–10.00 ms, and the CNNs of width 1 and 2 0.4–10.0 ms, more than in float32 TensorFlow Lite. Removing the default XNNPACK delegate changes the int8 times by at most 4%, so in TensorFlow Lite 2.10 on this CPU the accelerated path serves the float32 models while the int8 graphs run on the interpreter's own integer kernels. Full-integer models with int8 input and output take 0.06–10.07 ms, are faster than the float-fallback int8 models in 1 of 18 cases, and lose at most 0.020 seed-0 macro-F1 against float32. On this CPU, int8 therefore saves memory, not time, relative to float32 TensorFlow Lite; relative to float32 TensorFlow, whose fixed per-call overhead dominates, it is faster in some cases (Tables 3 and 4). The cost of a tree ensemble grows with the number of trees it evaluates: in scikit-learn the random forests take about 1.2–1.4 ms with 25 trees and 4.1–4.3 ms with 100, the GBDTs of the main grid 4.1–83.8 ms and the capped GBDTs 3.4–19.7 ms, about 7–10 µs per tree. The most accurate tree models are slower than every network: the best GBDT on TON-IoT evaluates 1,210 trees per record in seed 0 (580–1,210 over the seeds, one tree per class and iteration) and takes 9.8 ms (4.7–10.2 ms over the seeds), 23 times the best network; on UNSW-NB15 it takes 84 ms against 1.59 ms. The tree models statistically tied with the best, however, take 9.8–82.1 ms on TON-IoT and 6.7–83.8 ms on UNSW-NB15, and the fastest tree model within 0.005 of the best is GBDT with 31 leaves and at most 100 iterations (0.958, 4.02 ms) on TON-IoT and GBDT with 31 leaves and at most 200 iterations (0.608, 6.68 ms) on UNSW-NB15: the latency cost belongs to the choice of a large GBDT, not to accuracy as such. At the budget of Section 5.2, the best network takes 2.85 ms and the best tree model 4.13 ms per record on TON-IoT. Batching removes most of the per-call overhead: 1,000 records in one call cost the GBDTs 46–782 ms, 65–155 times less per record than single calls, the float32 networks 0.8–88.2 ms and the int8 networks 26.2–9630.3 ms. All of these times describe runtimes as much as models. The compiled tree library evaluates a tree in 0.07–0.14 µs, 59–108 times less than scikit-learn's per-tree loop; at that cost the best GBDT on TON-IoT would take about 0.16 ms per record. The cross-family latency ranking above is therefore a property of scikit-learn and TensorFlow on a laptop CPU, not of the model families.

| Model | TON-IoT ms/record | TON-IoT ms/1,000 | UNSW-NB15 ms/record | UNSW-NB15 ms/1,000 | N-BaIoT ms/record | N-BaIoT ms/1,000 |
|---|---|---|---|---|---|---|
| RF, 25 trees, depth 8 | 1.352 | 3.3 | 1.228 | 2.7 | 1.231 | 2.6 |
| RF, 25 trees, depth 12 | 1.230 | 3.3 | 1.251 | 3.2 | 1.222 | 2.7 |
| RF, 25 trees, depth 16 | 1.232 | 3.7 | 1.251 | 4.4 | 1.227 | 2.8 |
| RF, 100 trees, depth 8 | 4.136 | 10.9 | 4.127 | 10.0 | 4.103 | 9.4 |
| RF, 100 trees, depth 12 | 4.186 | 13.0 | 4.121 | 12.6 | 4.120 | 10.0 |
| RF, 100 trees, depth 16 | 4.200 | 14.5 | 4.257 | 17.9 | 4.187 | 10.7 |
| RF, 200 trees, depth unlimited | 8.090 | 32.8 | 8.552 | 66.7 | 8.049 | 20.7 |
| RF, earlier configuration (200 trees, depth 20) | 8.088 | 29.6 | 9.123 | 44.4 | 8.020 | 20.8 |
| GBDT, 7 leaves, lr 0.1 | 20.082 | 160.0 | 13.826 | 104.9 | 10.506 | 91.8 |
| GBDT, 15 leaves, lr 0.1 | 5.127 | 51.5 | 7.489 | 65.6 | 6.112 | 67.4 |
| GBDT, 31 leaves, lr 0.1 | 4.063 | 46.4 | 7.190 | 79.0 | 6.828 | 82.0 |
| GBDT, 63 leaves, lr 0.1 | 9.759 | 130.0 | 12.135 | 124.4 | 6.642 | 102.6 |
| GBDT, 7 leaves, lr 0.05 | 23.800 | 192.0 | 66.274 | 428.2 | 16.543 | 152.1 |
| GBDT, 15 leaves, lr 0.05 | 82.066 | 702.0 | 82.321 | 604.8 | 10.303 | 114.1 |
| GBDT, 31 leaves, lr 0.05 | 35.819 | 383.2 | 83.772 | 708.3 | 7.253 | 103.5 |
| GBDT, 63 leaves, lr 0.05 | 19.371 | 259.5 | 79.982 | 782.4 | 14.968 | 212.7 |
| LR | 0.075 | 0.2 | 0.075 | 0.2 | 0.076 | 0.5 |
| CNN width 0.5 | 0.400 | 6.5 | 0.398 | 3.0 | 0.409 | 27.3 |
| CNN width 0.5 (int8) | 0.216 | 184.3 | 0.121 | 88.0 | 0.747 | 717.9 |
| CNN width 1 | 0.410 | 11.2 | 0.394 | 4.9 | 0.432 | 45.6 |
| CNN width 1 (int8) | 0.764 | 658.3 | 0.420 | 318.9 | 2.664 | 2562.9 |
| CNN width 2 | 0.423 | 23.9 | 0.419 | 9.8 | 0.472 | 88.2 |
| CNN width 2 (int8) | 2.849 | 2501.5 | 1.589 | 1208.5 | 9.998 | 9630.3 |
| MLP width 0.5 | 0.360 | 0.8 | 0.357 | 0.8 | 0.353 | 1.0 |
| MLP width 0.5 (int8) | 0.059 | 32.7 | 0.049 | 26.2 | 0.116 | 67.8 |
| MLP width 1 | 0.356 | 1.1 | 0.356 | 1.1 | 0.361 | 1.3 |
| MLP width 1 (int8) | 0.174 | 101.8 | 0.153 | 89.5 | 0.289 | 171.9 |
| MLP width 2 | 0.367 | 2.2 | 0.369 | 2.0 | 0.410 | 3.3 |
| MLP width 2 (int8) | 0.588 | 353.3 | 0.547 | 327.6 | 0.843 | 494.0 |

| Network | Dataset | TFLite float32 (ms) | int8, float fallback (ms) | same, no XNNPACK (ms) | Full-integer int8 (ms) | File sizes f32 / int8 / full-int8 (KB) | Seed-0 macro-F1 f32 / int8 / full-int8 |
|---|---|---|---|---|---|---|---|
| CNN width 0.5 | TON-IoT | 0.007 | 0.216 | 0.216 | 0.230 | 32 / 17 / 17 | 0.938 / 0.938 / 0.938 |
| CNN width 0.5 | UNSW-NB15 | 0.006 | 0.121 | 0.122 | 0.133 | 32 / 17 / 17 | 0.529 / 0.508 / 0.508 |
| CNN width 0.5 | N-BaIoT | 0.014 | 0.747 | 0.751 | 0.769 | 32 / 17 / 17 | 0.999 / 0.999 / 0.999 |
| CNN width 1 | TON-IoT | 0.011 | 0.764 | 0.795 | 0.784 | 101 / 36 / 35 | 0.948 / 0.947 / 0.947 |
| CNN width 1 | UNSW-NB15 | 0.008 | 0.420 | 0.422 | 0.435 | 101 / 36 / 35 | 0.548 / 0.546 / 0.546 |
| CNN width 1 | N-BaIoT | 0.029 | 2.664 | 2.659 | 2.685 | 100 / 36 / 35 | 0.998 / 0.997 / 0.997 |
| CNN width 2 | TON-IoT | 0.026 | 2.849 | 2.853 | 2.895 | 369 / 106 / 106 | 0.952 / 0.950 / 0.950 |
| CNN width 2 | UNSW-NB15 | 0.016 | 1.589 | 1.580 | 1.594 | 369 / 106 / 106 | 0.568 / 0.564 / 0.564 |
| CNN width 2 | N-BaIoT | 0.076 | 9.998 | 10.010 | 10.075 | 369 / 106 / 106 | 0.999 / 0.998 / 0.998 |
| MLP width 0.5 | TON-IoT | 0.004 | 0.059 | 0.059 | 0.071 | 23 / 9.3 / 9.0 | 0.944 / 0.939 / 0.939 |
| MLP width 0.5 | UNSW-NB15 | 0.004 | 0.049 | 0.048 | 0.060 | 19 / 8.4 / 8.1 | 0.538 / 0.523 / 0.523 |
| MLP width 0.5 | N-BaIoT | 0.004 | 0.116 | 0.117 | 0.129 | 46 / 15 / 15 | 0.997 / 0.996 / 0.996 |
| MLP width 1 | TON-IoT | 0.005 | 0.174 | 0.173 | 0.186 | 67 / 21 / 20 | 0.948 / 0.947 / 0.947 |
| MLP width 1 | UNSW-NB15 | 0.005 | 0.153 | 0.154 | 0.166 | 60 / 19 / 19 | 0.555 / 0.565 / 0.565 |
| MLP width 1 | N-BaIoT | 0.006 | 0.289 | 0.287 | 0.302 | 112 / 32 / 32 | 0.997 / 0.994 / 0.994 |
| MLP width 2 | TON-IoT | 0.008 | 0.588 | 0.588 | 0.598 | 227 / 61 / 61 | 0.950 / 0.949 / 0.949 |
| MLP width 2 | UNSW-NB15 | 0.008 | 0.547 | 0.545 | 0.561 | 212 / 58 / 57 | 0.561 / 0.554 / 0.554 |
| MLP width 2 | N-BaIoT | 0.010 | 0.843 | 0.817 | 0.824 | 317 / 84 / 84 | 0.998 / 0.994 / 0.994 |

Table 5 converts these times into the number of flow records one core can classify per second, the quantity a gateway is sized by. Classifying one record per call, the most accurate GBDT in scikit-learn sustains 102 flows per second on TON-IoT and 12 on UNSW-NB15, the fastest tree model within 0.005 of it 249 and 150, and the most accurate network 2,366 and 629. At an illustrative 10,000 new flows per second, the most accurate GBDT on TON-IoT would need about 98 cores in this form. Batching changes the picture: with 1,000 records per call the same GBDT sustains 7,694 flows per second, the fastest near-best tree model 21,739 and the network 41,823, and a compiled tree library would bring the most accurate GBDT to about 6,093 flows per second one record at a time. A detector that can collect flows into batches, as a gateway exporting flow records at fixed intervals does, can run any model in Table 5 on one core at 827 flows per second or more; one that must decide on each flow as it ends cannot use a large GBDT in scikit-learn. These rates exclude feature extraction (Section 6.3).

| Model (TON-IoT; UNSW-NB15) | TON-IoT single | TON-IoT batched | UNSW-NB15 single | UNSW-NB15 batched |
|---|---|---|---|---|
| Most accurate tree model (GBDT with 63 leaves at lr 0.1; GBDT with 31 leaves at lr 0.05) | 102 | 7,694 | 12 | 1,412 |
| Fastest tree model within 0.005 of it (GBDT with 31 leaves and at most 100 iterations; GBDT with 31 leaves and at most 200 iterations) | 249 | 21,739 | 150 | 13,825 |
| Most accurate network (CNN width 2; CNN width 2 (int8)) | 2,366 | 41,823 | 629 | 827 |
| Most accurate tree model, compiled-library estimate | 6,093 | – | 738 | – |

### 5.5 Input-scaling and size-accounting pitfalls

Our own earlier, unpublished draft of this comparison set a CNN of 350 KB on disk against a random forest of 95 MB in pickled form on TON-IoT, and timed both on a GPU workstation. Three of its choices illustrate pitfalls that are easy to miss.

**CNN input scaling.** The earlier draft trained its CNN on min–max scaled features and reported macro-F1 0.880 on TON-IoT. In the present pipeline, the CNN of width 1 trained on min–max inputs reaches 0.900 ± 0.005 on TON-IoT, and the quantile transform adds +0.047 (Welch p < 0.001); on UNSW-NB15 it adds +0.020 (p 0.021). Byte and packet counts are heavy-tailed, so min–max scaling compresses most records into a narrow range near zero; the quantile transform spreads them over [0, 1]. The earlier comparison therefore understated the CNN. Part of the remaining gap between the earlier draft's 0.880 and the present min–max CNN comes from other pipeline changes, which we did not separate.

**LSTM.** The earlier LSTM read windows of ten consecutive records after a stratified shuffle. Consecutive records were then unrelated flows, so its inputs were sets of random flows, not sequences, and its score says nothing about temporal modelling. None of the datasets as split here preserves flow order, so we do not include a recurrent model.

**Forest size.** The earlier forest configuration (200 trees, depth 20) occupies 97,704 KB when pickled, the 95 MB of the earlier draft, but 10,171 KB in the compact encoding. The 350 KB of the earlier CNN was also a serialised size: the saved Keras file, about 3.7 times its 94 KB of float32 parameters. The forest's macro-F1 on TON-IoT, 0.957, is exceeded by the best tree model (0.962; Welch p < 0.001) with a compact size 11 times smaller, and 25 trees of depth 16 come within 0.004 of it at 1,028 KB. The size advantage claimed for the CNN came mostly from comparing it with an oversized forest in its pickled form.

### 5.6 Port numbers

Without the two port numbers, macro-F1 on TON-IoT falls by 0.015 for the medium forest, 0.015 for the earlier forest, 0.015 for the CNN and 0.014 for the MLP (Table 6). The rolled-back GBDTs with 31 leaves change by −0.013 at learning rate 0.05 and −0.015 at 0.1. Ransomware loses most for every forest and network (by 0.06 F1 for the earlier forest); for the GBDT at learning rate 0.05 the class that loses most is ransomware (by 0.05). The models therefore use the testbed's port assignment, mostly for one class, but most of their accuracy does not depend on it.

| Model | With ports | Without ports | Difference | Welch p |
|---|---|---|---|---|
| RF, 100 trees, depth 16 | 0.955 | 0.940 | −0.015 | < 0.001 |
| RF, earlier configuration (200 trees, depth 20) | 0.957 | 0.942 | −0.015 | < 0.001 |
| GBDT, 31 leaves, lr 0.1 | 0.958 | 0.942 | −0.015 | < 0.001 |
| GBDT, 31 leaves, lr 0.05 | 0.960 | 0.947 | −0.013 | < 0.001 |
| CNN width 1 | 0.947 | 0.932 | −0.015 | < 0.001 |
| MLP width 1 | 0.948 | 0.935 | −0.014 | < 0.001 |

## 6. Discussion

### 6.1 What "lightweight" should mean

"Lightweight" is not a property of a model family but of a budget, and of what the budget counts. At the size of the largest int8 network, size-budgeted trees were more accurate on TON-IoT and size-budgeted trees were more accurate on UNSW-NB15, and int8 quantisation costs the CNN almost nothing; on TON-IoT the trees lead only because the networks' input table is counted, which a deployment must store. From 201 KB on TON-IoT and 97 KB on UNSW-NB15, a tree model was significantly more accurate than every network we trained, the best tree by 0.010 and 0.046, and an unbounded forest was no more accurate than the best smaller tree model on all test records. Our earlier draft's evidence for a compact CNN rested on a forest far larger than its accuracy required, measured in its pickled form. Size and speed rank the families differently only under particular runtimes: in scikit-learn the most accurate GBDTs cost 23 times the per-record latency of the best network on TON-IoT, but smaller GBDTs within 0.005 of the best accuracy and a compiled tree library close most of that gap (Section 5.4). These statements concern the compact CNN and MLP architectures typical of the IoT intrusion-detection literature, trained with one fixed recipe at three widths; the tree families were explored far more widely (Section 6.3), and a tuned or modern tabular network may do better. A detector that must classify each flow on a single core within a few milliseconds can use a compact network, a small forest or a compiled GBDT; a large GBDT evaluated tree by tree in scikit-learn is the one option to avoid.

### 6.2 Recommendations

For studies that claim a lightweight detector we recommend:

1. Compare against tree ensembles held to the same size budget, at several budgets, not against an unbounded forest.
2. Report size in one deployable unit (bytes of a plain encoding or of the deployed file), and state how it was measured; the pickled size of a scikit-learn forest is about ten times its compact size.
3. Report single-thread CPU latency on the target class of hardware, with the thread count of every library fixed, and name the runtime; where possible, time trees through a compiled runtime as well.
4. Train a strong tree reference on the same features and treat a large gap to it as a sign of a training problem, as it was for the earlier CNN; check the reference itself across seeds, and evaluate early-stopped models at their best validation iteration (scikit-learn's histogram GBDT keeps the iterations after it), and regularise the leaf values: the divergence we saw is consistent with leaf values −G/(H + λ) growing without bound where λ = 0 (scikit-learn's default) and the Hessians are near zero in nearly pure, oversampled regions, although we did not test this mechanism.
5. Report several seeds, select configurations on validation data, and evaluate quantised models per seed.
6. Test whether accuracy survives without testbed-specific fields such as port numbers.

### 6.3 Limitations

Latency was measured on one thread of a laptop CPU, not on an embedded device; absolute times will differ on ARM gateways and microcontrollers, and microcontroller deployment would need further work (for example tree-to-C compilation or integer-only kernels). Feature extraction, which can dominate the cost of a flow-based detector, is not included. The size budgets are a grid, not an optimisation, so a better model may exist at any size. All three datasets are testbed or laboratory captures, UNSW-NB15 is not IoT traffic, and N-BaIoT is easy; conclusions about real deployments need data from operational networks. The five seeds vary network initialisation and batch order, the forests' bootstrap samples and the boosting's internal validation split, but not the data split or the oversampled training set, so the Welch tests capture training randomness only (five further data splits, one seed each, gave the same ranking; Section 5.2); test-sample variance, which matters for the small MITM and worms classes (a few records can move macro-F1 by about 0.01), enters only the bootstrap intervals of Section 5.2. Feature selection uses a random-forest wrapper, which may favour tree models, and on UNSW-NB15 it stopped at the floor of ten features. The networks were not tuned beyond the three widths (37 tree and LR configurations against six networks), and the GBDT only at two learning rates, so the comparison favours the more thoroughly explored family. The int8 timings come from TensorFlow Lite 2.10 on x86, whose integer kernels were not accelerated in this build; newer runtimes and ARM builds have optimised int8 paths, so the finding that int8 saves no time may not hold there. The compiled-tree evidence comes from a LightGBM proxy, not from the evaluated models.

## 7. Conclusion

We compared tree models and compact neural networks at several size budgets, in float32 and int8, on three datasets with five seeds, selecting configurations on validation data. Which family is better depends on the budget. At the size of the largest int8 network with its input table (204 KB on TON-IoT and 145 KB on UNSW-NB15), the best tree model was more accurate than the best network, by 0.004 and 0.012. From 201 KB on TON-IoT and 97 KB on UNSW-NB15, a tree model was significantly more accurate than every network, the best tree by 0.010 and 0.046 macro-F1. Int8 quantisation cost the CNNs at most 0.010 and the MLPs up to 0.021 macro-F1 on average over seeds. Per-record latency ranked the families by runtime more than by model: in scikit-learn the most accurate GBDTs were slower than every network (9.8 ms on TON-IoT against 0.42 ms for the best network), but the fastest tree model within 0.005 of the best accuracy, on TON-IoT a GBDT with 31 leaves and at most 100 iterations (0.958, 4.02 ms), and a compiled tree library were much faster. Early-stopped GBDTs in scikit-learn must be evaluated at their best validation iteration; without rollback they collapsed on some seeds. Removing port numbers cost about 0.015 macro-F1 on TON-IoT. Lightweight claims should be made against size-budgeted tree models, in deployable bytes and single-thread latency with a named runtime, over several seeds.

## Declarations

**Data availability.** TON-IoT, UNSW-NB15 and N-BaIoT are publicly available from their providers. The code, configurations and all result files are available at https://github.com/Amoakoemma123/iot-ids-pipeline. The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors.

**CRediT authorship contribution statement.** Emmanuel Kwame Amoako: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft, Writing – review & editing. Mahamudu Amadu: Conceptualization, Methodology, Validation. James Quayson: Conceptualization, Methodology, Writing – review & editing. Kingsford Donkor: Validation, Writing – review & editing. Daniel Asamoah Apraku: Validation, Writing – review & editing.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the authors used Claude (Anthropic) to write and edit the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The released scripts compute every reported number from the result files, and an automated check compares each quantitative statement in the text with those files.

## Appendix A. Full configuration grid

Selected features. TON-IoT (25): src_port, dst_port, duration, src_bytes, dst_bytes, src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes, dns_qclass, dns_qtype, dns_rcode, proto_tcp, proto_udp, service_dns, service_http, conn_state_rej, conn_state_s0, conn_state_s1, conn_state_sf, conn_state_sh, dns_RD_t, dns_RA_f, dns_rejected_f, dns_rejected_t. UNSW-NB15 (10): sbytes, dbytes, sttl, sload, dload, ackdat, smean, ct_state_ttl, ct_dst_src_ltm, ct_srv_dst. N-BaIoT: all 115 Kitsune statistics.

| Dataset | Model | Macro-F1 (mean ± SD) | Validation macro-F1 | Benign FPR | Attack detection | Macro PR-AUC | Compact size (KB) | Pickled size (KB) |
|---|---|---|---|---|---|---|---|---|
| TON-IoT | Decision tree, depth 4 | 0.552 ± 0.000 | 0.555 | 0.441 | 0.998 | 0.561 | 0.3 | 4.4 |
| TON-IoT | Decision tree, depth 6 | 0.836 ± 0.000 | 0.838 | 0.038 | 0.991 | 0.818 | 1.1 | 11 |
| TON-IoT | Decision tree, depth 8 | 0.900 ± 0.000 | 0.899 | 0.042 | 0.998 | 0.892 | 3.0 | 30 |
| TON-IoT | RF, 10 trees, depth 4 | 0.791 ± 0.030 | 0.791 | 0.161 | 0.987 | 0.888 | 4.4 | 46 |
| TON-IoT | RF, 25 trees, depth 4 | 0.797 ± 0.025 | 0.799 | 0.167 | 0.989 | 0.904 | 11 | 113 |
| TON-IoT | Decision tree, depth 12 | 0.942 ± 0.000 | 0.943 | 0.017 | 1.000 | 0.938 | 12 | 119 |
| TON-IoT | RF, 10 trees, depth 6 | 0.883 ± 0.007 | 0.884 | 0.037 | 0.996 | 0.930 | 15 | 146 |
| TON-IoT | RF, 25 trees, depth 6 | 0.897 ± 0.006 | 0.896 | 0.033 | 0.996 | 0.934 | 36 | 358 |
| TON-IoT | GBDT, 7 leaves, ≤ 50 iterations | 0.941 ± 0.001 | 0.940 | 0.019 | 0.998 | 0.961 | 43 | 438 |
| TON-IoT | GBDT, 7 leaves, ≤ 100 iterations | 0.944 ± 0.003 | 0.944 | 0.017 | 0.998 | 0.966 | 66 | 660 |
| TON-IoT | GBDT, 15 leaves, ≤ 50 iterations | 0.949 ± 0.003 | 0.947 | 0.014 | 0.999 | 0.974 | 82 | 740 |
| TON-IoT | GBDT, 7 leaves, ≤ 200 iterations | 0.946 ± 0.005 | 0.947 | 0.015 | 0.999 | 0.967 | 96 | 949 |
| TON-IoT | LR | 0.874 ± 0.000 | 0.872 | 0.038 | 0.997 | 0.898 | 99 | 2.7 |
| TON-IoT | RF, 25 trees, depth 8 | 0.923 ± 0.001 | 0.923 | 0.023 | 0.997 | 0.946 | 101 | 978 |
| TON-IoT | GBDT, 7 leaves, lr 0.1 | 0.947 ± 0.006 | 0.947 | 0.015 | 0.999 | 0.967 | 105 | 1,027 |
| TON-IoT | GBDT, 7 leaves, ≤ 400 iterations | 0.947 ± 0.006 | 0.947 | 0.015 | 0.999 | 0.967 | 105 | 1,027 |
| TON-IoT | MLP width 0.5 (int8) | 0.941 ± 0.001 | 0.938* | 0.019 | 0.999 | 0.961 | 107 | – |
| TON-IoT | GBDT, 15 leaves, ≤ 100 iterations | 0.950 ± 0.004 | 0.948 | 0.014 | 0.999 | 0.974 | 107 | 961 |
| TON-IoT | GBDT, 15 leaves, lr 0.1 | 0.951 ± 0.004 | 0.948 | 0.014 | 0.999 | 0.974 | 110 | 981 |
| TON-IoT | GBDT, 15 leaves, ≤ 200 iterations | 0.951 ± 0.004 | 0.948 | 0.014 | 0.999 | 0.974 | 110 | 981 |
| TON-IoT | GBDT, 15 leaves, ≤ 400 iterations | 0.951 ± 0.004 | 0.948 | 0.014 | 0.999 | 0.974 | 110 | 981 |
| TON-IoT | CNN width 0.5 (int8) | 0.938 ± 0.001 | 0.936* | 0.022 | 0.998 | 0.957 | 115 | – |
| TON-IoT | MLP width 1 (int8) | 0.946 ± 0.001 | 0.947* | 0.017 | 0.999 | 0.968 | 118 | – |
| TON-IoT | MLP width 0.5 | 0.944 ± 0.000 | 0.943* | 0.017 | 0.999 | 0.967 | 119 | – |
| TON-IoT | CNN width 0.5 | 0.940 ± 0.001 | 0.938* | 0.018 | 0.998 | 0.963 | 122 | – |
| TON-IoT | CNN width 1 (int8) | 0.947 ± 0.001 | 0.947* | 0.017 | 0.999 | 0.968 | 133 | – |
| TON-IoT | MLP width 2 (int8) | 0.950 ± 0.001 | 0.950* | 0.012 | 0.999 | 0.968 | 159 | – |
| TON-IoT | MLP width 1 | 0.948 ± 0.001 | 0.949* | 0.015 | 0.999 | 0.972 | 165 | – |
| TON-IoT | CNN width 1 | 0.947 ± 0.001 | 0.947* | 0.016 | 0.999 | 0.972 | 191 | – |
| TON-IoT | GBDT, 31 leaves, ≤ 50 iterations | 0.955 ± 0.001 | 0.955 | 0.010 | 0.999 | 0.978 | 201 | 1,703 |
| TON-IoT | CNN width 2 (int8) | 0.951 ± 0.001 | 0.948* | 0.014 | 0.999 | 0.970 | 204 | – |
| TON-IoT | GBDT, 31 leaves, ≤ 100 iterations | 0.958 ± 0.003 | 0.957 | 0.009 | 0.999 | 0.978 | 289 | 2,433 |
| TON-IoT | MLP width 2 | 0.951 ± 0.002 | 0.949* | 0.012 | 0.999 | 0.973 | 328 | – |
| TON-IoT | GBDT, 31 leaves, ≤ 200 iterations | 0.958 ± 0.003 | 0.957 | 0.009 | 0.999 | 0.978 | 360 | 3,029 |
| TON-IoT | GBDT, 31 leaves, lr 0.1 | 0.958 ± 0.003 | 0.957 | 0.009 | 0.999 | 0.978 | 384 | 3,225 |
| TON-IoT | GBDT, 31 leaves, ≤ 400 iterations | 0.958 ± 0.003 | 0.957 | 0.009 | 0.999 | 0.978 | 384 | 3,225 |
| TON-IoT | RF, 100 trees, depth 8 | 0.925 ± 0.001 | 0.925 | 0.023 | 0.997 | 0.953 | 402 | 3,892 |
| TON-IoT | RF, 25 trees, depth 12 | 0.946 ± 0.001 | 0.946 | 0.013 | 0.999 | 0.970 | 450 | 4,330 |
| TON-IoT | CNN width 2 | 0.952 ± 0.001 | 0.951* | 0.012 | 0.999 | 0.974 | 461 | – |
| TON-IoT | GBDT, 7 leaves, lr 0.05 | 0.957 ± 0.004 | 0.956 | 0.010 | 0.999 | 0.977 | 665 | 6,384 |
| TON-IoT | GBDT, 63 leaves, lr 0.1 | 0.962 ± 0.001 | 0.961 | 0.008 | 0.999 | 0.980 | 891 | 7,295 |
| TON-IoT | RF, 25 trees, depth 16 | 0.953 ± 0.000 | 0.952 | 0.011 | 1.000 | 0.977 | 1,028 | 9,878 |
| TON-IoT | GBDT, 31 leaves, lr 0.05 | 0.960 ± 0.003 | 0.959 | 0.008 | 0.999 | 0.980 | 1,491 | 12,456 |
| TON-IoT | GBDT, 15 leaves, lr 0.05 | 0.962 ± 0.001 | 0.961 | 0.007 | 0.999 | 0.978 | 1,628 | 14,221 |
| TON-IoT | RF, 100 trees, depth 12 | 0.947 ± 0.000 | 0.947 | 0.013 | 0.999 | 0.972 | 1,777 | 17,090 |
| TON-IoT | GBDT, 63 leaves, lr 0.05 | 0.962 ± 0.001 | 0.960 | 0.007 | 0.999 | 0.981 | 1,918 | 15,673 |
| TON-IoT | RF, 100 trees, depth 16 | 0.955 ± 0.000 | 0.953 | 0.010 | 1.000 | 0.978 | 4,173 | 40,090 |
| TON-IoT | RF, earlier configuration (200 trees, depth 20) | 0.957 ± 0.000 | 0.955 | 0.008 | 1.000 | 0.980 | 10,171 | 97,704 |
| TON-IoT | RF, 200 trees, depth unlimited | 0.960 ± 0.001 | 0.960 | 0.007 | 1.000 | 0.978 | 22,486 | 215,928 |
| UNSW-NB15 | Decision tree, depth 4 | 0.264 ± 0.000 | 0.263 | 0.223 | 0.999 | 0.346 | 0.4 | 4.9 |
| UNSW-NB15 | Decision tree, depth 6 | 0.474 ± 0.000 | 0.480 | 0.223 | 1.000 | 0.443 | 1.2 | 12 |
| UNSW-NB15 | Decision tree, depth 8 | 0.516 ± 0.000 | 0.511 | 0.199 | 0.999 | 0.528 | 3.5 | 35 |
| UNSW-NB15 | RF, 10 trees, depth 4 | 0.423 ± 0.024 | 0.417 | 0.226 | 0.999 | 0.451 | 4.3 | 45 |
| UNSW-NB15 | RF, 25 trees, depth 4 | 0.440 ± 0.010 | 0.437 | 0.224 | 0.999 | 0.475 | 11 | 112 |
| UNSW-NB15 | RF, 10 trees, depth 6 | 0.474 ± 0.010 | 0.473 | 0.209 | 0.999 | 0.517 | 14 | 142 |
| UNSW-NB15 | Decision tree, depth 12 | 0.548 ± 0.000 | 0.557 | 0.159 | 0.979 | 0.596 | 27 | 257 |
| UNSW-NB15 | RF, 25 trees, depth 6 | 0.480 ± 0.012 | 0.484 | 0.208 | 1.000 | 0.535 | 36 | 350 |
| UNSW-NB15 | LR | 0.405 ± 0.000 | 0.410 | 0.234 | 0.994 | 0.366 | 39 | 1.6 |
| UNSW-NB15 | GBDT, 7 leaves, ≤ 50 iterations | 0.552 ± 0.003 | 0.548 | 0.200 | 0.998 | 0.609 | 43 | 427 |
| UNSW-NB15 | MLP width 0.5 (int8) | 0.526 ± 0.004 | 0.522* | 0.199 | 0.998 | 0.580 | 47 | – |
| UNSW-NB15 | CNN width 0.5 (int8) | 0.518 ± 0.009 | 0.514* | 0.201 | 0.999 | 0.579 | 56 | – |
| UNSW-NB15 | MLP width 0.5 | 0.535 ± 0.002 | 0.531* | 0.198 | 0.999 | 0.609 | 57 | – |
| UNSW-NB15 | MLP width 1 (int8) | 0.532 ± 0.025 | 0.556* | 0.185 | 0.990 | 0.578 | 58 | – |
| UNSW-NB15 | CNN width 0.5 | 0.528 ± 0.005 | 0.531* | 0.200 | 0.999 | 0.599 | 64 | – |
| UNSW-NB15 | CNN width 1 (int8) | 0.547 ± 0.003 | 0.545* | 0.199 | 0.999 | 0.606 | 75 | – |
| UNSW-NB15 | GBDT, 7 leaves, ≤ 100 iterations | 0.567 ± 0.002 | 0.569 | 0.197 | 0.997 | 0.622 | 86 | 832 |
| UNSW-NB15 | MLP width 2 (int8) | 0.546 ± 0.007 | 0.542* | 0.189 | 0.995 | 0.593 | 97 | – |
| UNSW-NB15 | GBDT, 15 leaves, ≤ 50 iterations | 0.574 ± 0.003 | 0.577 | 0.196 | 0.997 | 0.634 | 97 | 861 |
| UNSW-NB15 | MLP width 1 | 0.553 ± 0.002 | 0.553* | 0.192 | 0.997 | 0.628 | 99 | – |
| UNSW-NB15 | RF, 25 trees, depth 8 | 0.517 ± 0.012 | 0.518 | 0.198 | 1.000 | 0.586 | 106 | 1,029 |
| UNSW-NB15 | CNN width 1 | 0.550 ± 0.002 | 0.543* | 0.198 | 0.999 | 0.624 | 133 | – |
| UNSW-NB15 | CNN width 2 (int8) | 0.563 ± 0.007 | 0.569* | 0.195 | 0.998 | 0.607 | 145 | – |
| UNSW-NB15 | GBDT, 7 leaves, ≤ 200 iterations | 0.580 ± 0.004 | 0.585 | 0.190 | 0.996 | 0.633 | 158 | 1,506 |
| UNSW-NB15 | GBDT, 15 leaves, ≤ 100 iterations | 0.590 ± 0.005 | 0.590 | 0.184 | 0.993 | 0.645 | 185 | 1,619 |
| UNSW-NB15 | GBDT, 7 leaves, lr 0.1 | 0.586 ± 0.010 | 0.588 | 0.184 | 0.994 | 0.634 | 198 | 1,889 |
| UNSW-NB15 | GBDT, 7 leaves, ≤ 400 iterations | 0.586 ± 0.010 | 0.588 | 0.184 | 0.994 | 0.634 | 198 | 1,889 |
| UNSW-NB15 | GBDT, 31 leaves, ≤ 50 iterations | 0.595 ± 0.003 | 0.595 | 0.182 | 0.992 | 0.648 | 207 | 1,740 |
| UNSW-NB15 | MLP width 2 | 0.557 ± 0.004 | 0.554* | 0.179 | 0.991 | 0.631 | 254 | – |
| UNSW-NB15 | GBDT, 15 leaves, ≤ 200 iterations | 0.594 ± 0.010 | 0.595 | 0.176 | 0.991 | 0.648 | 268 | 2,330 |
| UNSW-NB15 | GBDT, 15 leaves, lr 0.1 | 0.594 ± 0.009 | 0.595 | 0.175 | 0.990 | 0.648 | 290 | 2,525 |
| UNSW-NB15 | GBDT, 15 leaves, ≤ 400 iterations | 0.594 ± 0.009 | 0.595 | 0.175 | 0.990 | 0.648 | 290 | 2,525 |
| UNSW-NB15 | CNN width 2 | 0.561 ± 0.006 | 0.561* | 0.189 | 0.996 | 0.623 | 402 | – |
| UNSW-NB15 | GBDT, 31 leaves, ≤ 100 iterations | 0.606 ± 0.003 | 0.602 | 0.164 | 0.986 | 0.655 | 404 | 3,374 |
| UNSW-NB15 | RF, 100 trees, depth 8 | 0.523 ± 0.002 | 0.526 | 0.198 | 1.000 | 0.595 | 416 | 4,029 |
| UNSW-NB15 | GBDT, 31 leaves, ≤ 200 iterations | 0.608 ± 0.004 | 0.606 | 0.155 | 0.981 | 0.655 | 610 | 5,078 |
| UNSW-NB15 | RF, 25 trees, depth 12 | 0.572 ± 0.004 | 0.577 | 0.192 | 0.998 | 0.630 | 744 | 7,149 |
| UNSW-NB15 | GBDT, 31 leaves, lr 0.1 | 0.608 ± 0.004 | 0.607 | 0.151 | 0.979 | 0.653 | 785 | 6,534 |
| UNSW-NB15 | GBDT, 31 leaves, ≤ 400 iterations | 0.608 ± 0.004 | 0.607 | 0.151 | 0.979 | 0.653 | 785 | 6,534 |
| UNSW-NB15 | GBDT, 7 leaves, lr 0.05 | 0.602 ± 0.005 | 0.602 | 0.164 | 0.986 | 0.653 | 811 | 7,657 |
| UNSW-NB15 | GBDT, 15 leaves, lr 0.05 | 0.606 ± 0.004 | 0.609 | 0.147 | 0.979 | 0.663 | 1,580 | 13,642 |
| UNSW-NB15 | GBDT, 63 leaves, lr 0.1 | 0.606 ± 0.007 | 0.609 | 0.136 | 0.972 | 0.652 | 2,238 | 18,247 |
| UNSW-NB15 | RF, 100 trees, depth 12 | 0.578 ± 0.004 | 0.579 | 0.192 | 0.998 | 0.632 | 2,926 | 28,121 |
| UNSW-NB15 | RF, 25 trees, depth 16 | 0.601 ± 0.002 | 0.605 | 0.175 | 0.993 | 0.653 | 3,218 | 30,898 |
| UNSW-NB15 | GBDT, 31 leaves, lr 0.05 | 0.608 ± 0.002 | 0.614 | 0.127 | 0.968 | 0.657 | 4,112 | 34,130 |
| UNSW-NB15 | GBDT, 63 leaves, lr 0.05 | 0.601 ± 0.003 | 0.611 | 0.118 | 0.962 | 0.648 | 8,514 | 69,354 |
| UNSW-NB15 | RF, 100 trees, depth 16 | 0.603 ± 0.003 | 0.605 | 0.177 | 0.993 | 0.654 | 12,787 | 122,783 |
| UNSW-NB15 | RF, earlier configuration (200 trees, depth 20) | 0.605 ± 0.002 | 0.616 | 0.150 | 0.984 | 0.657 | 54,254 | 520,899 |
| UNSW-NB15 | RF, 200 trees, depth unlimited | 0.606 ± 0.003 | 0.609 | 0.110 | 0.959 | 0.629 | 197,983 | 1,900,701 |
| N-BaIoT | Decision tree, depth 4 | 0.701 ± 0.000 | 0.703 | 0.014 | 1.000 | 0.754 | 0.3 | 4.2 |
| N-BaIoT | Decision tree, depth 6 | 0.947 ± 0.000 | 0.950 | 0.005 | 1.000 | 0.932 | 0.7 | 7.9 |
| N-BaIoT | Decision tree, depth 8 | 0.969 ± 0.000 | 0.972 | 0.001 | 1.000 | 0.981 | 1.0 | 11 |
| N-BaIoT | Decision tree, depth 12 | 0.996 ± 0.000 | 0.996 | 0.001 | 1.000 | 0.999 | 1.9 | 20 |
| N-BaIoT | RF, 10 trees, depth 4 | 0.850 ± 0.033 | 0.852 | 0.002 | 0.999 | 0.924 | 3.6 | 40 |
| N-BaIoT | RF, 10 trees, depth 6 | 0.954 ± 0.010 | 0.956 | 0.001 | 1.000 | 0.977 | 8.7 | 88 |
| N-BaIoT | RF, 25 trees, depth 4 | 0.838 ± 0.018 | 0.840 | 0.002 | 0.999 | 0.945 | 9.0 | 96 |
| N-BaIoT | RF, 25 trees, depth 6 | 0.952 ± 0.006 | 0.955 | 0.001 | 1.000 | 0.984 | 22 | 221 |
| N-BaIoT | GBDT, 7 leaves, ≤ 50 iterations | 0.999 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 39 | 606 |
| N-BaIoT | RF, 25 trees, depth 8 | 0.991 ± 0.002 | 0.992 | 0.000 | 1.000 | 0.999 | 41 | 410 |
| N-BaIoT | GBDT, 7 leaves, ≤ 100 iterations | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 77 | 978 |
| N-BaIoT | RF, 25 trees, depth 12 | 0.999 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 78 | 765 |
| N-BaIoT | GBDT, 15 leaves, ≤ 50 iterations | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 88 | 1,000 |
| N-BaIoT | RF, 25 trees, depth 16 | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 93 | 910 |
| N-BaIoT | GBDT, 7 leaves, lr 0.1 | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 100 | 1,222 |
| N-BaIoT | GBDT, 7 leaves, ≤ 200 iterations | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 100 | 1,222 |
| N-BaIoT | GBDT, 7 leaves, ≤ 400 iterations | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 100 | 1,222 |
| N-BaIoT | GBDT, 15 leaves, ≤ 100 iterations | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 140 | 1,462 |
| N-BaIoT | GBDT, 15 leaves, ≤ 200 iterations | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 153 | 1,615 |
| N-BaIoT | GBDT, 15 leaves, lr 0.1 | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 154 | 1,634 |
| N-BaIoT | GBDT, 15 leaves, ≤ 400 iterations | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 154 | 1,634 |
| N-BaIoT | RF, 100 trees, depth 8 | 0.990 ± 0.001 | 0.992 | 0.000 | 1.000 | 0.999 | 164 | 1,625 |
| N-BaIoT | GBDT, 31 leaves, ≤ 50 iterations | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 183 | 1,756 |
| N-BaIoT | GBDT, 7 leaves, lr 0.05 | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 195 | 2,142 |
| N-BaIoT | GBDT, 31 leaves, ≤ 100 iterations | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 252 | 2,352 |
| N-BaIoT | GBDT, 31 leaves, ≤ 200 iterations | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 255 | 2,404 |
| N-BaIoT | GBDT, 31 leaves, lr 0.1 | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 256 | 2,417 |
| N-BaIoT | GBDT, 31 leaves, ≤ 400 iterations | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 256 | 2,417 |
| N-BaIoT | GBDT, 15 leaves, lr 0.05 | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 276 | 2,653 |
| N-BaIoT | RF, 100 trees, depth 12 | 0.999 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 310 | 3,047 |
| N-BaIoT | RF, 100 trees, depth 16 | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 372 | 3,640 |
| N-BaIoT | GBDT, 31 leaves, lr 0.05 | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 432 | 3,841 |
| N-BaIoT | LR | 0.979 ± 0.000 | 0.978 | 0.007 | 1.000 | 0.993 | 453 | 8.9 |
| N-BaIoT | MLP width 0.5 (int8) | 0.989 ± 0.008 | 0.998* | 0.001 | 1.000 | 0.998 | 464 | – |
| N-BaIoT | CNN width 0.5 (int8) | 0.998 ± 0.000 | 0.999* | 0.000 | 1.000 | 1.000 | 466 | – |
| N-BaIoT | GBDT, 63 leaves, lr 0.1 | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 473 | 4,164 |
| N-BaIoT | CNN width 0.5 | 0.999 ± 0.000 | 0.999* | 0.000 | 1.000 | 1.000 | 474 | – |
| N-BaIoT | MLP width 1 (int8) | 0.989 ± 0.014 | 0.994* | 0.000 | 1.000 | 0.999 | 481 | – |
| N-BaIoT | CNN width 1 (int8) | 0.998 ± 0.001 | 0.998* | 0.000 | 1.000 | 1.000 | 485 | – |
| N-BaIoT | MLP width 0.5 | 0.997 ± 0.000 | 0.998* | 0.000 | 1.000 | 1.000 | 493 | – |
| N-BaIoT | MLP width 2 (int8) | 0.982 ± 0.033 | 0.996* | 0.001 | 1.000 | 0.989 | 533 | – |
| N-BaIoT | CNN width 1 | 0.999 ± 0.000 | 0.999* | 0.000 | 1.000 | 1.000 | 543 | – |
| N-BaIoT | CNN width 2 (int8) | 0.998 ± 0.001 | 0.999* | 0.000 | 1.000 | 1.000 | 555 | – |
| N-BaIoT | MLP width 1 | 0.998 ± 0.001 | 0.998* | 0.000 | 1.000 | 1.000 | 561 | – |
| N-BaIoT | RF, earlier configuration (200 trees, depth 20) | 1.000 ± 0.000 | 0.999 | 0.000 | 1.000 | 1.000 | 726 | 7,114 |
| N-BaIoT | MLP width 2 | 0.999 ± 0.001 | 0.998* | 0.001 | 1.000 | 1.000 | 769 | – |
| N-BaIoT | RF, 200 trees, depth unlimited | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 770 | 7,542 |
| N-BaIoT | CNN width 2 | 0.999 ± 0.000 | 0.999* | 0.000 | 1.000 | 1.000 | 812 | – |
| N-BaIoT | GBDT, 63 leaves, lr 0.05 | 1.000 ± 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 952 | 8,055 |

| Dataset | Leaves | lr 0.1 as run: mean (min seed) | lr 0.1 rolled back: mean (min seed) | lr 0.05 rolled back: mean (min seed) | Iterations kept (lr 0.1; lr 0.05) |
|---|---|---|---|---|---|
| TON-IoT | 7 | 0.750 (0.441) | 0.947 (0.942) | 0.957 (0.951) | 58–248; 293–1000 |
| TON-IoT | 15 | 0.827 (0.334) | 0.951 (0.944) | 0.962 (0.960) | 26–106; 768–999 |
| TON-IoT | 31 | 0.927 (0.867) | 0.958 (0.954) | 0.960 (0.955) | 49–224; 91–460 |
| TON-IoT | 63 | 0.935 (0.825) | 0.962 (0.961) | 0.962 (0.961) | 58–121; 197–253 |
| UNSW-NB15 | 7 | 0.589 (0.580) | 0.586 (0.574) | 0.602 (0.598) | 137–321; 761–1000 |
| UNSW-NB15 | 15 | 0.591 (0.584) | 0.594 (0.582) | 0.606 (0.601) | 74–258; 459–1000 |
| UNSW-NB15 | 31 | 0.591 (0.548) | 0.608 (0.604) | 0.608 (0.605) | 89–306; 966–1000 |
| UNSW-NB15 | 63 | 0.557 (0.365) | 0.606 (0.598) | 0.601 (0.597) | 69–679; 999–1000 |
| N-BaIoT | 7 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 105–178; 223–338 |
| N-BaIoT | 15 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 62–241; 130–230 |
| N-BaIoT | 31 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 53–230; 92–171 |
| N-BaIoT | 63 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 51–269; 109–219 |

## References

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., & Anwar, A. (2020). TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems. *IEEE Access, 8*, 165130–165150. https://doi.org/10.1109/ACCESS.2020.3022862

Apruzzese, G., Laskov, P., & Schneider, J. (2023). SoK: Pragmatic assessment of machine learning for network intrusion detection. In *2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P)* (pp. 592–614). IEEE. https://doi.org/10.1109/EuroSP57164.2023.00042

Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., & Rieck, K. (2022). Dos and don'ts of machine learning in computer security. In *31st USENIX Security Symposium* (pp. 3971–3988). USENIX Association. https://www.usenix.org/conference/usenixsecurity22/presentation/arp

Booij, T. M., Chiscop, I., Meeuwissen, E., Moustafa, N., & den Hartog, F. T. H. (2022). ToN_IoT: The role of heterogeneity and the need for standardization of features and attack types in IoT network intrusion data sets. *IEEE Internet of Things Journal, 9*(1), 485–496. https://doi.org/10.1109/JIOT.2021.3085194

Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2024). Deep neural networks and tabular data: A survey. *IEEE Transactions on Neural Networks and Learning Systems, 35*(6), 7499–7519. https://doi.org/10.1109/TNNLS.2022.3229161

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

De Bernardis, E. P., Kuznetsov, O., Arnesano, M., Zhansaya, P., & Sydykova, M. (2026). Lightweight machine learning intrusion detection for IoT/IIoT networks: Quantisation strategies and physical deployment on resource-constrained microcontrollers. *Electronics, 15*(13), Article 2869. https://doi.org/10.3390/electronics15132869

D'hooge, L., Verkerken, M., Volckaert, B., Wauters, T., & De Turck, F. (2022). Establishing the contaminating effect of metadata feature inclusion in machine-learned network intrusion detection models. In *Detection of Intrusions and Malware, and Vulnerability Assessment (DIMVA 2022)* (pp. 23–41). Springer. https://doi.org/10.1007/978-3-031-09484-2_2

Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection dataset: The CICIDS2017 case study. In *2021 IEEE Security and Privacy Workshops (SPW)* (pp. 7–12). IEEE. https://doi.org/10.1109/SPW53761.2021.00009

Flood, R., Engelen, G., Aspinall, D., & Desmet, L. (2024). Bad design smells in benchmark NIDS datasets. In *2024 IEEE 9th European Symposium on Security and Privacy (EuroS&P)* (pp. 658–675). IEEE. https://doi.org/10.1109/EuroSP60621.2024.00042

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence, 2*(11), 665–673. https://doi.org/10.1038/s42256-020-00257-z

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://doi.org/10.52202/068431-0037

Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2704–2713). IEEE. https://doi.org/10.1109/CVPR.2018.00286

Javed, A., Awais, M. N., Qureshi, A.-H., Jawad, M., Arshad, J., & Larijani, H. (2024). Embedding tree-based intrusion detection system in smart thermostats for enhanced IoT security. *Sensors, 24*(22), Article 7320. https://doi.org/10.3390/s24227320

Lahasan, B., & Samma, H. (2022). Optimized deep autoencoder model for Internet of Things intruder detection. *IEEE Access, 10*, 8434–8448. https://doi.org/10.1109/ACCESS.2022.3144208

Layeghy, S., & Portmann, M. (2023). Explainable cross-domain evaluation of ML-based network intrusion detection systems. *Computers and Electrical Engineering, 108*, Article 108692. https://doi.org/10.1016/j.compeleceng.2023.108692

Mazinani, A., Antonucci, D., Davoli, L., & Ferrari, G. (2026). Performance assessment of DL for network intrusion detection on a constrained IoT device. *Future Internet, 18*(1), Article 34. https://doi.org/10.3390/fi18010034

McElfresh, D., Khandagale, S., Valverde, J., Prasad C, V., Ramakrishnan, G., Goldblum, M., & White, C. (2023). When do neural nets outperform boosted trees on tabular data? In *Advances in Neural Information Processing Systems 36* (pp. 76336–76369). Curran Associates. https://doi.org/10.52202/075280-3337

Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Shabtai, A., Breitenbacher, D., & Elovici, Y. (2018). N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders. *IEEE Pervasive Computing, 17*(3), 12–22. https://doi.org/10.1109/MPRV.2018.03367731

Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). Kitsune: An ensemble of autoencoders for online network intrusion detection. In *Network and Distributed System Security Symposium (NDSS 2018)*. https://doi.org/10.14722/ndss.2018.23204

Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society, 72*, 102994. https://doi.org/10.1016/j.scs.2021.102994

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In *2015 Military Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE. https://doi.org/10.1109/MilCIS.2015.7348942

Musthafa, M. B., Huda, S., Nguyen, T. T., Kodera, Y., & Nogami, Y. (2025). Optimized ensemble deep learning for real-time intrusion detection on resource-constrained Raspberry Pi devices. *IEEE Access, 13*, 113544–113556. https://doi.org/10.1109/ACCESS.2025.3584373

Ngo, D.-M., Lightbody, D., Temko, A., Pham-Quoc, C., Tran, N.-T., Murphy, C. C., & Popovici, E. (2023). HH-NIDS: Heterogeneous hardware-based network intrusion detection framework for IoT security. *Future Internet, 15*(1), Article 9. https://doi.org/10.3390/fi15010009

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Sharmila, B. S., & Nagapadma, R. (2023). Quantized autoencoder (QAE) intrusion detection system for anomaly detection in resource-constrained IoT devices using RT-IoT2022 dataset. *Cybersecurity, 6*(1), Article 41. https://doi.org/10.1186/s42400-023-00178-5

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011

Soe, Y. N., Feng, Y., Santosa, P. I., Hartanto, R., & Sakurai, K. (2020). Towards a lightweight detection system for cyber attacks in the IoT environment using corresponding features. *Electronics, 9*(1), Article 144. https://doi.org/10.3390/electronics9010144

Sommer, R., & Paxson, V. (2010). Outside the closed world: On using machine learning for network intrusion detection. In *2010 IEEE Symposium on Security and Privacy* (pp. 305–316). IEEE. https://doi.org/10.1109/SP.2010.25

Tekin, N., Acar, A., Aris, A., Uluagac, A. S., & Gungor, V. C. (2023). Energy consumption of on-device machine learning models for IoT intrusion detection. *Internet of Things, 21*, Article 100670. https://doi.org/10.1016/j.iot.2022.100670

Ullah, S., Wu, J., Kamal, M. M., Alzaylaee, M. K., & Mohamed, H. G. (2026). Quantized intrusion detection for resource-constrained IoT: A comparative evaluation of efficiency and adversarial robustness. *Computers, Materials & Continua, 88*(3). https://doi.org/10.32604/cmc.2026.084409

Zakariyya, I., Kalutarage, H., & Al-Kadri, M. O. (2023). Towards a robust, effective and resource efficient machine learning technique for IoT security monitoring. *Computers & Security, 133*, Article 103388. https://doi.org/10.1016/j.cose.2023.103388

Zhao, R., Gui, G., Xue, Z., Yin, J., Ohtsuki, T., Adebisi, B., & Gacanin, H. (2022). A novel intrusion detection method based on lightweight neural network for Internet of Things. *IEEE Internet of Things Journal, 9*(12), 9960–9972. https://doi.org/10.1109/JIOT.2021.3119055
