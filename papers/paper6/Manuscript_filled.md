# Paper 6 v3 manuscript (double-brace placeholders are filled from values.json by build.py)

## Title

**Fusing Feature Groups of One Flow Record: A Controlled Evaluation of Multi-Stream Intrusion Detection on IoT and Network Benchmarks**

## Abstract

Intrusion detection studies increasingly encode semantic groups of flow features with separate neural branches and fuse them, reporting gains for this "multimodal" design. Such gains can also come from more features, extra parameters or a weak baseline. We test multi-stream fusion with controls for each on TON-IoT, the only non-saturated IoT benchmark in the study, on the enterprise UNSW-NB15 and on the saturated N-BaIoT: two semantic streams, compact 1D-CNN encoders, early, late and attention fusion, five seeds, same-size random feature splits, capacity-matched networks and tree ensembles. One CNN on all features (early fusion) exceeded the better single stream by +0.044 macro-F1 on TON-IoT and +0.060 on UNSW-NB15; a random forest (RF) gained +0.032 and +0.061 from the same features. Late fusion added +0.005 on TON-IoT (significant after Holm correction), a gain that late fusion with both encoders on all features largely reproduced, and +0.016 on UNSW-NB15 (not significant after Holm correction). Attention fusion added nothing significant. Semantic instead of random streams of the same sizes changed late-fusion macro-F1 by +0.001 on TON-IoT and +0.009 on UNSW-NB15, where an effect up to about 0.02 cannot be excluded. Tuned tree ensembles beat every untuned fusion network in macro-F1 except on N-BaIoT, but the tuned forests occupy 221–636 MB against 343 kB for late fusion, and a 4.4 MB forest scored below late fusion on TON-IoT. Attention weights reflected routing, not stream usefulness. With min–max inputs, the volume-stream CNN failed silently on TON-IoT (macro-F1 0.028); a tree reference exposed the failure.

**Keywords:** intrusion detection; Internet of Things; multimodal fusion; feature groups; deep learning; random forest; evaluation methodology

## 1. Introduction

Machine-learning intrusion detection systems (IDS) for the Internet of Things (IoT) are usually trained on flow records, vectors of a few dozen statistics computed per connection. A growing line of work treats subsets of these statistics as separate "modalities" or "views": byte and packet counts in one branch, protocol, port and connection-state fields in another, each encoded by its own network and fused by concatenation or attention. The motivation is borrowed from multimodal learning on images, audio and text (Baltrušaitis et al., 2019), where separate encoders are needed because the inputs differ in structure. Reported results are usually favourable: the fused model beats each single branch and often a single network on all features.

Such comparisons are hard to interpret. A fused model sees more features than either branch, so beating a branch says little about fusion. Fusion heads add parameters, so a gain over a single network may come from capacity rather than from the grouping. If the grouping matters, a semantic split should outperform an arbitrary one of the same sizes, a control we have not seen reported. And tree ensembles remain strong on tabular data (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022), so a fusion network that is not compared with one on the same features may be solving a problem that does not exist. These are instances of the evaluation pitfalls documented for machine learning in security (Arp et al., 2022).

We carried out this comparison with these controls in place. The study replaces an earlier, unpublished version of this work, which reported a dual-stream framework evaluated on a single seed; on re-running it with a corrected data pipeline we found that its central observation, a flow-feature branch that "fails" on TON-IoT, was a training failure caused by input scaling (Section 5.5). Our contributions are:

1. A controlled evaluation of early, late and attention fusion of two semantic feature streams with five seeds, paired comparisons and a Holm-corrected family of tests, on TON-IoT (the only non-saturated IoT benchmark in the study), on UNSW-NB15 as a non-IoT replication and on N-BaIoT as a ceiling check.
2. Controls that, to our knowledge, have not been combined in fusion-IDS work: random feature splits of the same sizes, single networks matched to late fusion in parameters and head, late fusion with both encoders on all features, tuned tree ensembles on all features and on each stream, an order-independent encoder, and repeated data splits.
3. An analysis of attention weights as explanations: whether they follow which stream classifies a class better, and whether they are stable across seeds.
4. A documented silent failure: with min–max scaled, heavy-tailed flow counts the batch-normalised CNN learns almost nothing from the volume stream, without any error; a tree reference on the same features is a cheap and reliable way to detect it.

Section 2 reviews related work, Section 3 describes the data and streams, Section 4 the models and protocol, Section 5 the results, and Section 6 discusses implications and limitations.

## 2. Related Work

### 2.1 Fusion of feature groups in intrusion detection

Learning-based IDS for IoT have moved from single networks on flow features (Diro & Chilamkurti, 2018; Imrana et al., 2021; see the surveys of Chaabouni et al., 2019, and Lansky et al., 2021) to designs with several input branches. Some fuse inputs that differ in kind. Kiflay et al. (2024) combine six flow features with the first payload bytes of each UNSW-NB15 flow, using two random forests and decision-level fusion; Xu et al. (2025) fuse a byte-level image of the first packets of a flow with the flow's statistics; Ullah et al. (2024) fuse text-like and image-like encodings of packet contents; and Singh et al. (2026) combine network and host audit data. Others derive their branches from one flow table, which is the setting we test. Attota et al. (2021) train an ensemble of networks on multiple views of IoT network data in a federated multi-view IDS; Ayantayo et al. (2023) split UNSW-NB15 and NSL-KDD features by data type and compare early and late fusion; Dinh et al. (2024a) give each feature group its own sub-encoder in a multi-input autoencoder; Sanjalawe et al. (2025) use a dual-stream attention network on selected IIoT features; and AboulEla and Kashef (2026) fuse numeric, text, graph and quantum-inspired representations of the same NetFlow records. Attention over features is common (Al Qathrady et al., 2024; Ishtiaq et al., 2025), and some "dual-path" designs feed the same full feature vector to both paths (Alabbadi & Bajaber, 2025). Reported accuracies on IoT benchmarks are often near 100% (Ishtiaq et al., 2025; Afraji et al., 2025, for binary classification), where differences between architectures cannot be resolved.

The comparisons offered for fusion are usually against the model's own single branches. We examined 14 recent fusion, multi-view and attention-based IDS papers (Appendix A, Table A.1; two of them, Al Qathrady et al., 2024, and Ishtiaq et al., 2025, are single-network attention designs). Of the eight read in full text, none reported a random grouping of features or a single network with a matched parameter count, two trained a tree ensemble on the same data, and none reported results over several seeds or repeated runs (one gives a ten-fold cross-validation mean without its spread, and one names the absence of repeated runs as a limitation). The abstracts of the other six mention neither of the first two controls nor repeated runs, and one of them (Singh et al., 2026) mentions a tree ensemble; absence from an abstract is weak evidence. The three tree comparisons are informative. In Singh et al. (2026), tree ensembles on the fused features reached higher accuracy than the fused CNN; in an extended version of the multi-input autoencoder study (Dinh et al., 2024b), a random forest on the original features scored the same F-score on UNSW-NB15 as the multi-input autoencoder followed by a random forest (0.727 against 0.728); and Kiflay et al. (2024) found decision-level fusion better than one forest on the concatenated features, a positive result for fusing inputs that differ in kind. AboulEla and Kashef (2026) report only modest, dataset-dependent gains, with no fusion model above the best single representation on one dataset. Li et al. (2025), comparing multi-view and single-view deep learning on TON-IoT and UNSW-NB15 in host and network scenarios, found multi-view models generally, but not always, better.

The closest studies to ours are Ayantayo et al. (2023), who compare early and late fusion of data-type groups of UNSW-NB15 features, and Li et al. (2025). We differ from the first in holding the parameter count and the head fixed and in replacing the semantic grouping by random ones, and from the second in that our streams come from one flow record rather than from different sources. Our question is narrower than theirs: given one set of flow features, does grouping and fusing them help?

### 2.2 Multimodal learning

Multimodal learning combines inputs of different structure, such as images, audio and text (Baltrušaitis et al., 2019; Ramachandram & Taylor, 2017). Separate encoders are needed there because a convolution suited to images does not suit text, and fusion may be early (inputs), late (representations or decisions) or attention-based (Bahdanau et al., 2015; Vaswani et al., 2017; Nagrani et al., 2021). Even with true modalities, joint training is not automatically better than the best single modality (Wang et al., 2020), and one modality can dominate optimisation and leave the other under-used (Peng et al., 2022; Wu et al., 2022; Huang et al., 2022). Du et al. (2023) show that jointly trained multimodal networks can learn weaker uni-modal features than separately trained ones, and that ensembles of uni-modal models can match joint training; Liang et al. (2024) survey these questions. Attention weights, often presented as explanations, need not agree with other measures of importance (Jain & Wallace, 2019), although whether they can serve as explanations depends on how they are tested (Wiegreffe & Pinter, 2019).

### 2.3 Baselines and evaluation in ML-based security

On tabular data, tree ensembles remain strong competitors of deep networks (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022; McElfresh et al., 2023). Security-specific pitfalls of machine-learning evaluation, including inappropriate baselines and spurious correlations, have been documented repeatedly (Sommer & Paxson, 2010; Arp et al., 2022; Apruzzese et al., 2023), and benchmark datasets themselves contain errors (Engelen et al., 2021). Our study applies these lessons to fusion architectures: it asks what the fusion adds once the features, the parameter count and a strong tabular baseline are controlled.

## 3. Data and Feature Streams

### 3.1 Datasets and pre-processing

TON-IoT contains network flows from an IoT/IIoT testbed with nine attack classes (backdoor, DDoS, DoS, injection, MITM, password, ransomware, scanning, XSS) and benign traffic (Alsaedi et al., 2020; Moustafa, 2021). UNSW-NB15 contains flows from an enterprise-style testbed with nine attack categories (Moustafa & Slay, 2015); it is not an IoT dataset, and we keep it as a second, independent benchmark. Both pass through the corrected pre-processing pipeline of our companion studies, released with this paper, in this order. Identifier and free-text columns are removed; categorical fields with at most 30 distinct values (on TON-IoT protocol, service, connection state and DNS flags) are one-hot encoded (the category sets are fixed on the full data before the split, without labels, and the first category is dropped) and those with more are dropped; exact duplicates are removed after all column drops. After the split described next, numeric columns with more than 40% missing values in the training split are dropped, remaining gaps are filled with the training median, infinite values are replaced by the training 99th percentile, and constant features are removed. The data are then split into stratified training, validation and test sets (70/15/15), and validation and test records identical to a training record (all columns and the label) are removed. Feature selection is fitted on the training split: a mutual-information filter (threshold 0.05) followed by recursive feature elimination with three-fold cross-validation (RFECV), which ranks features with a 50-tree random forest, scores subsets by macro-F1 on a 50,000-record subsample and keeps at least ten features. The selected features are min–max scaled (fitted on the training split), and the training split is oversampled with SMOTE (Chawla et al., 2002); validation and test sets are never resampled. SMOTE interpolates between records, so synthetic records can carry fractional values in one-hot columns. On UNSW-NB15 the cross-validated macro-F1 of RFECV was highest at its floor (0.554 with ten features against 0.526 with all 40), so only ten features remain and the streams there are small.

N-BaIoT contains benign, BASHLITE and Mirai traffic of nine commercial IoT devices, described by 115 stream statistics computed by the Kitsune feature extractor (Meidan et al., 2018; Mirsky et al., 2018). We use the preparation of our federated-learning study: a uniform 3% sample of every device–class file, duplicates removed at 32-bit precision, the two BASHLITE flood classes dropped (their records collapse to fewer than 100 distinct vectors), a 70/15/15 split by device and class with the same cross-split guard, min–max scaling fitted on the training split, all features kept and no oversampling, because the classes are nearly balanced. Kitsune features are statistics over sliding time windows, so neighbouring windows of one device can fall into both the training and the test set; together with its saturated scores (Section 5.1), this makes N-BaIoT a ceiling check rather than a test of the architectures. Table 1 summarises the three datasets.

| Dataset | Classes | Features (stream 1 + stream 2) | Training (after SMOTE) | Validation | Test | Stream 1 | Stream 2 |
|---|---|---|---|---|---|---|---|
| TON-IoT | 10 | 25 (7 + 18) | 128,484 (293,550) | 27,533 | 27,533 | volume and timing: duration, bytes and packets per direction | ports, protocol, service, connection state, DNS fields |
| UNSW-NB15 | 10 | 10 (6 + 4) | 74,167 (363,220) | 15,894 | 15,894 | volume and timing: bytes and load per direction, mean source packet size, ACK delay | TTL and connection-count fields (sttl, ct_state_ttl, ct_dst_src_ltm, ct_srv_dst) |
| N-BaIoT | 9 | 115 (30 + 85) | 110,227 (no SMOTE) | 22,503 | 22,453 | host-level statistics (MI_dir_*, H_*) | channel and socket statistics (HH_*, HH_jit_*, HpHp_*) |

Every model is evaluated on the same test records of a split. Two classes are small: TON-IoT's MITM class has 156 test records and UNSW-NB15's worms class 19, so their F1 scores are noisy. Because the cross-split guard compares all columns before feature selection, some test records still coincide with a training record on the selected features (0.1% of the TON-IoT and 16.3% of the UNSW-NB15 test records). We report results on the full test set and repeat the main comparisons on the test records without such a match (Section 5.2).

### 3.2 Semantic streams

Each dataset's selected features are split into two streams by a fixed rule (Table 1). The rule was fixed before any model of this study was trained, but it is the split of the earlier version of this work, whose results we had seen. On TON-IoT and UNSW-NB15, stream 1 holds the volume and timing of the flow (duration, bytes and packets per direction, load, mean packet size, ACK delay) and stream 2 everything else: on TON-IoT ports, protocol, service, connection state and DNS fields, on UNSW-NB15 the time-to-live and connection-count fields. Grouping the features of one flow table into separately encoded subsets is common in the fusion literature, although with other rules (Ayantayo et al., 2023, group by data type; Dinh et al., 2024a, give each feature group its own sub-encoder); we know of no study that uses exactly this volume-versus-behaviour split. On N-BaIoT, stream 1 holds the host-level statistics (Kitsune's MI_dir and H groups, aggregated over all traffic from the source) and stream 2 the channel and socket statistics (HH, HH_jit and HpHp groups), a split defined by the feature extractor itself. The streams are unequal in size: 7 and 18 features on TON-IoT, 6 and 4 on UNSW-NB15, and 30 and 85 on N-BaIoT.

## 4. Models and Protocol

### 4.1 Encoders and fusion

Every stream is encoded by the same compact 1D-CNN, which treats the feature vector as a one-channel sequence: a convolution with 64 filters (kernel 3) and batch normalisation, max-pooling, a convolution with 128 filters and batch normalisation, global average pooling and a 64-unit dense embedding. Five architectures are compared (Figure 1):

- **Stream 1 only** and **stream 2 only**: one encoder on one stream, dropout 0.3 and a softmax layer.
- **Early fusion**: the same single encoder on the concatenated features of both streams, which is simply one CNN on all features (34,634 parameters on TON-IoT).
- **Late fusion**: one encoder per stream, the two embeddings concatenated and passed through a 128-unit dense layer, dropout and softmax (85,770 parameters).
- **Attention fusion**: one encoder per stream; a dense layer with a two-way softmax on the concatenated embeddings produces one weight per stream and record, the embeddings are summed with these weights and passed through a 64-unit dense layer, dropout and softmax (73,036 parameters, fewer than late fusion because of the smaller head). The weight of stream 1 is computed for every test record; class means and within-class spreads are saved.

All networks are trained with Adam (learning rate 10⁻³, halved after 4 epochs without improvement of the validation loss, minimum 10⁻⁵), batch size 512, on the oversampled training set without class weights, for at most 120 epochs with early stopping on validation loss (patience 10). The weights of the epoch with the lowest validation loss are restored in every run, including runs that reach the budget; 1 of the 90 network runs of the main and capacity experiments used the full budget (N-BaIoT stream 2, seed 4, best epoch 118, so it was still improving at the cap; N-BaIoT is saturated and outside the confirmatory family). Network inputs are the selected features after a quantile transform to the uniform distribution (1,000 quantiles, fitted on a 200,000-record subsample of the oversampled training set, or on the whole training set of N-BaIoT, which is smaller and not oversampled); Section 5.5 explains why. Fitting the transform on the oversampled set lets synthetic SMOTE records shape the quantiles; fitting it on real records only would have been an equally valid choice.

### 4.2 Controls

**Tree ensembles.** A random forest (RF; scikit-learn, Pedregosa et al., 2011; 200 trees) and a histogram gradient-boosting classifier (GBDT; up to 1,000 iterations with early stopping after 20 iterations without improvement on an internal 10% split) are trained on all features and on each stream, on the min–max scaled, oversampled training set. Their hyperparameters (RF: maximum depth 20 or unlimited, minimum leaf size 1 or 2; GBDT: learning rate 0.05 or 0.1, 31 or 127 leaves) are chosen on the validation macro-F1 with all features and then fixed. Every GBDT is evaluated at its best internal-validation iteration: scikit-learn stops 20 iterations after it but keeps those iterations, in which boosting can diverge. On UNSW-NB15 the GBDT ran to 966–1,000 iterations, at or near the cap. Tree splits depend only on the order of values within a feature, so a monotone transform such as the quantile transform would change the trees little (split thresholds and the histogram bins of the GBDT can still move); we therefore train them on the min–max features. The networks, in contrast, use one fixed configuration and were not tuned per dataset (Section 6.3). Because the RFECV selector itself uses a random forest, the selected features may suit tree models; we return to this in Section 6.3.

**Capacity.** Late fusion has about 2.5 times as many parameters as the single network, and a different head. Three further networks separate these factors: early fusion with every layer widened by a factor of 1.6 (**wide early fusion**, 86,613 parameters on TON-IoT, against 85,770 for late fusion); early fusion with the same 128-unit head as late fusion (**early fusion with head**); and late fusion in which both encoders see all features (**late fusion on all features**), which keeps the late-fusion architecture but removes the grouping.

**Random feature splits.** To test whether the semantic grouping matters, the features are also split at random into two streams of the same sizes as the semantic ones (10 splits per dataset), in the spirit of the random subspace method (Ho, 1998). The single-stream networks, late and attention fusion and the per-stream trees are trained on every random split; split r uses seed r mod 5, so each seed occurs with exactly two random splits. Of the 120 random-split network runs, 104 (and the 15 N-BaIoT runs of the stream-1, early- and late-fusion networks on the semantic split) are taken from an earlier run; they include all 30 late-fusion random-split runs and 17 of the 20 attention-fusion random-split runs on TON-IoT and UNSW-NB15, whereas every semantic-split late- and attention-fusion run on these two datasets is new. The earlier run was the same experiment except for a 60-epoch cap and for restoring the best weights only when early stopping fired. These runs stopped early before the cap, so they were scored with the weights of their best epoch, and the cap was never binding; they are therefore valid runs of the present protocol. They are not, however, the runs the present code would produce with the same seed: of 51 runs that stopped early in the earlier run and were repeated with the same seed, 9 reached their best epoch at the same point, and macro-F1 changed by 0.003 on average (at most 0.018), because GPU training is not deterministic; per dataset the change was 0.001 on average on TON-IoT (at most 0.002; 21 runs), 0.006 on UNSW-NB15 (at most 0.018; 25 runs) and 0.000 on N-BaIoT (5 runs). Seed pairing between reused and new runs is therefore nominal, and this run-to-run variation is of the order of the smaller effects we report. Their predictions were not saved, so they have no PR-AUC or duplicate-free score; the latter equals macro-F1 on N-BaIoT, where no test record matches a training record.

**Column order and encoder type.** A 1D convolution depends on the order of its inputs, so a random split changes not only the grouping but also which features are neighbours. Early fusion is therefore repeated on five random orders of all features (order k trained with seed k, so order and seed are confounded; we compare the spread over orders with the spread over seeds). On TON-IoT and UNSW-NB15, all five architectures and the random-split experiment are also repeated with a multilayer-perceptron encoder (two 128-unit layers with batch normalisation and a 64-unit embedding), which does not depend on feature order.

**Data splits.** Seeds vary initialisation, batch order and dropout, but not the data. On TON-IoT and UNSW-NB15 the experiment is therefore repeated on five further stratified 70/15/15 splits (single-stream, early, late and attention networks, the RF on all features and on each stream, and the GBDT on all features; one seed per split). The selected features are kept from the original split, and scaling, oversampling and the quantile transform are refitted on each new training split. Feature selection used labels of records that now fall into the new test sets; this affects all models alike but makes the replicates slightly optimistic.

### 4.3 Evaluation

The primary metric is macro-F1 on the test set; we also report the benign false-positive rate (FPR, the share of benign test records classified as an attack), the attack detection rate and the macro-averaged area under the precision–recall curve (one-vs-rest). On the original split, each configuration is trained with five seeds (0–4), and the same seed is used for every architecture. GPU training is not bit-wise deterministic, so repeated runs with one seed can differ slightly. Networks are compared by paired differences over seeds, reported as the mean with a t-based 95% confidence interval and the number of seeds in which the difference is positive; differences below 10⁻⁶ count as ties. Tree ensembles and networks share no initialisation, so they are compared with Welch's t-test. The confirmatory family was fixed before the experiments reported here were run. It consists of four paired t-tests on each of TON-IoT and UNSW-NB15: late against early fusion, attention against early fusion, late fusion against wide early fusion, and late fusion against late fusion on all features. The first two were planned from the start; the two capacity comparisons were added after review of an earlier draft, before their runs were made. Holm correction is applied over these 8 tests. N-BaIoT is at the ceiling and is not part of the family, and all other comparisons are descriptive. In particular, the direct comparison of semantic and random splits (Section 5.3), which addresses the central grouping question, is descriptive; late fusion against late fusion on all features is its confirmatory proxy. Appendix B (Table B.1) lists the whole family with unadjusted and Holm-adjusted p-values; the confidence intervals reported in the text and tables are unadjusted. A non-significant test is not evidence of no effect. We therefore also read each difference against a margin of ±0.01 macro-F1, which we regard as too small to matter for deployment: a difference whose confidence interval lies inside the margin is called negligible, and one whose interval crosses it is called inconclusive. The margin was chosen after the first results were seen, so it is a reading aid, not a pre-registered equivalence test. The five further data splits give split-level replicates: the original split (mean over seeds) and the five new splits form six paired observations per comparison. Semantic and random splits are compared within seeds: for each seed, the semantic run minus the mean of the two random splits trained with that seed, giving five paired differences. Experiments ran on one laptop (NVIDIA GPU for network training); inference latency was measured separately on its CPU with no other jobs running.

## 5. Results

### 5.1 Overview

Table 2 reports all models on the semantic split. On TON-IoT the volume stream is the stronger one (CNN 0.904, RF 0.928) and the protocol stream the weaker (0.829, RF 0.796); the fusion models score between 0.948 and 0.953, and the tree ensembles on all features score highest (RF 0.961, GBDT 0.962; benign FPR 0.007 and 0.008). UNSW-NB15 is hard for every model: the best CNN reaches 0.573 and the RF 0.607, and all misclassify a large share of benign records (FPR 0.127–0.205); 82% of the benign records misclassified by late fusion, and 82% of those misclassified by the RF, are labelled fuzzers, so with ten selected features benign and fuzzer flows overlap for every model. On N-BaIoT every model, including each stream alone, scores between 0.994 and 1.000; this dataset cannot separate the architectures, and we report it as a ceiling case. For the five main CNNs the standard deviations over seeds are small on TON-IoT (at most 0.001) and larger on UNSW-NB15 (up to 0.011, for early fusion; some control networks in Table 2 vary more), so on TON-IoT differences of a few thousandths can be systematic, although they are not necessarily meaningful in practice. PR-AUC ranks the models broadly as macro-F1 does, with some ties; the benign FPR does not always follow macro-F1 (on UNSW-NB15 the GBDT has a lower FPR than the RF and attention fusion a higher one than early fusion). Table B.3 in Appendix B gives macro-F1, PR-AUC, detection rate and FPR for every model of Table 2. Macro PR-AUC on TON-IoT is 0.974, 0.975 and 0.974 for early, late and attention fusion against 0.978 for the RF and 0.981 for the GBDT, and on UNSW-NB15 0.621, 0.635 and 0.632 against 0.651 and 0.657. The attack detection rate (attack records flagged as any attack) is at least 0.996 for every model on TON-IoT (at least 0.997 for the five main CNNs) and 0.968 on UNSW-NB15, so the models differ mainly in which attack class they assign and in false alarms.

| Model | Params (TON-IoT) | TON-IoT macro-F1 | TON-IoT FPR | UNSW-NB15 macro-F1 | UNSW-NB15 FPR | N-BaIoT macro-F1 | N-BaIoT FPR |
|---|---|---|---|---|---|---|---|
| Stream 1 only (CNN) | 34,634 | 0.904 ± 0.001 | 0.073 | 0.497 ± 0.004 | 0.205 | 1.000 ± 0.000 | 0.001 |
| Stream 2 only (CNN) | 34,634 | 0.829 ± 0.001 | 0.047 | 0.361 ± 0.003 | 0.201 | 0.994 ± 0.001 | 0.004 |
| Early fusion (CNN, all features) | 34,634 | 0.948 ± 0.001 | 0.016 | 0.557 ± 0.011 | 0.188 | 0.999 ± 0.000 | 0.000 |
| Late fusion | 85,770 | 0.953 ± 0.001 | 0.011 | 0.573 ± 0.001 | 0.186 | 1.000 ± 0.000 | 0.000 |
| Attention fusion | 73,036 | 0.949 ± 0.001 | 0.016 | 0.565 ± 0.003 | 0.195 | 1.000 ± 0.000 | 0.000 |
| Wide early fusion (×1.6) | 86,613 | 0.950 ± 0.000 | 0.014 | 0.565 ± 0.005 | 0.185 | – | – |
| Early fusion with head | 43,594 | 0.951 ± 0.001 | 0.013 | 0.563 ± 0.003 | 0.186 | – | – |
| Late fusion on all features | 85,770 | 0.951 ± 0.001 | 0.012 | 0.562 ± 0.006 | 0.184 | – | – |
| MLP stream 1 | 27,466 | 0.904 ± 0.002 | 0.077 | 0.498 ± 0.002 | 0.202 | – | – |
| MLP stream 2 | 28,874 | 0.829 ± 0.001 | 0.047 | 0.374 ± 0.019 | 0.202 | – | – |
| MLP early fusion | 29,770 | 0.948 ± 0.001 | 0.015 | 0.552 ± 0.011 | 0.188 | – | – |
| MLP late fusion | 72,842 | 0.953 ± 0.001 | 0.011 | 0.569 ± 0.009 | 0.184 | – | – |
| MLP attention fusion | 60,108 | 0.951 ± 0.001 | 0.014 | 0.564 ± 0.014 | 0.197 | – | – |
| RF, all features | – | 0.961 ± 0.001 | 0.007 | 0.607 ± 0.003 | 0.147 | 1.000 ± 0.000 | 0.000 |
| RF, stream 1 | – | 0.928 ± 0.000 | 0.016 | 0.546 ± 0.001 | 0.151 | 1.000 ± 0.000 | 0.000 |
| RF, stream 2 | – | 0.796 ± 0.000 | 0.039 | 0.368 ± 0.000 | 0.193 | 0.998 ± 0.000 | 0.000 |
| GBDT, all features | – | 0.962 ± 0.001 | 0.008 | 0.608 ± 0.002 | 0.127 | 1.000 ± 0.000 | 0.000 |
| GBDT, stream 1 | – | 0.925 ± 0.001 | 0.045 | 0.540 ± 0.002 | 0.155 | 1.000 ± 0.000 | 0.000 |
| GBDT, stream 2 | – | 0.836 ± 0.001 | 0.036 | 0.365 ± 0.001 | 0.196 | 0.998 ± 0.000 | 0.001 |

### 5.2 Does fusion beat one network on all features?

Combining the two streams helps a great deal relative to either stream alone (Table 3): early fusion exceeds the better single stream by +0.044 on TON-IoT and +0.060 on UNSW-NB15, in every seed. But early fusion is just one CNN on all features, and the RF gains a similar amount from the same features (+0.032 and +0.061). The gain over a single stream therefore measures the value of the extra features, not of a fusion design. A fusion paper that reports only branch-versus-fused comparisons cannot distinguish fusion from this effect.

| Comparison | TON-IoT | UNSW-NB15 | N-BaIoT |
|---|---|---|---|
| Early fusion − better single stream | +0.044 [+0.043, +0.045] (5/5) | +0.060 [+0.042, +0.077] (5/5) | 0.000 [−0.001, 0.000] (1/5) |
| Late − early fusion | +0.005 [+0.004, +0.007] (5/5) | +0.016 [+0.001, +0.031] (5/5) | 0.000 [0.000, +0.001] (3/5) |
| Attention − early fusion | +0.001 [0.000, +0.003] (4/5) | +0.008 [−0.004, +0.021] (3/5) | 0.000 [0.000, +0.001] (4/5) |
| Late − attention fusion | +0.004 [+0.002, +0.005] (5/5) | +0.008 [+0.004, +0.011] (5/5) | 0.000 [0.000, 0.000] (3/5) |
| Wide − standard early fusion | +0.002 [+0.001, +0.003] (5/5) | +0.008 [−0.008, +0.024] (4/5) | – |
| Early fusion with head − early fusion | +0.004 [+0.001, +0.006] (5/5) | +0.006 [−0.010, +0.022] (3/5) | – |
| Late fusion − early fusion with head | +0.001 [−0.001, +0.004] (4/5) | +0.010 [+0.004, +0.015] (5/5) | – |
| Late fusion − wide early fusion | +0.003 [+0.002, +0.004] (5/5) | +0.008 [+0.001, +0.014] (5/5) | – |
| Late fusion − late fusion on all features | +0.001 [0.000, +0.003] (4/5) | +0.011 [+0.004, +0.018] (5/5) | – |
| MLP: late − early fusion | +0.005 [+0.002, +0.007] (5/5) | +0.017 [−0.002, +0.037] (4/5) | – |
| MLP: attention − early fusion | +0.003 [+0.001, +0.005] (5/5) | +0.012 [−0.009, +0.033] (3/5) | – |
| RF (all features) − late fusion (Welch) | +0.008 (p < 0.001) | +0.034 (p < 0.001) | 0.000 (p = 0.404) |
| GBDT (all features) − late fusion (Welch) | +0.009 (p < 0.001) | +0.036 (p < 0.001) | 0.000 (p = 0.180) |

The fusion architectures themselves differ little. Late fusion beats early fusion by +0.005 on TON-IoT (5/5 seeds; Holm-adjusted p = 0.006) and by +0.016 [+0.001, +0.031] on UNSW-NB15 (5/5; unadjusted p = 0.038, Holm-adjusted p = 0.135), so the UNSW-NB15 difference is not significant after Holm correction. Attention fusion does not differ significantly from early fusion (+0.001 on TON-IoT and +0.008 on UNSW-NB15, Holm-adjusted p = 0.135 and 0.135; 0.000 on N-BaIoT) and falls below late fusion, with which it shares the two encoders. With the MLP encoder of Section 5.3, attention fusion does exceed early fusion on TON-IoT (+0.003 [+0.001, +0.005], 5/5 seeds; descriptive), a small gain inside the ±0.01 margin.

The capacity controls locate the small advantage of late fusion. Widening early fusion to the parameter count of late fusion added +0.002 [+0.001, +0.003] on TON-IoT (about 41% of the late-fusion advantage) and +0.008 on UNSW-NB15, and late fusion stayed above the wide network (TON-IoT +0.003, Holm-adjusted p = 0.009; UNSW-NB15 +0.008, Holm-adjusted p = 0.135). Giving early fusion the extra 128-unit dense layer of the late-fusion head recovered +0.004 [+0.001, +0.006] of the +0.005 TON-IoT advantage (about 71%), and late fusion minus early fusion with head is +0.001 [−0.001, +0.004] (Table 3). Late fusion with both encoders on all features reproduced about 73% of the advantage and came within +0.001 [0.000, +0.003] of late fusion on the semantic streams (unadjusted p = 0.032, Holm-adjusted p = 0.135). On TON-IoT the advantage therefore comes mainly from the extra dense layer and less from the parameter count (the intervals of the two controls overlap, and they were not compared directly), and hardly from the semantic grouping. On UNSW-NB15 the head does not explain it: late fusion stays above early fusion with head by +0.010 [+0.004, +0.015] (5/5 seeds; descriptive). On UNSW-NB15 no comparison of the family is significant, but the intervals are wide: the half-width of the 95% confidence intervals is about 0.010 (median over the family) against 0.001 on TON-IoT, so effects of about 0.01 can be neither confirmed nor excluded there. Late fusion on the semantic streams exceeded late fusion on all features by +0.011 [+0.004, +0.018] in 5/5 seeds (unadjusted p = 0.013, Holm-adjusted p = 0.076), a hint, of about the size of the margin, that the grouping itself may help on this dataset; Section 5.3 tests the grouping directly. The UNSW-NB15 effects are also not stable between runs. In the earlier run of the main experiment, in which every UNSW-NB15 network stopped early and was scored at its best epoch, late minus early fusion was +0.007 (4/5 seeds) and attention minus early fusion −0.003. On the test records without a training match on the selected features, these differences change by at most a few thousandths on TON-IoT; on UNSW-NB15, where 16.3% of test records are removed, late minus early fusion becomes +0.025 [+0.001, +0.048] (Table B.2).

Per class (Table 4 for TON-IoT, Table 5 for UNSW-NB15), the fusion models differ most on small or hard classes: MITM on TON-IoT (0.62 early, 0.66 late, 0.62 attention; 156 test records), and worms (0.27, 0.34, 0.34; 19 records) and DoS (0.33, 0.35, 0.35) on UNSW-NB15.

No CNN beats the tuned RF on all features in macro-F1; the networks were not tuned (Section 4.2), so this is a comparison of tuned trees with untuned networks. On TON-IoT the RF exceeds late fusion by +0.008 (Welch p < 0.001) with a lower benign FPR (0.007 against 0.011); a margin of this size might be closed by tuning the networks. On UNSW-NB15 the RF leads by +0.034, with the largest margins on worms, shellcode and exploits, while the CNNs are ahead on analysis and backdoor (Table 5). On N-BaIoT both are at the ceiling.

Test records that coincide with a training record on the selected features matter on UNSW-NB15 (2,586 records, 16.3% of the test set; on TON-IoT only 35). Some classes are affected far more than others: 81% of the backdoor, 80% of the generic and 71% of the analysis test records have a match, against 10% of the benign ones. Table B.2 reports the main models on all test records, on the records without a match and on the two subsets separately. On the matched records every model is about equally accurate (accuracy 0.635–0.644, against 0.786–0.835 on the other records), and the RF is no better than late fusion there (−0.007 in accuracy, against +0.036 on the unmatched records). Removing them therefore raises the RF lead in macro-F1 from +0.034 to +0.073: the advantage of the RF lies in the records without a match, not in memorised duplicates. The low accuracy of every model on records whose ten-feature vector also occurs in training suggests that many of them share that vector with training records of another class, that is, label conflicts created by reducing the features to ten; the label of the matching training record was not saved, so we cannot report the share of matches with a different label. If that share is large, part of the full-test error on UNSW-NB15 is irreducible for any model on these features.

**Other data splits.** The ranking does not depend on the particular data split. Over the original and five further stratified splits (six split-level observations, descriptive), late fusion exceeded early fusion on TON-IoT in 6 of six (+0.005 [+0.004, +0.006]) and on UNSW-NB15 in 5 of six (+0.011 [+0.001, +0.020]). Attention fusion differed from early fusion by +0.002 and +0.006 on average (per split 0.000 to +0.004 on TON-IoT and −0.010 to +0.017 on UNSW-NB15). The RF exceeded late fusion on every split of both datasets (+0.006 [+0.005, +0.008] and +0.041 [+0.035, +0.046]), and early fusion exceeded the better single stream by +0.045 and +0.050.

| Class | Test n | Stream 1 only | Stream 2 only | Early fusion | Late fusion | Attention fusion | RF, all features | GBDT, all features |
|---|---|---|---|---|---|---|---|---|
| backdoor | 2,807 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| ddos | 2,998 | 0.96 | 0.93 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| dos | 2,200 | 0.97 | 0.94 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| injection | 2,995 | 0.96 | 0.65 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 |
| mitm | 156 | 0.52 | 0.27 | 0.62 | 0.66 | 0.62 | 0.71 | 0.73 |
| normal | 6,290 | 0.96 | 0.97 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |
| password | 2,977 | 0.99 | 0.71 | 0.99 | 0.99 | 0.99 | 1.00 | 0.99 |
| ransomware | 2,140 | 0.82 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| scanning | 2,792 | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| xss | 2,178 | 0.86 | 0.86 | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 |

| Class | Test n | Stream 1 only | Stream 2 only | Early fusion | Late fusion | Attention fusion | RF, all features | GBDT, all features |
|---|---|---|---|---|---|---|---|---|
| analysis | 194 | 0.29 | 0.19 | 0.28 | 0.28 | 0.29 | 0.24 | 0.22 |
| backdoor | 191 | 0.21 | 0.18 | 0.27 | 0.27 | 0.26 | 0.23 | 0.22 |
| dos | 527 | 0.26 | 0.07 | 0.33 | 0.35 | 0.35 | 0.37 | 0.38 |
| exploits | 2,933 | 0.69 | 0.68 | 0.73 | 0.75 | 0.74 | 0.80 | 0.80 |
| fuzzers | 2,378 | 0.63 | 0.59 | 0.69 | 0.71 | 0.70 | 0.73 | 0.73 |
| generic | 620 | 0.88 | 0.69 | 0.87 | 0.88 | 0.87 | 0.89 | 0.89 |
| normal | 7,784 | 0.88 | 0.89 | 0.89 | 0.89 | 0.89 | 0.91 | 0.92 |
| reconnaissance | 1,084 | 0.77 | 0.14 | 0.79 | 0.79 | 0.79 | 0.78 | 0.79 |
| shellcode | 164 | 0.23 | 0.17 | 0.45 | 0.47 | 0.43 | 0.56 | 0.53 |
| worms | 19 | 0.12 | 0.02 | 0.27 | 0.34 | 0.34 | 0.54 | 0.61 |

### 5.3 Does the semantic grouping matter?

If the semantic streams carried structure that separate encoders can exploit, fusion should work better on them than on arbitrary groupings of the same features. We drew ten random partitions of each dataset's selected features into two streams of the semantic sizes and trained the single-stream CNNs, late and attention fusion and the per-stream RFs on each (Table 6, Figure 2; split r uses seed r mod 5).

The grouping matters for the single streams but not for the fused models. On TON-IoT the semantic protocol stream is weaker than every random stream of the same size (CNN 0.829 against 0.929–0.946; RF 0.796 against 0.944–0.964), because the semantic rule places most of the informative volume features in the other stream; random seven-feature streams range from 0.579 to 0.909, depending on which features they draw. On UNSW-NB15 both semantic streams fall below nearly all random streams of their size. Once the streams are fused the point estimates shrink to at most about 0.01: late fusion scores 0.953 on the semantic split against a random-split mean of 0.951 on TON-IoT (range 0.949–0.953; the semantic five-seed mean exceeds 8 of the ten single-run random-split values; within-seed difference +0.001 [0.000, +0.002], 5/5 seeds), and 0.573 against 0.563 on UNSW-NB15 (0.543–0.575; semantic mean above 9 of the ten; +0.009 [−0.001, +0.019], 5/5 seeds). For attention fusion the within-seed differences are 0.000 [−0.001, +0.001] and +0.001 [−0.004, +0.007]. On N-BaIoT the fused models differ between semantic and random splits by about 0.001 (within-seed 0.000 for late and +0.001 for attention fusion), although the semantic attention-fusion mean is above all ten random-split values there. Because each random split is a single run while the semantic value is a five-seed mean, these counts include seed noise on one side only; the within-seed differences are the like-for-like statistic.

Two further controls make the encoder an unlikely explanation, although on UNSW-NB15 they cannot rule it out. A 1D convolution depends on the order of its inputs, but early fusion on five random column orders scored 0.946–0.950 on TON-IoT (semantic order, stream 1 then stream 2: 0.948) and 0.527–0.565 on UNSW-NB15 (0.557; standard deviation 0.015 over orders against 0.011 over seeds). On UNSW-NB15 the spread over column orders is thus larger than the spread over seeds and comparable to the late-minus-early difference, so for differences below about 0.02 on that dataset the order-independent MLP results below are the cleaner evidence. With the order-independent MLP encoder, semantic minus random late fusion was +0.001 [−0.001, +0.003] on TON-IoT and +0.007 [−0.004, +0.018] on UNSW-NB15. Late MLP fusion again exceeded early MLP fusion on TON-IoT (+0.005, 5/5 seeds) and, not significantly, on UNSW-NB15 (+0.017 [−0.002, +0.037]). Late MLP fusion scored within a few thousandths of late CNN fusion (0.953 and 0.569); some other MLP models differ more from their CNN counterparts (Table 2).

The point estimates of the semantic split's advantage for late fusion over the random-split mean are therefore at most about 0.01 of macro-F1 (TON-IoT +0.001, UNSW-NB15 +0.009). On TON-IoT the within-seed difference is small (+0.001, interval [0.000, +0.002] excluding zero at full precision, unadjusted p = 0.027) and negligible by our margin; the same holds for the even smaller N-BaIoT differences. It should not be over-read: all late-fusion random-split runs come from the earlier run, whereas the semantic late-fusion runs on TON-IoT are new, and same-seed reruns on TON-IoT differ by up to 0.002 (Section 4.2), about the size of this difference. On UNSW-NB15, the dataset where it is largest, it is not significant but inconclusive: its interval [−0.001, +0.019] reaches beyond the margin, so a grouping effect of up to about 0.02 cannot be excluded there. The larger "fusion gain" of the semantic split over its better stream (+0.049 against +0.012 on random splits for TON-IoT, +0.076 against +0.029 for UNSW-NB15) comes from the semantic streams being weaker on their own, not from a better fused model. On UNSW-NB15 several descriptive results lean the same way (the semantic mean above 9 of the ten random splits, late fusion above late fusion on all features in 5/5 seeds, and a larger late-minus-early difference on the records without a training match), a consistent but unresolved hint that the grouping may help a little there. Early fusion has no streams, so it was not trained on random splits; compared with the five-seed mean of early fusion on the semantic order, late fusion on random splits stays above it on TON-IoT (by +0.004 on average, all ten splits), so the small advantage of late over early fusion in Section 5.2 does not depend on the semantic grouping.

| Dataset | Model | Semantic split (mean of 5 seeds) | Random splits: mean | Random splits: min–max | Within-seed difference, semantic − random [95% CI] |
|---|---|---|---|---|---|
| TON-IoT | Stream 1 only | 0.904 | 0.824 | 0.579–0.909 | +0.080 [+0.001, +0.158] (5/5) |
| TON-IoT | Stream 2 only | 0.829 | 0.939 | 0.929–0.946 | −0.110 [−0.114, −0.106] (0/5) |
| TON-IoT | Late fusion | 0.953 | 0.951 | 0.949–0.953 | +0.001 [0.000, +0.002] (5/5) |
| TON-IoT | Attention fusion | 0.949 | 0.949 | 0.946–0.952 | 0.000 [−0.001, +0.001] (4/5) |
| TON-IoT | RF, stream 1 | 0.928 | 0.855 | 0.609–0.935 | +0.073 [−0.010, +0.157] (5/5) |
| TON-IoT | RF, stream 2 | 0.796 | 0.956 | 0.944–0.964 | −0.159 [−0.164, −0.154] (0/5) |
| TON-IoT | GBDT, stream 1 | 0.925 | 0.849 | 0.600–0.927 | +0.076 [−0.006, +0.157] (5/5) |
| TON-IoT | GBDT, stream 2 | 0.836 | 0.955 | 0.943–0.961 | −0.119 [−0.123, −0.115] (0/5) |
| TON-IoT | MLP stream 1 | 0.904 | 0.825 | 0.580–0.912 | +0.079 [+0.002, +0.156] (5/5) |
| TON-IoT | MLP stream 2 | 0.829 | 0.940 | 0.925–0.945 | −0.111 [−0.115, −0.107] (0/5) |
| TON-IoT | MLP late fusion | 0.953 | 0.952 | 0.949–0.953 | +0.001 [−0.001, +0.003] (4/5) |
| TON-IoT | MLP attention fusion | 0.951 | 0.950 | 0.947–0.952 | +0.002 [−0.001, +0.004] (3/5) |
| UNSW-NB15 | Stream 1 only | 0.497 | 0.530 | 0.472–0.550 | −0.033 [−0.050, −0.017] (0/5) |
| UNSW-NB15 | Stream 2 only | 0.361 | 0.466 | 0.374–0.517 | −0.105 [−0.142, −0.068] (0/5) |
| UNSW-NB15 | Late fusion | 0.573 | 0.563 | 0.543–0.575 | +0.009 [−0.001, +0.019] (5/5) |
| UNSW-NB15 | Attention fusion | 0.565 | 0.564 | 0.551–0.578 | +0.001 [−0.004, +0.007] (3/5, 1 tie) |
| UNSW-NB15 | RF, stream 1 | 0.546 | 0.579 | 0.501–0.608 | −0.033 [−0.052, −0.014] (0/5) |
| UNSW-NB15 | RF, stream 2 | 0.368 | 0.503 | 0.365–0.595 | −0.135 [−0.183, −0.088] (0/5) |
| UNSW-NB15 | GBDT, stream 1 | 0.540 | 0.569 | 0.490–0.595 | −0.028 [−0.050, −0.007] (0/5) |
| UNSW-NB15 | GBDT, stream 2 | 0.365 | 0.486 | 0.365–0.558 | −0.121 [−0.161, −0.080] (0/5) |
| UNSW-NB15 | MLP stream 1 | 0.498 | 0.528 | 0.466–0.556 | −0.030 [−0.042, −0.018] (0/5) |
| UNSW-NB15 | MLP stream 2 | 0.374 | 0.463 | 0.373–0.513 | −0.089 [−0.111, −0.067] (0/5) |
| UNSW-NB15 | MLP late fusion | 0.569 | 0.562 | 0.551–0.571 | +0.007 [−0.004, +0.018] (4/5) |
| UNSW-NB15 | MLP attention fusion | 0.564 | 0.563 | 0.552–0.574 | +0.002 [−0.015, +0.018] (4/5) |
| N-BaIoT | Stream 1 only | 1.000 | 0.999 | 0.998–0.999 | +0.001 [+0.001, +0.001] (5/5) |
| N-BaIoT | Stream 2 only | 0.994 | 0.999 | 0.999–0.999 | −0.005 [−0.006, −0.003] (0/5) |
| N-BaIoT | Late fusion | 1.000 | 0.999 | 0.998–1.000 | 0.000 [0.000, +0.001] (5/5) |
| N-BaIoT | Attention fusion | 1.000 | 0.999 | 0.998–0.999 | +0.001 [0.000, +0.002] (5/5) |
| N-BaIoT | RF, stream 1 | 1.000 | 1.000 | 0.999–1.000 | 0.000 [0.000, 0.000] (5/5) |
| N-BaIoT | RF, stream 2 | 0.998 | 1.000 | 1.000–1.000 | −0.001 [−0.001, −0.001] (0/5) |
| N-BaIoT | GBDT, stream 1 | 1.000 | 1.000 | 0.999–1.000 | 0.000 [0.000, 0.000] (5/5) |
| N-BaIoT | GBDT, stream 2 | 0.998 | 1.000 | 1.000–1.000 | −0.001 [−0.002, −0.001] (0/5) |

### 5.4 What do the attention weights say?

Attention fusion assigns each record a weight for each stream, and such weights are often presented as an explanation of which "modality" drives a decision. Averaged over test records, stream 1 receives 0.59 of the weight on TON-IoT (range over seeds 0.55–0.63), 0.46 on UNSW-NB15 and 0.73 on N-BaIoT (0.66–0.81). The weights vary between classes (on TON-IoT from 0.30 for dos to 0.88 for backdoor) and also within a class (mean within-class standard deviation 0.14).

If the weights explained the model, a class should receive more weight on the stream that classifies it better. We correlated each class's mean stream-1 weight with its stream advantage (Figure 3), the per-class F1 of the stream-1 CNN minus that of the stream-2 CNN (Spearman's ρ over classes, computed for each seed's weights against the seed-averaged stream advantage). The correlation is weak on TON-IoT (mean ρ = 0.16, range over seeds −0.21 to +0.32) and moderate on UNSW-NB15 (0.44, +0.38 to +0.50); with ten classes, a correlation of this size is imprecise (permutation p for the seed-averaged weights 0.56 and 0.22). An in-model ablation separates what the weights describe from what they are taken to mean. Replacing one stream's embedding by its mean over the test set, with the attention weights kept, lowered macro-F1 by 0.459 for stream 1 and 0.248 for stream 2 on TON-IoT (0.288 and 0.142 on UNSW-NB15, 0.678 and 0.079 on N-BaIoT). We define a class's reliance on stream 1 as the F1 lost when stream 1 is removed minus the F1 lost when stream 2 is removed. Reliance correlates strongly with the class's mean stream-1 weight (ρ = 0.94, 0.93 and 0.93; permutation p < 0.01 on each dataset), and similarly with the weight-scaled contribution of each stream's embedding. The weights thus describe how the trained model routes information. That routing, however, is not tied to which stream classifies the class better: reliance and stream advantage correlate at ρ = 0.03 on TON-IoT (p = 0.95) and 0.54 on UNSW-NB15 (p = 0.11). On TON-IoT, for example, the DoS class gets the lowest mean stream-1 weight (the lowest in 4 of five seeds) although the volume stream classifies it slightly better than the protocol stream (0.97 against 0.94). On N-BaIoT, where both streams classify every class almost perfectly, the weights still range from 0.38 (mirai_udpplain) to 0.99 (gafgyt_junk), so they reflect how the model happened to route information, not which stream is needed. On TON-IoT and N-BaIoT the ranking of classes by weight also changes between seeds: the mean pairwise Spearman correlation between seeds is 0.74 on TON-IoT (lowest pair 0.47) and 0.55 on N-BaIoT (lowest −0.17), against 0.89 on UNSW-NB15. Stream weights from a single trained model should therefore not be read as evidence of which feature group matters for an attack; this agrees with earlier findings that attention weights are not, by themselves, explanations (Jain & Wallace, 2019).

### 5.5 A silent failure: input scaling

Our first run of these experiments fed the CNNs the min–max scaled features used by the RF. Table 7 compares it with a repeat that changed only the input scaling to the quantile transform (same 60-epoch cap and weight restoration as the first run), and with the final run of Table 2. On TON-IoT, the stream-1 CNN of the first run reached a macro-F1 of 0.028 and labelled 0.753 of the benign test records as attacks, while the RF on the same seven features scored 0.927. Late fusion, which contains the same encoder, collapsed to 0.176. The best validation epoch of the stream-1 CNN was at most 4, a sign that the model degraded rather than learned, and its macro-F1 is below that of a constant prediction of the majority class (0.037).

| Dataset | Input scaling | Stream 1 only | Stream 2 only | Early fusion | Late fusion | Attention fusion | RF, stream 1 |
|---|---|---|---|---|---|---|---|
| TON-IoT | min–max, first run | 0.028 | 0.809 | 0.864 | 0.176 | 0.670 | 0.927 |
| TON-IoT | quantile, first-run protocol | 0.904 | 0.829 | 0.948 | 0.953 | 0.949 | 0.927 |
| TON-IoT | quantile, final run | 0.904 | 0.829 | 0.948 | 0.953 | 0.949 | 0.928 |
| UNSW-NB15 | min–max, first run | 0.450 | 0.384 | 0.535 | 0.523 | 0.514 | 0.545 |
| UNSW-NB15 | quantile, first-run protocol | 0.498 | 0.362 | 0.562 | 0.569 | 0.559 | 0.545 |
| UNSW-NB15 | quantile, final run | 0.497 | 0.361 | 0.557 | 0.573 | 0.565 | 0.546 |

The cause is the shape of the volume features. After min–max scaling, byte and packet counts are concentrated near zero with rare large values (on TON-IoT the stream-1 features have training medians of at most 0.0002 and standard deviations between 0.0027 and 0.014). The most likely mechanism is batch normalisation (Ioffe & Szegedy, 2015): its statistics estimated during training may not match evaluation for such inputs. The first run saved summary results (such as the best epoch) but not the training curves, so we cannot show them, and we did not isolate this mechanism with a separate ablation. Heavy-tailed inputs are a known pitfall, usually handled by standardisation or a log transform; the point here is that the failure was silent. A quantile transform, which is monotone and therefore barely affects the trees, removes the problem with nothing else changed: the stream-1 CNN reaches 0.904, within 0.03 of the RF, and the final run, with the longer budget, gives 0.904. The training-protocol change between the first and final runs therefore plays no part in the collapse.

Read naively, the first run says that on TON-IoT the protocol stream dominates and the volume stream is nearly useless, which is the opposite of the final result, and that late fusion is a poor architecture. The earlier version of this study reported exactly this pattern. The failure produced no error and plausible numbers for the other stream. Training curves and a trivial baseline would have revealed it to a careful reader, but the RF trained on the same features made it obvious at no cost. Any comparison of feature groups with neural encoders should include such a reference, and a branch that falls far below a tree model on the same features should be treated as a training failure until shown otherwise.

### 5.6 Cost

Table 8 lists CPU inference times (Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz; TensorFlow for the networks, scikit-learn for the trees). TensorFlow used its default thread pools on all logical cores, the GBDT used all cores in both modes (which probably inflates its single-record times), and the RF used one thread for single records and all cores for batches. Prediction time does not depend on the trained weights, so the networks were timed untrained; the trees were trained with their tuned settings. One record at a time, early fusion took 0.41 ms on TON-IoT, late and attention fusion 0.55 and 0.57 ms; per call, this time is dominated by framework overhead. For 1,000 records in one call the three architectures were equally fast (8.4, 8.1 and 8.2 ms), while the wide early fusion and late fusion on all features took about twice as long (16.2 and 17.6 ms). Fusion therefore costs little at inference.

The trees are slower and much larger. The RF took 8.04 ms per record, about 14.5 times as long as late fusion, and its fitted model occupies 221 MB on TON-IoT and 636 MB on UNSW-NB15 (4,416,482 nodes), against 343 kB of 32-bit weights for late fusion. The GBDT is smaller (23 and 35 MB) but slower per record on UNSW-NB15 (114.95 ms). The trees were tuned for validation macro-F1 only, and smaller forests trade accuracy for size. A companion study of the accuracy–footprint trade-off, not yet published, uses the same features and data split; its forests were trained and scored with the same pipeline, and the values below are read from its result files: a forest of 25 trees of depth 12 occupies 4.4 MB on TON-IoT and 7.3 MB on UNSW-NB15 and scores 0.946 and 0.572, about the level of late fusion (0.953 and 0.573). On TON-IoT this forest is below late fusion, by more than the late-minus-early difference, while still about 13 times larger. Keeping most of the tuned forest's accuracy takes 100 trees of depth 16 (0.955 at 41.1 MB, 0.603 at 125.7 MB). The accuracy advantage of the trees in Section 5.2 thus comes with a memory cost that matters on constrained devices, whereas the fusion architectures cost hardly more than one CNN.

| Model | TON-IoT (ms/record) | UNSW-NB15 (ms/record) | N-BaIoT (ms/record) | TON-IoT (ms / 1,000 records) | UNSW-NB15 (ms / 1,000 records) | N-BaIoT (ms / 1,000 records) |
|---|---|---|---|---|---|---|
| Stream 1 only (CNN) | 0.41 | 0.41 | 0.43 | 3.2 | 3.5 | 10.5 |
| Stream 2 only (CNN) | 0.41 | 0.41 | 0.48 | 6.3 | 2.3 | 34.5 |
| Early fusion (CNN, all features) | 0.41 | 0.41 | 0.58 | 8.4 | 4.1 | 42.9 |
| Late fusion | 0.55 | 0.57 | 0.61 | 8.1 | 4.5 | 39.1 |
| Attention fusion | 0.57 | 0.56 | 0.62 | 8.2 | 4.8 | 39.6 |
| Wide early fusion (×1.6) | 0.51 | 0.46 | 0.56 | 16.2 | 6.5 | 76.0 |
| Early fusion with head | 0.42 | 0.41 | 0.52 | 9.7 | 4.3 | 40.9 |
| Late fusion on all features | 0.55 | 0.54 | 0.65 | 17.6 | 6.8 | 77.9 |
| MLP early fusion | 0.36 | 0.37 | 0.37 | 1.1 | 1.1 | 1.3 |
| MLP late fusion | 0.51 | 0.50 | 0.49 | 1.7 | 1.7 | 2.0 |
| RF, all features | 8.04 | 8.04 | 8.02 | 46.6 | 47.4 | 49.7 |
| GBDT, all features | 19.10 | 114.95 | 26.73 | 78.0 | 332.8 | 100.7 |

## 6. Discussion

### 6.1 What the results mean for multi-stream IDS

Four conclusions follow for flow-feature IDS. First, the benefit attributed to fusion is mostly the benefit of using all features: combining streams helps relative to each stream, but one network on all features captures nearly all of it, and a random forest captures it without any neural network. Second, the architecture of fusion matters little; late fusion gave a small but consistent gain on TON-IoT that late fusion on all features largely reproduced and that comes mainly from its extra 128-unit dense layer, and attention fusion gave no significant gain. Third, the semantic grouping itself contributes little: random groupings of the same sizes change the single streams considerably but the fused models by point estimates of about 0.01 at most. On TON-IoT this effect is negligible (about 0.001); on UNSW-NB15 it is not significant, but our data cannot exclude an effect of up to about 0.02, and several descriptive results there lean towards a small benefit of the grouping. The larger fusion gain of the semantic split mostly reflects its weaker individual streams. Fourth, the tuned trees are the right diagnostic reference, but their accuracy lead over untuned networks was measured with forests of hundreds of megabytes; a forest of a few megabytes scored below late fusion on TON-IoT (Section 5.6). We therefore see no evidence that the features of one flow record behave like modalities in the multimodal-learning sense. They come from the same observation and share a time base and, after transformation, a scale, and on these benchmarks they did not need different encoders.

These conclusions rest mainly on TON-IoT, the only non-saturated IoT dataset in the study, and on UNSW-NB15, an enterprise benchmark with ten selected features. They concern two small encoder families and one semantic split per dataset. Multi-stream designs may still be useful where the streams really are different sources: packet payloads or byte sequences alongside flow statistics, host logs alongside network flows, or features arriving from different sensors at different times, where one stream can be missing. Our results do not test these settings. Within a flow record, the gains they found are no larger than those of one network, or a tree ensemble, on all features.

### 6.2 Recommendations

For studies that propose fusion of feature groups for intrusion detection we recommend: (i) compare against one network on all features with matched parameters, not only against single branches; (ii) include random groupings of the same sizes; (iii) include a tree ensemble on the same features and on each group, which also detects training failures such as the one in Section 5.5; (iv) report several seeds with paired differences; and (v) do not interpret fusion weights as explanations without checking them against per-group performance and across seeds.

### 6.3 Limitations

The selected features come from a selector that ranks features with a random forest, which may favour the tree references; SMOTE produces fractional values in one-hot columns, which affects stream 2 on TON-IoT most. N-BaIoT shares time windows between training and test records and is saturated, and UNSW-NB15 has only ten selected features. The streams are one semantic split per dataset; other groupings (for example by protocol layer) were not tested, and ten random splits sample only a small part of the possible groupings. The encoders are small 1D-CNNs and MLPs, chosen for IoT deployment, with one fixed configuration, whereas the trees were tuned on the validation set; tuned or larger encoders, or tabular transformers, might use separate streams differently and might narrow the gap to the trees, as tuned deep tabular models do on general benchmarks (Gorishniy et al., 2021). Our tree-versus-network comparison is therefore one of tuned trees against untuned networks, and on TON-IoT, where the RF leads late fusion by +0.008, tuning could plausibly change the ranking. UNSW-NB15 is not IoT traffic and N-BaIoT is saturated, so TON-IoT is the only non-saturated IoT benchmark in the study and the IoT evidence rests on it; a second non-saturated IoT benchmark such as CICIoT2023 (Neto et al., 2023) or Edge-IIoTset (Ferrag et al., 2022) would strengthen it. Five seeds give 95% intervals of about ±0.001 on TON-IoT but about ±0.010 on UNSW-NB15, so UNSW-NB15 can resolve only effects of about 0.01 and larger. Run-to-run variation on the GPU (Section 4.2) is of the same order as the smaller effects, and 104 of the 120 random-split networks, including all late-fusion random-split runs, come from the earlier run, so the semantic-versus-random comparison mixes runs whose seed pairing is nominal. On UNSW-NB15 the column order of the CNN input matters about as much as the seed. The training curves of the failed min–max run were not saved. The data-split replicates keep the features selected on the original split. The CNN inputs use a quantile transform; with other remedies for heavy-tailed inputs (for example a log transform) the absolute numbers would differ slightly.

## 7. Conclusion

On the flow-feature benchmarks we studied, splitting the features of a record into semantic streams and fusing them added little beyond what one network, or a tree ensemble, achieves on all features: late fusion gained a few thousandths of macro-F1 on TON-IoT, a gain that did not depend on the grouping, and nothing significant after Holm correction on UNSW-NB15, and attention fusion gave no significant gain. The gains such designs report over their own branches are real but come mostly from the features, not from the fusion. A random grouping of the same features served the fused model almost as well as the semantic one: within 0.002 on TON-IoT, and by a point estimate of about 0.01 on UNSW-NB15, where an effect of up to about 0.02 cannot be excluded. The tuned forests beat the untuned networks in macro-F1 at hundreds of megabytes; forests of a few megabytes scored at or below late fusion. Attention weights over the streams describe how the model routes information, but are only loosely tied to stream usefulness and, on two of the three datasets, change between seeds. A misconfigured input scaling can silently disable one branch and reverse conclusions about which stream matters, and a tree-ensemble reference catches it. We release the code, the result files and the stream definitions so that these controls can be applied to other fusion designs.

## Declarations

**Data availability.** TON-IoT, UNSW-NB15 and N-BaIoT are publicly available from their providers. The code, stream definitions and all result files are available at https://github.com/Amoakoemma123/iot-ids-pipeline. The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors.

**CRediT authorship contribution statement.** Emmanuel Kwame Amoako: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft, Writing – review & editing. Mahamudu Amadu: Conceptualization, Methodology, Validation. James Quayson: Conceptualization, Methodology, Writing – review & editing. Kingsford Donkor: Validation, Writing – review & editing. Daniel Asamoah Apraku: Validation, Writing – review & editing.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the authors used Claude (Anthropic) to write and edit the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The released scripts compute every reported number from the result files, and an automated check compares each quantitative statement in the text with those files.

## Appendix A. Fusion studies examined

Table A.1 lists the 14 fusion, multi-view and attention-based IDS studies behind the statements in Section 2.1. "Not reported" means that the control does not appear in the full text; "not in abstract" means that only the abstract was available to us, which is weak evidence of absence.

| Study | Datasets | Branch inputs | Source | Random grouping | Matched-parameter single network | Tree ensemble on the same data | Several seeds or runs |
|---|---|---|---|---|---|---|---|
| Ayantayo et al. (2023) | UNSW-NB15, NSL-KDD | feature groups by data type | FT | not reported | not reported | not reported (values quoted from other papers) | not reported |
| Kiflay et al. (2024) | UNSW-NB15 | six flow features; first 32 payload bytes | FT | not reported | not applicable (forests only) | yes (one forest on concatenated inputs) | ten-fold CV mean, no spread |
| Xu et al. (2025) | CICIDS2017, CICIDS2018 | byte-level image of the first packets; flow statistics | FT | not reported | not reported | not reported | not reported |
| Ullah et al. (2024) | CIC-IoT 2022, CICIoT2023, Edge-IIoTset | text-like and image-like encodings of packet contents | FT | not reported | not reported | not reported | not reported |
| Dinh et al. (2024a, 2024b) | NSL-KDD, UNSW-NB15, CIC-IDS2017 | one sub-encoder per feature group | FT (extended version, 2024b) | not reported | not reported | yes (forest on the original features) | not reported |
| AboulEla & Kashef (2026) | four NetFlow benchmarks | numeric, text, graph and quantum-inspired representations of the same records | FT | not reported | not reported | not reported (gradient boosting only as meta-learner) | no (stated as a limitation) |
| Alabbadi & Bajaber (2025) | TON-IoT, NSL-KDD, CICIoMT2024 | the same full feature vector to both paths | FT | not reported | not reported | not reported | not reported |
| Ishtiaq et al. (2025) | Edge-IIoTset | one network with channel and temporal attention | FT | not reported | not reported | not reported | not reported |
| Li et al. (2025) | TON-IoT, UNSW-NB15 | host and network scenarios | Abs | not in abstract | not in abstract | not in abstract | not in abstract |
| Singh et al. (2026) | one IoT dataset with network and host data | network and audit telemetry | Abs | not in abstract | not in abstract | yes (gradient boosting and forest on fused features) | not in abstract |
| Attota et al. (2021) | IoT network data (not named in the abstract) | multiple views of the network data | Abs | not in abstract | not in abstract | not in abstract | not in abstract |
| Sanjalawe et al. (2025) | CICAPT-IIoT, Edge-IIoTset, X-IIoTID, WUSTL-IIoT-2021 | dual-stream network on selected features | Abs | not in abstract | not in abstract | not in abstract | not in abstract |
| Afraji et al. (2025) | TON-IoT, CICIDS2017 | CNN, LSTM and GRU branches | Abs | not in abstract | not in abstract | not in abstract | not in abstract |
| Al Qathrady et al. (2024) | Edge-IIoTset, X-IIoTID | self-attention followed by a CNN | Abs | not in abstract | not in abstract | not in abstract | not in abstract |

## Appendix B. Supplementary analyses

Table B.1 lists the confirmatory family of Section 4.3 with unadjusted and Holm-adjusted p-values and the reading against the ±0.01 margin. Table B.2 separates the UNSW-NB15 and TON-IoT test records that coincide with a training record on the selected features from the others (Section 5.2); it uses the saved per-record predictions of the semantic-split runs (random-split runs reused from the earlier run have none). Table B.3 gives the threshold-free and detection metrics behind Section 5.1.

| Dataset | Comparison | Mean difference [95% CI] (seeds > 0) | Unadjusted p | Holm-adjusted p | Margin reading |
|---|---|---|---|---|---|
| TON-IoT | Late − early fusion | +0.005 [+0.004, +0.007] (5/5) | 0.001 | 0.006 | negligible (CI inside ±0.01) |
| TON-IoT | Attention − early fusion | +0.001 [0.000, +0.003] (4/5) | 0.056 | 0.135 | negligible (CI inside ±0.01) |
| TON-IoT | Late fusion − wide early fusion | +0.003 [+0.002, +0.004] (5/5) | 0.001 | 0.009 | negligible (CI inside ±0.01) |
| TON-IoT | Late fusion − late fusion on all features | +0.001 [0.000, +0.003] (4/5) | 0.032 | 0.135 | negligible (CI inside ±0.01) |
| UNSW-NB15 | Late − early fusion | +0.016 [+0.001, +0.031] (5/5) | 0.038 | 0.135 | inconclusive (CI crosses ±0.01) |
| UNSW-NB15 | Attention − early fusion | +0.008 [−0.004, +0.021] (3/5) | 0.128 | 0.135 | inconclusive (CI crosses ±0.01) |
| UNSW-NB15 | Late fusion − wide early fusion | +0.008 [+0.001, +0.014] (5/5) | 0.027 | 0.135 | inconclusive (CI crosses ±0.01) |
| UNSW-NB15 | Late fusion − late fusion on all features | +0.011 [+0.004, +0.018] (5/5) | 0.013 | 0.076 | inconclusive (CI crosses ±0.01) |

| Dataset | Model | Macro-F1, all test records | Macro-F1, without a training match | Accuracy on records with a training match | Accuracy on records without a match |
|---|---|---|---|---|---|
| TON-IoT | Early fusion | 0.948 | 0.948 | 0.960 | 0.982 |
| TON-IoT | Late fusion | 0.953 | 0.953 | 0.977 | 0.984 |
| TON-IoT | Attention fusion | 0.949 | 0.949 | 0.971 | 0.983 |
| TON-IoT | Early fusion with head | 0.951 | 0.951 | 0.971 | 0.983 |
| TON-IoT | Late fusion on all features | 0.951 | 0.951 | 0.977 | 0.984 |
| TON-IoT | RF, all features | 0.961 | 0.961 | 1.000 | 0.987 |
| TON-IoT | GBDT, all features | 0.962 | 0.962 | 1.000 | 0.987 |
| UNSW-NB15 | Early fusion | 0.557 | 0.554 | 0.642 | 0.786 |
| UNSW-NB15 | Late fusion | 0.573 | 0.579 | 0.643 | 0.796 |
| UNSW-NB15 | Attention fusion | 0.565 | 0.568 | 0.644 | 0.788 |
| UNSW-NB15 | Early fusion with head | 0.563 | 0.563 | 0.643 | 0.791 |
| UNSW-NB15 | Late fusion on all features | 0.562 | 0.567 | 0.638 | 0.793 |
| UNSW-NB15 | RF, all features | 0.607 | 0.652 | 0.636 | 0.831 |
| UNSW-NB15 | GBDT, all features | 0.608 | 0.645 | 0.635 | 0.835 |

| Model | TON-IoT macro-F1 | TON-IoT PR-AUC | TON-IoT detection rate | TON-IoT FPR | UNSW-NB15 macro-F1 | UNSW-NB15 PR-AUC | UNSW-NB15 detection rate | UNSW-NB15 FPR |
|---|---|---|---|---|---|---|---|---|
| Stream 1 only (CNN) | 0.904 | 0.946 | 0.998 | 0.073 | 0.497 | 0.549 | 0.997 | 0.205 |
| Stream 2 only (CNN) | 0.829 | 0.865 | 0.998 | 0.047 | 0.361 | 0.364 | 0.998 | 0.201 |
| Early fusion (CNN, all features) | 0.948 | 0.974 | 0.999 | 0.016 | 0.557 | 0.621 | 0.995 | 0.188 |
| Late fusion | 0.953 | 0.975 | 0.999 | 0.011 | 0.573 | 0.635 | 0.995 | 0.186 |
| Attention fusion | 0.949 | 0.974 | 0.999 | 0.016 | 0.565 | 0.632 | 0.998 | 0.195 |
| Wide early fusion (×1.6) | 0.950 | 0.974 | 0.999 | 0.014 | 0.565 | 0.625 | 0.994 | 0.185 |
| Early fusion with head | 0.951 | 0.973 | 0.999 | 0.013 | 0.563 | 0.627 | 0.994 | 0.186 |
| Late fusion on all features | 0.951 | 0.973 | 0.999 | 0.012 | 0.562 | 0.619 | 0.993 | 0.184 |
| MLP stream 1 | 0.904 | 0.943 | 0.998 | 0.077 | 0.498 | 0.559 | 0.997 | 0.202 |
| MLP stream 2 | 0.829 | 0.865 | 0.998 | 0.047 | 0.374 | 0.368 | 0.998 | 0.202 |
| MLP early fusion | 0.948 | 0.972 | 0.999 | 0.015 | 0.552 | 0.619 | 0.994 | 0.188 |
| MLP late fusion | 0.953 | 0.976 | 0.999 | 0.011 | 0.569 | 0.624 | 0.994 | 0.184 |
| MLP attention fusion | 0.951 | 0.973 | 0.999 | 0.014 | 0.564 | 0.628 | 0.999 | 0.197 |
| RF, all features | 0.961 | 0.978 | 1.000 | 0.007 | 0.607 | 0.651 | 0.982 | 0.147 |
| RF, stream 1 | 0.928 | 0.946 | 0.998 | 0.016 | 0.546 | 0.594 | 0.971 | 0.151 |
| RF, stream 2 | 0.796 | 0.814 | 0.997 | 0.039 | 0.368 | 0.377 | 0.997 | 0.193 |
| GBDT, all features | 0.962 | 0.981 | 0.999 | 0.008 | 0.608 | 0.657 | 0.968 | 0.127 |
| GBDT, stream 1 | 0.925 | 0.962 | 0.998 | 0.045 | 0.540 | 0.588 | 0.974 | 0.155 |
| GBDT, stream 2 | 0.836 | 0.873 | 0.998 | 0.036 | 0.365 | 0.374 | 0.998 | 0.196 |

## References

AboulEla, S., & Kashef, R. (2026). MM-NIDS: A novel multimodal ensemble fusion network intrusion detection system using numeric, text, graph, and quantum representations. *Sensors, 26*(13), 4196. https://doi.org/10.3390/s26134196

Afraji, D. M. A. A., Lloret, J., & Peñalver, L. (2025). An integrated hybrid deep learning framework for intrusion detection in IoT and IIoT networks using CNN-LSTM-GRU architecture. *Computation, 13*(9), 222. https://doi.org/10.3390/computation13090222

Al Qathrady, M., Ullah, S., Alshehri, M. S., Ahmad, J., Almakdi, S., Alqhtani, S. M., Khan, M. A., & Ghaleb, B. (2024). SACNN-IDS: A self-attention convolutional neural network for intrusion detection in industrial internet of things. *CAAI Transactions on Intelligence Technology, 9*(6), 1398–1411. https://doi.org/10.1049/cit2.12352

Alabbadi, A., & Bajaber, F. (2025). X-FuseRLSTM: A cross-domain explainable intrusion detection framework in IoT using the attention-guided dual-path feature fusion and residual LSTM. *Sensors, 25*(12), 3693. https://doi.org/10.3390/s25123693

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., & Anwar, A. (2020). TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems. *IEEE Access, 8*, 165130–165150. https://doi.org/10.1109/ACCESS.2020.3022862

Apruzzese, G., Laskov, P., & Schneider, J. (2023). SoK: Pragmatic assessment of machine learning for network intrusion detection. In *2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P)* (pp. 592–614). IEEE. https://doi.org/10.1109/EuroSP57164.2023.00042

Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., & Rieck, K. (2022). Dos and don'ts of machine learning in computer security. In *31st USENIX Security Symposium* (pp. 3971–3988). USENIX Association. https://www.usenix.org/conference/usenixsecurity22/presentation/arp

Attota, D. C., Mothukuri, V., Parizi, R. M., & Pouriyeh, S. (2021). An ensemble multi-view federated learning intrusion detection for IoT. *IEEE Access, 9*, 117734–117745. https://doi.org/10.1109/ACCESS.2021.3107337

Ayantayo, A., Kaur, A., Kour, A., Schmoor, X., Shah, F., Vickers, I., Kearney, P., & Abdelsamea, M. M. (2023). Network intrusion detection using feature fusion with deep learning. *Journal of Big Data, 10*(1), Article 167. https://doi.org/10.1186/s40537-023-00834-0

Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. In *3rd International Conference on Learning Representations (ICLR 2015)*. arXiv:1409.0473.

Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 41*(2), 423–443. https://doi.org/10.1109/TPAMI.2018.2798607

Chaabouni, N., Mosbah, M., Zemmari, A., Sauvignac, C., & Faruki, P. (2019). Network intrusion detection for IoT security based on learning techniques. *IEEE Communications Surveys & Tutorials, 21*(3), 2671–2701. https://doi.org/10.1109/COMST.2019.2896380

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Dinh, P. V., Hoang, D. T., Uy, N. Q., Nguyen, D. N., Bao, S. P., & Dutkiewicz, E. (2024a). Multiple-input auto-encoder for IoT intrusion detection systems with heterogeneous data. In *ICC 2024 – IEEE International Conference on Communications* (pp. 2707–2712). IEEE. https://doi.org/10.1109/ICC51166.2024.10622942

Dinh, P. V., Nguyen, D. N., Hoang, D. T., Nguyen, Q. U., Dutkiewicz, E., & Bao, S. P. (2024b). *Multiple-input auto-encoder guided feature selection for IoT intrusion detection systems* (arXiv:2403.15511v1). arXiv. https://doi.org/10.48550/arXiv.2403.15511

Diro, A. A., & Chilamkurti, N. (2018). Distributed attack detection scheme using deep learning approach for Internet of Things. *Future Generation Computer Systems, 82*, 761–768. https://doi.org/10.1016/j.future.2017.08.043

Du, C., Teng, J., Li, T., Liu, Y., Yuan, T., Wang, Y., Yuan, Y., & Zhao, H. (2023). On uni-modal feature learning in supervised multi-modal learning. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 8632–8656). PMLR.

Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection dataset: The CICIDS2017 case study. In *2021 IEEE Security and Privacy Workshops (SPW)* (pp. 7–12). IEEE. https://doi.org/10.1109/SPW53761.2021.00009

Ferrag, M. A., Friha, O., Hamouda, D., Maglaras, L., & Janicke, H. (2022). Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning. *IEEE Access, 10*, 40281–40306. https://doi.org/10.1109/ACCESS.2022.3165809

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. In *Advances in Neural Information Processing Systems 34* (pp. 18932–18943). Curran Associates. https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://doi.org/10.52202/068431-0037

Ho, T. K. (1998). The random subspace method for constructing decision forests. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 20*(8), 832–844. https://doi.org/10.1109/34.709601

Huang, Y., Lin, J., Zhou, C., Yang, H., & Huang, L. (2022). Modality competition: What makes joint training of multi-modal network fail in deep learning? (Provably). In *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 9226–9259). PMLR. https://proceedings.mlr.press/v162/huang22e.html

Imrana, Y., Xiang, Y., Ali, L., & Abdul-Rauf, Z. (2021). A bidirectional LSTM deep learning approach for intrusion detection. *Expert Systems with Applications, 185*, 115524. https://doi.org/10.1016/j.eswa.2021.115524

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (Vol. 37, pp. 448–456). PMLR. https://proceedings.mlr.press/v37/ioffe15.html

Ishtiaq, W., Zannat, A., Parvez, A. H. M. S., Hossain, M. A., Kanchan, M. H., & Tarek, M. M. (2025). CST-AFNet: A dual attention-based deep learning framework for intrusion detection in IoT networks. *Array, 27*, 100501. https://doi.org/10.1016/j.array.2025.100501

Jain, S., & Wallace, B. C. (2019). Attention is not explanation. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Vol. 1* (pp. 3543–3556). ACL. https://doi.org/10.18653/v1/N19-1357

Kiflay, A., Tsokanos, A., Fazlali, M., & Kirner, R. (2024). Network intrusion detection leveraging multimodal features. *Array, 22*, 100349. https://doi.org/10.1016/j.array.2024.100349

Lansky, J., Ali, S., Mohammadi, M., Majeed, M. K., Karim, S. H. T., Rashidi, S., Hosseinzadeh, M., & Rahmani, A. M. (2021). Deep learning-based intrusion detection systems: A systematic review. *IEEE Access, 9*, 101574–101599. https://doi.org/10.1109/ACCESS.2021.3097247

Li, M., Qiao, Y., & Lee, B. (2025). A comparative analysis of single and multi-view deep learning for cybersecurity anomaly detection. *IEEE Access, 13*, 83996–84012. https://doi.org/10.1109/ACCESS.2025.3564066

Liang, P. P., Zadeh, A., & Morency, L.-P. (2024). Foundations & trends in multimodal machine learning: Principles, challenges, and open questions. *ACM Computing Surveys, 56*(10), 1–42. https://doi.org/10.1145/3656580

McElfresh, D., Khandagale, S., Valverde, J., Prasad C, V., Ramakrishnan, G., Goldblum, M., & White, C. (2023). When do neural nets outperform boosted trees on tabular data? In *Advances in Neural Information Processing Systems 36* (pp. 76336–76369). Curran Associates. https://doi.org/10.52202/075280-3337

Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Shabtai, A., Breitenbacher, D., & Elovici, Y. (2018). N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders. *IEEE Pervasive Computing, 17*(3), 12–22. https://doi.org/10.1109/MPRV.2018.03367731

Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). Kitsune: An ensemble of autoencoders for online network intrusion detection. In *Network and Distributed System Security Symposium (NDSS 2018)*. https://doi.org/10.14722/ndss.2018.23204

Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society, 72*, 102994. https://doi.org/10.1016/j.scs.2021.102994

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In *2015 Military Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE. https://doi.org/10.1109/MilCIS.2015.7348942

Nagrani, A., Yang, S., Arnab, A., Jansen, A., Schmid, C., & Sun, C. (2021). Attention bottlenecks for multimodal fusion. In *Advances in Neural Information Processing Systems 34* (pp. 14200–14213). Curran Associates.

Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A. (2023). CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment. *Sensors, 23*(13), 5941. https://doi.org/10.3390/s23135941

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), 2825–2830.

Peng, X., Wei, Y., Deng, A., Wang, D., & Hu, D. (2022). Balanced multimodal learning via on-the-fly gradient modulation. In *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 8228–8237). IEEE. https://doi.org/10.1109/CVPR52688.2022.00806

Ramachandram, D., & Taylor, G. W. (2017). Deep multimodal learning: A survey on recent advances and trends. *IEEE Signal Processing Magazine, 34*(6), 96–108. https://doi.org/10.1109/MSP.2017.2738401

Sanjalawe, Y., Fraihat, S., Al-E'mari, S., & Makhadmeh, S. N. (2025). AOA-SMA-EGRUAttNet: A hybrid feature selection and dual-stream attention-based intrusion detection framework for IIoT systems. *Internet of Things and Cyber-Physical Systems, 5*, 143–164. https://doi.org/10.1016/j.iotcps.2026.03.002

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90. https://doi.org/10.1016/j.inffus.2021.11.011

Singh, P., Huang, Y., Li, H., Oikonomou, G., & Pope, J. (2026). Multi-modal model for embedding network and audit data for IoT anomaly detection. In *Proceedings of the 12th International Conference on Information Systems Security and Privacy (ICISSP)* (pp. 396–405). SciTePress. https://doi.org/10.5220/0014470300004061

Sommer, R., & Paxson, V. (2010). Outside the closed world: On using machine learning for network intrusion detection. In *2010 IEEE Symposium on Security and Privacy* (pp. 305–316). IEEE. https://doi.org/10.1109/SP.2010.25

Ullah, F., Turab, A., Ullah, S., Cacciagrano, D., & Zhao, Y. (2024). Enhanced network intrusion detection system for Internet of Things security using multimodal big data representation with transfer learning and game theory. *Sensors, 24*(13), 4152. https://doi.org/10.3390/s24134152

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30* (NIPS 2017).

Wang, W., Tran, D., & Feiszli, M. (2020). What makes training multi-modal classification networks hard? In *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 12692–12702). IEEE. https://doi.org/10.1109/CVPR42600.2020.01271

Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 11–20). ACL. https://doi.org/10.18653/v1/D19-1002

Wu, N., Jastrzębski, S., Cho, K., & Geras, K. J. (2022). Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In *Proceedings of the 39th International Conference on Machine Learning* (PMLR 162, pp. 24043–24055).

Xu, C., Zhan, Y., Wang, Z., & Yang, J. (2025). Multimodal fusion based few-shot network intrusion detection system. *Scientific Reports, 15*(1), Article 21986. https://doi.org/10.1038/s41598-025-05217-4
