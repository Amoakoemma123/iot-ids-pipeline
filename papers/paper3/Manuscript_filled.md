# Paper 3 v2 manuscript (double-brace placeholders are filled from values.json by build.py)

## Title

**Learning New Attack Families After Deployment: A Controlled Evaluation of Class-Incremental Learning for Compact IoT Intrusion Detectors**

## Abstract

New attack families appear after an intrusion detection system (IDS) is deployed, and retraining from scratch requires keeping all past data. We evaluate class-incremental learning for a compact 1D-CNN detector on TON-IoT, UNSW-NB15 and N-BaIoT, comparing fine-tuning, online EWC, learning without forgetting (LwF), A-GEM and experience replay with 500 or 2,000 stored records against joint retraining and random-forest baselines. The protocol keeps benign traffic in every task, fits feature selection and scaling on the first task only, and uses five seeds on a semantic task order plus four random orders. Replay with 2,000 records recovered 81–91% of the macro-F1 difference between fine-tuning and retraining on the semantic order (64–91% over random orders), ahead of every non-replay strategy in every order. Online EWC did not significantly improve on fine-tuning, LwF helped little, and A-GEM improved on it significantly on TON-IoT and N-BaIoT, closing at most 57% of the gap. A random forest retrained on the same memory was better than the CNN with ER-2000 on all three datasets, but worse than the CNN with ER-2000-pool on UNSW-NB15. At the alert level forgetting was milder: fine-tuning still flagged 0.698–0.875 of the attack records of earlier tasks. Replay raised the benign false-positive rate on TON-IoT to 0.132; a half-benign memory lowered it to 0.075, and adding the memory to each task's training pool instead of every minibatch to 0.030. Fitting the pre-processing on future data changed mean final macro-F1 by at most 0.029 (inconclusive on N-BaIoT).

**Keywords:** intrusion detection; Internet of Things; continual learning; class-incremental learning; catastrophic forgetting; experience replay; evaluation methodology

## 1. Introduction

An intrusion detection system (IDS) deployed in an Internet of Things (IoT) network meets attacks that did not exist when it was trained. Retraining from scratch on all data collected so far is the simplest response, but it requires keeping every past record and repeating the full training, which may conflict with data-retention limits and is costly for operators who update often. Continual learning offers an alternative: update the deployed model with the new attack family while retaining what it learned before (De Lange et al., 2022; van de Ven et al., 2022). Continual learning for network intrusion detection has been studied on several benchmarks (e.g. Amalapuram et al., 2023; Talpini et al., 2024; Zhang et al., 2025).

How well these methods work for IDS depends on details of the evaluation that are easy to get wrong. If the pre-processing (feature selection and scaling) is fitted on data that include future attack families, the first model already "knows" the later tasks. If benign traffic appears only in the first task, later tasks reduce to separating attacks from each other, which overstates how hard it is to keep benign traffic recognised. Without an upper bound from joint retraining on all data so far, the cost of forgetting cannot be put in proportion, and a single seed or a single task order cannot separate a method's effect from chance. Finally, an IDS is judged first by whether it raises an alert at all; a model that confuses two attack families but still flags both behaves very differently from one that lets them pass as benign.

The general continual-learning literature predicts that regularisation fails and replay works in the class-incremental scenario (van de Ven et al., 2022), and Talpini et al. (2024) have evaluated class-incremental intrusion detection with benign traffic in every task and a full-data bound. This paper is a controlled replication and extension of these two lines of work for IoT intrusion detection with a compact detector. Our contributions are:

1. A class-incremental evaluation of fine-tuning, joint retraining, online EWC, LwF, experience replay (500 and 2,000 records) and A-GEM for a compact IoT detector on TON-IoT, UNSW-NB15 and N-BaIoT, with benign traffic in every task, feature selection and scaling fitted on the first task only, and five seeds on a semantic task order plus four random orders.
2. Reporting views that we recommend for IDS evaluations: retention measured at two levels, the correct attack class and any alert, together with the benign false-positive rate and the share of not-yet-learned attacks flagged as attacks.
3. An ablation that fits feature selection and scaling on all data instead of the first task; under our pipeline this common shortcut changed results only marginally on two datasets and inconclusively on the third.
4. Two random-forest baselines, retrained on all data and on the same replay memory as the network, which show what a cheap non-neural detector reaches with the same storage, and which revealed that the network was under-trained on badly scaled inputs (Section 5.6). A replay-weighting check shows that how the memory enters the loss is a design setting that moves replay's false-positive cost and where its remaining gap lies (Sections 5.2–5.3).

## 2. Related Work

### 2.1 Continual learning

A network trained sequentially on new data tends to lose what it learned before, a problem known since the early connectionist literature (McCloskey & Cohen, 1989; Robins, 1995) and reviewed extensively since (Parisi et al., 2019; De Lange et al., 2022; Wang et al., 2024). Three scenarios differ in difficulty: in task-incremental learning the task identity is given at test time, in domain-incremental learning the label space stays fixed, and in class-incremental learning new classes are added to a single classifier without task identity (van de Ven et al., 2022). Detecting new attack families with one detector is class-incremental. Methods fall into three groups. Regularisation methods penalise changes to parameters important for earlier tasks, such as elastic weight consolidation (EWC; Kirkpatrick et al., 2017) and its online variant (Schwarz et al., 2018), or distil the previous model's outputs, as in learning without forgetting (LwF; Li & Hoiem, 2018). Replay methods keep a small memory of past records, from plain experience replay (Chaudhry et al., 2019b) to exemplar-based classifiers such as iCaRL (Rebuffi et al., 2017). Gradient-projection methods constrain updates so that the loss on the memory does not increase (GEM, Lopez-Paz & Ranzato, 2017; A-GEM, Chaudhry et al., 2019a). Evaluation choices strongly affect how these methods compare: multi-head or task-incremental evaluations overstate their success (Farquhar & Gal, 2018), controlled comparisons show that implementation tricks and evaluation settings change the ranking of methods (Mai et al., 2022), and in the class-incremental scenario regularisation methods largely fail while replay works (van de Ven et al., 2022).

### 2.2 Continual learning and drift in intrusion detection

Security models age as the data they see drift away from their training data. In malware detection, spatial and temporal bias in evaluation inflates results (Pendlebury et al., 2019), drifting samples can be detected and rejected (Jordaney et al., 2017; Barbero et al., 2022; Yang et al., 2021), and adaptation without labels has clear limits (Kan et al., 2021). Rahman et al. (2022) found that, in class-incremental malware classification, regularisation methods and LwF performed close to no adaptation and only the exemplar-based iCaRL approached joint retraining; in the domain-incremental setting, replaying 20–50% of past data outperformed all continual-learning methods. For network intrusion detection, Andresini et al. (2021) study concept drift with label-scarce updates on a refined CICIDS-2017, including attacks that appear only after deployment, and Shyaa et al. (2024) survey drift-aware IDS.

Continual learning for network intrusion detection has grown quickly. One group has studied class imbalance in class-incremental NIDS (Amalapuram et al., 2021), compared EWC and GEM in class- and domain-incremental settings (Amalapuram et al., 2022), and proposed augmented replay (Amalapuram et al., 2023), semi-supervised continual learning (Amalapuram et al., 2024) and practical guidance (Amalapuram et al., 2025). Other work compares LwF and replay variants under covariate shift (Prasath et al., 2022), applies continual learning to multi-class detection with a CNN (Oikonomou et al., 2023), shares replay across distributed detectors (Faber et al., 2023), learns new attack classes in federated IIoT settings (Mao et al., 2024), and refines replay memories (Zhang et al., 2025; Guo et al., 2025; A H et al., 2025).

The protocols differ in ways that matter for our questions. Four studies are closest to ours; their protocols differ as follows. Amalapuram et al. (2023), the broadest benchmark, include benign traffic in every task, but their scenario is binary and domain-incremental, min–max scaling is applied to the whole dataset before training, and no joint-retraining bound is reported. Zhang et al. (2025) evaluate a binary drift stream against a static model rather than a joint bound. Andresini et al. (2021) use an oracle-label update as reference. Talpini et al. (2024) come closest to our protocol: class-incremental tasks with benign traffic always present, a full-data bound and 15 random task orders; their study uses CICIDS-2017 only and updates the standardisation with every task. None of these four separates whether an old attack is still named correctly from whether it is still flagged at all, measures how attacks not yet learned are flagged, or isolates the effect of fitting the pre-processing on future data. Our study adds these views on two IoT datasets and one enterprise dataset, with a compact detector, and tests whether the prediction from the general literature, that regularisation fails and replay works in the class-incremental scenario, holds for intrusion detection.

## 3. Data and Tasks

### 3.1 Datasets

TON-IoT contains network flows from an IoT/IIoT testbed with nine attack classes and benign traffic; we use its network dataset, train_test_network.csv (Moustafa, 2021; see also Alsaedi et al., 2020). UNSW-NB15 contains flows from an enterprise-style testbed with nine attack categories (Moustafa & Slay, 2015); we use its 175,341-record partitioned training set. It is not IoT traffic, and we keep it as a second benchmark. N-BaIoT contains benign, BASHLITE and Mirai traffic of nine commercial IoT devices, described by 115 statistics computed by the Kitsune feature extractor (Meidan et al., 2018; Mirsky et al., 2018). TON-IoT and UNSW-NB15 are loaded as follows: identifier and free-text columns removed, categorical fields with at most 30 values one-hot encoded, exact duplicates removed after all column drops, a stratified 70/15/15 split, and validation and test records identical to a training record removed. N-BaIoT is prepared by drawing a uniform 3% sample per device and class, which keeps each device's class mix and brings the 7.06 million records to a size comparable to the other datasets; exact duplicates at 32-bit precision and the 17 vectors that occur with conflicting labels are removed. The two BASHLITE flood classes (TCP and UDP) are dropped because only 40 and 32 distinct vectors remain after de-duplication. A stratified 70/15/15 split by device and class follows, and validation and test records identical to a training record are removed.

### 3.2 Tasks

Each dataset's attack classes are divided into three tasks (Table 1). The semantic order groups related families: on TON-IoT, scanning and DoS first, then DDoS, password and MITM, then backdoor, ransomware, injection and XSS; on UNSW-NB15, generic and exploits, then fuzzers, reconnaissance and DoS, then analysis, backdoor, shellcode and worms; on N-BaIoT, the BASHLITE classes, then three Mirai classes, then the two Mirai UDP classes. Four random orders of the same task sizes test whether conclusions depend on the order.

Benign training and validation records are divided into three disjoint shares, one per task, so that every task contains benign traffic, as a deployed detector would see. The test set is never divided: after every task the model is evaluated on the full test set, including attack families it has not yet seen.

| Dataset | Task | Attack classes | Training records | Attack test records |
|---|---|---|---|---|
| TON-IoT | 0 | scanning, DoS | 33,085 | 4,992 |
| TON-IoT | 1 | DDoS, password, MITM | 38,395 | 6,131 |
| TON-IoT | 2 | backdoor, ransomware, injection, XSS | 57,004 | 10,120 |
| UNSW-NB15 | 0 | generic, exploits | 28,688 | 3,553 |
| UNSW-NB15 | 1 | fuzzers, reconnaissance, DoS | 30,720 | 3,989 |
| UNSW-NB15 | 2 | analysis, backdoor, shellcode, worms | 14,759 | 568 |
| N-BaIoT | 0 | gafgyt combo, gafgyt junk, gafgyt scan | 25,481 | 3,931 |
| N-BaIoT | 1 | mirai scan, mirai ack, mirai syn | 44,092 | 8,390 |
| N-BaIoT | 2 | mirai udp, mirai udpplain | 40,654 | 7,647 |

### 3.3 Pre-processing fitted on the first task

Dataset-level cleaning uses the whole training split under both protocols below: numeric columns with more than 40% missing values are dropped, missing values are imputed with the median, infinite values are replaced by the training 99th percentile, zero-variance columns are dropped, and the one-hot schema is fixed. Feature selection and scaling, by contrast, are fitted on the Task-0 training records only, because the later attack families do not exist when the detector is first built. For TON-IoT and UNSW-NB15, a mutual-information filter (threshold 0.05) is followed by recursive feature elimination with cross-validation (Guyon et al., 2002), using a 50-tree random forest, three-fold stratified cross-validation scored by macro-F1, at least 10 features and a subsample of at most 50,000 records; it keeps 13 features on TON-IoT and 39 on UNSW-NB15. N-BaIoT uses all 115 features without selection. Min–max scaling is then fitted on the Task-0 records. After scaling, every feature is compressed by the fixed monotone map x ↦ asinh(1000x)/asinh(1000), which leaves 0 and 1 unchanged, has no fitted parameters and extrapolates smoothly to later-task values outside the Task-0 range; Section 5.6 shows why the network needs it. Finally, the compressed values are clipped to [−1, 2], which corresponds to about −1 to 2,000 in min–max units. The **oracle** ablation fits the feature selection and scaling on all training records instead (fine-tuning, ER-2000 and joint retraining, five seeds, all three datasets); on N-BaIoT it changes only the min–max range.

## 4. Models and Strategies

### 4.1 Network

All strategies use a compact 1D-CNN: two convolutional blocks (32 and 64 filters, kernel 3, batch normalisation (Ioffe & Szegedy, 2015) with momentum 0.9, max-pooling), global average pooling, dense layers of 128 and 64 units with dropout 0.3 and a single output layer over all classes, 23,946 parameters on TON-IoT and UNSW-NB15 and 23,881 on N-BaIoT. A kernel-3 convolution over a tabular feature vector treats neighbouring features as related, an assumption we did not test (Section 6.2). Training and prediction are restricted to the classes seen so far (logits of unseen classes are masked). The loss is class-weighted cross-entropy on the training pool of the current stage (weights inversely proportional to class frequency in that pool, capped at 50). The network is trained with Adam (Kingma & Ba, 2015): on Task 0 and for joint retraining with learning rate 10⁻³ for at most 100 epochs (patience 15), on later tasks with 5 × 10⁻⁴ for at most 50 epochs (patience 10), batch size 512. Early stopping monitors macro-F1 on the validation records of the current task (its benign share and its new attack classes); the incremental strategies do not keep validation data of earlier tasks, whereas joint retraining uses the validation records of all tasks so far. The best weights are always restored. Task 0 is trained once per seed, and the same weights are reused by every strategy.

### 4.2 Strategies

- **Fine-tuning** (lower bound): train on each new task's data only.
- **Joint retraining** (upper bound): retrain a fresh network on all training data seen so far.
- **Online EWC** (Kirkpatrick et al., 2017; Schwarz et al., 2018): quadratic penalty on parameter changes weighted by a per-sample empirical Fisher information, estimated on 1,000 training records of the task just learned with their true labels and summed over tasks without decay (γ = 1); the penalty is anchored at the most recent parameters.
- **LwF** (Li & Hoiem, 2018): distillation of the previous model's outputs, computed on the current task's data, over all previously seen classes, benign included (temperature 2).
- **Experience replay** (ER; Chaudhry et al., 2019b): a class-balanced episodic memory of at most 500 or 2,000 training records (equal slots per seen class; at the end 1,889 records on UNSW-NB15, where some rare classes have fewer records than their share). Each minibatch of 512 current-task records is paired with 512 memory records drawn with replacement, so the memory supplies half of every gradient step, and the class weights of the loss are computed over the current task and the memory together. Each memory class holds only a few hundred records and therefore receives a high weight, so earlier classes are up-weighted twice, by oversampling and by the loss weights. The replay-weighting check of Section 4.3 removes the oversampling and keeps the loss weights.
- **A-GEM** (Chaudhry et al., 2019a): the same 2,000-record memory serves only as a constraint. At each step a reference gradient is computed on 512 memory records; if the current-task gradient has a negative inner product with it, the gradient is projected so that, to first order, the memory loss does not increase. The memory loss weights every stored class equally (class weights computed over the memory) and is evaluated in inference mode, so the memory batch neither updates the batch-normalisation statistics nor uses dropout. A first implementation weighted the memory loss with the current task's class weights, which gave the earlier attack classes zero weight; all A-GEM results reported here use the corrected version.

The EWC penalty weight (10⁰–10⁷ in factors of 10) and the LwF weight (0.1, 0.25, 0.5, 1, 2, 4, 8, 16 and 32) are tuned on a separate development seed, choosing the value with the highest validation macro-F1 over all seen classes after the last task. ER and A-GEM were not tuned (memory sampling ratio and learning rate fixed), which favours the tuned strategies; the replay-weighting check (Section 5.2) shows that the way the memory is sampled can matter.

### 4.3 References and checks

Two random forests (200 trees, depth 20, minimum leaf size 2, same Task-0 pre-processing) serve as non-neural baselines. The first is retrained on all task data so far, after every task. The second (RF + memory) is retrained after every task on the current task's data plus the same 2,000-record memory as ER-2000; for a given seed the memory holds the same records. It is an engineering baseline at the storage cost of replay.

Four further checks address specific questions raised by the results:

- **Replay-weighting check** (ER-2000-pool; all three datasets, five seeds): the same memory is added once to the training pool of each new task instead of being mixed 1:1 into every minibatch. The class weights over task and memory are unchanged, so every seen class receives the same total weight per epoch (up to the weight cap of 50). The check also changes how often the memory is revisited: each memory record is used once per epoch, instead of on average as many times as the current task has records divided by the memory size. It tests whether the location of replay's shortfall and its benign false-positive cost depend on the double weighting of Section 4.2.
- **Benign-share memory** (ER-2000b; TON-IoT, five seeds): half of the 2,000-record memory is reserved for benign records and the rest divided equally among the attack classes, to test whether the higher benign false-positive rate of class-balanced replay comes from under-representing benign traffic.
- **Budget check** (all three datasets, five seeds): ER-2000 is rerun with the Task-0/joint budget (at most 100 epochs, patience 15, learning rate 10⁻³) on every incremental task (Section 5.2).
- **Input-map × budget factorial** (TON-IoT, five seeds): joint retraining is rerun with each combination of input map (min–max only, or min–max followed by the compression map) and training budget (at most 40 epochs with patience 5, as in a first run of this protocol, or the present 100 epochs with patience 15; joint retraining uses this budget at every stage) (Section 5.6).

### 4.4 Metrics

After learning task t, the full test set is predicted and the confusion matrix stored. For each earlier or current task j, **class-level retention** R[t, j] is the mean recall of task j's attack classes, and **detection-level retention** Rd[t, j] the share of task j's attack records flagged as any attack. From these we compute average accuracy A = mean_j R[T, j] after the last task, plasticity = mean_j R[j, j], forgetting FM = mean over j < T of (max over t < T of R[t, j] − R[T, j]) and backward transfer BWT = mean over j < T of (R[T, j] − R[j, j]), and the same quantities for Rd. When each task's best retention is reached as it is learned, as for almost all strategies here, BWT equals −FM; Table 2 therefore reports FM and plasticity. Retention of the earlier tasks alone (mean over j < T of R[T, j] or Rd[T, j]) excludes the task just learned. We also report final macro-F1 over all classes, the benign false-positive rate after each task, and **not-yet-learned flagging**: for tasks not yet learned, the share of their attack records flagged as an attack (summarised as the mean over all stages and not-yet-learned tasks).

### 4.5 Statistics

Each strategy is run with five seeds (0–4) on the semantic order and once on each of four random orders, with the seed equal to the order index, so order and seed effects are confounded there. The seed controls network initialisation, batch order and memory sampling; the data split, the benign shares of each order and the pre-processing fit are the same for all seeds, so the reported variability covers training randomness only. The confirmatory comparisons are the six strategies against fine-tuning on final macro-F1, paired over seeds (paired t-test, Holm correction over the six comparisons within each dataset); we report the paired differences with 95% t-intervals. All other comparisons, including the checks of Section 4.3, are exploratory and reported with uncorrected p-values.

## 5. Results

### 5.1 Overview

Table 2 summarises the semantic task order over five seeds. For a given seed, all seven strategies start from the same Task-0 model, and the stored Task-0 predictions are identical across strategies on all three datasets. That model reaches a macro-F1 of 0.987 on TON-IoT, 0.921 on UNSW-NB15 and 0.987 on N-BaIoT over benign traffic and the Task-0 attack classes.

After the last task, joint retraining reaches a final macro-F1 of 0.922, 0.520 and 0.987. The random forest retrained on all data reaches 0.965, 0.575 and 1.000, so the compact network's upper bound lies 0.042, 0.054 and 0.013 below the tree reference. The low UNSW-NB15 values reflect that dataset's small, overlapping attack classes, which limit every model.

Fine-tuning ends at 0.358, 0.130 and 0.253, with class-level forgetting of 0.970, 0.872 and 0.956. After the last task it recognises none of the Task-0 attack classes on any dataset (retention 0.000, 0.000, 0.000). Joint retraining also shows class-level forgetting on UNSW-NB15 (FM 0.156): even a model retrained from scratch recognises earlier classes less well once overlapping new classes are added.

| Dataset | Strategy | Final macro-F1 | A (class) | FM (class) | Plasticity (class) | A (detection) | FM (detection) | Benign FPR |
|---|---|---|---|---|---|---|---|---|
| TON-IoT | Fine-tuning | 0.358 ± 0.001 | 0.328 | 0.970 | 0.975 | 0.916 | 0.113 | 0.005 |
| TON-IoT | Online EWC | 0.355 ± 0.002 | 0.327 | 0.971 | 0.975 | 0.925 | 0.101 | 0.009 |
| TON-IoT | LwF | 0.420 ± 0.015 | 0.377 | 0.898 | 0.976 | 0.965 | 0.042 | 0.020 |
| TON-IoT | A-GEM | 0.531 ± 0.048 | 0.512 | 0.694 | 0.975 | 0.928 | 0.098 | 0.007 |
| TON-IoT | ER-500 | 0.855 ± 0.036 | 0.919 | 0.026 | 0.936 | 0.992 | 0.002 | 0.116 |
| TON-IoT | ER-2000 | 0.872 ± 0.008 | 0.941 | 0.001 | 0.941 | 0.997 | -0.006 | 0.132 |
| TON-IoT | Joint retraining | 0.922 ± 0.007 | 0.957 | 0.018 | 0.969 | 0.993 | 0.000 | 0.032 |
| TON-IoT | Random forest (all data) | 0.965 ± 0.001 | 0.971 | | | 0.998 | | 0.005 |
| UNSW-NB15 | Fine-tuning | 0.130 ± 0.001 | 0.254 | 0.872 | 0.835 | 0.794 | 0.296 | 0.039 |
| UNSW-NB15 | Online EWC | 0.172 ± 0.043 | 0.232 | 0.662 | 0.673 | 0.963 | 0.049 | 0.189 |
| UNSW-NB15 | LwF | 0.194 ± 0.003 | 0.328 | 0.763 | 0.837 | 0.991 | 0.009 | 0.191 |
| UNSW-NB15 | A-GEM | 0.167 ± 0.047 | 0.307 | 0.798 | 0.839 | 0.818 | 0.263 | 0.048 |
| UNSW-NB15 | ER-500 | 0.373 ± 0.017 | 0.452 | 0.107 | 0.524 | 0.997 | -0.017 | 0.218 |
| UNSW-NB15 | ER-2000 | 0.461 ± 0.004 | 0.541 | 0.001 | 0.541 | 0.998 | -0.007 | 0.205 |
| UNSW-NB15 | Joint retraining | 0.520 ± 0.007 | 0.670 | 0.156 | 0.774 | 0.999 | -0.001 | 0.203 |
| UNSW-NB15 | Random forest (all data) | 0.575 ± 0.003 | 0.574 | | | 0.945 | | 0.081 |
| N-BaIoT | Fine-tuning | 0.253 ± 0.034 | 0.337 | 0.956 | 0.975 | 0.917 | 0.125 | 0.000 |
| N-BaIoT | Online EWC | 0.307 ± 0.013 | 0.390 | 0.854 | 0.960 | 0.944 | 0.084 | 0.000 |
| N-BaIoT | LwF | 0.224 ± 0.011 | 0.301 | 0.989 | 0.961 | 0.998 | 0.003 | 0.001 |
| N-BaIoT | A-GEM | 0.670 ± 0.204 | 0.706 | 0.402 | 0.974 | 0.997 | 0.005 | 0.000 |
| N-BaIoT | ER-500 | 0.798 ± 0.031 | 0.767 | 0.020 | 0.780 | 0.999 | 0.001 | 0.002 |
| N-BaIoT | ER-2000 | 0.851 ± 0.035 | 0.812 | 0.017 | 0.823 | 1.000 | 0.000 | 0.002 |
| N-BaIoT | Joint retraining | 0.987 ± 0.006 | 0.987 | 0.007 | 0.991 | 1.000 | 0.000 | 0.001 |
| N-BaIoT | Random forest (all data) | 1.000 ± 0.000 | 1.000 | | | 1.000 | | 0.000 |

### 5.2 Replay against regularisation

Replay reduced forgetting most. ER-2000 improved final macro-F1 over fine-tuning by +0.514 on TON-IoT (95% CI [+0.502, +0.525]), +0.330 on UNSW-NB15 ([+0.325, +0.336]) and +0.598 on N-BaIoT ([+0.552, +0.644]; Holm-adjusted p < 0.001, < 0.001 and < 0.001). That closes 91%, 85% and 81% of the gap between fine-tuning and joint retraining. With a quarter of the memory, ER-500 closed 88%, 62% and 74%.

The other strategies did much less:

- **Online EWC** changed final macro-F1 by -0.003 on TON-IoT (95% CI [-0.006, +0.000]), +0.042 on UNSW-NB15 ([-0.011, +0.095]) and +0.054 on N-BaIoT ([+0.010, +0.098]); none of these differences is significant after Holm correction (adjusted p 0.056, 0.19 and 0.053). With five seeds the test has little power, so the small gains on UNSW-NB15 and N-BaIoT are not shown to be zero, but they are far below those of replay.
- **A-GEM**, with the corrected memory constraint (Section 4.2), changed final macro-F1 by +0.173 on TON-IoT (95% CI [+0.114, +0.232]), +0.036 on UNSW-NB15 ([-0.023, +0.096]) and +0.417 on N-BaIoT ([+0.182, +0.652]; Holm-adjusted p 0.003, 0.19 and 0.024), closing 31%, 9% and 57% of the gap with the same memory as ER-2000. Used only to constrain the gradient direction, the memory recovered a sizeable part of what replay recovers on TON-IoT and N-BaIoT, but far less on UNSW-NB15; on N-BaIoT the seeds ranged from 0.319 to 0.809.
- **LwF** improved on fine-tuning on TON-IoT (+0.061, 95% CI [+0.042, +0.081], Holm-adjusted p 0.003) and UNSW-NB15 (+0.063, [+0.061, +0.066], Holm-adjusted p < 0.001), but this closed only 11% and 16% of the gap. On N-BaIoT it did not improve (-0.028, [-0.080, +0.023]). Extending the LwF grid down to 0.1 left the selected weight unchanged on all three datasets (4 on TON-IoT, 1 on UNSW-NB15, 0.5 on N-BaIoT). On N-BaIoT, where 0.5 had been the smallest value of the original grid, the lower weights reached lower validation macro-F1 on the development seed (0.226 at 0.25 and 0.214 at 0.1, against 0.266), so the lack of gain there is not an artefact of the grid. The EWC weights were chosen inside the grid on every dataset.

That regularisation fails in the class-incremental scenario matches the general literature (van de Ven et al., 2022) and the malware results of Rahman et al. (2022). Unlike in their class-incremental experiments, where plain experience replay also performed poorly and only the exemplar-based iCaRL approached joint retraining, plain replay worked well here.

Replay still trails joint retraining by 0.050, 0.060 and 0.136. The per-task retention shows where this gap lies.

| Dataset | Task | ER-2000 | Joint retraining |
|---|---|---|---|
| TON-IoT | 0 | 0.957 | 0.967 |
| TON-IoT | 1 | 0.955 | 0.937 |
| TON-IoT | 2 (newest) | 0.911 | 0.968 |
| UNSW-NB15 | 0 | 0.690 | 0.691 |
| UNSW-NB15 | 1 | 0.714 | 0.635 |
| UNSW-NB15 | 2 (newest) | 0.218 | 0.684 |
| N-BaIoT | 0 | 0.979 | 0.977 |
| N-BaIoT | 1 | 0.972 | 0.997 |
| N-BaIoT | 2 (newest) | 0.484 | 0.986 |

Table 3 lists the retention of each task after the last one. ER-2000 keeps the two earlier tasks at or near the joint level (on UNSW-NB15 Task 1 even above it, 0.714 against 0.635); the attack-class shortfall is on the task learned last. On TON-IoT part of the macro-F1 gap also comes from benign traffic: ER-2000 recognises 0.87 of the benign test records against 0.97 for joint retraining (Section 5.3). Two cases illustrate the newest-task shortfall:

- On UNSW-NB15 the last task holds four rare classes (Table 1). ER-2000 recognises backdoor with recall 0.01 against 0.65 for joint retraining.
- On N-BaIoT, ER-2000 confuses the two Mirai UDP floods (recall 0.34 and 0.63, against 0.99 and 0.99 for joint retraining).

**Is the shortfall a weighting effect?** ER-2000 up-weights the earlier classes twice (Section 4.2), which pushes it towards stability. With the memory added once to the training pool instead (ER-2000-pool, Section 4.3), retention of the newest task was 0.946 on TON-IoT, 0.679 on UNSW-NB15 and 0.829 on N-BaIoT (ER-2000: 0.911, 0.218 and 0.484), mean retention of the earlier tasks was 0.916, 0.641 and 0.852 (ER-2000: 0.956, 0.702 and 0.976), and final macro-F1 changed by +0.011, +0.037 and +0.001 (uncorrected p 0.048, < 0.001 and 0.94; marginal on TON-IoT). The newest-task shortfall shrank by more than half on all three datasets; at least one earlier task now fell as far below the joint level as the newest task, or further, on all three datasets (TON-IoT Task 0: 0.900 against 0.967; UNSW-NB15 Task 1: 0.611 against 0.635; N-BaIoT Task 1: 0.755 against 0.997), so the location of the shortfall depends partly on the double weighting. Final macro-F1 rose clearly only on UNSW-NB15, so the balance between stability and plasticity moved, but the gap to joint retraining did not close.

**Is the shortfall a budget effect?** Incremental tasks are trained with a smaller budget than joint retraining (Section 4.1), so we reran ER-2000 with the joint budget on every task (five seeds, same Task-0 models). Retention of the newest task rose on all three datasets (TON-IoT 0.911 → 0.924, UNSW-NB15 0.218 → 0.315, N-BaIoT 0.484 → 0.745), but mean retention of the earlier tasks fell on UNSW-NB15 (0.702 → 0.687) and N-BaIoT (0.976 → 0.903). Final macro-F1 did not change significantly (paired differences +0.009, +0.001 and +0.028; uncorrected p 0.12, 0.74 and 0.23). Longer training therefore moves replay along the stability–plasticity trade-off; it does not close the gap to joint retraining.

Raising the memory from 500 to 2,000 records clearly helped only on UNSW-NB15; final macro-F1 changed by +0.017, +0.087 and +0.053 (uncorrected paired p 0.29, < 0.001, 0.14; exploratory). Stored as 32-bit values, 2,000 records take 102 KiB on TON-IoT, 305 KiB on UNSW-NB15 and 898 KiB on N-BaIoT, whereas all training data of a sequence, which joint retraining keeps, take 6,525, 11,299 and 49,516 KiB.

**A tree model on the same memory.** A random forest retrained after each task on the same memory plus the new task's data (RF + memory, Section 4.3) reached a final macro-F1 of 0.939, 0.473 and 0.991, against 0.872, 0.461 and 0.851 for the CNN with ER-2000 (paired differences +0.067, +0.013 and +0.141; uncorrected p < 0.001, 0.023 and < 0.001), so it was better than the CNN with ER-2000 on all three datasets, but worse than the CNN with ER-2000-pool on UNSW-NB15. Against the CNN with ER-2000-pool (above; 0.883, 0.497 and 0.852) the paired differences were +0.056, -0.024 and +0.139 (uncorrected p < 0.001, < 0.001 and < 0.001). The forest on the memory was also above CNN joint retraining on TON-IoT and N-BaIoT (by +0.017 and +0.004). RF + memory retained 0.933, 0.526 and 0.991 of the earlier classes (ER-2000: 0.956, 0.702 and 0.976) and flagged 0.991, 0.702 and 1.000 of their attack records (ER-2000: 0.996, 0.998 and 1.000); on UNSW-NB15 it flagged only 0.499 of the Task-1 attacks. It ended at benign FPRs of 0.005, 0.015 and 0.000, so on UNSW-NB15 its low FPR comes with missed earlier attacks. One retraining sequence took 5, 7 and 16 s. Retrained on all data after every task, the forest's class-level forgetting was 0.015, 0.043 and 0.000.

### 5.3 Forgetting the class, keeping the alert

Detection-level retention tells a different story from class-level retention. Below, retention of the earlier tasks is averaged over Tasks 0 and 1 after the last task; unlike A in Table 2, it excludes the task just learned. Figure 1 follows the first task through the sequence. Under fine-tuning, EWC and LwF, recall of the Task-0 classes drops to near zero as soon as Task 1 is learned, while most Task-0 attack records are still flagged as attacks:

- **Fine-tuning** recognises 0.000, 0.000 and 0.037 of the earlier classes, yet still flags 0.875, 0.698 and 0.875 of their attack records as attacks. Most old attack records are relabelled as a newer attack family, not as benign traffic.
- **LwF** flags 0.948, 0.987 and 0.997 of the earlier attack records, although it recognises only 0.075, 0.109 and 0.004 of their classes.
- **A-GEM** recognises 0.277, 0.074 and 0.591 of the earlier classes and flags 0.892, 0.729 and 0.995 of their attack records.
- **Both replay variants** flag at least 0.990 of the earlier attack records on every dataset.

Detection-level retention has to be read together with the benign false-positive rate (FPR), because a model can keep alerting by flagging more of everything:

- **N-BaIoT:** the benign FPR stays at or below 0.003 on average for every strategy at every stage, so its high detection-level retention is genuine.
- **UNSW-NB15:** learning Task 1 raises the benign FPR to 0.182–0.196 for fine-tuning, EWC, LwF, both replay variants and joint retraining (0.191 for joint retraining), so the rise comes with the Task-1 classes rather than with any strategy; A-GEM reaches 0.184. Fine-tuning and A-GEM fall back to 0.039 and 0.048 after Task 2 only because they forget the Task-1 classes; EWC, LwF, replay and joint retraining stay near this level (0.189, 0.191, 0.205 and 0.203). The random forest ends at 0.081.

Replay has a cost of its own on TON-IoT. ER-500 and ER-2000 end at benign FPRs of 0.116 and 0.132, against 0.032 for joint retraining and 0.005 for the random forest. About one benign flow in eight is flagged. Two properties of the replay loss can contribute: the class-balanced memory gives benign traffic the same number of slots as each attack class, so benign traffic is under-represented relative to its share of the data, and the double weighting of Section 4.2 further favours the earlier attack classes. Reserving half of the memory for benign records (ER-2000b, Section 4.3) lowered the final benign FPR to 0.075 (paired difference -0.057, uncorrected p = 0.0032). That removes 57% of replay’s excess over joint retraining. Final macro-F1 rose slightly, by +0.013 (uncorrected p = 0.009), and detection-level retention was unchanged (0.996 against 0.997). With the class-balanced memory added once to the training pool instead (ER-2000-pool), the final benign FPR on TON-IoT was 0.030 (paired difference to ER-2000 -0.101, uncorrected p 0.0011), about the 0.032 of joint retraining, at the price of lower earlier-task retention (class level 0.916 against 0.956, detection level 0.980 against 0.996). Removing the per-minibatch oversampling alone thus brought the FPR to the joint level, and a half-benign memory under the original scheme removed 57% of the excess. Both act on the effective weight of benign traffic relative to the earlier attack classes and were not combined, so we do not apportion the cost between them. How the memory enters the loss and what it holds are both settings worth tuning.

### 5.4 Not-yet-learned attacks

Table 4 shows how often attack families that the model has not yet learned raise an alert. After Task 0, the model flags 0.738 of Task-1 and 0.779 of Task-2 attack records on TON-IoT, 0.523 and 0.895 on UNSW-NB15, and 0.461 and 0.430 on N-BaIoT. The benign FPR at that point is only 0.007, 0.029 and 0.003, so these rates are not produced by flagging everything. Because the same Task-0 model is shared, they are identical for all strategies. On N-BaIoT, a detector trained on BASHLITE misses more than half of the Mirai traffic.

After Task 1, flagging of the still-unseen Task-2 families is 0.865–0.954 on TON-IoT, 0.944–0.981 on UNSW-NB15 and 0.984–1.000 on N-BaIoT (range over strategies), higher than after Task 0 for every strategy. Once three Mirai classes are known, the two Mirai UDP floods are flagged almost completely.

How well a closed-set detector covers new families therefore depends on how closely they resemble the families it already knows. The rates of 0.430–0.461 on N-BaIoT suggest that families unlike anything learned would need an open-set or anomaly component, which we did not test.

| Dataset | Task 1 before learning it (after Task 0) | Task 2 after Task 0 | Task 2 after Task 1, range over strategies | Benign FPR after Task 0 | Benign FPR after Task 2, range over strategies |
|---|---|---|---|---|---|
| TON-IoT | 0.738 | 0.779 | 0.865–0.954 | 0.007 | 0.005–0.132 |
| UNSW-NB15 | 0.523 | 0.895 | 0.944–0.981 | 0.029 | 0.039–0.218 |
| N-BaIoT | 0.461 | 0.430 | 0.984–1.000 | 0.003 | 0.000–0.002 |

### 5.5 Task order and pre-processing scope

**Task order.** Over the four random task orders (Table 5), replay beat fine-tuning, online EWC and LwF in every order on every dataset (4/4, 4/4, 4/4). Replay also beat A-GEM in every order. The smallest margin between the weaker replay variant and the best of fine-tuning, EWC and LwF was 0.391, 0.098 and 0.492. Averaged over the random orders, ER-2000 closed 91%, 64% and 84% of the gap between fine-tuning and joint retraining, less than on the semantic order on UNSW-NB15.

Absolute levels depend on the order:

- On UNSW-NB15, fine-tuning averages 0.257 over the random orders but 0.130 on the semantic order.
- On UNSW-NB15, ER-500 and ER-2000 are indistinguishable over the random orders (0.423 and 0.424) but differ by +0.087 on the semantic order, which places the four rare classes last.

This suggests that placing the rare classes last hurts fine-tuning and the small memory, whereas ER-2000 scores higher on the semantic order than over the random ones (0.461 against 0.424). The benefit of a larger memory depends on the order. On the same dataset, joint retraining ranges from 0.500 to 0.532 over the random orders.

| Dataset | Fine-tuning | Online EWC | LwF | A-GEM | ER-500 | ER-2000 | Joint retraining |
|---|---|---|---|---|---|---|---|
| TON-IoT | 0.336 (0.291–0.382) | 0.339 (0.281–0.375) | 0.351 (0.293–0.436) | 0.542 (0.457–0.648) | 0.868 (0.828–0.899) | 0.881 (0.856–0.922) | 0.933 (0.922–0.939) |
| UNSW-NB15 | 0.257 (0.213–0.282) | 0.260 (0.238–0.311) | 0.285 (0.264–0.320) | 0.250 (0.198–0.268) | 0.423 (0.418–0.433) | 0.424 (0.413–0.439) | 0.518 (0.500–0.532) |
| N-BaIoT | 0.217 (0.185–0.237) | 0.232 (0.172–0.332) | 0.205 (0.179–0.226) | 0.344 (0.188–0.470) | 0.858 (0.836–0.900) | 0.864 (0.823–0.906) | 0.989 (0.984–0.992) |

**Pre-processing scope.** Fitting feature selection and scaling on all training data instead of Task 0 alone changed mean final macro-F1 by at most 0.029 for fine-tuning, ER-2000 and joint retraining on the three datasets (five seeds; Table 6). On TON-IoT and UNSW-NB15, every 95% confidence interval of the paired difference lies within ±0.025, so any effect of the shortcut there is small; the only change with an uncorrected p below 0.05 was a small loss for joint retraining on TON-IoT (-0.016, uncorrected p = 0.0084; Holm over the three TON-IoT tests, p = 0.025). N-BaIoT varies more across seeds: the largest mean change, -0.029 for fine-tuning, is not significant (uncorrected p = 0.25), but its interval [-0.089, +0.031] cannot exclude an effect of a few hundredths in either direction. Not-yet-learned flagging changed by at most 0.059, downwards on TON-IoT and N-BaIoT and upwards on UNSW-NB15. A lower rate under the all-data fit is the opposite of what leakage of the later families into the first model would be expected to produce; we did not investigate its cause. Under our pipeline, therefore, the shortcut did not inflate final results on TON-IoT and UNSW-NB15 by more than about 0.02, and the test is inconclusive on N-BaIoT.

How much the ablation changes the inputs bounds this conclusion. After min–max scaling, 0.0078% of TON-IoT, 0.0037% of UNSW-NB15 and 0.67% of N-BaIoT test values lie outside the Task-0 range [0, 1] (0.1%, 0.1% and 30.8% of the test records in at least one feature; for the training data of the later tasks, which is what the all-data fit would see, 0.02%, 0.0029% and 0.86% of the values). The compression map extrapolates these values smoothly, and only 0.002% of TON-IoT test values lie beyond the clipping limit of about 2,000 in min–max units. The all-data fit keeps 25 features on TON-IoT and 10 on UNSW-NB15, of which 13 and 10 are also among the 13 and 39 features selected on Task 0; on N-BaIoT only the min–max range changes. The shortcut may matter more with features whose range shifts strongly between attack families, or with a pipeline that does not extrapolate smoothly.

| Dataset | Strategy | Task-0 pre-processing | All-data pre-processing | Difference | Not-yet-learned flagging difference |
|---|---|---|---|---|---|
| TON-IoT | ER-2000 | 0.872 | 0.879 | +0.007 | -0.058 |
| TON-IoT | Joint retraining | 0.922 | 0.906 | -0.016 | -0.059 |
| TON-IoT | Fine-tuning | 0.358 | 0.359 | +0.001 | -0.058 |
| UNSW-NB15 | ER-2000 | 0.461 | 0.465 | +0.004 | +0.034 |
| UNSW-NB15 | Joint retraining | 0.520 | 0.519 | -0.001 | +0.026 |
| UNSW-NB15 | Fine-tuning | 0.130 | 0.130 | -0.001 | +0.032 |
| N-BaIoT | ER-2000 | 0.851 | 0.843 | -0.008 | -0.041 |
| N-BaIoT | Joint retraining | 0.987 | 0.987 | -0.000 | -0.036 |
| N-BaIoT | Fine-tuning | 0.253 | 0.224 | -0.029 | -0.041 |

### 5.6 Input scaling for a neural detector on flow features

In a first run of the present protocol (five seeds, benign traffic in every task), the network received min–max-scaled inputs without the compression map of Section 3.3 and was trained for at most 40 epochs on Task 0 and for joint retraining, and 20 epochs on later tasks (patience 5).

On TON-IoT, joint retraining reached 0.768 in that run against 0.922 now, and ER-2000 0.507 against 0.872, while fine-tuning moved only from 0.348 to 0.358, because it fails for a different reason (forgetting). The random forest scores 0.965. A tree is unaffected by monotone rescaling of its inputs, so up to the clipping of a few extreme values this reference applies to both runs, and it showed that the network was under-trained rather than the task being hard.

**The other datasets:**
- On UNSW-NB15, joint retraining rose from 0.467 to 0.520 and ER-2000 from 0.408 to 0.461.
- On N-BaIoT, whose features are window statistics rather than raw flow counts, joint retraining hardly changed (0.983 and 0.987), and ER-2000 was higher in the first run (0.879 against 0.851).

On TON-IoT and UNSW-NB15, the badly scaled inputs lowered the upper bound and replay much more than fine-tuning, which shrank the apparent benefit of replay.

**Which change mattered.** The correction changed two things at once, the input map and the training budget. Table 7 separates them for joint retraining on TON-IoT.

| Input map | Earlier budget | Present budget |
|---|---|---|
| Min–max only | 0.805 ± 0.027 | 0.871 ± 0.011 |
| Min–max + compression | 0.908 ± 0.010 | 0.922 ± 0.007 |

Averaged over the two budgets, the compression map contributed more than the longer budget (+0.077 against +0.040). The map raised final macro-F1 by +0.103 under the earlier budget and by +0.051 under the present one; the longer budget added +0.066 with min–max inputs but only +0.015 with the compression map, so under the present budget the two changes are of similar size. More training thus partly compensates for badly scaled inputs, but neither change alone reaches the combined result, which still stays below the random forest (0.965). These differences are descriptive, on one dataset and one strategy, and we did not measure the mechanism, for example through the batch-normalisation statistics.

The min–max, earlier-budget cell reproduces the first run only approximately: 0.805 against 0.768 (paired difference +0.037, uncorrected p = 0.3). The Task-0 confusion matrices are identical for every seed, so the difference arises in the later tasks. There that configuration is unstable: its final scores varied by seed with a standard deviation of 0.044. The present code also computes the loss differently, so exact reproduction of the later tasks is not expected.

## 6. Discussion

### 6.1 Implications for deployed detectors

**Keep a small replay memory.** It is the one strategy that clearly worked for the network. On these datasets, at most 2,000 stored records (102–898 KiB) recovered 81–91% of the difference between fine-tuning and full retraining on the semantic order and 64–91% over random orders. Regularisation did not help enough to justify its complexity in the class-incremental setting; only LwF gave a significant but small gain, on two datasets. Gradient projection with the same memory (A-GEM) improved on it significantly on TON-IoT and N-BaIoT, closing at most 57% of the gap, so where stored records are available, replaying them was the better use. How the memory enters the loss also matters: adding it once to the training pool instead of to every minibatch lowered the benign FPR on TON-IoT from 0.132 to 0.030 (Section 5.3). A random forest retrained on the same memory had a higher macro-F1 and a lower benign FPR than the CNN with ER-2000 on all three datasets; on UNSW-NB15 the CNN with ER-2000-pool had the higher macro-F1, and the forest flagged only 0.702 of the earlier attack records, against 0.999 for that network. Where a tree model fits the deployment, it is a strong alternative to a continually trained network and should be the reference any such network is compared with.

**Forgetting mostly misnames attacks rather than hiding them.** Even plain fine-tuning kept flagging most old attack records. This matters for deployments that act on the alert rather than on the attack class. It holds only when read alongside the benign FPR:

- on UNSW-NB15 every strategy that kept the Task-1 classes, joint retraining included, also kept a benign FPR near 0.2, and fine-tuning's lower FPR came only from forgetting;
- replay on TON-IoT raised the benign FPR to 0.116–0.132, a cost that depends on how the memory is used (added to the training pool instead of every minibatch: 0.030; a half-benign memory: 0.075).

Benign FPRs of about 0.2 on UNSW-NB15 for every network, joint retraining included (random forest 0.081), or above 0.1 for replay with minibatch mixing on TON-IoT (ER-2000-pool: 0.030) would be unusable in practice without a second filtering stage or a recalibrated threshold. We therefore recommend reporting class-level retention, detection-level retention and the benign FPR together.

**New families are often, but not reliably, flagged before they are learned.** Coverage depended on how closely the new families resembled the known ones, from under half of Mirai traffic for a BASHLITE-trained model to almost all of it once other Mirai variants were known. A closed-set detector would likely need to be paired with an open-set or anomaly component.

**Joint retraining remained the best option for the network.** At these data sizes a full retraining sequence took about five to ten minutes of training on a laptop; learning Tasks 1 and 2 took a median of 272, 425 and 355 s with joint retraining against 246, 261 and 140 s with ER-2000 (indicative only; Section 6.2). The case for continual learning in this setting therefore rests on data retention limits more than on compute: all training data of a sequence take 37–64 times the space of a 2,000-record memory. A replay memory stores real past records, so it weakens that case only in proportion to its size, which is small. The network itself has about 24,000 parameters (under 100 KiB at 32 bits). The forests are far larger: the final RF + memory model has 215,144, 348,392 and 26,370 tree nodes (30,332, 49,070 and 3,575 KiB serialised, 38–525 times the network) and the all-data forest 66,609, 203,780 and 7,157 KiB. On one CPU thread, predicting 10,000 records took 179, 214 and 164 ms with RF + memory against 145, 272 and 621 ms with the network (seed 0, on a laptop running other work, so indicative only). Where a detector must fit in a few hundred kilobytes, the network is the option that does; otherwise the forest is the stronger detector on two of the three datasets. We did not measure peak memory or energy on an edge device.

**Use a non-neural reference.** A tree model on the same features is a cheap check that a network is not failing quietly. In this study it showed that the network was under-trained on min–max-scaled inputs in a first run of the protocol (Section 5.6).

### 6.2 Limitations

**Data and setting:**
- All three datasets come from testbeds, and UNSW-NB15 is not IoT traffic.
- Each sequence has three tasks, and the benign traffic is divided artificially into thirds.
- The splits are random at record level. N-BaIoT features are sliding-window statistics, so near-duplicate records across the split remain likely after exact de-duplication, and absolute N-BaIoT scores are probably optimistic.
- We study class-incremental learning only, not drift of benign traffic or of known attacks over time.
- One compact CNN is evaluated. Its kernel-3 convolution assumes that neighbouring features are related, and we did not compare it with a multilayer perceptron of similar size.
- Training times are indicative only: they were measured on a laptop that was also running other work. We report no inference latency, peak memory or energy.

**Tuning and statistics:**
- Regularisation strengths were tuned on a single development seed, using validation data from all tasks after the last one. A deployed learner would not have these in advance, so the EWC and LwF results are, if anything, optimistic. ER and A-GEM were not tuned.
- Early stopping uses the current task's validation records only, which favours plasticity over retention; a replay strategy could use its memory for validation.
- The five seeds vary training randomness only; the data split and the benign shares are fixed. With five seeds the paired tests have little power, so non-significant differences are not evidence of equivalence.
- The compression map and its constant were fixed after the under-training on TON-IoT was observed, not tuned further; the factorial of Table 7 is scored on the test set and is diagnostic, not a selection.

**Unresolved questions:**
- The pre-processing ablation covers three strategies, not EWC, LwF or A-GEM, and is inconclusive on N-BaIoT.
- The replay loss up-weights the earlier classes twice; the ER-2000-pool check tests one alternative, not the full range of weighting and sampling schemes, and it was not combined with the half-benign memory.
- Detection-level retention is measured at the argmax operating point. A threshold-free measure, such as attack-versus-benign AUROC or recall at a fixed benign FPR, would compare strategies at equal false-alarm rates.
- Per-class recalls of the rare UNSW-NB15 classes rest on few test records (Table 1).
- Stronger class-incremental methods (nearest-mean classification as in iCaRL, bias correction, dark experience replay) and IDS-specific replay schemes were not evaluated. Plain replay already recovers most of the gap, and the replay-weighting check shows that the remaining shortfall moves with the replay scheme, so such methods would compete with simple tuning of how the memory is used.
- An incremental tree ensemble was not evaluated; the random forest retrained on the replay memory is the closest baseline here.
- The benign-share memory and the input-map × budget factorial were run on TON-IoT only, and the factorial on joint retraining only.
- The benign share of the memory was fixed at one half rather than tuned.

## 7. Conclusion

We evaluated class-incremental learning for a compact IoT intrusion detector on TON-IoT, UNSW-NB15 and N-BaIoT under a strict protocol: benign traffic in every task, feature selection and scaling fitted on the first task only, a joint-retraining bound, random-forest baselines, and five seeds on a semantic task order plus four random orders.

Experience replay with 2,000 stored records recovered 81–91% of the difference between fine-tuning and retraining on the semantic order and 64–91% over random orders, ahead of every non-replay strategy in every order. Online EWC did not significantly improve on fine-tuning, LwF helped only a little, and A-GEM improved on it significantly on TON-IoT and N-BaIoT, closing at most 57% of the gap. On the semantic order, replay's remaining gap to retraining lay mainly in learning the newest family and, on TON-IoT, in benign false positives, rather than in forgetting the older ones. Both depend partly on the double weighting of the earlier classes in the replay loss: with the memory added once to the training pool, the newest-task shortfall shrank by more than half on all three datasets, the benign FPR on TON-IoT fell from 0.132 to 0.030, and part of the gap moved to the earlier tasks; final macro-F1 rose clearly only on UNSW-NB15.

Forgotten classes were mostly still flagged as attacks, and replay raised the benign false-positive rate on TON-IoT; reserving half of the memory for benign records removed part of that cost, and adding the memory to the training pool instead of every minibatch brought it to the joint-retraining level. Unseen families were flagged at rates from under half to nearly all, depending on their similarity to known ones. Fitting the pre-processing on future data changed results only marginally on TON-IoT and UNSW-NB15 and inconclusively on N-BaIoT.

A random forest retrained on the same memory better than the CNN with ER-2000 on all three datasets, but worse than the CNN with ER-2000-pool on UNSW-NB15; it is far larger than the network, but it is the reference a continually trained network should be compared with. Finally, a random-forest reference showed that the network was under-trained on min–max-scaled, heavy-tailed flow features; a fixed compression map together with a longer budget closed most of that gap. Such a reference should be standard in IDS evaluations.

## Declarations

**Data availability.** TON-IoT, UNSW-NB15 and N-BaIoT are publicly available from their providers. The code, task definitions and all result files are available at https://github.com/Amoakoemma123/iot-ids-pipeline. The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors.

**CRediT authorship contribution statement.** Emmanuel Kwame Amoako: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft, Writing – review & editing. Mahamudu Amadu: Conceptualization, Methodology, Validation. James Quayson: Conceptualization, Methodology, Writing – review & editing. Kingsford Donkor: Validation, Writing – review & editing. Daniel Asamoah Apraku: Validation, Writing – review & editing.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the authors used Claude (Anthropic) to write and edit the manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. The released scripts compute every reported number from the result files, and an automated check compares each quantitative statement in the text with those files.

## References

A H, N. F., Khraisat, A., S P, S. I., & Li, G. (2025). Adaptive memory replay for network intrusion detection: Tackling data drift and catastrophic forgetting. *Computer Networks, 272*, Article 111712. https://doi.org/10.1016/j.comnet.2025.111712

Alsaedi, A., Moustafa, N., Tari, Z., Mahmood, A., & Anwar, A. (2020). TON_IoT telemetry dataset: A new generation dataset of IoT and IIoT for data-driven intrusion detection systems. *IEEE Access, 8*, 165130–165150. https://doi.org/10.1109/ACCESS.2020.3022862

Amalapuram, S. K., Channappayya, S. S., & Tamma, B. R. (2023). Augmented memory replay-based continual learning approaches for network intrusion detection. In *Advances in Neural Information Processing Systems 36* (pp. 17156–17169). Curran Associates. https://doi.org/10.52202/075280-0750

Amalapuram, S. K., Reddy, T. T., Channappayya, S. S., & Tamma, B. R. (2021). On handling class imbalance in continual learning based network intrusion detection systems. In *Proceedings of the First International Conference on AI-ML Systems (AIMLSystems '21)* (pp. 1–7). ACM. https://doi.org/10.1145/3486001.3486231

Amalapuram, S. K., Tadwai, A., Vinta, R., Channappayya, S. S., & Tamma, B. R. (2022). Continual learning for anomaly based network intrusion detection. In *2022 14th International Conference on COMmunication Systems & NETworkS (COMSNETS)* (pp. 497–505). IEEE. https://doi.org/10.1109/COMSNETS53615.2022.9668482

Amalapuram, S. K., Tamma, B. R., & Channappayya, S. S. (2024). SPIDER: A semi-supervised continual learning-based network intrusion detection system. In *IEEE INFOCOM 2024 – IEEE Conference on Computer Communications* (pp. 571–580). IEEE. https://doi.org/10.1109/INFOCOM52122.2024.10621428

Amalapuram, S. K., Tamma, B. R., & Channappayya, S. S. (2025). Experimental insights on continual learning for network intrusion detection. In *2025 IEEE International Conference on Data Mining Workshops (ICDMW)* (pp. 1551–1557). IEEE. https://doi.org/10.1109/ICDMW69685.2025.00186

Andresini, G., Pendlebury, F., Pierazzi, F., Loglisci, C., Appice, A., & Cavallaro, L. (2021). INSOMNIA: Towards concept-drift robustness in network intrusion detection. In *Proceedings of the 14th ACM Workshop on Artificial Intelligence and Security (AISec '21)* (pp. 111–122). ACM. https://doi.org/10.1145/3474369.3486864

Barbero, F., Pendlebury, F., Pierazzi, F., & Cavallaro, L. (2022). Transcending TRANSCEND: Revisiting malware classification in the presence of concept drift. In *2022 IEEE Symposium on Security and Privacy (SP)* (pp. 805–823). IEEE. https://doi.org/10.1109/SP46214.2022.9833659

Chaudhry, A., Ranzato, M., Rohrbach, M., & Elhoseiny, M. (2019a). Efficient lifelong learning with A-GEM. In *International Conference on Learning Representations (ICLR 2019)*. arXiv:1812.00420.

Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019b). *On tiny episodic memories in continual learning* (arXiv:1902.10486). arXiv. https://doi.org/10.48550/arXiv.1902.10486

De Lange, M., Aljundi, R., Masana, M., Parisot, S., Jia, X., Leonardis, A., Slabaugh, G., & Tuytelaars, T. (2022). A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(7), 3366–3385. https://doi.org/10.1109/TPAMI.2021.3057446

Faber, K., Sniezynski, B., & Corizzo, R. (2023). Distributed continual intrusion detection: A collaborative replay framework. In *2023 IEEE International Conference on Big Data (BigData)* (pp. 3255–3263). IEEE. https://doi.org/10.1109/BigData59044.2023.10386211

Farquhar, S., & Gal, Y. (2018). *Towards robust evaluations of continual learning* (arXiv:1805.09733). arXiv. https://doi.org/10.48550/arXiv.1805.09733

Guo, C., Li, X., Cheng, J., Yang, S., & Gong, H. (2025). Continual learning for intrusion detection under evolving network threats. *Future Internet, 17*(10), Article 456. https://doi.org/10.3390/fi17100456

Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning, 46*(1–3), 389–422. https://doi.org/10.1023/A:1012487302797

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (Vol. 37, pp. 448–456). PMLR. https://proceedings.mlr.press/v37/ioffe15.html

Jordaney, R., Sharad, K., Dash, S. K., Wang, Z., Papini, D., Nouretdinov, I., & Cavallaro, L. (2017). Transcend: Detecting concept drift in malware classification models. In *26th USENIX Security Symposium* (pp. 625–642). USENIX Association.

Kan, Z., Pendlebury, F., Pierazzi, F., & Cavallaro, L. (2021). Investigating labelless drift adaptation for malware detection. In *Proceedings of the 14th ACM Workshop on Artificial Intelligence and Security (AISec '21)* (pp. 123–134). ACM. https://doi.org/10.1145/3474369.3486873

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR 2015)*. arXiv:1412.6980.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

Li, Z., & Hoiem, D. (2018). Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 40*(12), 2935–2947. https://doi.org/10.1109/TPAMI.2017.2773081

Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. In *Advances in Neural Information Processing Systems 30* (pp. 6467–6476). Curran Associates.

Mai, Z., Li, R., Jeong, J., Quispe, D., Kim, H., & Sanner, S. (2022). Online continual learning in image classification: An empirical survey. *Neurocomputing, 469*, 28–51. https://doi.org/10.1016/j.neucom.2021.10.021

Mao, J., Wei, Z., Li, B., Zhang, R., & Song, L. (2024). Toward ever-evolution network threats: A hierarchical federated class-incremental learning approach for network intrusion detection in IIoT. *IEEE Internet of Things Journal, 11*(18), 29864–29877. https://doi.org/10.1109/JIOT.2024.3408634

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation, 24*, 109–165. https://doi.org/10.1016/S0079-7421(08)60536-8

Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Shabtai, A., Breitenbacher, D., & Elovici, Y. (2018). N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders. *IEEE Pervasive Computing, 17*(3), 12–22. https://doi.org/10.1109/MPRV.2018.03367731

Mirsky, Y., Doitshman, T., Elovici, Y., & Shabtai, A. (2018). Kitsune: An ensemble of autoencoders for online network intrusion detection. In *Network and Distributed System Security Symposium (NDSS 2018)*. https://doi.org/10.14722/ndss.2018.23204

Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society, 72*, Article 102994. https://doi.org/10.1016/j.scs.2021.102994

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set). In *2015 Military Communications and Information Systems Conference (MilCIS)* (pp. 1–6). IEEE. https://doi.org/10.1109/MilCIS.2015.7348942

Oikonomou, C., Iliopoulos, I., Ioannidis, D., & Tzovaras, D. (2023). A multi-class intrusion detection system based on continual learning. In *2023 IEEE International Conference on Cyber Security and Resilience (CSR)* (pp. 86–91). IEEE. https://doi.org/10.1109/CSR57506.2023.10224974

Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., & Wermter, S. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks, 113*, 54–71. https://doi.org/10.1016/j.neunet.2019.01.012

Pendlebury, F., Pierazzi, F., Jordaney, R., Kinder, J., & Cavallaro, L. (2019). TESSERACT: Eliminating experimental bias in malware classification across space and time. In *28th USENIX Security Symposium* (pp. 729–746). USENIX Association.

Prasath, S., Sethi, K., Mohanty, D., Bera, P., & Samantaray, S. R. (2022). Analysis of continual learning models for intrusion detection system. *IEEE Access, 10*, 121444–121464. https://doi.org/10.1109/ACCESS.2022.3222715

Rahman, M. S., Coull, S., & Wright, M. (2022). On the limitations of continual learning for malware classification. In *Proceedings of the 1st Conference on Lifelong Learning Agents* (Vol. 199, pp. 564–582). PMLR. https://proceedings.mlr.press/v199/rahman22a.html

Rebuffi, S.-A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (2017). iCaRL: Incremental classifier and representation learning. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5533–5542). IEEE. https://doi.org/10.1109/CVPR.2017.587

Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318

Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y. W., Pascanu, R., & Hadsell, R. (2018). Progress & compress: A scalable framework for continual learning. In *Proceedings of the 35th International Conference on Machine Learning* (Vol. 80, pp. 4528–4537). PMLR.

Shyaa, M. A., Ibrahim, N. F., Zainol, Z., Abdullah, R., Anbar, M., & Alzubaidi, L. (2024). Evolving cybersecurity frontiers: A comprehensive survey on concept drift and feature dynamics aware machine and deep learning in intrusion detection systems. *Engineering Applications of Artificial Intelligence, 137*, Article 109143. https://doi.org/10.1016/j.engappai.2024.109143

Talpini, J., Sartori, F., & Savi, M. (2024). Hierarchical multiclass continual learning for network intrusion detection. In *2024 IEEE 10th International Conference on Network Softwarization (NetSoft)* (pp. 263–267). IEEE. https://doi.org/10.1109/NetSoft60951.2024.10588909

van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2022). Three types of incremental learning. *Nature Machine Intelligence, 4*(12), 1185–1197. https://doi.org/10.1038/s42256-022-00568-3

Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 46*(8), 5362–5383. https://doi.org/10.1109/TPAMI.2024.3367329

Yang, L., Guo, W., Hao, Q., Ciptadi, A., Ahmadzadeh, A., Xing, X., & Wang, G. (2021). CADE: Detecting and explaining concept drift samples for security applications. In *30th USENIX Security Symposium* (pp. 2327–2344). USENIX Association.

Zhang, X., Zhao, R., Jiang, Z., Chen, H., Ding, Y., Ngai, E. C. H., & Yang, S.-H. (2025). Continual learning with strategic selection and forgetting for network intrusion detection. In *IEEE INFOCOM 2025 – IEEE Conference on Computer Communications* (pp. 1–10). IEEE. https://doi.org/10.1109/INFOCOM55648.2025.11044615
