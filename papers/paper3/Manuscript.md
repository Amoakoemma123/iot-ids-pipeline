# Paper 3 v2 manuscript (double-brace placeholders are filled from values.json by build.py)

## Title

**Learning New Attack Families After Deployment: A Controlled Evaluation of Class-Incremental Learning for Compact IoT Intrusion Detectors**

## Abstract

New attack families appear after an intrusion detection system (IDS) is deployed, and retraining from scratch requires keeping all past data. We evaluate class-incremental learning for a compact 1D-CNN detector on TON-IoT, UNSW-NB15 and N-BaIoT, comparing fine-tuning, online EWC, learning without forgetting (LwF), A-GEM and experience replay with 500 or 2,000 stored records against joint retraining and random-forest baselines. The protocol keeps benign traffic in every task, fits feature selection and scaling on the first task only, and uses five seeds on a semantic task order plus four random orders. Replay with 2,000 records recovered {{NB_er2000_gap_closed}}–{{TON_er2000_gap_closed}}% of the macro-F1 difference between fine-tuning and retraining on the semantic order ({{UNSW_er2000_orders_gap_closed}}–{{TON_er2000_orders_gap_closed}}% over random orders), ahead of {{P3F_orders_nonreplay}} in every order. Online EWC did not significantly improve on fine-tuning, LwF helped little, and A-GEM {{P3F_agem_verdict_short}}. A random forest retrained on the same memory was {{P3F_rfmem_verdict}}. At the alert level forgetting was milder: fine-tuning still flagged {{UNSW_naive_old_det}}–{{TON_naive_old_det}} of the attack records of earlier tasks. Replay raised the benign false-positive rate on TON-IoT to {{TON_er2000_fpr_final}}; a half-benign memory lowered it to {{TON_er2000b_fpr_final}}, and adding the memory to each task's training pool instead of every minibatch to {{P3F_TON_er2000p_fpr_final}}. Fitting the pre-processing on future data changed mean final macro-F1 by at most {{oracle_max_abs}} (inconclusive on N-BaIoT).

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

{{table:tasks}}

### 3.3 Pre-processing fitted on the first task

Dataset-level cleaning uses the whole training split under both protocols below: numeric columns with more than 40% missing values are dropped, missing values are imputed with the median, infinite values are replaced by the training 99th percentile, zero-variance columns are dropped, and the one-hot schema is fixed. Feature selection and scaling, by contrast, are fitted on the Task-0 training records only, because the later attack families do not exist when the detector is first built. For TON-IoT and UNSW-NB15, a mutual-information filter (threshold 0.05) is followed by recursive feature elimination with cross-validation (Guyon et al., 2002), using a 50-tree random forest, three-fold stratified cross-validation scored by macro-F1, at least 10 features and a subsample of at most 50,000 records; it keeps {{TON_n_features}} features on TON-IoT and {{UNSW_n_features}} on UNSW-NB15. N-BaIoT uses all {{NB_n_features}} features without selection. Min–max scaling is then fitted on the Task-0 records. After scaling, every feature is compressed by the fixed monotone map x ↦ asinh(1000x)/asinh(1000), which leaves 0 and 1 unchanged, has no fitted parameters and extrapolates smoothly to later-task values outside the Task-0 range; Section 5.6 shows why the network needs it. Finally, the compressed values are clipped to [−1, 2], which corresponds to about −1 to 2,000 in min–max units. The **oracle** ablation fits the feature selection and scaling on all training records instead (fine-tuning, ER-2000 and joint retraining, five seeds, all three datasets); on N-BaIoT it changes only the min–max range.

## 4. Models and Strategies

### 4.1 Network

All strategies use a compact 1D-CNN: two convolutional blocks (32 and 64 filters, kernel 3, batch normalisation (Ioffe & Szegedy, 2015) with momentum 0.9, max-pooling), global average pooling, dense layers of 128 and 64 units with dropout 0.3 and a single output layer over all classes, {{TON_params}} parameters on TON-IoT and UNSW-NB15 and {{NB_params}} on N-BaIoT. A kernel-3 convolution over a tabular feature vector treats neighbouring features as related, an assumption we did not test (Section 6.2). Training and prediction are restricted to the classes seen so far (logits of unseen classes are masked). The loss is class-weighted cross-entropy on the training pool of the current stage (weights inversely proportional to class frequency in that pool, capped at 50). The network is trained with Adam (Kingma & Ba, 2015): on Task 0 and for joint retraining with learning rate 10⁻³ for at most 100 epochs (patience 15), on later tasks with 5 × 10⁻⁴ for at most 50 epochs (patience 10), batch size 512. Early stopping monitors macro-F1 on the validation records of the current task (its benign share and its new attack classes); the incremental strategies do not keep validation data of earlier tasks, whereas joint retraining uses the validation records of all tasks so far. The best weights are always restored. Task 0 is trained once per seed, and the same weights are reused by every strategy.

### 4.2 Strategies

- **Fine-tuning** (lower bound): train on each new task's data only.
- **Joint retraining** (upper bound): retrain a fresh network on all training data seen so far.
- **Online EWC** (Kirkpatrick et al., 2017; Schwarz et al., 2018): quadratic penalty on parameter changes weighted by a per-sample empirical Fisher information, estimated on 1,000 training records of the task just learned with their true labels and summed over tasks without decay (γ = 1); the penalty is anchored at the most recent parameters.
- **LwF** (Li & Hoiem, 2018): distillation of the previous model's outputs, computed on the current task's data, over all previously seen classes, benign included (temperature 2).
- **Experience replay** (ER; Chaudhry et al., 2019b): a class-balanced episodic memory of at most 500 or 2,000 training records (equal slots per seen class; at the end {{UNSW_er2000_mem_final}} records on UNSW-NB15, where some rare classes have fewer records than their share). Each minibatch of 512 current-task records is paired with 512 memory records drawn with replacement, so the memory supplies half of every gradient step, and the class weights of the loss are computed over the current task and the memory together. Each memory class holds only a few hundred records and therefore receives a high weight, so earlier classes are up-weighted twice, by oversampling and by the loss weights. The replay-weighting check of Section 4.3 removes the oversampling and keeps the loss weights.
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

Table 2 summarises the semantic task order over five seeds. For a given seed, all seven strategies start from the same Task-0 model, and the stored Task-0 predictions are identical across strategies on all three datasets. That model reaches a macro-F1 of {{TON_task0_f1}} on TON-IoT, {{UNSW_task0_f1}} on UNSW-NB15 and {{NB_task0_f1}} on N-BaIoT over benign traffic and the Task-0 attack classes.

After the last task, joint retraining reaches a final macro-F1 of {{TON_joint_f1}}, {{UNSW_joint_f1}} and {{NB_joint_f1}}. The random forest retrained on all data reaches {{TON_rf_f1}}, {{UNSW_rf_f1}} and {{NB_rf_f1}}, so the compact network's upper bound lies {{TON_joint_rf_gap}}, {{UNSW_joint_rf_gap}} and {{NB_joint_rf_gap}} below the tree reference. The low UNSW-NB15 values reflect that dataset's small, overlapping attack classes, which limit every model.

Fine-tuning ends at {{TON_naive_f1}}, {{UNSW_naive_f1}} and {{NB_naive_f1}}, with class-level forgetting of {{TON_naive_FM}}, {{UNSW_naive_FM}} and {{NB_naive_FM}}. After the last task it recognises none of the Task-0 attack classes on any dataset (retention {{TON_naive_Rlast_t0}}, {{UNSW_naive_Rlast_t0}}, {{NB_naive_Rlast_t0}}). Joint retraining also shows class-level forgetting on UNSW-NB15 (FM {{UNSW_joint_FM}}): even a model retrained from scratch recognises earlier classes less well once overlapping new classes are added.

{{table:main}}

### 5.2 Replay against regularisation

Replay reduced forgetting most. ER-2000 improved final macro-F1 over fine-tuning by {{TON_er2000_vs_naive_d}} on TON-IoT (95% CI {{TON_er2000_vs_naive_ci}}), {{UNSW_er2000_vs_naive_d}} on UNSW-NB15 ({{UNSW_er2000_vs_naive_ci}}) and {{NB_er2000_vs_naive_d}} on N-BaIoT ({{NB_er2000_vs_naive_ci}}; Holm-adjusted p {{TON_er2000_vs_naive_pholm}}, {{UNSW_er2000_vs_naive_pholm}} and {{NB_er2000_vs_naive_pholm}}). That closes {{TON_er2000_gap_closed}}%, {{UNSW_er2000_gap_closed}}% and {{NB_er2000_gap_closed}}% of the gap between fine-tuning and joint retraining. With a quarter of the memory, ER-500 closed {{TON_er500_gap_closed}}%, {{UNSW_er500_gap_closed}}% and {{NB_er500_gap_closed}}%.

The other strategies did much less:

- **Online EWC** changed final macro-F1 by {{TON_ewc_vs_naive_d}} on TON-IoT (95% CI {{TON_ewc_vs_naive_ci}}), {{UNSW_ewc_vs_naive_d}} on UNSW-NB15 ({{UNSW_ewc_vs_naive_ci}}) and {{NB_ewc_vs_naive_d}} on N-BaIoT ({{NB_ewc_vs_naive_ci}}); none of these differences is significant after Holm correction (adjusted p {{TON_ewc_vs_naive_pholm}}, {{UNSW_ewc_vs_naive_pholm}} and {{NB_ewc_vs_naive_pholm}}). With five seeds the test has little power, so the small gains on UNSW-NB15 and N-BaIoT are not shown to be zero, but they are far below those of replay.
- **A-GEM**, with the corrected memory constraint (Section 4.2), changed final macro-F1 by {{P3F_TON_agem_vs_naive_d}} on TON-IoT (95% CI {{P3F_TON_agem_vs_naive_ci}}), {{P3F_UNSW_agem_vs_naive_d}} on UNSW-NB15 ({{P3F_UNSW_agem_vs_naive_ci}}) and {{P3F_NB_agem_vs_naive_d}} on N-BaIoT ({{P3F_NB_agem_vs_naive_ci}}; Holm-adjusted p {{P3F_TON_agem_vs_naive_pholm}}, {{P3F_UNSW_agem_vs_naive_pholm}} and {{P3F_NB_agem_vs_naive_pholm}}), closing {{P3F_TON_agem_gap_closed}}%, {{P3F_UNSW_agem_gap_closed}}% and {{P3F_NB_agem_gap_closed}}% of the gap with the same memory as ER-2000. {{P3F_agem_mechanism}}
- **LwF** improved on fine-tuning on TON-IoT ({{TON_lwf_vs_naive_d}}, 95% CI {{TON_lwf_vs_naive_ci}}, Holm-adjusted p {{TON_lwf_vs_naive_pholm}}) and UNSW-NB15 ({{UNSW_lwf_vs_naive_d}}, {{UNSW_lwf_vs_naive_ci}}, Holm-adjusted p {{UNSW_lwf_vs_naive_pholm}}), but this closed only {{TON_lwf_gap_closed}}% and {{UNSW_lwf_gap_closed}}% of the gap. On N-BaIoT it did not improve ({{NB_lwf_vs_naive_d}}, {{NB_lwf_vs_naive_ci}}). Extending the LwF grid down to 0.1 left the selected weight unchanged on all three datasets ({{TON_lwf_alpha}} on TON-IoT, {{UNSW_lwf_alpha}} on UNSW-NB15, {{NB_lwf_alpha}} on N-BaIoT). On N-BaIoT, where {{NB_lwf_alpha}} had been the smallest value of the original grid, the lower weights reached lower validation macro-F1 on the development seed ({{NB_lwf_tune_0p25}} at 0.25 and {{NB_lwf_tune_0p1}} at 0.1, against {{NB_lwf_tune_0p5}}), so the lack of gain there is not an artefact of the grid. The EWC weights were chosen inside the grid on every dataset.

That regularisation fails in the class-incremental scenario matches the general literature (van de Ven et al., 2022) and the malware results of Rahman et al. (2022). Unlike in their class-incremental experiments, where plain experience replay also performed poorly and only the exemplar-based iCaRL approached joint retraining, plain replay worked well here.

Replay still trails joint retraining by {{TON_er2000_joint_gap}}, {{UNSW_er2000_joint_gap}} and {{NB_er2000_joint_gap}}. The per-task retention shows where this gap lies.

| Dataset | Task | ER-2000 | Joint retraining |
|---|---|---|---|
| TON-IoT | 0 | {{TON_er2000_Rlast_t0}} | {{TON_joint_Rlast_t0}} |
| TON-IoT | 1 | {{TON_er2000_Rlast_t1}} | {{TON_joint_Rlast_t1}} |
| TON-IoT | 2 (newest) | {{TON_er2000_Rlast_t2}} | {{TON_joint_Rlast_t2}} |
| UNSW-NB15 | 0 | {{UNSW_er2000_Rlast_t0}} | {{UNSW_joint_Rlast_t0}} |
| UNSW-NB15 | 1 | {{UNSW_er2000_Rlast_t1}} | {{UNSW_joint_Rlast_t1}} |
| UNSW-NB15 | 2 (newest) | {{UNSW_er2000_Rlast_t2}} | {{UNSW_joint_Rlast_t2}} |
| N-BaIoT | 0 | {{NB_er2000_Rlast_t0}} | {{NB_joint_Rlast_t0}} |
| N-BaIoT | 1 | {{NB_er2000_Rlast_t1}} | {{NB_joint_Rlast_t1}} |
| N-BaIoT | 2 (newest) | {{NB_er2000_Rlast_t2}} | {{NB_joint_Rlast_t2}} |

Table 3 lists the retention of each task after the last one. ER-2000 keeps the two earlier tasks at or near the joint level (on UNSW-NB15 Task 1 even above it, {{UNSW_er2000_Rlast_t1}} against {{UNSW_joint_Rlast_t1}}); the attack-class shortfall is on the task learned last. On TON-IoT part of the macro-F1 gap also comes from benign traffic: ER-2000 recognises {{TON_er2000_rec_normal}} of the benign test records against {{TON_joint_rec_normal}} for joint retraining (Section 5.3). Two cases illustrate the newest-task shortfall:

- On UNSW-NB15 the last task holds four rare classes (Table 1). ER-2000 recognises backdoor with recall {{UNSW_er2000_rec_backdoor}} against {{UNSW_joint_rec_backdoor}} for joint retraining.
- On N-BaIoT, ER-2000 confuses the two Mirai UDP floods (recall {{NB_er2000_rec_mirai_udp}} and {{NB_er2000_rec_mirai_udpplain}}, against {{NB_joint_rec_mirai_udp}} and {{NB_joint_rec_mirai_udpplain}} for joint retraining).

**Is the shortfall a weighting effect?** ER-2000 up-weights the earlier classes twice (Section 4.2), which pushes it towards stability. With the memory added once to the training pool instead (ER-2000-pool, Section 4.3), retention of the newest task was {{P3F_TON_er2000p_Rlast_t2}} on TON-IoT, {{P3F_UNSW_er2000p_Rlast_t2}} on UNSW-NB15 and {{P3F_NB_er2000p_Rlast_t2}} on N-BaIoT (ER-2000: {{TON_er2000_Rlast_t2}}, {{UNSW_er2000_Rlast_t2}} and {{NB_er2000_Rlast_t2}}), mean retention of the earlier tasks was {{P3F_TON_er2000p_old_cls}}, {{P3F_UNSW_er2000p_old_cls}} and {{P3F_NB_er2000p_old_cls}} (ER-2000: {{TON_er2000_old_cls}}, {{UNSW_er2000_old_cls}} and {{NB_er2000_old_cls}}), and final macro-F1 changed by {{P3F_TON_er2000p_d}}, {{P3F_UNSW_er2000p_d}} and {{P3F_NB_er2000p_d}} (uncorrected p {{P3F_TON_er2000p_p}}, {{P3F_UNSW_er2000p_p}} and {{P3F_NB_er2000p_p}}; marginal on TON-IoT). {{P3F_er2000p_shortfall_verdict}} Final macro-F1 rose clearly only on UNSW-NB15, so the balance between stability and plasticity moved, but the gap to joint retraining did not close.

**Is the shortfall a budget effect?** Incremental tasks are trained with a smaller budget than joint retraining (Section 4.1), so we reran ER-2000 with the joint budget on every task (five seeds, same Task-0 models). Retention of the newest task rose on all three datasets (TON-IoT {{TON_er2000_Rlast_t2}} → {{TON_er2000fb_Rlast_t2}}, UNSW-NB15 {{UNSW_er2000_Rlast_t2}} → {{UNSW_er2000fb_Rlast_t2}}, N-BaIoT {{NB_er2000_Rlast_t2}} → {{NB_er2000fb_Rlast_t2}}), but mean retention of the earlier tasks fell on UNSW-NB15 ({{UNSW_er2000_old_cls}} → {{UNSW_er2000fb_old_cls}}) and N-BaIoT ({{NB_er2000_old_cls}} → {{NB_er2000fb_old_cls}}). Final macro-F1 did not change significantly (paired differences {{TON_er2000fb_d}}, {{UNSW_er2000fb_d}} and {{NB_er2000fb_d}}; uncorrected p {{TON_er2000fb_p}}, {{UNSW_er2000fb_p}} and {{NB_er2000fb_p}}). Longer training therefore moves replay along the stability–plasticity trade-off; it does not close the gap to joint retraining.

Raising the memory from 500 to 2,000 records clearly helped only on UNSW-NB15; final macro-F1 changed by {{TON_er2000_vs_er500}}, {{UNSW_er2000_vs_er500}} and {{NB_er2000_vs_er500}} (uncorrected paired p {{TON_er2000_vs_er500_p}}, {{UNSW_er2000_vs_er500_p}}, {{NB_er2000_vs_er500_p}}; exploratory). Stored as 32-bit values, 2,000 records take {{TON_buffer_kb_2000}} KiB on TON-IoT, {{UNSW_buffer_kb_2000}} KiB on UNSW-NB15 and {{NB_buffer_kb_2000}} KiB on N-BaIoT, whereas all training data of a sequence, which joint retraining keeps, take {{TON_store_all_kib}}, {{UNSW_store_all_kib}} and {{NB_store_all_kib}} KiB.

**A tree model on the same memory.** A random forest retrained after each task on the same memory plus the new task's data (RF + memory, Section 4.3) reached a final macro-F1 of {{P3F_TON_rfmem_f1}}, {{P3F_UNSW_rfmem_f1}} and {{P3F_NB_rfmem_f1}}, against {{TON_er2000_f1}}, {{UNSW_er2000_f1}} and {{NB_er2000_f1}} for the CNN with ER-2000 (paired differences {{P3F_TON_rfmem_vs_er2000_d}}, {{P3F_UNSW_rfmem_vs_er2000_d}} and {{P3F_NB_rfmem_vs_er2000_d}}; uncorrected p {{P3F_TON_rfmem_vs_er2000_p}}, {{P3F_UNSW_rfmem_vs_er2000_p}} and {{P3F_NB_rfmem_vs_er2000_p}}), so it was {{P3F_rfmem_verdict}}. Against the CNN with ER-2000-pool (above; {{P3F_TON_er2000p_f1}}, {{P3F_UNSW_er2000p_f1}} and {{P3F_NB_er2000p_f1}}) the paired differences were {{P3F_TON_rfmem_vs_er2000p_d}}, {{P3F_UNSW_rfmem_vs_er2000p_d}} and {{P3F_NB_rfmem_vs_er2000p_d}} (uncorrected p {{P3F_TON_rfmem_vs_er2000p_p}}, {{P3F_UNSW_rfmem_vs_er2000p_p}} and {{P3F_NB_rfmem_vs_er2000p_p}}). The forest on the memory was also above CNN joint retraining on {{P3F_rfmem_above_joint}} (by {{P3F_TON_rfmem_vs_joint_cnn}} and {{P3F_NB_rfmem_vs_joint_cnn}}). RF + memory retained {{P3F_TON_rfmem_old_cls}}, {{P3F_UNSW_rfmem_old_cls}} and {{P3F_NB_rfmem_old_cls}} of the earlier classes (ER-2000: {{TON_er2000_old_cls}}, {{UNSW_er2000_old_cls}} and {{NB_er2000_old_cls}}) and flagged {{P3F_TON_rfmem_old_det}}, {{P3F_UNSW_rfmem_old_det}} and {{P3F_NB_rfmem_old_det}} of their attack records (ER-2000: {{TON_er2000_old_det}}, {{UNSW_er2000_old_det}} and {{NB_er2000_old_det}}); on UNSW-NB15 it flagged only {{P3F_UNSW_rfmem_old_det_t1}} of the Task-1 attacks. It ended at benign FPRs of {{P3F_TON_rfmem_fpr_final}}, {{P3F_UNSW_rfmem_fpr_final}} and {{P3F_NB_rfmem_fpr_final}}, so on UNSW-NB15 its low FPR comes with missed earlier attacks. One retraining sequence took {{P3F_TON_rfmem_train_s}}, {{P3F_UNSW_rfmem_train_s}} and {{P3F_NB_rfmem_train_s}} s. Retrained on all data after every task, the forest's class-level forgetting was {{P3F_TON_rfjoint_FM}}, {{P3F_UNSW_rfjoint_FM}} and {{P3F_NB_rfjoint_FM}}.

### 5.3 Forgetting the class, keeping the alert

Detection-level retention tells a different story from class-level retention. Below, retention of the earlier tasks is averaged over Tasks 0 and 1 after the last task; unlike A in Table 2, it excludes the task just learned. Figure 1 follows the first task through the sequence. Under fine-tuning, EWC and LwF, recall of the Task-0 classes drops to near zero as soon as Task 1 is learned, while most Task-0 attack records are still flagged as attacks:

- **Fine-tuning** recognises {{TON_naive_old_cls}}, {{UNSW_naive_old_cls}} and {{NB_naive_old_cls}} of the earlier classes, yet still flags {{TON_naive_old_det}}, {{UNSW_naive_old_det}} and {{NB_naive_old_det}} of their attack records as attacks. Most old attack records are relabelled as a newer attack family, not as benign traffic.
- **LwF** flags {{TON_lwf_old_det}}, {{UNSW_lwf_old_det}} and {{NB_lwf_old_det}} of the earlier attack records, although it recognises only {{TON_lwf_old_cls}}, {{UNSW_lwf_old_cls}} and {{NB_lwf_old_cls}} of their classes.
- **A-GEM** recognises {{P3F_TON_agem_old_cls}}, {{P3F_UNSW_agem_old_cls}} and {{P3F_NB_agem_old_cls}} of the earlier classes and flags {{P3F_TON_agem_old_det}}, {{P3F_UNSW_agem_old_det}} and {{P3F_NB_agem_old_det}} of their attack records.
- **Both replay variants** flag at least {{TON_er500_old_det}} of the earlier attack records on every dataset.

Detection-level retention has to be read together with the benign false-positive rate (FPR), because a model can keep alerting by flagging more of everything:

- **N-BaIoT:** the benign FPR stays at or below {{NB_fpr_all_max}} on average for every strategy at every stage, so its high detection-level retention is genuine.
- **UNSW-NB15:** learning Task 1 raises the benign FPR to {{UNSW_fpr_t1_min_noagem}}–{{UNSW_fpr_t1_max_noagem}} for fine-tuning, EWC, LwF, both replay variants and joint retraining ({{UNSW_joint_fpr_t1}} for joint retraining), so the rise comes with the Task-1 classes rather than with any strategy; A-GEM reaches {{P3F_UNSW_agem_fpr_t1}}. Fine-tuning and A-GEM fall back to {{UNSW_naive_fpr_final}} and {{P3F_UNSW_agem_fpr_final}} after Task 2 only because they forget the Task-1 classes; EWC, LwF, replay and joint retraining stay near this level ({{UNSW_ewc_fpr_final}}, {{UNSW_lwf_fpr_final}}, {{UNSW_er2000_fpr_final}} and {{UNSW_joint_fpr_final}}). The random forest ends at {{UNSW_rf_fpr}}.

Replay has a cost of its own on TON-IoT. ER-500 and ER-2000 end at benign FPRs of {{TON_er500_fpr_final}} and {{TON_er2000_fpr_final}}, against {{TON_joint_fpr_final}} for joint retraining and {{TON_rf_fpr}} for the random forest. About one benign flow in eight is flagged. Two properties of the replay loss can contribute: the class-balanced memory gives benign traffic the same number of slots as each attack class, so benign traffic is under-represented relative to its share of the data, and the double weighting of Section 4.2 further favours the earlier attack classes. Reserving half of the memory for benign records (ER-2000b, Section 4.3) lowered the final benign FPR to {{TON_er2000b_fpr_final}} (paired difference {{TON_er2000b_fpr_d}}, uncorrected p = {{TON_er2000b_fpr_p}}). That removes {{TON_er2000b_fpr_excess_removed}}% of replay’s excess over joint retraining. Final macro-F1 rose slightly, by {{TON_er2000b_f1_d}} (uncorrected p = {{TON_er2000b_f1_p}}), and detection-level retention was unchanged ({{TON_er2000b_det_A}} against {{TON_er2000_det_A}}). With the class-balanced memory added once to the training pool instead (ER-2000-pool), the final benign FPR on TON-IoT was {{P3F_TON_er2000p_fpr_final}} (paired difference to ER-2000 {{P3F_TON_er2000p_fpr_d}}, uncorrected p {{P3F_TON_er2000p_fpr_p}}), about the {{TON_joint_fpr_final}} of joint retraining, at the price of lower earlier-task retention (class level {{P3F_TON_er2000p_old_cls}} against {{TON_er2000_old_cls}}, detection level {{P3F_TON_er2000p_old_det}} against {{TON_er2000_old_det}}). Removing the per-minibatch oversampling alone thus brought the FPR to the joint level, and a half-benign memory under the original scheme removed {{TON_er2000b_fpr_excess_removed}}% of the excess. Both act on the effective weight of benign traffic relative to the earlier attack classes and were not combined, so we do not apportion the cost between them. How the memory enters the loss and what it holds are both settings worth tuning.

### 5.4 Not-yet-learned attacks

Table 4 shows how often attack families that the model has not yet learned raise an alert. After Task 0, the model flags {{TON_zd_stage0_t1}} of Task-1 and {{TON_zd_stage0_t2}} of Task-2 attack records on TON-IoT, {{UNSW_zd_stage0_t1}} and {{UNSW_zd_stage0_t2}} on UNSW-NB15, and {{NB_zd_stage0_t1}} and {{NB_zd_stage0_t2}} on N-BaIoT. The benign FPR at that point is only {{TON_fpr_t0}}, {{UNSW_fpr_t0}} and {{NB_fpr_t0}}, so these rates are not produced by flagging everything. Because the same Task-0 model is shared, they are identical for all strategies. On N-BaIoT, a detector trained on BASHLITE misses more than half of the Mirai traffic.

After Task 1, flagging of the still-unseen Task-2 families is {{TON_zd_stage1_min}}–{{TON_zd_stage1_max}} on TON-IoT, {{UNSW_zd_stage1_min}}–{{UNSW_zd_stage1_max}} on UNSW-NB15 and {{NB_zd_stage1_min}}–{{NB_zd_stage1_max}} on N-BaIoT (range over strategies), higher than after Task 0 for every strategy. Once three Mirai classes are known, the two Mirai UDP floods are flagged almost completely.

How well a closed-set detector covers new families therefore depends on how closely they resemble the families it already knows. The rates of {{NB_zd_stage0_t2}}–{{NB_zd_stage0_t1}} on N-BaIoT suggest that families unlike anything learned would need an open-set or anomaly component, which we did not test.

{{table:zeroday}}

### 5.5 Task order and pre-processing scope

**Task order.** Over the four random task orders (Table 5), replay beat fine-tuning, online EWC and LwF in every order on every dataset ({{TON_orders_replay_wins_noagem}}, {{UNSW_orders_replay_wins_noagem}}, {{NB_orders_replay_wins_noagem}}). {{P3F_orders_agem_sentence}} The smallest margin between the weaker replay variant and the best of fine-tuning, EWC and LwF was {{TON_orders_replay_margin_min_noagem}}, {{UNSW_orders_replay_margin_min_noagem}} and {{NB_orders_replay_margin_min_noagem}}. Averaged over the random orders, ER-2000 closed {{TON_er2000_orders_gap_closed}}%, {{UNSW_er2000_orders_gap_closed}}% and {{NB_er2000_orders_gap_closed}}% of the gap between fine-tuning and joint retraining, less than on the semantic order on UNSW-NB15.

Absolute levels depend on the order:

- On UNSW-NB15, fine-tuning averages {{UNSW_naive_orders_f1}} over the random orders but {{UNSW_naive_f1}} on the semantic order.
- On UNSW-NB15, ER-500 and ER-2000 are indistinguishable over the random orders ({{UNSW_er500_orders_f1}} and {{UNSW_er2000_orders_f1}}) but differ by {{UNSW_er2000_vs_er500}} on the semantic order, which places the four rare classes last.

This suggests that placing the rare classes last hurts fine-tuning and the small memory, whereas ER-2000 scores higher on the semantic order than over the random ones ({{UNSW_er2000_f1}} against {{UNSW_er2000_orders_f1}}). The benefit of a larger memory depends on the order. On the same dataset, joint retraining ranges from {{UNSW_joint_orders_f1_min}} to {{UNSW_joint_orders_f1_max}} over the random orders.

{{table:orders}}

**Pre-processing scope.** Fitting feature selection and scaling on all training data instead of Task 0 alone changed mean final macro-F1 by at most {{oracle_max_abs}} for fine-tuning, ER-2000 and joint retraining on the three datasets (five seeds; Table 6). On TON-IoT and UNSW-NB15, every 95% confidence interval of the paired difference lies within ±{{oracle_ci_bound_TU}}, so any effect of the shortcut there is small; the only change with an uncorrected p below 0.05 was a small loss for joint retraining on TON-IoT ({{TON_joint_oracle_d}}, uncorrected p = {{TON_joint_oracle_p}}; Holm over the three TON-IoT tests, p = {{TON_joint_oracle_pholm3}}). N-BaIoT varies more across seeds: the largest mean change, {{NB_naive_oracle_d}} for fine-tuning, is not significant (uncorrected p = {{NB_naive_oracle_p}}), but its interval {{NB_naive_oracle_ci}} cannot exclude an effect of a few hundredths in either direction. Not-yet-learned flagging changed by at most {{oracle_zd_max_abs}}, downwards on TON-IoT and N-BaIoT and upwards on UNSW-NB15. A lower rate under the all-data fit is the opposite of what leakage of the later families into the first model would be expected to produce; we did not investigate its cause. Under our pipeline, therefore, the shortcut did not inflate final results on TON-IoT and UNSW-NB15 by more than about 0.02, and the test is inconclusive on N-BaIoT.

How much the ablation changes the inputs bounds this conclusion. After min–max scaling, {{P3F_TON_out01_te}}% of TON-IoT, {{P3F_UNSW_out01_te}}% of UNSW-NB15 and {{P3F_NB_out01_te}}% of N-BaIoT test values lie outside the Task-0 range [0, 1] ({{P3F_TON_out01_rows_te}}%, {{P3F_UNSW_out01_rows_te}}% and {{P3F_NB_out01_rows_te}}% of the test records in at least one feature; for the training data of the later tasks, which is what the all-data fit would see, {{P3F_TON_out01_later_tr}}%, {{P3F_UNSW_out01_later_tr}}% and {{P3F_NB_out01_later_tr}}% of the values). The compression map extrapolates these values smoothly, and only {{TON_clip_share}}% of TON-IoT test values lie beyond the clipping limit of about 2,000 in min–max units. The all-data fit keeps {{P3F_TON_feat_oracle}} features on TON-IoT and {{P3F_UNSW_feat_oracle}} on UNSW-NB15, of which {{P3F_TON_feat_common}} and {{P3F_UNSW_feat_common}} are also among the {{TON_n_features}} and {{UNSW_n_features}} features selected on Task 0; on N-BaIoT only the min–max range changes. The shortcut may matter more with features whose range shifts strongly between attack families, or with a pipeline that does not extrapolate smoothly.

{{table:oracle}}

### 5.6 Input scaling for a neural detector on flow features

In a first run of the present protocol (five seeds, benign traffic in every task), the network received min–max-scaled inputs without the compression map of Section 3.3 and was trained for at most 40 epochs on Task 0 and for joint retraining, and 20 epochs on later tasks (patience 5).

On TON-IoT, joint retraining reached {{TON_v2_joint_f1}} in that run against {{TON_joint_f1}} now, and ER-2000 {{TON_v2_er2000_f1}} against {{TON_er2000_f1}}, while fine-tuning moved only from {{TON_v2_naive_f1}} to {{TON_naive_f1}}, because it fails for a different reason (forgetting). The random forest scores {{TON_rf_f1}}. A tree is unaffected by monotone rescaling of its inputs, so up to the clipping of a few extreme values this reference applies to both runs, and it showed that the network was under-trained rather than the task being hard.

**The other datasets:**
- On UNSW-NB15, joint retraining rose from {{UNSW_v2_joint_f1}} to {{UNSW_joint_f1}} and ER-2000 from {{UNSW_v2_er2000_f1}} to {{UNSW_er2000_f1}}.
- On N-BaIoT, whose features are window statistics rather than raw flow counts, joint retraining hardly changed ({{NB_v2_joint_f1}} and {{NB_joint_f1}}), and ER-2000 was higher in the first run ({{NB_v2_er2000_f1}} against {{NB_er2000_f1}}).

On TON-IoT and UNSW-NB15, the badly scaled inputs lowered the upper bound and replay much more than fine-tuning, which shrank the apparent benefit of replay.

**Which change mattered.** The correction changed two things at once, the input map and the training budget. Table 7 separates them for joint retraining on TON-IoT.

| Input map | Earlier budget | Present budget |
|---|---|---|
| Min–max only | {{TON_fact_minmax_v2}} ± {{TON_fact_minmax_v2_sd}} | {{TON_fact_minmax_v3}} ± {{TON_fact_minmax_v3_sd}} |
| Min–max + compression | {{TON_fact_asinh_v2}} ± {{TON_fact_asinh_v2_sd}} | {{TON_fact_asinh_v3}} ± {{TON_fact_asinh_v3_sd}} |

Averaged over the two budgets, the compression map contributed more than the longer budget ({{TON_fact_map_effect_avg}} against {{TON_fact_budget_effect_avg}}). The map raised final macro-F1 by {{TON_fact_map_effect_v2budget}} under the earlier budget and by {{TON_fact_map_effect_v3budget}} under the present one; the longer budget added {{TON_fact_budget_effect_minmax}} with min–max inputs but only {{TON_fact_budget_effect_asinh}} with the compression map, so under the present budget the two changes are of similar size. More training thus partly compensates for badly scaled inputs, but neither change alone reaches the combined result, which still stays below the random forest ({{TON_rf_f1}}). These differences are descriptive, on one dataset and one strategy, and we did not measure the mechanism, for example through the batch-normalisation statistics.

The min–max, earlier-budget cell reproduces the first run only approximately: {{TON_fact_minmax_v2}} against {{TON_v2_joint_f1}} (paired difference {{TON_fact_vs_v2_d}}, uncorrected p = {{TON_fact_vs_v2_p}}). The Task-0 confusion matrices are identical for every seed, so the difference arises in the later tasks. There that configuration is unstable: its final scores varied by seed with a standard deviation of {{TON_v2_joint_sd}}. The present code also computes the loss differently, so exact reproduction of the later tasks is not expected.

## 6. Discussion

### 6.1 Implications for deployed detectors

**Keep a small replay memory.** It is the one strategy that clearly worked for the network. On these datasets, at most 2,000 stored records ({{TON_buffer_kb_2000}}–{{NB_buffer_kb_2000}} KiB) recovered {{NB_er2000_gap_closed}}–{{TON_er2000_gap_closed}}% of the difference between fine-tuning and full retraining on the semantic order and {{UNSW_er2000_orders_gap_closed}}–{{TON_er2000_orders_gap_closed}}% over random orders. Regularisation did not help enough to justify its complexity in the class-incremental setting; only LwF gave a significant but small gain, on two datasets. Gradient projection with the same memory (A-GEM) {{P3F_agem_verdict_short}}, so where stored records are available, replaying them was the better use. How the memory enters the loss also matters: adding it once to the training pool instead of to every minibatch lowered the benign FPR on TON-IoT from {{TON_er2000_fpr_final}} to {{P3F_TON_er2000p_fpr_final}} (Section 5.3). {{P3F_rfmem_advice}}

**Forgetting mostly misnames attacks rather than hiding them.** Even plain fine-tuning kept flagging most old attack records. This matters for deployments that act on the alert rather than on the attack class. It holds only when read alongside the benign FPR:

- on UNSW-NB15 every strategy that kept the Task-1 classes, joint retraining included, also kept a benign FPR near 0.2, and fine-tuning's lower FPR came only from forgetting;
- replay on TON-IoT raised the benign FPR to {{TON_er500_fpr_final}}–{{TON_er2000_fpr_final}}, a cost that depends on how the memory is used (added to the training pool instead of every minibatch: {{P3F_TON_er2000p_fpr_final}}; a half-benign memory: {{TON_er2000b_fpr_final}}).

Benign FPRs of about 0.2 on UNSW-NB15 for every network, joint retraining included (random forest {{UNSW_rf_fpr}}), or above 0.1 for replay with minibatch mixing on TON-IoT (ER-2000-pool: {{P3F_TON_er2000p_fpr_final}}) would be unusable in practice without a second filtering stage or a recalibrated threshold. We therefore recommend reporting class-level retention, detection-level retention and the benign FPR together.

**New families are often, but not reliably, flagged before they are learned.** Coverage depended on how closely the new families resembled the known ones, from under half of Mirai traffic for a BASHLITE-trained model to almost all of it once other Mirai variants were known. A closed-set detector would likely need to be paired with an open-set or anomaly component.

**Joint retraining remained the best option for the network.** At these data sizes a full retraining sequence took about five to ten minutes of training on a laptop; learning Tasks 1 and 2 took a median of {{TON_joint_upd_s}}, {{UNSW_joint_upd_s}} and {{NB_joint_upd_s}} s with joint retraining against {{TON_er2000_upd_s}}, {{UNSW_er2000_upd_s}} and {{NB_er2000_upd_s}} s with ER-2000 (indicative only; Section 6.2). The case for continual learning in this setting therefore rests on data retention limits more than on compute: all training data of a sequence take {{UNSW_store_all_ratio}}–{{TON_store_all_ratio}} times the space of a 2,000-record memory. A replay memory stores real past records, so it weakens that case only in proportion to its size, which is small. The network itself has about 24,000 parameters (under 100 KiB at 32 bits). The forests are far larger: the final RF + memory model has {{P3F_TON_sz_rf_er2000_nodes}}, {{P3F_UNSW_sz_rf_er2000_nodes}} and {{P3F_NB_sz_rf_er2000_nodes}} tree nodes ({{P3F_TON_sz_rf_er2000_kib}}, {{P3F_UNSW_sz_rf_er2000_kib}} and {{P3F_NB_sz_rf_er2000_kib}} KiB serialised, {{P3F_sz_ratio_min}}–{{P3F_sz_ratio_max}} times the network) and the all-data forest {{P3F_TON_sz_rf_joint_kib}}, {{P3F_UNSW_sz_rf_joint_kib}} and {{P3F_NB_sz_rf_joint_kib}} KiB. On one CPU thread, predicting 10,000 records took {{P3F_TON_sz_rf_er2000_ms10k}}, {{P3F_UNSW_sz_rf_er2000_ms10k}} and {{P3F_NB_sz_rf_er2000_ms10k}} ms with RF + memory against {{P3F_TON_sz_cnn_ms10k}}, {{P3F_UNSW_sz_cnn_ms10k}} and {{P3F_NB_sz_cnn_ms10k}} ms with the network (seed 0, on a laptop running other work, so indicative only). Where a detector must fit in a few hundred kilobytes, the network is the option that does; otherwise the forest is the stronger detector on two of the three datasets. We did not measure peak memory or energy on an edge device.

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

Experience replay with 2,000 stored records recovered {{NB_er2000_gap_closed}}–{{TON_er2000_gap_closed}}% of the difference between fine-tuning and retraining on the semantic order and {{UNSW_er2000_orders_gap_closed}}–{{TON_er2000_orders_gap_closed}}% over random orders, ahead of {{P3F_orders_nonreplay}} in every order. Online EWC did not significantly improve on fine-tuning, LwF helped only a little, and A-GEM {{P3F_agem_verdict_short}}. On the semantic order, replay's remaining gap to retraining lay mainly in learning the newest family and, on TON-IoT, in benign false positives, rather than in forgetting the older ones. Both depend partly on the double weighting of the earlier classes in the replay loss: with the memory added once to the training pool, the newest-task shortfall shrank by more than half on all three datasets, the benign FPR on TON-IoT fell from {{TON_er2000_fpr_final}} to {{P3F_TON_er2000p_fpr_final}}, and part of the gap moved to the earlier tasks; final macro-F1 rose clearly only on UNSW-NB15.

Forgotten classes were mostly still flagged as attacks, and replay raised the benign false-positive rate on TON-IoT; reserving half of the memory for benign records removed part of that cost, and adding the memory to the training pool instead of every minibatch brought it to the joint-retraining level. Unseen families were flagged at rates from under half to nearly all, depending on their similarity to known ones. Fitting the pre-processing on future data changed results only marginally on TON-IoT and UNSW-NB15 and inconclusively on N-BaIoT.

A random forest retrained on the same memory {{P3F_rfmem_verdict}}; it is far larger than the network, but it is the reference a continually trained network should be compared with. Finally, a random-forest reference showed that the network was under-trained on min–max-scaled, heavy-tailed flow features; a fixed compression map together with a longer budget closed most of that gap. Such a reference should be standard in IDS evaluations.

## Declarations

**Data availability.** TON-IoT, UNSW-NB15 and N-BaIoT are publicly available from their providers. The code, task definitions and all result files are available at {{REPO}}. The analysis code was written with the assistance of Claude (Anthropic) and checked by the authors.

**CRediT authorship contribution statement.** Emmanuel Kwame Amoako: Conceptualization, Methodology, Software, Formal analysis, Investigation, Visualization, Writing – original draft, Writing – review & editing. Mahamudu Amadu: Conceptualization, Methodology, Validation. James Quayson: Conceptualization, Methodology, Writing – review & editing. Kingsford Donkor: Validation, Writing – review & editing. Daniel Asamoah Apraku: Validation, Writing – review & editing.

**Declaration of competing interest.** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Declaration of generative AI and AI-assisted technologies in the writing process.** {{AI_DECLARATION}}

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
