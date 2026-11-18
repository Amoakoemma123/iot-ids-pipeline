# Paper 6: reference and citation-use check (review round 1)

Checked on 2026-11-01. The manuscript was not edited.

## How the check was done

- **DOIs.** All 32 DOIs in the References section were resolved against `api.crossref.org/works/<doi>`. For each record I compared title, authors and their order, year, venue, volume, issue, pages or article number, and the DOI itself.
- **Items without a DOI.** These were checked on the publisher's own page:
  - PMLR v37 `ioffe15`, v162 `huang22e` and v162 `wu22d`
  - JMLR v12 `pedregosa11a`
  - USENIX Security '22 `arp`
  - NeurIPS proceedings for Nagrani 2021, Vaswani 2017 and Grinsztajn 2022
  - the arXiv abstract page for Bahdanau (its comment reads "Accepted at ICLR 2015 as oral presentation")
- **Abstracts.** These came from Crossref, or from OpenAlex where Crossref had none.
- **Full texts read in this round (each one grepped):**
  - Alabbadi & Bajaber 2025, PMC12196925
  - AboulEla & Kashef 2026, PMC13363868
  - Ullah et al. 2024, PMC11243847
  - Xu et al. 2025, PMC12219291
  - Ayantayo et al. 2023, SpringerOpen HTML
  - Kiflay et al. 2024, author copy at uhra.herts.ac.uk
  - Ishtiaq et al. 2025, arXiv:2510.02717
  - Dinh et al., the extended version arXiv:2403.15511v2. The ICC 2024 paper itself is not open access.
- **Full texts I could not retrieve:**
  - Attota et al. 2021 (IEEE and the repository blocked the download)
  - Li et al. 2025 (IEEE blocked the download)
  - Singh et al. 2026 (the Bristol repository returned HTML instead of the PDF)

  For these three, the abstract is all I checked this time.

## 1. Reference table

Key: **OK** = exists, and every metadata field matches Crossref or the publisher. **OK\*** = correct, with an APA or completeness improvement suggested. No reference was fabricated, and none was not found.

| # | Reference | Status | Notes / correction |
|---|---|---|---|
| 1 | AboulEla & Kashef (2026), *Sensors 26*(13), 4196 | OK | Title, authors, volume, issue and article number match Crossref (issued 2026-07-02). |
| 2 | Afraji, Lloret & Peñalver (2025), *Computation 13*(9), 222 | OK | Crossref lists the first author as "Afraji, Doaa Mohsin Abd Ali", so "Afraji, D. M. A. A." is correct. |
| 3 | Al Qathrady et al. (2024), *CAAI TIT 9*(6), 1398–1411 | OK | Crossref stores the surname as "Qathrady" with given name "Mimonah Al". The paper itself uses "Al Qathrady", so keep that spelling. All 8 authors and their order match. |
| 4 | Alabbadi & Bajaber (2025), *Sensors 25*(12), 3693 | OK | |
| 5 | Alsaedi et al. (2020), *IEEE Access 8*, 165130–165150 | OK | |
| 6 | Arp et al. (2022), USENIX Security, pp. 3971–3988 | OK\* | Authors and pages match usenix.org. APA 7 wants a URL for an item without a DOI: add `https://www.usenix.org/conference/usenixsecurity22/presentation/arp`. |
| 7 | Attota et al. (2021), *IEEE Access 9*, 117734–117745 | OK | |
| 8 | Ayantayo et al. (2023), *J. Big Data 10*, 167 | OK\* | Crossref has issue 1. APA form: *Journal of Big Data, 10*(1), Article 167. |
| 9 | Bahdanau, Cho & Bengio (2015), ICLR | OK\* | ICLR acceptance is confirmed by the arXiv comment. Optionally add `https://arxiv.org/abs/1409.0473`. |
| 10 | Baltrušaitis et al. (2019), *IEEE TPAMI 41*(2), 423–443 | OK | Crossref drops the diacritic ("Baltrusaitis"). The author's own spelling, as used, is fine. |
| 11 | Chaabouni et al. (2019), *IEEE COMST 21*(3), 2671–2701 | OK | |
| 12 | Chawla et al. (2002), *JAIR 16*, 321–357 | OK | |
| 13 | Dinh et al. (2024), ICC 2024, pp. 2707–2712 | OK | The author order (Dinh, Hoang, Uy, Nguyen, Bao, Dutkiewicz) matches Crossref. |
| 14 | Diro & Chilamkurti (2018), *FGCS 82*, 761–768 | OK | |
| 15 | Engelen, Rimmer & Joosen (2021), IEEE SPW, pp. 7–12 | OK | |
| 16 | Grinsztajn, Oyallon & Varoquaux (2022), NeurIPS 35 | **Incomplete** | Crossref and the NeurIPS page give **pp. 507–520**; add them. APA form: *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://doi.org/10.52202/068431-0037. The "typical" in the title is correct. |
| 17 | Huang et al. (2022), ICML, PMLR 162, 9226–9259 | OK\* | Authors and pages match PMLR. APA form: *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 9226–9259). PMLR. Add `https://proceedings.mlr.press/v162/huang22e.html`. |
| 18 | Imrana et al. (2021), *ESWA 185*, 115524 | OK\* | APA: "Article 115524". |
| 19 | Ioffe & Szegedy (2015), ICML, PMLR 37, 448–456 | OK\* | Same as #17: use "(Vol. 37, pp. 448–456). PMLR." and add `https://proceedings.mlr.press/v37/ioffe15.html`. |
| 20 | Ishtiaq et al. (2025), *Array 27*, 100501 | OK | Crossref splits the surnames as "Shahariar Parvez", "Alamgir Hossain", "Hasan Kanchan" and "Masud Tarek". The initials as given (Parvez, A. H. M. S.; Hossain, M. A.; Kanchan, M. H.; Tarek, M. M.) are a defensible reading; check them against the article's byline. APA: "Article 100501". |
| 21 | Jain & Wallace (2019), NAACL-HLT, pp. 3543–3556 | OK | |
| 22 | Kiflay et al. (2024), *Array 22*, 100349 | OK | APA: "Article 100349". |
| 23 | Lansky et al. (2021), *IEEE Access 9*, 101574–101599 | OK | |
| 24 | Li, Qiao & Lee (2025), *IEEE Access 13*, 83996–84012 | OK | |
| 25 | Meidan et al. (2018), *IEEE Pervasive Computing 17*(3), 12–22 | OK | |
| 26 | Mirsky et al. (2018), NDSS | OK\* | DOI resolves. Optionally add the publisher: "Internet Society." |
| 27 | Moustafa (2021), *Sustainable Cities and Society 72*, 102994 | OK | APA: "Article 102994". |
| 28 | Moustafa & Slay (2015), MilCIS, pp. 1–6 | OK | The year 2015 is correct. |
| 29 | Nagrani et al. (2021), NeurIPS 34 | **Incomplete** | The NeurIPS page gives **pp. 14200–14213**; add them, plus "Curran Associates" and the proceedings URL. |
| 30 | Pedregosa et al. (2011), *JMLR 12*(85), 2825–2830 | OK | The JMLR page metadata gives issue 85 and pp. 2825–2830. Optionally add the jmlr.org URL. |
| 31 | Peng et al. (2022), CVPR, pp. 8228–8237 | OK | |
| 32 | Ramachandram & Taylor (2017), *IEEE SPM 34*(6), 96–108 | OK | |
| 33 | Shwartz-Ziv & Armon (2022), *Information Fusion 81*, 84–90 | OK | |
| 34 | Singh et al. (2026), ICISSP, pp. 396–405 | OK | Matches Crossref, including all 5 authors. |
| 35 | Sommer & Paxson (2010), IEEE S&P, pp. 305–316 | OK | |
| 36 | Ullah et al. (2024), *Sensors 24*(13), 4152 | OK | |
| 37 | Vaswani et al. (2017), NIPS 30 | OK\* | The official page shows no page numbers, so leaving them out is acceptable. Add "Curran Associates" and the URL for APA. |
| 38 | Wang, Tran & Feiszli (2020), CVPR, pp. 12692–12702 | OK | The third author is correct (Feiszli). |
| 39 | Wu et al. (2022), ICML, PMLR 162, 24043–24055 | OK\* | Same format note as #17. URL: `https://proceedings.mlr.press/v162/wu22d.html`. |
| 40 | Xu, Zhan, Wang & Yang (2025), *Sci. Rep. 15*, 21986 | OK\* | Crossref has issue 1. APA form: *Scientific Reports, 15*(1), Article 21986. |

**Cross-check.** All 40 entries are cited in the text, and every in-text citation has an entry. The list is in correct alphabetical order: "Al Qathrady" before "Alabbadi" follows APA's nothing-before-something rule.

**APA consistency.** Article-number journals (Elsevier, Springer, Nature) should consistently use "Article N", while MDPI is conventionally printed without the word. Conference items without a DOI should carry a URL.

## 2. How the citations are used in the text

| Claim in manuscript (Sec. 2.1) | Source checked | Verdict |
|---|---|---|
| Kiflay et al. (2024) combine flow features with the first payload bytes of each UNSW-NB15 flow | Full text | **Supported** (six flow features plus the first 32 bytes of the protocol payload). **Wording problem:** the paragraph opens with "Deep-learning IDS … moved … to designs with several input branches", but Kiflay uses two **random forests** with soft voting, and no neural network at all. Reword it, e.g. "Some fuse genuinely different sources, with neural or tree models: …". Kiflay also reports that decision-level fusion beat early fusion (an RF on the concatenated features), which is worth noting because it is a positive result for fusion of *different* sources. |
| Singh et al. (2026) combine network and host audit data | Abstract | **Supported**: network (inter-host) and audit (intra-host) telemetry, Windows 7/10. |
| Singh et al.: tree ensembles on the fused features reached higher accuracy than the fused CNN | Abstract | **Supported**: the 1D-CNN reached 99.32%, while XGBoost and RF "achieved ideal accuracy". |
| Attota et al. (2021) train one network per view (bi-flow, uni-flow, packet features) of an MQTT dataset | Abstract only (full text blocked) | **Plausible, not re-verified.** The abstract confirms multi-view, federated learning, an ensemble and IoT data, but it names neither the dataset nor the views. The three views match the published feature levels of MQTT-IoT-IDS2020 (Hindy et al.), and the earlier notes record a full-text read. Confirm against the PDF before submission. |
| Ayantayo et al. (2023) split UNSW-NB15 and NSL-KDD features by data type and compare early and late fusion | Full text | **Supported**: float, integer and string groups; early fusion, late fusion and late-ensemble models; Table 4 RF/DT figures are quoted from other papers, not re-run. |
| Dinh et al. (2024) give each feature group its own sub-encoder | ICC abstract plus extended arXiv full text | **Supported.** In the extended version the groups come from KL-divergence clustering of feature distributions. **However**, see the 14-paper claim below: this paper *does* compare against a standalone RF on the original data. |
| Xu et al. (2025) fuse a graph and a feature-set representation of the same traffic | Full text | **Misleading.** The "traffic feature graph" is a 64×64 greyscale **image of the raw first 256 bytes of the first 16 packets**, headers and payload included. The second input is flow statistics. This is a raw-bytes plus flow-statistics design, the same kind of pairing as Kiflay's payload plus flow, yet Kiflay sits in the "genuinely different sources" group and Xu in the "same flow table or packet capture" group. Both take their inputs from the same PCAP. Use one criterion for both. Suggested wording: "Xu et al. (2025) fuse a byte-level image of the first packets of a flow with the flow's statistics." |
| Ullah et al. (2024) fuse text-like and image-like encodings of the same packets | Full text | **Broadly supported**: word2vec features from crawled PCAP content plus a bytes-to-image ResNet. The same consistency point as Xu applies: this also uses raw bytes, not a split of a flow table. |
| AboulEla & Kashef (2026) fuse numeric, text, graph and quantum-inspired representations of the same NetFlow records | Full text | **Supported**; the authors write "complementary views of the same NetFlow records". |
| AboulEla & Kashef report only modest, dataset-dependent gains | Full text | **Supported**, and if anything understated. On one dataset no fusion model beat the best single model (M2, the text-based model) on F1. |
| Attention over branches or features is common (Al Qathrady 2024; Ishtiaq 2025) | Abstracts and Ishtiaq full text | **Supported** for attention over *features*. Al Qathrady uses self-attention on the inputs; Ishtiaq uses channel and temporal attention inside a single network. Neither has attention over *branches*, so "over features" is the part these citations support. |
| Alabbadi & Bajaber (2025) feed the same full feature vector to both paths | Full text, Algorithm 1 | **Supported**: both the deep encoder and the sparse transformer map R^d, applied to the same X_train, followed by concatenation and PCA. |
| Accuracies are often near 100% (Ishtiaq 2025; Afraji 2025) | Abstracts | **Supported**: Ishtiaq 99.97% on Edge-IIoTset; Afraji 100% binary on TON_IoT. Afraji's multiclass accuracy is 97%, so make clear the 100% figure is binary. |
| Li et al. (2025) conclude that multi-view learning is not the best choice in all scenarios ("point in our direction") | Abstract only | **Overstated / selectively quoted.** The quote is accurate, but the same abstract says multi-view AE and CNN models "performed better in general than the corresponding single view model". The multi-view CNN got the best F1 on TON_IoT, and the authors "confirmed the effectiveness of multi-view methods". The "not best in all scenarios" remark refers to DGCCA losing to some single-view models. Li's views are also **host and network data**, i.e. different sources, so the paper does not test splitting one flow table. Either drop it from "point in our direction" or describe it fairly, e.g. "Li et al. (2025) found multi-view models generally, but not always, better than single-view ones". |
| In a sample of 14 fusion papers (5 full text, 9 abstracts), none reported random groupings, none a matched-parameter single network, and at most two a tree ensemble on the same split | 8 full texts plus 6 abstracts | **Partly supported; needs correction.** Details below. |

**The 14-paper claim in detail:**

- **The sample cannot be identified from the paper.** Thirteen of the 14 sample papers are cited, but the 14th (Sanjalawe et al., 2025, *IoT and Cyber-Physical Systems 5*, 143–164) is not, so the sample cannot be checked from the manuscript. List the 14 papers in a supplementary table, as in Table B of `literature_verified.md`.
- **Random groupings: 0 of 14 holds.** I grepped the 8 full texts now available, plus Attota per the earlier notes, and none uses random groupings.
- **Matched-parameter single network: 0 of 14 holds.** No full text matches parameter counts.
- **Tree ensembles: "at most two" is probably wrong; it should be three.**
  - The two already counted are Kiflay (RF on the concatenated data) and Singh (XGBoost and RF on the fused features).
  - The third is **Dinh et al.** The extended version (Table V) compares MIAE+RF with LR, SVM, DT and **RF on the original data** ("standalone"). On UNSW the F-scores are 0.727 for standalone RF against 0.728 for MIAE+RF, essentially equal, which supports this paper's position.
  - The ICC abstract also reports comparisons against "conventional classifiers". The earlier note saying Dinh "uses RF only on top of the representation" is incorrect.
  - Ullah, Xu, AboulEla, Alabbadi, Ishtiaq and Ayantayo do not re-run a tree ensemble on the same split. In AboulEla, XGBoost appears only as a meta-fuser.
- **Update the counts.** Change the sentence to "(eight in full text, six from their abstracts)" and "at most three a tree ensemble trained on the same data". Alternatively, say "none of the full texts and none of the abstracts mentions …", because absence from an abstract is weak evidence.

**Other usage notes:**

- **Sec. 3.2.** The manuscript says the flow-versus-behaviour split is used "in similar forms, in the fusion literature" but cites nothing. Cite Ayantayo et al. (2023) and Dinh et al. (2024), the closest examples.
- **Sec. 2.2 and 2.3.** The Wang, Peng, Wu, Huang, Jain & Wallace, Grinsztajn, Shwartz-Ziv, Arp, Sommer & Paxson and Engelen citations are used consistently with what those papers are known for. No problems found.

## 3. Missing literature a referee would expect

All of these were confirmed on Crossref or PMLR in this check.

**1. Random-split control**
- Ho, T. K. (1998). The random subspace method for constructing decision forests. *IEEE TPAMI, 20*(8), 832–844. https://doi.org/10.1109/34.709601
- This is the classic precedent for random feature subsets and gives the random-split control a lineage. It is pre-2023 but expected.

**2. Multimodal joint training vs unimodal learning (2023–2024)**
- Du, C., Teng, J., Li, T., Liu, Y., Yuan, T., Wang, Y., Yuan, Y., & Zhao, H. (2023). On uni-modal feature learning in supervised multi-modal learning. *Proc. 40th ICML* (Vol. 202, pp. 8632–8656). PMLR.
  - It shows that simple uni-modal ensembles can match or beat joint multimodal training, which directly supports Sec. 2.2.
- Liang, P. P., Zadeh, A., & Morency, L.-P. (2024). Foundations & trends in multimodal machine learning: Principles, challenges, and open questions. *ACM Computing Surveys, 56*(10), 1–42. https://doi.org/10.1145/3656580
  - This is the current survey; Baltrušaitis 2019 and Ramachandram 2017 alone look dated.

**3. Tabular deep learning vs trees (2023–2024)**
- McElfresh, D., et al. (2023). When do neural nets outperform boosted trees on tabular data? *NeurIPS 36* (pp. 76336–76369). https://doi.org/10.52202/075280-3337
- Borisov, V., et al. (2024). Deep neural networks and tabular data: A survey. *IEEE TNNLS, 35*(6), 7499–7519. https://doi.org/10.1109/TNNLS.2022.3229161

**4. NIDS evaluation pitfalls (2023–2024)**
- Apruzzese, G., Laskov, P., & Schneider, J. (2023). SoK: Pragmatic assessment of machine learning for network intrusion detection. *2023 IEEE EuroS&P* (pp. 592–614). https://doi.org/10.1109/EuroSP57164.2023.00042
  - A referee in this area will almost certainly expect it next to Arp et al.
- Flood, R., Engelen, G., Aspinall, D., & Desmet, L. (2024). Bad design smells in benchmark NIDS datasets. *2024 IEEE EuroS&P* (pp. 658–675). https://doi.org/10.1109/EuroSP60621.2024.00042
  - It is relevant because the paper relies on TON-IoT and UNSW-NB15 and reports a saturated N-BaIoT.
- Lanvin, M., et al. (2023). Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes. In *CRiSIS 2022*, LNCS (pp. 18–33). Springer. https://doi.org/10.1007/978-3-031-31108-6_2
  - Optional; it pairs with Engelen 2021.

**5. Attention as explanation**
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation. *EMNLP-IJCNLP 2019* (pp. 11–20). https://doi.org/10.18653/v1/D19-1002
- A referee will expect this counterpoint whenever Jain & Wallace is cited. Pre-2023, but standard.

**6. Flow feature sets**
- Sarhan, M., Layeghy, S., & Portmann, M. (2022). Towards a standard feature set for network intrusion detection system datasets. *Mobile Networks and Applications, 27*(1), 357–370. https://doi.org/10.1007/s11036-021-01843-0
- Useful because AboulEla's MM-NIDS uses these NF-* datasets, and because it bears on how flow features are grouped.

## 4. Priority fixes

1. Correct the 14-paper sentence: Dinh's standalone-RF comparison makes the tree-ensemble count three; update the full-text/abstract split; list the sample in a supplement.
2. Reframe Li et al. (2025) fairly, or remove it from "point in our direction".
3. Use one criterion for "genuinely different sources" across Kiflay, Xu and Ullah, and fix the "graph" description of Xu.
4. Fix the "deep-learning" framing around Kiflay, who uses random forests.
5. Add pages to Grinsztajn (507–520) and Nagrani (14200–14213), and URLs to the items without a DOI.
6. Confirm Attota's MQTT dataset and the three views against the PDF.
7. Consider adding Apruzzese 2023, Du 2023, McElfresh 2023, Liang 2024 and Wiegreffe & Pinter 2019.
