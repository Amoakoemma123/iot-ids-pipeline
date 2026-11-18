# Paper 4 v2: Reference Verification, Round 4

Checked 2026-10-26 against `Manuscript.md` (38 references; numbering has changed since round 3, which had 42).

**How I checked:**
- All 25 DOIs were checked against the Crossref REST API (`api.crossref.org/works/<doi>`), covering title, authors, container, volume, issue, pages or article number, and print/online dates.
- Entries without a DOI were checked against the arXiv API (author lists and titles), the PMLR proceedings pages (citation_firstpage and citation_lastpage), and the MLSys and NeurIPS proceedings pages and their official BibTeX.
- For the citation-support checks I read the abstracts and, where it mattered, the full text: the Rey et al. arXiv PDF and the Bagdasaryan et al. NeurIPS PDF.
- No manuscript file was modified.

**Bottom line:**
- All 38 references exist. None is fabricated or unfindable.
- **35 verified, 2 metadata or format errors ([33], [35]), 0 not found, 0 misattributed.**
- **1 citation-support problem**: [37] and [38] are offered as "datasets with separate sites".
- Numbering is clean. Every reference [1]–[38] is cited, every citation has an entry, and first citations run in strict order 1, 2, …, 38. The only range is "[4]–[6]", written correctly with an en dash. The Supplement contains no numbered citations.

## 1. Per-reference table

| # | Reference (short) | Status | Correction / note |
|---|---|---|---|
| 1 | Diro & Chilamkurti, FGCS 82:761–768, 2018 | Verified | |
| 2 | Agrawal et al., Comput. Commun. 195:346–361, 2022 | Verified | 9 authors, so "et al." is correct. |
| 3 | McMahan et al., FedAvg, AISTATS, PMLR 54:1273–1282, 2017 | Verified | Authors match arXiv:1602.05629. Pages match PMLR. |
| 4 | Nguyen et al., DÏoT, ICDCS 2019, pp. 756–767 | Verified | 6 authors, all listed. |
| 5 | Hernandez-Ramos et al., ACM CSUR 57(12):1–65, 2025 | Verified | 7 authors, so "et al." is correct. |
| 6 | Mármol Campos et al., Comput. Netw. 203, 108661, 2022 | Verified | Surname now correct ("Mármol Campos", 7 authors). |
| 7 | Hsu, Qi & Brown, arXiv:1909.06335, 2019 | Verified | |
| 8 | Q. Li, Diao, Chen & He, ICDE 2022, pp. 965–978 | Verified | |
| 9 | Abadi et al., CCS 2016, pp. 308–318 | Verified | |
| 10 | McMahan, Ramage, Talwar & Zhang, ICLR 2018 | Verified | arXiv:1710.06963. |
| 11 | Alsaedi et al., TON_IoT telemetry, IEEE Access 8:165130–165150, 2020 | Verified | |
| 12 | Moustafa, Network TON_IoT, Sustain. Cities Soc. 72, 102994, 2021 | Verified | |
| 13 | Moustafa & Slay, UNSW-NB15, MilCIS 2015, pp. 1–6 | Verified | |
| 14 | Meidan et al., N-BaIoT, IEEE Pervasive Comput. 17(3):12–22, 2018 | Verified | 7 authors, so "et al." is correct. Nine devices, BASHLITE and Mirai: correct. |
| 15 | T. Li et al., FedProx, MLSys vol. 2, pp. 429–450, 2020 | Verified | Pages confirmed by the official MLSys BibTeX. |
| 16 | Reddi et al., Adaptive federated optimization, ICLR 2021 | Verified | 8 authors, so "et al." is correct. |
| 17 | Karimireddy et al., SCAFFOLD, PMLR 119:5132–5143, 2020 | Verified | 6 authors, all listed. Pages match PMLR. |
| 18 | J. Zhang et al., FedLC, PMLR 162:26311–26329, 2022 | Verified | 7 authors, so "et al." is correct. Pages match PMLR (zhang22p). |
| 19 | Hsieh et al., Non-IID data quagmire, PMLR 119:4387–4398, 2020 | Verified | Contribution 5 is now reworded ("in line with Hsieh et al.'s recommendation of group normalisation"). The round-3 misattribution is fixed. |
| 20 | Wu & He, Group normalization, ECCV, LNCS 11217, 2018, pp. 3–19 | Verified | |
| 21 | Mironov, Rényi DP, IEEE CSF 2017, pp. 263–275 | Verified | |
| 22 | Mironov, Talwar & Zhang, arXiv:1908.10530, 2019 | Verified | |
| 23 | Rey et al., Comput. Netw. 204, 108693, 2022 | Verified | The journal version has 4 authors (the arXiv version adds M. Jaggi), so the entry is correct. The claim is supported. |
| 24 | Lazzarini, Tianfield & Charissis, AI 4(3):509–530, 2023 | Verified | |
| 25 | Popoola et al., IEEE TCE 70(1):1610–1622, 2024 | Verified | 6 authors, all listed. |
| 26 | Olanrewaju-George & Pranggono, Cyber Secur. Appl. 3, 100068, 2025 | Verified | Online 2024; vol. 3 issue dated 2025. |
| 27 | Peng, Wu & Xiao, FD-IDS, Sensors 25(14), 4309, 2025 | Verified | |
| 28 | Nguyen, Rieger, Miettinen & Sadeghi, DISS 2020 | Verified | |
| 29 | Zhao et al., arXiv:1806.00582, 2018 | Verified | 6 authors, all listed. |
| 30 | X. Li et al. (5 authors, all listed), FedBN, ICLR 2021 | Verified | |
| 31 | T. Li, Hu, Beirami & Smith, Ditto, PMLR 139:6357–6368, 2021 | Verified | |
| 32 | Fu et al., DP-FL systematic review, ACM CSUR 58(11):1–38, 2026 | Verified | 9 authors, so "et al." is correct. Online 2026-04, print 2026-08. |
| 33 | Ruzafa-Alcazar et al., IEEE TII 19(2):1145–1154, 2023 | **Format error (IEEE "et al." rule)** | The metadata is correct and all **7** authors are now listed. However, IEEE style (and every other entry in this list: [2], [5], [6], [9], [14], [16], [18], [32]) uses "et al." when there are more than 6 authors. **Change to "P. Ruzafa-Alcazar et al., …"**. This also removes the name-form inconsistency: Crossref spells the third author "E. Marmol-Campos", while [6] gives "E. Mármol Campos". |
| 34 | Friha et al., 2DF-IDS, Comput. Secur. 127, 103097, 2023 | Verified | 6 authors, all listed. |
| 35 | Bagdasaryan & Shmatikov, NeurIPS vol. 32, 2019, pp. 15479–15488 | **Metadata error (missing author)** | The NeurIPS 2019 proceedings version (PDF and official BibTeX) has **three** authors: Eugene Bagdasaryan, **Omid Poursaeed**, Vitaly Shmatikov. Only the arXiv preprint (1905.12101) lists two. **Fix:** "E. Bagdasaryan, O. Poursaeed, and V. Shmatikov, …". The NeurIPS proceedings give **no page numbers** (the official BibTeX has `pages = {}`), and I could not verify "pp. 15479–15488" from a primary source, so **drop the pages** or replace them with the paper URL. |
| 36 | Mirsky et al., Kitsune, NDSS 2018 | Verified | DOI 10.14722/ndss.2018.23204 verified. N-BaIoT's 115 features come from the Kitsune (AfterImage) extractor, so the citation is appropriate. |
| 37 | Neto et al., CICIoT2023, Sensors 23(13), 5941, 2023 | Verified (metadata); **support problem**, see §2 | |
| 38 | Ferrag et al., Edge-IIoTset, IEEE Access 10:40281–40306, 2022 | Verified (metadata); **support problem**, see §2 | |

## 2. Citation-support problems

1. **[37], [38] in §VI Limitations: "Datasets with separate sites, such as CICIoT2023 [37] or Edge-IIoTset [38], would be more realistic."** Neither dataset was captured at separate sites.
   - The CICIoT2023 abstract: "33 attacks are executed in an IoT topology composed of 105 devices". This is one laboratory topology.
   - Edge-IIoTset is generated on a single seven-layer testbed. It is designed *for* centralised and federated learning, but its FL clients are partitions of one capture, just as in this paper.
   - Citing them as examples of multi-site data overstates what they offer, and a reviewer familiar with them will notice. **Suggested rewording:** "Datasets with more devices, such as CICIoT2023 (105 devices) [37], or designed for federated evaluation, such as Edge-IIoTset [38], would allow richer device-level partitions; traffic captured at genuinely separate sites remains scarce."

All other citing sentences are supported. I checked the following:
- [2], [5] identify heterogeneity, communication and privacy–utility trade-offs as open problems. Supported.
- [4] DÏoT trains device-type-specific models federatedly. Supported.
- [23] Rey et al. The full text says: "clients that own the data from a single device each", on N-BaIoT, with robust aggregation. Supported. Strictly, one device is held out as the "new device", so the training federation has fewer than nine clients. The wording is still accurate.
- [6] evaluates FL-IDS on ToN_IoT with different data distributions. Supported.
- [24] compares aggregation algorithms. Supported.
- [25] covers consumer IoT. Supported.
- [26] covers unsupervised and supervised models. Supported.
- [27] uses knowledge distillation. Supported.
- [28] covers poisoning. Supported.
- [29] FedAvg degrades under label skew. Supported.
- [15], [16], [17], [18] algorithm descriptions are supported: FedProx's proximal term, FedAdam's server Adam, SCAFFOLD option II with doubled communication, and FedLC's τ·n^(−1/4) calibration.
- [30] FedBN keeps BN statistics local. Supported.
- [31] Ditto personalises. Supported.
- [7] covers Dirichlet partitioning. Supported.
- [8] reports that no algorithm dominates. Supported.
- [19] treats BN as a failure point under skew and recommends GN. Supported.
- [9] covers DP-SGD with per-example clipping, Gaussian noise and Poisson ("lot") sampling. Supported.
- [21], [22] cover the RDP and subsampled-Gaussian accountant. Supported.
- [10], [32] client-level DP needs many clients. Supported. McMahan et al. state this directly, so the claim no longer needs the dropped Wei et al. reference.
- [33], [34] apply DP to FL-IDS for IIoT. Supported.
- [35] DP hurts under-represented classes more. Supported; the abstract also says "more complex subgroups", which is consistent with "extends here beyond small classes".
- [11], [12] TON-IoT has nine attack classes plus benign. Supported.
- [13] UNSW-NB15 has nine attack categories. Supported.
- [14] N-BaIoT has nine devices and the BASHLITE and Mirai botnets. Supported.

## 3. Other observations (minor, optional)

1. **Uncited companion work (carried over from round 3).** §III-A says "the corrected pipeline of our companion studies, released with this paper", with no citation. Either cite the companion paper(s) (preprint or "under review") or say "our corrected pipeline, released with this paper".
2. **Holm correction is uncited.** §III-D uses Holm-corrected p-values. Optionally cite S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979. This is optional, since many IEEE venues do not require it.
3. **Conference locations are inconsistent.** Only [28] (DISS) gives a location ("San Diego, CA, USA"). Either remove it or add locations to all conference entries. Removing it is simpler.
4. **arXiv entries.** [7], [22], [29] read "arXiv:XXXX, 2019". This is acceptable, but the current IEEE reference guide prefers "…," 2019, *arXiv:1909.06335*. Optional.
5. **Abbreviations and article numbers** are consistent: "in *Proc. X*", "PMLR vol. N", "Art. no.", and "doi:" placed last.

## 4. Status of round-3 findings

| Round-3 finding | Status now |
|---|---|
| Khraisat et al. year (2024 → 2025) | Resolved: the reference was removed |
| "E. M. Campos" → "E. Mármol Campos" | Fixed [6] |
| Ruzafa-Alcazar missing 7th author | Fixed (Skarmeta added), but the entry should now use "et al." (see [33]) |
| Hsieh "confirming" misattribution | Fixed (contribution 5 reworded) |
| Uncited companion studies | **Still open** (§3.1) |
| CICIoT2023 / Edge-IIoTset uncited | Fixed: cited as [37], [38] (but see §2 for their wording) |
| Optional LNCS volume [20], NeurIPS pages [35] | LNCS volume added. NeurIPS pages added, but they are unverifiable (see [35]) |
