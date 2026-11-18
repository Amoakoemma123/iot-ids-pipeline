# Paper 4 v2 — Reference Verification, Round 3

Checked 2026-10-25 against `Manuscript_filled.md` (42 references). I checked every DOI against the Crossref REST API (`api.crossref.org/works/<doi>`), and every entry without a DOI against Semantic Scholar (graph API), the arXiv API and the PMLR proceedings pages (volume and page ranges). I compared each citing sentence with the source's abstract, title and known content. No other file was modified.

**Bottom line:** none of the 42 references is fabricated and all exist. **38 OK, 3 wrong metadata (all minor), 1 misattributed (one sentence), 0 not found.** Numbering is clean: every [1]-[42] is cited, every citation has an entry, and first citations run in strict numeric order (1, 2, ..., 42). All of the round-1 faults are fixed.

## 1. Per-reference table

| # | Reference (short) | Status | Correction / note |
|---|---|---|---|
| 1 | Diro & Chilamkurti, FGCS 82:761-768, 2018 | OK | Title now complete. DOI verified. |
| 2 | Agrawal et al., Comput. Commun. 195:346-361, 2022 | OK | 9 authors, so "et al." is correct. |
| 3 | Voigt & von dem Bussche, GDPR Practical Guide, Springer 2017 | OK | The CCPA claim has been removed, and "privacy regulation" is now supported. |
| 4 | McMahan et al., FedAvg, AISTATS, PMLR 54:1273-1282, 2017 | OK | Full authors per arXiv:1602.05629. PMLR pages verified. |
| 5 | Nguyen et al., DÏoT, ICDCS 2019, pp. 756-767 | OK | Claim corrected to device-type-specific GRU models, which matches the paper. |
| 6 | Khraisat et al., ACM CSUR 57(1):1-38, **2024** | Wrong metadata (year) | Vol. 57, no. 1 is the **January 2025** issue (Crossref published-print 2025-01-31; online 2024-10-07). IEEE style uses the issue year, so change to **2025**. |
| 7 | Hernandez-Ramos et al., ACM CSUR 57(12):1-65, 2025 | OK | DOI 10.1145/3731596 verified. |
| 8 | **E. M. Campos** et al., Comput. Netw. 203, 108661, 2022 | Wrong metadata (author name, minor) | The first author is Enrique **Mármol Campos**, a compound surname. Use "E. Mármol Campos et al.". [39] spells the same person "E. Marmol-Campos", so the list is inconsistent. The claim (FL-IDS evaluated on ToN_IoT under non-IID splits) is supported. |
| 9 | Vyas et al., IEEE Access 12:127018-127050, 2024 | OK | |
| 10 | Hsu, Qi & Brown, arXiv:1909.06335, 2019 | OK | Correct source for Dirichlet partitioning. This was the fix for round-1 item 14. |
| 11 | Q. Li, Diao, Chen & He, ICDE 2022, pp. 965-978 | OK | First author corrected. The "no algorithm dominates" claim matches the paper's main finding. |
| 12 | Zhao et al., arXiv:1806.00582, 2018 | OK | Supports the claim that FedAvg degrades under label skew. |
| 13 | Abadi et al., CCS 2016, pp. 308-318 | OK | DP-SGD with per-example clipping, Gaussian noise and Poisson ("lot") sampling, which matches §IV-B. |
| 14 | McMahan, Ramage, Talwar & Zhang, ICLR 2018 | OK | Now cited for the record-level vs client-level distinction, which is correct (user-level DP). The earlier per-sample misstatement is gone. |
| 15 | Alsaedi et al., TON_IoT telemetry, IEEE Access 8:165130-165150, 2020 | OK | |
| 16 | Moustafa, Network TON_IoT, Sustain. Cities Soc. 72, 102994, 2021 | OK | New entry. This is the correct citation for the network subset used here. |
| 17 | Moustafa & Slay, UNSW-NB15, MilCIS 2015, pp. 1-6 | OK | Pages and DOI added. The nine attack categories are correct. |
| 18 | T. Li et al., FedProx, MLSys vol. 2, pp. 429-450, 2020 | OK | |
| 19 | Reddi et al., Adaptive federated optimization, ICLR 2021 | OK | β1 = 0.9 and β2 = 0.99 are as in the paper. τ is a tuned value there, which is fine. |
| 20 | Karimireddy et al., SCAFFOLD, ICML, PMLR 119:5132-5143, 2020 | OK | Pages verified on PMLR. The option II update and the unweighted server control variate match the paper. |
| 21 | J. Zhang et al., FedLC, ICML, PMLR 162:26311-26329, 2022 | OK | New entry. 7 authors, so "et al." is correct. Pages verified on PMLR (zhang22p). |
| 22 | Hsieh et al., Non-IID data quagmire, ICML, PMLR 119:4387-4398, 2020 | Misattributed (one sentence) | Metadata verified. §II-B is accurate: BN is a failure point under label skew and GroupNorm is recommended. Contribution 4 ("**confirming** Hsieh et al. [22] for IDS models: batch normalisation reaches similar accuracy at its best round, but its running statistics make the evaluated model swing") is not what Hsieh show. They report an *accuracy loss* of BN under skew, caused by cross-client statistic mismatch. The manuscript instead finds similar best-round accuracy, and a swing that also occurs in the **centralised** model (§V-D: "not specific to federation"). Reword, e.g. "in line with Hsieh et al.'s recommendation of group normalisation [22], we find that ...", or "extending". |
| 23 | Wu & He, Group normalization, ECCV, LNCS, 2018, pp. 3-19 | OK | Optional: add the LNCS volume ("vol. 11217"). |
| 24 | Mironov, Rényi differential privacy, CSF 2017, pp. 263-275 | OK | Title corrected. |
| 25 | Mironov, Talwar & Zhang, RDP of the sampled Gaussian mechanism, arXiv:1908.10530, 2019 | OK | New entry. This is exactly the Poisson-subsampled Gaussian accountant described in §IV-B. |
| 26 | Rey et al., Comput. Netw. 204, 108693, 2022 | OK | Claim corrected to N-BaIoT with robust aggregation, which matches the paper. |
| 27 | Lazzarini, Tianfield & Charissis, AI 4(3):509-530, 2023 | OK | The abstract confirms a comparison of FedAvg, FedAvgM, FedAdam and FedAdagrad on ToN_IoT/CICIDS2017. |
| 28 | Popoola et al., IEEE TCE 70(1):1610-1622, 2024 | OK | |
| 29 | Olanrewaju-George & Pranggono, Cyber Secur. Appl. 3, 100068, 2025 | OK | The title confirms the supervised and unsupervised comparison. Vol. 3 is the 2025 volume (online 2024). |
| 30 | Peng, Wu & Xiao, FD-IDS, Sensors 25(14), 4309, 2025 | OK | The abstract confirms non-IID handling with KD plus a proximal term. |
| 31 | Nguyen, Rieger, Miettinen & Sadeghi, Poisoning attacks on FL-based IoT IDS, DISS 2020 | OK | DOI 10.14722/diss.2020.23003 verified. |
| 32 | Nguyen et al., FLAME, USENIX Security 2022, pp. 1415-1432 | OK | 13 authors. Pages verified. |
| 33 | Acar et al., FedDyn, ICLR 2021 | OK | 6 authors, listed in full and correctly. |
| 34 | Q. Li, He & Song, MOON, CVPR 2021, pp. 10708-10717 | OK | DOI and pages verified. |
| 35 | X. Li et al. (5 authors), FedBN, ICLR 2021 | OK | Authors match arXiv:2102.07623. |
| 36 | T. Li, Hu, Beirami & Smith, Ditto, ICML, PMLR 139:6357-6368, 2021 | OK | Pages verified on PMLR (li21h). |
| 37 | Fu et al., DP-FL systematic review, ACM CSUR 58(11):1-38, 2026 | OK | Verified. Supports the record-level vs client-level taxonomy. The "needs many clients" claim is carried mainly by [38], and [37] is adequate as co-support. |
| 38 | Wei et al., IEEE TIFS 15:3454-3469, 2020 | OK | Supports the claim that client-level DP utility improves with the number of clients. |
| 39 | Ruzafa-Alcazar et al., IEEE TII 19(2):1145-1154, 2023 | Wrong metadata (missing author) | Crossref lists **7** authors. The entry names 6 with no "et al." and omits **A. F. Skarmeta**. Add him, or use "P. Ruzafa-Alcazar et al." (IEEE uses "et al." for more than 6 authors). The same omission was in round-1 suggestion S1. |
| 40 | Friha et al., 2DF-IDS, Comput. Secur. 127, 103097, 2023 | OK | 6 authors, complete. |
| 41 | Bagdasaryan & Shmatikov, NeurIPS vol. 32, 2019 | OK | Supports the claim that DP hurts under-represented classes more. Optional: add pp. 15479-15488. |
| 42 | Ponomareva et al., How to DP-fy ML, JAIR 77:1113-1201, 2023 | OK | DOI 10.1613/jair.1.14649 verified. |

## 2. Status of round-1 findings

| Round-1 finding | Status now |
|---|---|
| Diro title shortened | Fixed [1] |
| GDPR book cited for CCPA | Fixed: CCPA removed [3] |
| McMahan 2017 initials / PMLR vol. | Fixed [4] |
| TON_IoT network subset uncited | Fixed: Moustafa 2021 added [16] |
| UNSW-NB15 pages/DOI | Fixed [17] |
| FedProx / SCAFFOLD volume and pages | Fixed [18], [20] |
| Li X. vs Li Q. (ICDE 2022) | Fixed [11] |
| Hsieh cited for Dirichlet partitioning | Fixed: Hsu et al. [10] added. Hsieh is now used for BN, but see the new wording issue at [22] |
| Rey et al. CICIDS-2017 / "within 5%" | Fixed [26] |
| DÏoT "LSTM, non-IID degradation" | Fixed [5] |
| Mothukuri survey misused | Fixed: removed; replaced by FL-IDS surveys [6], [7] |
| McMahan 2018 "per-sample clipping" | Fixed [14] |
| Flower [19] never cited | Fixed: removed (Flower is no longer mentioned) |
| Mironov wrong title | Fixed [24]; Mironov et al. 2019 added [25] |
| Hernandez-Ramos DOI / hyphenation | Fixed [7] |
| "Ye, M." wrong authors (arXiv:2411.12377) | Fixed: reference removed |
| Placeholder self-citation [7] | Removed (but see 3.1 below) |
| Numbering out of order / entry after Data Availability | Fixed: 1-42 in citation order, all above Declarations |
| "(Emmanuel, 2026)" author-year citation | Fixed: removed |
| Dead Zenodo DOI | Fixed: replaced by the GitHub URL |

## 3. Other observations

1. **Uncited self-reference.** §III-A says "the corrected pipeline of our companion studies", with no citation. Cite the companion paper(s), as a preprint or "under review", or drop "of our companion studies".
2. **Named datasets without citations** (§VI-C). Both were verified on Crossref:
   - CICIoT2023: E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, Art. no. 5941, 2023, doi: 10.3390/s23135941.
   - Edge-IIoTset: M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, and H. Janicke, "Edge-IIoTset: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning," *IEEE Access*, vol. 10, pp. 40281-40306, 2022, doi: 10.1109/ACCESS.2022.3165809.
3. **IEEE style consistency.** Formatting is consistent: abbreviated journal names, "in *Proc. X*", "PMLR vol. N", "Art. no." for article numbers, and "doi:" at the end. The use of "et al." is consistent (full list for 6 or fewer authors, "et al." for more than 6), except for [39]. The only remaining differences are optional items: the LNCS volume for [23], NeurIPS pages for [41], and a location for the DISS workshop [31].
4. **In-text citation format.** All citations are numeric, and ranges use an en dash ("[1]–[3]", "[5]–[9]"). There are no author-year citations left.
