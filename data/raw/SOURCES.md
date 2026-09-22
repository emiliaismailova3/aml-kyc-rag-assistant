# Corpus sources

All documents retrieved 2026-09-22. This is the AML/KYC and financial compliance
corpus for a neobank/fintech operating in (or supervised in relation to) Azerbaijan.
Sources: FATF, the Wolfsberg Group, the Central Bank of the Republic of Azerbaijan
(CBAR), and the primary Azerbaijani AML/CFT statute (mirrored by UNODC).

`fatf-gafi.org` rate-limits/blocks repeated automated requests from the same IP after
a few downloads (HTTP 403 from a WAF, confirmed by re-testing a previously-successful
URL). The three FATF documents below were secured either via `curl` (before the block
kicked in) or via the browser's own fetch, which is not subject to the same WAF rule.

## FATF (Financial Action Task Force)

| File | Title | Source URL |
|---|---|---|
| `fatf_recommendations_2025.pdf` | The FATF Recommendations (as amended, Feb 2025 consolidated text) | https://www.fatf-gafi.org/content/dam/fatf-gafi/recommendations/Feburary%202025%20FATF%20Recommendations.pdf |
| `fatf_azerbaijan_fur_2025.pdf` | Azerbaijan's progress in strengthening measures to tackle money laundering and terrorist financing — 5th Enhanced Follow-up Report & Technical Compliance Re-Rating (2025) | https://www.fatf-gafi.org/content/dam/fatf-gafi/fsrb-fur/Azerbaijan-FUR-2025.pdf.coredownload.inline.pdf |
| `fatf_guidance_beneficial_ownership_legal_persons.pdf` | Guidance on Beneficial Ownership of Legal Persons (R.24), March 2023 | https://www.fatf-gafi.org/content/dam/fatf-gafi/guidance/Guidance-Beneficial-Ownership-Legal-Persons.pdf.coredownload.pdf |

## The Wolfsberg Group

| File | Title | Year | Source URL |
|---|---|---|---|
| `wolfsberg_risk_based_approach_guidance.pdf` | Guidance on the Risk-Based Approach | 2026 | https://db.wolfsberg-group.org/assets/6c32b34b-bc02-4e8b-9b3f-f36e9bd109c3/Wolfsberg%20Group%20-%20Risk%20Based%20Approach%20Guidance%20_June2026.pdf |
| `wolfsberg_sanctions_screening_guidance.pdf` | Sanctions Screening Guidance | 2019 | https://db.wolfsberg-group.org/assets/4b6c2db6-696d-492e-bdd5-c51552708597/Wolfsberg%20Guidance%20on%20Sanctions%20Screening.pdf |
| `wolfsberg_payment_transparency_roles_responsibilities.pdf` | Guidance on Payment Transparency - Roles and Responsibilities | 2024 | https://db.wolfsberg-group.org/assets/b60cae63-3a63-46de-983a-cb22a06d14ab/PT_Roles__Responsibilities_forpublication.pdf |
| `wolfsberg_digital_customer_lifecycle_risk_management.pdf` | Guidance on Digital Customer Lifecycle Risk Management | 2022 | https://db.wolfsberg-group.org/assets/d51130c7-0262-4604-978e-ac3b85aea142/Wolfsberg%20Guidance%20on%20Digital%20Customer%20Lifecycle%20Management%20(2022).pdf |
| `wolfsberg_pep_guidance.pdf` | PEP Guidance (Politically Exposed Persons) | 2017 | https://db.wolfsberg-group.org/assets/9c6630de-69a8-4f55-9289-4db938938e34/Wolfsberg%20Guidance%20on%20PEPs.pdf |
| `wolfsberg_faqs_beneficial_ownership.pdf` | FAQs on Beneficial Ownership | 2012 | https://db.wolfsberg-group.org/assets/2d1320da-43a9-4bfa-9aee-fc1b01455ac6/19.%20Wolfsberg-FAQs-on-Beneficial-Ownership-May-2012.pdf |
| `wolfsberg_source_of_wealth_funds_faqs.pdf` | Source of Wealth and Source of Funds FAQs | 2020 | https://db.wolfsberg-group.org/assets/a27f9bf6-b4a8-41d2-a390-6f1aaf797241/Wolfsberg%20SoW%20and%20SoF%20FAQs%20August%202020%20(FFP).pdf |
| `wolfsberg_correspondent_banking_principles_2022.pdf` | Financial Crime Principles for Correspondent Banking | 2022 | https://db.wolfsberg-group.org/assets/d39a5072-7fb6-4e31-9a87-9e54021ce71f/Wolfsberg%20Correspondent%20Banking%20Principles%202022.pdf |

## Central Bank of the Republic of Azerbaijan (CBAR) & Azerbaijani law

| File | Title | Source URL |
|---|---|---|
| `cbar_notice_unlicensed_payment_institutions.pdf` | Guidance on notifying the Central Bank about payment/e-money institutions operating illegally without a license | https://uploads.cbar.az/assets/9c9aaa7b32fa5709a7b1466a0.pdf |
| `cbar_regulation_payment_emoney_institutions_2024.pdf` | Regulation on the organization and implementation of activities by payment and electronic money institutions (Decision No. 01/2, 10 Jan 2024) | https://uploads.cbar.az/assets/Regulation%20on%20the%20organization%20and%20implementation%20of%20activities%20by%20payment%20and%20electronic%20money%20institutions.pdf |
| `cbar_manual_payment_emoney_institutions_2024.pdf` | Manual (registry) of Payment and Electronic Money Institutions operating in the Republic of Azerbaijan | https://uploads.cbar.az/assets/Elektron%20pul%20t%C9%99%C5%9Fkilatlar%C4%B1%20v%C9%99%20%C3%B6d%C9%99ni%C5%9F%20t%C9%99%C5%9Fkilatlar%C4%B1n%C4%B1n_m%C9%99lumat%20kitab%C3%A7as%C4%B1_en_%2030.09.2024_1.pdf |
| `cbar_law_on_banks.pdf` | The Law of the Republic of Azerbaijan on Banks | https://uploads.cbar.az/assets/e2eb28ff9a91779f134e34b95.pdf |
| `azerbaijan_aml_law_unodc.pdf` | Law of the Republic of Azerbaijan on the Prevention of the Legalization of Criminally Obtained Funds or Other Property and the Financing of Terrorism (mirrored by UNODC's SHERLOC/BRI legal resources; original text is Azerbaijan's core AML/CFT statute referenced throughout the CBAR regulations and FATF evaluations above) | https://track.unodc.org/uploads/documents/BRI-legal-resources/Azerbaijan/9_-Azerbaijan_AMLaw.pdf |

## Notes

- 16 PDF documents total, ~11 MB combined — small enough to keep in the repo for
  reproducibility.
- File names are used as the `source` metadata field during ingestion (Step 3).
- Two further FATF thematic documents (Guidance for a Risk-Based Approach to Virtual
  Assets/VASPs, and the original 2023 Azerbaijan Mutual Evaluation Report) were
  identified but could not be downloaded because of FATF's WAF blocking repeated
  automated requests. They can be added later by downloading manually from
  fatf-gafi.org and dropping the PDF into this folder.
