# Text workflow data card

**All application text and all public PDFs in this directory are synthetic.** They are not freMTPL2 records and are not used to train or evaluate loss prediction.

- `applications.json`: 12 project-authored English development scenarios. Nine are scored for typed field extraction; three exercise conflict/document/time gates. `gold` is a hand-authored oracle, never provided to the extractor. Existing GLM fields, expected mileage, and usage appear in the examples; no real identities are present. Project BSD-3-Clause terms apply to these original fixtures.
- `example_input.json`: an input-only CLI example without gold answers.
- `public/`: two original, unchanged motor policy declaration PDFs from **Nikolai Sachok, Strata Insurance Corpus**; all people and policies are fictional. Corresponding `.txt` files are derived with pypdf 6.1.3, with LF line endings. Both originals and derived text are **CC BY 4.0**, not the repository's code license. Source links and checksums are in `manifest.json`.
- `manifest.json`: immutable source revision, original-file and derived-text hashes, pinned model-file hashes, attribution and questions. Vehicle labels follow upstream ground truth; dates follow the visible policy period. The questions and extraction evaluation are additions by this project.

Source: https://github.com/NikolaiSachok/strata-insurance-corpus/tree/c9ad2b0b4cdc4041bda304390988ee8ce8386091/sample

Data license: https://creativecommons.org/licenses/by/4.0/

Upstream data card: https://github.com/NikolaiSachok/strata-insurance-corpus/blob/c9ad2b0b4cdc4041bda304390988ee8ce8386091/docs/data-card.md

There is **no join** from Strata to freMTPL2. The PDFs are issued contracts used only for extraction. Their quoted premiums are not claim labels and do not enter the model. No claim reports or source identity images were downloaded.

The schema, questions and development fixtures were inspected during implementation; performance here is not a representative or independent test. Raw failures are retained in `reports/text_pricing/results.json`.
