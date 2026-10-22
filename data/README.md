# Data card

**Datasets:** freMTPL2freq and freMTPL2sev; French motor third-party liability policies and claim amounts.

**Catalogue:** OpenML IDs [41214](https://www.openml.org/d/41214) and [41215](https://www.openml.org/d/41215).

**Actual default download:** original authors' [CASdatasets historical snapshot](https://github.com/dutangc/CASdatasets/tree/c49cbbb37235fc49616cac8ccac32e1491cdc619/data), commit `c49cbbb37235fc49616cac8ccac32e1491cdc619`. This was selected because OpenML could not be reached. We do not claim byte equality with the OpenML export.

`manifest.json` records exact URLs, row counts, columns, original R-file hashes and local CSV hashes. `nonlife.data` checks the local hashes on each run. Numeric IDs are validated before joining, claims are aggregated by ID, and unmatched severity IDs are reported.

## Fields used

| Field | Meaning |
|---|---|
| IDpol | Policy identifier, used to join tables, excluded from predictors |
| ClaimNb | Claim count, target component |
| Exposure | Covered time in years, model/evaluation weight |
| ClaimAmount | Claim amount in EUR, aggregated to the policy level |
| VehPower, VehAge, VehBrand, VehGas | Vehicle attributes |
| DrivAge | Driver age |
| BonusMalus | Bonus-malus score |
| Area, Density, Region | Geographic descriptors |

## Storage and terms

Raw `.rda` and `.csv.gz` files are downloaded locally, not committed. The source CASdatasets package is distributed under GPL (>= 2); original data terms and attribution remain separate from this project's code license. Consult the upstream package documentation when redistributing source data. Outputs committed here are aggregated analyses, not the source tables.

No synthetic replacement is used for the real-policy GLM/MLP pricing results. Unit-test fixtures and the separate `text_demo/` workflow demonstration are explicitly synthetic. The latter contains authored application scenarios and two openly licensed public synthetic PDFs; none is used to fit or evaluate the claims-pricing models. See [text-source assessment](../docs/TEXT_DATASETS.md) and [text-demo data card](text_demo/README.md).
