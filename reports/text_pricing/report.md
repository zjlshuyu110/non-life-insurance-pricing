# Text → AI extraction → GLM: measured workflow demonstration

**Synthetic workflow demonstration, not evidence of better claims prediction.**

[Open the interactive trace](https://zjlshuyu110.github.io/non-life-insurance-pricing/text-to-pricing.html)

The local pretrained MiniLM reader selects answer spans; typed schema checks validate them. Four fields (driver age, vehicle age, bonus–malus and fuel) enter the existing Poisson × Gamma / Tweedie models. Five remaining model inputs are an explicit illustrative profile. Mileage and vehicle use are extracted but not priced. GLMs are not retrained.

## Recorded results

| Measure | Result |
|---|---:|
| Authored application scenarios | 12 |
| Priced / review / blocked | 6 / 4 / 2 |
| Validated field correctness, including abstentions | 54 / 54 |
| Present fields recovered | 44 / 44 |
| Absent or unsupported fields left empty | 10 / 10 |
| Raw public-PDF question exact matches | 3 / 6 |
| Maximum relative pricing difference vs gold inputs, priced cases only | 0.0% |
| CPU inference + handoff, excluding model loading | 2.20 s |

The nine scored applications are short, authored development fixtures, inspected during implementation. Three document-gate scenarios are excluded from field accuracy. These numbers are **not** an independent extraction accuracy estimate. The model and question wording were not fine-tuned; a usage-enum check was added after inspecting a development error. There is no test of real pricing uplift.

## Preserved failures and boundaries

- In `missing_extra`, the raw reader answers `diesel` for vehicle use with a score around 0.72. The schema rejects it. Scores are not calibrated correctness probabilities; raw answers and exact offsets remain in results.json.
- Public PDF answers remain uncorrected: a full period is returned instead of the start date and one vehicle is missed. These are issued synthetic contracts, used only for extraction. They are not pre-quote applications and cannot be joined to freMTPL2.
- Missing required pricing values and unsupported electric fuel stop handoff. Explicit conflicting age statements trigger a narrow rule; it is not a general contradiction detector.
- Input document type and timezone-aware availability timestamps are supplied by trusted ingestion. The metadata gate does not infer trustworthy provenance from arbitrary text.
- Changing only annual mileage from 8,000 to 40,000 keeps the existing model output unchanged. A new feature needs linked exposure/loss data and a new validated model before it can influence pricing.
- Age, car age and bonus ranges are conservative demo bounds; out-of-range values require review. English only. Unknown usage phrases remain empty under the explicit demo taxonomy.

## Case results

| Case | Decision | Frequency × severity (€ / year) |
|---|---|---:|
| commuter | priced | 98.49 |
| delivery | priced | 149.87 |
| retired | priced | 90.88 |
| young | priced | 346.36 |
| missing_bonus | review | — |
| missing_details | review | — |
| high_mileage | priced | 98.49 |
| missing_extra | priced | 98.49 |
| electric | review | — |
| conflict | review | — |
| claim | blocked | — |
| late | blocked | — |

## Reproduce

First reproduce the base GLM study; then:

```bash
python -m pip install -r requirements-neural.txt
python -m pip install -r requirements-text.txt
python scripts/prepare_text_demo.py --download-model
python -m nonlife.text_pricing
python scripts/build_text_showcase.py
```

For a new application: `python -m nonlife.text_pricing --input data/text_demo/example_input.json --output result.json`. New text runs locally; the website only browses recorded results. Do not publish private application text.

## Sources and audit

[Dataset assessment](../../docs/TEXT_DATASETS.md) · [Source manifest](../../data/text_demo/manifest.json) · [Application fixtures](../../data/text_demo/applications.json) · [Configuration](../../text_config.json) · [Full run JSON](results.json)

Public documents: Nikolai Sachok, Strata Insurance Corpus, CC BY 4.0. Two original PDFs are unchanged; pypdf 6.1.3 extracts their text. Local model: deepset/minilm-uncased-squad2, CC BY 4.0, pinned revision 934656cdda79824eabf503ed56e15c01ddbdbe3f. See THIRD_PARTY_NOTICES.md for attribution.

## 展示时的说法

我把 AI 接在信息提取环节，实际运行了开源语言模型，保留每个字段的原文证据，并用校验与人工复核入口控制错误。通过的字段进入已经训练好的 GLM。公开保险文档可用于抽取演示，但还没有核实到合适的真实投保文本与后续赔款配对数据，所以这个版本证明流程可运行，不声称定价精度提高。
