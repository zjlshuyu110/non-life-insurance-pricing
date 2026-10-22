# 公开文本数据筛选与 AI → GLM 集成

核实日期：2026-10-22。目标是寻找**投保前原始文本 + 可关联的后续赔款 + 暴露期**，检验文本新增信息是否改善非寿险定价。在本次核实的候选中，尚未找到满足这三个条件的合适真实开放数据集；这不是“任何地方都不存在”的断言。

已完成的项目入口：[Text → Pricing AI Lab](https://zjlshuyu110.github.io/non-life-insurance-pricing/text-to-pricing.html)。它实际运行本地神经语言模型抽取，并调用已有 GLM；是工作流程演示，不是文本带来真实定价提升的实验。

## 核实到的数据

| 数据 | 公开内容与性质 | 能否直接检验真实定价提升？ | 本项目用途 |
|---|---|---|---|
| [RISCBAC](https://huggingface.co/datasets/davebulaval/RISCBAC) / [RISC 论文](https://arxiv.org/abs/2304.04212) | 10,000 对英法合成车险合同，源于 Quebec 合同格式。数据 CC BY 4.0；生成器另有 LGPL 许可。 | 不能：不是原始投保叙述，没有真实后续赔款标签。合同内保费不是赔款目标。 | 可扩展双语抽取。未下载 714 MB 完整压缩包。 |
| [Strata Insurance Corpus](https://github.com/NikolaiSachok/strata-insurance-corpus) / [数据卡](https://github.com/NikolaiSachok/strata-insurance-corpus/blob/c9ad2b0b4cdc4041bda304390988ee8ce8386091/docs/data-card.md) | 虚构保险公司的合成保单、理赔和知识文档；结构化实体与问答标签。数据 CC BY 4.0，生成代码 MIT。 | 不能：虽有跨文档关联键与合成赔款，风险与赔款来自生成过程，不是真实经验。已签发保单也不自动等于报价前可获得材料。 | 已下载并运行两个原始车险 PDF 的抽取；未与 freMTPL2 强行拼接。 |
| [QuoteSweep Stated-Appetite Corpus](https://www.quotesweep.com/research/observed-appetite) / [DOI](https://doi.org/10.5281/zenodo.20280436) | 509 家商业财险公司的公开承保偏好资料；发布方标注 CC BY 4.0。 | 不能：这是公司层面的承保知识，不是保单层面的投保文本与赔款。论文的生产级 observed-appetite 测量仍列为后续工作。 | 可另做承保规则检索/RAG；本次未加入模型。 |
| [ascentTraining::policy_data](https://search.r-project.org/CRAN/refmans/ascentTraining/html/policy_data.html) | 926 行模拟表格，含 Mileage、Usage、GrossIncurred、Exposure，无原始文本。 | 不能验证真实信息增益：模拟数据；把表格改写成文本也没有增加新信息。 | 可做独立的模拟实验；本次未复制或使用其数据。 |
| [freMTPL2 / CASdatasets](https://github.com/dutangc/CASdatasets/tree/c49cbbb37235fc49616cac8ccac32e1491cdc619/data) | 本项目已有的真实法国车险结构化特征、赔款和暴露期。 | 能检验结构化模型；不能验证原始文本特征，因没有原始文本。 | 继续提供已训练的 GLM，不修改原有 GLM/MLP 对比结果。 |

不同数据集的保单编号不共享语义，不能根据行号、相似年龄、车型或随机匹配拼出“文本 + 真实赔款”。跨国家、保障责任与定价年代也不能被这种拼接消除。

## 本次实际加入的两条路径

1. **公开文档抽取**：Strata 两份未修改的 synthetic motor declarations PDF → pypdf 提取文本 → MiniLM 回答车型、起保日、终保日 → 原始答案与预期答案逐条展示。每份 PDF 通过上游 manifest 的 SHA-256 校验；原文件、文本、来源和许可保存在 data/text_demo。这是公开样本，但不是现实客户数据。
2. **完整工作流**：12 份本项目编写的英文合成投保文本 → 文件类型与可得时间检查 → MiniLM 抽取 → 原文位置、分数与字段校验 → 4 个已有定价字段 + 5 个明确给定的固定档案字段 → 既有 Poisson × Gamma / Tweedie 模型。完整字段缺失时不补默认值；对应案例送复核。

第二条路径中的文本是作者为展示编写的，并非 RISCBAC/Strata 原始投保文本，也不是对真实测试保单的改写。没有把公开合同伪装成申请书。它测量“是否正确送入已有模型”，不测量“是否更准确预测赔款”。

## AI 在哪里

[deepset/minilm-uncased-squad2](https://huggingface.co/deepset/minilm-uncased-squad2) 是经过 SQuAD 2.0 问答训练的神经语言模型，约 33.4M 参数，英文，CC BY 4.0。它不是规则匹配，也不是生成式聊天大模型：它从文本中选择答案跨度，或给出无答案。模型权重约 133 MB，本机 CPU 推理，不需要 API key。

字段问题与阈值见 [text_config.json](../text_config.json) 和 [nonlife/text_pricing.py](../nonlife/text_pricing.py)。模型选出跨度后，代码负责数字/燃料/用途枚举校验，不替模型另找答案。保留原文证据能便于审计，但不能证明语义一定正确。分数也不是经过校准的正确率。

| 字段 | 当前处理 |
|---|---|
| DrivAge、VehAge、BonusMalus、VehGas | 抽取、校验后进入已有 GLM |
| VehBrand、VehPower、Region、Area、Density | 来自页面明确展示的示例档案，非 AI 抽取，也不声称来自公开合同 |
| AnnualKm、Usage | 展示原文和抽取结果；当前 GLM 不使用，无价格影响 |
| ClaimAmount、ClaimNb、事后事故描述 | 不进入定价输入；标为理赔文档或报价后才可得的材料在入口阻断 |

材料类型和时间戳来自可信接收流程，不从语言模型输出获取；演示不验证任意输入者是否伪造这些元数据。冲突年龄规则仅覆盖明确的有限表达，不能当作通用矛盾检测器。实际部署仍需要人工复核和更全面验证。

## 实测与失败案例

见 [完整运行报告](../reports/text_pricing/report.md) 和 [原始记录](../reports/text_pricing/results.json)。

- 12 个场景覆盖正常、缺失、超出支持的燃料、矛盾、事后材料和晚到材料。
- 九个短文本开发样例的字段正确率不是独立泛化评估。实现中检查过这些文本，并在发现错误后增加了用途枚举校验。
- 缺少用途时，原始模型返回 diesel，分数约 0.72；校验拒绝该值并保留原始输出。这说明高分并不保证语义正确。
- 两份公开 PDF 的起保日期可能被抽成完整区间；一个车型未回答。未修改这些失败样本或将它们剔除。
- “8,000 → 40,000 公里”案例的其他字段一致，两个模型的输出也一致。不能人为加一个里程系数，然后宣称它是从真实数据学到的。

## 怎样才能回答“AI 提升定价了吗”

需要收集有合法使用权限的同一业务数据：保单/投保关联键、原始申请文本、文档类型与可得时间、报价时点、结构化基准特征、后续保期内赔款，以及对应暴露期。只从当时已知材料提取预定义特征；缺失与抽取失败也要保留。

在未触碰的时间外样本上比较：现有结构化 GLM vs 同样建模流程但加入文本抽取特征的 GLM。模型、阈值、特征选择与超参数只在训练/验证阶段确定。使用 exposure 加权 deviance、总体及分组校准、风险排序，并报告抽取错误、人工复核比例和不确定性。原项目当前测试集已被查看，不能继续调参再把它称为新的独立验证。

纯文本再表格化带来的流程效率，可以通过每单处理时间、人工修正量和可追溯性衡量；真实风险信息增益必须由上述关联数据实验回答。

## 展示建议

先选 commuter 展示原文高亮与四个 GLM 输入；再选 missing_bonus 展示拒绝缺失字段；选 missing_extra 展示高分错误被取值校验拦截；点击 mileage 对照解释尚未使用的新特征；最后展示公开 PDF 的成功与失败，以及数据缺口。

可以说：“我让 AI 负责把非结构化材料转为有证据的字段，GLM 保留为当前经过验证的定价核心。这个版本已经复现流程；真实定价提升仍需要关联业务数据验证。”

