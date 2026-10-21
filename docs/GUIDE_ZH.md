# Non-life 车险定价项目：阅读与复现说明

这个项目回答：给定驾驶人和车辆特征，每单位保险暴露的预期赔款是多少？

建议先打开 [高级交互工作台](https://zjlshuyu110.github.io/non-life-insurance-pricing/)，再读 `reports/report.md` 的实际结果，最后用 notebook 和源码理解计算过程。原版分析报告仍保留在 `reports/dashboard.html`。

新版工作台可以旋转三维模型曲面、调整年龄和 bonus–malus、联动筛选真实测试集、点击地区热力图，并对比不同赔款封顶情景。三维图是固定其他特征后的模型预测；测试集图表是真实保单汇总，页面中分别标注。直接打开 `docs/index.html` 也可离线使用。

## 两条定价路线

1. Poisson 模型预测出险频率，Gamma 模型预测案均赔款，两者相乘得到年化纯保费。
2. Tweedie 模型直接预测年化纯保费。

Exposure 是承保时间，0.5 表示半年；必须纳入模型权重。这里的纯保费只覆盖预期赔款，未加入费用、利润、税费或未来通胀。

## 怎么运行

在项目文件夹打开终端。当前电脑已经建好了 `.venv`：

```powershell
.\.venv\Scripts\python.exe -m nonlife.run --sensitivity
.\.venv\Scripts\python.exe -m pytest -q
```

在其他电脑，按根目录 README 创建环境、安装 `requirements-lock.txt`，再先运行 `python -m nonlife.run --download-only`。

## 图应该怎么看

- **Calibration**：实际赔款与预测赔款是否接近。总体正确不代表每个年龄组都正确。
- **Deviance**：在相同目标口径和 power 下越低越好。不能拿不同封顶情景的 deviance 直接比较优劣。
- **Gini / Lorenz**：模型能否区分高低风险，不代表预测金额准确。
- **Sensitivity**：赔款封顶会改变预测目标，也可能改变两个模型的相对表现。

原教程把每份保单的累计赔款封顶到 20 万欧元，并调整 exposure 和赔案数。项目保留这些规则以复现方法，同时记录修改了多少行、减少了多少赔款。封顶后的结果不能代表完整尾部风险。

## 数据来源和复现边界

OpenML 下载接口在本次运行时超时，因此使用原作者 CASdatasets 仓库的固定历史版本，并保存下载地址、提交号和 SHA-256。这属于方法复现，未宣称与 OpenML 文件逐字节一致。

项目只用训练集学习分箱、类别和标准化参数。模型参数沿用基准，不根据测试集挑选。因此数值不要求与网页示例完全相同。测试集按 policy ID 随机划分；不同 ID 下潜在的重复风险以及时间外验证仍是局限。

## 面试或展示时可以讲的内容

先解释业务问题，然后解释 exposure、频率与赔款强度为何分开建模，再展示测试集校准及封顶敏感性。最后说明局限和下一步验证。README 已明确标注参考来源和新增部分，可以据此准确描述自己的贡献。

适合后续拓展：分组交叉验证、bootstrap 不确定性区间、GLM 与 boosting 对比、赔款尾部单独建模。先把当前方法和结果讲清楚，再增加复杂模型。
