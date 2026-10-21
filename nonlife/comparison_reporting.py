"""Generate a portable model-comparison report from measured experiment outputs."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import sha256

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"Baseline": "Portfolio mean", "FrequencySeverity": "Poisson x Gamma",
          "Tweedie": "Reference Tweedie", "TunedTweedie": "Tuned Tweedie",
          "MLP_seed0": "MLP / seed 0", "MLP_seed1": "MLP / seed 1", "MLP_seed2": "MLP / seed 2"}
MAIN = ["FrequencySeverity", "Tweedie", "TunedTweedie", "MLP_seed0"]
COLORS = {"Baseline": "#81949f", "FrequencySeverity": "#ed966a", "Tweedie": "#cfb57a",
          "TunedTweedie": "#7699e7", "MLP_seed0": "#45d6c0", "MLP_seed1": "#45d6c0", "MLP_seed2": "#45d6c0"}


def build_report(output: Path):
    scores = pd.read_csv(output / "metrics.csv")
    test = scores.query("split == 'test'").set_index("model")
    history = pd.read_csv(output / "learning_curves.csv")
    search = pd.read_csv(output / "validation_search.csv")
    intervals = pd.read_csv(output / "paired_bootstrap.csv")
    meta = json.loads((output / "run_metadata.json").read_text())
    selected = meta["selection"]
    names = [f"MLP_seed{s}" for s in meta["experiment"]["final_seeds"]]
    seed_results = test.loc[names]
    seed_results[["deviance_p1.9", "gini", "predicted_actual"]].agg(["mean", "std", "min", "max"]).to_csv(output / "seed_summary.csv")
    glm = test.loc["TunedTweedie"]
    primary = test.loc["MLP_seed0"]
    difference = primary["deviance_p1.9"] - glm["deviance_p1.9"]
    gain = -difference / glm["deviance_p1.9"] * 100
    interval = intervals.set_index("comparator").loc["TunedTweedie"]
    winner = "MLP" if difference < 0 else "Tuned Tweedie GLM"
    direction = "lower" if difference < 0 else "higher"
    ci_text = ("The interval crosses zero; this run does not establish a deviance advantage."
               if interval.ci95_low <= 0 <= interval.ci95_high else
               f"The conditional interval favors {'MLP' if interval.ci95_high < 0 else 'the tuned GLM'} on this holdout.")
    headline = ("The plain MLP improves on the tuned Tweedie GLM." if difference < 0
                else "The plain MLP does not beat the GLM benchmark.")
    verdict = (f"The predeclared primary MLP (seed 0) has {abs(gain):.2f}% {direction} weighted Tweedie deviance "
               f"than the validation-tuned GLM. {ci_text}")
    observations = {
        "primary_comparison": "MLP seed 0 versus validation-tuned Tweedie GLM",
        "primary_metric": "exposure-weighted Tweedie deviance, p=1.9",
        "mlp_improvement_percent": float(gain), "difference": float(difference),
        "conditional_ci95": [float(interval.ci95_low), float(interval.ci95_high)],
        "winner_by_point_estimate": winner,
        "mlp_seed_deviance_range": [float(seed_results["deviance_p1.9"].min()), float(seed_results["deviance_p1.9"].max())],
        "mlp_seed_gini_range": [float(seed_results.gini.min()), float(seed_results.gini.max())],
    }
    (output / "conclusions.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
    table_header = "| Model | Test deviance p=1.9 ↓ | Gini ↑ | Expected / observed | Train deviance |\n|---|---:|---:|---:|---:|"
    train = scores.query("split == 'train'").set_index("model")
    table_rows = [f"| {LABELS[name]} | {test.loc[name, 'deviance_p1.9']:.6f} | {test.loc[name, 'gini']:.4f} | {test.loc[name, 'predicted_actual']:.2%} | {train.loc[name, 'deviance_p1.9']:.6f} |" for name in LABELS]
    ci_rows = [f"| {LABELS[r.comparator]} | {r.deviance_difference_mlp_minus_glm:.6f} | [{r.ci95_low:.6f}, {r.ci95_high:.6f}] |" for r in intervals.itertuples()]
    md = f"""# GLM versus ordinary neural networks / 实测对比

{headline} {verdict}

**中文结论：** 主比较中，MLP（预先固定 seed 0）的测试集 deviance 比调参后的 Tweedie GLM {'低' if gain > 0 else '高'} **{abs(gain):.2f}%**。这不是神经网络一定更强或更弱的普遍结论；仅针对当前数据、特征、训练预算和随机留出集。

{table_header}
{chr(10).join(table_rows)}

Deviance 越低越好；Gini 衡量风险排序，越高越好；Expected / observed 越接近 100% 越好。它们回答不同的问题，不能互相替代。金额误差 MAE 和 p=1.5 deviance 也保存在 `metrics.csv`。

![Measured model comparison](comparison.png)

## Experimental protocol

- **Data:** all {meta['train_policies'] + meta['test_policies']:,} policies, the same source hashes and EUR 200,000 adjusted-loss cap as the existing study.
- **Outer split:** {meta['train_policies']:,} training / {meta['test_policies']:,} test policies, original seed 0. Existing GLM predictions were reproduced and checked numerically.
- **Selection split:** {meta['inner_fit_policies']:,} inner fit / {meta['validation_policies']:,} validation, seed 42, both entirely inside the original training sample.
- **Features:** identical {meta['feature_count']} binned / one-hot / continuous features. MLP inputs additionally receive train-only scaling. No policy ID, claim amount, claim count or exposure is used as a predictor. Exposure weights the loss.
- **Reference GLMs:** original Poisson x Gamma and Tweedie settings remain unchanged. An additional Tweedie GLM selects alpha from {meta['experiment']['glm_alphas']} using validation deviance; selected alpha = **{selected['glm_alpha']}**. The search is small and the optimum may lie outside this grid.
- **Ordinary MLP:** two fully connected ReLU layers; positive log-link output; exposure-weighted Tweedie objective with p=1.9. No pretrained network, attention, GLM offset or ensemble. Target scaling preserves the loss optimum. Log-relative predictions are guarded to [-8, 8] for numerical stability.
- **Selection:** widths {meta['experiment']['mlp_candidates']}, AdamW learning rate {meta['experiment']['learning_rate']}, weight decay {meta['experiment']['weight_decay']}, batch size {meta['experiment']['batch_size']}, gradient clipping {meta['experiment']['gradient_clip']}, at most {meta['experiment']['max_epochs']} epochs, patience {meta['experiment']['patience']}. Widths and epoch are selected only on validation deviance.
- **Selected network:** **{' → '.join(map(str, [meta['feature_count'], *selected['mlp_widths'], 1]))}**, {meta['final_mlp_runs'][0]['parameters']:,} parameters, **{selected['mlp_epochs']} epochs**. It is refitted from scratch on all outer training policies with seeds 0, 1 and 2 at the same epoch budget. Seed 0 is the primary result, not the best test seed. No post-hoc calibration is applied.
- **Runtime:** {meta['elapsed_seconds']:.1f} seconds on CPU for selection, refits, evaluation and bootstrap; individual neural fitting times are in `training_runs.csv`. These timings are hardware-specific and include differing diagnostic work.

## Stability and uncertainty

Three MLP seeds produced test deviance from **{seed_results['deviance_p1.9'].min():.6f} to {seed_results['deviance_p1.9'].max():.6f}**, and Gini from **{seed_results.gini.min():.4f} to {seed_results.gini.max():.4f}**. These are three runs, not a confidence interval.

Paired bootstrap of {meta['experiment']['bootstrap_replicates']:,} test-policy samples, with models held fixed. Difference is **MLP seed 0 minus GLM**; negative favors MLP.

| Comparator | Deviance difference | Conditional 95% percentile interval |
|---|---:|---:|
{chr(10).join(ci_rows)}

{ci_text} These intervals assume IID policies, ignore repeated insured entities and training uncertainty, and are exploratory without multiple-comparison adjustment. The primary comparison is against the validation-tuned Tweedie GLM. They do not establish performance in a future year.

## Interpretation / 面试可以怎么讲

1. Explain the frequency–severity decomposition and why exposure is a weight rather than an ordinary input.
2. Compare GLM with a plain MLP under the **same Tweedie objective and target**, rather than fitting a generic MSE network to a mostly-zero heavy-tailed target.
3. Distinguish financial calibration, ranking and predictive loss. A model may improve one and worsen another.
4. Show the validation curves and all three seeds. Additional network capacity does not guarantee improved generalization, especially with only a small fraction of positive claims.

The test set has now been used for this comparison. Further model changes should be evaluated using a new holdout or nested / temporal validation; repeatedly using these test results to tune would contaminate the benchmark. The current experiment uses one random split, a narrow architecture / regularization search and an adjusted loss target, so it is not a broad benchmark of all neural models.

## Reproduce

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-neural.txt
python -m nonlife.run --download-only
python -m nonlife.run --sensitivity
python -m nonlife.compare_neural
python -m pytest -q
```

To regenerate only the plots and page from committed aggregate results: `python -m nonlife.compare_neural --report-only` (PyTorch environment required by the CLI). Local model binaries and policy-level predictions are Git-ignored. The generated page is `docs/model-comparison.html`.

Sources: [scikit-learn insurance example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html), [Tweedie deviance definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_tweedie_deviance.html), [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html). Values in this report come from this repository's actual run, not from those reference pages.
"""
    (output / "report.md").write_text(md, encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#153149",
                         "axes.labelcolor": "#153149", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), layout="constrained")
    fig.suptitle("GLM vs ordinary MLP | measured on the same holdout\n169,504 policies · exposure-weighted · no test-set tuning", fontsize=19, fontweight="bold")
    y = np.arange(len(MAIN))
    ax = axes[0, 0]
    for i, name in enumerate(MAIN):
        value = test.loc[name, "deviance_p1.9"]
        ax.scatter(value, i, s=95, color=COLORS[name], edgecolor="#153149", lw=.5)
        ax.annotate(f" {value:.4f}", (value, i), xytext=(5, 6), textcoords="offset points", fontsize=9)
    ax.set(yticks=y, yticklabels=[LABELS[m] for m in MAIN], title="A | Predictive loss (lower is better)", xlabel="Tweedie deviance, p = 1.9", ylim=(-.6, 3.7))
    values = test.loc[MAIN, "deviance_p1.9"]
    pad = max(.08, np.ptp(values) * .3)
    ax.set_xlim(values.min() - pad, values.max() + pad * 2)
    ax.grid(axis="x", alpha=.15)
    ax = axes[0, 1]
    for i, name in enumerate(MAIN):
        ax.scatter(100 * test.loc[name, "predicted_actual"], i, s=95, color=COLORS[name], edgecolor="#153149", lw=.5)
    ax.axvline(100, color="#153149", ls="--", lw=1)
    ax.set(yticks=y, yticklabels=[LABELS[m] for m in MAIN], title="B | Portfolio calibration (target = 100%)", xlabel="Expected / observed loss (%)", ylim=(-.6, 3.7))
    ax.grid(axis="x", alpha=.15)
    ax = axes[1, 0]
    lorenz = pd.read_csv(output / "lorenz.csv")
    ax.plot([0, 1], [0, 1], "--", color="#81949f", lw=1)
    for name in MAIN:
        line = lorenz.query("model == @name")
        ax.plot(line.exposure_share, line.loss_share, color=COLORS[name], lw=2,
                label=f"{LABELS[name]}  {test.loc[name, 'gini']:.3f}")
    ax.set(title="C | Risk ranking (legend shows Gini)", xlabel="Cumulative exposure: low to high risk", ylabel="Cumulative share of observed losses")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 1]
    for tag, group in history.query("stage == 'selection'").groupby("run"):
        ax.plot(group.epoch, group.validation_deviance, lw=2, label=f"MLP {tag}")
    val_glm = search.query("family == 'TweedieGLM'").validation_deviance.min()
    ax.axhline(val_glm, color=COLORS["TunedTweedie"], ls="--", label="Best validation GLM")
    ax.set(title="D | Validation-only model selection", xlabel="Epoch", ylabel="Validation deviance, p = 1.9")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)

    plots = []
    def add_plot(title, subtitle, traces, layout=None):
        figure = go.Figure(traces)
        settings = dict(template="plotly_dark", paper_bgcolor="#0c1e2b", plot_bgcolor="#0c1e2b",
                        font=dict(family="Segoe UI, Arial", color="#e7eff2"), height=365,
                        margin=dict(l=65, r=35, t=35, b=75), legend=dict(orientation="h", y=-.24),
                        hovermode="closest")
        figure.update_layout(**(settings | (layout or {})))
        figure.update_xaxes(gridcolor="#233945", zeroline=False)
        figure.update_yaxes(gridcolor="#233945", zeroline=False)
        plots.append(f'<article class="panel"><h2>{title}</h2><p>{subtitle}</p>' +
                     figure.to_html(full_html=False, include_plotlyjs=False, div_id=f"comparison-plot-{len(plots) + 1}",
                                    config={"responsive": True, "displaylogo": False}) + "</article>")
    add_plot("01 / Predictive loss", "Lower is better. All models use identical test policies and targets.",
             [go.Bar(x=[LABELS[m] for m in MAIN], y=test.loc[MAIN, "deviance_p1.9"], marker_color=[COLORS[m] for m in MAIN],
                     text=[f"{v:.4f}" for v in test.loc[MAIN, "deviance_p1.9"]], textposition="outside", cliponaxis=False)],
             {"yaxis_title": "Weighted Tweedie deviance, p = 1.9", "yaxis_range": [0, test.loc[MAIN, "deviance_p1.9"].max() * 1.15]})
    add_plot("02 / Financial calibration", "The dashed line marks perfect aggregate calibration. This is not ranking quality.",
             [go.Bar(x=[LABELS[m] for m in MAIN], y=test.loc[MAIN, "predicted_actual"] * 100, marker_color=[COLORS[m] for m in MAIN],
                     text=[f"{v:.1%}" for v in test.loc[MAIN, "predicted_actual"]], textposition="outside", cliponaxis=False)],
             {"yaxis_title": "Expected / observed (%)", "yaxis_range": [0, max(120, test.loc[MAIN, "predicted_actual"].max() * 115)],
              "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 100, "y1": 100, "line": {"color": "#ed966a", "dash": "dot"}}]})
    add_plot("03 / Ranking the portfolio", "Ordered Lorenz curves, from lower to higher predicted risk. Click legends to compare.",
             [go.Scatter(x=lorenz.query("model == @m").exposure_share, y=lorenz.query("model == @m").loss_share,
                         mode="lines", name=f"{LABELS[m]} · Gini {test.loc[m, 'gini']:.3f}", line=dict(color=COLORS[m])) for m in MAIN],
             {"xaxis_title": "Cumulative exposure", "yaxis_title": "Cumulative observed loss"})
    traces = []
    for tag, group in history.query("stage == 'selection'").groupby("run"):
        traces.append(go.Scatter(x=group.epoch, y=group.validation_deviance, mode="lines+markers", name=f"MLP {tag}"))
    add_plot("04 / What validation selected", "No test values appear in this chart. The horizontal line is the best validation GLM.", traces,
             {"xaxis_title": "Epoch", "yaxis_title": "Validation Tweedie deviance",
              "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": val_glm, "y1": val_glm, "line": {"color": "#7699e7", "dash": "dot"}}]})
    add_plot("05 / Random-seed stability", "Same architecture, data and epoch count; independent initialization and batch order. Seed 0 is primary.",
             [go.Scatter(x=[0, 1, 2], y=seed_results["deviance_p1.9"], mode="markers+lines", name="MLP", marker=dict(size=12, color="#45d6c0"))],
             {"xaxis_title": "Random seed", "xaxis_dtick": 1, "yaxis_title": "Test Tweedie deviance",
              "shapes": [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": glm["deviance_p1.9"], "y1": glm["deviance_p1.9"], "line": {"color": "#7699e7", "dash": "dot"}}]})
    add_plot("06 / How certain is the difference?", "Paired policy bootstrap, 1,000 replicates. Negative favors MLP; intervals condition on these fitted models.",
             [go.Scatter(x=intervals.deviance_difference_mlp_minus_glm, y=[LABELS[m] for m in intervals.comparator], mode="markers",
                         marker=dict(size=11, color="#45d6c0"),
                         error_x=dict(type="data", symmetric=False,
                                      array=intervals.ci95_high - intervals.deviance_difference_mlp_minus_glm,
                                      arrayminus=intervals.deviance_difference_mlp_minus_glm - intervals.ci95_low))],
             {"xaxis_title": "Deviance: MLP seed 0 minus comparator", "margin": dict(l=140, r=25, t=35, b=65),
              "shapes": [{"type": "line", "yref": "paper", "y0": 0, "y1": 1, "x0": 0, "x1": 0, "line": {"color": "#ed966a", "dash": "dot"}}]})
    # Aggregate-only table and curves; no individual-policy data enter the page.
    html_rows = "".join(f"<tr><td>{LABELS[m]}</td><td>{test.loc[m, 'deviance_p1.9']:.5f}</td><td>{test.loc[m, 'gini']:.4f}</td><td>{test.loc[m, 'predicted_actual']:.1%}</td></tr>" for m in LABELS)
    plotly_hash = sha256(ROOT / "docs/assets/plotly.min.js")[:12]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM vs MLP / Risk Observatory</title><meta name="description" content="A measured comparison of actuarial GLMs and ordinary neural networks on 169,504 held-out motor policies.">
<script src="assets/plotly.min.js?v={plotly_hash}"></script><style>
*{{box-sizing:border-box}}body{{margin:0;background:#07131c;color:#e7eff2;font-family:'Segoe UI',Arial,sans-serif;line-height:1.6}}main{{max-width:1280px;margin:auto;padding:35px 32px 65px}}a{{color:#45d6c0;text-decoration:none}}header{{display:flex;justify-content:space-between;border-bottom:1px solid #233945;padding-bottom:24px;font-size:13px;gap:15px}}.eyebrow{{color:#cfb57a;font:11px Consolas,monospace;letter-spacing:2px;margin-top:60px}}h1{{font-size:clamp(36px,5vw,66px);font-weight:500;line-height:1.12;max-width:950px;margin:20px 0}}.intro{{color:#90a6b1;max-width:850px;font-size:17px}}.verdict{{padding:26px;border:1px solid #2c706b;background:#0c252d;margin:30px 0}}.verdict strong{{font-size:24px;color:#45d6c0}}.facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:28px 0}}.fact{{border-top:1px solid #233945;padding-top:16px;color:#90a6b1;font-size:12px}}.fact b{{display:block;color:#e7eff2;font-size:25px;font-weight:500}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.panel{{background:#0c1e2b;border:1px solid #233945;border-radius:10px;padding:20px;min-width:0}}h2{{font-size:19px;font-weight:500;margin:0}}.panel p{{color:#90a6b1;font-size:12px;min-height:40px}}.table-wrap{{overflow:auto;margin:40px 0}}table{{width:100%;border-collapse:collapse;text-align:right;font-size:14px}}th,td{{padding:13px;border-bottom:1px solid #233945}}th:first-child,td:first-child{{text-align:left}}th{{color:#90a6b1;font-weight:400}}.method{{margin-top:35px;color:#90a6b1;max-width:1000px}}.method h2{{color:#e7eff2}}footer{{border-top:1px solid #233945;margin-top:50px;padding-top:24px;color:#90a6b1;font-size:12px}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}.facts{{grid-template-columns:1fr 1fr}}main{{padding:24px 16px}}.panel{{padding:15px}}}}
</style></head><body><main><header><a href="index.html">N/ &nbsp; RISK OBSERVATORY</a><a href="https://github.com/zjlshuyu110/non-life-insurance-pricing/tree/main/reports/model_comparison">Inspect measured results ↗</a></header>
<p class="eyebrow">EXPERIMENT 002 / GLM × NEURAL NETWORK</p><h1>Does more flexibility<br>mean better pricing?</h1><p class="intro">An ordinary two-layer MLP meets the actuarial GLM benchmark. Same policies. Same target. Validation-only choices. Three neural seeds.</p>
<div class="verdict"><strong>{html.escape(headline)}</strong><p>{html.escape(verdict)}</p></div>
<div class="facts"><div class="fact"><b>{meta['test_policies']:,}</b>untouched test policies until evaluation</div><div class="fact"><b>{' → '.join(map(str, selected['mlp_widths']))}</b>selected hidden-layer widths</div><div class="fact"><b>{selected['mlp_epochs']}</b>validation-selected epochs</div><div class="fact"><b>03</b>neural seeds, all reported</div></div>
<div class="grid">{''.join(plots)}</div><div class="table-wrap"><table><thead><tr><th>Test model</th><th>Deviance p=1.9 ↓</th><th>Gini ↑</th><th>Expected / observed</th></tr></thead><tbody>{html_rows}</tbody></table></div>
<section class="method"><h2>What was controlled?</h2><p>{meta['inner_fit_policies']:,} policies for inner fitting; {meta['validation_policies']:,} for validation. Binning, categories and scaling were learned on the permitted training split. A {meta['feature_count']}-feature MLP with ReLU hidden layers and a positive log link minimized exposure-weighted Tweedie loss (p=1.9). Validation selected the architecture and epoch, then each seed was refitted on all {meta['train_policies']:,} outer training policies.</p><p>The reference GLMs retain their original settings. The additional tuned Tweedie uses alpha {selected['glm_alpha']}, selected from a four-value validation grid. MLP seed 0 was predeclared as primary; no seed was selected by test performance and no post-hoc calibration was applied.</p><h2>Where the conclusion stops</h2><p>This is one random holdout, a limited model search and a EUR 200k capped target. Calibration, ranking and deviance measure different properties. Bootstrap intervals assume independent policies, condition on fitted models and omit training uncertainty. Future-year results may differ. Further tuning needs fresh evaluation data.</p><p><a href="https://github.com/zjlshuyu110/non-life-insurance-pricing/blob/main/reports/model_comparison/report.md">Read the full English / 中文 experiment report ↗</a></p></section><footer>Python · PyTorch · scikit-learn · Plotly / measured results, not illustrative scores</footer></main></body></html>"""
    (ROOT / "docs/model-comparison.html").write_text(page, encoding="utf-8", newline="\n")
    print("Wrote comparison figures, bilingual report and docs/model-comparison.html", flush=True)
