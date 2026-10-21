"""Static portfolio figures and a standalone, offline Plotly report."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .evaluation import LABELS, MODELS, ordered_lorenz

NAVY = "#153149"
TEAL = "#007f82"
CORAL = "#dc7250"
GRAY = "#8b99a5"
COLORS = {"Baseline": GRAY, "FrequencySeverity": CORAL, "Tweedie": TEAL}


def style_plotly(fig: go.Figure, title: str, y: str = "Pure premium (EUR / exposure-year)") -> go.Figure:
    fig.update_layout(template="plotly_white", title=dict(text=title, font=dict(size=20, color=NAVY)),
                      font=dict(family="Arial, sans-serif", size=13, color=NAVY), height=400,
                      margin=dict(l=65, r=30, t=70, b=65), paper_bgcolor="white", plot_bgcolor="white",
                      legend=dict(orientation="h", y=-0.2, x=0), hovermode="x unified")
    fig.update_yaxes(title_text=y, gridcolor="#e9eef1", zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    return fig


def calibration_figure(table: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    x = table["group"].astype(str).tolist()
    fig.add_trace(go.Scatter(x=x, y=table.actual_rate, name="Observed", mode="lines+markers",
                            line=dict(color=NAVY, width=3), marker=dict(size=7)))
    for name in ["FrequencySeverity", "Tweedie"]:
        fig.add_trace(go.Scatter(x=x, y=table[f"{name}_rate"], name=LABELS[name], mode="lines+markers",
                                line=dict(color=COLORS[name], width=2),
                                customdata=table[["exposure", "policies"]].to_numpy(),
                                hovertemplate="%{y:,.1f} EUR<br>Exposure: %{customdata[0]:,.0f}<br>Policies: %{customdata[1]:,.0f}<extra>%{fullData.name}</extra>"))
    return style_plotly(fig, title)


def static_figures(output: Path, test: pd.DataFrame, predictions: pd.DataFrame,
                   results: pd.DataFrame, tables: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": NAVY, "text.color": NAVY, "axes.titleweight": "bold",
                         "axes.edgecolor": "#ced8df", "xtick.color": NAVY, "ytick.color": NAVY,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("FRENCH MOTOR INSURANCE\nPricing models | held-out validation", fontsize=23, fontweight="bold", x=0.04, ha="left")
    table = tables["risk_decile"]
    ax = axes[0, 0]
    ax.plot(table.group, table.actual_rate, "o-", color=NAVY, label="Observed", lw=2.5)
    for name in ["FrequencySeverity", "Tweedie"]:
        ax.plot(table.group, table[f"{name}_rate"], "o-", color=COLORS[name], label=LABELS[name], lw=2)
    ax.set(title="A  |  Calibration by predicted risk", xlabel="Exposure decile (Tweedie score)", ylabel="EUR / exposure-year")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "--", color=GRAY, label="No ranking")
    for name in ["FrequencySeverity", "Tweedie"]:
        x, y, gini = ordered_lorenz(test.ClaimAmount.to_numpy(), test.Exposure.to_numpy(), predictions[name].to_numpy())
        ax.plot(x, y, color=COLORS[name], lw=2.5, label=f"{LABELS[name]}  Gini {gini:.3f}")
    ax.set(title="B  |  Risk ranking", xlabel="Cumulative exposure (low to high risk)", ylabel="Cumulative share of losses", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=9)
    scored = results.query("split == 'test'").set_index("model")
    ax = axes[1, 0]
    ratios = scored.loc[MODELS, "predicted_actual"].to_numpy() * 100
    bars = ax.barh([LABELS[m] for m in MODELS], ratios, color=[COLORS[m] for m in MODELS], height=.55)
    ax.axvline(100, color=NAVY, ls="--", lw=1.2)
    ax.bar_label(bars, labels=[f"{r:.1f}%" for r in ratios], padding=5)
    ax.set(title="C  |  Portfolio calibration", xlabel="Predicted / observed total losses (%)", xlim=(0, max(120, ratios.max() * 1.16)))
    ax = axes[1, 1]
    values = scored.loc[MODELS, "deviance_p1.9"].to_numpy()
    bars = ax.barh([LABELS[m] for m in MODELS], values, color=[COLORS[m] for m in MODELS], height=.55)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=5)
    ax.set(title="D  |  Predictive loss", xlabel="Exposure-weighted Tweedie deviance, p = 1.9 (lower is better)", xlim=(0, values.max() * 1.2))
    for ax in axes.flat:
        ax.grid(axis="y" if ax in axes[0] else "x", color="#edf0f3", zorder=0)
        ax.set_axisbelow(True)
    fig.savefig(output / "figures" / "overview.png", dpi=180)
    fig.savefig(output / "figures" / "overview.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, key, title in zip(axes, ["driver_age", "vehicle_age"], ["Driver age", "Vehicle age"]):
        table = tables[key]
        x = np.arange(len(table))
        ax.plot(x, table.actual_rate, "o-", color=NAVY, label="Observed", lw=2)
        for name in ["FrequencySeverity", "Tweedie"]:
            ax.plot(x, table[f"{name}_rate"], "o-", color=COLORS[name], label=LABELS[name])
        ax.set_xticks(x, table.group.astype(str), rotation=25, ha="right")
        ax.set(title=title + " | held-out calibration", ylabel="EUR / exposure-year")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis="y", color="#edf0f3")
    fig.savefig(output / "figures" / "segment_calibration.png", dpi=180)
    plt.close(fig)


def generate_reports(output: Path, frame: pd.DataFrame, test: pd.DataFrame, predictions: pd.DataFrame,
                     results: pd.DataFrame, tables: dict, sensitivity: pd.DataFrame, metadata: dict) -> None:
    (output / "figures").mkdir(exist_ok=True)
    static_figures(output, test, predictions, results, tables)
    test_metrics = results.query("split == 'test'").set_index("model")
    figures = [
        ("risk", calibration_figure(tables["risk_decile"], "01 / Calibration across risk groups"),
         "Groups use approximately equal exposure, ordered by the Tweedie score. Both models are evaluated on these same groups; tied scores stay together."),
        ("age", calibration_figure(tables["driver_age"], "02 / Driver age: where calibration changes"),
         "Observed and predicted losses are divided by total exposure within each age band. Hover to inspect group size; small groups can be noisy."),
        ("vehicle", calibration_figure(tables["vehicle_age"], "03 / Vehicle age: segment diagnostics"),
         "Group differences describe this held-out sample. They are associations, not causal effects."),
    ]
    lorenz = go.Figure()
    lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="No ranking", line=dict(color=GRAY, dash="dash")))
    for name in ["FrequencySeverity", "Tweedie"]:
        x, y, gini = ordered_lorenz(test.ClaimAmount.to_numpy(), test.Exposure.to_numpy(), predictions[name].to_numpy())
        # Downsample only for drawing; metrics use every distinct prediction score.
        idx = np.unique(np.linspace(0, len(x) - 1, min(1200, len(x))).astype(int))
        lorenz.add_trace(go.Scatter(x=x[idx], y=y[idx], name=f"{LABELS[name]} (Gini {gini:.3f})", mode="lines", line=dict(color=COLORS[name], width=3)))
    style_plotly(lorenz, "04 / Ranking risks", "Cumulative share of observed losses")
    lorenz.update_xaxes(title="Cumulative exposure, from lowest to highest predicted risk", tickformat=".0%")
    lorenz.update_yaxes(tickformat=".0%")
    figures.append(("ranking", lorenz, "The ordered Lorenz Gini measures ranking, not absolute calibration. These are raw Ginis, not normalized by an oracle."))
    region = tables["region"].sort_values("tweedie_actual_expected")
    fig_region = go.Figure(go.Bar(x=region.tweedie_actual_expected, y=region.group.astype(str), orientation="h",
                                 marker_color=[CORAL if value > 1 else TEAL for value in region.tweedie_actual_expected],
                                 customdata=region.exposure, hovertemplate="Actual / predicted: %{x:.2f}<br>Exposure: %{customdata:,.0f}<extra></extra>"))
    style_plotly(fig_region, "05 / Regional calibration: Tweedie", "Region")
    fig_region.update_layout(height=470)
    fig_region.update_xaxes(title="Actual / predicted losses (>1 means underprediction)")
    fig_region.add_vline(x=1, line_dash="dash", line_color=NAVY)
    figures.append(("region", fig_region, "A descriptive diagnostic across all regions. No credibility adjustment or statistical significance claim is applied."))
    if not sensitivity.empty:
        fig_sens = go.Figure()
        for name in MODELS:
            rows = sensitivity[sensitivity.model == name].copy()
            labels = rows.loss_cap.map(lambda value: "Uncapped" if str(value) == "Uncapped" else f"EUR {float(value):,.0f}")
            fig_sens.add_trace(go.Bar(x=labels, y=100 * rows.predicted_actual, name=LABELS[name], marker_color=COLORS[name]))
        style_plotly(fig_sens, "06 / Sensitivity to the loss cap", "Predicted / observed losses (%)")
        fig_sens.add_hline(y=100, line_dash="dash", line_color=NAVY)
        figures.append(("sensitivity", fig_sens, "Each scenario refits on the same training IDs and evaluates the matching loss definition on the same test IDs. Deviance values across different caps are not comparable."))
    cards = "".join(f'<article class="kpi"><span>{label}</span><strong>{value}</strong><small>{note}</small></article>' for label, value, note in [
        ("POLICIES", f"{len(frame):,}", "Full portfolio" if metadata["full_data_run"] else "Sample run"),
        ("HELD-OUT POLICIES", f"{len(test):,}", "25% random split · seed 0" if metadata["config"]["test_size"] == .25 and metadata["config"]["seed"] == 0 else "Configured random holdout"),
        ("ANNUAL PURE PREMIUM", f"EUR {frame.ClaimAmount.sum() / frame.Exposure.sum():,.0f}", "Observed, under the benchmark cap"),
        ("TWEEDIE CALIBRATION", f"{test_metrics.loc['Tweedie', 'predicted_actual']:.1%}", "Predicted / observed test losses"),
    ])
    sections = "".join(f'<section class="chart" id="{key}">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True}, div_id=key + "-plot")}<p>{html.escape(note)}</p></section>' for key, fig, note in figures)
    display = test_metrics[["predicted_actual", "deviance_p1.9", "gini", "weighted_mae"]].copy()
    display.index = [LABELS[m] for m in display.index]
    display.columns = ["Predicted / actual", "Deviance (p=1.9)", "Ranking Gini", "Weighted MAE"]
    table_html = display.to_html(float_format=lambda value: f"{value:,.4f}", border=0, classes="metrics")
    css = """
    :root{--ink:#153149;--teal:#007f82;--muted:#5b6e7d;--line:#e2e9ed}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f2f5f7;color:var(--ink);font-family:Arial,sans-serif;line-height:1.55}
    header{background:var(--ink);color:white;padding:52px max(6vw,24px) 44px;border-bottom:6px solid var(--teal)}
    .eyebrow{font-size:12px;letter-spacing:2px;color:#9de0d9}h1{font-size:clamp(32px,5vw,52px);line-height:1.12;margin:14px 0}header p{max-width:760px;color:#d0dde5;font-size:17px}
    .badge{display:inline-block;border:1px solid #557085;padding:5px 12px;border-radius:20px;font-size:12px;margin:6px 8px 0 0}
    main{max-width:1250px;margin:auto;padding:28px 28px 60px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
    .kpi,.chart,.panel{background:white;border:1px solid var(--line);border-radius:12px}.kpi{padding:22px}.kpi span{font-size:10px;font-weight:bold;letter-spacing:1.1px;color:var(--muted)}
    .kpi strong{display:block;font-size:29px;margin:10px 0;color:var(--teal)}.kpi small{font-size:12px;color:var(--muted)}
    nav{display:flex;gap:18px;flex-wrap:wrap;margin:28px 0;font-size:13px}a{color:var(--teal)}nav a{text-decoration:none;font-weight:bold}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.chart{overflow:hidden;padding:10px}.chart p{margin:4px 20px 18px;font-size:12px;color:var(--muted)}
    .panel{padding:26px;margin:26px 0}.panel h2{font-size:21px;margin-top:0}.panel p,.panel li{font-size:14px;color:var(--muted)}
    table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:12px;border-bottom:1px solid var(--line)}th:first-child{text-align:left}
    .table-scroll{overflow-x:auto}footer{font-size:12px;color:var(--muted);margin-top:26px}.note{border-left:4px solid var(--teal);padding-left:16px}
    @media(max-width:900px){.kpis{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}
    @media(max-width:520px){main{padding:18px 12px}.kpi{padding:15px}.kpi strong{font-size:23px}header{padding:32px 22px}.panel{padding:18px}}
    """
    actual_sources = " · ".join(f'<a href="{html.escape(url, quote=True)}">{html.escape(name)}</a>' for name, url in metadata["data_sources"].items())
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>French motor insurance | Pricing study</title><style>{css}</style><script>{get_plotlyjs()}</script></head><body>
    <header><div class="eyebrow">NON-LIFE ACTUARIAL PORTFOLIO / STUDY 01</div><h1>French motor insurance pricing</h1><p>From claim frequency and severity to pure premium. A reproducible comparison of two actuarial GLM approaches, with held-out calibration and risk diagnostics.</p><span class="badge">Poisson × Gamma</span><span class="badge">Tweedie GLM</span><span class="badge">Training-only preprocessing</span><span class="badge">Offline interactive report</span></header>
    <main><div class="kpis">{cards}</div><nav><a href="#results">Model results</a><a href="#risk">Risk groups</a><a href="#age">Age segments</a><a href="#ranking">Ranking</a><a href="#region">Regions</a><a href="#method">Method & scope</a></nav>
    <section class="panel" id="results"><h2>Held-out model comparison</h2><div class="table-scroll">{table_html}</div><p>Calibration closer to 1 is better; deviance and MAE are lower-is-better. Gini measures ranking only. Results use the same holdout and the same loss definition.</p></section>
    <div class="grid">{sections}</div>
    <section class="panel" id="method"><h2>What was reproduced</h2><p>The scikit-learn French MTPL worked example: exposure-weighted Poisson frequency, claim-count-weighted Gamma severity, and an exposure-weighted Tweedie pure-premium GLM (p = 1.9). Hyperparameters are fixed from the reference, with no test-set tuning.</p><p class="note">This is a methodological reproduction. Unlike the reference, bins, category levels and scaling are learned only from training data. Numerical equality with the online tutorial is therefore not expected.</p><h2>How to read the results</h2><ul><li>The benchmark caps policy-level aggregate losses at EUR 200,000, annual exposure at 1, and claim count at 4. Zero-loss policies have their claim count reset to zero. The resulting target is modified, not uncapped commercial pricing.</li><li>There are {metadata['test_positive_loss_policies']:,} positive-loss policies in the test set. Sparse large losses can materially affect comparisons.</li><li>This is one random holdout, not a future-year validation or evidence of statistically significant model superiority.</li><li>Pure premium covers expected claims only. Expenses, profit margins, taxes, inflation and reserve development are outside this study.</li></ul><p>Sources: <a href="https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html">scikit-learn example</a> · <a href="https://www.openml.org/d/41214">freMTPL2freq</a> · <a href="https://www.openml.org/d/41215">freMTPL2sev</a>. Source provenance and checksums are saved in data/manifest.json.</p></section>
    <footer>Actual data downloads: {actual_sources}. Generated from this run's actual results. Charts are interactive and work offline; click a legend entry to toggle a model. Configuration and environment: run_metadata.json.</footer></main></body></html>"""
    (output / "dashboard.html").write_text(html_doc, encoding="utf-8")
    lines = ["# Run report", "", "This file is generated from the actual run; no metrics are manually entered.", "",
             f"Policies: {len(frame):,}. Training: {metadata['train_policies']:,}. Test: {len(test):,}.",
             f"Positive-loss test policies: {metadata['test_positive_loss_policies']:,}.", "",
             "| Model | Predicted / actual | Deviance (p=1.9) | Ranking Gini |",
             "|---|---:|---:|---:|"]
    for name in MODELS:
        row = test_metrics.loc[name]
        lines.append(f"| {LABELS[name]} | {row.predicted_actual:.4f} | {row['deviance_p1.9']:.4f} | {row.gini:.4f} |")
    improvement = 100 * (1 - test_metrics.loc["Tweedie", "deviance_p1.9"] / test_metrics.loc["Baseline", "deviance_p1.9"])
    lines.extend(["", f"Tweedie deviance reduction relative to the training-portfolio-mean baseline: {improvement:.2f}%.",
                  "This is descriptive performance on a single holdout, not a significance test.", "",
                  "![Model overview](figures/overview.png)", "", "![Segment calibration](figures/segment_calibration.png)", "",
                  "## Cleaning audit", "", "```json", __import__("json").dumps(metadata["cleaning_audit"], indent=2), "```", "",
                  "## Interpretation", "", "- Compare model deviances only under an identical target definition and Tweedie power.",
                  "- Do not interpret a high Gini as evidence of accurate total premiums; assess calibration separately.",
                  "- Caps change the target. The EUR 200k study does not price the excluded tail.",
                  "- Frequency is adjusted to positive-loss claims, following the reference; it is not all reported claims.",
                  "- Quantile bins and scaling are fit only on training policies, unlike the original example.",
                  "- The sample has no usable calendar-year holdout here; random splitting does not establish temporal generalization.", ""])
    if not sensitivity.empty:
        lines.extend(["## Loss-cap sensitivity", "", "The following are refitted models on identical policy splits. Compare models within each target definition.", "",
                      "| Policy loss cap | Model | Predicted / actual | Deviance (p=1.9) |",
                      "|---|---|---:|---:|"])
        for _, row in sensitivity.iterrows():
            cap_label = "Uncapped" if str(row.loss_cap) == "Uncapped" else f"EUR {float(row.loss_cap):,.0f}"
            lines.append(f"| {cap_label} | {LABELS[row.model]} | {row.predicted_actual:.4f} | {row['deviance_p1.9']:.4f} |")
        lines.extend(["", "The uncapped baseline is useful context: a train/test difference in average loss cost can affect both GLMs. A larger calibration gap does not by itself identify a modelling defect. Investigate tail concentration and repeated splits before making a production decision.", ""])
    lines.extend(["## Merge audit", "", f"Severity rows without a matching frequency policy: {metadata['merge_audit']['orphan_severity_rows']:,}; unmatched claim amount: EUR {metadata['merge_audit']['orphan_claim_amount']:,.2f}. These do not enter the matched policy-level study.", "",
                  "Actual sources (also recorded with hashes in the manifest):", "",
                  *[f"- [{name}]({url})" for name, url in metadata["data_sources"].items()], ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
