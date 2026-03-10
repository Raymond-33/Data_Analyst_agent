"""
Visualization Engine — Enhanced
=================================
Comprehensive chart generation for the AI Data Analyst dashboard.
All charts use Plotly with a consistent dark theme, the project accent
palette, and modern styling. Organized into:

  - Auto-EDA charts (distribution, correlation, composition, comparison)
  - Data quality charts (missing values, outliers, class balance)
  - Statistical charts (feature importance, clustering, trends)
  - On-demand chart generator for conversational queries
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


# ─────────────────────────────────────────────
# Theme Constants
# ─────────────────────────────────────────────

TEMPLATE = "plotly_dark"
ACCENT = "#64FFDA"
ACCENT_2 = "#4D96FF"
ACCENT_3 = "#FF6B6B"
ACCENT_4 = "#C084FC"
ACCENT_5 = "#FBBF24"
BG_COLOR = "#112240"
PAPER_COLOR = "#0A192F"
GRID_COLOR = "rgba(100, 255, 218, 0.1)"

# Curated palette for multi-series charts
PALETTE = [ACCENT, ACCENT_2, ACCENT_3, ACCENT_4, ACCENT_5,
           "#34D399", "#F472B6", "#FB923C", "#818CF8", "#A78BFA"]


def _apply_theme(fig, title: str = "", description: str = ""):
    """Applies consistent professional styling to any Plotly figure."""
    # Build title with optional subtitle description
    if description:
        full_title = (
            f"<b>{title}</b>"
            f"<br><span style='font-size:11px;color:#8892B0'>{description}</span>"
        )
    else:
        full_title = f"<b>{title}</b>"

    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=full_title, font=dict(size=16, color=ACCENT)),
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, sans-serif", color="#a8b2d1", size=12),
        margin=dict(l=40, r=30, t=70, b=50),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        legend=dict(
            bgcolor="rgba(17,34,64,0.8)",
            bordercolor="rgba(100,255,218,0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor=BG_COLOR, font_size=12),
    )
    return fig


# ═══════════════════════════════════════════════
# SECTION A: Auto-EDA Charts (existing, enhanced)
# ═══════════════════════════════════════════════

def generate_all_charts(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Dict[str, go.Figure]:
    """
    Backward-compatible: generates the original 5 chart types
    with improved styling, descriptions, and legends.
    """
    charts = {}

    # 1. Distribution — highest variance numeric
    if num_cols:
        target_num = df[num_cols].var().idxmax()
        col_data = df[target_num].dropna()
        mean_val = col_data.mean()
        std_val = col_data.std()
        fig = px.histogram(
            df, x=target_num,
            color_discrete_sequence=[ACCENT],
            opacity=0.85,
        )
        fig.update_traces(
            marker_line_color=ACCENT_2, marker_line_width=1,
            name=f"{target_num} frequency",
            showlegend=True,
        )
        # Add mean line
        fig.add_vline(
            x=mean_val, line_dash="dash", line_color=ACCENT_3,
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right",
            annotation_font_color=ACCENT_3,
        )
        desc = f"Mean={mean_val:.2f}, Std={std_val:.2f}, N={len(col_data):,} | Selected as highest-variance numeric column"
        fig.update_layout(xaxis_title=target_num, yaxis_title="Frequency")
        charts["dist"] = _apply_theme(fig, f"Distribution of {target_num}", desc)

    # 2. Composition — first categorical as donut
    if cat_cols:
        n_unique = df[cat_cols[0]].nunique()
        top_val = df[cat_cols[0]].value_counts().index[0]
        top_pct = (df[cat_cols[0]].value_counts().iloc[0] / len(df) * 100)
        fig = px.pie(
            df, names=cat_cols[0],
            hole=0.45,
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        desc = f"{n_unique} unique categories | Most common: '{top_val}' ({top_pct:.1f}%)"
        charts["comp"] = _apply_theme(fig, f"Composition by {cat_cols[0]}", desc)

    # 3. Relationship — top 2 numeric scatter
    if len(num_cols) >= 2:
        corr_val = df[num_cols[0]].corr(df[num_cols[1]])
        corr_str = f"Pearson r = {corr_val:.3f}" if not pd.isna(corr_val) else "No correlation computed"
        fig = px.scatter(
            df, x=num_cols[0], y=num_cols[1],
            color=cat_cols[0] if cat_cols else None,
            color_discrete_sequence=PALETTE,
            opacity=0.7,
        )
        fig.update_layout(xaxis_title=num_cols[0], yaxis_title=num_cols[1])
        desc = f"Scatter plot of {len(df):,} data points | {corr_str}"
        charts["rel"] = _apply_theme(fig, f"{num_cols[0]} vs {num_cols[1]}", desc)

    # 4. Comparison — category vs numeric average
    if cat_cols and num_cols:
        agg = df.groupby(cat_cols[0])[num_cols[0]].mean().reset_index()
        overall_mean = df[num_cols[0]].mean()
        fig = px.bar(
            agg, x=cat_cols[0], y=num_cols[0],
            color_discrete_sequence=[ACCENT_2],
        )
        fig.update_traces(
            marker_line_color=ACCENT, marker_line_width=1,
            name=f"Mean {num_cols[0]}",
            showlegend=True,
        )
        # Add overall mean line
        fig.add_hline(
            y=overall_mean, line_dash="dot", line_color=ACCENT_5,
            annotation_text=f"Overall mean: {overall_mean:.2f}",
            annotation_font_color=ACCENT_5,
        )
        fig.update_layout(
            xaxis_title=cat_cols[0],
            yaxis_title=f"Average {num_cols[0]}",
        )
        desc = f"Average {num_cols[0]} across {agg.shape[0]} categories | Overall mean: {overall_mean:.2f}"
        charts["bar"] = _apply_theme(fig, f"Avg {num_cols[0]} per {cat_cols[0]}", desc)

    # 5. Correlation heatmap
    if len(num_cols) > 1:
        corr = df[num_cols].corr()
        max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        if len(max_corr) > 0:
            strongest = max_corr.abs().idxmax()
            strongest_val = max_corr.loc[strongest]
            corr_desc = f"Strongest pair: {strongest[0]} × {strongest[1]} (r={strongest_val:.3f})"
        else:
            corr_desc = ""
        fig = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            labels=dict(color="Pearson r"),
        )
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
        )
        desc = f"Pearson correlation matrix of {len(num_cols)} numeric features | {corr_desc}"
        charts["heat"] = _apply_theme(fig, "Correlation Matrix", desc)

    return charts


# ═══════════════════════════════════════════════
# SECTION B: Data Quality Charts
# ═══════════════════════════════════════════════

def plot_missing_values(missing_report: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Horizontal bar chart showing missing-value percentages per column.
    Only shows columns with at least 1 missing value.
    """
    cols_with_missing = [
        c for c in missing_report["columns"] if c["missing_count"] > 0
    ]
    if not cols_with_missing:
        return None

    df_plot = pd.DataFrame(cols_with_missing)
    df_plot = df_plot.sort_values("missing_pct", ascending=True)

    # Color by severity with legend traces
    color_map = {
        "low": ACCENT_2,
        "moderate": ACCENT_5,
        "high": "#FB923C",
        "critical": ACCENT_3,
    }
    colors = [color_map.get(s, ACCENT) for s in df_plot["severity"]]

    fig = go.Figure(go.Bar(
        x=df_plot["missing_pct"],
        y=df_plot["column"],
        orientation="h",
        marker_color=colors,
        text=[f"{v}% ({c} rows)" for v, c in zip(df_plot["missing_pct"], df_plot["missing_count"])],
        textposition="outside",
        name="Missing %",
    ))

    total_missing = sum(c["missing_count"] for c in cols_with_missing)
    desc = (
        f"{len(cols_with_missing)} columns with missing data | "
        f"Total missing cells: {total_missing:,} | "
        f"Color: 🔵low 🟡moderate 🟠high 🔴critical"
    )
    fig = _apply_theme(fig, "Missing Values by Column", desc)
    fig.update_layout(
        xaxis_title="Percentage Missing (%)",
        yaxis_title="Column Name",
        height=max(300, len(cols_with_missing) * 40 + 120),
        showlegend=False,
    )
    return fig


def plot_outliers(outlier_report: Dict[str, Any], df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Box plots for columns that have outliers, highlighting the outlier points.
    Uses IQR method: points beyond Q1-1.5×IQR or Q3+1.5×IQR are outliers.
    """
    cols_with_outliers = [
        c for c in outlier_report["columns"] if c["outlier_count"] > 0
    ]
    if not cols_with_outliers:
        return None

    # Limit to top 6 to keep the chart readable
    cols_to_plot = [c["column"] for c in cols_with_outliers[:6]]
    outlier_counts = {c["column"]: c["outlier_count"] for c in cols_with_outliers[:6]}
    total_outliers = sum(outlier_counts.values())

    fig = go.Figure()
    for i, col in enumerate(cols_to_plot):
        fig.add_trace(go.Box(
            y=df[col],
            name=f"{col} ({outlier_counts[col]} outliers)",
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.5,
        ))

    desc = (
        f"Box plots showing {total_outliers:,} total outliers across {len(cols_to_plot)} columns | "
        f"IQR method (1.5× interquartile range) | Dashed line = mean, solid = median"
    )
    fig = _apply_theme(fig, "Outlier Distribution", desc)
    fig.update_layout(
        showlegend=True,
        height=450,
        yaxis_title="Value",
    )
    return fig


def plot_class_balance(
    imbalance_report: List[Dict[str, Any]],
) -> Optional[go.Figure]:
    """
    Bar chart showing class distribution for imbalanced columns.
    Highlights target columns differently.
    """
    if not imbalance_report:
        return None

    # Pick the most relevant column (target first, then most imbalanced)
    target_cols = [c for c in imbalance_report if c["is_target_column"]]
    col_info = target_cols[0] if target_cols else imbalance_report[0]

    dist = col_info["class_distribution"]
    classes = list(dist.keys())
    values = list(dist.values())
    ratio = col_info.get("imbalance_ratio", "N/A")

    fig = go.Figure(go.Bar(
        x=classes,
        y=values,
        marker_color=PALETTE[:len(classes)],
        text=[f"{v}%" for v in values],
        textposition="outside",
        name=col_info["column"],
    ))

    # Add balanced reference line
    balanced_pct = 100 / len(classes) if classes else 0
    fig.add_hline(
        y=balanced_pct, line_dash="dot", line_color=ACCENT,
        annotation_text=f"Balanced: {balanced_pct:.1f}%",
        annotation_font_color=ACCENT,
    )

    severity_tag = f" ({col_info['severity']})" if col_info.get("severity") != "balanced" else ""
    target_tag = " 🎯 Target Column" if col_info.get("is_target_column") else ""
    desc = (
        f"Imbalance ratio: {ratio}x | {len(classes)} classes{target_tag} | "
        f"Dotted line shows ideal balanced distribution"
    )
    fig = _apply_theme(fig, f"Class Distribution: {col_info['column']}{severity_tag}", desc)
    fig.update_layout(
        xaxis_title="Class Label",
        yaxis_title="Percentage (%)",
        height=400,
    )
    return fig


# ═══════════════════════════════════════════════
# SECTION C: Statistical Charts
# ═══════════════════════════════════════════════

def plot_distributions_grid(
    df: pd.DataFrame,
    num_cols: List[str],
    max_cols: int = 6,
) -> Optional[go.Figure]:
    """
    Grid of histograms for multiple numeric columns with KDE-style overlay.
    """
    cols = num_cols[:max_cols]
    if not cols:
        return None

    n = len(cols)
    n_rows = (n + 1) // 2
    n_grid_cols = min(n, 2)

    fig = make_subplots(
        rows=n_rows, cols=n_grid_cols,
        subplot_titles=[c for c in cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    for i, col in enumerate(cols):
        row = i // n_grid_cols + 1
        col_idx = i % n_grid_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                marker_color=PALETTE[i % len(PALETTE)],
                opacity=0.8,
                name=col,
                showlegend=False,
                nbinsx=30,
            ),
            row=row, col=col_idx,
        )

    desc = f"Frequency histograms for {len(cols)} numeric columns (30 bins each) | Showing data shape and spread"
    fig = _apply_theme(fig, "Distribution Analysis", desc)
    fig.update_layout(height=300 * n_rows)
    return fig


def plot_feature_importance(
    importance_report: Dict[str, Any],
    top_n: int = 10,
) -> Optional[go.Figure]:
    """
    Horizontal bar chart of feature importance scores.
    """
    rankings = importance_report.get("rankings", [])
    if not rankings:
        return None

    top = rankings[:top_n]
    top.reverse()  # Bottom-to-top for horizontal bar

    fig = go.Figure(go.Bar(
        x=[r["score"] for r in top],
        y=[r["feature"] for r in top],
        orientation="h",
        marker=dict(
            color=[r["score"] for r in top],
            colorscale=[[0, ACCENT_2], [1, ACCENT]],
        ),
        text=[f"{r['score']:.4f}" for r in top],
        textposition="outside",
    ))

    method = importance_report.get("method", "")
    target = importance_report.get("target_used", "")
    subtitle = f" (target: {target})" if target else ""
    top_feature = rankings[0]["feature"] if rankings else "N/A"
    top_score = rankings[0]["score"] if rankings else 0

    desc = (
        f"Method: {method}{subtitle} | "
        f"Top feature: {top_feature} (score={top_score:.4f}) | "
        f"Higher score = more predictive power"
    )
    fig = _apply_theme(fig, f"Feature Importance — {method}{subtitle}", desc)
    fig.update_layout(
        xaxis_title="Importance Score",
        height=max(300, len(top) * 35 + 120),
    )
    return fig


def plot_correlation_top_pairs(
    corr_report: Dict[str, Any],
    top_n: int = 10,
) -> Optional[go.Figure]:
    """
    Horizontal bar chart of top correlated feature pairs (positive & negative).
    """
    top_pos = corr_report.get("top_positive", [])
    top_neg = corr_report.get("top_negative", [])
    pairs = (top_pos[:top_n // 2] + top_neg[:top_n // 2])

    if not pairs:
        return None

    pairs.sort(key=lambda x: x["correlation"])

    labels = [f"{p['col1']} × {p['col2']}" for p in pairs]
    values = [p["correlation"] for p in pairs]
    colors = [ACCENT if v > 0 else ACCENT_3 for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    desc = (
        f"{len(pairs)} feature pairs | "
        f"🟢 Positive = move together | 🔴 Negative = move oppositely | "
        f"Range: -1 (perfect inverse) to +1 (perfect correlation)"
    )
    fig = _apply_theme(fig, "Top Correlated Feature Pairs", desc)
    fig.update_layout(
        xaxis_title="Pearson Correlation Coefficient (r)",
        xaxis=dict(range=[-1.1, 1.1]),
        height=max(300, len(pairs) * 35 + 120),
    )
    return fig


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Optional[List[str]] = None,
    max_lines: int = 4,
) -> Optional[go.Figure]:
    """
    Line chart for time series data. Automatically selects numeric columns
    if value_cols is not provided.
    """
    if date_col not in df.columns:
        return None

    try:
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[date_col]).sort_values(date_col)
    except Exception:
        return None

    if value_cols is None:
        value_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
    value_cols = value_cols[:max_lines]

    if not value_cols:
        return None

    fig = go.Figure()
    for i, col in enumerate(value_cols):
        fig.add_trace(go.Scatter(
            x=df_ts[date_col],
            y=df_ts[col],
            mode="lines",
            name=col,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2),
        ))

    date_range = f"{df_ts[date_col].min().strftime('%Y-%m-%d')} to {df_ts[date_col].max().strftime('%Y-%m-%d')}"
    desc = f"Time series of {len(value_cols)} metrics | Date range: {date_range} | {len(df_ts):,} data points"
    fig = _apply_theme(fig, "Time Series Trends", desc)
    fig.update_layout(
        xaxis_title=f"Date ({date_col})",
        yaxis_title="Value",
        hovermode="x unified",
        height=450,
    )
    return fig


def plot_clustering(
    df: pd.DataFrame,
    labels: Any,
    num_cols: List[str],
) -> Optional[go.Figure]:
    """
    2D scatter plot of clustered data, using the first two numeric columns.
    Handles length mismatch when clustering was done on NaN-dropped data.
    """
    if len(num_cols) < 2 or labels is None:
        return None

    # Align: clustering was run on dropna subset, so filter to matching rows
    df_num = df[num_cols].dropna()
    n_labels = len(labels)
    if len(df_num) > n_labels:
        df_num = df_num.head(n_labels)
    elif len(df_num) < n_labels:
        labels = labels[:len(df_num)]

    df_plot = df_num[[num_cols[0], num_cols[1]]].copy()
    df_plot["Cluster"] = [f"Cluster {l}" for l in labels]

    fig = px.scatter(
        df_plot,
        x=num_cols[0],
        y=num_cols[1],
        color="Cluster",
        color_discrete_sequence=PALETTE,
        opacity=0.7,
    )
    n_clusters = len(set(labels))
    desc = (
        f"{n_clusters} clusters identified via K-Means | "
        f"{len(df_plot):,} data points | "
        f"Axes: {num_cols[0]} × {num_cols[1]} (first two numeric features)"
    )
    fig = _apply_theme(fig, f"K-Means Clustering ({num_cols[0]} vs {num_cols[1]})", desc)
    fig.update_layout(
        height=450,
        xaxis_title=num_cols[0],
        yaxis_title=num_cols[1],
    )
    return fig


def plot_silhouette_scores(k_scores: List[Dict[str, Any]]) -> Optional[go.Figure]:
    """
    Line chart showing silhouette scores for different k values.
    """
    if not k_scores:
        return None

    ks = [s["k"] for s in k_scores]
    scores = [s["silhouette"] for s in k_scores]
    best_k = ks[scores.index(max(scores))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=scores,
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=8),
        name="Silhouette Score",
    ))
    # Highlight optimal k
    fig.add_trace(go.Scatter(
        x=[best_k], y=[max(scores)],
        mode="markers",
        marker=dict(size=14, color=ACCENT_3, symbol="star"),
        name=f"Optimal k={best_k}",
    ))

    desc = (
        f"Tested k={min(ks)} to k={max(ks)} | "
        f"Best k={best_k} (silhouette={max(scores):.3f}) | "
        f"Higher silhouette = better-separated clusters (max=1.0)"
    )
    fig = _apply_theme(fig, "Optimal Cluster Selection", desc)
    fig.update_layout(
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score (higher = better)",
        height=350,
    )
    return fig


# ═══════════════════════════════════════════════
# SECTION D: Quality Score Gauge
# ═══════════════════════════════════════════════

def plot_quality_gauge(quality_score: Dict[str, Any]) -> go.Figure:
    """
    Gauge chart showing the overall data quality score (0-100).
    """
    total = quality_score["total"]
    grade = quality_score["grade"]

    # Color based on score
    if total >= 90:
        bar_color = ACCENT
    elif total >= 75:
        bar_color = ACCENT_2
    elif total >= 60:
        bar_color = ACCENT_5
    elif total >= 40:
        bar_color = "#FB923C"
    else:
        bar_color = ACCENT_3

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total,
        number=dict(suffix=f" ({grade})", font=dict(size=36, color=ACCENT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=ACCENT),
            bar=dict(color=bar_color),
            bgcolor=BG_COLOR,
            bordercolor=ACCENT,
            steps=[
                dict(range=[0, 40], color="rgba(255,107,107,0.15)"),
                dict(range=[40, 60], color="rgba(251,191,36,0.1)"),
                dict(range=[60, 75], color="rgba(77,150,255,0.1)"),
                dict(range=[75, 100], color="rgba(100,255,218,0.1)"),
            ],
            threshold=dict(line=dict(color=ACCENT_3, width=2), value=total),
        ),
        title=dict(text="Data Quality Score", font=dict(size=18, color="#a8b2d1")),
    ))

    fig.update_layout(
        paper_bgcolor=PAPER_COLOR,
        font=dict(family="Inter, sans-serif", color="#a8b2d1"),
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    return fig


def plot_quality_dimensions(quality_score: Dict[str, Any]) -> go.Figure:
    """
    Radar chart showing the four quality dimensions
    (completeness, uniqueness, consistency, validity).
    """
    dims = quality_score["dimensions"]
    categories = list(dims.keys())
    values = list(dims.values())

    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=[c.title() for c in categories],
        fill="toself",
        fillcolor="rgba(100, 255, 218, 0.15)",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=8, color=ACCENT),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 25], gridcolor=GRID_COLOR, color="#a8b2d1"),
            angularaxis=dict(gridcolor=GRID_COLOR, color=ACCENT),
            bgcolor=BG_COLOR,
        ),
        paper_bgcolor=PAPER_COLOR,
        font=dict(family="Inter, sans-serif", color="#a8b2d1", size=13),
        title=dict(text="Quality Dimensions (out of 25)", font=dict(size=16, color=ACCENT)),
        height=380,
        margin=dict(l=50, r=50, t=60, b=40),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════
# SECTION E: Full Dashboard Chart Generator
# ═══════════════════════════════════════════════

def generate_dashboard_charts(
    df: pd.DataFrame,
    schema_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    profiling_report: Dict[str, Any],
) -> Dict[str, Optional[go.Figure]]:
    """
    Master function: generates ALL charts for the full dashboard.
    Returns a dict of chart_name → figure. None values indicate
    the chart wasn't applicable for this dataset.
    """
    cats = schema_report["column_categories"]
    num_cols = cats.get("numeric", [])
    cat_cols = cats.get("categorical", [])
    ts = schema_report["time_series"]

    charts: Dict[str, Optional[go.Figure]] = {}

    # ── Data Quality Section ──
    charts["quality_gauge"] = plot_quality_gauge(quality_report["overall_score"])
    charts["quality_dimensions"] = plot_quality_dimensions(quality_report["overall_score"])
    charts["missing_values"] = plot_missing_values(quality_report["missing"])
    charts["outliers"] = plot_outliers(quality_report["outliers"], df)
    charts["class_balance"] = plot_class_balance(quality_report["class_imbalance"])

    # ── Statistical Section ──
    charts["distributions"] = plot_distributions_grid(df, num_cols)
    charts["feature_importance"] = plot_feature_importance(
        profiling_report["importance"]
    )
    charts["correlation_pairs"] = plot_correlation_top_pairs(
        profiling_report["correlations"]
    )

    # ── Core EDA Charts ──
    eda = generate_all_charts(df, num_cols, cat_cols)
    charts.update(eda)

    # ── Time Series ──
    if ts["is_time_series"] and ts["best_time_column"]:
        charts["time_series"] = plot_time_series(
            df, ts["best_time_column"], num_cols[:4]
        )

    # ── Clustering ──
    clust = profiling_report.get("clustering")
    if clust and clust.get("labels") is not None:
        charts["clustering"] = plot_clustering(df, clust["labels"], num_cols)
        charts["silhouette"] = plot_silhouette_scores(clust["k_scores"])

    return charts