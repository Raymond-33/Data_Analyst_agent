"""
AI Autonomous Data Analyst — Main Application
================================================
Streamlit app with two modes:
  MODE 1: Autonomous Initial Analysis (Auto-EDA) — triggers automatically on upload
  MODE 2: Conversational Data Intelligence — AI consultant with analytical reasoning

Architecture:
  CSV Upload → Data Handler → Schema Intel + Quality + Profiling → LLM → Charts → Memory
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from config import set_page_config, inject_custom_css
from utils.memory import (
    init_memory, clear_memory, store_dataset, store_reports,
    store_executive_summary, store_llm_context, mark_analysis_complete,
    get_clean_data, get_schema_report, get_quality_report,
    get_profiling_report, get_executive_summary, get_llm_context,
    get_messages, add_message, store_query_result,
    has_dataset, is_analysis_ready, get_file_name,
    get_dataset_info_summary, get_conversation_context,
)
from utils.data_handler import (
    load_file_robust, normalize_column_names, clean_dataframe,
    get_col_types, prepare_llm_context,
)
from utils.schema_intel import (
    get_schema_intelligence_report, schema_report_to_dataframe,
    format_schema_summary,
)
from utils.data_quality import (
    get_quality_report as compute_quality_report,
    missing_report_to_dataframe, outlier_report_to_dataframe,
    format_quality_summary,
)
from utils.stats_profiler import (
    get_profiling_report as compute_profiling_report,
    format_profiling_summary, correlation_pairs_to_dataframe,
)
from utils.plots import generate_dashboard_charts
from utils.ai_agent import (
    generate_executive_summary, render_executive_summary_md,
    query_data_consultant,
)

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
load_dotenv()
set_page_config()
inject_custom_css()
init_memory()


# ─────────────────────────────────────────────
# Data Upload Handler
# ─────────────────────────────────────────────

def handle_file_upload(uploaded_file):
    """Handles new file upload — loads, analyzes, and stores everything."""
    if uploaded_file is None:
        return

    # Skip if same file already loaded
    if get_file_name() == uploaded_file.name and has_dataset():
        return

    # New file → clear everything and run full pipeline
    clear_memory()

    with st.spinner("Loading and preparing dataset..."):
        # Step 1: Load CSV
        df_raw, file_size = load_file_robust(uploaded_file)
        if df_raw is None:
            st.error("Failed to load file. Please check the format.")
            return

        # Step 2: Normalize column names
        col_map = dict(zip(df_raw.columns.tolist(), [None]))  # placeholder
        df_raw = normalize_column_names(df_raw)

        # Step 3: Schema Intelligence (on raw)
        schema_report = get_schema_intelligence_report(df_raw)

        # Step 4: Data Quality (on raw)
        quality_report = compute_quality_report(
            df_raw, schema_report.get("target_candidates")
        )

        # Step 5: Clean
        df_clean = clean_dataframe(df_raw)

        # Step 6: Store dataset
        store_dataset(df_raw, df_clean, file_size, uploaded_file.name)

    with st.spinner("Running statistical profiling..."):
        # Step 7: Statistical Profiling (on clean)
        target_col = None
        targets = schema_report.get("target_candidates", [])
        if targets:
            target_col = targets[0]["column"]

        profiling_report = compute_profiling_report(
            df_clean, target_col=target_col, run_clustering=True
        )

        # Step 8: Store reports
        store_reports(schema_report, quality_report, profiling_report)

    with st.spinner("Generating AI executive summary..."):
        # Step 9: Build LLM context
        llm_context = prepare_llm_context(
            df_clean, file_size, schema_report, quality_report
        )
        store_llm_context(llm_context)

        # Step 10: Executive summary
        summary = generate_executive_summary(llm_context)
        store_executive_summary(summary)

    # Done
    mark_analysis_complete()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar():
    """Renders the sidebar with navigation, upload, and dataset info."""
    st.sidebar.title("🤖 AI Data Analyst")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "💬 AI Consultant"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV / Excel)", type=["csv", "xlsx", "xls"]
    )

    # Dataset info card
    info = get_dataset_info_summary()
    if info.get("loaded"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 Dataset Info")
        st.sidebar.markdown(f"**File:** {info['file_name']}")
        st.sidebar.markdown(f"**Size:** {info['rows']:,} rows × {info['columns']} cols")
        st.sidebar.markdown(f"**Memory:** {info['file_size_mb']:.2f} MB")

        if info.get("quality_grade"):
            grade = info["quality_grade"]
            score = info["quality_score"]
            grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}
            icon = grade_colors.get(grade, "⚪")
            st.sidebar.markdown(f"**Quality:** {icon} {score}/100 (Grade {grade})")

        if info.get("target_column"):
            st.sidebar.markdown(f"**Target:** `{info['target_column']}`")

        if info.get("has_time_series"):
            st.sidebar.markdown("**Type:** 📅 Time Series")

    return page, uploaded_file


# ─────────────────────────────────────────────
# Landing Page
# ─────────────────────────────────────────────

def render_landing():
    """Renders the welcome page when no data is loaded."""
    st.title("📊 AI Autonomous Data Analyst")

    st.markdown("""
    ### Welcome to your professional AI data companion.
    
    Upload a CSV or Excel file in the sidebar and the system will **automatically**:
    
    1. 🧠 **Schema Intelligence** — Detect types, targets, time series, primary keys
    2. 🔍 **Data Quality Audit** — Missing values, outliers, duplicates, format issues  
    3. 📈 **Statistical Profiling** — Distributions, correlations, feature importance, clustering
    4. 📊 **16+ Visualizations** — Auto-generated professional charts
    5. 🤖 **AI Executive Summary** — Structured insights, anomalies, recommendations
    6. 💬 **Conversational AI** — Ask any question about your data
    
    *No configuration needed. Just upload and go.*
    """)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 🔬 Mode 1: Auto-EDA
        Instant analysis the moment you upload. Schema detection, quality scoring,
        statistical profiling, and AI insights — all automatic.
        """)
    with c2:
        st.markdown("""
        #### 💬 Mode 2: Chat
        Ask questions in natural language. The AI understands your data context
        and provides analytical answers with code suggestions.
        """)
    with c3:
        st.markdown("""
        #### 📊 16+ Charts
        Distribution plots, correlation heatmaps, outlier visualization,
        time series trends, clustering — all auto-generated.
        """)

    st.info("👈 Upload a CSV or Excel file in the sidebar to begin.")


# ─────────────────────────────────────────────
# Dashboard Page
# ─────────────────────────────────────────────

def render_dashboard():
    """Full analysis dashboard with tabbed sections."""
    if not is_analysis_ready():
        st.info("⏳ Analysis is running... please wait.")
        return

    df = get_clean_data()
    schema = get_schema_report()
    quality = get_quality_report()
    profiling = get_profiling_report()
    summary = get_executive_summary()

    st.header("📊 Analysis Dashboard")

    # ── Tabs ──
    tab_summary, tab_schema, tab_quality, tab_stats, tab_charts = st.tabs([
        "🤖 Executive Summary",
        "🧠 Schema Intelligence",
        "🔍 Data Quality",
        "📈 Statistical Profile",
        "📊 Visualizations",
    ])

    # ── Tab 1: Executive Summary ──
    with tab_summary:
        if summary:
            st.markdown(render_executive_summary_md(summary))

            # Show raw JSON in expander
            with st.expander("📄 View Raw JSON Report"):
                st.json(summary)
        else:
            st.warning("Executive summary not available.")

    # ── Tab 2: Schema Intelligence ──
    with tab_schema:
        st.subheader("📋 Column Analysis")
        schema_df = schema_report_to_dataframe(schema)
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

        # Schema summary
        with st.expander("📝 Schema Summary", expanded=True):
            st.markdown(format_schema_summary(schema))

        # Column categories breakdown
        cats = schema["column_categories"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Numeric Columns", len(cats.get("numeric", [])))
            st.metric("Boolean Columns", len(cats.get("boolean", [])))
        with c2:
            st.metric("Categorical Columns", len(cats.get("categorical", [])))
            st.metric("DateTime Columns", len(cats.get("datetime", [])))
        with c3:
            st.metric("Text Columns", len(cats.get("text", [])))
            st.metric("Identifier Columns", len(cats.get("identifier", [])))

        # Special detections
        if schema["mixed_columns"]:
            st.warning(f"⚠️ {len(schema['mixed_columns'])} mixed-type column(s) detected")
            for mc in schema["mixed_columns"]:
                st.markdown(f"- `{mc['column']}`: types = {mc['types_found']}")

        if schema["primary_keys"]:
            st.success(f"🔑 Primary key candidates: {', '.join(pk['column'] for pk in schema['primary_keys'])}")

        if schema["time_series"]["is_time_series"]:
            ts = schema["time_series"]
            st.info(
                f"📅 Time series detected on `{ts['best_time_column']}` "
                f"— {ts['detected_frequency']} frequency"
            )

    # ── Tab 3: Data Quality ──
    with tab_quality:
        # Generate quality charts
        from utils.plots import plot_quality_gauge, plot_quality_dimensions, plot_missing_values, plot_outliers, plot_class_balance

        # Quality score gauge + dimensions
        c1, c2 = st.columns(2)
        with c1:
            fig = plot_quality_gauge(quality["overall_score"])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = plot_quality_dimensions(quality["overall_score"])
            st.plotly_chart(fig, use_container_width=True)

        # Missing values
        st.subheader("📉 Missing Values")
        st.markdown(quality["missing"]["pattern_summary"])
        miss_df = missing_report_to_dataframe(quality["missing"])
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        fig = plot_missing_values(quality["missing"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Duplicates
        st.subheader("🔄 Duplicates")
        dupes = quality["duplicates"]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Exact Duplicates", f"{dupes['exact_duplicates']:,}")
        with c2:
            st.metric("Duplicate %", f"{dupes['exact_pct']}%")
        st.markdown(dupes["recommendation"])

        # Outliers
        st.subheader("📍 Outliers")
        out_df = outlier_report_to_dataframe(quality["outliers"])
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        fig = plot_outliers(quality["outliers"], df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Class imbalance
        if quality["class_imbalance"]:
            st.subheader("⚖️ Class Imbalance")
            for ci in quality["class_imbalance"][:3]:
                tag = " 🎯" if ci["is_target_column"] else ""
                st.markdown(f"**`{ci['column']}`{tag}** — Ratio: {ci['imbalance_ratio']}x ({ci['severity']})")
                st.markdown(f"_{ci['recommendation']}_")

            fig = plot_class_balance(quality["class_imbalance"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Format issues
        if quality["inconsistencies"]:
            st.subheader("📝 Format Inconsistencies")
            for inc in quality["inconsistencies"]:
                issues = [i["type"] for i in inc["issues"]]
                st.markdown(f"- `{inc['column']}`: {', '.join(issues)}")

    # ── Tab 4: Statistical Profile ──
    with tab_stats:
        from utils.plots import plot_feature_importance, plot_correlation_top_pairs, plot_clustering, plot_silhouette_scores

        # Descriptive stats table
        st.subheader("📊 Descriptive Statistics")
        desc = profiling["descriptive"]
        if not desc["summary_table"].empty:
            st.dataframe(desc["summary_table"], use_container_width=True, hide_index=True)

        if desc["skewed_columns"]:
            st.warning("⚠️ **Skewed columns detected:**")
            for sc in desc["skewed_columns"]:
                st.markdown(f"- `{sc['column']}`: skewness={sc['skewness']} ({sc['direction']}) — {sc['suggestion']}")

        # Correlations
        st.subheader("🔗 Correlations")
        corrs = profiling["correlations"]
        st.markdown(corrs["summary"])
        corr_df = correlation_pairs_to_dataframe(corrs)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        fig = plot_correlation_top_pairs(corrs)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.subheader("🏆 Feature Importance")
        imp = profiling["importance"]
        st.markdown(f"*Method: {imp['method']}*" + (f" | *Target: `{imp['target_used']}`*" if imp["target_used"] else ""))
        fig = plot_feature_importance(imp)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Distribution analysis
        dists = profiling["distributions"]
        normal_cols = [d["column"] for d in dists if d["is_normal"]]
        if normal_cols:
            st.success(f"✅ Normally distributed: {', '.join(normal_cols)}")

        # Clustering
        clust = profiling.get("clustering")
        if clust:
            st.subheader("🔮 Clustering Analysis")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Optimal k", clust["optimal_k"])
            with c2:
                st.metric("Silhouette Score", clust["silhouette_score"])
            with c3:
                st.metric("Data Points", sum(clust["cluster_sizes"].values()))

            num_cols = schema["column_categories"].get("numeric", [])
            fig = plot_clustering(df, clust["labels"], num_cols)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            fig = plot_silhouette_scores(clust["k_scores"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Visualizations ──
    with tab_charts:
        st.subheader("📊 Auto-Generated Visualizations")

        charts = generate_dashboard_charts(df, schema, quality, profiling)

        # Layout: show charts in organized groups
        # Row 1: Distribution + Scatter
        c1, c2 = st.columns(2)
        if charts.get("dist"):
            with c1:
                st.plotly_chart(charts["dist"], use_container_width=True)
        if charts.get("rel"):
            with c2:
                st.plotly_chart(charts["rel"], use_container_width=True)

        # Row 2: Composition + Comparison
        c3, c4 = st.columns(2)
        if charts.get("comp"):
            with c3:
                st.plotly_chart(charts["comp"], use_container_width=True)
        if charts.get("bar"):
            with c4:
                st.plotly_chart(charts["bar"], use_container_width=True)

        # Row 3: Heatmap (full width)
        if charts.get("heat"):
            st.plotly_chart(charts["heat"], use_container_width=True)

        # Row 4: Time series (full width)
        if charts.get("time_series"):
            st.plotly_chart(charts["time_series"], use_container_width=True)

        # Row 5: Distributions grid
        if charts.get("distributions"):
            st.plotly_chart(charts["distributions"], use_container_width=True)


# ─────────────────────────────────────────────
# AI Consultant Page
# ─────────────────────────────────────────────

def render_ai_consultant():
    """Conversational AI data consultant with analytical reasoning."""
    st.title("💬 AI Data Consultant")

    if not is_analysis_ready():
        st.warning("⚠️ Please upload a dataset first. Analysis will run automatically.")
        return

    st.caption("Ask any question about your dataset — the AI has full context of the analysis.")

    # Query suggestions
    with st.expander("💡 Example Questions", expanded=False):
        suggestions = [
            "What does this dataset represent?",
            "Which features are most important?",
            "Are there any concerning patterns in the data?",
            "Show me the Pandas code to filter high-value records.",
            "What model would you recommend for prediction?",
            "Why might column X correlate with column Y?",
            "Suggest feature engineering steps.",
            "How should I handle the outliers?",
        ]
        for s in suggestions:
            st.markdown(f"- *{s}*")

    # Chat history display
    messages = get_messages()
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your dataset..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        add_message("user", prompt)

        # Get enriched context (includes previous query results)
        context = get_conversation_context()

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = query_data_consultant(
                    context,
                    messages,  # Pass history for multi-turn
                    prompt,
                )
                st.markdown(response)

        add_message("assistant", response)
        store_query_result(prompt, response)

    # Query history in sidebar
    if messages:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Query History")
        user_msgs = [m for m in messages if m["role"] == "user"]
        for i, msg in enumerate(user_msgs, 1):
            text = msg["content"][:40]
            st.sidebar.caption(f"{i}. {text}...")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    page, uploaded_file = render_sidebar()

    # Handle file upload (triggers auto-analysis)
    if uploaded_file:
        handle_file_upload(uploaded_file)

    # Route to page
    if not has_dataset():
        render_landing()
    elif page == "📊 Dashboard":
        render_dashboard()
    else:
        render_ai_consultant()


if __name__ == "__main__":
    main()