"""
Data Handler — Enhanced
========================
Loads, validates, and cleans CSV data. Runs schema intelligence and data
quality analysis on the RAW data BEFORE any cleaning, preserving the
original quality picture. Returns cleaned data + full metadata for
downstream modules (dashboard, LLM, visualizations).
"""

import pandas as pd
import streamlit as st
import os
import io
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from utils.schema_intel import (
    get_schema_intelligence_report,
    schema_report_to_dataframe,
    format_schema_summary,
)
from utils.data_quality import (
    get_quality_report,
    format_quality_summary,
)


# ─────────────────────────────────────────────
# 1. File Loading — CSV & Excel with Robust Fallback
# ─────────────────────────────────────────────

def load_file_robust(uploaded_file) -> Tuple[Optional[pd.DataFrame], float]:
    """
    Loads a CSV or Excel file.
    
    - CSV: encoding fallback chain utf-8 → latin-1 → cp1252.
    - Excel (.xlsx, .xls): uses openpyxl / xlrd via pandas.
    
    Returns:
        (DataFrame | None, file_size_mb)
    """
    # Determine file size
    uploaded_file.seek(0, os.SEEK_END)
    file_size_mb = uploaded_file.tell() / (1024 * 1024)
    uploaded_file.seek(0)

    file_name = getattr(uploaded_file, "name", "unknown.csv").lower()

    # ── Excel branch ──
    if file_name.endswith((".xlsx", ".xls")):
        try:
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            return df, round(file_size_mb, 2)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return None, 0

    # ── CSV branch (default) ──
    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            return df, round(file_size_mb, 2)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, 0

    st.error("Failed to read file with any supported encoding (utf-8, latin-1, cp1252).")
    return None, 0


# Backward-compatible alias
load_csv_robust = load_file_robust


# ─────────────────────────────────────────────
# 2. Column Name Normalization
# ─────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names:
      - Strip leading/trailing whitespace
      - Replace spaces and special chars with underscores
      - Convert to lowercase
      - Remove consecutive underscores
    """
    df = df.copy()
    clean_names = []
    for col in df.columns:
        name = str(col).strip()
        name = name.lower()
        # Replace spaces, dashes, dots with underscores
        for char in [" ", "-", ".", "/", "\\", "(", ")", "[", "]"]:
            name = name.replace(char, "_")
        # Remove consecutive underscores
        while "__" in name:
            name = name.replace("__", "_")
        # Strip leading/trailing underscores
        name = name.strip("_")
        clean_names.append(name)

    # Handle duplicate names after normalization
    final_names = []
    seen = {}
    for name in clean_names:
        if name in seen:
            seen[name] += 1
            final_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            final_names.append(name)

    df.columns = final_names
    return df


# ─────────────────────────────────────────────
# 3. Data Cleaning
# ─────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame:
      - Numeric columns: fill NaN with median
      - Categorical columns: fill NaN with mode (or 'Unknown')
      - Strip whitespace from string values
      - Drop fully empty rows
      - Reset index
    """
    df = df.copy()

    # Drop rows where ALL values are NaN
    df = df.dropna(how="all")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            # Strip whitespace from string values first
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            # Fill remaining NaN with mode
            modes = df[col].mode()
            mode_val = modes.iloc[0] if not modes.empty else "Unknown"
            df[col] = df[col].fillna(mode_val)

    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 4. Master Pipeline — Load → Analyze Raw → Clean
# ─────────────────────────────────────────────

@st.cache_data
def load_and_process_data(uploaded_file) -> Dict[str, Any]:
    """
    Full data pipeline:
      1. Load file — CSV or Excel (robust encoding)
      2. Normalize column names
      3. Run schema intelligence on RAW data
      4. Run data quality analysis on RAW data
      5. Clean the data
      6. Return everything

    Returns:
        {
            "df_raw": pd.DataFrame,          # Original with normalized names
            "df_clean": pd.DataFrame,         # Cleaned version
            "file_size_mb": float,
            "schema_report": dict,            # From schema_intel
            "quality_report": dict,           # From data_quality
            "column_name_map": dict,          # original → normalized name mapping
        }
      Returns None on failure.
    """
    # Step 1: Load
    df_raw, file_size_mb = load_file_robust(uploaded_file)
    if df_raw is None:
        return None

    # Step 2: Store original names, then normalize
    original_names = df_raw.columns.tolist()
    df_raw = normalize_column_names(df_raw)
    name_map = dict(zip(original_names, df_raw.columns.tolist()))

    # Step 3: Schema intelligence on RAW data
    schema_report = get_schema_intelligence_report(df_raw)

    # Step 4: Data quality on RAW data (with target candidates from schema)
    quality_report = get_quality_report(
        df_raw,
        target_candidates=schema_report.get("target_candidates"),
    )

    # Step 5: Clean
    df_clean = clean_dataframe(df_raw)

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "file_size_mb": file_size_mb,
        "schema_report": schema_report,
        "quality_report": quality_report,
        "column_name_map": name_map,
    }


# ─────────────────────────────────────────────
# 5. Backward-Compatible Wrappers
# ─────────────────────────────────────────────
# These keep the old API working so app.py doesn't break
# until we upgrade the dashboard in a later step.

@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Legacy wrapper — returns (df_clean, file_size_mb).
    Calls the new pipeline internally.
    """
    result = load_and_process_data(uploaded_file)
    if result is None:
        return None, 0
    return result["df_clean"], result["file_size_mb"]


def get_col_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Returns (numeric_columns, categorical_columns) for chart generation."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if df[col].nunique() < 30
    ]
    return num_cols, cat_cols


def get_dataset_description(df: pd.DataFrame) -> str:
    """One-line dataset description."""
    return f"This dataset has {df.shape[0]:,} rows and {df.shape[1]} columns."


def get_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy schema table. Still works but the dashboard will
    switch to schema_report_to_dataframe() in Phase 6.
    """
    schema_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": [df[col].nunique() for col in df.columns],
    })
    return schema_df


# ─────────────────────────────────────────────
# 6. Enhanced LLM Context Builder
# ─────────────────────────────────────────────

def prepare_llm_context(
    df: pd.DataFrame,
    file_size_mb: float,
    schema_report: Optional[Dict] = None,
    quality_report: Optional[Dict] = None,
) -> str:
    """
    Builds a rich textual context for the LLM, incorporating:
      - Dataset summary statistics
      - Schema intelligence summary (if available)
      - Data quality summary (if available)
      - Representative data sample

    Falls back gracefully if reports aren't provided (backward compat).
    """
    sections = []

    # Section 1: Basic stats
    sections.append("=" * 50)
    sections.append("DATASET OVERVIEW")
    sections.append("=" * 50)
    sections.append(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    sections.append(f"File Size: {file_size_mb:.2f} MB")
    sections.append(f"Columns: {', '.join(df.columns.tolist())}")
    sections.append("")

    # Section 2: Schema intelligence
    if schema_report:
        sections.append("=" * 50)
        sections.append("SCHEMA INTELLIGENCE")
        sections.append("=" * 50)
        sections.append(format_schema_summary(schema_report))
        sections.append("")

    # Section 3: Data quality
    if quality_report:
        sections.append("=" * 50)
        sections.append("DATA QUALITY ASSESSMENT")
        sections.append("=" * 50)
        sections.append(format_quality_summary(quality_report))
        sections.append("")

    # Section 4: Descriptive statistics
    sections.append("=" * 50)
    sections.append("DESCRIPTIVE STATISTICS")
    sections.append("=" * 50)
    sections.append(df.describe(include="all").to_string())
    sections.append("")

    # Section 5: Data sample (size-aware)
    sections.append("=" * 50)
    sections.append("DATA SAMPLE")
    sections.append("=" * 50)
    if file_size_mb > 50:
        sample_df = df.sample(n=min(500, len(df)), random_state=42)
        sections.append(
            f"NOTE: Large dataset ({file_size_mb:.1f} MB). "
            f"Showing random sample of {len(sample_df)} rows."
        )
    else:
        sample_df = df.head(min(100, len(df)))
        sections.append(f"Showing first {len(sample_df)} rows.")

    sections.append(sample_df.to_string())

    return "\n".join(sections)