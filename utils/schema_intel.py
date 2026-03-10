"""
Schema Intelligence Engine
===========================
Analyzes dataset structure: data types, mixed columns, primary keys,
target-like columns, and time-series patterns. Runs automatically on
CSV upload to build a rich metadata layer for downstream modules.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


# ─────────────────────────────────────────────
# 1. Enhanced Data Type Detection
# ─────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Returns a per-column dictionary with detected semantic type, 
    pandas dtype, sample values, and uniqueness ratio.
    
    Semantic types: numeric, categorical, boolean, datetime, text, identifier
    """
    type_report = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        pandas_dtype = str(series.dtype)
        n_unique = series.nunique()
        null_pct = round(series.isnull().mean() * 100, 2)
        uniqueness_ratio = round(n_unique / n_rows, 4) if n_rows > 0 else 0

        # Determine semantic type
        semantic_type = _infer_semantic_type(series, n_unique, n_rows)

        type_report[col] = {
            "pandas_dtype": pandas_dtype,
            "semantic_type": semantic_type,
            "n_unique": n_unique,
            "uniqueness_ratio": uniqueness_ratio,
            "null_count": int(series.isnull().sum()),
            "null_pct": null_pct,
            "sample_values": series.dropna().head(3).tolist(),
        }

    return type_report


def _infer_semantic_type(series: pd.Series, n_unique: int, n_rows: int) -> str:
    """Infers semantic type from a pandas Series."""

    # Boolean check first — exactly 2 unique non-null values
    non_null = series.dropna()
    if n_unique == 2:
        vals = set(non_null.unique())
        bool_patterns = [
            {0, 1}, {True, False}, {"yes", "no"}, {"true", "false"},
            {"y", "n"}, {"0", "1"}, {"Yes", "No"}, {"True", "False"},
            {"Y", "N"}, {"Male", "Female"}, {"male", "female"},
            {"M", "F"}, {"m", "f"},
        ]
        if vals in bool_patterns:
            return "boolean"

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 2:
            return "boolean"
        if n_unique <= 20 and n_rows > 50:
            return "categorical"  # Low-cardinality numeric acts categorical
        return "numeric"

    # Datetime — try parsing
    if _is_datetime_like(series):
        return "datetime"

    # Object / String
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        if n_unique == n_rows and n_rows > 10:
            return "identifier"
        if n_unique <= 30 or (n_unique / n_rows < 0.05 and n_rows > 100):
            return "categorical"
        avg_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
        if avg_len > 50:
            return "text"
        return "categorical"

    return "unknown"


def _is_datetime_like(series: pd.Series) -> bool:
    """Attempts to detect if a series contains datetime-like values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    # Try to parse a small sample of non-null string values
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        success_rate = parsed.notna().mean()
        return success_rate >= 0.8  # 80%+ successfully parsed
    except Exception:
        return False


# ─────────────────────────────────────────────
# 2. Mixed Column Detection
# ─────────────────────────────────────────────

def detect_mixed_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identifies columns containing mixed data types (e.g., numbers and strings
    coexisting in the same column). This is a common data-quality signal.
    
    Returns a list of dicts:
        {"column": str, "types_found": list, "sample_conflicts": list}
    """
    mixed = []

    for col in df.columns:
        # Only relevant for object-dtype columns (strings / mixed)
        if df[col].dtype != object:
            continue

        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        # Check what Python types are present
        type_counts: Dict[str, int] = {}
        samples_by_type: Dict[str, list] = {}
        for val in non_null.head(500):  # Sample for performance
            t = type(val).__name__
            type_counts[t] = type_counts.get(t, 0) + 1
            if t not in samples_by_type:
                samples_by_type[t] = []
            if len(samples_by_type[t]) < 3:
                samples_by_type[t].append(val)

        if len(type_counts) > 1:
            mixed.append({
                "column": col,
                "types_found": list(type_counts.keys()),
                "type_counts": type_counts,
                "sample_conflicts": {k: v for k, v in samples_by_type.items()},
            })

        # Also flag "numeric strings" — object col that is mostly numeric
        elif "str" in type_counts:
            numeric_parseable = pd.to_numeric(non_null, errors="coerce").notna().mean()
            if 0.3 < numeric_parseable < 0.95:
                mixed.append({
                    "column": col,
                    "types_found": ["str (mixed numeric/text)"],
                    "type_counts": {"numeric_parseable_pct": round(numeric_parseable * 100, 1)},
                    "sample_conflicts": non_null.head(5).tolist(),
                })

    return mixed


# ─────────────────────────────────────────────
# 3. Target-like Column Detection
# ─────────────────────────────────────────────

def identify_target_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Heuristically identifies columns that are likely prediction targets
    based on naming patterns, cardinality, and distribution.
    
    Scoring logic:
      - Name contains 'target', 'label', 'class', 'outcome', 'flag', 'status' → +3
      - Binary column (2 unique values) → +2
      - Low cardinality categorical (2-10 unique) → +1
      - Appears to be an imbalanced binary → +1 (indicates classification target)
    """
    target_keywords = [
        "target", "label", "class", "outcome", "result", "flag",
        "status", "churn", "default", "fraud", "response", "survived",
        "diagnosis", "prediction", "y", "output",
    ]

    candidates = []
    n_rows = len(df)

    for col in df.columns:
        score = 0
        reasons = []
        col_lower = col.lower().strip()

        # Name-based scoring
        for keyword in target_keywords:
            if keyword in col_lower:
                score += 3
                reasons.append(f"Name contains '{keyword}'")
                break

        n_unique = df[col].nunique()

        # Binary column
        if n_unique == 2:
            score += 2
            reasons.append("Binary column (2 unique values)")
            # Check imbalance
            value_counts = df[col].value_counts(normalize=True)
            minority_pct = value_counts.min() * 100
            if minority_pct < 30:
                score += 1
                reasons.append(f"Imbalanced ({minority_pct:.1f}% minority class)")

        # Low cardinality
        elif 2 < n_unique <= 10 and n_rows > 50:
            score += 1
            reasons.append(f"Low cardinality ({n_unique} classes)")

        if score >= 2:
            candidates.append({
                "column": col,
                "score": score,
                "reasons": reasons,
                "n_unique": n_unique,
                "dtype": str(df[col].dtype),
            })

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


# ─────────────────────────────────────────────
# 4. Primary Key Detection
# ─────────────────────────────────────────────

def detect_primary_keys(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identifies columns that could serve as primary keys.
    
    Criteria:
      - 100% unique values (uniqueness_ratio == 1.0)
      - No null values
      - Prefers integer or sequential patterns
    """
    pk_candidates = []
    n_rows = len(df)

    if n_rows == 0:
        return pk_candidates

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique()
        has_nulls = series.isnull().any()

        if n_unique == n_rows and not has_nulls:
            pk_info = {
                "column": col,
                "dtype": str(series.dtype),
                "is_sequential": False,
                "is_integer": False,
                "confidence": "high",
            }

            # Check if integer type
            if pd.api.types.is_integer_dtype(series):
                pk_info["is_integer"] = True
                # Check if monotonically increasing (sequential ID)
                if series.is_monotonic_increasing:
                    pk_info["is_sequential"] = True
                    pk_info["confidence"] = "very_high"

            # String IDs — check if they have a consistent pattern
            elif series.dtype == object:
                lengths = series.astype(str).str.len()
                if lengths.std() == 0:
                    pk_info["confidence"] = "very_high"  # All same length → ID format

            pk_candidates.append(pk_info)

    return pk_candidates


# ─────────────────────────────────────────────
# 5. Time Series Structure Detection
# ─────────────────────────────────────────────

def detect_time_series(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects if the dataset has time-series properties.
    
    Returns:
        {
            "is_time_series": bool,
            "datetime_columns": [...],
            "best_time_column": str | None,
            "detected_frequency": str | None,
            "temporal_span": {...} | None,
        }
    """
    result: Dict[str, Any] = {
        "is_time_series": False,
        "datetime_columns": [],
        "best_time_column": None,
        "detected_frequency": None,
        "temporal_span": None,
    }

    datetime_cols = []

    for col in df.columns:
        series = df[col]

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            datetime_cols.append(col)
            continue

        # Try to parse object columns
        if series.dtype == object:
            if _is_datetime_like(series):
                datetime_cols.append(col)

    result["datetime_columns"] = datetime_cols

    if not datetime_cols:
        return result

    result["is_time_series"] = True

    # Pick the best datetime column (prefer columns named date/time/timestamp)
    best_col = _pick_best_datetime_col(datetime_cols, df)
    result["best_time_column"] = best_col

    # Analyse temporal span and frequency
    try:
        dt_series = pd.to_datetime(df[best_col], errors="coerce").dropna().sort_values()
        if len(dt_series) > 1:
            result["temporal_span"] = {
                "start": str(dt_series.iloc[0]),
                "end": str(dt_series.iloc[-1]),
                "duration_days": (dt_series.iloc[-1] - dt_series.iloc[0]).days,
                "n_records": len(dt_series),
            }
            # Detect frequency
            freq = _detect_frequency(dt_series)
            result["detected_frequency"] = freq
    except Exception:
        pass

    return result


def _pick_best_datetime_col(datetime_cols: List[str], df: pd.DataFrame) -> str:
    """Picks the most likely primary datetime column from a list."""
    time_keywords = ["date", "time", "timestamp", "datetime", "created", "updated", "period"]
    
    for col in datetime_cols:
        for kw in time_keywords:
            if kw in col.lower():
                return col
    
    # Fallback: pick the one with the most unique values (finest granularity)
    best = max(datetime_cols, key=lambda c: df[c].nunique())
    return best


def _detect_frequency(dt_series: pd.Series) -> Optional[str]:
    """Detects the most common time interval in a datetime series."""
    if len(dt_series) < 3:
        return None
    
    diffs = dt_series.diff().dropna()
    if len(diffs) == 0:
        return None

    median_diff = diffs.median()
    days = median_diff.days

    if days == 0:
        hours = median_diff.total_seconds() / 3600
        if hours < 1:
            return "sub-hourly"
        elif hours <= 1.5:
            return "hourly"
        else:
            return "intra-day"
    elif days == 1:
        return "daily"
    elif 6 <= days <= 8:
        return "weekly"
    elif 13 <= days <= 16:
        return "bi-weekly"
    elif 27 <= days <= 33:
        return "monthly"
    elif 85 <= days <= 95:
        return "quarterly"
    elif 360 <= days <= 370:
        return "yearly"
    else:
        return f"~{days}-day intervals"


# ─────────────────────────────────────────────
# 6. Full Schema Intelligence Report
# ─────────────────────────────────────────────

def get_schema_intelligence_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Master function: runs all schema intelligence checks and returns
    a unified report dictionary used by downstream modules.
    
    Returns:
        {
            "shape": (rows, cols),
            "memory_mb": float,
            "column_types": {...},
            "mixed_columns": [...],
            "target_candidates": [...],
            "primary_keys": [...],
            "time_series": {...},
            "column_categories": {
                "numeric": [...], "categorical": [...], "datetime": [...],
                "boolean": [...], "text": [...], "identifier": [...]
            }
        }
    """
    # Core analyses
    col_types = detect_column_types(df)
    mixed = detect_mixed_columns(df)
    targets = identify_target_columns(df)
    pks = detect_primary_keys(df)
    ts = detect_time_series(df)

    # Build categorized column lists 
    column_categories: Dict[str, List[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "text": [],
        "identifier": [],
        "unknown": [],
    }
    for col, info in col_types.items():
        stype = info["semantic_type"]
        if stype in column_categories:
            column_categories[stype].append(col)
        else:
            column_categories["unknown"].append(col)

    # Memory usage
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    return {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "memory_mb": memory_mb,
        "column_types": col_types,
        "mixed_columns": mixed,
        "target_candidates": targets,
        "primary_keys": pks,
        "time_series": ts,
        "column_categories": column_categories,
    }


# ─────────────────────────────────────────────
# 7. Schema Report → Display-ready DataFrames
# ─────────────────────────────────────────────

def schema_report_to_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts the column_types section of the schema report into a 
    DataFrame for rendering in the Streamlit UI.
    """
    rows = []
    for col, info in report["column_types"].items():
        rows.append({
            "Column": col,
            "Pandas Type": info["pandas_dtype"],
            "Semantic Type": info["semantic_type"].title(),
            "Unique Values": info["n_unique"],
            "Uniqueness %": f"{info['uniqueness_ratio'] * 100:.1f}%",
            "Nulls": info["null_count"],
            "Null %": f"{info['null_pct']}%",
            "Sample Values": str(info["sample_values"][:3]),
        })
    return pd.DataFrame(rows)


def format_schema_summary(report: Dict[str, Any]) -> str:
    """
    Returns a concise markdown summary string for the schema report,
    suitable for embedding in LLM prompts or displaying as text.
    """
    shape = report["shape"]
    cats = report["column_categories"]
    ts = report["time_series"]
    
    lines = [
        f"**Dataset Shape:** {shape['rows']:,} rows × {shape['cols']} columns",
        f"**Memory Usage:** {report['memory_mb']} MB",
        "",
        "**Column Breakdown:**",
        f"  - Numeric: {len(cats['numeric'])} columns",
        f"  - Categorical: {len(cats['categorical'])} columns",
        f"  - Boolean: {len(cats['boolean'])} columns",
        f"  - DateTime: {len(cats['datetime'])} columns",
        f"  - Text: {len(cats['text'])} columns",
        f"  - Identifier: {len(cats['identifier'])} columns",
    ]

    if report["mixed_columns"]:
        lines.append(f"\n⚠️ **Mixed Columns Detected:** {len(report['mixed_columns'])}")
        for mc in report["mixed_columns"]:
            lines.append(f"  - `{mc['column']}`: types = {mc['types_found']}")

    if report["primary_keys"]:
        pk_names = [pk["column"] for pk in report["primary_keys"]]
        lines.append(f"\n🔑 **Primary Key Candidates:** {', '.join(pk_names)}")

    if report["target_candidates"]:
        top = report["target_candidates"][0]
        lines.append(f"\n🎯 **Likely Target Column:** `{top['column']}` (score={top['score']})")
        for r in top["reasons"]:
            lines.append(f"  - {r}")

    if ts["is_time_series"]:
        lines.append(f"\n📅 **Time Series Detected:**")
        lines.append(f"  - Best column: `{ts['best_time_column']}`")
        if ts["detected_frequency"]:
            lines.append(f"  - Frequency: {ts['detected_frequency']}")
        if ts["temporal_span"]:
            span = ts["temporal_span"]
            lines.append(f"  - Span: {span['start']} → {span['end']} ({span['duration_days']} days)")

    return "\n".join(lines)
