"""
Data Quality Analysis Engine
==============================
Audits dataset health: missing values, duplicates, inconsistent formats,
outliers (IQR-based), and class imbalance. Produces a unified quality
report consumed by the dashboard, LLM context, and visualization modules.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple


# ─────────────────────────────────────────────
# 1. Missing Values Analysis
# ─────────────────────────────────────────────

def missing_values_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive missing-value audit.

    Returns:
        {
            "total_missing": int,
            "total_cells": int,
            "overall_pct": float,
            "columns": [
                {
                    "column": str,
                    "missing_count": int,
                    "missing_pct": float,
                    "severity": "none" | "low" | "moderate" | "high" | "critical"
                }, ...
            ],
            "pattern_summary": str,
            "rows_with_any_missing": int,
            "rows_complete": int,
        }
    """
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    col_reports = []
    total_missing = 0

    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        total_missing += n_miss
        pct = round(n_miss / n_rows * 100, 2) if n_rows > 0 else 0.0

        severity = _missing_severity(pct)

        col_reports.append({
            "column": col,
            "missing_count": n_miss,
            "missing_pct": pct,
            "severity": severity,
        })

    # Sort by missing count descending
    col_reports.sort(key=lambda x: x["missing_count"], reverse=True)

    rows_with_missing = int(df.isnull().any(axis=1).sum())
    rows_complete = n_rows - rows_with_missing

    overall_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0.0

    # Pattern summary
    cols_with_missing = [c for c in col_reports if c["missing_count"] > 0]
    if not cols_with_missing:
        pattern = "No missing values detected — dataset is fully complete."
    elif len(cols_with_missing) <= 3:
        names = [c["column"] for c in cols_with_missing]
        pattern = f"Missing values concentrated in {len(cols_with_missing)} column(s): {', '.join(names)}."
    else:
        pattern = f"Missing values spread across {len(cols_with_missing)} columns — may indicate systemic data collection issues."

    return {
        "total_missing": total_missing,
        "total_cells": total_cells,
        "overall_pct": overall_pct,
        "columns": col_reports,
        "pattern_summary": pattern,
        "rows_with_any_missing": rows_with_missing,
        "rows_complete": rows_complete,
    }


def _missing_severity(pct: float) -> str:
    """Classifies missing-value percentage into severity buckets."""
    if pct == 0:
        return "none"
    elif pct < 5:
        return "low"
    elif pct < 15:
        return "moderate"
    elif pct < 40:
        return "high"
    else:
        return "critical"


# ─────────────────────────────────────────────
# 2. Duplicate Detection
# ─────────────────────────────────────────────

def duplicate_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detects exact duplicate rows and near-duplicate patterns.

    Returns:
        {
            "exact_duplicates": int,
            "exact_pct": float,
            "duplicate_subset_candidates": [
                {"columns": [...], "duplicates": int}, ...
            ],
            "recommendation": str,
        }
    """
    n_rows = len(df)
    exact_dupes = int(df.duplicated().sum())
    exact_pct = round(exact_dupes / n_rows * 100, 2) if n_rows > 0 else 0.0

    # Check subsets — find column combos that create a lot of duplication
    subset_candidates = []

    # Check all non-numeric columns (identifiers often create key constraints)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols and len(non_numeric_cols) <= 15:
        for col in non_numeric_cols:
            dupes = int(df.duplicated(subset=[col]).sum())
            if dupes > 0:
                subset_candidates.append({
                    "columns": [col],
                    "duplicates": dupes,
                    "pct": round(dupes / n_rows * 100, 1),
                })

    # Sort by duplicate count
    subset_candidates.sort(key=lambda x: x["duplicates"], reverse=True)
    subset_candidates = subset_candidates[:5]  # Top 5

    # Recommendation
    if exact_dupes == 0:
        recommendation = "No exact duplicate rows found. Data appears clean."
    elif exact_pct < 1:
        recommendation = f"Minor duplication: {exact_dupes} rows ({exact_pct}%). Likely safe to deduplicate."
    elif exact_pct < 10:
        recommendation = f"Moderate duplication: {exact_dupes} rows ({exact_pct}%). Review before removal — could be valid repeated records."
    else:
        recommendation = f"Heavy duplication: {exact_dupes} rows ({exact_pct}%). Investigate data pipeline for repeat ingestion."

    return {
        "exact_duplicates": exact_dupes,
        "exact_pct": exact_pct,
        "duplicate_subset_candidates": subset_candidates,
        "recommendation": recommendation,
    }


# ─────────────────────────────────────────────
# 3. Inconsistent Format Detection
# ─────────────────────────────────────────────

def inconsistent_format_check(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detects formatting inconsistencies in string/object columns:
      - Leading/trailing whitespace
      - Mixed case variants of the same value
      - Inconsistent separators or special characters

    Returns a list of issue dicts per column.
    """
    issues = []

    obj_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in obj_cols:
        col_issues = []
        non_null = df[col].dropna()

        if len(non_null) == 0:
            continue

        str_vals = non_null.astype(str)

        # 1. Whitespace issues
        has_leading = (str_vals != str_vals.str.lstrip()).any()
        has_trailing = (str_vals != str_vals.str.rstrip()).any()
        if has_leading or has_trailing:
            affected = int(((str_vals != str_vals.str.strip())).sum())
            col_issues.append({
                "type": "whitespace",
                "description": "Leading/trailing whitespace found",
                "affected_count": affected,
            })

        # 2. Mixed case variants
        case_groups = _find_case_variants(str_vals)
        if case_groups:
            col_issues.append({
                "type": "mixed_case",
                "description": "Same value appears with different casing",
                "variants": case_groups[:5],  # Top 5 groups
            })

        # 3. Empty strings (different from NaN)
        empty_count = int((str_vals.str.strip() == "").sum())
        if empty_count > 0:
            col_issues.append({
                "type": "empty_strings",
                "description": "Empty strings found (not NaN)",
                "affected_count": empty_count,
            })

        if col_issues:
            issues.append({
                "column": col,
                "issues": col_issues,
                "issue_count": len(col_issues),
            })

    return issues


def _find_case_variants(series: pd.Series) -> List[Dict[str, Any]]:
    """
    Finds values that differ only by case.
    Example: 'New York', 'new york', 'NEW YORK' → grouped together.
    """
    value_counts = series.value_counts()
    lower_map: Dict[str, List[str]] = {}

    for val in value_counts.index:
        key = str(val).strip().lower()
        if key not in lower_map:
            lower_map[key] = []
        lower_map[key].append(str(val))

    # Only return groups where multiple case variants exist
    variants = []
    for key, vals in lower_map.items():
        if len(vals) > 1:
            counts = {v: int(value_counts.get(v, 0)) for v in vals}
            variants.append({
                "canonical": key,
                "variants": counts,
                "total": sum(counts.values()),
            })

    variants.sort(key=lambda x: x["total"], reverse=True)
    return variants


# ─────────────────────────────────────────────
# 4. Outlier Detection (IQR Method)
# ─────────────────────────────────────────────

def outlier_detection(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict[str, Any]:
    """
    IQR-based outlier detection for all numeric columns.

    Returns:
        {
            "total_outliers": int,
            "columns": [
                {
                    "column": str,
                    "outlier_count": int,
                    "outlier_pct": float,
                    "lower_bound": float,
                    "upper_bound": float,
                    "min_val": float,
                    "max_val": float,
                    "severity": str,
                    "sample_outliers": [...]
                }, ...
            ],
            "summary": str,
        }
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    total_outliers = 0
    col_reports = []

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Guard against zero IQR (constant or near-constant columns)
        if iqr == 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_mask = (series < lower) | (series > upper)
        n_outliers = int(outlier_mask.sum())
        total_outliers += n_outliers
        pct = round(n_outliers / len(series) * 100, 2)

        severity = _outlier_severity(pct)

        # Collect sample outlier values
        sample_outliers = series[outlier_mask].head(5).tolist() if n_outliers > 0 else []

        col_reports.append({
            "column": col,
            "outlier_count": n_outliers,
            "outlier_pct": pct,
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "min_val": round(float(series.min()), 4),
            "max_val": round(float(series.max()), 4),
            "severity": severity,
            "sample_outliers": [round(float(v), 4) for v in sample_outliers],
        })

    # Sort by outlier count
    col_reports.sort(key=lambda x: x["outlier_count"], reverse=True)

    cols_with_outliers = [c for c in col_reports if c["outlier_count"] > 0]
    if not cols_with_outliers:
        summary = "No significant outliers detected across numeric columns."
    elif len(cols_with_outliers) <= 2:
        names = [f"{c['column']} ({c['outlier_count']})" for c in cols_with_outliers]
        summary = f"Outliers found in {len(cols_with_outliers)} column(s): {', '.join(names)}."
    else:
        summary = f"Outliers detected in {len(cols_with_outliers)} columns. Total outlier data points: {total_outliers}."

    return {
        "total_outliers": total_outliers,
        "columns": col_reports,
        "summary": summary,
    }


def _outlier_severity(pct: float) -> str:
    """Classifies outlier percentage into severity."""
    if pct == 0:
        return "none"
    elif pct < 2:
        return "low"
    elif pct < 5:
        return "moderate"
    elif pct < 15:
        return "high"
    else:
        return "extreme"


# ─────────────────────────────────────────────
# 5. Class Imbalance Detection
# ─────────────────────────────────────────────

def class_imbalance_check(
    df: pd.DataFrame,
    target_candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Checks for class imbalance in target-like or low-cardinality categorical columns.
    Accepts target_candidates from the schema_intel module to focus analysis.

    Returns a list of imbalance reports per column.
    """
    results = []

    # Determine which columns to check
    columns_to_check = []
    if target_candidates:
        columns_to_check = [tc["column"] for tc in target_candidates if tc["column"] in df.columns]

    # Also check any categorical column with <= 15 unique values
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 15 and col not in columns_to_check:
            columns_to_check.append(col)

    for col in columns_to_check:
        if col not in df.columns:
            continue

        vc = df[col].value_counts(normalize=True)
        if len(vc) < 2:
            continue

        majority_pct = round(vc.iloc[0] * 100, 1)
        minority_pct = round(vc.iloc[-1] * 100, 1)
        imbalance_ratio = round(vc.iloc[0] / vc.iloc[-1], 2) if vc.iloc[-1] > 0 else float("inf")

        is_target = any(tc["column"] == col for tc in (target_candidates or []))

        severity = _imbalance_severity(imbalance_ratio)

        results.append({
            "column": col,
            "n_classes": len(vc),
            "majority_class": str(vc.index[0]),
            "majority_pct": majority_pct,
            "minority_class": str(vc.index[-1]),
            "minority_pct": minority_pct,
            "imbalance_ratio": imbalance_ratio,
            "is_target_column": is_target,
            "severity": severity,
            "class_distribution": {str(k): round(v * 100, 1) for k, v in vc.items()},
            "recommendation": _imbalance_recommendation(severity, is_target, col),
        })

    # Sort: target columns first, then by imbalance ratio
    results.sort(key=lambda x: (-x["is_target_column"], -x["imbalance_ratio"]))
    return results


def _imbalance_severity(ratio: float) -> str:
    """Classifies imbalance ratio into severity."""
    if ratio < 1.5:
        return "balanced"
    elif ratio < 3:
        return "mild"
    elif ratio < 10:
        return "moderate"
    elif ratio < 50:
        return "severe"
    else:
        return "extreme"


def _imbalance_recommendation(severity: str, is_target: bool, col: str) -> str:
    """Generates recommendations based on imbalance severity."""
    if severity == "balanced":
        return f"'{col}' has well-balanced class distribution."
    
    prefix = f"Target column '{col}'" if is_target else f"Column '{col}'"

    if severity == "mild":
        return f"{prefix} shows mild imbalance. Standard ML models should handle this."
    elif severity == "moderate":
        return f"{prefix} has moderate imbalance. Consider stratified sampling or class weights."
    elif severity == "severe":
        return f"{prefix} is severely imbalanced. Use SMOTE, class weights, or specialized techniques."
    else:
        return f"{prefix} has extreme imbalance. Anomaly detection approach may be more appropriate than classification."


# ─────────────────────────────────────────────
# 6. Full Data Quality Report
# ─────────────────────────────────────────────

def get_quality_report(
    df: pd.DataFrame,
    target_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Master function: runs all data quality checks and produces a unified report.

    Args:
        df: The DataFrame to analyse.
        target_candidates: Optional list from schema_intel.identify_target_columns().

    Returns a dictionary with keys:
        missing, duplicates, inconsistencies, outliers, class_imbalance, overall_score
    """
    missing = missing_values_report(df)
    dupes = duplicate_detection(df)
    inconsistencies = inconsistent_format_check(df)
    outliers = outlier_detection(df)
    imbalance = class_imbalance_check(df, target_candidates)

    # Compute overall quality score (0-100)
    score = _compute_quality_score(missing, dupes, outliers, inconsistencies)

    return {
        "missing": missing,
        "duplicates": dupes,
        "inconsistencies": inconsistencies,
        "outliers": outliers,
        "class_imbalance": imbalance,
        "overall_score": score,
    }


def _compute_quality_score(
    missing: Dict, dupes: Dict, outliers: Dict, inconsistencies: List,
) -> Dict[str, Any]:
    """
    Computes a 0-100 quality score across four dimensions.
    Each dimension contributes 25 points to the total.
    """
    # Completeness (25 pts) — penalise for missing values
    miss_pct = missing["overall_pct"]
    completeness = max(0, 25 - (miss_pct * 0.5))

    # Uniqueness (25 pts) — penalise for duplicates
    dupe_pct = dupes["exact_pct"]
    uniqueness = max(0, 25 - (dupe_pct * 0.5))

    # Consistency (25 pts) — penalise for format issues
    n_issues = sum(len(i["issues"]) for i in inconsistencies)
    consistency = max(0, 25 - (n_issues * 2))

    # Validity (25 pts) — penalise for outliers
    outlier_cols = [c for c in outliers["columns"] if c["outlier_count"] > 0]
    avg_outlier_pct = (
        np.mean([c["outlier_pct"] for c in outlier_cols]) if outlier_cols else 0
    )
    validity = max(0, 25 - (avg_outlier_pct * 1.5))

    total = round(completeness + uniqueness + consistency + validity, 1)

    # Grade
    if total >= 90:
        grade = "A"
    elif total >= 75:
        grade = "B"
    elif total >= 60:
        grade = "C"
    elif total >= 40:
        grade = "D"
    else:
        grade = "F"

    return {
        "total": total,
        "grade": grade,
        "dimensions": {
            "completeness": round(completeness, 1),
            "uniqueness": round(uniqueness, 1),
            "consistency": round(consistency, 1),
            "validity": round(validity, 1),
        },
    }


# ─────────────────────────────────────────────
# 7. Display Formatters
# ─────────────────────────────────────────────

def missing_report_to_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    """Converts missing values report to a display-ready DataFrame."""
    if not report["columns"]:
        return pd.DataFrame()

    rows = []
    for col_info in report["columns"]:
        if col_info["missing_count"] > 0:
            rows.append({
                "Column": col_info["column"],
                "Missing": col_info["missing_count"],
                "% Missing": f"{col_info['missing_pct']}%",
                "Severity": col_info["severity"].title(),
            })

    if not rows:
        return pd.DataFrame({"Status": ["No missing values detected"]})

    return pd.DataFrame(rows)


def outlier_report_to_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    """Converts outlier report to a display-ready DataFrame."""
    rows = []
    for col_info in report["columns"]:
        if col_info["outlier_count"] > 0:
            rows.append({
                "Column": col_info["column"],
                "Outliers": col_info["outlier_count"],
                "% Outlier": f"{col_info['outlier_pct']}%",
                "Lower Bound": col_info["lower_bound"],
                "Upper Bound": col_info["upper_bound"],
                "Severity": col_info["severity"].title(),
            })

    if not rows:
        return pd.DataFrame({"Status": ["No significant outliers detected"]})

    return pd.DataFrame(rows)


def format_quality_summary(report: Dict[str, Any]) -> str:
    """
    Returns a concise markdown string summarising the data quality report,
    suitable for LLM prompts or display.
    """
    score = report["overall_score"]
    missing = report["missing"]
    dupes = report["duplicates"]
    outliers = report["outliers"]
    imbalance = report["class_imbalance"]
    dims = score["dimensions"]

    lines = [
        f"## Data Quality Score: {score['total']}/100 (Grade: {score['grade']})",
        "",
        "**Dimension Scores:**",
        f"  - Completeness: {dims['completeness']}/25",
        f"  - Uniqueness: {dims['uniqueness']}/25",
        f"  - Consistency: {dims['consistency']}/25",
        f"  - Validity: {dims['validity']}/25",
        "",
        f"**Missing Values:** {missing['total_missing']:,} total ({missing['overall_pct']}%)",
        f"  - {missing['pattern_summary']}",
        f"  - Complete rows: {missing['rows_complete']:,} / {missing['rows_complete'] + missing['rows_with_any_missing']:,}",
        "",
        f"**Duplicates:** {dupes['exact_duplicates']:,} exact ({dupes['exact_pct']}%)",
        f"  - {dupes['recommendation']}",
        "",
        f"**Outliers:** {outliers['total_outliers']:,} data points",
        f"  - {outliers['summary']}",
    ]

    if imbalance:
        lines.append("")
        lines.append("**Class Imbalance:**")
        for ci in imbalance[:3]:
            lines.append(
                f"  - `{ci['column']}`: ratio={ci['imbalance_ratio']}x "
                f"({ci['severity']}) — {ci['recommendation']}"
            )

    if report["inconsistencies"]:
        lines.append("")
        lines.append(f"**Format Issues:** {len(report['inconsistencies'])} column(s) with inconsistencies")
        for inc in report["inconsistencies"][:3]:
            issue_types = [i["type"] for i in inc["issues"]]
            lines.append(f"  - `{inc['column']}`: {', '.join(issue_types)}")

    return "\n".join(lines)
