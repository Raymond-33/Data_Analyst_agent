"""
Statistical Profiling Engine
==============================
Deep statistical analysis: enhanced descriptive stats (skewness, kurtosis),
distribution normality tests, top correlations, feature importance
heuristics, and optional K-Means clustering. Produces a unified
profiling report for the dashboard, LLM, and visualization modules.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple


# ─────────────────────────────────────────────
# 1. Enhanced Descriptive Statistics
# ─────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extended descriptive statistics for numeric columns, including
    skewness and kurtosis beyond what df.describe() provides.

    Returns:
        {
            "summary_table": pd.DataFrame,   # Full stats table for display
            "skewed_columns": [...],          # Columns with |skew| > 1
            "high_kurtosis_columns": [...],   # Columns with kurtosis > 3
        }
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return {
            "summary_table": pd.DataFrame(),
            "skewed_columns": [],
            "high_kurtosis_columns": [],
        }

    rows = []
    skewed = []
    high_kurt = []

    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 3:
            continue

        skew_val = float(s.skew())
        kurt_val = float(s.kurtosis())  # Excess kurtosis (0 = normal)

        row = {
            "Column": col,
            "Count": int(s.count()),
            "Mean": round(float(s.mean()), 4),
            "Std": round(float(s.std()), 4),
            "Min": round(float(s.min()), 4),
            "25%": round(float(s.quantile(0.25)), 4),
            "Median": round(float(s.median()), 4),
            "75%": round(float(s.quantile(0.75)), 4),
            "Max": round(float(s.max()), 4),
            "Skewness": round(skew_val, 4),
            "Kurtosis": round(kurt_val, 4),
            "CV%": round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else 0,
        }
        rows.append(row)

        if abs(skew_val) > 1:
            direction = "right" if skew_val > 0 else "left"
            skewed.append({
                "column": col,
                "skewness": round(skew_val, 4),
                "direction": direction,
                "suggestion": f"Consider log or sqrt transform for '{col}'",
            })

        if abs(kurt_val) > 3:
            kurt_type = "leptokurtic (heavy tails)" if kurt_val > 0 else "platykurtic (light tails)"
            high_kurt.append({
                "column": col,
                "kurtosis": round(kurt_val, 4),
                "type": kurt_type,
            })

    return {
        "summary_table": pd.DataFrame(rows),
        "skewed_columns": skewed,
        "high_kurtosis_columns": high_kurt,
    }


# ─────────────────────────────────────────────
# 2. Distribution Analysis
# ─────────────────────────────────────────────

def distribution_analysis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Tests each numeric column for normality and classifies the distribution.

    Uses the D'Agostino-Pearson test (scipy) if available, falling back
    to a heuristic based on skewness/kurtosis.

    Returns a list of per-column distribution assessments.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = []

    has_scipy = False
    try:
        from scipy import stats as sp_stats
        has_scipy = True
    except ImportError:
        pass

    for col in num_cols:
        s = df[col].dropna()
        if len(s) < 20:
            continue

        skew = float(s.skew())
        kurt = float(s.kurtosis())

        info: Dict[str, Any] = {
            "column": col,
            "n_samples": len(s),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "is_normal": False,
            "normality_test": "n/a",
            "p_value": None,
            "distribution_type": "unknown",
        }

        # Normality test
        if has_scipy and len(s) >= 20:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_value = sp_stats.normaltest(s)
                info["normality_test"] = "D'Agostino-Pearson"
                info["p_value"] = round(float(p_value), 6)
                info["is_normal"] = p_value > 0.05
            except Exception:
                pass

        # Classify distribution shape
        info["distribution_type"] = _classify_distribution(skew, kurt, info["is_normal"])

        results.append(info)

    return results


def _classify_distribution(skew: float, kurt: float, is_normal: bool) -> str:
    """Classifies distribution based on shape statistics."""
    if is_normal and abs(skew) < 0.5:
        return "normal"
    elif abs(skew) < 0.5:
        return "approximately symmetric"
    elif skew > 2:
        return "highly right-skewed"
    elif skew > 0.5:
        return "right-skewed"
    elif skew < -2:
        return "highly left-skewed"
    elif skew < -0.5:
        return "left-skewed"
    elif kurt > 5:
        return "heavy-tailed"
    else:
        return "approximately symmetric"


# ─────────────────────────────────────────────
# 3. Correlation Analysis
# ─────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
    """
    Computes the correlation matrix for numeric columns and extracts
    the top positively and negatively correlated pairs.

    Returns:
        {
            "correlation_matrix": pd.DataFrame,
            "top_positive": [{"col1":, "col2":, "correlation":}, ...],
            "top_negative": [{"col1":, "col2":, "correlation":}, ...],
            "strong_count": int,  # pairs with |r| > 0.7
            "summary": str,
        }
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        return {
            "correlation_matrix": pd.DataFrame(),
            "top_positive": [],
            "top_negative": [],
            "strong_count": 0,
            "summary": "Insufficient numeric columns for correlation analysis.",
        }

    corr_matrix = df[num_cols].corr()

    # Extract unique pairs (upper triangle)
    pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr_matrix.iloc[i, j]
            if not np.isnan(r):
                pairs.append({
                    "col1": num_cols[i],
                    "col2": num_cols[j],
                    "correlation": round(float(r), 4),
                    "strength": _corr_strength(abs(r)),
                })

    # Sort for top positive / negative
    pairs_sorted = sorted(pairs, key=lambda x: x["correlation"], reverse=True)
    top_pos = [p for p in pairs_sorted if p["correlation"] > 0][:top_n]
    top_neg = [p for p in reversed(pairs_sorted) if p["correlation"] < 0][:top_n]
    strong = sum(1 for p in pairs if abs(p["correlation"]) > 0.7)

    # Summary
    if strong == 0:
        summary = "No strong correlations (|r| > 0.7) found between numeric features."
    elif strong <= 3:
        top = pairs_sorted[0] if pairs_sorted else None
        summary = (
            f"{strong} strong correlation(s) found. "
            f"Strongest: {top['col1']} & {top['col2']} (r={top['correlation']})."
            if top else f"{strong} strong correlation(s) found."
        )
    else:
        summary = (
            f"{strong} strong correlations found — possible multicollinearity. "
            f"Consider feature selection or PCA."
        )

    return {
        "correlation_matrix": corr_matrix,
        "top_positive": top_pos,
        "top_negative": top_neg,
        "strong_count": strong,
        "summary": summary,
    }


def _corr_strength(abs_r: float) -> str:
    """Classifies correlation strength."""
    if abs_r >= 0.9:
        return "very strong"
    elif abs_r >= 0.7:
        return "strong"
    elif abs_r >= 0.5:
        return "moderate"
    elif abs_r >= 0.3:
        return "weak"
    else:
        return "negligible"


# ─────────────────────────────────────────────
# 4. Feature Importance Heuristics
# ─────────────────────────────────────────────

def feature_importance_heuristic(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimates feature importance without training a full model.

    Strategy:
      - If target is provided and scikit-learn is available:
        uses mutual information (classification or regression).
      - Fallback: variance-based ranking for numeric columns +
        cardinality ranking for categoricals.

    Returns:
        {
            "method": str,
            "rankings": [{"feature": str, "score": float}, ...],
            "target_used": str | None,
        }
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Try mutual information if target is available
    if target_col and target_col in df.columns:
        mi_result = _mutual_info_importance(df, target_col)
        if mi_result is not None:
            return mi_result

    # Fallback: variance-based ranking
    return _variance_importance(df, num_cols)


def _mutual_info_importance(
    df: pd.DataFrame, target_col: str,
) -> Optional[Dict[str, Any]]:
    """Mutual information scoring against a target column."""
    try:
        from sklearn.feature_selection import (
            mutual_info_classif,
            mutual_info_regression,
        )
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return None

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        return None

    # Prepare features: encode categoricals, drop NaN
    df_work = df[feature_cols + [target_col]].dropna()
    if len(df_work) < 20:
        return None

    X = df_work[feature_cols].copy()
    y = df_work[target_col].copy()

    # Encode categorical features
    le_map = {}
    for col in X.columns:
        if X[col].dtype == object or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_map[col] = le

    # Encode target if categorical
    is_classification = y.dtype == object or y.nunique() <= 20
    if y.dtype == object or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Compute MI
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
    except Exception:
        return None

    rankings = []
    for col, score in zip(feature_cols, mi_scores):
        rankings.append({"feature": col, "score": round(float(score), 4)})

    rankings.sort(key=lambda x: x["score"], reverse=True)

    return {
        "method": f"mutual_info_{'classif' if is_classification else 'regression'}",
        "rankings": rankings,
        "target_used": target_col,
    }


def _variance_importance(
    df: pd.DataFrame, num_cols: List[str],
) -> Dict[str, Any]:
    """Variance-based feature ranking (no target needed)."""
    rankings = []

    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        # Normalize variance by dividing by mean^2 (coefficient of variation squared)
        mean = s.mean()
        var = s.var()
        if mean != 0:
            cv_sq = var / (mean ** 2)
        else:
            cv_sq = var
        rankings.append({"feature": col, "score": round(float(cv_sq), 6)})

    # Also rank categoricals by entropy
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        vc = df[col].value_counts(normalize=True)
        entropy = float(-(vc * np.log2(vc.clip(lower=1e-10))).sum())
        rankings.append({"feature": col, "score": round(entropy, 4)})

    rankings.sort(key=lambda x: x["score"], reverse=True)

    return {
        "method": "variance_entropy",
        "rankings": rankings,
        "target_used": None,
    }


# ─────────────────────────────────────────────
# 5. Basic Clustering
# ─────────────────────────────────────────────

def basic_clustering(
    df: pd.DataFrame,
    max_k: int = 8,
    max_samples: int = 5000,
) -> Optional[Dict[str, Any]]:
    """
    K-Means clustering with automatic k selection via silhouette score.

    Only uses numeric columns, scales data, tests k=2..max_k.
    Returns None if scikit-learn isn't available or data is unsuitable.

    Returns:
        {
            "optimal_k": int,
            "silhouette_score": float,
            "cluster_sizes": dict,
            "cluster_centers": pd.DataFrame,
            "labels": np.ndarray,
            "k_scores": [{"k": int, "silhouette": float}, ...],
        }
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return None

    # Prepare data: drop NaN, sample if large
    df_num = df[num_cols].dropna()
    if len(df_num) < 30:
        return None

    if len(df_num) > max_samples:
        df_num = df_num.sample(n=max_samples, random_state=42)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df_num)

    # Test k values
    k_scores = []
    best_k = 2
    best_score = -1

    actual_max_k = min(max_k, len(df_num) - 1)

    for k in range(2, actual_max_k + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=100)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels, sample_size=min(1000, len(X)))

            k_scores.append({"k": k, "silhouette": round(float(score), 4)})

            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    if not k_scores:
        return None

    # Final fit with optimal k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        final_labels = km_final.fit_predict(X)

    # Cluster centers in original scale
    centers_original = scaler.inverse_transform(km_final.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=num_cols)
    centers_df.index = [f"Cluster {i}" for i in range(best_k)]

    # Cluster sizes
    unique, counts = np.unique(final_labels, return_counts=True)
    cluster_sizes = {f"Cluster {int(u)}": int(c) for u, c in zip(unique, counts)}

    return {
        "optimal_k": best_k,
        "silhouette_score": round(best_score, 4),
        "cluster_sizes": cluster_sizes,
        "cluster_centers": centers_df,
        "labels": final_labels,
        "k_scores": k_scores,
    }


# ─────────────────────────────────────────────
# 6. Full Profiling Report
# ─────────────────────────────────────────────

def get_profiling_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    run_clustering: bool = True,
) -> Dict[str, Any]:
    """
    Master function: runs all statistical profiling and returns a
    unified report dictionary.

    Args:
        df: The (cleaned) DataFrame.
        target_col: Optional target column name (from schema_intel).
        run_clustering: Whether to run K-Means (can be slow on large data).

    Returns dict with keys:
        descriptive, distributions, correlations, importance, clustering
    """
    desc = descriptive_stats(df)
    dists = distribution_analysis(df)
    corrs = correlation_analysis(df)
    importance = feature_importance_heuristic(df, target_col)

    clustering_result = None
    if run_clustering:
        clustering_result = basic_clustering(df)

    return {
        "descriptive": desc,
        "distributions": dists,
        "correlations": corrs,
        "importance": importance,
        "clustering": clustering_result,
    }


# ─────────────────────────────────────────────
# 7. Display Formatters
# ─────────────────────────────────────────────

def format_profiling_summary(report: Dict[str, Any]) -> str:
    """
    Generates a markdown summary of the statistical profiling report,
    suitable for LLM context or display.
    """
    lines = []

    # Descriptive
    desc = report["descriptive"]
    if desc["skewed_columns"]:
        lines.append("**Skewed Columns:**")
        for sc in desc["skewed_columns"]:
            lines.append(f"  - `{sc['column']}`: skewness={sc['skewness']} ({sc['direction']}-skewed)")
            lines.append(f"    {sc['suggestion']}")
    else:
        lines.append("**Skewness:** All numeric columns are approximately symmetric.")

    if desc["high_kurtosis_columns"]:
        lines.append("")
        lines.append("**Heavy-Tailed Columns:**")
        for hk in desc["high_kurtosis_columns"]:
            lines.append(f"  - `{hk['column']}`: kurtosis={hk['kurtosis']} ({hk['type']})")

    # Distributions
    dists = report["distributions"]
    normal_cols = [d["column"] for d in dists if d["is_normal"]]
    non_normal = [d for d in dists if not d["is_normal"]]
    lines.append("")
    if normal_cols:
        lines.append(f"**Normally Distributed:** {', '.join(normal_cols)}")
    if non_normal:
        lines.append(f"**Non-Normal Distributions:**")
        for d in non_normal[:5]:
            lines.append(f"  - `{d['column']}`: {d['distribution_type']}")

    # Correlations
    corrs = report["correlations"]
    lines.append("")
    lines.append(f"**Correlations:** {corrs['summary']}")
    if corrs["top_positive"]:
        top = corrs["top_positive"][0]
        lines.append(f"  Strongest positive: {top['col1']} & {top['col2']} (r={top['correlation']})")
    if corrs["top_negative"]:
        top = corrs["top_negative"][0]
        lines.append(f"  Strongest negative: {top['col1']} & {top['col2']} (r={top['correlation']})")

    # Feature importance
    imp = report["importance"]
    lines.append("")
    lines.append(f"**Feature Importance** (method: {imp['method']}):")
    if imp["target_used"]:
        lines.append(f"  Target: `{imp['target_used']}`")
    for r in imp["rankings"][:5]:
        lines.append(f"  - `{r['feature']}`: score={r['score']}")

    # Clustering
    clust = report["clustering"]
    if clust:
        lines.append("")
        lines.append(f"**Clustering:** Optimal k={clust['optimal_k']} "
                     f"(silhouette={clust['silhouette_score']})")
        lines.append(f"  Cluster sizes: {clust['cluster_sizes']}")
    else:
        lines.append("")
        lines.append("**Clustering:** Not performed or insufficient data.")

    return "\n".join(lines)


def correlation_pairs_to_dataframe(corrs: Dict[str, Any]) -> pd.DataFrame:
    """Converts top correlation pairs into a display-ready DataFrame."""
    all_pairs = corrs["top_positive"] + corrs["top_negative"]
    if not all_pairs:
        return pd.DataFrame({"Status": ["No correlation pairs to display"]})

    all_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    rows = []
    for p in all_pairs[:15]:
        rows.append({
            "Feature 1": p["col1"],
            "Feature 2": p["col2"],
            "Correlation": p["correlation"],
            "Strength": p["strength"],
        })
    return pd.DataFrame(rows)
