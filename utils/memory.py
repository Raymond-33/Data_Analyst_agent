"""
Context Memory Layer
======================
Stores and manages the analytical context for the current dataset session:

  - Dataset metadata (schema, quality, profiling reports)
  - Executive summary
  - Conversation history with derived results
  - Provides enriched context for the LLM on follow-up queries

Uses Streamlit session_state as the backing store, with a clean API
that isolates the rest of the codebase from session_state details.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


# ─────────────────────────────────────────────
# Session State Key Constants
# ─────────────────────────────────────────────

_KEY_PREFIX = "mem_"
_DATA_RAW = f"{_KEY_PREFIX}data_raw"
_DATA_CLEAN = f"{_KEY_PREFIX}data_clean"
_FILE_SIZE = f"{_KEY_PREFIX}file_size_mb"
_FILE_NAME = f"{_KEY_PREFIX}file_name"
_SCHEMA_REPORT = f"{_KEY_PREFIX}schema_report"
_QUALITY_REPORT = f"{_KEY_PREFIX}quality_report"
_PROFILING_REPORT = f"{_KEY_PREFIX}profiling_report"
_EXECUTIVE_SUMMARY = f"{_KEY_PREFIX}executive_summary"
_LLM_CONTEXT = f"{_KEY_PREFIX}llm_context"
_MESSAGES = f"{_KEY_PREFIX}messages"
_QUERY_RESULTS = f"{_KEY_PREFIX}query_results"
_ANALYSIS_READY = f"{_KEY_PREFIX}analysis_ready"
_COL_NAME_MAP = f"{_KEY_PREFIX}col_name_map"
_ANALYSIS_TIMESTAMP = f"{_KEY_PREFIX}analysis_timestamp"


# ─────────────────────────────────────────────
# Initialization
# ─────────────────────────────────────────────

def init_memory():
    """
    Ensures all memory keys exist in session_state.
    Call once at app startup.
    """
    defaults = {
        _DATA_RAW: None,
        _DATA_CLEAN: None,
        _FILE_SIZE: 0,
        _FILE_NAME: None,
        _SCHEMA_REPORT: None,
        _QUALITY_REPORT: None,
        _PROFILING_REPORT: None,
        _EXECUTIVE_SUMMARY: None,
        _LLM_CONTEXT: None,
        _MESSAGES: [],
        _QUERY_RESULTS: [],
        _ANALYSIS_READY: False,
        _COL_NAME_MAP: None,
        _ANALYSIS_TIMESTAMP: None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def clear_memory():
    """
    Resets all memory — called on new dataset upload.
    """
    st.session_state[_DATA_RAW] = None
    st.session_state[_DATA_CLEAN] = None
    st.session_state[_FILE_SIZE] = 0
    st.session_state[_FILE_NAME] = None
    st.session_state[_SCHEMA_REPORT] = None
    st.session_state[_QUALITY_REPORT] = None
    st.session_state[_PROFILING_REPORT] = None
    st.session_state[_EXECUTIVE_SUMMARY] = None
    st.session_state[_LLM_CONTEXT] = None
    st.session_state[_MESSAGES] = []
    st.session_state[_QUERY_RESULTS] = []
    st.session_state[_ANALYSIS_READY] = False
    st.session_state[_COL_NAME_MAP] = None
    st.session_state[_ANALYSIS_TIMESTAMP] = None


# ─────────────────────────────────────────────
# Data Storage
# ─────────────────────────────────────────────

def store_dataset(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    file_size_mb: float,
    file_name: str,
    col_name_map: Optional[Dict[str, str]] = None,
):
    """Stores the loaded dataset in memory."""
    st.session_state[_DATA_RAW] = df_raw
    st.session_state[_DATA_CLEAN] = df_clean
    st.session_state[_FILE_SIZE] = file_size_mb
    st.session_state[_FILE_NAME] = file_name
    st.session_state[_COL_NAME_MAP] = col_name_map


def store_reports(
    schema_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    profiling_report: Dict[str, Any],
):
    """Stores the analysis reports in memory."""
    st.session_state[_SCHEMA_REPORT] = schema_report
    st.session_state[_QUALITY_REPORT] = quality_report
    st.session_state[_PROFILING_REPORT] = profiling_report


def store_executive_summary(summary: Dict[str, Any]):
    """Stores the LLM executive summary."""
    st.session_state[_EXECUTIVE_SUMMARY] = summary


def store_llm_context(context: str):
    """Stores the pre-built LLM context string."""
    st.session_state[_LLM_CONTEXT] = context


def mark_analysis_complete():
    """Marks that the full analysis pipeline has finished."""
    st.session_state[_ANALYSIS_READY] = True
    st.session_state[_ANALYSIS_TIMESTAMP] = datetime.now().isoformat()


# ─────────────────────────────────────────────
# Data Retrieval
# ─────────────────────────────────────────────

def get_raw_data() -> Optional[pd.DataFrame]:
    return st.session_state.get(_DATA_RAW)

def get_clean_data() -> Optional[pd.DataFrame]:
    return st.session_state.get(_DATA_CLEAN)

def get_file_size() -> float:
    return st.session_state.get(_FILE_SIZE, 0)

def get_file_name() -> Optional[str]:
    return st.session_state.get(_FILE_NAME)

def get_schema_report() -> Optional[Dict]:
    return st.session_state.get(_SCHEMA_REPORT)

def get_quality_report() -> Optional[Dict]:
    return st.session_state.get(_QUALITY_REPORT)

def get_profiling_report() -> Optional[Dict]:
    return st.session_state.get(_PROFILING_REPORT)

def get_executive_summary() -> Optional[Dict]:
    return st.session_state.get(_EXECUTIVE_SUMMARY)

def get_llm_context() -> Optional[str]:
    return st.session_state.get(_LLM_CONTEXT)

def get_col_name_map() -> Optional[Dict]:
    return st.session_state.get(_COL_NAME_MAP)

def is_analysis_ready() -> bool:
    return st.session_state.get(_ANALYSIS_READY, False)

def has_dataset() -> bool:
    return st.session_state.get(_DATA_CLEAN) is not None

def get_analysis_timestamp() -> Optional[str]:
    return st.session_state.get(_ANALYSIS_TIMESTAMP)


# ─────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────

def get_messages() -> List[Dict[str, str]]:
    """Returns the full chat history."""
    return st.session_state.get(_MESSAGES, [])


def add_message(role: str, content: str):
    """Appends a message to the chat history."""
    if _MESSAGES not in st.session_state:
        st.session_state[_MESSAGES] = []
    st.session_state[_MESSAGES].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })


def clear_messages():
    """Clears the chat history."""
    st.session_state[_MESSAGES] = []


def store_query_result(query: str, result: str, result_type: str = "text"):
    """
    Stores a derived query result for follow-up context.
    
    Args:
        query: The user's question.
        result: The computed answer/result.
        result_type: "text", "dataframe", "chart", "code"
    """
    if _QUERY_RESULTS not in st.session_state:
        st.session_state[_QUERY_RESULTS] = []
    st.session_state[_QUERY_RESULTS].append({
        "query": query,
        "result": result[:2000],  # Truncate to avoid oversized state
        "type": result_type,
        "timestamp": datetime.now().isoformat(),
    })


def get_query_results() -> List[Dict]:
    """Returns all stored query results."""
    return st.session_state.get(_QUERY_RESULTS, [])


# ─────────────────────────────────────────────
# Enriched Context for LLM Queries
# ─────────────────────────────────────────────

def get_conversation_context(max_recent: int = 5) -> str:
    """
    Builds an enriched context string for the LLM that includes:
      - The base LLM context (dataset + schema + quality + stats)
      - Recent query results (for follow-up intelligence)
      - Analysis timestamp

    This is used by the consultant to understand previous interactions.
    """
    sections = []

    # Base context
    base = get_llm_context()
    if base:
        sections.append(base)

    # Previous query results for continuity
    results = get_query_results()
    if results:
        recent = results[-max_recent:]
        sections.append("\n" + "=" * 50)
        sections.append("PREVIOUS ANALYSIS RESULTS")
        sections.append("=" * 50)
        for r in recent:
            sections.append(f"\nQuestion: {r['query']}")
            sections.append(f"Result: {r['result'][:500]}")

    # Metadata
    ts = get_analysis_timestamp()
    if ts:
        sections.append(f"\nAnalysis performed at: {ts}")

    return "\n".join(sections)


def get_dataset_info_summary() -> Dict[str, Any]:
    """
    Returns a lightweight summary of the current dataset state,
    useful for sidebar display or quick checks.
    """
    df = get_clean_data()
    schema = get_schema_report()
    quality = get_quality_report()

    if df is None:
        return {"loaded": False}

    info = {
        "loaded": True,
        "file_name": get_file_name(),
        "rows": len(df),
        "columns": len(df.columns),
        "file_size_mb": get_file_size(),
        "analysis_ready": is_analysis_ready(),
    }

    if quality and "overall_score" in quality:
        info["quality_grade"] = quality["overall_score"]["grade"]
        info["quality_score"] = quality["overall_score"]["total"]

    if schema:
        info["has_time_series"] = schema.get("time_series", {}).get("is_time_series", False)
        targets = schema.get("target_candidates", [])
        info["target_column"] = targets[0]["column"] if targets else None

    return info
