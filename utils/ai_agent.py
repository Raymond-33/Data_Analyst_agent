"""
AI Agent — Enhanced LLM Layer (Ollama + OpenRouter)
=====================================================
Handles all LLM interactions with a dual-provider strategy:

  PRIMARY:  Ollama (local) — no rate limits, fast, free
  FALLBACK: OpenRouter     — cloud-based, free tier

  1. Executive Summary — structured JSON output with dataset overview,
     key metrics, insights, anomalies, and recommendations.
  2. Conversational Consultant — analytical reasoning, intent detection,
     Pandas code suggestions, chart recommendations.
"""

import os
import json
import re
import time
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from openai import OpenAI


# ─────────────────────────────────────────────
# Client Setup — Ollama (primary) + OpenRouter (fallback)
# ─────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2:3b"

# OpenRouter fallback models (tried in order)
OPENROUTER_MODELS = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "qwen/qwen3-4b:free",
]

MAX_RETRIES = 3
BASE_RETRY_DELAY = 3  # seconds


def _get_ollama_client():
    """Returns an Ollama-connected OpenAI client if Ollama is running."""
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        client.models.list()
        return client
    except Exception:
        return None


def _get_openrouter_client():
    """Returns an OpenRouter-connected OpenAI client, or None if no API key."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# ─────────────────────────────────────────────
# Unified LLM Call — with Ollama → OpenRouter fallback
# ─────────────────────────────────────────────

def _llm_call(messages, temperature=0.3, max_tokens=4096):
    """
    Unified LLM call:
      1. Try Ollama locally (no rate limits)
      2. Fall back to OpenRouter with model rotation on 429 errors

    Returns the response text string, or raises on total failure.
    """
    # ── Try Ollama first ──
    ollama = _get_ollama_client()
    if ollama:
        try:
            response = ollama.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.warning(f"⚠️ Ollama failed ({e}), falling back to OpenRouter...")

    # ── Fallback: OpenRouter with model rotation ──
    openrouter = _get_openrouter_client()
    if not openrouter:
        raise RuntimeError(
            "No LLM provider available. "
            "Start Ollama (`ollama serve`) or set OPENROUTER_API_KEY in .env."
        )

    last_error = None
    for model in OPENROUTER_MODELS:
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = openrouter.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                error_str = str(e)
                is_rate_limit = (
                    "429" in error_str
                    or "RESOURCE_EXHAUSTED" in error_str
                    or "rate" in error_str.lower()
                )
                if is_rate_limit:
                    if attempt < MAX_RETRIES:
                        delay = BASE_RETRY_DELAY * (2 ** attempt)
                        st.warning(
                            f"⏳ Rate limit on `{model.split('/')[-1]}`. "
                            f"Retrying in {delay}s... ({attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        st.info("🔄 Switching to next model...")
                        break
                else:
                    raise

    raise last_error


# ═══════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY (Structured JSON Output)
# ═══════════════════════════════════════════════

EXECUTIVE_SUMMARY_PROMPT = """
You are a Senior AI Data Analyst producing an executive summary report.

DATASET CONTEXT:
{context}

INSTRUCTIONS:
Analyze the dataset context above and produce a response in EXACTLY this JSON format.
Do NOT include any text before or after the JSON block. Return ONLY valid JSON.

{{
  "dataset_overview": "A 2-3 sentence description of what this dataset represents, its domain, and purpose.",
  "key_metrics": {{
    "total_records": <int>,
    "total_features": <int>,
    "data_quality_grade": "<A/B/C/D/F from the quality report>",
    "notable_stat_1": "<key statistic with label>",
    "notable_stat_2": "<another key statistic with label>"
  }},
  "insights": [
    {{
      "title": "<concise insight title>",
      "description": "<1-2 sentence explanation with specific numbers>",
      "impact": "<high/medium/low>",
      "category": "<trend/correlation/distribution/quality/anomaly>"
    }},
    {{
      "title": "<insight 2>",
      "description": "<description>",
      "impact": "<level>",
      "category": "<category>"
    }},
    {{
      "title": "<insight 3>",
      "description": "<description>",
      "impact": "<level>",
      "category": "<category>"
    }}
  ],
  "anomalies": [
    {{
      "title": "<anomaly title>",
      "description": "<what is unusual and why it matters>",
      "affected_columns": ["<col1>", "<col2>"],
      "severity": "<critical/warning/info>"
    }}
  ],
  "recommendations": [
    {{
      "action": "<specific actionable step>",
      "rationale": "<why this matters based on the data>",
      "priority": "<high/medium/low>"
    }},
    {{
      "action": "<action 2>",
      "rationale": "<rationale>",
      "priority": "<priority>"
    }},
    {{
      "action": "<action 3>",
      "rationale": "<rationale>",
      "priority": "<priority>"
    }}
  ]
}}

Provide 3-5 insights, 1-3 anomalies, and 3-5 recommendations.
Be data-driven, cite specific numbers from the context. Return ONLY valid JSON.
"""


def generate_executive_summary(data_context: str) -> Dict[str, Any]:
    """
    Generates a structured JSON executive summary from the LLM.

    Returns the parsed dict on success, or a fallback dict with
    error information on failure.
    """
    prompt = EXECUTIVE_SUMMARY_PROMPT.format(context=data_context)

    try:
        raw = _llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        return _parse_json_response(raw)
    except Exception as e:
        return _error_summary(f"LLM Error: {str(e)}")


def _parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Robustly parses JSON from LLM output, handling markdown code fences
    and other common wrapping patterns.
    """
    text = raw.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object from surrounding text
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return the raw text wrapped in a fallback structure
    return {
        "dataset_overview": text[:500],
        "key_metrics": {},
        "insights": [{"title": "Raw Analysis", "description": text[:1000], "impact": "medium", "category": "general"}],
        "anomalies": [],
        "recommendations": [],
        "_parse_warning": "LLM response was not valid JSON. Showing raw text.",
    }


def _error_summary(message: str) -> Dict[str, Any]:
    """Returns a fallback summary dict when the LLM call fails."""
    return {
        "dataset_overview": f"Error: {message}",
        "key_metrics": {},
        "insights": [],
        "anomalies": [],
        "recommendations": [],
        "_error": message,
    }


# ─────────────────────────────────────────────
# Executive Summary → Markdown Rendering
# ─────────────────────────────────────────────

def render_executive_summary_md(summary: Dict[str, Any]) -> str:
    """
    Converts the structured executive summary dict into beautiful
    markdown for Streamlit rendering.
    """
    lines = []

    # Error state
    if "_error" in summary:
        return f"⚠️ **{summary['_error']}**"

    # Parse warning
    if "_parse_warning" in summary:
        lines.append(f"> ⚠️ {summary['_parse_warning']}\n")

    # Overview
    lines.append("### 📋 Dataset Overview")
    lines.append(summary.get("dataset_overview", "No overview available."))
    lines.append("")

    # Key Metrics
    metrics = summary.get("key_metrics", {})
    if metrics:
        lines.append("### 📊 Key Metrics")
        for key, val in metrics.items():
            label = key.replace("_", " ").title()
            lines.append(f"- **{label}:** {val}")
        lines.append("")

    # Insights
    insights = summary.get("insights", [])
    if insights:
        lines.append("### 💡 Key Insights")
        for i, ins in enumerate(insights, 1):
            impact_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                ins.get("impact", ""), "⚪"
            )
            lines.append(f"**{i}. {ins.get('title', 'Insight')}** {impact_icon}")
            lines.append(f"   {ins.get('description', '')}")
            lines.append("")

    # Anomalies
    anomalies = summary.get("anomalies", [])
    if anomalies:
        lines.append("### ⚠️ Anomalies Detected")
        for a in anomalies:
            sev_icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(
                a.get("severity", ""), "⚪"
            )
            cols = ", ".join(a.get("affected_columns", []))
            lines.append(f"- {sev_icon} **{a.get('title', 'Anomaly')}**")
            lines.append(f"  {a.get('description', '')}")
            if cols:
                lines.append(f"  *Columns:* `{cols}`")
            lines.append("")

    # Recommendations
    recs = summary.get("recommendations", [])
    if recs:
        lines.append("### 🎯 Recommendations")
        for i, r in enumerate(recs, 1):
            pri_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                r.get("priority", ""), "⚪"
            )
            lines.append(f"**{i}. {r.get('action', '')}** {pri_icon}")
            lines.append(f"   *{r.get('rationale', '')}*")
            lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════
# 2. BACKWARD-COMPATIBLE WRAPPER
# ═══════════════════════════════════════════════

def generate_insights_and_recommendations(data_context: str) -> str:
    """
    Backward-compatible wrapper. Returns markdown string.
    """
    summary = generate_executive_summary(data_context)
    return render_executive_summary_md(summary)


# ═══════════════════════════════════════════════
# 3. CONVERSATIONAL DATA CONSULTANT (Enhanced)
# ═══════════════════════════════════════════════

CONSULTANT_SYSTEM_PROMPT = """You are an expert AI Data Consultant with deep expertise in statistics, data engineering, and business analytics.

DATASET CONTEXT:
{context}

YOUR CAPABILITIES:
1. **Analytical Reasoning** — You can interpret data patterns, explain correlations vs causation, and identify statistical significance.
2. **Pandas Operations** — When the user's question requires data manipulation, suggest the specific Pandas code they would run. Format code in ```python blocks.
3. **Chart Recommendations** — When visualization would help, recommend the specific chart type and columns to use.
4. **Prediction & Forecasting** — When asked about future trends, explain what modeling approach would work and why.
5. **Data Quality Advice** — Flag data quality issues that could affect analysis accuracy.

RESPONSE GUIDELINES:
- Always ground your answers in the actual data provided in the context above.
- Use specific numbers, column names, and statistics from the dataset.
- If a question cannot be answered from the available data, explain what additional data would be needed.
- If the user's question is statistically invalid, politely explain why and suggest a better approach.
- When suggesting Pandas code, make it complete and runnable (assume `df` is the DataFrame).
- Keep responses focused and professional but conversational.
- If asked "why", go beyond surface-level — check correlations, suggest regression, explain confounding variables.
- For ambiguous queries, interpret charitably and clarify any assumptions you make.

ANTI-PATTERNS (things you must NOT do):
- Do NOT fabricate data or statistics that aren't in the context.
- Do NOT ignore data quality issues when they would affect the answer.
- Do NOT give overly simple answers to complex analytical questions.
- Do NOT confuse correlation with causation without disclaimers.
"""


def query_data_consultant(
    data_context: str,
    chat_history: list,
    user_query: str,
) -> str:
    """
    Enhanced conversational consultant with analytical reasoning.
    """
    system_instruction = CONSULTANT_SYSTEM_PROMPT.format(context=data_context)

    messages = [{"role": "system", "content": system_instruction}]

    for msg in chat_history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    messages.append({"role": "user", "content": user_query})

    try:
        return _llm_call(
            messages=messages,
            temperature=0.4,
            max_tokens=4096,
        )
    except Exception as e:
        return f"⚠️ Error consulting AI: {str(e)}"


# ═══════════════════════════════════════════════
# 4. QUICK ANALYSIS QUERIES (Utility)
# ═══════════════════════════════════════════════

def generate_column_analysis(data_context: str, column_name: str) -> str:
    """
    Generates a focused analysis on a specific column.
    """
    prompt = f"""
    Analyze the column '{column_name}' from this dataset:

    {data_context}

    Provide:
    1. **What it represents** — domain meaning
    2. **Distribution characteristics** — is it normal, skewed, etc.?
    3. **Notable patterns** — trends, clusters, gaps
    4. **Quality concerns** — missing values, outliers, encoding issues
    5. **Recommended next steps** — feature engineering, transformations, further analysis

    Be concise and data-driven. Use specific numbers.
    """

    try:
        return _llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as e:
        return f"Error: {str(e)}"


def generate_hypothesis(data_context: str, observation: str) -> str:
    """
    Given a user observation about the data, generates hypotheses
    and suggests how to test them.
    """
    prompt = f"""
    You are a data scientist. A user has made this observation about their dataset:

    "{observation}"

    DATASET CONTEXT:
    {data_context}

    Based on the data:
    1. Generate 2-3 plausible hypotheses for why this pattern exists.
    2. For each hypothesis, suggest a specific statistical test or analysis to validate it.
    3. Provide the Pandas/Python code to run each test (assume `df` is the DataFrame).
    4. Warn about any confounding variables or data quality issues that could affect results.

    Be rigorous and scientific in your reasoning.
    """

    try:
        return _llm_call(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except Exception as e:
        return f"Error: {str(e)}"