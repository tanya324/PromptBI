"""
llm.py — Gemini integration with structured prompt engineering.
Uses the new google-genai SDK (google-generativeai is deprecated).

Pipeline per query:
  1. Build a rich system prompt with full schema context
  2. Call Gemini → get JSON with { sql, charts[], insight }
  3. Run each SQL query via db.run_query()
  4. If SQL fails → self-correction pass (send error back to Gemini, retry once)
  5. Return final dashboard config to main.py
"""

import os
import json
import re
import logging
import time
from datetime import datetime

from google import genai
from google.genai import types

try:
    from backend.schema import SCHEMA_DESCRIPTION, CHART_GUIDANCE
    from backend.db import run_query
except Exception:  # pragma: no cover
    from schema import SCHEMA_DESCRIPTION, CHART_GUIDANCE
    from db import run_query

logger = logging.getLogger(__name__)

# Default to Gemini 2.0 Flash (override via env).
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = float(os.environ.get("GEMINI_TEMPERATURE", "0.1"))

# Request timeout in milliseconds (google-genai uses ms).
GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "120000"))  # 120s

# Keep token usage lower by default (override via env).
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

# Retry on transient Gemini errors (429/5xx/timeouts).
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
GEMINI_RETRY_BASE_DELAY_S = float(os.environ.get("GEMINI_RETRY_BASE_DELAY_S", "1.5"))

# Self-correction costs an extra Gemini call. Keep it off by default to reduce quota usage.
GEMINI_ENABLE_SELF_CORRECTION = os.environ.get("GEMINI_ENABLE_SELF_CORRECTION", "0") == "1"

# Short-lived cache to avoid repeated identical calls burning quota.
# Keyed by (query + conversation history).
GEMINI_CACHE_TTL_S = int(os.environ.get("GEMINI_CACHE_TTL_S", "120"))  # 2 minutes
_cache: dict[str, tuple[float, dict]] = {}

_client: genai.Client | None = None
_client_key: str | None = None
_client_timeout_ms: int | None = None


_SUPPORTED_CHART_TYPES = {"line", "bar", "pie", "area"}


def _normalize_chart_type(chart_type: str | None) -> str:
    """
    Frontend ChartRenderer supports only: line, bar, pie, area.
    Gemini may return scatter/other types; map them to a supported type.
    """
    t = (chart_type or "").strip().lower()
    if t in _SUPPORTED_CHART_TYPES:
        return t
    if t == "scatter":
        return "bar"
    return "bar"


def _build_primary_series(chart: dict) -> tuple[list, list, str]:
    """
    Convert chart {data, x_key, y_keys} into (labels, values, dataset_name)
    compatible with the current frontend.
    """
    rows = chart.get("data") or []
    if not isinstance(rows, list) or not rows:
        return [], [], ""

    x_key = (chart.get("x_key") or "").strip()
    y_keys = chart.get("y_keys") or []
    y_key = y_keys[0] if isinstance(y_keys, list) and y_keys else ""

    # Fallback: infer keys from first row if missing/invalid
    first = rows[0] if isinstance(rows[0], dict) else {}
    if isinstance(first, dict):
        if not x_key or x_key not in first:
            x_key = next(iter(first.keys()), "")
        if not y_key or y_key not in first:
            # pick first numeric-like column (skip x_key)
            for k, v in first.items():
                if k == x_key:
                    continue
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()):
                    y_key = k
                    break

    if not x_key or not y_key:
        return [], [], ""

    labels: list = []
    values: list = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        labels.append(row.get(x_key))
        v = row.get(y_key)
        if v is None:
            values.append(0)
        elif isinstance(v, (int, float)):
            values.append(v)
        else:
            try:
                values.append(float(v))
            except Exception:
                values.append(0)

    dataset_name = y_key or "Value"
    return labels, values, dataset_name


def _fallback_dashboard(user_query: str) -> dict:
    """
    Deterministic fallback when Gemini is unavailable (quota/network/etc).
    Produces a single chart from SQLite so the frontend can still render.
    """
    q = (user_query or "").lower()

    # Heuristic selection of a safe, fast aggregate query
    if any(k in q for k in ["month", "monthly", "trend", "over time", "time series", "timeline"]):
        title = "Monthly Views Trend"
        chart_type = "line"
        sql = """
        SELECT strftime('%Y-%m', timestamp) AS month,
               SUM(views) AS total_views
        FROM videos
        GROUP BY month
        ORDER BY month ASC
        LIMIT 60
        """
        x_key = "month"
        y_keys = ["total_views"]
    elif "region" in q:
        title = "Views by Region"
        chart_type = "bar"
        sql = """
        SELECT region AS region,
               SUM(views) AS total_views
        FROM videos
        GROUP BY region
        ORDER BY total_views DESC
        LIMIT 10
        """
        x_key = "region"
        y_keys = ["total_views"]
    elif "language" in q:
        title = "Views by Language"
        chart_type = "bar"
        sql = """
        SELECT language AS language,
               SUM(views) AS total_views
        FROM videos
        GROUP BY language
        ORDER BY total_views DESC
        LIMIT 10
        """
        x_key = "language"
        y_keys = ["total_views"]
    else:
        title = "Views by Category"
        chart_type = "bar"
        sql = """
        SELECT category AS category,
               SUM(views) AS total_views
        FROM videos
        GROUP BY category
        ORDER BY total_views DESC
        LIMIT 10
        """
        x_key = "category"
        y_keys = ["total_views"]

    db_result = run_query(sql)
    if not db_result["success"]:
        # Last-ditch fallback: return an empty-but-successful payload so UI doesn't crash.
        return {
            "success": True,
            "dashboard_title": "Dashboard (Fallback)",
            "insight": (
                "AI service is currently unavailable, and the fallback query also failed. "
                "Please check that your dataset is loaded (data.csv/upload) and try again."
            ),
            "charts": [],
            "failed_charts": [{"chart_id": "chart_1", "title": title, "chart_type": chart_type, "sql": sql, "error": db_result["error"]}],
            "query": user_query,
            "title": title,
            "chart_type": chart_type,
            "labels": [],
            "datasets": [],
        }

    chart = {
        "chart_id": "chart_1",
        "title": title,
        "chart_type": chart_type,
        "x_key": x_key,
        "y_keys": y_keys,
        "color_by": None,
        "description": "Auto-generated fallback chart (Gemini unavailable).",
        "data": db_result["rows"],
        "columns": db_result["columns"],
        "sql": db_result["sql"],
        "row_count": db_result["row_count"],
    }
    labels, values, dataset_name = _build_primary_series(chart)

    return {
        "success": True,
        "dashboard_title": "Dashboard (Fallback)",
        "insight": (
            "Gemini is currently unavailable (quota/billing/network). "
            "Showing a fallback chart generated directly from your SQLite data."
        ),
        "charts": [chart],
        "failed_charts": [],
        "query": user_query,
        "title": title,
        "chart_type": chart_type,
        "labels": labels,
        "datasets": ([{"name": dataset_name, "data": values}] if labels else []),
    }


def _get_client() -> genai.Client:
    global _client, _client_key, _client_timeout_ms
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    # If the key/timeout changed (e.g. you created a new key), rebuild the client.
    if _client is not None and _client_key == api_key and _client_timeout_ms == GEMINI_TIMEOUT_MS:
        return _client

    # Explicit timeout prevents hanging requests.
    _client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )
    _client_key = api_key
    _client_timeout_ms = GEMINI_TIMEOUT_MS
    return _client

SYSTEM_PROMPT = f"""
You are an expert data analyst and SQL engineer powering a Business Intelligence dashboard.
Your job is to convert a user's natural language question into one or more SQL queries
and a complete dashboard configuration.

DATABASE SCHEMA
{SCHEMA_DESCRIPTION}

CHART TYPE RULES
{CHART_GUIDANCE}

OUTPUT FORMAT — you must ALWAYS return valid JSON. No markdown, no explanation.

Return a JSON object with this exact structure:

{{
  "answerable": true,
  "dashboard_title": "Short descriptive title of the dashboard",
  "insight": "2-3 sentence plain-English summary of what the data shows.",
  "charts": [
    {{
      "chart_id": "chart_1",
      "title": "Chart title",
      "chart_type": "bar",
      "sql": "SELECT ... FROM videos ...",
      "x_key": "column name to use as x-axis or label",
      "y_keys": ["column name(s) to use as y-axis or value"],
      "color_by": null,
      "description": "One sentence describing what this chart shows"
    }}
  ]
}}

If the question CANNOT be answered with the available data, return:
{{
  "answerable": false,
  "reason": "Clear explanation of why this cannot be answered."
}}

RULES:
1. Generate 1-4 charts per dashboard.
2. Every SQL must be a valid SQLite SELECT query on table `videos`.
3. Every SQL must include LIMIT 100 or less for aggregated results.
4. Always alias computed columns: AVG(views) AS avg_views, COUNT(*) AS video_count etc.
5. For time-series, always ORDER BY time column ASC.
6. For rankings, ORDER BY value DESC LIMIT 10.
7. x_key and y_keys must exactly match the column aliases in your SELECT.
8. ads_enabled is TEXT: use WHERE ads_enabled = 'True', not WHERE ads_enabled = 1.
9. sentiment_score ranges from -1 to 1.
10. Never make up data. Never hallucinate column names.
11. When generating multiple charts, each must show a DIFFERENT dimension.
12. For pie charts, the y_keys value column must be numeric.
"""

CORRECTION_PROMPT_TEMPLATE = """
Your previous SQL query failed with this error:

  Query: {sql}
  Error: {error}

Please fix the SQL and return the corrected full JSON response.
Common issues:
  - Column names: timestamp, video_id, category, language, region,
    duration_sec, views, likes, comments, shares, sentiment_score, ads_enabled
  - ads_enabled is TEXT: WHERE ads_enabled = 'True'
  - Use strftime('%Y-%m', timestamp) for monthly grouping
  - Table name is exactly: videos
  - All aggregated columns must be aliased
Return only the corrected JSON. No explanation.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}\nRaw:\n{text[:500]}")
        raise ValueError(f"Gemini returned invalid JSON: {e}")


def _call_gemini(messages: list[dict]) -> str:
    client = _get_client()
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            )
        )

    last_err: Exception | None = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                ),
            )
            return response.text
        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Only retry likely-transient failures.
            is_quota_exhausted = ("resource_exhausted" in msg) or ("quota" in msg)
            is_rate_limited = ("rate limit" in msg) or ("429" in msg)
            is_transient = any(
                token in msg
                for token in [
                    "timeout",
                    "timed out",
                    "deadline",
                    "temporarily unavailable",
                    "unavailable",
                    "503",
                    "500",
                    "connection reset",
                    "connection aborted",
                ]
            )
            retryable = is_rate_limited or (is_transient and not is_quota_exhausted)

            if attempt >= GEMINI_MAX_RETRIES or not retryable:
                raise

            delay = GEMINI_RETRY_BASE_DELAY_S * (2 ** attempt)
            logger.warning(
                f"Gemini call failed (attempt {attempt + 1}/{GEMINI_MAX_RETRIES + 1}); "
                f"retrying in {delay:.1f}s. Error: {e}"
            )
            time.sleep(delay)

    # Should be unreachable.
    raise last_err or RuntimeError("Gemini call failed for unknown reasons.")


def process_query(user_query: str, conversation_history: list[dict] = None) -> dict:
    if not os.environ.get("GEMINI_API_KEY", ""):
        # No key → no Gemini. Still return a renderable dashboard from DB.
        return _fallback_dashboard(user_query)

    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_query})

    cache_key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    cached = _cache.get(cache_key)
    now = time.time()
    if cached and (now - cached[0]) <= GEMINI_CACHE_TTL_S:
        return cached[1]

    try:
        raw_response = _call_gemini(messages)
        logger.info(f"Gemini response (first 300 chars): {raw_response[:300]}")
    except Exception as e:
        # Never fail the endpoint due to upstream LLM issues; fallback to DB-generated chart.
        logger.error(f"Gemini unavailable, using fallback. Error: {e}")
        return _fallback_dashboard(user_query)

    try:
        llm_output = _extract_json(raw_response)
    except ValueError as e:
        return {"success": False, "error": str(e)}

    if not llm_output.get("answerable", True):
        return {
            "success":    False,
            "answerable": False,
            "reason":     llm_output.get("reason", "This question cannot be answered with the available data."),
        }

    charts_config = llm_output.get("charts", [])
    if not charts_config:
        return {"success": False, "error": "Gemini returned no charts for this query."}

    charts_output = []

    # First pass: run all chart SQL once without correction.
    for chart in charts_config:
        sql         = chart.get("sql", "")
        chart_id    = chart.get("chart_id", "chart_1")
        chart_type  = _normalize_chart_type(chart.get("chart_type", "bar"))
        x_key       = chart.get("x_key", "")
        y_keys      = chart.get("y_keys", [])
        color_by    = chart.get("color_by")
        title       = chart.get("title", "Chart")
        description = chart.get("description", "")

        db_result = run_query(sql)

        if db_result["success"]:
            charts_output.append({
                "chart_id":    chart_id,
                "title":       title,
                "chart_type":  chart_type,
                "x_key":       x_key,
                "y_keys":      y_keys,
                "color_by":    color_by,
                "description": description,
                "data":        db_result["rows"],
                "columns":     db_result["columns"],
                "sql":         db_result["sql"],
                "row_count":   db_result["row_count"],
            })
        else:
            charts_output.append({
                "chart_id":   chart_id,
                "title":      title,
                "chart_type": chart_type,
                "error":      db_result["error"],
                "sql":        sql,
            })

    # Optional second pass: if ANY chart SQL failed, do a SINGLE Gemini correction call
    # that fixes the whole JSON output at once. This prevents blowing quota by correcting
    # each chart individually.
    failed_initial = [c for c in charts_output if "error" in c]
    if failed_initial and GEMINI_ENABLE_SELF_CORRECTION:
        logger.warning(
            f"{len(failed_initial)} chart SQL queries failed; attempting single self-correction pass..."
        )

        errors_block = "\n".join(
            [
                f"- {c.get('chart_id')}: SQL={c.get('sql')} | Error={c.get('error')}"
                for c in failed_initial
            ]
        )

        correction_prompt = (
            "Your previous JSON response produced SQL errors for some charts.\n\n"
            f"{errors_block}\n\n"
            "Please return a corrected full JSON response (same schema as before), fixing ONLY the SQL/x_key/y_keys "
            "as needed so all queries run on SQLite table `videos`. Return only JSON."
        )

        try:
            correction_messages = messages + [
                {"role": "model", "content": raw_response},
                {"role": "user", "content": correction_prompt},
            ]
            corrected_raw = _call_gemini(correction_messages)
            corrected_output = _extract_json(corrected_raw)
            corrected_charts = corrected_output.get("charts", [])

            if corrected_charts:
                corrected_map = {
                    c.get("chart_id", f"chart_{i+1}"): c for i, c in enumerate(corrected_charts)
                }

                # Re-run SQL for any previously-failed chart if we got a replacement.
                for idx, chart_out in enumerate(charts_output):
                    if "error" not in chart_out:
                        continue

                    chart_id = chart_out.get("chart_id")
                    replacement = corrected_map.get(chart_id) or corrected_charts[0]
                    sql = replacement.get("sql", chart_out.get("sql", ""))
                    chart_type = replacement.get("chart_type", chart_out.get("chart_type", "bar"))
                    x_key = replacement.get("x_key", chart_out.get("x_key", ""))
                    y_keys = replacement.get("y_keys", chart_out.get("y_keys", []))

                    db_result = run_query(sql)
                    if db_result["success"]:
                        charts_output[idx] = {
                            "chart_id": chart_id,
                            "title": chart_out.get("title", "Chart"),
                            "chart_type": chart_type,
                            "x_key": x_key,
                            "y_keys": y_keys,
                            "color_by": chart_out.get("color_by"),
                            "description": chart_out.get("description", ""),
                            "data": db_result["rows"],
                            "columns": db_result["columns"],
                            "sql": db_result["sql"],
                            "row_count": db_result["row_count"],
                        }
                    else:
                        charts_output[idx] = {
                            "chart_id": chart_id,
                            "title": chart_out.get("title", "Chart"),
                            "chart_type": chart_type,
                            "error": db_result["error"],
                            "sql": sql,
                        }
        except Exception as e:
            logger.error(f"Single self-correction pass failed: {e}")

    successful = [c for c in charts_output if "error" not in c]
    failed     = [c for c in charts_output if "error" in c]

    if not successful:
        return {
            "success": False,
            "error":   "All SQL queries failed even after self-correction.",
            "charts":  failed,
        }

    # Ensure the first chart is a successful one (frontend renders the "primary" chart).
    charts_ordered = successful + failed

    primary = charts_ordered[0]
    # Provide a backwards-compatible "single chart" shape expected by the current UI.
    # The UI uses: title, chart_type, labels, datasets[0].data, datasets[0].name
    labels: list = []
    values: list = []
    dataset_name = ""
    if primary and "error" not in primary:
        labels, values, dataset_name = _build_primary_series(primary)

    result = {
        "success":         True,
        "dashboard_title": llm_output.get("dashboard_title", "Dashboard"),
        "insight":         llm_output.get("insight", ""),
        "charts":          charts_ordered,
        "failed_charts":   failed,
        "query":           user_query,

        # Primary chart (legacy fields used by DashboardPage.jsx)
        "title":           primary.get("title", "") if primary else "",
        "chart_type":      _normalize_chart_type(primary.get("chart_type", "bar")) if primary else "bar",
        "labels":          labels,
        "datasets":        ([{"name": dataset_name, "data": values}] if labels else []),
    }
    _cache[cache_key] = (time.time(), result)
    return result