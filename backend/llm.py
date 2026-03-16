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

from google import genai
from google.genai import types

from schema import SCHEMA_DESCRIPTION, CHART_GUIDANCE
from db import run_query

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client         = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME  = "gemini-2.0-flash"
TEMPERATURE = 0.1

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
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            )
        )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_output_tokens=4096,
        ),
    )
    return response.text


def process_query(user_query: str, conversation_history: list[dict] = None) -> dict:
    if not GEMINI_API_KEY:
        return {"success": False, "error": "GEMINI_API_KEY environment variable is not set."}

    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": user_query})

    try:
        raw_response = _call_gemini(messages)
        logger.info(f"Gemini response (first 300 chars): {raw_response[:300]}")
    except Exception as e:
        return {"success": False, "error": f"Gemini API error: {str(e)}"}

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

    for chart in charts_config:
        sql         = chart.get("sql", "")
        chart_id    = chart.get("chart_id", "chart_1")
        chart_type  = chart.get("chart_type", "bar")
        x_key       = chart.get("x_key", "")
        y_keys      = chart.get("y_keys", [])
        color_by    = chart.get("color_by")
        title       = chart.get("title", "Chart")
        description = chart.get("description", "")

        db_result = run_query(sql)

        if not db_result["success"]:
            logger.warning(f"SQL failed for {chart_id}, attempting self-correction...")
            correction_messages = messages + [
                {"role": "model", "content": raw_response},
                {"role": "user",  "content": CORRECTION_PROMPT_TEMPLATE.format(
                    sql=sql, error=db_result["error"]
                )},
            ]
            try:
                corrected_raw    = _call_gemini(correction_messages)
                corrected_output = _extract_json(corrected_raw)
                corrected_charts = corrected_output.get("charts", [])
                matching = next(
                    (c for c in corrected_charts if c.get("chart_id") == chart_id),
                    corrected_charts[0] if corrected_charts else None,
                )
                if matching:
                    sql        = matching.get("sql", sql)
                    chart_type = matching.get("chart_type", chart_type)
                    x_key      = matching.get("x_key", x_key)
                    y_keys     = matching.get("y_keys", y_keys)
                    db_result  = run_query(sql)
            except Exception as e:
                logger.error(f"Self-correction failed: {e}")

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

    successful = [c for c in charts_output if "error" not in c]
    failed     = [c for c in charts_output if "error" in c]

    if not successful:
        return {
            "success": False,
            "error":   "All SQL queries failed even after self-correction.",
            "charts":  failed,
        }

    return {
        "success":         True,
        "dashboard_title": llm_output.get("dashboard_title", "Dashboard"),
        "insight":         llm_output.get("insight", ""),
        "charts":          charts_output,
        "failed_charts":   failed,
        "query":           user_query,
    }