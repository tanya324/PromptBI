"""
db.py — Handles all database operations.
  - Loads CSV into SQLite once at startup (or reuses existing DB)
  - Validates SQL before execution (blocks destructive queries)
  - Runs queries and returns results as list of dicts
"""

import sqlite3
import pandas as pd
import os
import re
import logging

logger = logging.getLogger(__name__)

DB_PATH      = "videos.db"
TABLE_NAME   = "videos"
CSV_PATH     = "data.csv"          # default; overridden when user uploads
MAX_ROWS     = 500                  # hard cap on rows returned to frontend


# ── Dangerous SQL patterns we never allow ─────────────────────────────────────
BLOCKED_PATTERNS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bALTER\b",
    r"\bCREATE\b",
    r"\bTRUNCATE\b",
    r"\bREPLACE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bPRAGMA\b",
    r"--",           # SQL comment (could be used to escape clauses)
    r";.*SELECT",    # stacked queries
]


def _validate_sql(sql: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Blocks anything that isn't a read-only SELECT.
    """
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return False, "Only SELECT queries are allowed."

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, sql, re.IGNORECASE):
            return False, f"Query contains a disallowed operation: '{pattern}'"

    return True, "ok"


def _inject_limit(sql: str) -> str:
    """
    Ensures the query has a LIMIT clause capped at MAX_ROWS.
    If the query already has LIMIT N where N > MAX_ROWS, reduce it.
    If no LIMIT, append one.
    """
    sql = sql.rstrip().rstrip(";")

    limit_match = re.search(r"\bLIMIT\s+(\d+)", sql, re.IGNORECASE)
    if limit_match:
        existing = int(limit_match.group(1))
        if existing > MAX_ROWS:
            sql = re.sub(
                r"\bLIMIT\s+\d+", f"LIMIT {MAX_ROWS}", sql, flags=re.IGNORECASE
            )
    else:
        sql = f"{sql} LIMIT {MAX_ROWS}"

    return sql


def load_csv_to_db(csv_path: str = CSV_PATH, db_path: str = DB_PATH) -> dict:
    """
    Loads a CSV file into SQLite.
    Called once at startup for the default dataset, and again each time a
    user uploads their own CSV.

    Returns metadata about the loaded table.
    """
    logger.info(f"Loading CSV: {csv_path} → {db_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    # ── Basic cleaning ────────────────────────────────────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # ── Write to SQLite ───────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    # Add indexes on the columns most likely to appear in WHERE / GROUP BY
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_category  ON {TABLE_NAME}(category)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_region    ON {TABLE_NAME}(region)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_language  ON {TABLE_NAME}(language)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_timestamp ON {TABLE_NAME}(timestamp)")
    conn.commit()
    conn.close()

    # ── Return metadata ───────────────────────────────────────────────────────
    meta = {
        "table":    TABLE_NAME,
        "rows":     len(df),
        "columns":  list(df.columns),
        "dtypes":   {col: str(dtype) for col, dtype in df.dtypes.items()},
        "db_path":  db_path,
    }

    logger.info(f"Loaded {meta['rows']:,} rows, {len(meta['columns'])} columns")
    return meta


def run_query(sql: str, db_path: str = DB_PATH) -> dict:
    """
    Validates and executes a SQL query.

    Returns:
        {
            "success": True,
            "columns": ["col1", "col2", ...],
            "rows":    [{"col1": val, "col2": val}, ...],
            "row_count": N,
            "sql": "SELECT ..."       # the actual SQL that was run
        }

    Or on failure:
        {
            "success": False,
            "error":   "human-readable error message",
            "sql":     "SELECT ..."
        }
    """
    # 1. Validate
    is_safe, reason = _validate_sql(sql)
    if not is_safe:
        return {"success": False, "error": reason, "sql": sql}

    # 2. Enforce row limit
    sql_limited = _inject_limit(sql)

    # 3. Execute
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row          # lets us access columns by name
        cur  = conn.cursor()
        cur.execute(sql_limited)
        raw_rows = cur.fetchall()
        columns  = [desc[0] for desc in cur.description]
        conn.close()

        rows = [dict(zip(columns, row)) for row in raw_rows]

        return {
            "success":   True,
            "columns":   columns,
            "rows":      rows,
            "row_count": len(rows),
            "sql":       sql_limited,
        }

    except sqlite3.OperationalError as e:
        logger.error(f"SQL error: {e}\nQuery: {sql_limited}")
        return {
            "success": False,
            "error":   f"SQL execution error: {str(e)}",
            "sql":     sql_limited,
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "success": False,
            "error":   f"Unexpected error: {str(e)}",
            "sql":     sql_limited,
        }


def get_table_preview(db_path: str = DB_PATH, n: int = 5) -> dict:
    """
    Returns the first N rows of the videos table.
    Used by the frontend to show a data preview after CSV upload.
    """
    return run_query(f"SELECT * FROM {TABLE_NAME} LIMIT {n}", db_path)


def get_db_stats(db_path: str = DB_PATH) -> dict:
    """
    Returns basic stats about the loaded dataset.
    Shown in the UI sidebar.
    """
    result = run_query(
        f"""
        SELECT
            COUNT(*)                                        AS total_videos,
            COUNT(DISTINCT category)                        AS categories,
            COUNT(DISTINCT region)                          AS regions,
            COUNT(DISTINCT language)                        AS languages,
            ROUND(AVG(views), 0)                           AS avg_views,
            MAX(views)                                      AS max_views,
            strftime('%Y-%m', MIN(timestamp))               AS earliest,
            strftime('%Y-%m', MAX(timestamp))               AS latest
        FROM {TABLE_NAME}
        """,
        db_path,
    )

    if result["success"] and result["rows"]:
        return result["rows"][0]
    return {}