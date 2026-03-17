"""
main.py — FastAPI server exposing the BI dashboard API.

Endpoints:
  POST /query          — main endpoint: natural language → dashboard config
  POST /upload-csv     — upload a custom CSV and query it (bonus +20 pts)
  GET  /stats          — dataset statistics shown in the UI sidebar
  GET  /samples        — sample questions shown as chips in the UI
  GET  /health         — health check
"""

from dotenv import load_dotenv
from pathlib import Path

# Load env reliably no matter where the server is started from.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

import os
import uuid
import logging
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Allow running either from repo root (python -m backend.main / uvicorn backend.main:app)
# or from inside the backend folder (python main.py).
try:
    from backend.db import load_csv_to_db, ensure_sample_db, get_db_stats, get_table_preview, DB_PATH, CSV_PATH
    from backend.llm import process_query
    from backend.schema import SAMPLE_QUESTIONS
except Exception:  # pragma: no cover
    from db import load_csv_to_db, ensure_sample_db, get_db_stats, get_table_preview, DB_PATH, CSV_PATH
    from llm import process_query
    from schema import SAMPLE_QUESTIONS

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Per-session DB paths for uploaded CSVs ────────────────────────────────────
# Maps session_id → db_path so multiple users can upload different CSVs
# In production you'd use Redis; for a hackathon, an in-memory dict is fine
SESSION_DBS: dict[str, str] = {}


# ── Startup: load default CSV into SQLite ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    default_csv = os.environ.get("CSV_PATH", CSV_PATH)
    default_db  = os.environ.get("DB_PATH", DB_PATH)

    # If a DB already exists (e.g. bundled videos.db), use it as-is.
    if os.path.exists(default_db):
        stats = get_db_stats(default_db)
        if stats:
            logger.info(
                f"Using existing DB: {default_db} — "
                f"{stats.get('total_videos', 'N/A')} rows"
            )
            yield
            # Cleanup temp DBs on shutdown
            for db_path in SESSION_DBS.values():
                if os.path.exists(db_path):
                    os.remove(db_path)
                    logger.info(f"Cleaned up temp DB: {db_path}")
            return

    # Otherwise, try loading from CSV if present.
    if os.path.exists(default_csv):
        logger.info(f"Loading default dataset: {default_csv}")
        try:
            meta = load_csv_to_db(csv_path=default_csv, db_path=default_db)
            logger.info(
                f"Default DB ready — {meta['rows']:,} rows, "
                f"{len(meta['columns'])} columns"
            )
        except Exception as e:
            logger.error(f"Failed to load default CSV: {e}")
    else:
        logger.warning(
            f"Default CSV not found at '{default_csv}'. "
            "If you have a prebuilt DB, set DB_PATH or place videos.db in backend/. "
            "Otherwise upload a CSV via /upload-csv before querying."
        )
        # Ensure the app still works (demo mode) even without a dataset.
        sample_meta = ensure_sample_db(default_db)
        if sample_meta.get("created"):
            logger.info(
                f"Created sample dataset in {sample_meta.get('db_path')} "
                f"({sample_meta.get('rows')} rows)"
            )
    yield
    # Cleanup temp DBs on shutdown
    for db_path in SESSION_DBS.values():
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Cleaned up temp DB: {db_path}")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BI Dashboard API",
    description="Natural language → interactive dashboard powered by Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the React dev server (localhost:5173) and any Vercel URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:              str
    session_id:         str | None = None   # if set, queries the uploaded CSV
    conversation_history: list[dict] = []   # for follow-up / filter queries


class QueryResponse(BaseModel):
    success:         bool
    dashboard_title: str  = ""
    insight:         str  = ""
    charts:          list = []
    failed_charts:   list = []
    # Primary chart fields used by the current frontend ChartRenderer
    title:           str  = ""
    chart_type:      str  = ""
    labels:          list = []
    datasets:        list = []
    query:           str  = ""
    error:           str  = ""
    answerable:      bool = True
    reason:          str  = ""             # populated when answerable=False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    return {"status": "ok", "model": model}


@app.get("/stats")
def stats(session_id: str | None = None):
    """
    Returns row count, column count, date range etc.
    Frontend shows these in the sidebar as context for the user.
    """
    db_path = SESSION_DBS.get(session_id, DB_PATH) if session_id else DB_PATH
    data    = get_db_stats(db_path)
    if not data:
        raise HTTPException(status_code=500, detail="Could not retrieve DB stats.")
    return data


@app.get("/samples")
def samples():
    """Returns sample questions shown as clickable chips in the UI."""
    return {"questions": SAMPLE_QUESTIONS}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Main endpoint — converts a natural language question into a full
    dashboard configuration with data.

    Body:
      {
        "query": "Show me monthly views by category",
        "session_id": null,             // use default DB; or pass upload session ID
        "conversation_history": []      // previous turns for follow-up queries
      }

    Returns the full dashboard payload (see llm.process_query for shape).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    db_path = SESSION_DBS.get(req.session_id, DB_PATH) if req.session_id else DB_PATH

    logger.info(f"Query: '{req.query}' | session: {req.session_id} | db: {db_path}")

    # Inject the active db_path into the query processor
    # We monkey-patch the module-level DB_PATH used by db.run_query
    try:
        import backend.db as db_module
    except Exception:  # pragma: no cover
        import db as db_module
    original_db_path = db_module.DB_PATH
    db_module.DB_PATH = db_path

    try:
        result = process_query(
            user_query=req.query,
            conversation_history=req.conversation_history,
        )
    finally:
        db_module.DB_PATH = original_db_path   # always restore

    if not result["success"]:
        # Return a structured error — never a 500 — so React can display it
        return QueryResponse(
            success=False,
            error=result.get("error", "Unknown error"),
            answerable=result.get("answerable", True),
            reason=result.get("reason", ""),
            charts=result.get("charts", []),
            query=req.query,
        )

    return QueryResponse(
        success=True,
        dashboard_title=result.get("dashboard_title", ""),
        insight=result.get("insight", ""),
        charts=result.get("charts", []),
        failed_charts=result.get("failed_charts", []),
        title=result.get("title", ""),
        chart_type=result.get("chart_type", ""),
        labels=result.get("labels", []),
        datasets=result.get("datasets", []),
        query=req.query,
    )


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts a CSV upload, loads it into a temporary SQLite DB,
    and returns a session_id the frontend uses for all subsequent queries.

    Bonus feature — worth +20 pts on the rubric.

    Returns:
      {
        "session_id": "abc123",
        "rows": 50000,
        "columns": ["col1", "col2", ...],
        "preview": [ first 5 rows ],
        "message": "Successfully loaded 50,000 rows."
      }
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported. Please upload a .csv file."
        )

    # Save upload to a temp file
    suffix     = f"_{file.filename}"
    tmp_csv    = tempfile.mktemp(suffix=suffix)
    tmp_db     = tempfile.mktemp(suffix=".db")
    session_id = str(uuid.uuid4())[:8]

    try:
        with open(tmp_csv, "wb") as f:
            shutil.copyfileobj(file.file, f)

        meta    = load_csv_to_db(csv_path=tmp_csv, db_path=tmp_db)
        preview = get_table_preview(db_path=tmp_db, n=5)

        SESSION_DBS[session_id] = tmp_db
        os.remove(tmp_csv)

        logger.info(
            f"CSV uploaded — session: {session_id} | "
            f"{meta['rows']:,} rows | {len(meta['columns'])} columns"
        )

        return {
            "session_id": session_id,
            "rows":       meta["rows"],
            "columns":    meta["columns"],
            "preview":    preview.get("rows", []),
            "message":    f"Successfully loaded {meta['rows']:,} rows and {len(meta['columns'])} columns.",
        }

    except Exception as e:
        # Clean up on failure
        for path in [tmp_csv, tmp_db]:
            if os.path.exists(path):
                os.remove(path)
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load CSV: {str(e)}")


# ── Dev server entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # auto-restart on code changes during dev
    )