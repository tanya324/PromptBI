/**
 * api.js — All backend API calls in one place.
 * 
 * HOW TO USE:
 * 1. Copy this file into your React project: src/services/api.js
 * 2. Create a .env file in your React project root with:
 *    VITE_API_URL=http://localhost:8000
 * 3. Import any function you need:
 *    import { submitQuery, getStats, getSamples, uploadCSV } from './services/api'
 * 
 * IMPORTANT:
 * - Make sure the backend server is running on http://localhost:8000
 * - Make sure you have axios installed: npm install axios
 */

import axios from "axios";

// ── Base URL — reads from .env file ──────────────────────────────────────────
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Axios instance — all requests go through this ────────────────────────────
const api = axios.create({
  baseURL: API_URL,
  timeout: 120000, // 120 seconds — Gemini can take a moment on complex queries
  headers: {
    "Content-Type": "application/json",
  },
});


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FUNCTION 1 — submitQuery
// Send the user's natural language question to the backend.
// Returns a full dashboard config with charts and insight.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @param {string} query               - The user's natural language question
 * @param {string|null} sessionId      - Pass this if user uploaded their own CSV
 * @param {Array} conversationHistory  - Pass previous turns for follow-up queries
 * 
 * @returns {Object} result
 * 
 * SUCCESS:
 * {
 *   success: true,
 *   dashboard_title: "Views by Region Analysis",
 *   insight: "The US leads in total views...",
 *   charts: [
 *     {
 *       chart_id:    "chart_1",
 *       title:       "Views by Region",
 *       chart_type:  "bar",           // "bar" | "line" | "pie" | "scatter" | "area"
 *       x_key:       "region",        // use this as X axis in Recharts
 *       y_keys:      ["avg_views"],   // use this as Y axis in Recharts
 *       description: "Average views per region",
 *       data: [                       // pass this directly to Recharts
 *         { region: "US", avg_views: 73000 },
 *         { region: "IN", avg_views: 65000 }
 *       ]
 *     }
 *   ]
 * }
 * 
 * UNANSWERABLE (Gemini says it can't answer):
 * {
 *   success: false,
 *   answerable: false,
 *   reason: "This dataset does not contain salary information."
 * }
 * 
 * ERROR:
 * {
 *   success: false,
 *   error: "Something went wrong..."
 * }
 */
export async function submitQuery(
  query,
  sessionId = null,
  conversationHistory = []
) {
  try {
    const response = await api.post("/query", {
      query:                query,
      session_id:           sessionId,
      conversation_history: conversationHistory,
    });
    return response.data;

  } catch (error) {
    // Network error or server is down
    if (error.code === "ECONNREFUSED" || error.code === "ERR_NETWORK") {
      return {
        success: false,
        error:   "Cannot connect to the backend server. Make sure it is running on port 8000.",
      };
    }
    return {
      success: false,
      error:   error.response?.data?.detail || error.message || "Unknown error occurred.",
    };
  }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FUNCTION 2 — getStats
// Fetch dataset statistics to show in the UI sidebar or header.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @param {string|null} sessionId - Pass if user uploaded their own CSV
 * 
 * @returns {Object}
 * {
 *   total_videos: 1000000,
 *   categories:   6,
 *   regions:      5,
 *   languages:    5,
 *   avg_views:    73412,
 *   max_views:    147139,
 *   earliest:     "2024-01",
 *   latest:       "2025-12"
 * }
 */
export async function getStats(sessionId = null) {
  try {
    const params   = sessionId ? { session_id: sessionId } : {};
    const response = await api.get("/stats", { params });
    return { success: true, data: response.data };

  } catch (error) {
    return {
      success: false,
      error:   error.response?.data?.detail || error.message,
    };
  }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FUNCTION 3 — getSamples
// Fetch sample questions to show as clickable chips in the UI.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @returns {Object}
 * {
 *   success:   true,
 *   questions: [
 *     "Show me monthly view trends by category",
 *     "Which region has the highest engagement rate?",
 *     ...
 *   ]
 * }
 */
export async function getSamples() {
  try {
    const response = await api.get("/samples");
    return { success: true, questions: response.data.questions };

  } catch (error) {
    return {
      success:   false,
      questions: [],
      error:     error.response?.data?.detail || error.message,
    };
  }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FUNCTION 4 — uploadCSV
// Upload a CSV file and get back a session_id for querying it.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @param {File} file             - The CSV File object from an <input type="file">
 * @param {Function} onProgress   - Optional callback: (percent) => void
 * 
 * @returns {Object}
 * 
 * SUCCESS:
 * {
 *   success:    true,
 *   session_id: "abc123",       // SAVE THIS — pass it to submitQuery
 *   rows:       50000,
 *   columns:    ["col1", "col2", ...],
 *   preview:    [ first 5 rows as array of objects ],
 *   message:    "Successfully loaded 50,000 rows and 8 columns."
 * }
 * 
 * ERROR:
 * {
 *   success: false,
 *   error:   "Only CSV files are supported."
 * }
 */
export async function uploadCSV(file, onProgress = null) {
  try {
    // Must use FormData for file uploads — not JSON
    const formData = new FormData();
    formData.append("file", file);

    const response = await api.post("/upload-csv", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percent = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          onProgress(percent);
        }
      },
    });

    return { success: true, ...response.data };

  } catch (error) {
    return {
      success: false,
      error:   error.response?.data?.detail || error.message || "Upload failed.",
    };
  }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FUNCTION 5 — checkHealth
// Check if the backend server is running. 
// Call this on app load to show a "server offline" warning if needed.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * @returns {Object}
 * { success: true,  status: "ok", model: "gemini-2.0-flash" }
 * { success: false, error: "Cannot connect to backend" }
 */
export async function checkHealth() {
  try {
    const response = await api.get("/health");
    return { success: true, ...response.data };

  } catch (error) {
    return {
      success: false,
      error:   "Cannot connect to backend server.",
    };
  }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// USAGE EXAMPLES — copy paste these into your components
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/*

// ── Example 1: Basic query ───────────────────────────────────────────────────
const result = await submitQuery("Show me views by region")
if (result.success) {
  console.log(result.dashboard_title)  // "Views by Region Analysis"
  console.log(result.insight)          // "The US leads in total views..."
  console.log(result.charts)           // array of chart objects
} else if (result.answerable === false) {
  console.log(result.reason)           // "This data doesn't have salary info"
} else {
  console.log(result.error)            // "Something went wrong"
}


// ── Example 2: Follow-up query (chat with dashboard) ────────────────────────
const history = [
  { role: "user",  content: "Show me views by region" },
  { role: "model", content: JSON.stringify(previousResult) }
]
const followUp = await submitQuery("Now filter this to only the US", null, history)


// ── Example 3: Upload CSV then query it ─────────────────────────────────────
const fileInput = document.querySelector('input[type="file"]')
const file      = fileInput.files[0]

const upload = await uploadCSV(file, (percent) => {
  console.log(`Upload progress: ${percent}%`)
})

if (upload.success) {
  const sessionId = upload.session_id   // save this!
  const result    = await submitQuery("Show me the top categories", sessionId)
}


// ── Example 4: Load stats on app startup ────────────────────────────────────
useEffect(() => {
  async function loadStats() {
    const stats = await getStats()
    if (stats.success) {
      setDatasetStats(stats.data)  // { total_videos: 1000000, ... }
    }
  }
  loadStats()
}, [])


// ── Example 5: Load sample questions as chips ───────────────────────────────
useEffect(() => {
  async function loadSamples() {
    const result = await getSamples()
    if (result.success) {
      setSampleQuestions(result.questions)
    }
  }
  loadSamples()
}, [])


// ── Example 6: Check if backend is online ───────────────────────────────────
useEffect(() => {
  async function ping() {
    const health = await checkHealth()
    if (!health.success) {
      setError("Backend server is offline. Please start it on port 8000.")
    }
  }
  ping()
}, [])

*/
