<div align="center">

<br/>

### **Conversational AI for Instant Business Intelligence**
*Ask a question in plain English. Get a live interactive dashboard in seconds.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlite.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<br/>

> *"Show me monthly view trends broken down by category and highlight the top-performing region"*
> → **Full interactive dashboard. No SQL. No configuration. Just ask.**

<br/>

</div>

---

## 📖 Table of Contents

- [What is Query Dash?](#-what-is-query-dash)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Docker Deployment](#-docker-deployment)
- [Example Queries](#-example-queries)
- [Team](#-team)

---

## 🧠 What is Query Dash?

**Query Dash** is an AI-powered Business Intelligence platform that eliminates the gap between business questions and data answers.

In most organisations, getting a simple chart requires:
1. Writing a ticket to the data team
2. Waiting 2–3 days
3. Receiving a static screenshot

**Query Dash changes this entirely.** A non-technical executive types a question in plain English — and within seconds, a fully interactive, multi-chart dashboard appears. No SQL. No BI tool training. No waiting.

### Who is it for?

| Persona | Problem Solved |
|---|---|
| 🏢 CXO / Executive | Gets instant answers without depending on data teams |
| 📊 Business Analyst | Explores data 10x faster without writing queries |
| 🎓 Student / Researcher | Analyses large datasets through conversation |

---

## 🎬 Live Demo

> Try these three queries to see the full power of the system:

| Query | What it demonstrates |
|---|---|
| `"Show me monthly view trends for 2024 broken down by category"` | Time-series line chart with multi-category breakdown |
| `"Which region has the highest average engagement rate and how does sentiment compare?"` | Multi-chart dashboard with computed metrics |
| `"Compare total views for videos with ads enabled vs disabled across all languages"` | Grouped bar chart with filter logic |

---

## ✨ Features

### Core Features
- 🗣️ **Natural Language Queries** — Type questions exactly as you'd ask a colleague
- 📊 **Auto Chart Selection** — AI picks the right chart type (bar, line, pie, scatter, area) automatically
- 🧩 **Multi-Chart Dashboards** — One question generates 1–4 complementary charts
- 💡 **Plain-English Insights** — Every dashboard includes a human-readable summary
- ⚡ **Real-Time Rendering** — Charts appear within seconds of asking

### Intelligence Features
- 🤖 **Gemini 2.0 Flash** — Powered by Google's latest fast LLM
- 🛡️ **Hallucination Guard** — System reports when it can't answer rather than making up data
- 🔄 **Agentic Self-Correction** — If SQL fails, AI automatically fixes and retries
- 📐 **Schema-Aware Prompting** — Full database context injected into every query

### UX Features
- 💬 **Follow-up Queries** — Chat with your dashboard: *"Now filter this to only the US"*
- 📁 **CSV Upload** — Upload any CSV and immediately start querying it
- 🎯 **Sample Question Chips** — One-click starter questions for new users
- ⏳ **Animated Loading States** — Progress indicators while dashboard generates
- 📱 **Responsive Design** — Works on desktop and tablet

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   User types: "Show monthly views by category"              │
│                        │                                    │
│                        ▼                                    │
│   ┌─────────────────────────────────────────┐               │
│   │           FASTAPI BACKEND               │               │
│   │                                         │               │
│   │  1. Inject schema context into prompt   │               │
│   │  2. Call Gemini 2.0 Flash               │               │
│   │  3. Receive SQL + chart config as JSON  │               │
│   │  4. Validate & execute SQL on SQLite    │               │
│   │  5. Self-correct if SQL fails           │               │
│   │  6. Return chart data to frontend       │               │
│   └─────────────────────────────────────────┘               │
│                        │                                    │
│                        ▼                                    │
│   React renders interactive Recharts dashboard              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Prompt Engineering Pipeline

What makes Query Dash accurate is not just calling an LLM — it's **what we tell the LLM before every query**:

1. **Schema Injection** — Full table structure, column names, data types, and value ranges are prepended to every prompt so Gemini never guesses column names
2. **Chart Selection Rules** — Explicit rules map data shapes to chart types (time-series → line, categorical → bar, parts-of-whole → pie)
3. **SQL Guard Rules** — Edge cases like `ads_enabled` being stored as TEXT `'True'`/`'False'` are documented so SQL is correct first time
4. **Structured JSON Output** — Gemini is instructed to return a strict JSON schema — no free-text parsing needed
5. **Self-Correction Loop** — Failed SQL is sent back to Gemini with the error message for automatic fixing

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Google Gemini 2.0 Flash | Natural language → SQL + chart config |
| **Backend** | Python 3.11 + FastAPI | REST API, query orchestration |
| **Database** | SQLite | Fast local query execution on 1M rows |
| **Frontend** | React 18 + Vite | Interactive UI |
| **Charts** | Recharts | Responsive interactive visualisations |
| **Styling** | Tailwind CSS | Dark theme dashboard UI |
| **HTTP** | Axios | Frontend ↔ Backend communication |
| **Deployment** | Docker + Docker Compose | One-command deployment |

---

## 📁 Project Structure

```
Query-Dash/
│
├── 📂 backend/
│   ├── main.py           ← FastAPI server (API routes)
│   ├── llm.py            ← Gemini integration + prompt engineering
│   ├── db.py             ← SQLite query runner + SQL validation
│   ├── schema.py         ← Database schema metadata for AI context
│   ├── requirements.txt  ← Python dependencies
│   └── Dockerfile        ← Backend container
│
├── 📂 frontend/
│   ├── src/
│   │   ├── App.jsx                  ← Main app shell
│   │   ├── components/
│   │   │   ├── QueryInput.jsx       ← Search bar + sample chips
│   │   │   ├── DashboardGrid.jsx    ← Chart layout grid
│   │   │   ├── ChartCard.jsx        ← Individual chart renderer
│   │   │   └── InsightCard.jsx      ← AI insight display
│   │   └── services/
│   │       └── api.js               ← All backend API calls
│   ├── Dockerfile         ← Frontend container
│   └── nginx.conf         ← Production static file server
│
├── docker-compose.yml     ← Runs everything with one command
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have these installed:
- [Python 3.11+](https://python.org/downloads)
- [Node.js 18+](https://nodejs.org)
- [Git](https://git-scm.com)
- A free [Gemini API Key](https://aistudio.google.com/apikey)

---

### Option A — Run Locally (Development)

**1. Clone the repository**
```bash
git clone https://github.com/tanya324/Query-Dash.git
cd Query-Dash
```

**2. Set up the backend**
```bash
cd backend
python -m pip install -r requirements.txt
```

**3. Add your API key**

Create a `.env` file inside the `backend` folder:
```
GEMINI_API_KEY=your_api_key_here
```

**4. Add your dataset**

Place your CSV file inside the `backend` folder and rename it to `data.csv`.

**5. Start the backend**
```bash
python main.py
```
✅ Backend running at `http://localhost:8000`
✅ API docs at `http://localhost:8000/docs`

**6. Set up and start the frontend** (new terminal)
```bash
cd frontend
npm install
npm run dev
```
✅ Frontend running at `http://localhost:5173`

---

### Option B — Run with Docker (One Command)

```bash
git clone https://github.com/tanya324/Query-Dash.git
cd Query-Dash

# Add your API key to a .env file in the root
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Place your data.csv in the backend folder, then:
docker-compose up --build
```

✅ App live at `http://localhost`

---

## 📡 API Reference

### `POST /query`
Convert a natural language question into a dashboard.

**Request:**
```json
{
  "query": "Show me monthly views by category",
  "session_id": null,
  "conversation_history": []
}
```

**Response:**
```json
{
  "success": true,
  "dashboard_title": "Monthly Views by Category — 2024",
  "insight": "Education and Tech Reviews show strong growth through Q3...",
  "charts": [
    {
      "chart_id": "chart_1",
      "title": "Monthly Views Trend",
      "chart_type": "line",
      "x_key": "month",
      "y_keys": ["total_views"],
      "data": [{ "month": "2024-01", "total_views": 4231500 }]
    }
  ]
}
```

### `GET /stats`
Returns dataset statistics for the UI header.

### `GET /samples`
Returns 8 sample questions shown as clickable chips.

### `POST /upload-csv`
Upload a custom CSV file. Returns a `session_id` for querying it.

### `GET /health`
Health check endpoint.

---

## 🐳 Docker Deployment

The entire stack is containerised and connects automatically:

```
┌─────────────────────────────────────────┐
│          Docker Compose Network         │
│                                         │
│  ┌──────────────┐   ┌────────────────┐  │
│  │   Frontend   │──▶│    Backend     │  │
│  │  React+Nginx │   │    FastAPI     │  │
│  │   Port: 80   │   │   Port: 8000   │  │
│  └──────────────┘   └───────┬────────┘  │
│                             │           │
│                      ┌──────▼──────┐    │
│                      │   SQLite    │    │
│                      │  videos.db  │    │
│                      └─────────────┘    │
└─────────────────────────────────────────┘
```

```bash
docker-compose up --build    # Start everything
docker-compose down          # Stop everything
docker-compose logs backend  # View backend logs
```

---

## 💬 Example Queries

Here are queries you can try, from simple to complex:

**Simple**
```
Show me total views by region
```
```
What are the top 5 categories by likes?
```

**Intermediate**
```
Compare average sentiment score across all languages
```
```
Show videos with ads enabled vs disabled — which gets more views?
```

**Complex**
```
Show me monthly view trends for 2024 broken down by category
and tell me which category peaked earliest
```
```
Give me a full engagement analysis — views, likes, comments
and shares broken down by region
```

**Follow-up (chat with dashboard)**
```
First ask: "Show me views by region"
Then ask:  "Now filter this to only show English language videos"
Then ask:  "Which of these regions has the best sentiment score?"
```

---

## 👥 Team

| Name | Role |
|---|---|
| Tanya Srivastava | Backend + AI/ML + Project Lead |
| Suraj Singh Guleria | Frontend + UI/UX |
| Samkit Jain | Frontend + UI/UX |

*B.Tech — Artificial Intelligence & Data Science*
*Semester 4 — Under the guidance of Dr. Neha Garg*

---

## 📄 License

This project was built as part of a B.Tech academic hackathon project.

---

<div align="center">

<br/>

*Built with 🤖 Gemini AI · ⚡ FastAPI · ⚛️ React*

**Query Dash — because data should answer to you, not the other way around.**

<br/>

</div>