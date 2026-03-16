"""
schema.py — Database schema metadata injected into every Gemini prompt.
This is what lets Gemini generate accurate SQL without hallucinating column names.
"""

# Human-readable schema description passed to Gemini as context
SCHEMA_DESCRIPTION = """
You have access to a SQLite database with a single table called `videos`.

TABLE: videos
=============
Columns:
  - timestamp       (TEXT)     : Date and time the video was posted. Format: 'YYYY-MM-DD HH:MM:SS'
                                  Use strftime('%Y-%m', timestamp) for monthly grouping.
                                  Use strftime('%Y', timestamp) for yearly grouping.
                                  Use cast(strftime('%m', timestamp) as integer) for month number.
  - video_id        (TEXT)     : Unique identifier for each video. e.g. 'VID_589765'
  - category        (TEXT)     : Content category. Possible values: 'Vlogs', 'Education', 'Coding', 'Gaming', 'Music', 'Tech Reviews'
  - language        (TEXT)     : Language of the video. Possible values: 'English', 'Hindi', 'Urdu', 'Spanish', 'Japanese'
  - region          (TEXT)     : Region/country code. Possible values: 'US', 'PK', 'BR', 'IN', 'UK'
  - duration_sec    (INTEGER)  : Duration of the video in seconds. Range: 60 to 3599.
                                  To convert to minutes: ROUND(duration_sec / 60.0, 1)
  - views           (INTEGER)  : Total view count. Range: 0 to ~147,000
  - likes           (INTEGER)  : Total like count.
  - comments        (INTEGER)  : Total comment count.
  - shares          (INTEGER)  : Total share count.
  - sentiment_score (REAL)     : Sentiment score of comments. Range: -1.0 (negative) to +1.0 (positive)
  - ads_enabled     (TEXT)     : Whether ads are enabled. Values: 'True' or 'False' (stored as text, not boolean)

TOTAL ROWS: ~1,000,000

USEFUL DERIVED METRICS (compute these in SQL):
  - Engagement rate  : ROUND((likes + comments + shares) * 100.0 / NULLIF(views, 0), 2)
  - Like rate        : ROUND(likes * 100.0 / NULLIF(views, 0), 2)
  - Duration minutes : ROUND(duration_sec / 60.0, 1)

IMPORTANT SQL RULES:
  - Always use LIMIT 500 or less to avoid memory issues.
  - For time-series queries, always ORDER BY the time column ASC.
  - For ranking queries, always include ORDER BY ... DESC LIMIT N.
  - ads_enabled is stored as TEXT 'True'/'False' — use WHERE ads_enabled = 'True', NOT WHERE ads_enabled = 1.
  - NEVER use SELECT * — always name the columns you need.
  - When computing averages of sentiment_score, use ROUND(AVG(sentiment_score), 3).
  - Always alias computed columns clearly: e.g. AS engagement_rate, AS avg_views, AS total_views.
"""

# Mapping of column names to friendly display labels (used in chart axis labels)
COLUMN_LABELS = {
    "timestamp":       "Timestamp",
    "video_id":        "Video ID",
    "category":        "Category",
    "language":        "Language",
    "region":          "Region",
    "duration_sec":    "Duration (sec)",
    "views":           "Views",
    "likes":           "Likes",
    "comments":        "Comments",
    "shares":          "Shares",
    "sentiment_score": "Sentiment Score",
    "ads_enabled":     "Ads Enabled",
    # derived
    "engagement_rate": "Engagement Rate (%)",
    "like_rate":       "Like Rate (%)",
    "avg_views":       "Avg Views",
    "total_views":     "Total Views",
    "avg_sentiment":   "Avg Sentiment",
    "total_likes":     "Total Likes",
    "video_count":     "Video Count",
    "avg_duration":    "Avg Duration (min)",
}

# Chart type guidance injected into Gemini prompt
CHART_GUIDANCE = """
CHART TYPE SELECTION RULES — choose the best chart_type for the data:

  - "line"    : Time-series data (monthly trends, daily views over time). 
                Use when x-axis is a date/month/year.
  - "bar"     : Comparing discrete categories (views by region, likes by category).
                Use when x-axis is a categorical label and there are 2–12 categories.
  - "pie"     : Parts of a whole (share of total views by category, distribution by language).
                Use ONLY when there are 2–7 slices and they sum to a meaningful total.
  - "scatter" : Correlation between two numeric values (views vs likes, duration vs engagement).
                Use when the user asks about relationship or correlation.
  - "area"    : Cumulative or stacked time-series (total views over time across categories).
                Use when emphasising volume trend over time.

DEFAULT FALLBACK: if unsure, use "bar".
"""

# Sample questions shown in the UI (helps users understand what to ask)
SAMPLE_QUESTIONS = [
    "Show me monthly view trends for 2024 broken down by category",
    "Which region has the highest average engagement rate?",
    "Compare total views by language",
    "Show the relationship between video duration and views",
    "What is the average sentiment score per category?",
    "Which category has the most videos with ads enabled?",
    "Show me top 5 categories by total likes in the US",
    "Compare average views for videos with and without ads",
]