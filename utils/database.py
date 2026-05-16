import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

DB_PATH = Path("pipeline.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ran_at            TEXT NOT NULL,
    niche             TEXT NOT NULL,
    domain            TEXT,
    stage_reached     TEXT DEFAULT 'started',
    channels_found    INTEGER,
    videos_collected  INTEGER,
    trend_topic       TEXT,
    trend_hook        TEXT,
    trend_emotion     TEXT,
    video_prompt      TEXT,
    youtube_title     TEXT,
    youtube_tags      TEXT,
    thumbnail_concept TEXT,
    video_provider    TEXT,
    video_file        TEXT,
    video_duration    REAL,
    youtube_video_id  TEXT,
    youtube_url       TEXT,
    status            TEXT DEFAULT 'pending',
    error_message     TEXT,
    duration_seconds  REAL
);

CREATE TABLE IF NOT EXISTS published_topics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT NOT NULL,
    published_at TEXT NOT NULL,
    youtube_url  TEXT
);

CREATE TABLE IF NOT EXISTS channel_cache (
    channel_id       TEXT PRIMARY KEY,
    channel_name     TEXT,
    subscriber_count INTEGER,
    niche            TEXT,
    cached_at        TEXT
);

CREATE TABLE IF NOT EXISTS stage1_cache (
    niche      TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    cached_at  TEXT NOT NULL
);
"""


def init_db(path: Path | None = None) -> None:
    db = path or DB_PATH
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn(path: Path | None = None) -> Generator[sqlite3.Connection, None, None]:
    db = path or DB_PATH
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_run(ran_at: str, niche: str, domain: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (ran_at, niche, domain) VALUES (?, ?, ?)",
            (ran_at, niche, domain),
        )
        return cur.lastrowid  # type: ignore[return-value]


def update_run(run_id: int, **kwargs: Any) -> None:
    if not kwargs:
        return
    cols = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values()) + [run_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE runs SET {cols} WHERE id = ?", vals)


def get_recent_topics(days: int) -> list[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT topic FROM published_topics "
            "WHERE published_at >= datetime('now', ? || ' days') "
            "ORDER BY published_at DESC",
            (f"-{days}",),
        ).fetchall()
    return [r["topic"] for r in rows]


def log_published_topic(topic: str, published_at: str, youtube_url: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO published_topics (topic, published_at, youtube_url) VALUES (?, ?, ?)",
            (topic, published_at, youtube_url),
        )


def get_cached_channels(niche: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM channel_cache "
            "WHERE niche = ? AND cached_at >= datetime('now', '-1 day')",
            (niche,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stage1_cache(niche: str) -> dict | None:
    """Return cached Stage 1 output if fetched within the last 20 hours, else None."""
    import json
    with get_conn() as conn:
        row = conn.execute(
            "SELECT payload FROM stage1_cache "
            "WHERE niche = ? AND cached_at >= datetime('now', '-20 hours')",
            (niche,),
        ).fetchone()
    return json.loads(row["payload"]) if row else None


def save_stage1_cache(niche: str, payload: dict) -> None:
    import json
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO stage1_cache (niche, payload, cached_at) VALUES (?, ?, datetime('now'))",
            (niche, json.dumps(payload)),
        )


def upsert_channel_cache(channels: list[dict], niche: str) -> None:
    with get_conn() as conn:
        for ch in channels:
            conn.execute(
                "INSERT OR REPLACE INTO channel_cache "
                "(channel_id, channel_name, subscriber_count, niche, cached_at) "
                "VALUES (?, ?, ?, ?, datetime('now'))",
                (ch["channel_id"], ch["channel_name"], ch["subscriber_count"], niche),
            )
