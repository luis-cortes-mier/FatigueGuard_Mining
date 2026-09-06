"""Small SQLite persistence layer shared by realtime and the dashboard."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

try:
    from .config import DATABASE_PATH
except ImportError:  # Supports direct execution from src/.
    from config import DATABASE_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS current_status (
    session_id TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    probability REAL,
    positive_windows INTEGER NOT NULL,
    last_update TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    equipment_id TEXT NOT NULL,
    timestamp_start TEXT NOT NULL,
    timestamp_end TEXT NOT NULL,
    duration_seconds REAL NOT NULL,
    alert_reason TEXT NOT NULL,
    max_probability REAL,
    mean_probability REAL,
    mean_perclos REAL,
    max_eye_closure_seconds REAL NOT NULL,
    alert_triggered INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_current_status_last_update
ON current_status(last_update DESC);

CREATE INDEX IF NOT EXISTS idx_events_timestamp_start
ON events(timestamp_start DESC);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_database(db_path: Path = DATABASE_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path = DATABASE_PATH) -> Path:
    path = Path(db_path)
    with closing(connect_database(path)) as connection:
        connection.executescript(SCHEMA)
        connection.commit()
    return path


def upsert_current_status(
    session_id: str,
    equipment_id: str,
    status: str,
    probability: float | None,
    positive_windows: int,
    last_update: str | None = None,
    db_path: Path = DATABASE_PATH,
) -> None:
    initialize_database(db_path)
    values = (
        session_id,
        equipment_id,
        status,
        probability,
        int(positive_windows),
        last_update or utc_now_iso(),
    )
    query = """
        INSERT INTO current_status (
            session_id, equipment_id, status, probability, positive_windows, last_update
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            equipment_id = excluded.equipment_id,
            status = excluded.status,
            probability = excluded.probability,
            positive_windows = excluded.positive_windows,
            last_update = excluded.last_update
    """
    with closing(connect_database(db_path)) as connection:
        connection.execute(query, values)
        connection.commit()


def insert_event(event: Mapping, db_path: Path = DATABASE_PATH) -> None:
    initialize_database(db_path)
    columns = [
        "event_id",
        "session_id",
        "equipment_id",
        "timestamp_start",
        "timestamp_end",
        "duration_seconds",
        "alert_reason",
        "max_probability",
        "mean_probability",
        "mean_perclos",
        "max_eye_closure_seconds",
        "alert_triggered",
    ]
    values = [event.get(column) for column in columns]
    values[-1] = int(bool(values[-1]))
    placeholders = ", ".join("?" for _ in columns)
    query = f"INSERT INTO events ({', '.join(columns)}) VALUES ({placeholders})"
    with closing(connect_database(db_path)) as connection:
        connection.execute(query, values)
        connection.commit()


def get_current_status(db_path: Path = DATABASE_PATH) -> dict | None:
    initialize_database(db_path)
    query = "SELECT * FROM current_status ORDER BY last_update DESC LIMIT 1"
    with closing(connect_database(db_path)) as connection:
        row = connection.execute(query).fetchone()
    return dict(row) if row else None


def get_recent_events(limit: int = 100, db_path: Path = DATABASE_PATH) -> list[dict]:
    initialize_database(db_path)
    safe_limit = max(1, int(limit))
    query = "SELECT * FROM events ORDER BY timestamp_start DESC LIMIT ?"
    with closing(connect_database(db_path)) as connection:
        rows = connection.execute(query, (safe_limit,)).fetchall()
    return [dict(row) for row in rows]


def count_events(db_path: Path = DATABASE_PATH) -> int:
    initialize_database(db_path)
    with closing(connect_database(db_path)) as connection:
        value = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return int(value)
