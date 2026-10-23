"""SQLite history of release experiments."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class ExperimentStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    git_sha TEXT,
                    gate_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def save(self, experiment_id: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, created_at, name, config_hash, git_sha, gate_status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    payload["created_at"],
                    payload["name"],
                    payload["config_hash"],
                    payload.get("environment", {}).get("git_sha"),
                    payload["decision"]["status"],
                    json.dumps(payload, sort_keys=True),
                ),
            )

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        loaded = json.loads(row["payload_json"])
        return loaded if isinstance(loaded, dict) else None

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT experiment_id, created_at, name, gate_status, config_hash, git_sha
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
