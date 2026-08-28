from __future__ import annotations

import sqlite3

from app.config import DATABASE_PATH, ROOT_DIR


def _read_sql(name: str) -> str:
    return (ROOT_DIR / "database" / name).read_text(encoding="utf-8")


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_read_sql("schema.sql"))
        connection.executescript(_read_sql("seed.sql"))
        connection.commit()
