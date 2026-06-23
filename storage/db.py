"""SQLite persistence layer for the SPY option chain snapshot."""
import json
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS option_chain (
    ticker          TEXT NOT NULL,
    underlying      TEXT NOT NULL,
    strike          REAL NOT NULL,
    expiration      TEXT NOT NULL,
    option_type     TEXT NOT NULL,
    close_price     REAL NOT NULL,
    fetched_at      TEXT NOT NULL,
    PRIMARY KEY (ticker)
)
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE)
        conn.commit()


def insert_chain(db_path: Path, records: list[dict]) -> None:
    """Insert or replace snapshot records."""
    fetched_at = datetime.utcnow().isoformat()
    rows = [
        (
            r["ticker"],
            r["underlying"],
            r["strike"],
            r["expiration"],
            r["option_type"],
            r["close_price"],
            fetched_at,
        )
        for r in records
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO option_chain "
            "(ticker, underlying, strike, expiration, option_type, close_price, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def load_chain(db_path: Path) -> pd.DataFrame:
    """Load the full snapshot into a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM option_chain ORDER BY expiration, option_type, strike",
            conn,
        )
    return df


def save_calibration_cache(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
