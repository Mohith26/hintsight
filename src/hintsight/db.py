"""Download the ASSISTments 2009-2010 skill-builder log and load it into DuckDB."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

DATA_URL = (
    "https://huggingface.co/datasets/OloriBern/assistments-dataset/resolve/main/"
    "skill_builder_data_corrected.csv"
)
DATA_SHA256 = "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da"
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "skill_builder_data_corrected.csv"
DB_PATH = DATA_DIR / "assistments09.duckdb"
SQL_PATH = Path(__file__).with_name("sql") / "build.sql"

RAW_COLUMNS = [
    "order_id", "assignment_id", "user_id", "assistment_id", "problem_id", "original",
    "correct", "attempt_count", "ms_first_response", "tutor_mode", "answer_type",
    "sequence_id", "student_class_id", "position", "type", "base_sequence_id", "skill_id",
    "skill_name", "teacher_id", "school_id", "hint_count", "hint_total", "overlap_time",
    "template_id", "first_action", "bottom_hint", "opportunity", "opportunity_original",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if force or not CSV_PATH.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, CSV_PATH)
    digest = sha256(CSV_PATH)
    if digest != DATA_SHA256:
        raise RuntimeError(f"checksum mismatch for {CSV_PATH}: {digest}")
    return CSV_PATH


def load_raw_frame(path: Path = CSV_PATH) -> pd.DataFrame:
    # The file is not valid UTF-8 (answer_text has Windows-1252 bytes); answer_text is not used.
    return pd.read_csv(path, encoding="latin-1", usecols=RAW_COLUMNS, low_memory=False)


def build_tables(con: duckdb.DuckDBPyConnection, raw: pd.DataFrame) -> None:
    con.register("raw_df", raw)
    con.execute("CREATE OR REPLACE TABLE raw AS SELECT * FROM raw_df")
    con.unregister("raw_df")
    con.execute(SQL_PATH.read_text())


def connect(rebuild: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with all analysis tables built."""
    if rebuild or not DB_PATH.exists():
        download()
        if DB_PATH.exists():
            DB_PATH.unlink()
        con = duckdb.connect(str(DB_PATH))
        build_tables(con, load_raw_frame())
        return con
    return duckdb.connect(str(DB_PATH))
