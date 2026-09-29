"""Пути к данным, кэшу и answer.csv."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

TRAIN_PATH = DATA_DIR / "train.parquet"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
ANSWER_PATH = ROOT / "answer.csv"


def cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / name
