"""Чтение parquet-файлов задачи."""

import pandas as pd

from ..paths import ITEMS_PATH, QUERIES_PATH, TRAIN_PATH

QUERY_COLS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]
# описания из train не нужны, а весят сотни мегабайт
TRAIN_ITEM_COLS = ["item_id", "item_title_raw", "item_location_id", "item_microcat_id", "item_category_id"]


def load_items() -> pd.DataFrame:
    items = pd.read_parquet(ITEMS_PATH)
    for col in ("item_price", "item_latitude", "item_longitude"):
        items[col] = items[col].astype("float64")  # в parquet это decimal
    for col in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        items[col] = items[col].fillna("")
    return items


def load_benchmark() -> pd.DataFrame:
    queries = pd.read_parquet(QUERIES_PATH)
    queries["search_query"] = queries["search_query"].fillna("")
    queries["search_infm_params_text"] = queries["search_infm_params_text"].fillna("")
    return queries


def load_train() -> pd.DataFrame:
    train = pd.read_parquet(TRAIN_PATH, columns=QUERY_COLS + TRAIN_ITEM_COLS)
    train["search_query"] = train["search_query"].fillna("")
    train["search_infm_params_text"] = train["search_infm_params_text"].fillna("")
    return train
