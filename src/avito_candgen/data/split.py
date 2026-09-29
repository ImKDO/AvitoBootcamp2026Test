"""Разбиение train на логи поиска (knowledge) и валидацию на настоящем корпусе (labelled)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..paths import ITEMS_PATH, cache_path
from .load import QUERY_COLS, load_train


@dataclass
class Split:
    knowledge: pd.DataFrame
    labelled: pd.DataFrame


def make_split(seed: int = 0) -> Split:
    """Делим train на «логи» и валидацию.

    Объявления из train почти не пересекаются с корпусом бенчмарка (~5%), и это удобно:
    - knowledge — строки, где объявления нет в корпусе. Только из них берём статистику
      по логам, поэтому ответы валидации туда не утекают;
    - labelled — группы запросов, которые кликнули объявление из корпуса. Это готовая
      разметка: ищем по настоящему корпусу и проверяем, нашлись ли эти объявления.
    """
    path = cache_path("split.pkl")
    if path.exists():
        return pd.read_pickle(path)

    train = load_train()
    corpus_ids = set(pd.read_parquet(ITEMS_PATH, columns=["item_id"])["item_id"])
    in_corpus = train["item_id"].isin(corpus_ids)
    knowledge = train[~in_corpus].reset_index(drop=True)

    groups = (
        train[in_corpus]
        .groupby(QUERY_COLS, sort=False)["item_id"]
        .agg(lambda s: sorted(set(s)))
        .rename("relevant")
        .reset_index()
    )
    # в бенчмарке все тексты запросов разные — оставляем одну случайную группу на текст
    groups = groups.iloc[np.random.default_rng(seed).permutation(len(groups))]
    labelled = groups.drop_duplicates("search_query").reset_index(drop=True)
    labelled["seen"] = labelled["search_query"].isin(set(knowledge["search_query"]))

    split = Split(knowledge, labelled)
    pd.to_pickle(split, path)
    return split
