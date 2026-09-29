"""Оценка качества: Recall@50, кросс-валидация реранкера, поправка на состав бенчмарка."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .reranker import Reranker


def recall_at_k(predicted: Sequence[Sequence[str]], relevant: Sequence[Sequence[str]], k: int = 50) -> np.ndarray:
    """Доля правильных объявлений, попавших в топ-k, — по каждому запросу."""
    return np.array([len(set(rel).intersection(pred[:k])) / len(set(rel)) for pred, rel in zip(predicted, relevant)])


def cross_val_scores(pool: pd.DataFrame, y: np.ndarray, texts: list[str], n_folds: int = 5) -> np.ndarray:
    """Out-of-fold скоры реранкера; фолды по тексту запроса, чтобы текст не попал и в train, и в test."""
    uniq = sorted(set(texts))
    fold_of = dict(zip(uniq, np.random.default_rng(0).permutation(len(uniq)) % n_folds))
    fold = np.array([fold_of[t] for t in texts])[pool["q"].to_numpy()]
    oof = np.zeros(len(pool))
    for f in range(n_folds):
        model = Reranker(seeds=(0,)).fit(pool[fold != f].reset_index(drop=True), y[fold != f])
        oof[fold == f] = model.predict(pool[fold == f])
    return oof


# Бенчмарк сложнее валидации: больше незнакомых запросов и запросов без фильтров.
# Поэтому recall ещё пересчитываем с долями этих групп как в бенчмарке.
def _strata(queries: pd.DataFrame, seen: np.ndarray) -> np.ndarray:
    return 2 * seen.astype(int) + (queries["search_infm_params_text"] != "").to_numpy().astype(int)


def benchmark_weights(benchmark: pd.DataFrame, log_texts: set[str]) -> np.ndarray:
    seen = benchmark["search_query"].isin(log_texts).to_numpy()
    return np.bincount(_strata(benchmark, seen), minlength=4) / len(benchmark)


def report(name: str, preds, labelled: pd.DataFrame, weights: np.ndarray) -> None:
    seen = labelled["seen"].to_numpy()
    r = recall_at_k(preds, labelled["relevant"].tolist())
    strata = _strata(labelled, seen)
    est = sum(w * r[strata == c].mean() for c, w in enumerate(weights) if w > 0)
    print(
        f"{name:<24} recall@50={r.mean():.4f}  seen={r[seen].mean():.4f}  "
        f"unseen={r[~seen].mean():.4f}  benchmark-weighted={est:.4f}",
        flush=True,
    )
