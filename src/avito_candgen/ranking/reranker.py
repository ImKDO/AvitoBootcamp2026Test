import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = {
    "objective": "binary",  # для Recall@50 порядок внутри топа не важен, хватает «выберут / нет»
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "deterministic": True,  # чтобы повторный запуск давал тот же answer.csv
    "force_row_wise": True,
}
NUM_ROUNDS = 600
NEG_PER_QUERY = 150


def label(pool: pd.DataFrame, item_ids: np.ndarray, relevant: list[list[str]]) -> np.ndarray:
    rel = {(q, r) for q, rels in enumerate(relevant) for r in rels}
    pairs = zip(pool["q"].to_numpy(), item_ids[pool["item_idx"].to_numpy()])
    return np.fromiter(((q, iid) in rel for q, iid in pairs), dtype=np.int8, count=len(pool))


def _downsample(pool: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    # Позитивов ~1 на 300 кандидатов: берём все позитивы, все трудные негативы из топ-100
    # смеси и немного случайных — обучение в разы быстрее без потери качества.
    r = np.random.default_rng(seed).random(len(pool))
    n = pool.groupby("q")["q"].transform("size").to_numpy()
    keep = (y == 1) | (r < NEG_PER_QUERY / n) | (pool["rank_blend"].to_numpy() <= 100)
    # если правильного ответа нет в пуле, запрос ничему не учит
    has_pos = pd.Series(y).groupby(pool["q"].to_numpy()).transform("max").to_numpy() > 0
    return np.nonzero(keep & has_pos)[0]


class Reranker:
    """LightGBM поверх пула кандидатов, скор — среднее по нескольким сидам."""

    def __init__(self, seeds=(0, 1, 2)):
        self.seeds = seeds

    def fit(self, pool: pd.DataFrame, y: np.ndarray):
        self.features_ = [c for c in pool.columns if c not in ("q", "item_idx")]
        self.models_ = []
        for seed in self.seeds:
            idx = _downsample(pool, y, seed)
            ds = lgb.Dataset(pool.iloc[idx][self.features_], y[idx])
            self.models_.append(lgb.train({**PARAMS, "seed": seed}, ds, NUM_ROUNDS))
        return self

    def predict(self, pool: pd.DataFrame) -> np.ndarray:
        x = pool[self.features_]
        return np.mean([m.predict(x, raw_score=True) for m in self.models_], axis=0)


def top_k(pool: pd.DataFrame, scores: np.ndarray, item_ids: np.ndarray, n_queries: int, k: int = 50):
    order = np.lexsort((-scores, pool["q"].to_numpy()))
    q = pool["q"].to_numpy()[order]
    items = item_ids[pool["item_idx"].to_numpy()[order]]
    starts = np.searchsorted(q, np.arange(n_queries))
    ends = np.searchsorted(q, np.arange(n_queries), side="right")
    return [list(items[s : min(e, s + k)]) for s, e in zip(starts, ends)]
