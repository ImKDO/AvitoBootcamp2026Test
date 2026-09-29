"""Сборка решения: данные → генератор кандидатов → пулы (с кэшем) → разметка для реранкера."""

import numpy as np
import pandas as pd

from .candidates.embeddings import embed, item_texts, query_texts
from .candidates.generator import CandidateGenerator
from .data.load import load_benchmark, load_items
from .data.split import make_split
from .paths import cache_path
from .ranking.reranker import label


class Workspace:
    """Данные и генератор кандидатов, общие для avito-predict и avito-cv."""

    def __init__(self):
        self.split = make_split()
        self.items = load_items()
        self.benchmark = load_benchmark()
        self.item_ids = self.items["item_id"].to_numpy()
        self._generator = None

    @property
    def generator(self) -> CandidateGenerator:
        if self._generator is None:
            # локации бенчмарка заводим в модель, даже если их нет в логах
            locs = np.concatenate([self.split.labelled["search_location_id"], self.benchmark["search_location_id"]])
            log_texts = pd.Series(pd.unique(self.split.knowledge["search_query"]))
            item_emb = embed("items", item_texts(self.items))
            log_emb = embed("logs", query_texts(log_texts))
            self._generator = CandidateGenerator().fit(self.split.knowledge, self.items, item_emb, log_emb, locs)
        return self._generator

    def _pool(self, name: str, queries: pd.DataFrame) -> pd.DataFrame:
        path = cache_path(f"pool_{name}.parquet")
        if not path.exists():
            q_emb = embed(name, query_texts(queries["search_query"]))
            self.generator.pool(queries, q_emb).to_parquet(path)
        return pd.read_parquet(path)

    def labelled_pool(self) -> tuple[pd.DataFrame, np.ndarray]:
        pool = self._pool("labelled", self.split.labelled)
        return pool, label(pool, self.item_ids, self.split.labelled["relevant"].tolist())

    def benchmark_pool(self) -> pd.DataFrame:
        return self._pool("benchmark", self.benchmark)
