from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data.text import tokens
from .bm25 import FieldedBM25


def _row_normalize(m: sp.csr_matrix) -> sp.csr_matrix:
    s = np.asarray(m.sum(1)).ravel()
    s[s == 0] = 1.0
    return (sp.diags(1.0 / s) @ m).tocsr()


def _counts(rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> sp.csr_matrix:
    ok = (rows >= 0) & (cols >= 0)
    return sp.csr_matrix((np.ones(ok.sum(), np.float32), (rows[ok], cols[ok])), shape=shape)


class QueryLogModel:
    """Что пользователи выбирали по похожим запросам.

    Пользователь пишет «автоподбор», а исполнитель — «осмотр авто перед покупкой».
    Логи связывают эти формулировки: находим похожие запросы и смотрим, в каких
    микрокатегориях и с какими заголовками были выбранные объявления.
    """

    def __init__(self, n_neighbors: int = 20, min_rel_sim: float = 0.35):
        self.n_neighbors = n_neighbors
        self.min_rel_sim = min_rel_sim

    def fit(self, knowledge: pd.DataFrame, microcats: pd.Index, item_bm25: FieldedBM25):
        self.texts_ = pd.Index(pd.unique(knowledge["search_query"]))
        t_idx = self.texts_.get_indexer(knowledge["search_query"])
        self.text_bm25_ = FieldedBM25(k1=1.2, b=0.5).fit({"q": [tokens(t) for t in self.texts_]}, {"q": 1.0})
        self.text_freq_ = np.bincount(t_idx, minlength=len(self.texts_)).astype(np.float32)

        mc = microcats.get_indexer(knowledge["item_microcat_id"])
        self.text_mc_ = _row_normalize(_counts(t_idx, mc, (len(self.texts_), len(microcats))))

        # слова заголовков выбранных объявлений — в словаре BM25 корпуса
        title_terms = item_bm25.query_matrix([tokens(x) for x in knowledge["item_title_raw"]])
        per_text = _counts(t_idx, np.arange(len(knowledge)), (len(self.texts_), len(knowledge)))
        self.text_terms_ = _row_normalize(per_text @ title_terms)

        params = knowledge["search_infm_params_text"]
        self.params_ = pd.Index(pd.unique(params[params != ""]))
        p_idx = self.params_.get_indexer(params)
        self.params_mc_ = _row_normalize(_counts(p_idx, mc, (len(self.params_), len(microcats)))).toarray()
        self.mc_global_ = np.bincount(mc[mc >= 0], minlength=len(microcats)).astype(np.float32)
        self.mc_global_ /= self.mc_global_.sum()
        return self

    def neighbors(self, queries: Sequence[str]) -> sp.csr_matrix:
        """Веса похожих запросов из логов: строка на запрос, сумма по строке = 1."""
        sims = self.text_bm25_.scores(self.text_bm25_.query_matrix([tokens(t) for t in queries]))
        n = self.n_neighbors
        top = np.argpartition(-sims, n, axis=1)[:, :n]
        vals = np.take_along_axis(sims, top, 1)
        best = vals.max(1, keepdims=True)
        # слабых соседей отбрасываем, среди остальных предпочитаем близкие и частые запросы;
        # точное совпадение текста само оказывается самым близким
        w = np.where((vals > 0) & (vals >= self.min_rel_sim * best), vals / np.maximum(best, 1e-9), 0.0)
        w = w**2 * np.log1p(self.text_freq_[top])
        rows = np.repeat(np.arange(len(queries)), n)
        m = sp.csr_matrix((w.ravel(), (rows, top.ravel())), shape=(len(queries), len(self.texts_)))
        m.eliminate_zeros()
        return _row_normalize(m)

    def best_similarity(self, queries: Sequence[str]) -> np.ndarray:
        return self.text_bm25_.scores(self.text_bm25_.query_matrix([tokens(t) for t in queries])).max(1)

    def microcat_dist(self, nbrs: sp.csr_matrix, smoothing: float = 0.05) -> np.ndarray:
        d = (nbrs @ self.text_mc_).toarray()
        has = d.sum(1, keepdims=True) > 0
        # немного глобального распределения, чтобы редкие микрокатегории не обнулялись
        return np.where(has, (1 - smoothing) * d + smoothing * self.mc_global_, self.mc_global_)

    def params_microcat_dist(self, params: Sequence[str]) -> np.ndarray:
        idx = self.params_.get_indexer(pd.Index(list(params)))
        out = np.tile(self.mc_global_, (len(idx), 1))
        hit = idx >= 0
        out[hit] = 0.9 * self.params_mc_[idx[hit]] + 0.1 * self.mc_global_
        return out

    def expansion(self, nbrs: sp.csr_matrix, top_terms: int = 30) -> sp.csr_matrix:
        """Расширение запроса: самые весомые слова заголовков, выбранных соседями."""
        e = (nbrs @ self.text_terms_).tocsr()
        rows, cols, vals = [], [], []
        for i in range(e.shape[0]):
            d, c = e.data[e.indptr[i] : e.indptr[i + 1]], e.indices[e.indptr[i] : e.indptr[i + 1]]
            if len(d) > top_terms:
                keep = np.argpartition(-d, top_terms)[:top_terms]
                d, c = d[keep], c[keep]
            rows.append(np.full(len(d), i)), cols.append(c), vals.append(d)
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=e.shape, dtype=np.float32
        )
