from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data.text import item_tokens, tokens
from .bm25 import FieldedBM25
from .features import MC_EPS, add_rank_features, item_features, query_features
from .location import LocationModel
from .search_logs import QueryLogModel

# Веса подобраны на валидации. Описание весит как заголовок:
# исполнители часто пишут ключевые слова только там.
FIELD_WEIGHTS = {"title": 1.0, "params": 0.5, "desc": 1.0}
LOC_WEIGHT = 3.0
LOC_EPS = 1e-4
CHUNK = 128  # матрица 128 × 189k спокойно влезает в память
# сколько объявлений каждый источник кладёт в пул
SOURCES = {"blend": 300, "lexical_global": 20, "expansion_loc": 50, "dense_loc": 100}
DENSE_WEIGHT = 80.0


@dataclass
class QueryBatch:
    """Всё, что по запросам считается один раз, до прохода по корпусу."""

    tokens: list[list[str]]
    terms: sp.csr_matrix  # слова запроса
    filter_terms: sp.csr_matrix  # слова фильтров поиска
    n_terms: np.ndarray
    neighbors: sp.csr_matrix  # похожие запросы из логов
    mc_dist: np.ndarray  # P(микрокатегория | запрос)
    filter_mc_dist: np.ndarray  # P(микрокатегория | фильтры)
    expansion_terms: sp.csr_matrix  # слова заголовков, выбранных по похожим запросам
    location: np.ndarray
    q_emb: np.ndarray  # эмбеддинги запросов
    mc_dense: np.ndarray  # P(микрокатегория | запрос) по соседям-эмбеддингам
    expansion_dense: sp.csr_matrix


class CandidateGenerator:
    """Отбирает из корпуса ~300 кандидатов на запрос и считает для них признаки."""

    def fit(self, knowledge, items, item_emb: np.ndarray, log_emb: np.ndarray, extra_search_locs=()):
        self.items = items
        self.item_emb, self.log_emb = item_emb, log_emb
        self.item_ids = items["item_id"].to_numpy()
        fields = item_tokens(items)
        self.bm25 = FieldedBM25().fit(fields, FIELD_WEIGHTS)
        self.presence = {f: (self.bm25.field_matrices_[f] > 0).astype(np.float32).tocsr() for f in ("title", "desc")}
        self.location = LocationModel().fit(knowledge, items, extra_search_locs)
        self.item_loc = self.location.index(items["item_location_id"])
        microcats = pd.Index(
            np.unique(np.concatenate([items["item_microcat_id"].unique(), knowledge["item_microcat_id"].unique()]))
        )
        self.item_mc = microcats.get_indexer(items["item_microcat_id"])
        self.logs = QueryLogModel().fit(knowledge, microcats, self.bm25)
        self.item_feats = item_features(items, fields, self.item_mc, self.logs.mc_global_)
        return self

    def pool(self, queries: pd.DataFrame, q_emb: np.ndarray) -> pd.DataFrame:
        """Пул кандидатов с признаками: строка на пару (запрос, объявление)."""
        batch = self._prepare(queries, q_emb)
        frames = []
        for start in range(0, len(queries), CHUNK):
            rows = slice(start, min(start + CHUNK, len(queries)))
            frames.append(self._select(*self._signals(batch, rows), rows))

        pool = pd.concat(frames, ignore_index=True)
        add_rank_features(pool)
        self._add_pair_features(pool, batch)
        for c in self.item_feats:
            pool[c] = self.item_feats[c].to_numpy()[pool["item_idx"]]
        qf = query_features(queries, batch.tokens, self.location, self.logs, batch.neighbors, batch.mc_dist)
        for c in qf:
            pool[c] = qf[c].to_numpy()[pool["q"]]
        return pool

    def _prepare(self, queries: pd.DataFrame, q_emb: np.ndarray) -> QueryBatch:
        text = queries["search_query"].tolist()
        filters = queries["search_infm_params_text"].tolist()
        toks = [tokens(t) for t in text]
        terms = self.bm25.query_matrix(toks)
        neighbors = self.logs.neighbors(text)
        dense_nbrs = self.logs.dense_neighbors(q_emb, self.log_emb)
        return QueryBatch(
            tokens=toks,
            terms=terms,
            filter_terms=self.bm25.query_matrix([tokens(t) for t in filters]),
            n_terms=np.maximum(np.asarray(terms.sum(1)).ravel(), 1),
            neighbors=neighbors,
            mc_dist=self.logs.microcat_dist(neighbors).astype(np.float32),
            filter_mc_dist=self.logs.params_microcat_dist(filters).astype(np.float32),
            expansion_terms=self.logs.expansion(neighbors),
            location=self.location.index(queries["search_location_id"]),
            q_emb=q_emb,
            mc_dense=self.logs.microcat_dist(dense_nbrs).astype(np.float32),
            expansion_dense=self.logs.expansion(dense_nbrs),
        )

    def _signals(self, b: QueryBatch, rows: slice) -> tuple[dict, dict]:
        """Сигналы для пачки запросов — матрицы «запросы × весь корпус»."""
        fm = self.bm25.field_matrices_
        terms = b.terms[rows]
        s = {
            "bm25_title": (terms @ fm["title"]).toarray(),
            "bm25_params": (terms @ fm["params"]).toarray(),
            "bm25_desc": (terms @ fm["desc"]).toarray(),
            "bm25_sparams": (b.filter_terms[rows] @ fm["params"]).toarray(),
            "cov_title": (terms @ self.presence["title"]).toarray() / b.n_terms[rows, None],
            "cov_desc": (terms @ self.presence["desc"]).toarray() / b.n_terms[rows, None],
            "loc_prob": self.location.prob(b.location[rows], self.item_loc),
            "mc_logp": np.log(b.mc_dist[rows][:, self.item_mc] + MC_EPS),
            "pmc_logp": np.log(b.filter_mc_dist[rows][:, self.item_mc] + MC_EPS),
            "expansion": (b.expansion_terms[rows] @ self.bm25.doc_matrix_).toarray(),
        }
        s["bm25"] = sum(FIELD_WEIGHTS[f] * s[f"bm25_{f}"] for f in FIELD_WEIGHTS)
        # Локация — мягкий штраф log P, а не фильтр:
        # жёсткий фильтр терял объявления из соседних городов (0.69 против 0.83).
        loc = LOC_WEIGHT * np.log(s["loc_prob"] + LOC_EPS)
        s["blend"] = s["bm25"] + loc + s["mc_logp"] + s["expansion"]
        # эмбеддинги ловят перефразирования: «аренда помещения на праздник» → «пространство для мероприятий»
        s["dense"] = b.q_emb[rows] @ self.item_emb.T
        s["blend"] = s["blend"] + DENSE_WEIGHT * s["dense"]
        s["mc_logp_dense"] = np.log(b.mc_dense[rows][:, self.item_mc] + MC_EPS)
        s["expansion_dense"] = (b.expansion_dense[rows] @ self.bm25.doc_matrix_).toarray()
        derived = {"expansion_loc": s["expansion"] + loc, "dense_loc": DENSE_WEIGHT * s["dense"] + loc + s["mc_logp"]}
        return s, derived

    def _select(self, signals: dict, derived: dict, rows: slice) -> pd.DataFrame:
        """Объединяем топы источников и снимаем значения сигналов у отобранных объявлений."""
        # BM25 без локации — на случай, когда выбрали объявление в другом городе
        sources = {"blend": signals["blend"], "lexical_global": signals["bm25"], **derived}
        n = rows.stop - rows.start
        picked = np.zeros((n, len(self.item_ids)), dtype=bool)
        for name, k in SOURCES.items():
            top = np.argpartition(-sources[name], k, axis=1)[:, :k]
            picked[np.arange(n)[:, None], top] = True
        qi, ii = np.nonzero(picked)

        feats = {name: m[qi, ii].astype(np.float32) for name, m in signals.items()}
        feats.update({f"src_{name}": m[qi, ii].astype(np.float32) for name, m in derived.items()})
        df = pd.DataFrame(feats)
        df.insert(0, "item_idx", ii.astype(np.int32))
        df.insert(0, "q", (qi + rows.start).astype(np.int32))
        return df

    def _add_pair_features(self, pool: pd.DataFrame, b: QueryBatch) -> None:
        q_loc = b.location[pool["q"]]
        item_idx = pool["item_idx"].to_numpy()
        pool["same_loc"] = (self.item_loc[item_idx] == q_loc).astype(np.int8)
        pool["dist_km"] = self.location.distance_km(
            q_loc,
            self.items["item_latitude"].to_numpy()[item_idx],
            self.items["item_longitude"].to_numpy()[item_idx],
        )
        pool["mc_top1"] = (b.mc_dist.argmax(1)[pool["q"]] == self.item_mc[item_idx]).astype(np.int8)
