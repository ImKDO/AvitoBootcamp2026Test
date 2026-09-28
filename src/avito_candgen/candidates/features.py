import numpy as np
import pandas as pd
import scipy.sparse as sp

from .location import LocationModel
from .search_logs import QueryLogModel

MC_EPS = 1e-4
RANKED_SIGNALS = ["blend", "bm25", "expansion", "mc_logp", "loc_prob"]


def item_features(items: pd.DataFrame, fields: dict, item_mc: np.ndarray, mc_global: np.ndarray) -> pd.DataFrame:
    """Свойства объявления: из десятка одинаковых чаще выбирают те, где есть отзывы, рейтинг, описание."""
    loc_density = items.groupby("item_location_id")["item_id"].transform("size")
    price = items["item_price"].where(items["item_price"] > 0)  # -1 — цена не указана
    return pd.DataFrame(
        {
            "item_rating": items["item_rating"],
            "item_reviews_log": np.log1p(items["item_rating_reviews_count"].fillna(0)),
            "item_price_log": np.log1p(price),
            "item_phone_hidden": items["item_is_phone_hidden"].astype(np.int8),
            "item_msg_forbidden": items["item_is_message_forbidden"].astype(np.int8),
            "item_title_len": np.array([len(t) for t in fields["title"]]),
            "item_desc_len": np.log1p(np.array([len(t) for t in fields["desc"]])),
            "item_is_services": (items["item_category_id"] == 114).astype(np.int8),
            "item_loc_density": np.log1p(loc_density.to_numpy()),
            "item_mc_prior": np.log(mc_global[item_mc] + MC_EPS),
        }
    )


def query_features(
    queries: pd.DataFrame,
    query_tokens: list[list[str]],
    location: LocationModel,
    logs: QueryLogModel,
    neighbors: sp.csr_matrix,
    mc_dist: np.ndarray,
) -> pd.DataFrame:
    """Насколько мы уверены в запросе: незнакомым и размытым запросам приоры из логов помогают меньше."""
    q_loc = location.index(queries["search_location_id"])
    return pd.DataFrame(
        {
            "q_ntok": [len(t) for t in query_tokens],
            "q_has_params": (queries["search_infm_params_text"] != "").astype(np.int8).to_numpy(),
            "q_log_sim": logs.best_similarity(queries["search_query"].tolist()),
            "q_mc_entropy": -(mc_dist * np.log(mc_dist + 1e-12)).sum(1),
            "q_mc_max": mc_dist.max(1),
            "q_loc_self": location.prob_[q_loc, q_loc],
            "q_loc_searches": np.log1p(location.search_count_[q_loc]),
            "q_nbrs": np.diff(neighbors.indptr),
        }
    )


def add_rank_features(pool: pd.DataFrame) -> None:
    # скоры разных запросов несравнимы, а место внутри своего пула — сравнимо
    grp = pool.groupby("q", sort=False)
    for c in RANKED_SIGNALS:
        pool[f"rank_{c}"] = grp[c].rank(ascending=False, method="min").astype(np.float32)
