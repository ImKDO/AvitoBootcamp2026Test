import numpy as np
import pandas as pd


class LocationModel:
    """P(локация объявления | локация поиска) по логам.

    83% кликов — в той же локации, где искали. Но есть «региональные» локации поиска
    (область, вся страна) без своих объявлений: оттуда клики расходятся по городам.
    """

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def fit(self, knowledge: pd.DataFrame, items: pd.DataFrame, extra_search_locs=()):
        self.locs_ = pd.Index(
            pd.unique(
                np.concatenate(
                    [
                        items["item_location_id"].to_numpy(),
                        knowledge["item_location_id"].to_numpy(),
                        knowledge["search_location_id"].to_numpy(),
                        np.asarray(list(extra_search_locs), dtype=np.int64),
                    ]
                )
            )
        )
        n = len(self.locs_)
        counts = np.zeros((n, n))
        np.add.at(counts, (self.index(knowledge["search_location_id"]), self.index(knowledge["item_location_id"])), 1.0)
        # сглаживание в сторону «та же локация» — для локаций, которых нет в логах
        counts[np.arange(n), np.arange(n)] += self.alpha
        self.prob_ = (counts / counts.sum(1, keepdims=True)).astype(np.float32)
        self.search_count_ = counts.sum(1) - self.alpha

        # центр локации - медиана координат её объявлений; у региональных объявлений нет,
        # их центр — среднее городов, куда из них кликают
        lat, lon = np.full(n, np.nan), np.full(n, np.nan)
        g = items.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
        idx = self.index(g.index)
        lat[idx], lon[idx] = g["item_latitude"].to_numpy(), g["item_longitude"].to_numpy()
        missing, known = np.isnan(lat), ~np.isnan(lat)
        w = self.prob_[missing][:, known]
        w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
        lat[missing], lon[missing] = w @ lat[known], w @ lon[known]
        self.lat_, self.lon_ = lat, lon
        return self

    def index(self, loc_ids) -> np.ndarray:
        return self.locs_.get_indexer(np.asarray(loc_ids))

    def prob(self, search_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        return self.prob_[search_idx][:, item_idx]

    def distance_km(self, search_idx: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        la1, lo1 = np.radians(self.lat_[search_idx]), np.radians(self.lon_[search_idx])
        la2, lo2 = np.radians(lat), np.radians(lon)
        a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return (2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)
