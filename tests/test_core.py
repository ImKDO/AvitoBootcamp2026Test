import numpy as np
import pandas as pd

from avito_candgen.candidates.bm25 import FieldedBM25
from avito_candgen.candidates.location import LocationModel
from avito_candgen.data.text import tokens
from avito_candgen.ranking.validation import recall_at_k


def test_recall_at_k():
    pred = [["a", "b", "c"], ["x", "y"]]
    rel = [["a", "z"], ["y"]]
    np.testing.assert_allclose(recall_at_k(pred, rel, k=2), [0.5, 1.0])
    assert recall_at_k(pred, rel, k=1)[1] == 0.0


def test_tokens_normalise_and_stem():
    assert tokens("Ремонт ТЕЛЕВИЗОРОВ и ёлки") == tokens("ремонт телевизор елки")
    assert "и" not in tokens("ремонт и монтаж")


def test_bm25_finds_matching_document():
    docs = {"title": [tokens("ремонт телевизоров"), tokens("баня на дровах")]}
    bm = FieldedBM25().fit(docs, {"title": 1.0})
    s = bm.scores(bm.query_matrix([tokens("баня")]))
    assert s[0, 1] > 0 and s[0, 0] == 0


def test_location_model_regional_search():
    knowledge = pd.DataFrame({"search_location_id": [1, 1, 1, 2], "item_location_id": [10, 10, 11, 2]})
    items = pd.DataFrame(
        {"item_location_id": [10, 11, 2], "item_latitude": [55.0, 56.0, 60.0], "item_longitude": [37.0, 38.0, 30.0]}
    )
    geo = LocationModel(alpha=1.0).fit(knowledge, items)
    p = geo.prob(geo.index([1]), geo.index([10, 11, 2]))[0]
    assert p[0] > p[1] > p[2] == 0  # из региона 1 кликают в основном в город 10
    assert geo.prob(geo.index([2]), geo.index([2]))[0, 0] == 1.0
    assert not np.isnan(geo.lat_[geo.index([1])][0])  # центр региона восстановлен по его городам
