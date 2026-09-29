"""Точки входа: avito-predict (обучение и answer.csv) и avito-cv (оценка на валидации)."""

import time

import pandas as pd

from .paths import ANSWER_PATH
from .pipeline import Workspace
from .ranking.reranker import Reranker, top_k
from .ranking.validation import benchmark_weights, cross_val_scores, report


def predict():
    """Обучает реранкер на валидации и пишет answer.csv для бенчмарка."""
    t0 = time.time()
    ws = Workspace()
    model = Reranker().fit(*ws.labelled_pool())
    bench_pool = ws.benchmark_pool()
    preds = top_k(bench_pool, model.predict(bench_pool), ws.item_ids, len(ws.benchmark))

    answer = pd.DataFrame({"query_id": ws.benchmark["query_id"], "answer": [" ".join(p) for p in preds]})
    answer.to_csv(ANSWER_PATH, index=False)
    print(f"wrote {ANSWER_PATH}: {len(answer)} queries ({time.time() - t0:.0f}s)")


def cv():
    """Recall@50 на валидации: смесь сигналов и реранкер (5-fold CV)."""
    t0 = time.time()
    ws = Workspace()
    lab = ws.split.labelled
    pool, y = ws.labelled_pool()
    print(f"queries: {len(lab)}  pool: {len(pool) / len(lab):.0f}/query  "
          f"pool recall: {y.sum() / lab['relevant'].str.len().sum():.4f}")

    w = benchmark_weights(ws.benchmark, set(ws.split.knowledge["search_query"]))
    report("blend (no reranker)", top_k(pool, pool["blend"].to_numpy(), ws.item_ids, len(lab)), lab, w)
    oof = cross_val_scores(pool, y, lab["search_query"].tolist())
    report("LightGBM reranker (OOF)", top_k(pool, oof, ws.item_ids, len(lab)), lab, w)
    print(f"done in {time.time() - t0:.0f}s")
