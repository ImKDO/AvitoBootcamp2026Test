from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def _identity(x):
    return x


class FieldedBM25:
    """BM25 по нескольким полям с общим словарём.

    Скоры — запросы x словарь @ словарь x документы,
    так что сразу против всего корпуса.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, fields: dict[str, Sequence[list[str]]], weights: dict[str, float]):
        self._vec = CountVectorizer(analyzer=_identity, dtype=np.float32)
        self._vec.fit([toks for docs in fields.values() for toks in docs])
        n_docs = len(next(iter(fields.values())))
        tfs = {name: self._vec.transform(docs).tocsr() for name, docs in fields.items()}
        # IDF общий для полей: слово считается один раз на документ
        any_field = sum((tf > 0).astype(np.float32) for tf in tfs.values())
        df = (any_field > 0).sum(0).A1.astype(np.float64)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        self.field_matrices_ = {}
        for name, tf in tfs.items():
            dl = tf.sum(1).A1
            norm = self.k1 * (1 - self.b + self.b * dl / max(dl.mean(), 1e-9))
            tf = tf.tocoo()
            data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
            self.field_matrices_[name] = sp.csr_matrix((data.astype(np.float32), (tf.col, tf.row)), shape=tf.shape[::-1])
        self.doc_matrix_ = sum(weights[n] * m for n, m in self.field_matrices_.items()).tocsr()
        return self

    def query_matrix(self, queries: Sequence[list[str]]) -> sp.csr_matrix:
        q = self._vec.transform(queries).tocsr()
        q.data[:] = 1.0  # повтор слова в запросе не удваивает его вес
        return q

    def scores(self, q: sp.csr_matrix) -> np.ndarray:
        return (q @ self.doc_matrix_).toarray()
