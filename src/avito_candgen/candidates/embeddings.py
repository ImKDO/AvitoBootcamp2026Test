import subprocess
import sys

import numpy as np
import pandas as pd

from ..paths import cache_path

# Небольшая модель на CPU: корпус кодируется ~10 минут, а результат не зависит от GPU.
MODEL = "intfloat/multilingual-e5-small"


def _encode(texts: list[str]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    model = SentenceTransformer(MODEL, device="cpu")
    model.max_seq_length = 64
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def embed(name: str, texts: list[str]) -> np.ndarray:
    path = cache_path(f"emb_{name}.npy")
    if not path.exists():
        # отдельный процесс: torch и LightGBM в одном процессе на macOS падают из-за OpenMP
        src = cache_path(f"texts_{name}.pkl")
        pd.to_pickle(texts, src)
        subprocess.run([sys.executable, "-m", __name__, str(src), str(path)], check=True)
        src.unlink()
    return np.load(path)


def item_texts(items: pd.DataFrame) -> list[str]:
    desc = items["item_description_raw"].str.slice(0, 300).str.replace(r"\s+", " ", regex=True)
    return ("passage: " + items["item_title_raw"] + ". " + desc).tolist()


def query_texts(queries: pd.Series) -> list[str]:
    return ("query: " + queries.str.strip()).tolist()


if __name__ == "__main__":
    np.save(sys.argv[2], _encode(pd.read_pickle(sys.argv[1])))
