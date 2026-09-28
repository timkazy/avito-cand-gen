# Поиск: BM25-индекс, гео-пулы и dense-косинусы.
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from . import text as T

K1 = 0.9
B = 0.3


def build_bm25(tokenized_docs):
    """BM25 по всему корпусу. k1/b подобрал сеткой на своей валидации."""
    return BM25Okapi(tokenized_docs, k1=K1, b=B)


def city_centers(train):
    """Центр города поиска — медиана координат выбранных из него объявлений."""
    lat = pd.to_numeric(train["item_latitude"], errors="coerce")
    lon = pd.to_numeric(train["item_longitude"], errors="coerce")
    return train.assign(lat=lat, lon=lon).groupby("search_location_id")[["lat", "lon"]].median()


def haversine(clat, clon, lats, lons):
    """Расстояние в км от точки до массивов координат."""
    radius = 6371.0
    a1 = np.radians(clat)
    a2 = np.radians(lats)
    d1 = np.radians(lats - clat)
    d2 = np.radians(lons - clon)
    h = np.sin(d1 / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(d2 / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(h))


def radius_pool(center_lat, center_lon, item_lat, item_lon, radius_km):
    """Позиции объявлений внутри радиуса от центра."""
    dist = haversine(center_lat, center_lon, item_lat, item_lon)
    return np.flatnonzero(dist <= radius_km).tolist(), dist


def top_positions(scores, k):
    """k лучших позиций без полной сортировки."""
    if len(scores) <= k:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def load_dense(models_dir="models/e5-avito", emb_dir="models"):
    """Веса bi-encoder и готовые эмбеддинги корпуса и запросов бенчмарка."""
    from pathlib import Path

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(Path(models_dir)))
    model.max_seq_length = 128
    emb = Path(emb_dir)
    item_emb = np.load(emb / "item_emb.npy")
    query_emb = np.load(emb / "query_emb.npy")
    query_ids = np.load(emb / "query_ids.npy", allow_pickle=True)
    return model, item_emb, query_emb, {qid: i for i, qid in enumerate(query_ids)}


def encode_queries(model, queries):
    """Кодирую запросы так же, как в обучении: префикс query, без фильтров."""
    texts = ["query: " + str(q) for q in queries["search_query"]]
    return model.encode(texts, batch_size=64, normalize_embeddings=True)
