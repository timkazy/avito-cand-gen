# Общее для экспериментов: данные, выборки с фиксированными сидами,
# честный подсчёт кликов, замер recall@50.
# Сам файл ничего не считает — только функции.
import numpy as np
import pandas as pd

from src import text as T
from src.data import item_index, load_data
from src.retrieval import build_bm25, city_centers, haversine

SMALL_N = 200
SMALL_SEED = 42
BIG_N = 500
BIG_SEED = 123


def get_ctx():
    """Гружу всё один раз: таблицы, индекс, центры городов, координаты."""
    train, queries, items = load_data()
    item_ids, id_to_pos, loc_to_pos = item_index(items)
    tokenized = [T.lemmatize(T.doc_text(row)) for _, row in items.iterrows()]
    ctx = {
        "train": train, "queries": queries, "items": items,
        "item_ids": item_ids, "id_to_pos": id_to_pos, "loc_to_pos": loc_to_pos,
        "bm25": build_bm25(tokenized),
        "centers": city_centers(train),
        "item_lat": pd.to_numeric(items["item_latitude"], errors="coerce").to_numpy(),
        "item_lon": pd.to_numeric(items["item_longitude"], errors="coerce").to_numpy(),
    }
    return ctx


def pairs_in_corpus(train, item_ids):
    """Только пары, чей ответ лежит в корпусе бенчмарка."""
    return train[train["item_id"].isin(set(item_ids))]


def small_sample(pairs):
    """200 пар для быстрых проверок. Сид зафиксирован."""
    return pairs.sample(SMALL_N, random_state=SMALL_SEED)


def big_sample(pairs):
    """500 пар для стабильных проверок. Сид зафиксирован."""
    return pairs.sample(BIG_N, random_state=BIG_SEED)


def short_key(frame):
    """Короткий ключ группы: текст запроса + город. По нему выкидываю
    проверяемые группы из подсчёта кликов."""
    return (frame["search_query"].str.lower().str.strip()
            + "||" + frame["search_location_id"].astype(str))


def popularity(train, item_ids, exclude=None):
    """Число выборов объявления. Если exclude задан — считаю без этих групп."""
    if exclude is not None:
        train = train[~short_key(train).isin(exclude)]
    counts = train["item_id"].value_counts().to_dict()
    return [counts.get(iid, 0) for iid in item_ids]


def recall_at(sample, hit_fn):
    """Доля пар, где цель попала в топ-50."""
    hits = sum(1 for _, row in sample.iterrows() if hit_fn(row))
    return round(hits / len(sample), 4)


def query_tokens(row, with_filters=True):
    """Токены запроса: текст + фильтры или чистый текст."""
    if with_filters:
        raw = str(row["search_query"]) + " " + str(row["search_infm_params_text"])
    else:
        raw = str(row["search_query"])
    return T.lemmatize(raw)
