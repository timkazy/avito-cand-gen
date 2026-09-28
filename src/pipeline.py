# Сборка answer.csv: для каждого запроса считаю пул и топ-50.
# Запуск: python -m src.pipeline
# Нужны: data/*.parquet и models/ (веса e5-avito + эмбеддинги).
import time

import numpy as np
import pandas as pd

from . import text as T
from .data import item_index, load_data
from .retrieval import (build_bm25, city_centers, encode_queries, haversine,
                        load_dense, radius_pool, top_positions)
from .scoring import RADIUS_KM, filter_bonus, fuse, geo_bonus


def main():
    started = time.time()
    train, queries, items = load_data()
    item_ids, id_to_pos, loc_to_pos = item_index(items)
    centers = city_centers(train)
    item_lat = pd.to_numeric(items["item_latitude"], errors="coerce").to_numpy()
    item_lon = pd.to_numeric(items["item_longitude"], errors="coerce").to_numpy()

    print("Токенизирую корпус...")
    tokenized = [T.lemmatize(T.doc_text(row)) for _, row in items.iterrows()]
    bm25 = build_bm25(tokenized)
    doc_params = [set(T.lemmatize(str(s))) for s in items["item_infm_params_text"]]
    print("Индекс готов за", round(time.time() - started), "сек")

    model, item_emb, query_emb, query_pos = load_dense()
    print("Эмбеддинги загружены:", item_emb.shape)

    score_cache = {}
    out_ids, out_answers = [], []
    total = len(queries)
    for n, (_, row) in enumerate(queries.iterrows(), 1):
        # Запрос для BM25: чистка раскладки, без фильтров (они идут бонусом).
        tokens = T.lemmatize(T.fix_layout(row["search_query"]))
        key = tuple(tokens)
        if key not in score_cache:
            score_cache[key] = bm25.get_scores(tokens).astype(np.float32)
        scores = score_cache[key]
        cosines = query_emb[query_pos[row["query_id"]]] @ item_emb.T

        loc = row["search_location_id"]
        if loc in centers.index:
            cand, dist = radius_pool(centers.loc[loc, "lat"], centers.loc[loc, "lon"],
                                     item_lat, item_lon, RADIUS_KM)
        else:
            cand, dist = loc_to_pos.get(loc, []), None
        if not cand:
            cand, dist = loc_to_pos.get(loc, []), None

        pool_text = [scores[p] for p in cand]
        text_max = max(pool_text) if max(pool_text) > 0 else 1.0
        pool_cos = [cosines[p] for p in cand]
        cos_max = max(pool_cos) if max(pool_cos) > 0 else 1.0
        filt = T.filter_tokens(row["search_infm_params_text"])

        ranked = []
        for i, pos in enumerate(cand):
            dist_km = dist[pos] if dist is not None else None
            bonus = filter_bonus(filt, doc_params[pos])
            value = fuse(pool_text[i] / text_max, geo_bonus(dist_km),
                         bonus, pool_cos[i] / cos_max)
            ranked.append((value, pos))
        ranked.sort(reverse=True)
        top = [p for _, p in ranked[:50]]
        if len(top) < 50:
            seen = set(top)
            for pos in top_positions(scores, 50):
                if pos not in seen:
                    top.append(int(pos))
                    if len(top) >= 50:
                        break
        out_ids.append(row["query_id"])
        out_answers.append(" ".join(item_ids[int(p)] for p in top))
        if n % 500 == 0:
            print("Готово", n, "из", total)

    pd.DataFrame({"query_id": out_ids, "answer": out_answers}).to_csv("answer.csv", index=False)
    print("Сохранил answer.csv за", round(time.time() - started), "сек")


if __name__ == "__main__":
    main()
