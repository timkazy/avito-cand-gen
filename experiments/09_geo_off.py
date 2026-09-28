import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 09. Онлайн-разведка и geo-off для дальних (строка README 9).
# Гипотезы 17-17б: части услуг город не нужен; дальним радиус вредит.
# Отклонено: масштаб крошечный, а правило по маркерам бьет по ближним.
# Мой результат: маркеров 28/2452; дальше 30 км 4.3%; ближние 0.9228/0.4593;
# дальние 0.0/0.75 (их всего 8).
import numpy as np
import pandas as pd

from experiments.common import (big_sample, get_ctx, pairs_in_corpus,
                                popularity, query_tokens, short_key)
from src.retrieval import haversine, top_positions


def run():
    ctx = get_ctx()
    pairs = pairs_in_corpus(ctx["train"], ctx["item_ids"])
    big = big_sample(pairs)
    markers = ["онлайн", "удален", "удалён", "дистанцион", "интернет"]
    flagged = [any(m in str(q).lower() for m in markers)
               for q in ctx["queries"]["search_query"]]
    print("запросов с маркерами:", sum(flagged), "из", len(ctx["queries"]))
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]
    pop = popularity(ctx["train"], ctx["item_ids"], exclude=set(short_key(big)))

    def champ(row):
        scores = bm25.get_scores(query_tokens(row, True))
        loc = row["search_location_id"]
        target = id_to_pos.get(row["item_id"], -1)
        if loc in centers.index:
            dist = haversine(centers.loc[loc, "lat"], centers.loc[loc, "lon"], lat, lon)
            cand = np.flatnonzero(dist <= 30).tolist()
        else:
            dist, cand = None, loc_to_pos.get(loc, [])
        if not cand:
            cand = loc_to_pos.get(loc, [])
        part = [(scores[p], p) for p in cand]
        text_max = max(s for s, _ in part) or 1.0
        ranked = []
        for score, pos in part:
            geo = 0.0
            if dist is not None and dist[pos] == dist[pos] and dist[pos] <= 30:
                geo = 1.0 - dist[pos] / 30.0
            ranked.append((score / text_max + 0.3 * geo + 0.1 * np.log1p(pop[pos]), pos))
        ranked.sort(reverse=True)
        return target in [p for _, p in ranked[:50]]

    def no_geo(row):
        scores = bm25.get_scores(query_tokens(row, True))
        target = id_to_pos.get(row["item_id"], -1)
        pool = [int(p) for p in top_positions(scores, 1000)]
        part = [(scores[p], p) for p in pool]
        text_max = max(s for s, _ in part) or 1.0
        ranked = [(s / text_max + 0.1 * np.log1p(pop[p]), p) for s, p in part]
        ranked.sort(reverse=True)
        return target in [p for _, p in ranked[:50]]

    plat = pd.to_numeric(big["item_latitude"], errors="coerce").to_numpy()
    plon = pd.to_numeric(big["item_longitude"], errors="coerce").to_numpy()
    clat = big["search_location_id"].map(centers["lat"]).to_numpy()
    clon = big["search_location_id"].map(centers["lon"]).to_numpy()
    far = pd.Series(haversine(clat, clon, plat, plon) > 100).fillna(False).to_numpy()
    print("дальних:", int(far.sum()), "ближних:", int((~far).sum()))
    for name, mask in (("ближние", ~far), ("дальние", far)):
        sub = big[mask]
        if len(sub) == 0:
            continue
        hits_c = sum(1 for _, r in sub.iterrows() if champ(r))
        hits_n = sum(1 for _, r in sub.iterrows() if no_geo(r))
        print(name, len(sub), round(hits_c / len(sub), 4), round(hits_n / len(sub), 4))


if __name__ == "__main__":
    run()
