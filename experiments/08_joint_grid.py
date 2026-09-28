import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 08. Совместная сетка и чистая дуэль (строка README 8).
# Гипотеза 16: веса зависят друг от друга, оптимум ищу сразу
# по радиусу, весу geo и весу популярности. Сетка была на грязных кликах,
# поэтому победителя проверяю честно на чистых.
# Мой результат: сетка (200) — лучшее (50, 0.2, 0.2) 0.93;
# чистая дуэль (500) — старый 0.908 против нового 0.900.
import numpy as np

from experiments.common import (big_sample, get_ctx, pairs_in_corpus,
                                popularity, query_tokens, recall_at,
                                short_key, small_sample)
from src.retrieval import haversine


def run():
    ctx = get_ctx()
    pairs = pairs_in_corpus(ctx["train"], ctx["item_ids"])
    small = small_sample(pairs)
    big = big_sample(pairs)
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]
    pop_dirty = popularity(ctx["train"], ctx["item_ids"])
    pop_clean = popularity(ctx["train"], ctx["item_ids"],
                           exclude=set(short_key(big)))

    scored = []
    for _, row in small.iterrows():
        tokens = query_tokens(row, True)
        loc = row["search_location_id"]
        if loc in centers.index:
            dist = haversine(centers.loc[loc, "lat"], centers.loc[loc, "lon"], lat, lon)
        else:
            dist = None
        scored.append((bm25.get_scores(tokens), dist, loc,
                       id_to_pos.get(row["item_id"], -1)))
    print("сетка 27 вариантов (грязные клики):")
    best = (None, None, None, -1.0)
    for radius in (20, 30, 50):
        pools = []
        for scores, dist, loc, _ in scored:
            if dist is not None:
                pools.append(np.flatnonzero(dist <= radius).tolist())
            else:
                pools.append(loc_to_pos.get(loc, []))
        for geo_w in (0.2, 0.3, 0.5):
            for pop_w in (0.05, 0.1, 0.2):
                hits = 0
                for k, (scores, dist, loc, target) in enumerate(scored):
                    cand = pools[k] or loc_to_pos.get(loc, [])
                    part = [(scores[p], p) for p in cand]
                    text_max = max(s for s, _ in part) or 1.0
                    ranked = []
                    for i, (score, pos) in enumerate(part):
                        geo = 0.0
                        if dist is not None and dist[pos] == dist[pos]:
                            geo = max(0.0, 1.0 - dist[pos] / radius)
                        value = (score / text_max + geo_w * geo
                                 + pop_w * np.log1p(pop_dirty[pos]))
                        ranked.append((value, pos))
                    ranked.sort(reverse=True)
                    if target in [p for _, p in ranked[:50]]:
                        hits += 1
                rec = round(hits / len(scored), 4)
                print("r =", radius, "w =", geo_w, "G =", pop_w, rec)
                if rec > best[3]:
                    best = (radius, geo_w, pop_w, rec)
    print("лучшее:", best)

    def duel(radius, geo_w, pop_w):
        def hit(row):
            scores = bm25.get_scores(query_tokens(row, True))
            loc = row["search_location_id"]
            target = id_to_pos.get(row["item_id"], -1)
            if loc in centers.index:
                dist = haversine(centers.loc[loc, "lat"], centers.loc[loc, "lon"], lat, lon)
                cand = np.flatnonzero(dist <= radius).tolist()
            else:
                dist, cand = None, loc_to_pos.get(loc, [])
            if not cand:
                cand = loc_to_pos.get(loc, [])
            part = [(scores[p], p) for p in cand]
            text_max = max(s for s, _ in part) or 1.0
            ranked = []
            for score, pos in part:
                geo = 0.0
                if dist is not None and dist[pos] == dist[pos]:
                    geo = max(0.0, 1.0 - dist[pos] / radius)
                value = (score / text_max + geo_w * geo
                         + pop_w * np.log1p(pop_clean[pos]))
                ranked.append((value, pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return recall_at(big, hit)

    print("чистая дуэль: старый", duel(30, 0.3, 0.1), "новый", duel(*best[:3]))


if __name__ == "__main__":
    run()
