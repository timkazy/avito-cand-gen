import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 10. Проверка на новых текстах (строка README 10).
# Беру 500 пар и выкидываю их тексты из подсчёта кликов целиком —
# это похоже на бенчмарк, где запросы новые. Сравниваю лучший вариант,
# вариант без популярности и широкое geo.
# Мой результат: лучший 0.906, без популярности 0.892, широкое geo 0.926.
import numpy as np

from experiments.common import (get_ctx, pairs_in_corpus, query_tokens,
                                recall_at)
from src.retrieval import haversine


def run():
    ctx = get_ctx()
    pairs = pairs_in_corpus(ctx["train"], ctx["item_ids"])
    ood = pairs.sample(500, random_state=456)
    ood_texts = set(ood["search_query"].str.lower().str.strip())
    fit = ctx["train"][~ctx["train"]["search_query"].str.lower().str.strip().isin(ood_texts)]
    counts = fit["item_id"].value_counts().to_dict()
    pop = [counts.get(iid, 0) for iid in ctx["item_ids"]]
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]

    def check(radius, geo_w, pop_w):
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
                value = score / text_max + geo_w * geo + pop_w * np.log1p(pop[pos])
                ranked.append((value, pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return recall_at(ood, hit)

    print("лучший (30, 0.3, 0.1):", check(30, 0.3, 0.1))
    print("без популярности:", check(30, 0.3, 0.0))
    print("широкое geo (50, 0.2, 0.1):", check(50, 0.2, 0.1))


if __name__ == "__main__":
    run()
