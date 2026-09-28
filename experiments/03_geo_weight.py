import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 03. Вес близости внутри пула (строка README 3).
# Гипотеза 9: балл = текст/max + w * (1 - км/радиус).
# Мой результат на 200 парах: w=0 → 0.86; w=0.3 → 0.88; w=1.0 → 0.79.
import numpy as np

from experiments.common import (get_ctx, pairs_in_corpus, query_tokens,
                                recall_at, small_sample)
from src.retrieval import haversine


def run():
    ctx = get_ctx()
    sample = small_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]

    def check(weight):
        def hit(row):
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
                ranked.append((score / text_max + weight * geo, pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return recall_at(sample, hit)

    for weight in (0.0, 0.3, 1.0):
        print("w =", weight, check(weight))


if __name__ == "__main__":
    run()
