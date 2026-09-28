import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 02. Мягкое geo: радиус вместо жесткого города (строка README 2).
# Гипотезы 7-8: пул по радиусу от центра найдет соседей; смесь 45+5.
# Мой результат на 200 парах: радиус 30 — пул 0.955/recall 0.86;
# радиус 100 — 0.97/0.84; смесь 45+5 — 0.83.
import numpy as np

from experiments.common import (get_ctx, pairs_in_corpus, query_tokens,
                                recall_at, small_sample)
from src.retrieval import haversine, top_positions


def run():
    ctx = get_ctx()
    sample = small_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]

    def check(radius):
        pool_hits, hits = 0, 0
        for _, row in sample.iterrows():
            scores = bm25.get_scores(query_tokens(row, True))
            loc = row["search_location_id"]
            target = id_to_pos.get(row["item_id"], -1)
            if loc in centers.index:
                dist = haversine(centers.loc[loc, "lat"], centers.loc[loc, "lon"], lat, lon)
                cand = np.flatnonzero(dist <= radius).tolist()
            else:
                cand = loc_to_pos.get(loc, [])
            if not cand:
                cand = loc_to_pos.get(loc, [])
            ordered = sorted(cand, key=lambda p: -scores[p])
            if target in ordered[:1000]:
                pool_hits += 1
            top = ordered[:50]
            if len(top) < 50:
                seen = set(cand)
                for pos in top_positions(scores, 50):
                    if pos not in seen:
                        top.append(int(pos))
                        if len(top) >= 50:
                            break
            if target in top:
                hits += 1
        n = len(sample)
        return round(pool_hits / n, 4), round(hits / n, 4)

    print("радиус 30:", check(30))
    print("радиус 100:", check(100))


if __name__ == "__main__":
    run()
