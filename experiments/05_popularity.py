import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 05. Популярность: грязная, чистая, по сегментам (строка README 5).
# Гипотезы 13-13б: бонус G * log(1 + клики). Плюс на грязных кликах оказался
# смесью сигнала и утечки: на кликнутых целях +0.05, на нулевках -0.022.
# Поэтому из финального ответа популярность убрана.
# Мой результат: грязная (200): 0.88/0.89/0.9; чистая (500): 0.89/0.902/0.908;
# с кликами 0.9353/0.8849; нулевки 0.8739/0.8964.
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

    def make_hit(sample_pop, weight):
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
                value = score / text_max + 0.3 * geo + weight * np.log1p(sample_pop[pos])
                ranked.append((value, pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return hit

    print("грязная, 200 пар:")
    for weight in (0.0, 0.03, 0.1):
        print("  G =", weight, recall_at(small, make_hit(pop_dirty, weight)))
    print("чистая, 500 пар:")
    for weight in (0.0, 0.03, 0.1):
        print("  G =", weight, recall_at(big, make_hit(pop_clean, weight)))

    clicked = [i for i, row in big.iterrows()
               if pop_clean[id_to_pos.get(row["item_id"], -1)] > 0]
    zero = [i for i in big.index if i not in set(clicked)]
    for name, idx in (("с кликами", clicked), ("нулевки", zero)):
        sub = big.loc[idx]
        with_pop = sum(1 for _, r in sub.iterrows() if make_hit(pop_clean, 0.1)(r))
        without = sum(1 for _, r in sub.iterrows() if make_hit(pop_clean, 0.0)(r))
        print(name, len(sub), round(with_pop / len(sub), 4), round(without / len(sub), 4))


if __name__ == "__main__":
    run()
