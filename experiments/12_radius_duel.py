import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 12. Дуэль радиусов на бенчмарк-сегменте (строка README 12).
# Сегмент как у бенчмарка: цель без кликов + текст вне подсчёта кликов.
# Сравниваю радиус 30 против 50 без популярности вообще.
# Мой результат: сегмент вышел крошечным (35 из 500), ничья 0.7714/0.7714 —
# на такой выборке это шум, радиус тогда не менял. Позже проверка с dense
# на полных 500 показала преимущество широкого geo, в финале радиус 50.
import numpy as np

from experiments.common import (big_sample, get_ctx, pairs_in_corpus,
                                short_key)
from src.retrieval import haversine
from src import text as T


def run():
    ctx = get_ctx()
    big = big_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    fit_keys = set(short_key(ctx["train"])) - set(short_key(big))
    fit_texts = set(ctx["train"][short_key(ctx["train"]).isin(fit_keys)]["search_query"]
                    .str.lower().str.strip())
    seg = [row for _, row in big.iterrows()
           if str(row["search_query"]).lower().strip() not in fit_texts]
    print("пар в сегменте:", len(seg), "из", len(big))
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]

    def check(radius):
        hits = 0
        for row in seg:
            tokens = T.lemmatize(str(row["search_query"]) + " "
                                 + str(row["search_infm_params_text"]))
            scores = bm25.get_scores(tokens)
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
                ranked.append((score / text_max + 0.3 * geo, pos))
            ranked.sort(reverse=True)
            if target in [p for _, p in ranked[:50]]:
                hits += 1
        return round(hits / len(seg), 4) if seg else 0.0

    print("радиус 30:", check(30))
    print("радиус 50:", check(50))


if __name__ == "__main__":
    run()
