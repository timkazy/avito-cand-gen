import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 07. История в общий пул (строка README 7).
# Гипотеза 15: исторические ответы добавляю в пул, ранжирую всех одной
# формулой. Отклонено: что знала история, текст уже нашел.
# Мой результат на 500 парах: 0.908 с историей и 0.908 без.
from collections import Counter

import numpy as np

from experiments.common import (big_sample, get_ctx, pairs_in_corpus,
                                popularity, query_tokens, short_key)
from src.retrieval import haversine


def run():
    ctx = get_ctx()
    pairs = pairs_in_corpus(ctx["train"], ctx["item_ids"])
    big = big_sample(pairs)
    fit = ctx["train"][~short_key(ctx["train"]).isin(set(short_key(big)))]
    history = {}
    for query, group in fit.groupby(fit["search_query"].str.lower().str.strip()):
        counts = Counter(i for i in group["item_id"] if i in set(ctx["item_ids"]))
        if counts:
            history[query] = [iid for iid, _ in counts.most_common()]
    print("текстов со страховкой:", len(history))
    pop = popularity(ctx["train"], ctx["item_ids"], exclude=set(short_key(big)))
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]
    hits, added = 0, 0
    for _, row in big.iterrows():
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
        seen = set(cand)
        for iid in history.get(str(row["search_query"]).lower().strip(), []):
            pos = id_to_pos.get(iid, -1)
            if pos != -1 and pos not in seen:
                cand.append(pos)
                seen.add(pos)
                added += 1
        part = [(scores[p], p) for p in cand]
        text_max = max(s for s, _ in part) or 1.0
        ranked = []
        for score, pos in part:
            geo = 0.0
            if dist is not None and dist[pos] == dist[pos] and dist[pos] <= 30:
                geo = 1.0 - dist[pos] / 30.0
            ranked.append((score / text_max + 0.3 * geo + 0.1 * np.log1p(pop[pos]), pos))
        ranked.sort(reverse=True)
        if target in [p for _, p in ranked[:50]]:
            hits += 1
    print("добавил в пул:", added, "кандидатов")
    print("с историей в пуле:", round(hits / len(big), 4))


if __name__ == "__main__":
    run()
