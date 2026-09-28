import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 13. Dense-добавка и вес склейки (строка README 13).
# Bi-encoder e5-small, доученный на парах запрос→товар, сам по себе слаб
# (0.294), но как добавка к формуле даёт +0.015 на всех радиусах.
# Вес 0.5 — середина плато 0.93–0.934.
# Мой результат на 500 парах: r=30: 0.906/0.918/0.918/0.922/0.924;
# r=50: 0.916/0.93/0.932/0.93/0.934 для D = 0/0.2/0.3/0.5/0.8.
# Нужны models/e5-avito и models/*.npy (см. README).
import numpy as np

from experiments.common import (big_sample, get_ctx, pairs_in_corpus,
                                recall_at)
from src import text as T
from src.retrieval import haversine, load_dense


def run():
    ctx = get_ctx()
    big = big_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    model, item_emb, _, _ = load_dense()
    model.max_seq_length = 128
    val_texts = ["query: " + T.fix_layout(str(r["search_query"]))
                 for _, r in big.iterrows()]
    val_emb = model.encode(val_texts, batch_size=64, normalize_embeddings=True)
    cos_all = val_emb @ item_emb.T
    targets = [ctx["id_to_pos"].get(r["item_id"], -1) for _, r in big.iterrows()]
    alone = sum(1 for i in range(len(big))
                if targets[i] in np.argpartition(-cos_all[i], 50)[:50])
    print("dense один:", round(alone / len(big), 4))

    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]
    items = ctx["items"]
    doc_params = [set(T.lemmatize(str(s))) for s in items["item_infm_params_text"]]

    scored = []
    for (_, row), cos in zip(big.iterrows(), cos_all):
        tokens = T.lemmatize(T.fix_layout(str(row["search_query"])))
        loc = row["search_location_id"]
        if loc in centers.index:
            dist = haversine(centers.loc[loc, "lat"], centers.loc[loc, "lon"], lat, lon)
        else:
            dist = None
        scored.append((bm25.get_scores(tokens).astype(np.float32), cos, dist, loc,
                       T.filter_tokens(str(row["search_infm_params_text"])),
                       id_to_pos.get(row["item_id"], -1)))

    def check(radius, dense_w):
        def hit(k):
            scores, cos, dist, loc, filt, target = scored[k]
            if dist is not None:
                cand = np.flatnonzero(dist <= radius).tolist()
            else:
                cand = loc_to_pos.get(loc, [])
            if not cand:
                cand = loc_to_pos.get(loc, [])
            part_t = [scores[p] for p in cand]
            tmax = max(part_t) if max(part_t) > 0 else 1.0
            part_c = [cos[p] for p in cand]
            cmax = max(part_c) if max(part_c) > 0 else 1.0
            ranked = []
            for i, pos in enumerate(cand):
                geo = 0.0
                if dist is not None and dist[pos] == dist[pos]:
                    geo = max(0.0, 1.0 - dist[pos] / radius)
                bonus = 0.0
                if filt and len(filt & doc_params[pos]) / len(filt) >= 0.9:
                    bonus = 0.55
                value = part_t[i] / tmax + 0.3 * geo + bonus + dense_w * part_c[i] / cmax
                ranked.append((value, pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return round(sum(1 for k in range(len(scored)) if hit(k)) / len(scored), 4)

    for radius in (30, 50):
        for dense_w in (0.0, 0.2, 0.3, 0.5, 0.8):
            print("r =", radius, "D =", dense_w, check(radius, dense_w))


if __name__ == "__main__":
    run()
