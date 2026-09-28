import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 06. Char-канал от опечаток (строка README 6).
# Гипотеза 14: TF-IDF по символьным 3-5-граммам заголовков, беру лучшее
# из двух каналов. Отклонено: леммы уже закрывают окончания.
# Мой результат на 200 парах: без char 0.9, с char 0.895.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from experiments.common import (get_ctx, pairs_in_corpus, query_tokens,
                                recall_at, small_sample)
from src.retrieval import haversine


def run():
    ctx = get_ctx()
    sample = small_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    titles = [str(t).lower() for t in ctx["items"]["item_title_raw"]]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=2, sublinear_tf=True, dtype=np.float32)
    matrix = vec.fit_transform(titles)
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]
    pop = ctx["train"]["item_id"].value_counts().to_dict()
    pop_full = [pop.get(iid, 0) for iid in ctx["item_ids"]]

    def check(use_char):
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
            char = None
            if use_char:
                char = (vec.transform([str(row["search_query"]).lower()])
                        @ matrix.T).toarray()[0]
            ranked = []
            for score, pos in part:
                geo = 0.0
                if dist is not None and dist[pos] == dist[pos] and dist[pos] <= 30:
                    geo = 1.0 - dist[pos] / 30.0
                value = score / text_max
                if use_char:
                    value = max(value, char[pos])
                ranked.append((value + 0.3 * geo + 0.1 * np.log1p(pop_full[pos]), pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return recall_at(sample, hit)

    print("без char:", check(False))
    print("с char:", check(True))


if __name__ == "__main__":
    run()
