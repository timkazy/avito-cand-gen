import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 01. Сетка текстовых вариантов (строка README 1).
# Гипотезы 1-6: заголовок x3, начало описания, фильтры в запросе,
# стоп-слова, k1/b. Меняю один пункт за раз на 200 парах.
# Мой результат: v2 0.84; без x3 0.845; без описания 0.78;
# без фильтров 0.82; без стоп-слов 0.84; k/b по умолчанию 0.82.
import numpy as np
from rank_bm25 import BM25Okapi

from experiments.common import (get_ctx, pairs_in_corpus, query_tokens,
                                recall_at, small_sample)
from src import text as T


def doc_variant(row, mode):
    title = str(row["item_title_raw"])
    params = str(row["item_infm_params_text"])
    desc = str(row["item_description_raw"])[:600]
    if mode == "t1":
        return title + " " + params + " " + desc
    if mode == "nodesc":
        return title + " " + title + " " + title + " " + params
    return title + " " + title + " " + title + " " + params + " " + desc


def run():
    ctx = get_ctx()
    sample = small_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]

    def check(bm25, query_fn):
        def hit(row):
            scores = bm25.get_scores(query_fn(row))
            cand = loc_to_pos.get(row["search_location_id"], [])
            order = np.argsort(-scores[cand]) if cand else []
            top = [cand[i] for i in order[:50]]
            return id_to_pos.get(row["item_id"], -1) in top
        return recall_at(sample, hit)

    items = ctx["items"]
    full = [T.lemmatize(doc_variant(r, "full")) for _, r in items.iterrows()]
    bm25_full = BM25Okapi(full, k1=0.9, b=0.3)
    print("v2 полный:", check(bm25_full, lambda r: query_tokens(r, True)))

    t1 = [T.lemmatize(doc_variant(r, "t1")) for _, r in items.iterrows()]
    print("без x3:", check(BM25Okapi(t1, k1=0.9, b=0.3), lambda r: query_tokens(r, True)))

    nodesc = [T.lemmatize(doc_variant(r, "nodesc")) for _, r in items.iterrows()]
    print("без описания:", check(BM25Okapi(nodesc, k1=0.9, b=0.3),
                                 lambda r: query_tokens(r, True)))

    print("без фильтров:", check(bm25_full, lambda r: query_tokens(r, False)))
    nostop = [T.get_words(doc_variant(r, "full")) for _, r in items.iterrows()]
    print("без стоп-слов:", check(BM25Okapi(nostop, k1=0.9, b=0.3),
                                  lambda r: T.get_words(str(r["search_query"]) + " "
                                                        + str(r["search_infm_params_text"]))))
    print("k/b по умолчанию:", check(BM25Okapi(full, k1=1.5, b=0.75),
                                     lambda r: query_tokens(r, True)))


if __name__ == "__main__":
    run()
