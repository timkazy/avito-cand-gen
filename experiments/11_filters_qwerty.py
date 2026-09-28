import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 11. Фильтры бонусом и чистка раскладки (строка README 11).
# Гипотезы 18-19: латиница вроде «ktrnhbrf» — это забытая раскладка;
# фильтры точнее считать отдельным бонусом, а не мешать в запрос.
# Числа взял круглые для первой пробы: бонус 0.55 поднимает точное
# совпадение в топ, порог 90% отсекает случайные совпадения.
# Мой результат на 200 парах без популярности: было 0.88, с бонусом 0.89,
# с раскладкой тоже 0.89. Бонус взял в финал, раскладку оставил безвредной.
import numpy as np

from experiments.common import (get_ctx, pairs_in_corpus, recall_at,
                                small_sample)
from src import text as T
from src.retrieval import haversine
from src.scoring import FILTER_BONUS, FILTER_COVER


def run():
    ctx = get_ctx()
    sample = small_sample(pairs_in_corpus(ctx["train"], ctx["item_ids"]))
    items = ctx["items"]
    doc_params = [set(T.lemmatize(str(s))) for s in items["item_infm_params_text"]]
    bm25, centers = ctx["bm25"], ctx["centers"]
    loc_to_pos, id_to_pos = ctx["loc_to_pos"], ctx["id_to_pos"]
    lat, lon = ctx["item_lat"], ctx["item_lon"]

    def check(mode):
        def hit(row):
            raw_q, raw_f = str(row["search_query"]), str(row["search_infm_params_text"])
            if "qw" in mode:
                raw_q = T.fix_layout(raw_q)
            if "nobonus" in mode:
                tokens, use_bonus = T.lemmatize(raw_q + " " + raw_f), False
            else:
                tokens, use_bonus = T.lemmatize(raw_q), True
            scores = bm25.get_scores(tokens)
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
            filt = T.filter_tokens(raw_f) if use_bonus else set()
            ranked = []
            for score, pos in part:
                geo = 0.0
                if dist is not None and dist[pos] == dist[pos] and dist[pos] <= 30:
                    geo = 1.0 - dist[pos] / 30.0
                bonus = 0.0
                if use_bonus and filt and len(filt & doc_params[pos]) / len(filt) >= FILTER_COVER:
                    bonus = FILTER_BONUS
                ranked.append((score / text_max + 0.3 * geo + bonus, pos))
            ranked.sort(reverse=True)
            return target in [p for _, p in ranked[:50]]
        return recall_at(sample, hit)

    print("как было:", check("nobonus"))
    print("без фильтров + бонус:", check("bonus"))
    print("+ раскладка:", check("bonusqw"))


if __name__ == "__main__":
    run()
