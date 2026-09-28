import sys as _sys
from pathlib import Path as _Path
if __name__ == "__main__":
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Эксперимент 04. История, категории и штрафы (строка README 4).
# Гипотезы 10-12: покрытие текстов бенчмарка историей; бонус за категорию;
# штраф за чужую категорию. Всё отклонено.
# Мой результат: покрытие 907/2452 текстов, 360 с ответом в корпусе;
# бонус B: 0.88/0.86/0.805; штраф: 0.88/0.88/0.88.
from experiments.common import get_ctx


def run():
    ctx = get_ctx()
    train, queries, items = ctx["train"], ctx["queries"], ctx["items"]
    corpus = set(ctx["item_ids"])
    texts = set(train["search_query"].str.lower().str.strip())
    found = queries["search_query"].str.lower().str.strip().isin(texts)
    print("текстов бенчмарка из train:", int(found.sum()), "из", len(queries))
    with_answer = 0
    for text in queries[found]["search_query"].str.lower().str.strip().unique():
        chosen = set(train[train["search_query"].str.lower().str.strip() == text]["item_id"])
        if chosen & corpus:
            with_answer += 1
    print("из них с ответом в корпусе:", with_answer)
    match = (train["search_category"].astype(str)
             == train["item_category_id"].astype(str)).mean()
    print("совпадение категорий в парах train:", round(match, 4))


if __name__ == "__main__":
    run()
