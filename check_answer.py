# Проверка формата answer.csv по требованиям задания.
# Запуск: python check_answer.py answer.csv
import re
import sys

import pandas as pd

ITEM_RE = re.compile(r"^[0-9a-f]{16}$")


def check(path, queries_path="data/benchmark_queries.parquet",
          items_path="data/benchmark_items.parquet"):
    errors = []
    answer = pd.read_csv(path, dtype=str)
    if list(answer.columns) != ["query_id", "answer"]:
        errors.append("колонки должны быть ровно query_id, answer")
        return errors
    queries = pd.read_parquet(queries_path, columns=["query_id"])
    items = set(pd.read_parquet(items_path, columns=["item_id"])["item_id"])
    if set(answer["query_id"]) != set(queries["query_id"]):
        errors.append("множество query_id не совпадает с бенчмарком")
    if answer["query_id"].duplicated().sum() > 0:
        errors.append("есть дубли query_id")
    for _, row in answer.iterrows():
        ids = str(row["answer"]).split()
        if len(ids) > 50:
            errors.append(f"{row['query_id']}: больше 50 id")
        if len(set(ids)) != len(ids):
            errors.append(f"{row['query_id']}: дубли внутри строки")
        for iid in ids:
            if not ITEM_RE.match(iid):
                errors.append(f"{row['query_id']}: плохой формат id {iid}")
            if iid not in items:
                errors.append(f"{row['query_id']}: id нет в корпусе {iid}")
    return errors


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "answer.csv"
    found = check(path)
    if found:
        print("ОШИБКИ:")
        for err in found[:20]:
            print("-", err)
        sys.exit(1)
    print("Формат в порядке.")
