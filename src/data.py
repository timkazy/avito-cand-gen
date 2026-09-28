# Загрузка данных. Все id читаю строго строками: ведущие нули важны.
from pathlib import Path

import pandas as pd

TRAIN_FILE = "train.parquet"
QUERIES_FILE = "benchmark_queries.parquet"
ITEMS_FILE = "benchmark_items.parquet"


def load_data(data_dir="data"):
    """Читаю три parquet-файла, возвращаю train, запросы и корпус."""
    root = Path(data_dir)
    train = pd.read_parquet(root / TRAIN_FILE)
    queries = pd.read_parquet(root / QUERIES_FILE)
    items = pd.read_parquet(root / ITEMS_FILE)
    for frame in (train, queries, items):
        for col in ("query_id", "item_id"):
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
    return train, queries, items


def item_index(items):
    """Списки id и раскладка позиций объявлений по городам."""
    item_ids = items["item_id"].tolist()
    id_to_pos = {iid: pos for pos, iid in enumerate(item_ids)}
    loc_to_pos = {}
    for pos, loc in enumerate(items["item_location_id"].tolist()):
        loc_to_pos.setdefault(loc, []).append(pos)
    return item_ids, id_to_pos, loc_to_pos
