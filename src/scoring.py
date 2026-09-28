# Финальная формула. Все слагаемые нормирую на максимум в пуле,
# чтобы текст, близость и косинус жили в одной шкале 0..1.
RADIUS_KM = 50
GEO_W = 0.3
FILTER_BONUS = 0.55
FILTER_COVER = 0.9
DENSE_W = 0.5


def geo_bonus(dist_km, radius_km=RADIUS_KM):
    """1 в центре, 0 на границе радиуса и дальше."""
    if dist_km is None or dist_km != dist_km:  # NaN
        return 0.0
    score = 1.0 - dist_km / radius_km
    return score if score > 0.0 else 0.0


def filter_bonus(filter_toks, doc_toks):
    """Бонус, если параметры объявления покрывают фильтр почти целиком."""
    if not filter_toks:
        return 0.0
    if len(filter_toks & doc_toks) / len(filter_toks) >= FILTER_COVER:
        return FILTER_BONUS
    return 0.0


def fuse(text_score, geo_score, bonus, dense_score):
    """Сумма с весами из валидации."""
    return text_score + GEO_W * geo_score + bonus + DENSE_W * dense_score
