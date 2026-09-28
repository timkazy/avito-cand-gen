# Тексты: токенизация, лемматизация, чистка раскладки, токены фильтров.
# Я пишу запросы коротко, а объявления длинно, поэтому документ собираю так:
# заголовок три раза + параметры + первые 600 букв описания.
import re

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_lemma_cache = {}

# Частые слова без смысла: есть везде, только тратят вес BM25.
STOP = {"и", "в", "на", "для", "с", "по", "от", "к", "а", "но", "или",
        "не", "это", "как", "так", "же", "все", "при"}

# Служебные фразы фильтров: сами по себе ничего не значат.
JUNK = ["вид услуги", "тип услуги", "место оказания услуг", "название услуги"]

# Перевод латиницы в кириллицу по клавишам (забытая раскладка).
_LAYOUT = str.maketrans("qwertyuiop[]asdfghjkl;'zxcvbnm,.",
                        "йцукенгшщзхъфывапролджэячсмитьбю")


def get_words(text):
    """Только буквы и цифры, маленькими, короче двух букв выбрасываю."""
    words = re.findall("[а-яёa-z0-9]+", str(text).lower())
    return [w for w in words if len(w) >= 2]


def get_lemma(word):
    """Начальная форма слова со словарем-кэшем."""
    if word not in _lemma_cache:
        _lemma_cache[word] = _morph.parse(word)[0].normal_form
    return _lemma_cache[word]


def lemmatize(text):
    """Текст в леммы без стоп-слов."""
    out = []
    for word in get_words(text):
        lemma = get_lemma(word)
        if lemma not in STOP:
            out.append(lemma)
    return out


def doc_text(row):
    """Текст объявления для индекса."""
    title = str(row["item_title_raw"])
    params = str(row["item_infm_params_text"])
    desc = str(row["item_description_raw"])[:600]
    return title + " " + title + " " + title + " " + params + " " + desc


def fix_layout(text):
    """Если в запросе латиницы много (4+ букв и больше, чем кириллицы),
    значит печатали русское слово не в той раскладке — перевожу."""
    low = str(text).lower()
    latin = sum(1 for ch in low if "a" <= ch <= "z")
    russian = sum(1 for ch in low if "а" <= ch <= "я" or ch == "ё")
    if latin >= 4 and latin > russian:
        return low.translate(_LAYOUT)
    return str(text)


def filter_tokens(filter_text):
    """Значимые слова фильтров без служебных фраз."""
    text = str(filter_text).lower()
    for junk in JUNK:
        text = text.replace(junk, " ")
    return set(lemmatize(text))
