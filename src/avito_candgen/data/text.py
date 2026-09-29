"""Нормализация и стемминг текста, токены полей объявлений."""

import re
from functools import lru_cache

import pandas as pd
import Stemmer

from ..paths import cache_path

_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_ru = Stemmer.Stemmer("russian")
_en = Stemmer.Stemmer("english")

STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
    ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
    быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где
    есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж
    тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее
    сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше
    тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой
    перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
    """.split()
)

ITEM_FIELDS = {"title": "item_title_raw", "params": "item_infm_params_text", "desc": "item_description_raw"}


# Стемминг, а не лемматизация: на 189k описаний pymorphy работал бы в разы дольше,
# а для совпадения основ Snowball хватает.
@lru_cache(maxsize=1_000_000)
def stem(token: str) -> str:
    if token[0] < "а":  # латиница или цифры: «iphone», «3d»
        return _en.stemWord(token) if token[0].isalpha() else token
    return _ru.stemWord(token)


def tokens(text: str | None) -> list[str]:
    words = _TOKEN_RE.findall((text or "").lower().replace("ё", "е"))
    return [stem(w) for w in words if w not in STOPWORDS]


def item_tokens(items: pd.DataFrame) -> dict[str, list[list[str]]]:
    path = cache_path("item_tokens.pkl")  # стемминг всех описаний — самое долгое место
    if path.exists():
        return pd.read_pickle(path)
    fields = {name: [tokens(t) for t in items[col]] for name, col in ITEM_FIELDS.items()}
    pd.to_pickle(fields, path)
    return fields
