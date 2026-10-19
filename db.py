"""Read-only пристап до базата на производи (PostgreSQL / Neon).

Овој модул САМО чита. Сите конекции се отвораат како `readonly`, а сите
барања се `SELECT`. Нема INSERT/UPDATE/DELETE и нема промена на структурата.

Конекцијата се чита од `DATABASE_URL` во `.env`. Ако не е поставена,
модулот не фрла грешка — `search_products` враќа празна листа и чатботот
продолжува да работи само со knowledge base-от.
"""

import os
import re
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()

# TODO : wtf tuka ima koloni so ne postoat u bazata
# Сите колони што ги читаме (не ги измислувај — ова се вистинските во "Product").
PRODUCT_COLUMNS = (
    "id, name, description, price, category, unit, active_substance, "
    "is_active, deadline, stock, default_dosage, manufacturer, formulation, "
    "frac_group, karenca_days, priority_weight, is_preferred, registration_note_mk"
)

# Производите со is_active = false се „избришани" (меко бришење). NULL = активен.
ACTIVE_FILTER = "is_active IS DISTINCT FROM false"

_pool = None
_pool_failed = False


def _create_pool():
    """Отвори мал pool конекции. Пробај verify-full, па fallback на require."""
    from psycopg2.pool import ThreadedConnectionPool

    try:
        return ThreadedConnectionPool(1, 4, dsn=DATABASE_URL)
    except Exception as exc:  # noqa: BLE001
        # Neon бара SSL. Ако verify-full не најде root cert локално, олабави на require.
        if "certificate" in str(exc).lower() and "sslmode=verify-full" in DATABASE_URL:
            relaxed = DATABASE_URL.replace("sslmode=verify-full", "sslmode=require")
            return ThreadedConnectionPool(1, 4, dsn=relaxed)
        raise


def _get_pool():
    global _pool, _pool_failed
    if _pool is not None:
        return _pool
    if _pool_failed or not DATABASE_URL:
        return None
    try:
        _pool = _create_pool()
    except Exception as exc:  # noqa: BLE001
        _pool_failed = True
        print(f"[db] не можам да се поврзам со базата, продолжувам без неа: {exc}")
    return _pool


def db_available() -> bool:
    """Дали имаме конфигурирано и достапно DATABASE_URL."""
    return _get_pool() is not None


@contextmanager
def _readonly_conn():
    pool = _get_pool()
    if pool is None:
        yield None
        return
    conn = pool.getconn()
    try:
        # Тврдо read-only на ниво на сесија — дури и случаен write би се одбил.
        conn.set_session(readonly=True, autocommit=True)
        yield conn
    finally:
        pool.putconn(conn)


def _clean(row: dict) -> dict:
    """Decimal/date -> JSON-пријатни вредности."""
    for key, value in row.items():
        if isinstance(value, Decimal):
            row[key] = int(value) if value == value.to_integral_value() else float(value)
        elif isinstance(value, (date, datetime)):
            row[key] = value.isoformat()
    return row


def _fetch(sql: str, params: list) -> list[dict]:
    with _readonly_conn() as conn:
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)  # само SELECT, параметризирано
                cols = [d[0] for d in cur.description]
                return [_clean(dict(zip(cols, row))) for row in cur.fetchall()]
        except Exception as exc:  # noqa: BLE001
            print(f"[db] барањето не успеа: {exc}")
            return []


# --- Транслитерација (за да работи „armetil“ ↔ „Арметил“) --------------------

_LAT_DIGRAPHS = [
    ("dzh", "џ"), ("lj", "љ"), ("nj", "њ"), ("gj", "ѓ"), ("kj", "ќ"),
    ("zh", "ж"), ("ch", "ч"), ("sh", "ш"), ("dj", "ѓ"), ("dz", "ѕ"),
]

_LAT_SINGLE = {
    "a": "а", "b": "б", "c": "ц", "č": "ч", "ć": "ќ", "d": "д", "đ": "ѓ",
    "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "ј", "k": "к",
    "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с",
    "š": "ш", "t": "т", "u": "у", "v": "в", "z": "з", "ž": "ж",
}

_CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ѓ": "gj", "е": "e",
    "ж": "zh", "з": "z", "ѕ": "dz", "и": "i", "ј": "j", "к": "k", "л": "l",
    "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "ќ": "kj", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "џ": "dzh", "ш": "sh",
}

# TODO : Switch model first oti moze i nema da mi treba vaa translation so nov model primer gemini
# nz realno dali mi trebat site via cyrillic  translations , da gi remove ako ne i da vidam dali mozam bez niv
def _latin_to_cyrillic(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        two = text[i:i + 3].lower()
        for digraph, cyr in _LAT_DIGRAPHS:
            if two.startswith(digraph):
                out.append(cyr)
                i += len(digraph)
                break
        else:
            ch = text[i].lower()
            out.append(_LAT_SINGLE.get(ch, text[i]))
            i += 1
    return "".join(out)


def _cyrillic_to_latin(text: str) -> str:
    return "".join(_CYR_TO_LAT.get(ch.lower(), ch) for ch in text)


_TRAILING = "иаеоуыieaous"


def _variants(token: str) -> list[str]:
    """Сите форми на еден збор: оригинал, транслитерации и скратена форма.

    Скратената форма помага за множина/падеж: „хербициди“ -> „хербицид“,
    „herbicidi“ -> „herbicid“.
    """
    bases = [token]
    if len(token) >= 5 and token[-1].lower() in _TRAILING:
        bases.append(token[:-1])

    seen, out = set(), []
    for base in bases:
        for candidate in (base, _latin_to_cyrillic(base), _cyrillic_to_latin(base)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    return out


# --- Токенизација ------------------------------------------------------------- 
# Stopwords se so cel da ne prebaruva products spored via zborovi
_STOPWORDS = {
    # македонски
    "и", "или", "за", "на", "во", "со", "да", "ли", "е", "се", "не", "што",
    "кој", "која", "кое", "кои", "колку", "има", "имате", "имаме", "дали",
    "ми", "ти", "ние", "вие", "дека", "од", "до", "по", "кај", "при", "ако",
    "тоа", "ова", "го", "ја", "ги", "му", "но", "а", "па", "ќе", "би",
    "кажи", "кажете", "цена", "цената", "кошта", "далие", "лие", "сакам",
    # англиски
    "the", "a", "an", "is", "are", "am", "do", "does", "did", "you", "your",
    "have", "has", "me", "i", "we", "for", "of", "and", "or", "to", "in",
    "on", "at", "what", "which", "how", "much", "many", "price", "cost",
    "there", "any", "my", "our", "can", "could", "please", "this", "that",
}

_WORD_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯѓќљњџѕЉЊЌЃЏЅčćšžđČĆŠŽĐ]+")

#  ги одdelува зборовите од прашањето, одфрла стоп-зборови (на македонски и англиски: „и“, „за“, „цена“, „the“ итн.).
def _tokens(term: str) -> list[str]:
    words = _WORD_RE.findall(term.lower())
    out = []
    for w in words:
        if len(w) < 2:
            continue
        # Стоп-зборот се проверува и во латинична и во кирилична форма,
        # за да се отфрлат и „koj“ и „кој“.
        if w in _STOPWORDS or _latin_to_cyrillic(w) in _STOPWORDS:
            continue
        out.append(w)
    return out


# --- Јавни функции ------------------------------------------------------------

def _run_search(where_clause: str, params: list, limit: int) -> list[dict]:
    sql = (
        f"SELECT {PRODUCT_COLUMNS} FROM \"Product\" "
        f"WHERE {ACTIVE_FILTER} AND ({where_clause}) "
        "ORDER BY is_preferred DESC NULLS LAST, priority_weight DESC NULLS LAST, name "
        "LIMIT %s"
    )
    return _fetch(sql, [*params, limit])


def search_products(term: str, limit: int = 6) -> list[dict]:
    """Барај активни производи по име, активна супстанца, категорија, производител или опис.

    Прво бара производи каде што се совпаѓаат СИТЕ зборови (поточно), а ако
    нема резултат — било кој збор (пошироко). Секој збор се проверува и во
    оригинал и во транслитерирана форма („armetil“ -> „арметил“).
    """
    tokens = _tokens(term)
    if not tokens:
        print(f"[db] прашање: {term!r} → нема клучни зборови (сите се стоп-зборови)")
        return []

    # Го печатам она што оди во пребарувањето, за да се види како се бара.
    keywords = [v for token in tokens for v in _variants(token)]
    print(f"[db] прашање: {term!r}")
    print(f"[db]   извлечени зборови: {tokens}  (стоп-зборовите се отфрлени)")
    print(f"[db]   LIKE клучни зборови: {keywords}")
    print("[db]   се бараат во: name, active_substance, category, manufacturer, description")

    groups: list[str] = []
    flat_clauses: list[str] = []
    params: list = []

    for token in tokens:
        ors = []
        for v in _variants(token):
            like = f"%{v}%"
            clause = (
                "(name ILIKE %s OR coalesce(active_substance, '') ILIKE %s "
                "OR coalesce(category, '') ILIKE %s OR coalesce(manufacturer, '') ILIKE %s "
                "OR coalesce(description, '') ILIKE %s)"
            )
            ors.append(clause)
            flat_clauses.append(clause)
            params.extend([like] * 5)
        groups.append("(" + " OR ".join(ors) + ")")

    # 1) Сите зборови мора да се совпаднат (најточно).
    rows = _run_search(" AND ".join(groups), params, limit)
    print(f"[db]   AND (сите зборови) → {len(rows)} производи")
    if rows:
        return rows

    # 2) Отстапка: било кој збор да се совпадне (пошироко).
    if len(tokens) > 1:
        rows = _run_search(" OR ".join(flat_clauses), params, limit)
        print(f"[db]   OR (било кој збор) → {len(rows)} производи")
    return rows


def format_products(rows: list[dict]) -> str:
    """Претвори ги редовите во читлив текст за LLM-от."""
    if not rows:
        return "NONE (нема пронајден производ во базата)"
    # TODO: Tuka ima labels so ne postoat, izbrishi gi
    labels = [
        ("category", "category"),
        ("active_substance", "active substance"),
        ("manufacturer", "manufacturer"),
        ("formulation", "formulation"),
        ("price", "price"),
        ("unit", "unit"),
        ("stock", "stock"),
        ("deadline", "deadline"),
        ("default_dosage", "default dosage"),
        ("karenca_days", "karenca (days)"),
        ("frac_group", "FRAC group"),
        ("registration_note_mk", "registration note"),
    ]

    lines = []
    for row in rows:
        parts = [str(row.get("name") or "?")]
        for key, label in labels:
            value = row.get(key)
            if value not in (None, "", 0):
                parts.append(f"{label}: {value}")
        lines.append("- " + " | ".join(parts))
    return "\n".join(lines)
