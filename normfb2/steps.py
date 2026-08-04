"""
Шаги нормализации 1-29
"""

import unicodedata
from typing import List, Tuple

import regex as re

from normfb2.data import (
    CURRENCIES,
    DECIMAL_PLACES,
    EN_LETTER_NAMES,
    GREEK_LETTER_NAMES,
    GREEK_TO_RUSSIAN,
    LATIN_DIACRITICS_MAP,
    LETTER_NAMES,
    MEASUREMENTS_AMBIGUOUS,
    MEASUREMENTS,
    MONTHS_GEN,
    MULTIPLIERS,
    SECTION_ABBR,
    SUFFIX_FORMS,
    SYMBOLS,
    TYPOGRAPHY_REPLACEMENTS,
    VULGAR_FRACTIONS,
    WEB_SYMBOLS,
)
from normfb2.dicts import AcronymDict
from normfb2.roman import (
    RE_ROMAN,
    RE_ROMAN_VALID,
    ROMAN_STOPLIST,
    is_valid_roman,
    ordinal_text,
)
from normfb2.utils import _feminine_last, _plural, number_to_words

COMBINING_DIACRITICAL = "\u0300-\u036f"
RUSSIAN_CHAR = f"[а-яёА-ЯЁ{COMBINING_DIACRITICAL}]"
TAG_EN_OPEN = "<tts_en>"
TAG_EN_CLOSE = "</tts_en>"
PROTECT_MARKER_PREFIX = "ЪььЬььЪ"
PROTECT_MARKER_SUFFIX = "ЪььЪььЪ"
# Буква-номер для маркера (без цифр)
_MARKER_LETTERS = "абвгдежзийклмнопрстуфхцчшщыэюя"


def _detect_year_case(w: str) -> str:
    """Определяет падеж для слов 'год' (аналогично detect_case для 'век')"""
    import unicodedata

    w_clean = __import__("re").sub(r"[)!?;:]+$", "", w)
    w_norm = "".join(
        c for c in unicodedata.normalize("NFD", w_clean) if not unicodedata.combining(c)
    )
    cases = {
        "год": "nom_m",
        "года": "gen",
        "году": "dat",
        "годе": "prep",
        "годом": "instr",
        "годов": "pl",
        "годам": "instr",
        "годами": "instr",
        "годах": "pl",
        "гг": "nom_m",
        "гг.": "nom_m",
        "годы": "nom_m",
    }
    return cases.get(w_norm, None)


# --- Шаг 1: Типографика ---
def normalize_typography(text: str) -> str:
    text = re.sub(r"\*\*|__|`", "", text)
    for old, new in TYPOGRAPHY_REPLACEMENTS.items():
        text = text.replace(old, new)
    # Преобразование формата +а → а́ (ударение после гласной)
    text = re.sub(r"([аеёиоуыэюяАЕЁИОУЫЭЮЯ])\+", r"\1\u0301", text)
    return text


# --- Шаг 2: URL и email ---
def normalize_web(text: str) -> str:
    def spell(m):
        s = m.group(0).rstrip(".,!?")
        for sym, word in WEB_SYMBOLS.items():
            s = s.replace(sym, word)
        return re.sub(r" {2,}", " ", s).strip()

    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[A-Za-zА-Яа-я]{2,}\b", spell, text)
    text = re.sub(
        r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|ru|org|net|info|io|edu|gov|рф)\b",
        spell,
        text,
        flags=re.I,
    )
    text = re.sub(r"#([A-Za-zА-Яа-яёЁ0-9_]+)", r"хештег \1", text)
    return text


# --- Шаг 3: RegEx замены ---
def normalize_regex_dict(text: str, acro_dict: AcronymDict) -> str:
    for compiled_pat, replacement in acro_dict._regex_compiled:
        if compiled_pat.search(text):
            text = compiled_pat.sub(replacement, text)
    return text


# --- Шаг 5: Структурные ссылки ---
def normalize_sections(text: str) -> str:
    return re.sub(
        r"(?<![" + RUSSIAN_CHAR[1:-1] + r"])(ст|пп|табл|рис|гл|п)\.\s*(?=\d)",
        lambda m: SECTION_ABBR[m.group(1).lower()] + " ",
        text,
        flags=re.I,
    )


# --- Шаг 6: Группы цифр ---
def normalize_number_groups(text: str) -> str:
    return re.sub(
        r"\b\d{1,3}(?: \d{3})+\b", lambda m: m.group(0).replace(" ", ""), text
    )


# --- Шаг 7: Диапазоны ---
def normalize_ranges(words: list, gaps: list) -> tuple:

    i = 0
    while i < len(words):
        if words[i].isdigit() and i + 1 < len(words) and gaps[i + 1] in ("–", "—", "-"):
            n1 = int(words[i])
            if words[i + 1].isdigit():
                n2 = int(words[i + 1])
                next_word = words[i + 2] if i + 2 < len(words) else ""
                case = (
                    _detect_year_case(next_word.lower().rstrip("."))
                    if next_word
                    else None
                )

                if case:
                    words[i] = ordinal_text(n1, case)
                    words[i + 1] = ordinal_text(n2, case)
                    # Убираем "одна́" перед "ты́сяча"
                    words[i] = words[i].replace("одна́ ты́сяча", "ты́сяча")
                    words[i + 1] = words[i + 1].replace("одна́ ты́сяча", "ты́сяча")

                else:
                    words[i] = number_to_words(n1)
                    gaps[i + 1] = " "
                    words[i + 1] = number_to_words(n2)
                i += 2
                continue
        if words[i].isdigit() and i > 0 and gaps[i] in ("–", "—", "-"):
            i += 1
            continue
        i += 1
    return words, gaps


# --- Шаг 8: Даты ---
def normalize_dates(text: str) -> str:
    month_by_num = {f"{i:02d}": m for i, m in enumerate(MONTHS_GEN, start=1)}

    def date_numeric(m):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mn = month_by_num.get(f"{month:02d}")
        if not mn:
            return m.group(0)
        return f"{ordinal_text(day, 'gen')} {mn} {number_to_words(year)} го́да"

    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", date_numeric, text)
    text = re.sub(
        r"\b(\d{4})\s*г\.(?!" + RUSSIAN_CHAR + r")",
        lambda m: f"{ordinal_text(int(m.group(1)), 'nom_m')} го́д",
        text,
    )
    return text


# --- Шаг 8.5: Десятичные дроби с дефисом ---
def normalize_decimal_hyphen(text: str) -> str:
    def repl(m):
        integer_part, frac_part, suffix = int(m.group(1)), m.group(2), m.group(3)
        frac_words = number_to_words(int(frac_part))
        places = {1: "деся́тых", 2: "со́тых", 3: "ты́сячных"}
        place = places.get(len(frac_part), "")
        if integer_part == 0:
            return f"{frac_words} {place} {suffix}"
        return f"{number_to_words(integer_part)} и {frac_words} {place} {suffix}"

    return re.sub(r"\b(\d+),(\d+)-([а-яёА-ЯЁ][а-яёА-ЯЁ]+)", repl, text)


# --- Шаг 9: Составные числительные ---
def normalize_compounds(text: str) -> str:
    def repl(m):
        n = int(m.group(1))
        words = number_to_words(n).split()
        prefix = words[-1]
        if n == 1:
            prefix = "одно́"
        elif n == 2:
            prefix = "дву́х"
        elif n == 3:
            prefix = "трё́х"
        elif n == 4:
            prefix = "четырё́х"
        elif prefix.endswith("ть"):
            prefix = prefix[:-1] + "и"
        suffix_part = re.sub(r"[" + COMBINING_DIACRITICAL + r"]", "", m.group(2))
        return prefix + suffix_part

    return re.sub(
        r"\b(\d+)-(["
        + RUSSIAN_CHAR[1:-1]
        + r"]{2,}(?:ий|ый|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ым|им|ом|ем)["
        + COMBINING_DIACRITICAL
        + r"]*)\b",
        repl,
        text,
    )


# --- Шаг 10: Время ---
def normalize_time(text: str) -> str:
    def repl(m):
        h, mn = int(m.group(1)), int(m.group(2))
        h_word = (
            "ча́с"
            if h == 1
            else f"{number_to_words(h)} {_plural(h, ['ча́с', 'часа́', 'часо́в'])}"
        )
        if mn == 0:
            return h_word
        mn_word = f"{' '.join(_feminine_last(number_to_words(mn).split()))} {_plural(mn, ['мину́та', 'мину́ты', 'мину́т'])}"
        return f"{h_word} {mn_word}"

    return re.sub(r"(?<![\d:])(\d{1,2}):([0-5]\d)(?![\d:])", repl, text)


# --- Шаг 11: Дроби ---
def normalize_fractions(text: str) -> str:
    # Числитель — количественное, знаменатель — порядковое в женском роде
    from normfb2.data import VULGAR_FRACTIONS
    from normfb2.roman import ordinal_text
    from normfb2.utils import _feminine_last, number_to_words

    for ch, (num, den) in VULGAR_FRACTIONS.items():
        if den == 0:
            continue
        num_words = " ".join(_feminine_last(number_to_words(num).split()))
        if num == 1:
            den_words = ordinal_text(den, "nom_f")
        else:
            den_words = ordinal_text(den, "pl")
        name = f"{num_words} {den_words}"
        text = text.replace(ch, name)
    return text


# --- Шаг 12: Проценты ---
def normalize_percent(text: str) -> str:
    # % и ‱ (базисный пункт)
    text = re.sub(
        r"(\d+)\s*%",
        lambda m: f"{number_to_words(int(m.group(1)))} {_plural(int(m.group(1)), ['проце́нт', 'проце́нта', 'проце́нтов'])}",
        text,
    )
    text = re.sub(
        r"(\d+)\s*‱",
        lambda m: f"{number_to_words(int(m.group(1)))} {_plural(int(m.group(1)), ['ба́зисный пу́нкт', 'ба́зисных пу́нкта', 'ба́зисных пу́нктов'])}",
        text,
    )
    return text


# --- Шаг 13: Множители ---
def normalize_multipliers(words: list, gaps: list) -> tuple:
    i = 0
    while i < len(words):
        if words[i].isdigit() and i + 1 < len(words):
            next_w = words[i + 1].lower().rstrip(".")
            if next_w in ("тыс", "млн", "млрд"):
                forms, feminine = MULTIPLIERS[next_w]
                n = int(words[i])
                w = number_to_words(n).split()
                if feminine:
                    _feminine_last(w)
                words[i] = " ".join(w)
                words[i + 1] = _plural(n, forms)
                i += 2
                continue
        i += 1
    return words, gaps


# --- Шаг 14a: Единицы измерения (основные) ---
def normalize_measurements(words: list, gaps: list) -> tuple:
    import unicodedata

    if not MEASUREMENTS:
        return words, gaps
    i = 0
    while i < len(words):
        if words[i].isdigit() and i + 1 < len(words):
            next_w = words[i + 1].rstrip(".")
            next_w_clean = "".join(
                c for c in unicodedata.normalize("NFD", next_w)
                if not unicodedata.combining(c)
            )
            if next_w_clean in MEASUREMENTS:
                forms, gender = MEASUREMENTS[next_w_clean]
                n = int(words[i])
                w = number_to_words(n).split()
                if gender == "f":
                    _feminine_last(w)
                words[i] = " ".join(w)
                words[i + 1] = _plural(n, forms)
                i += 2
                continue
        i += 1
    return words, gaps


# --- Шаг 14b: Единицы измерения (сомнительные) ---
def normalize_measurements_1letter(words: list, gaps: list) -> tuple:
    import unicodedata

    if not MEASUREMENTS_AMBIGUOUS:
        return words, gaps
    i = 0
    while i < len(words):
        if words[i].isdigit() and i + 1 < len(words):
            next_w = words[i + 1].rstrip(".")
            next_w_clean = "".join(
                c for c in unicodedata.normalize("NFD", next_w)
                if not unicodedata.combining(c)
            )
            if next_w_clean in MEASUREMENTS_AMBIGUOUS:
                forms, gender = MEASUREMENTS_AMBIGUOUS[next_w_clean]
                n = int(words[i])
                w = number_to_words(n).split()
                if gender == "f":
                    _feminine_last(w)
                words[i] = " ".join(w)
                words[i + 1] = _plural(n, forms)
                i += 2
                continue
        i += 1
    return words, gaps


# --- Шаг 15: Порядковые числительные с суффиксами ---
def normalize_ordinal_suffixes(text: str) -> str:
    def repl(m):
        n = int(m.group(1))
        suffix_clean = "".join(
            c
            for c in unicodedata.normalize("NFD", m.group(2))
            if not unicodedata.combining(c)
        )
        case = SUFFIX_FORMS.get(suffix_clean, "nom_m")
        return ordinal_text(n, case)

    suffix_patterns = []
    for suffix in SUFFIX_FORMS:
        s = ""
        for char in suffix:
            s += char + "[" + COMBINING_DIACRITICAL + "]?"
        suffix_patterns.append(s)
    return re.sub(
        r"(\d+)[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D]("
        + "|".join(suffix_patterns)
        + r")(?![а-яё"
        + COMBINING_DIACRITICAL
        + r"])",
        repl,
        text,
    )


# --- Шаг 16: Акронимы и римские цифры ---
def normalize_acronyms(text: str, acro_dict: AcronymDict) -> Tuple[str, List[str]]:
    protected = []

    def protect(value: str) -> str:
        idx = len(protected)
        protected.append(value)
        letter = _MARKER_LETTERS[idx % len(_MARKER_LETTERS)]
        return f"{PROTECT_MARKER_PREFIX}{letter}{PROTECT_MARKER_SUFFIX}"

    if acro_dict.acro_ru:
        ru_keys = sorted(acro_dict.acro_ru.keys(), key=len, reverse=True)
        ru_pattern = r"\b(?:" + "|".join(re.escape(k) for k in ru_keys) + r")\b"

        def ru_acro(m):
            word = m.group(0)
            pron = acro_dict.get_pronunciation(word, "ru")
            return protect(pron) if pron else word

        text = re.sub(ru_pattern, ru_acro, text)

    if acro_dict.acro_en:
        en_keys = sorted(acro_dict.acro_en.keys(), key=len, reverse=True)
        en_pattern = r"\b(?:" + "|".join(re.escape(k) for k in en_keys) + r")\b"

        def en_acro(m):
            word = m.group(0)
            if (
                RE_ROMAN_VALID.match(word.upper())
                and word.upper() not in ROMAN_STOPLIST
            ):
                return word

            en_pron, ru_pron = acro_dict.get_en_pronunciation(word)

            # Если ru есть, а en пустой — всегда русское произношение
            if ru_pron and not en_pron:
                return ru_pron

            if ru_pron and en_pron:
                # Оба есть — проверяем контекст
                start = m.start()
                end = m.end()
                left_char = ""
                for i in range(start - 1, -1, -1):
                    ch = text[i]
                    if ch.isalpha() or ("а" <= ch.lower() <= "я" or ch.lower() == "ё"):
                        left_char = ch
                        break
                    if ch not in " .,;:!?-\"'()":
                        left_char = ch
                        break
                right_char = ""
                for i in range(end, len(text)):
                    ch = text[i]
                    if ch.isalpha() or ("а" <= ch.lower() <= "я" or ch.lower() == "ё"):
                        right_char = ch
                        break
                    if ch not in " .,;:!?-\"'()":
                        right_char = ch
                        break
                near_cyrillic = (
                    left_char
                    and ("а" <= left_char.lower() <= "я" or left_char.lower() == "ё")
                ) or (
                    right_char
                    and ("а" <= right_char.lower() <= "я" or right_char.lower() == "ё")
                )
                if near_cyrillic:
                    return ru_pron
                else:
                    return protect(en_pron)

            if en_pron:
                return protect(en_pron)

            return word

        text = re.sub(en_pattern, en_acro, text)

    return text, protected


# --- Шаг 17: Имена правителей с римскими цифрами ---


# --- Новый шаг: Римские цифры (века, диапазоны, ссылки) ---
def normalize_roman_generic(words: list, gaps: list, acro_dict=None) -> tuple:
    import unicodedata

    from normfb2.data import GREEK_LETTER_NAMES, LETTER_NAMES
    from normfb2.roman import (
        RE_ROMAN,
        ROMAN_STOPLIST,
        is_valid_roman,
        ordinal_text,
        roman_to_int,
    )
    from normfb2.utils import number_to_words

    def spell_letters(letters: str) -> str:
        """Произносит буквы (латиница, греческие)"""
        result = []
        for ch in letters:
            name = LETTER_NAMES.get(ch.lower())
            if name:
                result.append(name)
            elif ch in GREEK_LETTER_NAMES:
                result.append(GREEK_LETTER_NAMES[ch])
            else:
                result.append(ch.lower())
        return "-".join(result)

    # Падежи для слов "век"
    def detect_case(w, is_range=False):
        w_clean = __import__("re").sub(r"[)!?;:]+$", "", w)
        w_norm = "".join(
            c
            for c in unicodedata.normalize("NFD", w_clean)
            if not unicodedata.combining(c)
        )
        # Омограф "века": ед.ч. род. "ве́ка" (ударение на Е) vs мн.ч. им. "века́" (ударение на А)
        if w_norm == "века":
            # Проверяем ударение в исходном слове
            if "́" in w_clean:
                # Есть ударение. Если на последней "а" — мн.ч., но числительные в ед.ч.
                if w_clean.endswith("а́") or w_clean.endswith("а́"):
                    return "nom_m"
                else:
                    return "gen"
            # Нет ударения — для диапазона обычно род. падеж (IV-V ве́ка)
            return "gen" if is_range else "gen"
        if w_norm == "веков":
            return "pl"  # gen_pl = pl (пя́тых)
        if w_norm == "векам":
            return "dat"
        if w_norm == "веками":
            return "instr"
        if w_norm == "веках":
            return "prep"
        cases = {
            "век": "nom_m",
            "веку": "dat",
            "века": "gen",
            "веке": "prep",
            "веком": "instr",
        }
        if w_norm in cases:
            return cases[w_norm]
        if w_norm in ("вв.", "вв"):
            return "nom_m"
        if w_norm in ("в.", "в"):
            return "nom_m"
        return None

    # Предлоги и требуемые падежи
    prep_cases = {
        "в": "prep",
        "во": "prep",
        "на": "prep",
        "о": "prep",
        "об": "prep",
        "при": "prep",
        "к": "dat",
        "ко": "dat",
        "по": "dat",
        "с": "gen",
        "со": "gen",
        "от": "gen",
        "до": "gen",
        "из": "gen",
        "у": "gen",
        "для": "gen",
        "без": "gen",
        "между": "instr",
    }

    # Ссылочные слова перед римской цифрой
    ref_words = {
        "кн": "кни́га",
        "т": "то́м",
        "ч": "ча́сть",
        "гл": "глава́",
        "разд": "разде́л",
    }

    i = 0
    while i < len(words):
        w = words[i]

        # Пропускаем не-римские
        if not (
            w
            and RE_ROMAN.match(w)
            and is_valid_roman(w)
            and w.upper() not in ROMAN_STOPLIST
        ):
            i += 1
            continue

        # Дефисные конструкции:
        if i + 1 < len(words) and gaps[i + 1] in ("-", "–", "—"):
            next_w = words[i + 1]
            w_rom = RE_ROMAN.match(w) and is_valid_roman(w)
            nw_rom = RE_ROMAN.match(next_w) and is_valid_roman(next_w)

            if w_rom and nw_rom:
                # Обе римские
                if len(w) == 1 and len(next_w) == 1:
                    # Обе односимвольные → буквы (I-D, V-X), пропускаем
                    i += 1
                    continue
                # Диапазон (V-VIII, IV-X) — не пропускаем, обработаем ниже
            elif w_rom and not nw_rom:
                # Римская + не-римская → число + буква (II-В, I-Δ)
                words[i] = number_to_words(roman_to_int(w))
                words[i + 1] = spell_letters(next_w)
                i += 2
                continue
            else:
                # Не-римская + что-то → пропускаем
                i += 1
                continue

        if i > 0 and gaps[i] in ("-", "–", "—"):
            # Вторая часть дефисной конструкции — всегда пропускаем
            i += 1
            continue

        # Пропускаем одиночные римские если это инициалы (V. I. Lenin)
        if len(w) == 1 and w.upper() in "IVXLCDM":
            # Проверим: это инициал? (после точка и ещё буква с точкой)
            is_initials = False
            if (
                i + 1 < len(gaps)
                and gaps[i + 1].startswith(".")
                and i + 1 < len(words)
                and len(words[i + 1]) == 1
                and words[i + 1].isalpha()
            ):
                if i + 2 < len(gaps) and gaps[i + 2].startswith("."):
                    is_initials = True

            if is_initials:
                i += 1
                continue

            # Проверим наличие латиницы вокруг
            has_latin = False
            for offset in range(-3, 4):
                idx = i + offset
                if 0 <= idx < len(words) and offset != 0:
                    word = words[idx]
                    if ":" in word:
                        continue
                    # Пропускаем другие одиночные римские цифры
                    if len(word) == 1 and word.upper() in "IVXLCDM":
                        continue
                    # Пропускаем слова которые сами римские цифры
                    if RE_ROMAN.match(word) and is_valid_roman(word):
                        continue
                    if any(c.isascii() and c.isalpha() for c in word):
                        has_latin = True
                        break
            if has_latin:
                i += 1
                continue

        roman_str = w
        num = roman_to_int(roman_str)

        # Проверяем, не имя ли правителя перед римской
        prev_word = words[i - 1] if i > 0 else ""
        # Проверим что между именем и римской только пробел (не скобки)
        if acro_dict and prev_word and gaps[i].strip() == "":
            ruler_form = acro_dict.get_ruler_form(prev_word)
            if ruler_form:
                # Это имя правителя — пропускаем, обработается в normalize_rulers
                i += 1
                continue

        # Проверяем ссылочные слова перед римской
        prev_clean = prev_word.lower().rstrip(".")
        prev_clean = (
            "".join(
                c
                for c in unicodedata.normalize("NFD", prev_clean)
                if not unicodedata.combining(c)
            )
            if prev_word
            else ""
        )
        if prev_clean in ref_words:
            ref_text = ref_words[prev_clean]
            words[i] = number_to_words(num)
            words[i - 1] = ref_text
            # Убираем точку из зазора между ссылкой и цифрой
            if gaps[i].startswith("."):
                gaps[i] = gaps[i][1:]
            # Убираем точку из зазора после цифры (если есть)
            if i + 1 < len(gaps) and gaps[i + 1].startswith("."):
                gaps[i + 1] = gaps[i + 1][1:]
            i += 1
            continue

        # Ищем диапазон
        range_end = i
        while (
            range_end + 1 < len(words)
            and gaps[range_end + 1] in ("–", "—", "-")
            and RE_ROMAN.match(words[range_end + 1])
            and is_valid_roman(words[range_end + 1])
        ):
            range_end += 1

        if range_end > i:
            # Диапазон: ищем падеж по слову после диапазона
            next_idx = range_end + 1
            next_word = words[next_idx] if next_idx < len(words) else ""
            case = detect_case(next_word, is_range=True) if next_word else None

            if case:
                # Определяем форму для каждой части диапазона
                for j in range(i, range_end + 1):
                    n = roman_to_int(words[j])
                    words[j] = ordinal_text(n, case)
                # Добавляем ударение в слово после диапазона если нужно
                if next_word:
                    stress_map = {
                        "века": "ве́ка" if case == "gen" else "века́",
                        "веков": "веко́в",
                        "векам": "века́м",
                        "веками": "века́ми",
                        "веках": "века́х",
                    }
                    if next_word in stress_map:
                        words[next_idx] = stress_map[next_word]
                i = range_end + 1
                continue
            else:
                # Диапазон без падежного слова — просто числа
                for j in range(i, range_end + 1):
                    n = roman_to_int(words[j])
                    words[j] = number_to_words(n)
                i = range_end + 1
                continue

        # Одиночная римская цифра
        next_word = words[i + 1] if i + 1 < len(words) else ""
        prev_w = words[i - 1] if i > 0 else ""

        # Определяем падеж
        case = None

        # Сначала проверим: если перед римской предлог
        prev_w_clean = "".join(
            c
            for c in unicodedata.normalize("NFD", prev_w.lower())
            if not unicodedata.combining(c)
        )
        if prev_w_clean in prep_cases:
            prep_case = prep_cases[prev_w_clean]
            if next_word.lower() in prep_cases:
                # "с I по V век" — для первой римской используем падеж первого предлога
                case = prep_case
            else:
                case = detect_case(next_word)
                if not case:
                    case = prep_case
        elif next_word:
            # Без предлога — падеж от следующего слова
            case = detect_case(next_word)

        if case:
            words[i] = ordinal_text(num, case)
            # Если next_word — "век/века/веку/..." — пропускаем его
            if next_word and detect_case(next_word):
                i += 2
            else:
                i += 1
            continue

        # Если следующее слово — предлог ("по", "до")
        # Тогда ищем падеж дальше
        if (
            "".join(
                c
                for c in unicodedata.normalize("NFD", next_word.lower())
                if not unicodedata.combining(c)
            )
            in prep_cases
            and len(next_word) <= 3
        ):
            lookahead = i + 2
            if (
                lookahead < len(words)
                and RE_ROMAN.match(words[lookahead])
                and is_valid_roman(words[lookahead])
            ):
                lookahead += 1
            if lookahead < len(words):
                case = detect_case(words[lookahead])
                if case:
                    words[i] = ordinal_text(num, case)
                    i += 1
                    continue

        # Fallback: римская цифра без контекста → количественное числительное
        words[i] = number_to_words(num)
        i += 1

    return words, gaps


def normalize_rulers(words: list, gaps: list, acro_dict: AcronymDict) -> tuple:
    import re
    import unicodedata

    from normfb2.roman import ordinal_text, roman_to_int
    from normfb2.utils import detokenize_words_gaps, tokenize_words_gaps

    text = detokenize_words_gaps(words, gaps)

    def repl_ruler(m):
        name, roman = m.group(1), m.group(2)
        ruler_form = acro_dict.get_ruler_form(name)
        if not ruler_form:
            return m.group(0)
        gender = ruler_form.get("gender", "mas")
        name_clean = "".join(
            c
            for c in unicodedata.normalize("NFD", name.lower())
            if not unicodedata.combining(c)
        )
        case_map = {
            "nom": "nom",
            "gen": "gen",
            "acc": "acc",
            "dat": "dat",
            "ins": "instr",
            "prep": "prep",
        }
        detected = None
        for form_key, case_key in case_map.items():
            form_val = ruler_form.get(form_key, "")
            form_clean = "".join(
                c
                for c in unicodedata.normalize("NFD", form_val.lower())
                if not unicodedata.combining(c)
            )
            if name_clean == form_clean:
                detected = case_key
                break
        if not detected:
            detected = "nom"
        if detected == "nom":
            detected = "nom_f" if gender == "fem" else "nom_m"
        elif detected == "acc":
            detected = "acc_f" if gender == "fem" else "acc_m"
        return f"{name} {ordinal_text(roman_to_int(roman), detected)}"

    text = re.sub(r"\b([А-ЯЁа-яё̀-ͯ]+)\s+([IVXLCDM]+)\b", repl_ruler, text)
    return tokenize_words_gaps(text)


def normalize_alphanumeric(words: list, gaps: list) -> tuple:
    from normfb2.roman import roman_to_int
    from normfb2.utils import detokenize_words_gaps

    def spell_letters(letters: str) -> str:
        # Если это многосимвольная римская цифра (II, VIII) — не разбираем по буквам
        if (
            len(letters) > 1
            and RE_ROMAN.match(letters)
            and is_valid_roman(letters)
            and letters.upper() not in ROMAN_STOPLIST
        ):
            return number_to_words(roman_to_int(letters))
        result = []
        for ch in letters:
            name = LETTER_NAMES.get(ch.lower())
            if name:
                result.append(name)
            elif ch in GREEK_LETTER_NAMES:
                result.append(GREEK_LETTER_NAMES[ch])
            else:
                result.append(ch.lower())
        return "-".join(result)

    i = 0
    while i < len(words):
        w = words[i]

        # Инициалы: А. Б. → а́-бэ́
        if (
            len(w) == 1
            and w[0].isalpha()
            and i + 1 < len(words)
            and gaps[i + 1].startswith(".")
            and len(words[i + 1]) == 1
            and words[i + 1].isalpha()
        ):
            j = i
            initials = []
            while (
                j < len(words)
                and len(words[j]) == 1
                and words[j].isalpha()
                and j + 1 < len(gaps)
                and gaps[j + 1].startswith(".")
            ):
                initials.append(words[j])
                j += 1
            if len(initials) >= 2 and not any(
                c in "IVXLCDM" for c in "".join(initials)
            ):
                spelled = "-".join(spell_letters(l) for l in initials) + ", "
                words[i] = spelled
                for x in range(i + 1, j):
                    words[x] = ""
                    gaps[x + 1] = ""
                gaps[i + 1] = ""
                i = j
                continue

        # Одиночная буква с точкой: М. → эм.
        # (но не "в." — век, не римские цифры, не если следующее слово цифра)
        if (
            len(w) == 1
            and w[0].isalpha()
            and w.lower() not in ("в",)
            and w not in "IVXLCDM"
            and gaps[i + 1].startswith(".")
            and not (i + 1 < len(words) and words[i + 1].isdigit())
        ):
            words[i] = spell_letters(w) + "."
            i += 1
            continue

        # Дефисные конструкции: только односимвольные буквы (I-D, V-Δ)
        # Длинные латинские слова (MS-DOS, Gerichtetsein-auf) не трогаем
        if i + 1 < len(words) and gaps[i + 1] in ("-", "–", "—"):
            next_w = words[i + 1]
            # Пропускаем если хотя бы одна часть содержит кириллицу (уже число)
            if any("а" <= c.lower() <= "я" or c.lower() == "ё" for c in w) or any(
                "а" <= c.lower() <= "я" or c.lower() == "ё" for c in next_w
            ):
                i += 2
                continue
            # Обрабатываем только если обе части односимвольные буквы
            if len(w) == 1 and len(next_w) == 1 and w.isalpha() and next_w.isalpha():
                words[i] = spell_letters(w)
                words[i + 1] = spell_letters(next_w)
                i += 2
                continue
            # Всё остальное пропускаем
            i += 2
            continue

        # Буквы-цифры через дефис: А-123 → а сто двадцать три
        # Только если слово короткое (1-2 буквы) или все заглавные (акроним)
        if all(c.isalpha() or ord(c) in range(0x0300, 0x0370) for c in w) and any(
            c.isalpha() for c in w
        ):
            # Считаем буквы без диакритики
            letters_only = [c for c in w if c.isalpha()]
            is_short = len(letters_only) <= 2
            is_acronym = len(letters_only) >= 2 and all(
                c.isupper() for c in letters_only
            )
            if is_short or is_acronym:
                if (
                    i + 1 < len(words)
                    and gaps[i + 1] in ("-", "–", "—")
                    and words[i + 1].isdigit()
                ):
                    words[i] = spell_letters(w)
                    words[i + 1] = number_to_words(int(words[i + 1]))
                    i += 2
                    continue
                # Буквы перед цифрами без пробела: А123 → а сто двадцать три
                if i + 1 < len(words) and words[i + 1].isdigit() and gaps[i + 1] == "":
                    words[i] = spell_letters(w)
                    gaps[i + 1] = " "
                    words[i + 1] = number_to_words(int(words[i + 1]))
                    i += 2
                    continue

        # Цифры перед буквами: 123с → сто двадцать три, эс
        # (только если буква не одиночная кириллическая — это предлог/союз)
        if (
            w.isdigit()
            and i + 1 < len(words)
            and words[i + 1].isalpha()
            and gaps[i + 1] == ""
            and (len(words[i + 1]) == 1 or words[i + 1].isupper())
        ):
            words[i] = number_to_words(int(w)) + ", "
            words[i + 1] = spell_letters(words[i + 1])
            i += 2
            continue

        # Одиночная не-кириллическая буква после дефиса: -D → -ди
        if (
            i > 0
            and gaps[i] in ("-", "–", "—")
            and len(w) == 1
            and w.isalpha()
            and w not in "IVXLCDM"
            and not ("а" <= w.lower() <= "я" or w.lower() == "ё")
        ):
            words[i] = spell_letters(w)
            i += 1
            continue

        i += 1

    # Убираем пустые слова
    result_words = []
    result_gaps = [gaps[0]]
    for w, g in zip(words, gaps[1:]):
        if w:
            result_words.append(w)
            result_gaps.append(g)
        else:
            result_gaps[-1] += g

    return result_words, result_gaps


# --- Шаг 19: Падежное согласование ---
def normalize_case_context(text: str) -> str:
    return text


# --- Шаг 20: Восстановление маркеров ---
def restore_protected(text: str, protected: List[str]) -> str:
    for i, value in enumerate(protected):
        letter = _MARKER_LETTERS[i % len(_MARKER_LETTERS)]
        marker = f"{PROTECT_MARKER_PREFIX}{letter}{PROTECT_MARKER_SUFFIX}"
        text = text.replace(marker, value)
    return text


# --- Шаг 21: Спецсимволы ---
def normalize_symbols(text: str) -> str:
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, " " + word + " ")
    return re.sub(r" {2,}", " ", text)


# --- Шаг 22: Десятичные дроби ---
def normalize_decimals(text: str) -> str:
    def repl(m):
        ip, fp = m.group(1), m.group(2)
        if set(fp) == {"0"}:
            return number_to_words(int(ip))
        place = DECIMAL_PLACES.get(len(fp))
        if not place:
            return m.group(0)
        int_words = _feminine_last(number_to_words(int(ip)).split())
        whole = "це́лая" if (int(ip) % 10 == 1 and int(ip) % 100 != 11) else "це́лых"
        frac_words = _feminine_last(number_to_words(int(fp)).split())
        return f"{' '.join(int_words)} {whole} и́ {' '.join(frac_words)} {place}"

    return re.sub(r"\b(\d+),(\d+)\b", repl, text)


# --- Шаг 23: Валюта ---
def normalize_currency(text: str) -> str:
    patterns = {
        "rub": [
            r"(\d+(?:\.\d\d)?)\s*((руб|ру́б)(л(е́й|я́|ь))?(?!" + RUSSIAN_CHAR + r")|₽)"
        ],
        "usd": [
            r"(\d+(?:\.\d\d)?)\s*(до́ллар(о́в|а|ы)?(?!" + RUSSIAN_CHAR + r")|\$)",
            r"\$(\d+(?:\.\d\d)?)",
        ],
        "eur": [
            r"(\d+(?:\.\d\d)?)\s*(е́вро(?!" + RUSSIAN_CHAR + r")|€)",
            r"€(\d+(?:\.\d\d)?)",
        ],
        "gbp": [
            r"(\d+(?:\.\d\d)?)\s*(фу́нт(?!" + RUSSIAN_CHAR + r")|£)",
            r"£(\d+(?:\.\d\d)?)",
        ],
        "jpy": [
            r"(\d+(?:\.\d\d)?)\s*(ие́н(?!" + RUSSIAN_CHAR + r")|¥)",
            r"¥(\d+(?:\.\d\d)?)",
        ],
        "uah": [
            r"(\d+(?:\.\d\d)?)\s*(гри́в(ен|на|ны)?(?!" + RUSSIAN_CHAR + r")|₴)",
            r"₴(\d+(?:\.\d\d)?)",
        ],
        "kzt": [
            r"(\d+(?:\.\d\d)?)\s*(тенге́(?!" + RUSSIAN_CHAR + r")|₸)",
            r"₸(\d+(?:\.\d\d)?)",
        ],
    }
    for code, pats in patterns.items():
        (main_units, main_fem), (sub_units, sub_fem) = CURRENCIES[code]
        for pat in pats:
            for m in re.finditer(pat, text):
                amount = float(m.group(1))
                main_n, sub_n = int(amount), int(round((amount - int(amount)) * 100))
                main_words = number_to_words(main_n)
                if main_fem:
                    main_words = " ".join(_feminine_last(main_words.split()))
                result = f"{main_words} {_plural(main_n, main_units)}"
                if sub_n > 0:
                    sub_words = number_to_words(sub_n)
                    if sub_fem:
                        sub_words = " ".join(_feminine_last(sub_words.split()))
                    result += f" {sub_words} {_plural(sub_n, sub_units)}"
                text = text.replace(m.group(0), result)
    return text


# --- Шаг 24: Отрицательные числа ---
def normalize_negatives(text: str) -> str:
    return re.sub(r"(?:(?<=^)|(?<=[\s(\[]))[-−](\d)", r"ми́нус \1", text)


# --- Шаг 25: Числа ---
def normalize_numbers(words: list, gaps: list) -> tuple:
    digit_words = [
        "но́ль",
        "оди́н",
        "два́",
        "три́",
        "четы́ре",
        "пя́ть",
        "ше́сть",
        "се́мь",
        "во́семь",
        "де́вять",
    ]

    # Предлоги для определения падежа
    prep_cases = {
        "в": "prep",
        "во": "prep",
        "на": "prep",
        "о": "prep",
        "об": "prep",
        "при": "prep",
        "к": "dat",
        "ко": "dat",
        "по": "dat",
        "с": "gen",
        "со": "gen",
        "от": "gen",
        "до": "gen",
        "из": "gen",
        "у": "gen",
        "для": "gen",
        "без": "gen",
        "между": "instr",
    }

    i = 0
    while i < len(words):
        w = words[i]
        if w.isdigit():
            n = int(w)
            next_word = words[i + 1] if i + 1 < len(words) else ""
            prev_word = words[i - 1] if i > 0 else ""

            case = None

            # Порядковое только если есть предлог
            prev_clean = (
                "".join(
                    c for c in unicodedata.normalize("NFD", prev_word.lower())
                    if not unicodedata.combining(c)
                ) if prev_word else ""
            )
            if prev_clean in prep_cases:
                case = prep_cases[prev_clean]
                # "по" + "год" (им.п.) → винительный (= именительный)
                if prev_clean == "по" and next_word == "год":
                    case = "nom_m"

            if case:
                replacement = ordinal_text(n, case)
                replacement = replacement.replace("одна́ ты́сяча", "ты́сяча")

            elif len(w) > 1 and w[0] == "0":
                replacement = " ".join(digit_words[int(d)] for d in w)
            else:
                replacement = number_to_words(n)

            words[i] = replacement
        i += 1
    return words, gaps


# --- Шаг 26: Транслитерация (отключена) ---
def cyrillize(text: str) -> str:
    return text


# --- Шаг 27: Греческий ---
def normalize_greek(text: str) -> str:
    text = re.sub(
        r"\b([Α-Ωα-ω])\.(?!\w)",
        lambda m: GREEK_LETTER_NAMES.get(m.group(1), m.group(1)) + ".",
        text,
    )
    return "".join(GREEK_TO_RUSSIAN.get(ch, ch) for ch in text)


# --- Шаг 28: Языковые теги ---
def normalize_language_tags(text: str, acro_dict) -> str:
    latin_block = re.compile(
        r"(?<![а-яё" + COMBINING_DIACRITICAL + r"\w])"
        r'((?:[a-zA-ZäöüßÄÖÜ]{1,}[\s.,;:!?/&\-–—«»"\'()]*)+)'
        r"(?![а-яё" + COMBINING_DIACRITICAL + r"\w])"
    )

    def process_block(m):
        block = m.group(1).rstrip()
        if not block or not re.search(r"[a-zA-Z]{3,}", block):
            return m.group(0)
        processed = block
        # Одиночные буквы с точкой: только если не инициалы
        processed = re.sub(
            r"\b([B-DF-HJ-NP-TV-Z])\.(?!\s*[A-Z]\.)",
            lambda m: EN_LETTER_NAMES.get(m.group(1), m.group(1).lower()),
            processed,
        )
        # Одиночные буквы без точки (не инициалы)
        processed = re.sub(
            r"\b([B-DF-HJ-NP-TV-Z])\b(?!\.)",
            lambda m: EN_LETTER_NAMES.get(m.group(1), m.group(1).lower()),
            processed,
        )
        words = re.findall(r"\b[A-Z]{2,}\b", block)
        for word in words:
            pron = acro_dict.get_pronunciation(word, "en")
            if pron:
                processed = processed.replace(word, pron)
        processed = transliterate_latin_diacritics(processed)
        if processed.strip():
            return f"{TAG_EN_OPEN}{processed.strip()}{TAG_EN_CLOSE}"
        return m.group(0)

    return latin_block.sub(process_block, text)


# --- Шаг 29: Латинская диакритика ---
def transliterate_latin_diacritics(text: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFC", text)
    return "".join(LATIN_DIACRITICS_MAP.get(ch, ch) for ch in text)
