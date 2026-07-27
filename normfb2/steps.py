"""
Шаги нормализации 1-29
"""

import re as re_module
import regex as re
import unicodedata
import os
from typing import Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path

# Константы из основного модуля
COMBINING_DIACRITICAL = "\u0300-\u036f"
RUSSIAN_CHAR = f"[а-яё{COMBINING_DIACRITICAL}]"
TAG_EN_OPEN = "<tts_en>"
TAG_EN_CLOSE = "</tts_en>"
PROTECT_MARKER_PREFIX = "ЪЪЪьЪЪЪ"
PROTECT_MARKER_SUFFIX = "ЬЬЬъЬЬЬ"

from normfb2.dicts import AcronymDict, replace_diacritics
from normfb2.roman import (
    roman_to_int, is_valid_roman, ordinal_text,
    ROMAN_VALUES, ROMAN_STOPLIST, CYRILLIC_TO_LATIN_ROMAN,
    RE_ROMAN, RE_ROMAN_VALID,
)

# Встроенные словари (нужны для шагов 4 и 14)
def _read_tsv(name: str):
    tsv = globals()[f"_{name.upper()}_TSV"]
    return [l.strip() for l in tsv.splitlines() if l.strip() and not l.strip().startswith("#")]

from normfb2.utils import (
    number_to_words, _feminine_last, _plural, _read_tsv, _ABBREVIATIONS_TSV, _MEASUREMENTS_TSV,
    NormalizationLogger, StatsCollector, find_replacements, apply_step,
)

# ============================================================================
# ФУНКЦИИ НОРМАЛИЗАЦИИ
# ============================================================================





# --- Шаг 1: Типографика ---
def normalize_typography(text: str) -> str:
    text = re.sub(r"[\u00a0\u2009\u202f\u2060]", " ", text)
    text = re.sub(r"\*\*|__|`", "", text)
    return text


# --- Шаг 2: URL и email ---
def normalize_web(text: str) -> str:
    web_symbols = {"@": " соба́ка ", ".": " то́чка ", "/": " слэ́ш ", ":": " двоето́чие ", "-": " дефи́с ", "_": " подчё́ркивание "}
    def spell(m):
        s = m.group(0).rstrip(".,!?")
        for sym, word in web_symbols.items():
            s = s.replace(sym, word)
        return re.sub(r" {2,}", " ", s).strip()
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[A-Za-zА-Яа-я]{2,}\b", spell, text)
    text = re.sub(r"\b(?:https?://|www\.)\S+|\b[\w-]+\.(?:com|ru|org|net|info|io|edu|gov|рф)\b", spell, text, flags=re.I)
    text = re.sub(r"#([A-Za-zА-Яа-яёЁ0-9_]+)", r"хештег \1", text)
    return text


# --- Шаг 3: RegEx замены ---
def normalize_regex_dict(text: str, acro_dict: AcronymDict) -> str:
    """Применяет regex-замены из словаря. \b учитывает ударения."""
    for pattern, replacement in acro_dict.regex_dict.items():
        try:
            p = pattern
            lb = r'(?<![а-яё\w' + '́̀-ͯ' + r'])' 
            la = r'(?![а-яё\w' + '́̀-ͯ' + r'])' 
            if p.startswith('\\b'):
                p = lb + p[2:]
            if p.endswith('\\b'):
                p = p[:-2] + la
            if re.search(p, text):
                text = re.sub(p, replacement, text)
        except re.error:
            pass
    return text


# --- Шаг 4: Аббревиатуры ---
def normalize_abbreviations(text: str) -> str:
    abbr_map = {}
    for line in _read_tsv("abbreviations"):
        if "\t" in line:
            key, value = line.split("\t", 1)
            abbr_map[re.sub(r"\s+", "", key).lower()] = value.strip()
    if not abbr_map:
        return text
    keys = sorted({l.split("\t", 1)[0] for l in _read_tsv("abbreviations") if "\t" in l}, key=len, reverse=True)
    def to_pattern(k):
        return "".join(r"\.\s*" if ch == "." else (r"\s*" if ch == " " else re.escape(ch)) for ch in k)
    pattern = r"(?<![" + RUSSIAN_CHAR[1:-1] + r"])(?:" + "|".join(to_pattern(k) for k in keys) + r")(?![" + RUSSIAN_CHAR[1:-1] + r"])"
    abbr_re = re.compile(pattern, re.IGNORECASE)
    def repl(m):
        canon = re.sub(r"\s+", "", m.group(0)).lower()
        return abbr_map.get(canon, m.group(0))
    return abbr_re.sub(repl, text)


# --- Шаг 5: Структурные ссылки ---
def normalize_sections(text: str) -> str:
    section_abbr = {"ст": "статья́", "пп": "подпу́нкт", "п": "пу́нкт", "рис": "рису́нок", "табл": "табли́ца", "гл": "глава́"}
    return re.sub(r"(?<![" + RUSSIAN_CHAR[1:-1] + r"])(ст|пп|табл|рис|гл|п)\.\s*(?=\d)", lambda m: section_abbr[m.group(1).lower()] + " ", text, flags=re.I)


# --- Шаг 6: Группы цифр ---
def normalize_number_groups(text: str) -> str:
    return re.sub(r"\b\d{1,3}(?: \d{3})+\b", lambda m: m.group(0).replace(" ", ""), text)


# --- Шаг 7: Диапазоны ---
def normalize_ranges(text: str) -> str:
    text = re.sub(r"\b(\d{3,4})\s*[-–—]\s*(\d{3,4})\s*(?:гг\.?|годы)(?![" + RUSSIAN_CHAR[1:-1] + r"])",
                  lambda m: f"{number_to_words(int(m.group(1)))} {number_to_words(int(m.group(2)))} го́ды", text)
    return re.sub(r"(?<=\d)\s*[–—]\s*(?=\d)", " ", text)


# --- Шаг 8: Даты ---
def normalize_dates(text: str) -> str:
    months_gen = ("января́", "февраля́", "ма́рта", "апре́ля", "ма́я", "ию́ня", "ию́ля", "а́вгуста", "сентября́", "октября́", "ноября́", "декабря́")
    month_by_num = {f"{i:02d}": m for i, m in enumerate(months_gen, start=1)}
    def date_numeric(m):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        mn = month_by_num.get(f"{month:02d}")
        if not mn:
            return m.group(0)
        return f"{ordinal_text(day, 'gen')} {mn} {number_to_words(year)} го́да"
    text = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", date_numeric, text)
    text = re.sub(r"\b(\d{4})\s*г\.(?!" + RUSSIAN_CHAR + r")", lambda m: f"{number_to_words(int(m.group(1)))} го́д", text)
    return text


# --- Шаг 8.5: Десятичные дроби с дефисом ---
def normalize_decimal_hyphen(text: str) -> str:
    """7,62-миллиметровая → семь и шестьдесят две сотых миллиметровая"""
    def repl(m):
        integer_part = int(m.group(1))
        frac_part = m.group(2)
        suffix = m.group(3)
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
        if n == 1: prefix = "одно́"
        elif n == 2: prefix = "дву́х"
        elif n == 3: prefix = "трё́х"
        elif n == 4: prefix = "четырё́х"
        elif prefix.endswith("ть"): prefix = prefix[:-1] + "и"
        suffix_part = re.sub(r'[' + COMBINING_DIACRITICAL + r']', '', m.group(2))
        return prefix + suffix_part
    return re.sub(r"\b(\d+)-([" + RUSSIAN_CHAR[1:-1] + r"]{2,}(?:ий|ый|ой|ая|яя|ое|ее|ые|ие|ого|его|ому|ему|ым|им|ом|ем)[" + COMBINING_DIACRITICAL + r"]*)\b", repl, text)


# --- Шаг 10: Время ---
def normalize_time(text: str) -> str:
    def repl(m):
        h, mn = int(m.group(1)), int(m.group(2))
        h_word = "ча́с" if h == 1 else f"{number_to_words(h)} {_plural(h, ['ча́с', 'часа́', 'часо́в'])}"
        if mn == 0:
            return h_word
        mn_word = f"{' '.join(_feminine_last(number_to_words(mn).split()))} {_plural(mn, ['мину́та', 'мину́ты', 'мину́т'])}"
        return f"{h_word} {mn_word}"
    return re.sub(r"(?<![\d:])(\d{1,2}):([0-5]\d)(?![\d:])", repl, text)


# --- Шаг 11: Дроби ---
def normalize_fractions(text: str) -> str:
    vulgar = {"½": (1, 2), "⅓": (1, 3), "¼": (1, 4), "¾": (3, 4)}
    for ch, (num, den) in vulgar.items():
        text = text.replace(ch, f"{number_to_words(num)} {number_to_words(den)}")
    return text


# --- Шаг 12: Проценты ---
def normalize_percent(text: str) -> str:
    def repl(m):
        n = int(m.group(1))
        return f"{number_to_words(n)} {_plural(n, ['проце́нт', 'проце́нта', 'проце́нтов'])}"
    return re.sub(r"(\d+)\s*%", repl, text)


# --- Шаг 13: Множители ---
def normalize_multipliers(text: str) -> str:
    multipliers = {
        "тыс": (["ты́сяча", "ты́сячи", "ты́сяч"], True),
        "млн": (["миллио́н", "миллио́на", "миллио́нов"], False),
        "млрд": (["миллиа́рд", "миллиа́рда", "миллиа́рдов"], False),
    }
    def repl(m):
        forms, feminine = multipliers[m.group(2).lower()]
        n = int(m.group(1))
        words = number_to_words(n).split()
        if feminine:
            _feminine_last(words)
        return " ".join(words) + " " + _plural(n, forms)
    return re.sub(r"\b(\d+)\s*(тыс|млн|млрд)\.?(?!" + RUSSIAN_CHAR + r")", repl, text, flags=re.I)


# --- Шаг 14: Единицы измерения ---
def normalize_measurements(text: str) -> str:
    units = {}
    for line in _read_tsv("measurements"):
        parts = line.split("\t")
        if len(parts) == 5:
            units[parts[0]] = (parts[1], parts[2], parts[3], parts[4])
    if not units:
        return text
    unit_alt = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    re_measure = re.compile(r"(?<![\d.,])(\d+)\s*(" + unit_alt + r")\.?(?!" + RUSSIAN_CHAR + r"|[A-Za-z])")
    def repl(m):
        one, few, many, gender = units[m.group(2)]
        n = int(m.group(1))
        words = number_to_words(n).split()
        if gender == "f":
            _feminine_last(words)
        return " ".join(words) + " " + _plural(n, (one, few, many))
    return re_measure.sub(repl, text) if units else text


# --- Шаг 15: Порядковые числительные с суффиксами ---
def normalize_ordinal_suffixes(text: str) -> str:
    suffix_forms = {"й": "nom_m", "го": "gen", "му": "dat", "м": "prep", "я": "nom_f", "ю": "acc_f", "е": "nom_pl", "х": "pl"}
    def repl(m):
        n = int(m.group(1))
        suffix_with_accent = m.group(2)
        suffix_clean = "".join(c for c in unicodedata.normalize("NFD", suffix_with_accent) if not unicodedata.combining(c))
        case = suffix_forms.get(suffix_clean, "nom_m")
        return ordinal_text(n, case)
    suffix_patterns = []
    for suffix in suffix_forms.keys():
        suffix_with_accents = ""
        for char in suffix:
            suffix_with_accents += char + "[" + COMBINING_DIACRITICAL + "]?"
        suffix_patterns.append(suffix_with_accents)
    return re.sub(r"(\d+)[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE63\uFF0D](" + "|".join(suffix_patterns) + r")(?![а-яё" + COMBINING_DIACRITICAL + r"])", repl, text)


# --- Шаг 16: Акронимы и римские цифры ---
def normalize_acronyms(text: str, acro_dict: AcronymDict) -> Tuple[str, List[str]]:
    protected = []
    def protect(value: str) -> str:
        idx = len(protected)
        protected.append(value)
        return f"{PROTECT_MARKER_PREFIX}{idx}{PROTECT_MARKER_SUFFIX}"
    # Русские акронимы — поиск по точным ключам словаря с границами слова
    if acro_dict.acro_ru:
        ru_keys = sorted(acro_dict.acro_ru.keys(), key=len, reverse=True)
        ru_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in ru_keys) + r')\b'
        def ru_acro(m):
            word = m.group(0)
            pron = acro_dict.get_pronunciation(word, "ru")
            return protect(pron) if pron else word
        text = re.sub(ru_pattern, ru_acro, text)
    # Английские акронимы — поиск по точным ключам словаря с границами слова
    if acro_dict.acro_en:
        en_keys = sorted(acro_dict.acro_en.keys(), key=len, reverse=True)
        en_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in en_keys) + r')\b'
        def en_acro(m):
            word = m.group(0)
            if RE_ROMAN_VALID.match(word.upper()) and word.upper() not in ROMAN_STOPLIST:
                return word
            pron = acro_dict.get_pronunciation(word, "en")
            return protect(pron) if pron else word
        text = re.sub(en_pattern, en_acro, text)
    return text, protected


# --- Шаг 17: Имена правителей с римскими цифрами ---
def normalize_rulers(text: str, acro_dict: AcronymDict) -> str:
    case_endings = {
        "gen": ("а", "я", "ы", "и"), "dat": ("у", "ю", "е", "и"),
        "acc": ("а", "я", "у", "ю"), "instr": ("ом", "ем", "ём", "ой", "ей", "ёй", "ою", "ею"), "prep": ("е", "и"),
    }
    def detect_case(name: str) -> str:
        name_clean = "".join(c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c))
        for case, endings in case_endings.items():
            for ending in endings:
                if name_clean.endswith(ending):
                    return case
        return "nom"
    def repl(m):
        name, roman = m.group(1), m.group(2)
        ruler_form = acro_dict.get_ruler_form(name)
        if not ruler_form:
            return m.group(0)
        gender = ruler_form.get("gender", "mas")
        import unicodedata
        name_clean = ''.join(c for c in unicodedata.normalize('NFD', name.lower()) if not unicodedata.combining(c))
        case_map = {"nom": "nom", "gen": "gen", "acc": "acc", "dat": "dat", "ins": "instr", "prep": "prep"}
        detected = None
        for form_key, case_key in case_map.items():
            form_val = ruler_form.get(form_key, '')
            form_clean = ''.join(c for c in unicodedata.normalize('NFD', form_val.lower()) if not unicodedata.combining(c))
            if name_clean == form_clean:
                detected = case_key
                break
        if not detected:
            detected = detect_case(name)
        if detected == "nom":
            detected = "nom_f" if gender == "fem" else "nom_m"
        elif detected == "acc":
            detected = "acc_f" if gender == "fem" else "acc_m"
        return f"{name} {ordinal_text(roman_to_int(roman), detected)}"
    text = re.sub(r"\b([А-ЯЁа-яё̀-ͯ]+)\s+([IVXLCDM]+)\b", repl, text)
    def roman_repl(m):
        roman_str = m.group(1)
        if roman_str.upper() in ROMAN_STOPLIST or not is_valid_roman(roman_str):
            return m.group(0)
        if len(roman_str) == 1:
            after = text[m.end():].lstrip()
            if re.match(r'\.', after):
                return m.group(0)
            if re.match(r'\d', after):
                return str(roman_to_int(roman_str))
            if re.match(r'в\.', after):
                return ordinal_text(roman_to_int(roman_str), "nom_m")
            if not after or after[0] in '.!?;:,':
                return str(roman_to_int(roman_str))
            return m.group(0)
        num = roman_to_int(roman_str)
        after = text[m.end():].lstrip()
        # Склонение по слову после римской цифры
        if re.match(r'ве́ка\b', after):
            return ordinal_text(num, "gen")
        if re.match(r'ве́ку\b', after):
            return ordinal_text(num, "dat")
        if re.match(r'ве́ке\b', after):
            return ordinal_text(num, "prep")
        if re.match(r'ве́ком\b', after):
            return ordinal_text(num, "instr")
        if re.match(r'веко́в\b', after):
            return ordinal_text(num, "pl")
        # Для дефисных конструкций с "веко́в" — второе число тоже pl
        if re.match(r'[-–—-]', after):
            # Проверим, что дальше идёт "веко́в"
            rest = text[m.end():].lstrip()
            rest_after_dash = re.sub(r'^[-–—-]\s*[IVXLCDM]+\s*', '', rest)
            if re.match(r'веко́в\b', rest_after_dash):
                return ordinal_text(num, "pl")
            return ordinal_text(num, "nom_m")
        if re.match(r'(век|века́|в\.|вв\.)\b', after):
            return ordinal_text(num, "nom_m")
        # "в." с точкой в конце
        if re.match(r'в\.', after):
            return ordinal_text(num, "nom_m")
        # "вв." с точкой в конце
        if re.match(r'вв\.', after):
            return ordinal_text(num, "nom_m")
        # Для дефисных конструкций (VI-VII) — второе число тоже порядковое
        if re.match(r'[-–—-]', after):
            return ordinal_text(num, "nom_m")
        if re.match(r'[-–—-]', after):
            return ordinal_text(num, "nom_m")
        return str(num)
    text = RE_ROMAN.sub(roman_repl, text)
    return text


# --- Шаг 18: Буквенно-цифровые обозначения ---
def normalize_alphanumeric(text: str) -> str:
    # Буквы произносятся как в аббревиатурах (побуквенно)
    letter_names = {
        'а': 'а́', 'б': 'бэ́', 'в': 'вэ́', 'г': 'гэ́', 'д': 'дэ́', 'е': 'е́', 'ё': 'ё́',
        'ж': 'жэ́', 'з': 'зэ́', 'и': 'и́', 'й': 'и кра́ткое', 'к': 'ка́', 'л': 'э́ль',
        'м': 'э́м', 'н': 'э́н', 'о': 'о́', 'п': 'пэ́', 'р': 'э́р', 'с': 'э́с', 'т': 'тэ́',
        'у': 'у́', 'ф': 'э́ф', 'х': 'ха́', 'ц': 'цэ́', 'ч': 'че́', 'ш': 'ша́', 'щ': 'ща́',
        'ъ': 'твё́рдый зна́к', 'ы': 'ы́', 'ь': 'мя́гкий зна́к', 'э': 'э́', 'ю': 'ю́', 'я': 'я́',
        'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
        'g': 'джи', 'h': 'эйч', 'i': 'ай', 'j': 'джей', 'k': 'кей', 'l': 'эл',
        'm': 'эм', 'n': 'эн', 'o': 'оу', 'p': 'пи', 'q': 'кью', 'r': 'ар',
        's': 'эс', 't': 'ти', 'u': 'ю', 'v': 'ви', 'w': 'дабл ю', 'x': 'экс',
        'y': 'уай', 'z': 'зед',
    }
    
    def spell_letters(letters: str) -> str:
        return ' '.join(letter_names.get(ch.lower(), ch.lower()) for ch in letters)
    
    # Инициалы: пары и тройки объединяем через дефис (А. Б. → а́-бэ́)
    def spell_initials(m):
        matched = m.group(0)
        # Сохраняем хвостовой пробел если есть
        trailing = ''
        if matched.endswith(' '):
            trailing = ' '
            matched = matched.rstrip()
        letters = re.findall(r'([А-ЯЁа-яё])\.', matched)
        return '-'.join(spell_letters(l) for l in letters) + trailing
    text = re.sub(
        r'\b(?:[А-ЯЁа-яё]\.\s*){2,}',
        spell_initials,
        text,
    )
    # Оставшиеся одиночные буквы с точкой: М. → эм.
    text = re.sub(
        r"\b([А-ЯЁа-яё]{1})\.(?!\w)",
        lambda m: f"{spell_letters(m.group(1))}.",
        text,
    )
    # Одиночные буквы без точки не трогаем
    # Буквы-цифры через дефис: А-123 → а сто двадцать три
    text = re.sub(
        r"\b([А-ЯЁа-яё" + COMBINING_DIACRITICAL + r"]{1,6})[-–—](\d+)\b",
        lambda m: f"{spell_letters(m.group(1))} {number_to_words(int(m.group(2)))}",
        text,
    )
    # Буквы перед цифрами: А123 → а сто двадцать три
    text = re.sub(
        r"\b([А-ЯЁа-яё" + COMBINING_DIACRITICAL + r"]{1,4})(\d+)\b",
        lambda m: f"{spell_letters(m.group(1))} {number_to_words(int(m.group(2)))}",
        text,
    )
    # Цифры перед буквами: 123с → сто двадцать три, эс
    text = re.sub(
        r"\b(\d+)([А-ЯЁа-яё" + COMBINING_DIACRITICAL + r"]{1,4})\b",
        lambda m: f"{number_to_words(int(m.group(1)))}, {spell_letters(m.group(2))}",
        text,
    )
    return text


# --- Шаг 19: Падежное согласование ---
def normalize_case_context(text: str) -> str:
    return text


# --- Шаг 20: Восстановление маркеров ---
def restore_protected(text: str, protected: List[str]) -> str:
    for i, value in enumerate(protected):
        marker = f"{PROTECT_MARKER_PREFIX}{i}{PROTECT_MARKER_SUFFIX}"
        text = text.replace(marker, value)
    return text


# --- Шаг 21: Спецсимволы ---
def normalize_symbols(text: str) -> str:
    symbols = {"°C": "гра́дусов це́льсия", "°С": "гра́дусов це́льсия", "±": "плю́с ми́нус", "≈": "приблизи́тельно равно́", "≠": "не́ равно́", "×": "умно́жить на́", "÷": "раздели́ть на́", "§": "пара́граф", "№": "но́мер", "&": "и́", "²": "в ква́драте", "³": "в ку́бе"}
    for sym, word in symbols.items():
        text = text.replace(sym, " " + word + " ")
    return re.sub(r" {2,}", " ", text)


# --- Шаг 22: Десятичные дроби ---
def normalize_decimals(text: str) -> str:
    decimal_places = {1: "деся́тых", 2: "со́тых", 3: "ты́сячных", 4: "десятиты́сячных"}
    def repl(m):
        ip, fp = m.group(1), m.group(2)
        if set(fp) == {"0"}:
            return number_to_words(int(ip))
        place = decimal_places.get(len(fp))
        if not place:
            return m.group(0)
        int_words = _feminine_last(number_to_words(int(ip)).split())
        whole = "це́лая" if (int(ip) % 10 == 1 and int(ip) % 100 != 11) else "це́лых"
        frac_words = _feminine_last(number_to_words(int(fp)).split())
        return f"{' '.join(int_words)} {whole} и́ {' '.join(frac_words)} {place}"
    return re.sub(r"\b(\d+),(\d+)\b", repl, text)


# --- Шаг 23: Валюта ---
def normalize_currency(text: str) -> str:
    currencies = {
        "rub": ((["ру́бль", "рубля́", "рубле́й"], False), (["копе́йка", "копе́йки", "копе́ек"], True)),
        "usd": ((["до́ллар", "до́ллара", "до́лларов"], False), (["це́нт", "це́нта", "це́нтов"], False)),
        "eur": ((["е́вро", "е́вро", "е́вро"], False), (["евроце́нт", "евроце́нта", "евроце́нтов"], False)),
    }
    patterns = {
        "rub": [r"(\d+(?:\.\d\d)?)\s*((руб|ру́б)(л(е́й|я́|ь))?(?!" + RUSSIAN_CHAR + r")|₽)"],
        "usd": [r"(\d+(?:\.\d\d)?)\s*(до́ллар(о́в|а|ы)?(?!" + RUSSIAN_CHAR + r")|\$)", r"\$(\d+(?:\.\d\d)?)"],
        "eur": [r"(\d+(?:\.\d\d)?)\s*(е́вро(?!" + RUSSIAN_CHAR + r")|€)", r"€(\d+(?:\.\d\d)?)"],
    }
    for code, pats in patterns.items():
        (main_units, main_fem), (sub_units, sub_fem) = currencies[code]
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
def normalize_numbers(text: str) -> str:
    digit_words = ["но́ль", "оди́н", "два́", "три́", "четы́ре", "пя́ть", "ше́сть", "се́мь", "во́семь", "де́вять"]
    for m in reversed(list(re.finditer(r"\b\d+\b", text))):
        digits = m.group(0)
        n = int(digits)
        replacement = " ".join(digit_words[int(d)] for d in digits) if (len(digits) > 1 and digits[0] == "0") else number_to_words(n)
        text = text[:m.start()] + replacement + text[m.end():]
    return text


# --- Шаг 26: Транслитерация (отключена) ---
def cyrillize(text: str) -> str:
    return text


GREEK_TO_RUSSIAN = None

def _init_greek_map():
    global GREEK_TO_RUSSIAN
    if GREEK_TO_RUSSIAN is not None:
        return
    GREEK_TO_RUSSIAN = {
        # Строчные без тона
        '\u03b1': '\u0430', '\u03b2': '\u0432', '\u03b3': '\u0433', '\u03b4': '\u0434', '\u03b5': '\u0435',
        '\u03b6': '\u0437', '\u03b7': '\u0438', '\u03b8': '\u0444', '\u03b9': '\u0438', '\u03ba': '\u043a',
        '\u03bb': '\u043b', '\u03bc': '\u043c', '\u03bd': '\u043d', '\u03be': '\u043a\u0441', '\u03bf': '\u043e',
        '\u03c0': '\u043f', '\u03c1': '\u0440', '\u03c3': '\u0441', '\u03c2': '\u0441', '\u03c4': '\u0442',
        '\u03c5': '\u0438', '\u03c6': '\u0444', '\u03c7': '\u0445', '\u03c8': '\u043f\u0441', '\u03c9': '\u043e',
        # Строчные с тоном (ά έ ή ί ό ύ ώ)
        '\u03ac': '\u0430\u0301', '\u03ad': '\u0435\u0301', '\u03ae': '\u0438\u0301', '\u03af': '\u0438\u0301',
        '\u03cc': '\u043e\u0301', '\u03cd': '\u0438\u0301', '\u03ce': '\u043e\u0301',
        # Строчные с облегчённым придыханием (ἀ ἁ ἐ ἑ ἠ ἡ ἰ ἱ ὀ ὁ ὐ ὑ ὠ ὡ)
        '\u1f00': '\u0430', '\u1f01': '\u0430', '\u1f10': '\u0435', '\u1f11': '\u0435',
        '\u1f20': '\u0438', '\u1f21': '\u0438', '\u1f30': '\u0438', '\u1f31': '\u0438',
        '\u1f40': '\u043e', '\u1f41': '\u043e', '\u1f50': '\u0438', '\u1f51': '\u0438',
        '\u1f60': '\u043e', '\u1f61': '\u043e',
        # Строчные с придыханием и тоном (ἄ ἅ ἔ ἕ ἤ ἥ ἴ ἵ ὄ ὅ ὔ ὕ ὤ ὥ)
        '\u1f04': '\u0430\u0301', '\u1f05': '\u0430\u0301', '\u1f14': '\u0435\u0301', '\u1f15': '\u0435\u0301',
        '\u1f24': '\u0438\u0301', '\u1f25': '\u0438\u0301', '\u1f34': '\u0438\u0301', '\u1f35': '\u0438\u0301',
        '\u1f44': '\u043e\u0301', '\u1f45': '\u043e\u0301', '\u1f54': '\u0438\u0301', '\u1f55': '\u0438\u0301',
        '\u1f64': '\u043e\u0301', '\u1f65': '\u043e\u0301',
        # Строчные с йотой подписной (ᾳ ῃ ῳ)
        '\u1fb3': '\u0430', '\u1fc3': '\u0438', '\u1ff3': '\u043e',
        # Строчные с йотой и тоном (ᾴ ῄ ῴ)
        '\u1fb4': '\u0430\u0301', '\u1fc4': '\u0438\u0301', '\u1ff4': '\u043e\u0301',
        # Строчные с диалитикой (ϊ ϋ)
        '\u03ca': '\u0438', '\u03cb': '\u0438',
        # Строчные с диалитикой и тоном (ΐ ΰ)
        '\u0390': '\u0438\u0301', '\u03b0': '\u0438\u0301',
        # Заглавные без тона
        '\u0391': '\u0410', '\u0392': '\u0412', '\u0393': '\u0413', '\u0394': '\u0414', '\u0395': '\u0415',
        '\u0396': '\u0417', '\u0397': '\u0418', '\u0398': '\u0424', '\u0399': '\u0418', '\u039a': '\u041a',
        '\u039b': '\u041b', '\u039c': '\u041c', '\u039d': '\u041d', '\u039e': '\u041a\u0441', '\u039f': '\u041e',
        '\u03a0': '\u041f', '\u03a1': '\u0420', '\u03a3': '\u0421', '\u03a4': '\u0422', '\u03a5': '\u0418',
        '\u03a6': '\u0424', '\u03a7': '\u0425', '\u03a8': '\u041f\u0441', '\u03a9': '\u041e',
        # Заглавные с тоном (Ά Έ Ή Ί Ό Ύ Ώ)
        '\u0386': '\u0410\u0301', '\u0388': '\u0415\u0301', '\u0389': '\u0418\u0301', '\u038a': '\u0418\u0301',
        '\u038c': '\u041e\u0301', '\u038e': '\u0418\u0301', '\u038f': '\u041e\u0301',
        # Заглавные с придыханием (Ἀ Ἁ Ἐ Ἑ Ἠ Ἡ Ἰ Ἱ Ὀ Ὁ Ὑ Ὠ Ὡ)
        '\u1f08': '\u0410', '\u1f09': '\u0410', '\u1f18': '\u0415', '\u1f19': '\u0415',
        '\u1f28': '\u0418', '\u1f29': '\u0418', '\u1f38': '\u0418', '\u1f39': '\u0418',
        '\u1f48': '\u041e', '\u1f49': '\u041e', '\u1f59': '\u0418', '\u1f68': '\u041e', '\u1f69': '\u041e',
        # Заглавные с придыханием и тоном (Ἄ Ἅ Ἔ Ἕ Ἤ Ἥ Ἴ Ἵ Ὄ Ὅ Ὕ Ὤ Ὥ)
        '\u1f0c': '\u0410\u0301', '\u1f0d': '\u0410\u0301', '\u1f1c': '\u0415\u0301', '\u1f1d': '\u0415\u0301',
        '\u1f2c': '\u0418\u0301', '\u1f2d': '\u0418\u0301', '\u1f3c': '\u0418\u0301', '\u1f3d': '\u0418\u0301',
        '\u1f4c': '\u041e\u0301', '\u1f4d': '\u041e\u0301', '\u1f5d': '\u0418\u0301',
        '\u1f6c': '\u041e\u0301', '\u1f6d': '\u041e\u0301',
        # Заглавные с йотой подписной (ᾼ ῌ ῼ)
        '\u1fbc': '\u0410', '\u1fcc': '\u0418', '\u1ffc': '\u041e',
        # Заглавные с диалитикой (Ϊ Ϋ)
        '\u03aa': '\u0418', '\u03ab': '\u0418',
    }


def normalize_greek(text: str) -> str:
    _init_greek_map()
    # Греческие инициалы: Ω. → оме́га
    greek_letter_names = {
        'Α': 'а́льфа', 'α': 'а́льфа', 'Β': 'бе́та', 'β': 'бе́та',
        'Γ': 'га́мма', 'γ': 'га́мма', 'Δ': 'де́льта', 'δ': 'де́льта',
        'Ε': 'э́псилон', 'ε': 'э́псилон', 'Ζ': 'дзе́та', 'ζ': 'дзе́та',
        'Η': 'э́та', 'η': 'э́та', 'Θ': 'те́та', 'θ': 'те́та',
        'Ι': 'йо́та', 'ι': 'йо́та', 'Κ': 'ка́ппа', 'κ': 'ка́ппа',
        'Λ': 'ля́мбда', 'λ': 'ля́мбда', 'Μ': 'мю', 'μ': 'мю',
        'Ν': 'ню', 'ν': 'ню', 'Ξ': 'кси', 'ξ': 'кси',
        'Ο': 'оми́крон', 'ο': 'оми́крон', 'Π': 'пи', 'π': 'пи',
        'Ρ': 'ро', 'ρ': 'ρο', 'Σ': 'си́гма', 'σ': 'си́гма', 'ς': 'си́гма',
        'Τ': 'та́у', 'τ': 'та́у', 'Υ': 'ю́псилон', 'υ': 'ю́псилон',
        'Φ': 'фи', 'φ': 'фи', 'Χ': 'хи', 'χ': 'хи',
        'Ψ': 'пси', 'ψ': 'пси', 'Ω': 'оме́га', 'ω': 'оме́га',
    }
    text = re.sub(
        r'\b([Α-Ωα-ω])\.(?!\w)',
        lambda m: greek_letter_names.get(m.group(1), m.group(1)) + '.',
        text,
    )
    return ''.join(GREEK_TO_RUSSIAN.get(ch, ch) for ch in text)


def normalize_language_tags(text: str, acro_dict) -> str:
    """Оборачивает латинские блоки в <tts_en>, группируя слова и разделители."""
    # Ищем непрерывные блоки: буквы + пробелы + знаки препинания (.,;:!?/&-)
    # с условием что хотя бы одно слово из 3+ букв
    latin_block = re.compile(
        r'(?<![а-яё' + COMBINING_DIACRITICAL + r'\w])'
        r'((?:[a-zA-Z]{1,}[\s.,;:!?/&\-–—«»"\'()]*)+)'
        r'(?![а-яё' + COMBINING_DIACRITICAL + r'\w])'
    )
    
    def process_block(m):
        block = m.group(1).rstrip()
        if not block:
            return m.group(0)
        # Проверяем, есть ли хоть одно слово из 3+ букв
        if not re.search(r'[a-zA-Z]{3,}', block):
            return m.group(0)
        # Обрабатываем аббревиатуры внутри блока
        words = re.findall(r'\b[A-Z]{2,}\b', block)
        processed = block
        for word in words:
            pron = acro_dict.get_pronunciation(word, "en")
            if pron:
                processed = processed.replace(word, pron)
        processed = transliterate_latin_diacritics(processed)
        if processed.strip():
            return f"{TAG_EN_OPEN}{processed.strip()}{TAG_EN_CLOSE}"
        return m.group(0)
    
    return latin_block.sub(process_block, text)


def transliterate_latin_diacritics(text: str) -> str:
    # Нормализуем декомпозированные символы (O + combining acute → Ó)
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    m = {
        '\u00c0': 'A', '\u00c1': 'A', '\u00c2': 'A', '\u00c3': 'A', '\u00c4': 'A', '\u00c5': 'A', '\u00c6': 'AE', '\u00c7': 'C',
        '\u00c8': 'E', '\u00c9': 'E', '\u00ca': 'E', '\u00cb': 'E', '\u00cc': 'I', '\u00cd': 'I', '\u00ce': 'I', '\u00cf': 'I',
        '\u00d0': 'D', '\u00d1': 'N', '\u00d2': 'O', '\u00d3': 'O', '\u00d4': 'O', '\u00d5': 'O', '\u00d6': 'O', '\u00d8': 'O',
        '\u00d9': 'U', '\u00da': 'U', '\u00db': 'U', '\u00dc': 'U', '\u00dd': 'Y', '\u00de': 'Th', '\u00df': 'ss',
        '\u00e0': 'a', '\u00e1': 'a', '\u00e2': 'a', '\u00e3': 'a', '\u00e4': 'a', '\u00e5': 'a', '\u00e6': 'ae', '\u00e7': 'c',
        '\u00e8': 'e', '\u00e9': 'e', '\u00ea': 'e', '\u00eb': 'e', '\u00ec': 'i', '\u00ed': 'i', '\u00ee': 'i', '\u00ef': 'i',
        '\u00f0': 'd', '\u00f1': 'n', '\u00f2': 'o', '\u00f3': 'o', '\u00f4': 'o', '\u00f5': 'o', '\u00f6': 'o', '\u00f8': 'o',
        '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u00fc': 'u', '\u00fd': 'y', '\u00ff': 'y',
        '\u0100': 'A', '\u0101': 'a', '\u0102': 'A', '\u0103': 'a', '\u0104': 'A', '\u0105': 'a',
        '\u0106': 'C', '\u0107': 'c', '\u0108': 'C', '\u0109': 'c', '\u010a': 'C', '\u010b': 'c', '\u010c': 'C', '\u010d': 'c',
        '\u010e': 'D', '\u010f': 'd', '\u0110': 'D', '\u0111': 'd',
        '\u0112': 'E', '\u0113': 'e', '\u0114': 'E', '\u0115': 'e', '\u0116': 'E', '\u0117': 'e', '\u0118': 'E', '\u0119': 'e', '\u011a': 'E', '\u011b': 'e',
        '\u011c': 'G', '\u011d': 'g', '\u011e': 'G', '\u011f': 'g',
        '\u0120': 'G', '\u0121': 'g', '\u0122': 'G', '\u0123': 'g',
        '\u0124': 'H', '\u0125': 'h', '\u0126': 'H', '\u0127': 'h',
        '\u0128': 'I', '\u0129': 'i', '\u012a': 'I', '\u012b': 'i', '\u012c': 'I', '\u012d': 'i', '\u012e': 'I', '\u012f': 'i', '\u0130': 'I', '\u0131': 'i',
        '\u0134': 'J', '\u0135': 'j',
        '\u0136': 'K', '\u0137': 'k',
        '\u0139': 'L', '\u013a': 'l', '\u013b': 'L', '\u013c': 'l', '\u013d': 'L', '\u013e': 'l', '\u013f': 'L', '\u0140': 'l', '\u0141': 'L', '\u0142': 'l',
        '\u0143': 'N', '\u0144': 'n', '\u0145': 'N', '\u0146': 'n', '\u0147': 'N', '\u0148': 'n',
        '\u014c': 'O', '\u014d': 'o', '\u014e': 'O', '\u014f': 'o', '\u0150': 'O', '\u0151': 'o',
        '\u0152': 'OE', '\u0153': 'oe',
        '\u0154': 'R', '\u0155': 'r', '\u0156': 'R', '\u0157': 'r', '\u0158': 'R', '\u0159': 'r',
        '\u015a': 'S', '\u015b': 's', '\u015c': 'S', '\u015d': 's', '\u015e': 'S', '\u015f': 's', '\u0160': 'S', '\u0161': 's',
        '\u0162': 'T', '\u0163': 't', '\u0164': 'T', '\u0165': 't',
        '\u0168': 'U', '\u0169': 'u', '\u016a': 'U', '\u016b': 'u', '\u016c': 'U', '\u016d': 'u', '\u016e': 'U', '\u016f': 'u', '\u0170': 'U', '\u0171': 'u',
        '\u0172': 'U', '\u0173': 'u',
        '\u0174': 'W', '\u0175': 'w',
        '\u0176': 'Y', '\u0177': 'y', '\u0178': 'Y',
        '\u0179': 'Z', '\u017a': 'z', '\u017b': 'Z', '\u017c': 'z', '\u017d': 'Z', '\u017e': 'z',
    }
    return ''.join(m.get(ch, ch) for ch in text)


