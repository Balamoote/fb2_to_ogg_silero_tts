"""
Римские цифры, порядковые числительные
"""

import regex as re
from normfb2.utils import number_to_words

ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
ROMAN_STOPLIST = {
    "CD", "DVD", "MD", "DC", "MC", "MI", "MM",
    "DI", "DIV", "MIX", "CIV", "LCD", "IL", "IC", "ID", "IM",
}
CYRILLIC_TO_LATIN_ROMAN = {}

RE_ROMAN = re.compile(r"\b([IVXLCDM]+)\b", re.IGNORECASE)
RE_ROMAN_VALID = re.compile(r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$")

from normfb2.data import ORDINAL_CASES




def roman_to_int(s: str) -> int:
    s_normalized = s.upper()
    for cyr, lat in CYRILLIC_TO_LATIN_ROMAN.items():
        s_normalized = s_normalized.replace(cyr.upper(), lat)
    total = prev = 0
    for ch in reversed(s_normalized):
        v = ROMAN_VALUES.get(ch, 0)
        total += -v if v < prev else v
        prev = v
    return total


def is_valid_roman(s: str) -> bool:
    s_normalized = s.upper()
    for cyr, lat in CYRILLIC_TO_LATIN_ROMAN.items():
        s_normalized = s_normalized.replace(cyr.upper(), lat)
    return bool(RE_ROMAN_VALID.match(s_normalized))


def ordinal_text(n: int, case: str = "nom_m") -> str:
    if n in ORDINAL_CASES.get(case, {}):
        return ORDINAL_CASES[case][n]
    if n < 20:
        return number_to_words(n)
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones > 0:
            return f"{number_to_words(tens)} {ordinal_text(ones, case)}"
        return number_to_words(n)
    if n < 1000:
        hundreds = (n // 100) * 100
        rest = n % 100
        if rest > 0:
            return f"{number_to_words(hundreds)} {ordinal_text(rest, case)}"
        return ordinal_text(hundreds, case)
    # Большие числа: раскладываем на разряды, последний — порядковый
    parts = []
    scales = [
        (10**9, ["миллиа́рд", "миллиа́рда", "миллиа́рдов"], False),
        (10**6, ["миллио́н", "миллио́на", "миллио́нов"], False),
        (10**3, ["ты́сяча", "ты́сячи", "ты́сяч"], True),
    ]
    remaining = n
    for value, forms, feminine in scales:
        count = remaining // value
        if count > 0:
            count_str = number_to_words(count)
            if feminine:
                w = count_str.split()
                if w and w[-1] == "оди́н":
                    w[-1] = "одна́"
                elif w and w[-1] == "два́":
                    w[-1] = "две́"
                count_str = " ".join(w)
            parts.append(f"{count_str} {forms[0] if count % 10 == 1 and count % 100 != 11 else forms[1] if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14 else forms[2]}")
            remaining %= value
    if remaining > 0:
        parts.append(ordinal_text(remaining, case))
    elif parts:
        last = parts[-1]
        parts[-1] = ordinal_text(int(last.split()[0]), case) + " " + " ".join(last.split()[1:])
    return " ".join(parts)