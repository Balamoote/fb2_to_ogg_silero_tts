"""
Модуль нормализации FB2 текста
"""

from normfb2.utils import (
    number_to_words, _feminine_last, _plural,
    tokenize_words_gaps, detokenize_words_gaps,
    NormalizationLogger, StatsCollector, find_replacements, apply_step,

)

from normfb2.dicts import (
    AcronymDict, replace_diacritics, DIACRITIC_MAP,
)

from normfb2.roman import (
    roman_to_int, is_valid_roman, ordinal_text,
    ROMAN_VALUES, ROMAN_STOPLIST, CYRILLIC_TO_LATIN_ROMAN,
    RE_ROMAN, RE_ROMAN_VALID,
)

from normfb2.data import ORDINAL_CASES
from normfb2.data import (
    CASE_ENDINGS, CURRENCIES, DECIMAL_PLACES, EN_LETTER_NAMES,
    GREEK_LETTER_NAMES, GREEK_TO_RUSSIAN, LATIN_DIACRITICS_MAP,
    LETTER_NAMES, MONTHS_GEN, MULTIPLIERS, SECTION_ABBR,
    SUFFIX_FORMS, SYMBOLS, VULGAR_FRACTIONS, WEB_SYMBOLS,
)

from normfb2.steps import (
    COMBINING_DIACRITICAL, RUSSIAN_CHAR, TAG_EN_OPEN, TAG_EN_CLOSE,
    PROTECT_MARKER_PREFIX, PROTECT_MARKER_SUFFIX,
    normalize_typography, normalize_web, normalize_regex_dict,
    normalize_sections, normalize_number_groups,
    normalize_ranges, normalize_dates, normalize_decimal_hyphen,
    normalize_compounds, normalize_time, normalize_fractions,
    normalize_percent, normalize_multipliers, normalize_measurements, normalize_measurements_1letter,
    normalize_ordinal_suffixes, normalize_acronyms, normalize_rulers,
    normalize_greek, normalize_alphanumeric, normalize_case_context,
    restore_protected, normalize_symbols, normalize_decimals,
    normalize_currency, normalize_negatives, normalize_numbers,
    cyrillize, transliterate_latin_diacritics, normalize_language_tags,
)