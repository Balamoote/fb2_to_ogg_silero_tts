#!/usr/bin/env python3
"""
Нормализация FB2 текста перед синтезом речи.
Версия: 3.5.0 (модульная)

Использование:
  python normalize_fb2.py book.fb2
"""

VERSION = "3.5.0"

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import regex as re

# Локальные модули
from normfb2 import (
    number_to_words, _feminine_last, _plural,
    NormalizationLogger, StatsCollector, find_replacements, apply_step,
    roman_to_int, is_valid_roman, ordinal_text,
    ROMAN_VALUES, ROMAN_STOPLIST, CYRILLIC_TO_LATIN_ROMAN,
    RE_ROMAN, RE_ROMAN_VALID, _ORDINAL_BASE, ORDINAL_CASES,
    AcronymDict, replace_diacritics, DIACRITIC_MAP,
    normalize_typography, normalize_web, normalize_regex_dict,
    normalize_abbreviations, normalize_sections, normalize_number_groups,
    normalize_ranges, normalize_dates, normalize_decimal_hyphen,
    normalize_compounds, normalize_time, normalize_fractions,
    normalize_percent, normalize_multipliers, normalize_measurements,
    normalize_ordinal_suffixes, normalize_acronyms, normalize_rulers,
    normalize_greek, normalize_alphanumeric, normalize_case_context,
    restore_protected, normalize_symbols, normalize_decimals,
    normalize_currency, normalize_negatives, normalize_numbers,
    cyrillize, transliterate_latin_diacritics, normalize_language_tags,
    COMBINING_DIACRITICAL, RUSSIAN_CHAR, TAG_EN_OPEN, TAG_EN_CLOSE,
    PROTECT_MARKER_PREFIX, PROTECT_MARKER_SUFFIX,
)

# Константы
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

GREEK_TO_RUSSIAN = None

logger = logging.getLogger(__name__)

FB2_NS = "http://www.gribuser.ru/xml/fictionbook/2.0"
XLINK_NS = "http://www.w3.org/1999/xlink"

# Выключатели шагов
STEP_ENABLED = {
    1: True, 2: True, 3: True, 4: True, 5: True,
    6: True, 7: True, 8: True, "8.5": True, 9: True, 10: True,
    11: True, 12: True, 13: True, 14: False, 15: True,
    16: True, 17: True, 18: True, 19: True, 20: True,
    21: True, 22: True, 23: True, 24: True, 25: True,
    26: False, 27: True, 28: True, 29: True,
}

STEP_NAMES = {
    1: "Типографика", 2: "URL/email", 3: "RegEx", 4: "Аббревиатуры",
    5: "Структурные ссылки", 6: "Группы цифр", 7: "Диапазоны", 8: "Даты",
    "8.5": "Десятичные с дефисом",
    9: "Составные числительные", 10: "Время", 11: "Дроби", 12: "Проценты",
    13: "Множители (тыс/млн)", 14: "Единицы измерения", 15: "Порядковые с суффиксами",
    16: "Акронимы и римские цифры", 17: "Имена правителей", 18: "Буквенно-цифровые",
    19: "Падежное согласование", 20: "Восстановление маркеров", 21: "Спецсимволы",
    22: "Десятичные дроби", 23: "Валюта", 24: "Отрицательные числа", 25: "Числа",
    26: "Транслитерация", 27: "Греческий->русский", 28: "Языковые теги",
    29: "Латинская диакритика",
}


# Встроенные словари
def _read_tsv(name: str):
    tsv = globals()[f"_{name.upper()}_TSV"]
    return [l.strip() for l in tsv.splitlines() if l.strip() and not l.strip().startswith("#")]

def normalize_russian_full(
    text: str, acro_dict: Optional[AcronymDict] = None,
    disabled_steps: Optional[Set] = None, stats: Optional[StatsCollector] = None,
    normalization_logger: Optional[NormalizationLogger] = None,
) -> str:
    if not text or not text.strip():
        return text
    if disabled_steps is None:
        disabled_steps = set()
    if stats is None:
        stats = StatsCollector(normalization_logger)

    disabled_steps = disabled_steps | {n for n, enabled in STEP_ENABLED.items() if not enabled}
    off = disabled_steps

    text = apply_step(text, normalize_typography, 1 in off, stats, "01.Типографика")
    text = apply_step(text, normalize_web, 2 in off, stats, "02.URL/email")
    if acro_dict:
        text = apply_step(text, normalize_regex_dict, 3 in off, stats, "03.RegEx", acro_dict)
    if acro_dict:
        text, protected = (normalize_acronyms(text, acro_dict) if 16 not in off else (text, []))
        if 16 not in off and protected:
            stats.add("16.Акронимы", len(protected))
    else:
        protected = []
    text = apply_step(text, normalize_abbreviations, 4 in off, stats, "04.Аббревиатуры")
    text = apply_step(text, normalize_sections, 5 in off, stats, "05.Структ.ссылки")
    text = apply_step(text, normalize_number_groups, 6 in off, stats, "06.Группы цифр")
    text = apply_step(text, normalize_ranges, 7 in off, stats, "07.Диапазоны")
    text = apply_step(text, normalize_dates, 8 in off, stats, "08.Даты")
    text = apply_step(text, normalize_decimal_hyphen, "8.5" in off, stats, "08.5.Десятичные с дефисом")
    text = apply_step(text, normalize_compounds, 9 in off, stats, "09.Составные числ.")
    text = apply_step(text, normalize_time, 10 in off, stats, "10.Время")
    text = apply_step(text, normalize_fractions, 11 in off, stats, "11.Дроби")
    text = apply_step(text, normalize_percent, 12 in off, stats, "12.Проценты")
    text = apply_step(text, normalize_multipliers, 13 in off, stats, "13.Множители")
    text = apply_step(text, normalize_measurements, 14 in off, stats, "14.Ед.измерения")
    text = apply_step(text, normalize_ordinal_suffixes, 15 in off, stats, "15.Порядковые суфф.")

    if acro_dict:
        text = apply_step(text, normalize_rulers, 17 in off, stats, "17.Имена правителей", acro_dict)

    text = apply_step(text, normalize_greek, 27 in off, stats, "27.Греческий->русский")
    text = apply_step(text, normalize_alphanumeric, 18 in off, stats, "18.Буквенно-цифровые")
    text = apply_step(text, normalize_case_context, 19 in off, stats, "19.Падежное согл.")

    if protected:
        old_text = text
        text = restore_protected(text, protected)
        if text != old_text:
            stats.add("20.Восст.маркеров", len(protected))

    text = apply_step(text, normalize_symbols, 21 in off, stats, "21.Спецсимволы")
    text = apply_step(text, normalize_decimals, 22 in off, stats, "22.Десятичные")
    text = apply_step(text, normalize_currency, 23 in off, stats, "23.Валюта")
    text = apply_step(text, normalize_negatives, 24 in off, stats, "24.Отрицательные")
    text = apply_step(text, normalize_numbers, 25 in off, stats, "25.Числа")
    text = apply_step(text, cyrillize, 26 in off, stats, "26.Транслитерация")
    text = apply_step(text, lambda t: transliterate_latin_diacritics(t), 29 in off, stats, "29.Латинская диакритика")
    if acro_dict:
        text = apply_step(text, normalize_language_tags, 28 in off, stats, "28.Языковые теги", acro_dict)

    text = re.sub(r" {2,}", " ", text)
    return text

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

STEP_NAMES = {
    1: "Типографика", 2: "URL/email", 3: "RegEx", 4: "Аббревиатуры",
    5: "Структурные ссылки", 6: "Группы цифр", 7: "Диапазоны", 8: "Даты",
    "8.5": "Десятичные с дефисом",
    9: "Составные числительные", 10: "Время", 11: "Дроби", 12: "Проценты",
    13: "Множители (тыс/млн)", 14: "Единицы измерения", 15: "Порядковые с суффиксами",
    16: "Акронимы и римские цифры", 17: "Имена правителей", 18: "Буквенно-цифровые",
    19: "Падежное согласование", 20: "Восстановление маркеров", 21: "Спецсимволы",
    22: "Десятичные дроби", 23: "Валюта", 24: "Отрицательные числа", 25: "Числа",
    26: "Транслитерация", 27: "Греческий->русский", 28: "Языковые теги",
    29: "Латинская диакритика",
}


def normalize_russian_full(
    text: str, acro_dict: Optional[AcronymDict] = None,
    disabled_steps: Optional[Set] = None, stats: Optional[StatsCollector] = None,
    normalization_logger: Optional[NormalizationLogger] = None,
) -> str:
    if not text or not text.strip():
        return text
    if disabled_steps is None:
        disabled_steps = set()
    if stats is None:
        stats = StatsCollector(normalization_logger)

    disabled_steps = disabled_steps | {n for n, enabled in STEP_ENABLED.items() if not enabled}
    off = disabled_steps

    text = apply_step(text, normalize_typography, 1 in off, stats, "01.Типографика")
    text = apply_step(text, normalize_web, 2 in off, stats, "02.URL/email")
    if acro_dict:
        text = apply_step(text, normalize_regex_dict, 3 in off, stats, "03.RegEx", acro_dict)
    # Акронимы по словарю (без побуквенной автогенерации)
    if acro_dict:
        text, protected = (normalize_acronyms(text, acro_dict) if 16 not in off else (text, []))
        if 16 not in off and protected:
            stats.add("16.Акронимы", len(protected))
    else:
        protected = []
    text = apply_step(text, normalize_abbreviations, 4 in off, stats, "04.Аббревиатуры")
    text = apply_step(text, normalize_sections, 5 in off, stats, "05.Структ.ссылки")
    text = apply_step(text, normalize_number_groups, 6 in off, stats, "06.Группы цифр")
    text = apply_step(text, normalize_ranges, 7 in off, stats, "07.Диапазоны")
    text = apply_step(text, normalize_dates, 8 in off, stats, "08.Даты")
    text = apply_step(text, normalize_decimal_hyphen, "8.5" in off, stats, "08.5.Десятичные с дефисом")
    text = apply_step(text, normalize_compounds, 9 in off, stats, "09.Составные числ.")
    text = apply_step(text, normalize_time, 10 in off, stats, "10.Время")
    text = apply_step(text, normalize_fractions, 11 in off, stats, "11.Дроби")
    text = apply_step(text, normalize_percent, 12 in off, stats, "12.Проценты")
    text = apply_step(text, normalize_multipliers, 13 in off, stats, "13.Множители")
    text = apply_step(text, normalize_measurements, 14 in off, stats, "14.Ед.измерения")
    text = apply_step(text, normalize_ordinal_suffixes, 15 in off, stats, "15.Порядковые суфф.")

    if acro_dict:
        text = apply_step(text, normalize_rulers, 17 in off, stats, "17.Имена правителей", acro_dict)

    text = apply_step(text, normalize_greek, 27 in off, stats, "27.Греческий->русский")
    text = apply_step(text, normalize_alphanumeric, 18 in off, stats, "18.Буквенно-цифровые")
    text = apply_step(text, normalize_case_context, 19 in off, stats, "19.Падежное согл.")

    if protected:
        old_text = text
        text = restore_protected(text, protected)
        if text != old_text:
            stats.add("20.Восст.маркеров", len(protected))

    text = apply_step(text, normalize_symbols, 21 in off, stats, "21.Спецсимволы")
    text = apply_step(text, normalize_decimals, 22 in off, stats, "22.Десятичные")
    text = apply_step(text, normalize_currency, 23 in off, stats, "23.Валюта")
    text = apply_step(text, normalize_negatives, 24 in off, stats, "24.Отрицательные")
    text = apply_step(text, normalize_numbers, 25 in off, stats, "25.Числа")
    text = apply_step(text, cyrillize, 26 in off, stats, "26.Транслитерация")
    # Шаг 29 должен быть до шага 28, чтобы диакритика нормализовалась до оборачивания в <tts_en>
    text = apply_step(text, lambda t: transliterate_latin_diacritics(t), 29 in off, stats, "29.Латинская диакритика")
    if acro_dict:
        text = apply_step(text, normalize_language_tags, 28 in off, stats, "28.Языковые теги", acro_dict)

    text = re.sub(r" {2,}", " ", text)
    return text


# ============================================================================
# FB2 ОБРАБОТКА
# ============================================================================


def md5_file(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_cache_path(input_file: str) -> Path:
    return Path(input_file).with_suffix(".norm_cache.json")


def load_cache(cache_path: Path) -> dict:
    return json.load(open(cache_path, "r")) if cache_path.exists() else {}


def save_cache(cache_path: Path, cache: dict):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def extract_texts_from_fb2(input_file: str) -> Tuple[ET.ElementTree, List[Dict]]:
    tree = ET.parse(input_file)
    root = tree.getroot()
    texts = []
    def walk(element, path=""):
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if element.text and element.text.strip():
            texts.append({"element": element, "type": "text", "text": element.text, "path": f"{path}/{tag}"})
        for child in element:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            walk(child, f"{path}/{tag}")
        if element.tail and element.tail.strip():
            texts.append({"element": element, "type": "tail", "text": element.tail, "path": f"{path}/{tag}"})
    for body in root.findall(f"{{{FB2_NS}}}body"):
        body_name = body.get("name", "")
        path = "/FictionBook/body[notes]" if body_name == "notes" else "/FictionBook/body"
        walk(body, path)
    return tree, texts


def normalize_fb2(
    input_file: str, output_file: str, config: dict,
    force: bool = False, disabled_steps: Optional[Set] = None,
    enable_logging: bool = True, log_dir: Optional[Path] = None,
) -> bool:
    start_time = time.time()
    cache_path = get_cache_path(input_file)
    cache = load_cache(cache_path)
    input_md5 = md5_file(input_file)

    if not force and cache.get("source_md5") == input_md5:
        if cache.get("normalized_file") and os.path.exists(cache["normalized_file"]):
            output_md5 = md5_file(cache["normalized_file"])
            if cache.get("normalized_md5") == output_md5:
                print(f"{GREEN}✓ Файл уже нормализован (MD5 совпадает){RESET}")
                return True

    acro_dict = AcronymDict(config)
    normalization_logger = NormalizationLogger(log_dir) if enable_logging else None

    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} {BOLD}FB2 Normalizer v{VERSION}{RESET}")
    print(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}{CYAN}║{RESET} {GREEN}Вход:{RESET}  {Path(input_file).name}")
    print(f"{BOLD}{CYAN}║{RESET} {GREEN}Выход:{RESET} {Path(output_file).name}")
    if enable_logging:
        print(f"{BOLD}{CYAN}║{RESET} {GREEN}Логи:{RESET}  {normalization_logger.log_dir}")
    if disabled_steps:
        print(f"{BOLD}{CYAN}║{RESET} {YELLOW}Отключены шаги:{RESET} {', '.join(str(s) for s in sorted(disabled_steps, key=lambda x: (str(x).replace('.',''), str(x))))}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}\n")

    print(f"{YELLOW}Извлечение текста из FB2...{RESET}")
    tree, texts = extract_texts_from_fb2(input_file)
    print(f"  Найдено текстовых элементов: {len(texts)}")

    print(f"{YELLOW}Нормализация...{RESET}")
    stats = StatsCollector(normalization_logger)
    changed = 0
    total = len(texts)

    for i, item in enumerate(texts):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (total - i)
            sys.stderr.write(f"\r  Прогресс: {i}/{total} ({i*100//total}%) | Прошло: {elapsed:.0f}с | Осталось: {eta:.0f}с  ")
            sys.stderr.flush()
        original = item["text"]
        normalized = normalize_russian_full(original, acro_dict, disabled_steps, stats, normalization_logger)
        if normalized != original:
            changed += 1
            if item["type"] == "text":
                item["element"].text = normalized
            else:
                item["element"].tail = normalized

    sys.stderr.write(f"\r  Прогресс: {total}/{total} (100%) | Изменено элементов: {changed}\n")
    sys.stderr.flush()

    if normalization_logger:
        normalization_logger.total_processed = total
        normalization_logger.total_changed = changed

    stats.print_report()

    if normalization_logger and enable_logging:
        print(f"\n{YELLOW}Сохранение логов нормализации...{RESET}")
        normalization_logger.log_stats()

    print(f"\n{YELLOW}Сохранение нормализованного FB2...{RESET}")
    ET.register_namespace("", FB2_NS)
    xml_str = ET.tostring(tree.getroot(), encoding="unicode")
    xml_str = xml_str.replace("&lt;tts_en&gt;", "<tts_en>")
    xml_str = xml_str.replace("&lt;/tts_en&gt;", "</tts_en>")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n' + xml_str)

    output_md5 = md5_file(output_file)
    cache = {
        "version": VERSION, "source_file": input_file, "source_md5": input_md5,
        "normalized_file": output_file, "normalized_md5": output_md5,
        "changed_elements": changed, "total_elements": total,
        "disabled_steps": list(disabled_steps) if disabled_steps else [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - start_time,
    }
    save_cache(cache_path, cache)

    elapsed = time.time() - start_time
    print(f"\n{GREEN}✓ Готово за {elapsed:.1f} сек{RESET}")
    print(f"  Изменено элементов: {YELLOW}{changed}{RESET} из {YELLOW}{total}{RESET}")
    if normalization_logger:
        print(f"  Логи сохранены в: {CYAN}{normalization_logger.log_dir}{RESET}")
    return True


def dry_run(input_file: str, config: dict, disabled_steps: Optional[Set] = None):
    acro_dict = AcronymDict(config)
    _, texts = extract_texts_from_fb2(input_file)
    print(f"\n{BOLD}Предпросмотр нормализации (первые 30 изменений):{RESET}\n")
    if disabled_steps:
        print(f"{YELLOW}Отключены шаги: {', '.join(str(s) for s in sorted(disabled_steps, key=lambda x: (str(x).replace('.',''), str(x))))}{RESET}\n")
    stats = StatsCollector()
    shown = 0
    for item in texts:
        original = item["text"]
        normalized = normalize_russian_full(original, acro_dict, disabled_steps, stats)
        if normalized != original:
            shown += 1
            print(f"{YELLOW}[{shown}]{RESET} {item['path']}")
            print(f"  {RED}-{RESET} {original[:150]}{'...' if len(original) > 150 else ''}")
            print(f"  {GREEN}+{RESET} {normalized[:150]}{'...' if len(normalized) > 150 else ''}")
            print()
            if shown >= 30:
                break
    print(f"{'Показано' if shown else 'Изменений не найдено.'} {shown} изменений. Всего: {len(texts)}")
    stats.print_report()


def print_steps_help():
    print(f"\n{BOLD}Шаги нормализации:{RESET}")
    for num in sorted(STEP_NAMES, key=lambda x: (str(x).replace('.', ''), str(x))):
        print(f"  {CYAN}{str(num):>4}{RESET} — {STEP_NAMES[num]}")
    print(f"\n  {YELLOW}--N-off{RESET} — отключить шаг N (например --14-off)")
    print(f"  {YELLOW}--list-steps{RESET} — показать этот список\n")


def find_new_acronyms(input_file: str, config: dict):
    print(f"Поиск акронимов в: {input_file}")
    acro_dict = AcronymDict(config)
    _, texts = extract_texts_from_fb2(input_file)
    all_text = " ".join(item["text"] for item in texts)
    ru_new = sorted(set(re.findall(r"\b[А-ЯЁ]{2,}\b", all_text)) - set(acro_dict.acro_ru.keys()))
    en_new = sorted(set(re.findall(r"\b[A-Z]{2,}\b", all_text)) - set(acro_dict.acro_en.keys()))
    script_dir = Path(__file__).parent
    for name, items in [("acro_ru_new.txt", ru_new), ("acro_en_new.txt", en_new)]:
        if items:
            with open(script_dir / name, "w", encoding="utf-8") as f:
                for a in items:
                    f.write(f"{a}\n")
            print(f"Новых акронимов: {len(items)} → {name}")


def main():
    parser = argparse.ArgumentParser(description=f"Нормализация FB2 текста v{VERSION}")
    parser.add_argument("input", help="Входной FB2 файл")
    parser.add_argument("-o", "--output", help="Выходной файл")
    parser.add_argument("--force", action="store_true", help="Принудительная нормализация")
    parser.add_argument("--dry-run", action="store_true", help="Показать изменения без сохранения")
    parser.add_argument("--find-acro", action="store_true", help="Найти новые акронимы")
    parser.add_argument("--list-steps", action="store_true", help="Показать список шагов")
    parser.add_argument("--debug", action="store_true", help="Отладочный вывод")
    parser.add_argument("--version", action="version", version=f"%(prog)s v{VERSION}")
    parser.add_argument("--no-logs", action="store_true", help="Отключить сохранение логов")
    parser.add_argument("--log-dir", help="Директория для сохранения логов")
    for num in STEP_NAMES:
        parser.add_argument(f"--{num}-off", action="store_true", help=f"Отключить шаг {num}: {STEP_NAMES[num]}")

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.list_steps:
        print_steps_help()
        return 0
    if not os.path.exists(args.input):
        print(f"{RED}Файл не найден: {args.input}{RESET}")
        return 1

    disabled_steps = set()
    for num in STEP_NAMES:
        if getattr(args, f"{num}_off", False):
            disabled_steps.add(num)

    config = {"acro_ru_dict": "acro_ru.gz", "acro_en_dict": "acro_en.gz"}

    if args.dry_run:
        dry_run(args.input, config, disabled_steps)
        return 0
    if args.find_acro:
        find_new_acronyms(args.input, config)
        return 0

    output_file = args.output or str(Path(args.input).with_suffix(".norm.fb2"))
    log_dir = Path(args.log_dir) if args.log_dir else None

    try:
        return 0 if normalize_fb2(args.input, output_file, config, force=args.force, disabled_steps=disabled_steps, enable_logging=not args.no_logs, log_dir=log_dir) else 1
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Прервано{RESET}")
        return 1
    except Exception as e:
        print(f"{RED}Ошибка: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
