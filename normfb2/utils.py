"""
Вспомогательные функции: числа, логирование, статистика
"""

import csv
import json
import re
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Цвета для вывода
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================


def number_to_words(n: int) -> str:
    if n == 0:
        return "но́ль"
    units = ["", "оди́н", "два́", "три́", "четы́ре", "пя́ть", "ше́сть", "се́мь", "во́семь", "де́вять"]
    teens = ["де́сять", "оди́ннадцать", "двена́дцать", "трина́дцать", "четы́рнадцать", "пятна́дцать", "шестна́дцать", "семна́дцать", "восемна́дцать", "девятна́дцать"]
    tens = ["", "де́сять", "два́дцать", "три́дцать", "со́рок", "пятьдеся́т", "шестьдеся́т", "се́мьдесят", "во́семьдесят", "девяно́сто"]
    hundreds = ["", "сто́", "две́сти", "три́ста", "четы́реста", "пятьсо́т", "шестьсо́т", "семьсо́т", "восемьсо́т", "девятьсо́т"]
    scales = [
        (10**15, ["квадриллио́н", "квадриллио́на", "квадриллио́нов"], False),
        (10**12, ["триллио́н", "триллио́на", "триллио́нов"], False),
        (10**9, ["миллиа́рд", "миллиа́рда", "миллиа́рдов"], False),
        (10**6, ["миллио́н", "миллио́на", "миллио́нов"], False),
        (10**3, ["ты́сяча", "ты́сячи", "ты́сяч"], True),
    ]

    def plural(num, forms):
        if num % 10 == 1 and num % 100 != 11:
            return forms[0]
        if 2 <= num % 10 <= 4 and not 12 <= num % 100 <= 14:
            return forms[1]
        return forms[2]

    def under_thousand(num):
        if num == 0:
            return []
        if num < 10:
            return [units[num]]
        if num < 20:
            return [teens[num - 10]]
        if num < 100:
            return [tens[num // 10]] + ([units[num % 10]] if num % 10 else [])
        return [hundreds[num // 100]] + under_thousand(num % 100)

    if n >= 10**18:
        digit_words = ["но́ль", "оди́н", "два́", "три́", "четы́ре", "пя́ть", "ше́сть", "се́мь", "во́семь", "де́вять"]
        return " ".join(digit_words[int(d)] for d in str(n))

    words = []
    for value, forms, feminine in scales:
        count = (n // value) % 1000
        if count:
            chunk = under_thousand(count)
            if feminine:
                if chunk[-1] == "оди́н":
                    chunk[-1] = "одна́"
                elif chunk[-1] == "два́":
                    chunk[-1] = "две́"
                if count == 1:
                    chunk = chunk[:-1]
            words += chunk + [plural(count, forms)]
    words += under_thousand(n % 1000)
    return " ".join(w for w in words if w)


def _feminine_last(words):
    if words and words[-1] == "оди́н":
        words[-1] = "одна́"
    elif words and words[-1] == "два́":
        words[-1] = "две́"
    return words


def _plural(n, forms):
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


# ============================================================================
# ЛОГИРОВАНИЕ
# ============================================================================


class NormalizationLogger:
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.replacements: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.counters: Dict[str, int] = defaultdict(int)
        self.total_processed = 0
        self.total_changed = 0
        self.en_phrases: List[str] = []
        self._clean_old_logs()

    def _clean_old_logs(self):
        if self.log_dir.exists():
            for f in self.log_dir.glob("normalization_*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def add_replacement(self, step: str, original: str, normalized: str, context: str = ""):
        self.replacements[step].append((original, normalized, context))
        self.counters[step] += 1

    def add_en_phrase(self, phrase: str):
        if phrase.strip():
            self.en_phrases.append(phrase.strip())

    def log_stats(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._write_detailed_log(timestamp)
        self._write_summary_json(timestamp)
        self._write_readable_report(timestamp)
        self._write_en_phrases_log(timestamp)

    def _write_detailed_log(self, timestamp: str):
        csv_path = self.log_dir / f"normalization_log_{timestamp}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["Шаг", "Оригинал", "Нормализация", "Контекст", "Длина ориг.", "Длина норм."])
            for step in sorted(self.replacements.keys(), key=lambda x: int(str(x).split('.')[0]) if str(x)[0].isdigit() else 999):
                for original, normalized, context in sorted(self.replacements[step], key=lambda x: len(x[0]), reverse=True):
                    writer.writerow([step, original.strip(), normalized.strip(), context.strip()[:100], len(original), len(normalized)])
        print(f"{GREEN}✓ Детальный лог сохранён: {csv_path}{RESET}")

    def _write_summary_json(self, timestamp: str):
        json_path = self.log_dir / f"normalization_summary_{timestamp}.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_processed": self.total_processed,
            "total_changed": self.total_changed,
            "change_percentage": round(self.total_changed / max(self.total_processed, 1) * 100, 2),
            "steps": {}
        }
        for step in sorted(self.replacements.keys(), key=lambda x: int(str(x).split('.')[0]) if str(x)[0].isdigit() else 999):
            replacements = self.replacements[step]
            replacement_freq = defaultdict(int)
            for orig, norm, ctx in replacements:
                replacement_freq[f"{orig} → {norm}"] += 1
            top_replacements = sorted(replacement_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            summary["steps"][step] = {
                "count": len(replacements),
                "unique_patterns": len(replacement_freq),
                "top_replacements": [{"pattern": pat, "count": cnt} for pat, cnt in top_replacements],
                "avg_original_length": round(sum(len(orig) for orig, _, _ in replacements) / len(replacements), 1) if replacements else 0,
                "avg_normalized_length": round(sum(len(norm) for _, norm, _ in replacements) / len(replacements), 1) if replacements else 0,
            }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"{GREEN}✓ Сводная статистика сохранена: {json_path}{RESET}")

    def _write_readable_report(self, timestamp: str):
        report_path = self.log_dir / f"normalization_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"ОТЧЁТ О НОРМАЛИЗАЦИИ ТЕКСТА\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Всего обработано элементов: {self.total_processed}\n")
            f.write(f"Изменено элементов: {self.total_changed}\n")
            f.write(f"Процент изменений: {round(self.total_changed / max(self.total_processed, 1) * 100, 2)}%\n\n")
            f.write("-" * 80 + "\n")
            f.write("СТАТИСТИКА ПО ШАГАМ (отсортировано по количеству замен)\n")
            f.write("-" * 80 + "\n\n")
            sorted_steps = sorted(self.replacements.items(), key=lambda x: len(x[1]), reverse=True)
            for step, replacements in sorted_steps:
                f.write(f"Шаг {step}\n")
                f.write(f"  Количество замен: {len(replacements)}\n")
                if replacements:
                    pattern_freq = defaultdict(list)
                    for orig, norm, ctx in replacements:
                        pattern_freq[f"{orig} → {norm}"].append(ctx)
                    sorted_patterns = sorted(pattern_freq.items(), key=lambda x: len(x[1]), reverse=True)[:5]
                    f.write(f"  Топ-5 замен:\n")
                    for i, (pattern, contexts) in enumerate(sorted_patterns, 1):
                        f.write(f"    {i}. {pattern} (встретилось {len(contexts)} раз)\n")
                        if contexts and contexts[0]:
                            f.write(f"       Контекст: ...{contexts[0][:80]}...\n")
                f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("ИТОГО ПО ВСЕМ ШАГАМ\n")
            f.write("-" * 80 + "\n")
            total_orig_len = sum(len(orig) for reps in self.replacements.values() for orig, _, _ in reps)
            total_norm_len = sum(len(norm) for reps in self.replacements.values() for _, norm, _ in reps)
            f.write(f"Суммарная длина оригиналов: {total_orig_len} символов\n")
            f.write(f"Суммарная длина после нормализации: {total_norm_len} символов\n")
            if total_orig_len > 0:
                f.write(f"Изменение размера: {round((total_norm_len - total_orig_len) / total_orig_len * 100, 2)}%\n")
        print(f"{GREEN}✓ Читаемый отчёт сохранён: {report_path}{RESET}")

    def _write_en_phrases_log(self, timestamp: str):
        if not self.en_phrases:
            return
        en_log_path = self.log_dir / f"tts_en_phrases_{timestamp}.txt"
        with open(en_log_path, "w", encoding="utf-8") as f:
            f.write(f"Английские фразы для озвучки (из <tts_en>)\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего фраз: {len(self.en_phrases)}\n")
            f.write("=" * 60 + "\n\n")
            for i, phrase in enumerate(self.en_phrases, 1):
                f.write(f"{i:4d}. {phrase}\n")
        print(f"{GREEN}✓ Список английских фраз сохранён: {en_log_path}{RESET}")


# ============================================================================
# СТАТИСТИКА
# ============================================================================


class StatsCollector:
    def __init__(self, logger: Optional[NormalizationLogger] = None):
        self.stats: Dict[str, int] = defaultdict(int)
        self.logger = logger

    def add(self, step: str, count: int = 1):
        self.stats[step] = self.stats.get(step, 0) + count

    def get(self, step: str) -> int:
        return self.stats.get(step, 0)

    def print_report(self):
        if not self.stats:
            print(f"{YELLOW}  Нет изменений{RESET}")
            return
        print(f"\n{BOLD}Статистика замен по шагам (отсортировано по количеству):{RESET}")
        sorted_steps = sorted(self.stats.items(), key=lambda x: x[1], reverse=True)
        total = sum(count for _, count in sorted_steps)
        max_width = max(len(str(name)) for name, _ in sorted_steps)
        for step_name, count in sorted_steps:
            bar_length = int(40 * count / max(total, 1))
            bar = "█" * bar_length + "░" * (40 - bar_length)
            percentage = round(count / max(total, 1) * 100, 1)
            print(f"  {CYAN}{step_name:<{max_width}}{RESET} {YELLOW}{count:>6}{RESET} {bar} {percentage:>5.1f}%")
        print(f"  {BOLD}{'─' * (max_width + 50)}{RESET}")
        print(f"  {BOLD}{'ВСЕГО':<{max_width}} {GREEN}{total:>6}{RESET}")



# Встроенные словари
_ABBREVIATIONS_TSV = """\
гг.	го́ды
р-н	райо́н
до́ н. э.	до́ на́шей э́ры
н. э.	на́шей э́ры
и т. д.	и та́к да́лее
и т. п.	и тому́ подо́бное
б/у	бы́вший в употребле́нии
и др.	и други́е
и пр.	и про́чие
т.е.	то́ е́сть
ру́б.	рубле́й
до́лл.	до́лларов
"""

_MEASUREMENTS_TSV = """\
км	киломе́тр	киломе́тра	киломе́тров	m
м	ме́тр	ме́тра	ме́тров	m
см	сантиме́тр	сантиме́тра	сантиме́тров	m
мм	миллиме́тр	миллиме́тра	миллиме́тров	m
кг	килогра́мм	килогра́мма	килогра́ммов	m
г	гра́мм	гра́мма	гра́ммов	m
л	ли́тр	ли́тра	ли́тров	m
ч	ча́с	часа́	часо́в	m
ми́н	мину́та	мину́ты	мину́т	f
се́к	секу́нда	секу́нды	секу́нд	f
га́	гекта́р	гекта́ра	гекта́ров	m
км/ч	киломе́тр в ча́с	киломе́тра в ча́с	киломе́тров в ча́с	m
м/с	ме́тр в секу́нду	ме́тра в секу́нду	ме́тров в секу́нду	m
Гц	ге́рц	ге́рца	ге́рц	m
Вт	ва́тт	ва́тта	ва́тт	m
В	во́льт	во́льта	во́льт	m
А	ампе́р	ампе́ра	ампе́р	m
°	гра́дус	гра́дуса	гра́дусов	m
"""

def _read_tsv(name: str):
    tsv = globals()[f"_{name.upper()}_TSV"]
    return [l.strip() for l in tsv.splitlines() if l.strip() and not l.strip().startswith("#")]

def find_replacements(original: str, normalized: str) -> list:
    """Находит различия между оригинальным и нормализованным текстом."""
    import regex as re
    import os
    replacements = []
    orig_words = re.findall(r"\S+", original)
    norm_words = re.findall(r"\S+", normalized)
    if len(orig_words) == len(norm_words):
        for i, (ow, nw) in enumerate(zip(orig_words, norm_words)):
            if ow != nw:
                start = max(0, i - 2)
                end = min(len(orig_words), i + 3)
                context = " ".join(orig_words[start:end])
                replacements.append((ow, nw, context))
    else:
        common_prefix_len = len(os.path.commonprefix([original, normalized]))
        context_start = max(0, common_prefix_len - 50)
        context = original[context_start:context_start + 100]
        replacements.append((original[:100], normalized[:100], context))
    return replacements


def apply_step(text: str, func, disabled: bool, stats, step_name: str, *args):
    """Применяет шаг нормализации, если он не отключён."""
    import regex as re
    if disabled:
        return text
    result = func(text, *args) if args else func(text)
    if result != text:
        old_words = set(re.findall(r"\S+", text))
        new_words = set(re.findall(r"\S+", result))
        changes = len(old_words - new_words) or 1
        stats.add(step_name, changes)
        if stats.logger:
            replacements = find_replacements(text, result)
            for orig, norm, ctx in replacements:
                stats.logger.add_replacement(step_name, orig, norm, ctx)
            if step_name == "28.Языковые теги":
                en_phrases = re.findall(r'<tts_en>(.*?)</tts_en>', result)
                for phrase in en_phrases:
                    stats.logger.add_en_phrase(phrase)
    return result
