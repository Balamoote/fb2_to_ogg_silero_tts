"""
Словари: акронимы, regex, правители, диакритика
"""

import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import regex as re

logger = logging.getLogger(__name__)

# ============================================================================
# ДИАКРИТИКА
# ============================================================================

DIACRITIC_MAP = {
    "Á": "а́", "á": "а́", "À": "а", "Â": "а", "Ã": "а", "Ā": "а",
    "à": "а", "â": "а", "ã": "а", "ā": "а",
    "É": "е́", "é": "е́", "È": "е", "Ê": "е", "Ë": "е", "Ē": "е",
    "è": "е", "ê": "е", "ë": "е", "ē": "е",
    "Í": "и́", "í": "и́", "Ì": "и", "Î": "и", "Ï": "и", "Ī": "и",
    "ì": "и", "î": "и", "ï": "и", "ī": "и",
    "Ñ": "нь", "ñ": "нь", "Ń": "нь", "ń": "нь",
    "Ó": "о́", "ó": "о́", "Ò": "о", "Ô": "о", "Õ": "о", "Ō": "о",
    "ò": "о", "ô": "о", "õ": "о", "ō": "о",
    "Ö": "ё", "ö": "ё", "Ø": "ё", "ø": "ё", "Ő": "ё", "ő": "ё",
    "Č": "ч", "č": "ч", "Ř": "рж", "ř": "рж",
    "Ś": "ш", "ś": "ш", "Ş": "ш", "ş": "ш", "Š": "ш", "š": "ш",
    "Ż": "ж", "Ž": "ж", "ż": "ж", "ž": "ж",
    "Α": "а", "α": "а", "ά": "а́",
    "Β": "б", "β": "б", "Γ": "г", "γ": "г",
    "Δ": "д", "δ": "д", "Ð": "д", "ð": "д",
    "Ε": "е", "ε": "е", "έ": "е́", "Є": "е", "є": "е",
    "Ζ": "з", "ζ": "з",
    "Η": "э", "η": "э", "ή": "э́", "Ä": "э", "ä": "э", "Æ": "э", "æ": "э",
    "Θ": "ф", "θ": "ф",
    "Ι": "и", "ι": "и", "І": "и", "і": "и",
    "Κ": "к", "κ": "к", "Λ": "л", "λ": "л", "Ł": "л", "ł": "л",
    "Μ": "м", "μ": "м", "Ν": "н", "ν": "н",
    "Ξ": "кс", "ξ": "кс", "Ο": "о", "ο": "о", "ό": "о́", "Å": "о", "å": "о",
    "Π": "п", "π": "п", "Ρ": "р", "ρ": "р",
    "Σ": "с", "σ": "с", "ς": "с", "Ç": "с", "ç": "с",
    "Τ": "т", "τ": "т",
    "Υ": "ю", "υ": "ю", "ύ": "ю́", "Ü": "ю", "ü": "ю",
    "Φ": "ф", "φ": "ф", "Χ": "х", "χ": "х",
    "Ψ": "кс", "ψ": "кс", "Ω": "о", "ω": "о",
    "Ї": "йи", "ї": "йи", "Ў": "у", "ў": "у",
}

def replace_diacritics(text: str) -> str:
    if not text:
        return text
    return "".join(DIACRITIC_MAP.get(ch, ch) for ch in text)


# ============================================================================
# СЛОВАРИ
# ============================================================================

class AcronymDict:
    ENGLISH_PHONETIC = {
        "A": "э́й", "B": "би́", "C": "си́", "D": "ди́", "E": "и́", "F": "э́ф",
        "G": "джи́", "H": "э́йч", "I": "а́й", "J": "дже́й", "K": "ке́й", "L": "э́л",
        "M": "э́м", "N": "э́н", "O": "о́у", "P": "пи́", "Q": "кью́", "R": "а́р",
        "S": "э́с", "T": "ти́", "U": "ю́", "V": "ви́", "W": "да́бл ю́", "X": "э́кс",
        "Y": "уа́й", "Z": "зе́д",
    }
    RUSSIAN_PHONETIC = {
        "А": "а", "Б": "бэ́", "В": "вэ́", "Г": "гэ́", "Д": "дэ́", "Е": "е́",
        "Ё": "ё́", "Ж": "жэ́", "З": "зэ́", "И": "и́", "Й": "и́ кра́ткое",
        "К": "ка́", "Л": "э́ль", "М": "э́м", "Н": "э́н", "О": "о́", "П": "пэ́",
        "Р": "э́р", "С": "э́с", "Т": "тэ́", "У": "у́", "Ф": "э́ф", "Х": "ха́",
        "Ц": "цэ́", "Ч": "че́", "Ш": "ша́", "Щ": "ща́", "Ъ": "твё́рдый зна́к",
        "Ы": "ы́", "Ь": "мя́гкий зна́к", "Э": "э́", "Ю": "ю́", "Я": "я́",
    }

    def __init__(self, config: dict):
        self.config = config
        self.acro_ru: Dict[str, str] = {}
        self.acro_en: Dict[str, str] = {}
        self.regex_dict: Dict[str, str] = {}
        self.rulers: Dict[str, Dict[str, str]] = {}
        self._load_dictionaries()

    @staticmethod
    def _parse_dict_file(filepath: Path) -> Dict[str, str]:
        result = {}
        if not filepath.exists():
            return result
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        if " = " in line:
                            parts = line.split(" = ", 1)
                        else:
                            parts = line.rsplit("=", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = parts[1].strip()
                            if key and value:
                                result[key] = value
                    else:
                        key = line.strip()
                        if key:
                            result[key] = ""
        except Exception as e:
            logger.warning(f"Ошибка парсинга словаря {filepath}: {e}")
        return result

    @staticmethod
    def _save_dict(filepath: Path, data: Dict[str, str]):
        try:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                for key in sorted(data.keys()):
                    if data[key]:
                        f.write(f"{key} = {data[key]}\n")
                    else:
                        f.write(f"{key}\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения словаря {filepath}: {e}")

    def _parse_rulers_file(self, filepath: Path) -> Dict[str, Dict[str, str]]:
        result = {}
        if not filepath.exists():
            return result
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 8:
                        gender, base, nom, gen, dat, acc, ins, prep = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]
                        result[base] = {
                            "gender": gender, "base": base,
                            "nom": nom, "gen": gen, "dat": dat,
                            "acc": acc, "ins": ins, "prep": prep,
                        }
        except Exception as e:
            logger.warning(f"Ошибка парсинга словаря правителей {filepath}: {e}")
        return result

    def _load_dictionaries(self):
        script_dir = Path(__file__).parent
        acro_ru_path = script_dir / self.config.get("acro_ru_dict", "acro_ru.gz")
        if acro_ru_path.exists():
            self.acro_ru = self._parse_dict_file(acro_ru_path)
        else:
            self._create_default_ru_dict()
        acro_en_path = script_dir / self.config.get("acro_en_dict", "acro_en.gz")
        if acro_en_path.exists():
            self.acro_en = self._parse_dict_file(acro_en_path)
        else:
            self._create_default_en_dict()
        regex_path = script_dir / "regex.gz"
        if regex_path.exists():
            self.regex_dict = self._parse_dict_file(regex_path)
        else:
            self._create_default_regex_dict()
        rulers_path = script_dir / "rulers.gz"
        if rulers_path.exists():
            self.rulers = self._parse_rulers_file(rulers_path)
        else:
            self._create_default_rulers_dict()
        self._compile_regex_dict()

    def _compile_regex_dict(self):
        """Компилирует все regex-паттерны для ускорения."""
        import regex as re
        self._regex_compiled = []
        for pattern, replacement in self.regex_dict.items():
            try:
                p = pattern
                lb = r'(?<![а-яё\w\u0301\u0300-\u036f])'
                la = r'(?![а-яё\w\u0301\u0300-\u036f])'
                if p.startswith('\\b'):
                    p = lb + p[2:]
                if p.endswith('\\b'):
                    p = p[:-2] + la
                self._regex_compiled.append((re.compile(p), replacement))
            except re.error:
                pass

    def _create_default_ru_dict(self):
        self.acro_ru = {"ВВС": "", "СВТ": "эс вэ тэ", "НКВД": "эн ка вэ дэ", "СМЕРШ": "сме́рш"}
        self._save_dict(Path(__file__).parent / "acro_ru.gz", self.acro_ru)

    def _create_default_en_dict(self):
        self.acro_en = {"BBC": "", "GPS": "", "CPU": "си пи ю", "GPU": "джи пи ю"}
        self._save_dict(Path(__file__).parent / "acro_en.gz", self.acro_en)

    def _create_default_regex_dict(self):
        self.regex_dict = {r"\bP\.S\.(?![а-яё\w])": "пост скри́птум", r"\betc\.(?![а-яё\w])": "эт цетера"}
        self._save_dict(Path(__file__).parent / "regex.gz", self.regex_dict)

    def _create_default_rulers_dict(self):
        self.rulers = {
            "Пётр": {"gender": "mas", "base": "Пётр", "nom": "Пё́тр", "gen": "Петра́", "dat": "Петру́", "acc": "Петра́", "ins": "Петро́м", "prep": "Петре́"},
            "Екатерина": {"gender": "fem", "base": "Екатерина", "nom": "Екатери́на", "gen": "Екатери́ны", "dat": "Екатери́не", "acc": "Екатери́ну", "ins": "Екатери́ной", "prep": "Екатери́не"},
        }
        self._save_rulers_dict(Path(__file__).parent / "rulers.gz")

    def _save_rulers_dict(self, filepath: Path):
        try:
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                f.write("# Имена правителей\n")
                f.write("# Формат: род имя И.п. Р.п. Д.п. В.п. Т.п. П.п.\n\n")
                for name in sorted(self.rulers.keys()):
                    forms = self.rulers[name]
                    f.write(f"{forms.get('gender', 'mas')} {forms.get('base', name)} {forms.get('nom', '')} {forms.get('gen', '')} {forms.get('dat', '')} {forms.get('acc', '')} {forms.get('ins', '')} {forms.get('prep', '')}\n")
        except Exception as e:
            logger.error(f"Ошибка сохранения словаря правителей {filepath}: {e}")

    def _make_spelling(self, word: str, language: str) -> str:
        phonetic = self.RUSSIAN_PHONETIC if language == "ru" else self.ENGLISH_PHONETIC
        return " ".join(phonetic.get(c, c.lower()) for c in word.upper() if c.strip())

    def get_pronunciation(self, word: str, language: str) -> Optional[str]:
        dictionary = self.acro_ru if language == "ru" else self.acro_en
        if word in dictionary:
            value = dictionary[word]
            return value if value else self._make_spelling(word, language)
        for key, value in dictionary.items():
            if word.upper() == key.upper():
                return value if value else self._make_spelling(key, language)
        return None

    def get_ruler_form(self, name: str) -> Optional[Dict]:
        import unicodedata
        name_clean = ''.join(c for c in unicodedata.normalize('NFD', name.lower()) if not unicodedata.combining(c))
        for key, forms in self.rulers.items():
            for k in ("base", "nom", "gen", "acc", "dat", "ins", "prep"):
                form_val = forms.get(k, '')
                form_clean = ''.join(c for c in unicodedata.normalize('NFD', form_val.lower()) if not unicodedata.combining(c))
                if name_clean == form_clean:
                    return forms
            key_clean = ''.join(c for c in unicodedata.normalize('NFD', key.lower()) if not unicodedata.combining(c))
            if name_clean == key_clean:
                return forms
        return None