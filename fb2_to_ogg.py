#!/usr/bin/env python3
"""
TTS конвертер FB2 в аудио (OGG/WAV) с поддержкой сносок и ударений
Использует Silero TTS для озвучивания текста

ТРЕБОВАНИЯ:
  pip install torch torchaudio silero-tts numpy pyyaml
  sudo apt install ffmpeg

  Перед использованием обязательно нормализовать FB2:
  python normalize_fb2.py book.fb2
"""

VERSION = "1.3.0"

import argparse
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Проверка всех необходимых пакетов перед работой
REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchcodec": "torchcodec",
    "silero-tts": "silero_tts",
    "numpy": "numpy",
    "pyyaml": "yaml",
    "pedalboard": "pedalboard",
    "scipy": "scipy",
}
MISSING = []
for pkg_name, import_name in REQUIRED_PACKAGES.items():
    try:
        __import__(import_name)
    except ImportError:
        MISSING.append(pkg_name)

if MISSING:
    print(f"\n{RED}Не установлены необходимые пакеты:{RESET}")
    for pkg in MISSING:
        print(f"  - {pkg}")
    print(f"\nУстановите:\n  pip install {' '.join(MISSING)}")
    exit(1)

import numpy as np
import torch
import torchaudio
import warnings

# Подавляем предупреждения о SSML тегах для моделей v3
warnings.filterwarnings("ignore", message="Current model doesn't support SSML tag")

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "ru_model": "v5_cis_base_nostress",
    "en_model": "v3_en",
    "put_accent": True,  # авто-ударения для слов без ручной разметки
    "put_yo": True,  # авто-расстановка буквы ё
    "put_stress_homo": True,  # авто-ударения для омографов
    "put_yo_homo": True,  # авто-ё для омографов
    "ru_speaker": None,
    "en_speaker": "en_0",
    "service_speaker": None,
    "sample_rate": 48000,
    "format": "ogg",
    "speed": 1.0,
    "device": "auto",
    "pause_between_paragraphs": 0.2,
    "pause_between_sentences": 0.05,
    "pause_comma": 0.05,
    "pause_dash": 0.2,  # пауза для тире (—)
    "pause_semicolon": 0.1,
    "pause_colon": 0.12,
    "force_punctuation": False,  # True = принудительные паузы по знакам, False = естественные паузы Silero
    "ssml_enabled": False,  # True = паузы через SSML, False = программные паузы
    "pause_between_chapters": 1.5,
    "pause_between_subsections": 0.8,  # пауза между подглавами (subtitle)
    "footnote_prefix": "Сноска",
    "footnote_suffix": "Конец сноски",
    "annotation_prefix": "Аннотация",
    "annotation_suffix": "Конец аннотации",
    "final_phrase": "Конец озвученного текста.",
    "skip_footnotes": False,
    "loudness_target": -23.0,
    "parallel_threads": 1,
    "max_chunk_length": 900,
    "blocks_per_part": 100,
    # Настройки для выделения курсива/emphasis
    "emphasis_as_text": False,  # True = курсив как обычный текст
    "emphasis_pause_before": 0.0,
    "emphasis_pause_after": 0.0,
    "emphasis_speaker": None,
    "emphasis_speed": 1.0,
    "emphasis_pitch": "medium",  # x-low, low, medium, high, x-high (для SSML)
    # Настройки для английского текста
    "en_pause_before": 0.0,
    "en_pause_after": 0.0,
    "en_speed": 1.0,
    # Настройки сносок
    "footnote_pause_before": 0.3,
    "footnote_pause_after": 0.3,
    "footnote_force_comma": True,
    # Постобработка ffmpeg (эквалайзер/компрессор)
    "ffmpeg_filter": None,
    "vorbis_quality": 6,  # при ffmpeg_filter: битрейт opus = vorbis_quality * 32 kbps
    "filter_threads": 4,  # потоков для аудиофильтров ffmpeg (не используется)
    # Постобработка pedalboard (рекомендуется)
    "pedalboard_enabled": False,
    "pedalboard_room_tone": 0,  # уровень комнатного шума dB (0 = выкл)
    "pb_highpass_hz": 85,  # обрезка инфраниза
    "pb_lowpass_hz": 11500,  # обрезка высоких (песок)
    "pb_warmth_hz": 280,  # частота "тепла"
    "pb_warmth_db": 1.8,  # усиление тепла
    "pb_clarity_hz": 3200,  # частота ясности
    "pb_clarity_db": 1.4,  # усиление ясности
    "pb_comp_threshold": -18,  # порог компрессора dB
    "pb_comp_ratio": 2.4,  # ratio компрессора
    "pb_comp_attack": 18,  # атака компрессора ms
    "pb_comp_release": 120,  # релиз компрессора ms
    "pb_reverb_room": 0.22,  # размер комнаты (0-1)
    "pb_reverb_damping": 0.55,  # damping реверба (0-1)
    "pb_reverb_wet": 0.09,  # уровень wet реверба (0-1)
    "pb_reverb_width": 0.6,  # ширина реверба (0-1)
    "pb_gain_db": 0.3,  # финальный гейн dB
    "pb_deharsh_hz": 5500,  # частота подавления резонансов
    "pb_deharsh_db": -2.5,  # ослабление резонансов dB
    "pb_deharsh_q": 2.0,  # ширина полосы подавления (меньше = шире)
    "pb_deharsh2_hz": 7800,  # вторая частота подавления
    "pb_deharsh2_db": -3.0,  # ослабление dB
    "pb_deharsh2_q": 2.0,  # ширина второй полосы
}

STRESS_MARK = "\u0301"
STRESS_PATTERN = re.compile(r"([аеёиоуыэюяАЕЁИОУЫЭЮЯ])" + STRESS_MARK, re.IGNORECASE)

MODEL_TYPES = {
    "v5_5_ru": "auto",
    "v5_5_ru_manual": "manual",  # v5_5_ru с ручными ударениями
    "v5_4_ru": "auto",
    "v5_3_ru": "auto",
    "v5_2_ru": "auto",
    "v5_1_ru": "auto",
    "v5_ru": "auto",
    "v5_cis_base": "manual",
    "v5_cis_base_nostress": "manual",
    "v5_cis_ext": "manual",
    "v4_ru": "manual",
    "ru_v3": "manual",
    "v3_1_ru": "manual",
}

LANG_FROM_MODEL = {
    "v3_en": "en",
    "v3_de": "de",
    "v3_fr": "fr",
    "v3_es": "es",
    "v3_en_indic": "en",
}

RU_SPEAKERS_STANDARD = ["aidar", "baya", "kseniya", "xenia", "eugene"]
RU_SPEAKERS_CIS = [
    "ru_aigul",
    "ru_albina",
    "ru_alexandr",
    "ru_bogdan",
    "ru_dmitriy",
    "ru_ekaterina",
    "ru_eduard",
    "ru_gamat",
    "ru_igor",
    "ru_karina",
    "ru_kejilgan",
    "ru_kermen",
    "ru_larisa",
    "ru_marat",
    "ru_miyau",
    "ru_nurgul",
    "ru_oksana",
    "ru_onaoy",
    "ru_ramilia",
    "ru_roman",
    "ru_safarhuja",
    "ru_saida",
    "ru_sibday",
    "ru_vika",
    "ru_zara",
    "ru_zhadyra",
    "ru_zhazira",
    "ru_zinaida",
]


def save_audio(audio: np.ndarray, filepath: str, sample_rate: int):
    """Сохраняет аудио в файл (torchaudio с torchcodec)."""
    try:
        audio_tensor = (
            torch.from_numpy(audio).float().unsqueeze(0)
            if isinstance(audio, np.ndarray)
            else audio.float().unsqueeze(0)
        )
        torchaudio.save(filepath, audio_tensor, sample_rate)
        return True
    except Exception:
        return False


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из файла."""
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                    import yaml

                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
                if loaded_config:
                    config.update(loaded_config)
        except Exception:
            pass
    
    # Валидация критических параметров
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.warning(f"Конфигурация: {error}")
    
    return config


def validate_config(config: dict) -> List[str]:
    """Проверяет корректность конфигурации."""
    errors = []
    
    # Проверка скорости
    speed = config.get('speed', 1.0)
    if speed <= 0:
        errors.append("speed должен быть > 0")
    elif speed > 4.0:
        errors.append("speed не должен превышать 4.0")
    
    # Проверка длины чанка
    chunk_length = config.get('max_chunk_length', 900)
    if chunk_length < 100:
        errors.append("max_chunk_length слишком мал (минимум 100)")
    elif chunk_length > 2000:
        errors.append("max_chunk_length слишком велик (максимум 2000)")
    
    # Проверка частоты дискретизации
    sample_rate = config.get('sample_rate', 48000)
    if sample_rate not in [8000, 16000, 24000, 48000]:
        errors.append(f"sample_rate {sample_rate} не поддерживается (используйте 24000 или 48000)")
    
    # Проверка LUFS
    loudness = config.get('loudness_target')
    if loudness is not None:
        if loudness > 0:
            errors.append("loudness_target должен быть отрицательным (например, -23.0)")
        elif loudness < -70:
            errors.append("loudness_target слишком низкий (минимум -70)")
    
    # Проверка количества блоков в части
    blocks_per_part = config.get('blocks_per_part', 100)
    if blocks_per_part < 1:
        errors.append("blocks_per_part должен быть >= 1")
    
    return errors


def change_speed_audio(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Изменяет скорость воспроизведения аудио."""
    if speed == 1.0 or len(audio) == 0:
        return audio
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    new_sample_rate = int(sample_rate * speed)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=new_sample_rate
    )
    speed_changed = resampler(waveform)
    resampler_back = torchaudio.transforms.Resample(
        orig_freq=new_sample_rate, new_freq=sample_rate
    )
    return resampler_back(speed_changed).squeeze(0).numpy()


def measure_rms(audio: np.ndarray) -> float:
    """Измеряет RMS громкость аудио."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio**2)))


def run_ffmpeg(args: List[str], description: str = "") -> bool:
    """Запускает ffmpeg с переданными аргументами."""
    try:
        subprocess.run(["ffmpeg", "-y"] + args, check=True, capture_output=True)
        if description:
            logger.info(description)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except FileNotFoundError:
        logger.error("ffmpeg не установлен. Установите: sudo apt install ffmpeg")
        return False


def concat_ogg_files(
    input_files: List[Path],
    output_file: str,
    ffmpeg_filter: str = None,
    vorbis_quality: int = 6,
) -> bool:
    """Склеивает OGG файлы через ffmpeg с опциональной постобработкой."""
    concat_file = Path(output_file).with_suffix(".concat.txt")
    with open(concat_file, "w") as f:
        for filepath in input_files:
            path = Path(filepath) if not isinstance(filepath, Path) else filepath
            f.write(f"file '{path.absolute()}'\n")
    if ffmpeg_filter:
        q = str(vorbis_quality)
        args = [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-af",
            ffmpeg_filter,
            "-c:a",
            "libvorbis",
            "-q:a",
            q,
            output_file,
        ]
        desc = f"Склеено {len(input_files)} OGG-файлов с фильтром: {ffmpeg_filter}"
    else:
        args = [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            output_file,
        ]
        desc = f"Склеено {len(input_files)} OGG-файлов"
    success = run_ffmpeg(args, desc)
    concat_file.unlink(missing_ok=True)
    return success


def split_long_text(text: str, max_length: int = 900) -> List[str]:
    """Разбивает длинный текст на части для синтеза."""
    if len(text) <= max_length:
        return [text] if text.strip() else []
    
    parts = []
    remaining = text
    
    while len(remaining) > max_length:
        chunk = remaining[:max_length]
        
        # Ищем конец предложения (знак + пробел + заглавная буква или конец)
        split_pos = _find_best_split_position(chunk, max_length)
        
        if split_pos <= 0:
            # Если не нашли хорошее место - берем max_length
            split_pos = max_length
        
        part = remaining[:split_pos].strip()
        if part and len(part) >= 2:
            parts.append(part)
        
        remaining = remaining[split_pos:].strip()
    
    if remaining and len(remaining) >= 2:
        parts.append(remaining)
    
    return parts


def _find_best_split_position(chunk: str, max_length: int) -> int:
    """Находит лучшую позицию для разбиения чанка.
    
    Приоритеты:
    1. Конец предложения (.!?… + пробел + заглавная буква/цифра)
    2. Двоеточие или точка с запятой (: или ;)
    3. Запятая
    4. Пробел (только если совсем нет других вариантов)
    5. Ничего - возвращаем max_length
    """
    
    # 1. Ищем конец предложения
    # Паттерн: знак препинания + пробел + заглавная буква или цифра
    for i in range(min(len(chunk), max_length) - 1, max_length // 2, -1):
        if chunk[i] in '.!?…':
            # Проверяем, что это действительно конец предложения
            if i + 1 < len(chunk):
                if chunk[i + 1] == ' ':
                    # Следующий символ после пробела
                    if i + 2 < len(chunk):
                        next_char = chunk[i + 2]
                        if next_char.isupper() or next_char.isdigit() or next_char in '«"\'(':
                            return i + 2  # Включая пробел
                    else:
                        return i + 1
            elif chunk[i + 1] in '»"\')':
                    # Закрывающая кавычка/скобка после знака
                    if i + 2 < len(chunk) and chunk[i + 2] == ' ':
                        if i + 3 < len(chunk):
                            next_char = chunk[i + 3]
                            if next_char.isupper() or next_char.isdigit():
                                return i + 3
                        else:
                            return i + 2
            else:
                # Знак в конце чанка
                return i + 1
    
    # 2. Ищем двоеточие или точку с запятой
    for i in range(min(len(chunk), max_length) - 1, max_length // 2, -1):
        if chunk[i] in ':;':
            if i + 1 < len(chunk) and chunk[i + 1] == ' ':
                return i + 2
            else:
                return i + 1
    
    # 3. Ищем запятую (только если за ней пробел)
    for i in range(min(len(chunk), max_length) - 1, max_length // 2, -1):
        if chunk[i] == ',':
            if i + 1 < len(chunk) and chunk[i + 1] == ' ':
                return i + 2
            else:
                return i + 1
    
    # 4. Ищем пробел (не разрезаем слово)
    for i in range(min(len(chunk), max_length) - 1, max_length // 2, -1):
        if chunk[i] == ' ':
            # Убеждаемся, что пробел не в начале и не в конце
            if i > 0 and i < len(chunk) - 1:
                return i + 1
    
    # 5. Не нашли хорошего места - возвращаем max_length
    return max_length


class TextCleaner:
    """Очистка и подготовка текста для TTS."""

    @staticmethod
    def clean_for_tts(text: str) -> str:
        """Очищает текст от неподдерживаемых символов."""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        replacements = {
            "…": "...",
            "—": " <dash> ",
            "–": " <dash> ",
            "-": " ",
            "\u2019": "'",
            "\u2018": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u00ab": '"',
            "\u00bb": '"',
            "&nbsp;": " ",
            "&mdash;": "-",
            "&laquo;": '"',
            "&raquo;": '"',
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "//": " ",
            "/*": " ",
            "*/": " ",  # убираем шипящие комбинации
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        result = []
        for ch in text:
            cat = unicodedata.category(ch)
            if (
                cat.startswith("L")
                or cat.startswith("N")
                or cat.startswith("Z")
                or cat.startswith("M")
            ):
                result.append(ch)
            elif cat.startswith("P") or ch in "+-":
                result.append(ch)
            else:
                result.append(" ")
        return re.sub(r"\s+", " ", "".join(result)).strip()

    @staticmethod
    def convert_stress_for_model(text: str, model_type: str) -> str:
        """Конвертирует ударения в формат модели."""
        if not text:
            return ""
        if model_type == "auto":
            return STRESS_PATTERN.sub(r"\1", text)
        return STRESS_PATTERN.sub(lambda m: f"+{m.group(1)}", text)


class FB2Parser:
    """Парсер FB2 файлов."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.namespace = "http://www.gribuser.ru/xml/fictionbook/2.0"
        self.xlink_namespace = "http://www.w3.org/1999/xlink"
        self.footnotes_map: Dict[str, str] = {}
        self.force_punctuation = False
        self.pause_between_subsections = 0.8
        self.annotation_prefix = "Аннотация"
        self.annotation_suffix = "Конец аннотации"

    def parse(self) -> List[List[Dict]]:
        """Парсит FB2 файл и возвращает список глав с блоками."""
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            self._collect_footnotes(root)
            chapters = []
            # Сначала аннотация из <description>
            desc = root.find(f"{{{self.namespace}}}description")
            if desc is not None:
                title_info = desc.find(f"{{{self.namespace}}}title-info")
                if title_info is not None:
                    # 1. Автор
                    author_elem = title_info.find(f"{{{self.namespace}}}author")
                    if author_elem is not None:
                        author_text = self._extract_text(author_elem)
                        if author_text.strip():
                            author_text = author_text.strip()
                            if not author_text.endswith(('.', '!', '?', '…')):
                                author_text += "."
                            chapters.append([
                                {"type": "text", "content": self._norm(author_text)}
                            ])
                    
                    # 2. Название книги
                    book_title_elem = title_info.find(f"{{{self.namespace}}}book-title")
                    if book_title_elem is not None:
                        title_text = self._extract_text(book_title_elem)
                        if title_text.strip():
                            title_text = title_text.strip()
                            if not title_text.endswith(('.', '!', '?', '…')):
                                title_text += "."
                            chapters.append([
                                {"type": "text", "content": self._norm(title_text)}
                            ])
                    
                    # 3. Аннотация со служебными фразами
                    annotation = title_info.find(f"{{{self.namespace}}}annotation")
                    if annotation is not None:
                        anno_blocks = []
                        # Служебная фраза перед аннотацией
                        anno_blocks.append({
                            "type": "text", 
                            "content": self.annotation_prefix
                        })
                        for elem in annotation:
                            self._process_element(elem, anno_blocks)
                        # Служебная фраза после аннотации
                        anno_blocks.append({
                            "type": "text", 
                            "content": self.annotation_suffix
                        })
                        if anno_blocks:
                            chapters.append(anno_blocks)
            for body in root.findall(f"{{{self.namespace}}}body"):
                if body.get("name") == "notes":
                    continue
                # Если body содержит section — каждая section = глава
                sections = body.findall(f"{{{self.namespace}}}section")
                if sections:
                    # Обрабатываем элементы до первой секции (title, image, etc.)
                    pre_section_blocks = []
                    for elem in body:
                        if elem.tag == f"{{{self.namespace}}}section":
                            break
                        if elem.tag != f"{{{self.namespace}}}title":
                            self._process_element(elem, pre_section_blocks)
                        else:
                            # Title обрабатываем как заголовок всей книги
                            self._process_element(elem, pre_section_blocks)
                    
                    if pre_section_blocks:
                        chapters.append(pre_section_blocks)
                    
                    for section in sections:
                        chapter_blocks = []
                        self._process_element(section, chapter_blocks)
                        if chapter_blocks:
                            chapters.append(chapter_blocks)
                else:
                    # Нет секций — весь body = глава
                    chapter_blocks = []
                    for elem in body:
                        self._process_element(elem, chapter_blocks)
                    if chapter_blocks:
                        chapters.append(chapter_blocks)
            return chapters
        except Exception:
            return []

    @staticmethod
    def _norm(text: str) -> str:
        """Нормализует текст в NFC."""
        if text:
            return unicodedata.normalize("NFC", text)
        return text

    def _collect_footnotes(self, root):
        """Собирает сноски из тела notes."""
        for body in root.findall(f"{{{self.namespace}}}body"):
            if body.get("name") == "notes":
                for section in body.findall(f"{{{self.namespace}}}section"):
                    section_id = section.get("id")
                    if section_id:
                        parts = []
                        for elem in section:
                            if elem.tag == f"{{{self.namespace}}}title":
                                continue
                            text = self._extract_text(elem)
                            if text.strip():
                                parts.append(text)
                        note_text = " ".join(parts)
                        if note_text.strip():
                            self.footnotes_map[section_id] = self._norm(note_text)

    def _process_element(self, element, blocks: List[Dict]):
        """Обрабатывает элемент FB2."""
        if element.tag in [
            f"{{{self.namespace}}}binary",
            f"{{{self.namespace}}}description",
        ]:
            return
        if element.tag == f"{{{self.namespace}}}subtitle":
            # Подглава: добавляем паузу перед текстом
            if blocks:
                blocks.append({
                    "type": "pause", 
                    "duration": self.pause_between_subsections
                })
            text = self._extract_text(element)
            if text.strip():
                blocks.append({"type": "text", "content": self._norm(text.strip())})
                # Пауза после заголовка (50% от паузы до)
                blocks.append({
                    "type": "pause",
                    "duration": self.pause_between_subsections * 0.5
                })
            return
        # Обработка tts_en тегов на верхнем уровне
        if element.tag == "tts_en" or (
            element.tag.startswith("{") and element.tag.endswith("}tts_en")
        ):
            inner = self._extract_text(element).strip()
            if inner:
                blocks.append(
                    {"type": "text", "content": f"<tts_en>{inner}</tts_en>"}
                )
            if element.tail and element.tail.strip():
                blocks.append({"type": "text", "content": self._norm(element.tail)})
            return
        if element.tag == f"{{{self.namespace}}}emphasis":
            text = self._extract_text(element)
            if text.strip():
                blocks.append({"type": "emphasis", "content": self._norm(text.strip())})
            return
        if element.tag == f"{{{self.namespace}}}epigraph":
            full_text = self._extract_text(element)
            if full_text.strip():
                blocks.append(
                    {"type": "text", "content": self._norm(full_text.strip())}
                )
            return
        if element.tag in [
            f"{{{self.namespace}}}p",
            f"{{{self.namespace}}}title",
            f"{{{self.namespace}}}subtitle",
            f"{{{self.namespace}}}cite",
            f"{{{self.namespace}}}emphasis",
            f"{{{self.namespace}}}strong",
            f"{{{self.namespace}}}annotation",
            f"{{{self.namespace}}}text-author",
            f"{{{self.namespace}}}poem",
            f"{{{self.namespace}}}stanza",
            f"{{{self.namespace}}}v",
            f"{{{self.namespace}}}table",
            f"{{{self.namespace}}}tr",
            f"{{{self.namespace}}}td",
            f"{{{self.namespace}}}th",
        ]:
            self._extract_text_with_footnotes(element, blocks)
        elif element.tag == f"{{{self.namespace}}}section":
            # Проверяем, есть ли title у секции
            title_elem = element.find(f"{{{self.namespace}}}title")
            
            if title_elem is not None:
                # Пауза перед заголовком
                if blocks:
                    blocks.append({
                        "type": "pause",
                        "duration": self.pause_between_subsections
                    })
                
                # Обрабатываем title
                self._process_element(title_elem, blocks)
                
                # Пауза после заголовка (50%)
                blocks.append({
                    "type": "pause",
                    "duration": self.pause_between_subsections * 0.5
                })
                
                # Обрабатываем остальные элементы (кроме title)
                for child in element:
                    if child is not title_elem:
                        self._process_element(child, blocks)
            else:
                # Нет title — просто обрабатываем все элементы
                for child in element:
                    self._process_element(child, blocks)
        elif element.tag == f"{{{self.namespace}}}empty-line":
            if blocks and blocks[-1]["type"] == "text":
                blocks.append({"type": "pause", "duration": 0.5})

    def _extract_text_with_footnotes(self, element, blocks: List[Dict]):
        """Извлекает текст с поддержкой сносок."""
        if element.text and element.text.strip():
            blocks.append({"type": "text", "content": self._norm(element.text)})
        for child in element:
            # Сохраняем tts_en теги (без namespace) как есть
            if child.tag == "tts_en" or (
                child.tag.startswith("{") and child.tag.endswith("}tts_en")
            ):
                inner = self._extract_text(child).strip()
                if inner:
                    blocks.append(
                        {"type": "text", "content": f"<tts_en>{inner}</tts_en>"}
                    )
                # tail всегда отдельным русским блоком (НЕ добавляем к английскому)
                if child.tail and child.tail.strip():
                    tail_text = self._norm(child.tail)
                    blocks.append({"type": "text", "content": tail_text})
                continue
            if child.tag == f"{{{self.namespace}}}a":
                link_type = child.get("type")
                if link_type == "note":
                    href = child.get(f"{{{self.xlink_namespace}}}href", "")
                    note_id = href[1:] if href.startswith("#") else href
                    if note_id in self.footnotes_map:
                        note_title = self._extract_text(child).strip()
                        blocks.append(
                            {
                                "type": "footnote",
                                "id": note_id,
                                "text": self.footnotes_map[note_id],
                                "title": self._norm(note_title),
                            }
                        )
                else:
                    link_text = self._extract_text(child)
                    if link_text.strip():
                        blocks.append(
                            {"type": "text", "content": self._norm(link_text)}
                        )
            else:
                self._extract_text_with_footnotes(child, blocks)
            if child.tail and child.tail.strip():
                blocks.append({"type": "text", "content": self._norm(child.tail)})

    def _extract_text(self, element) -> str:
        """Извлекает весь текст из элемента, сохраняя <tts_en> теги."""
        if element is None:
            return ""
        parts = []
        if element.text and element.text.strip():
            parts.append(self._norm(element.text.strip()))
        for child in element:
            # Сохраняем tts_en теги
            if child.tag == "tts_en" or (
                child.tag.startswith("{") and child.tag.endswith("}tts_en")
            ):
                # Извлекаем только текст внутри tts_en (без tail)
                inner = self._extract_text(child).strip()
                if inner:
                    parts.append(f"<tts_en>{inner}</tts_en>")
                # tail добавляем как отдельный русский текст
                if child.tail and child.tail.strip():
                    parts.append(self._norm(child.tail.strip()))
                continue
            if child.tag == f"{{{self.namespace}}}a" and child.get("type") == "note":
                continue
            child_text = self._extract_text(child)
            if child_text.strip():
                parts.append(child_text)
            if child.tail and child.tail.strip():
                parts.append(self._norm(child.tail.strip()))
        return " ".join(parts)


class LanguageDetector:
    """Определение языка текста."""

    @staticmethod
    def has_latin(text: str) -> bool:
        """Проверяет наличие латинских символов."""
        return bool(re.search(r"[a-zA-Z]", text))

    @staticmethod
    def split_by_language(text: str) -> List[Tuple[str, str]]:
        """Разбивает текст на блоки по языку — учитывает теги <tts_en>."""
        if not text:
            return []

        parts = re.split(r"(<tts_en>.*?</tts_en>)", text, flags=re.DOTALL)
        result = []
        for part in parts:
            if part.startswith("<tts_en>") and part.endswith("</tts_en>"):
                inner = part[len("<tts_en>") : -len("</tts_en>")]
                if inner.strip():
                    result.append(("en", inner.strip()))
            else:
                if part.strip():
                    result.append(("ru", part.strip()))

        return result if result else [("ru", text.strip())] if text.strip() else []


class SileroTTS:
    """Обёртка для Silero TTS моделей."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config["sample_rate"]
        self.device = (
            "cuda"
            if (config["device"] == "auto" and torch.cuda.is_available())
            else (config["device"] if config["device"] != "auto" else "cpu")
        )
        self.ru_model_name = config["ru_model"]
        self.en_model_name = config["en_model"]
        self.ru_model_type = MODEL_TYPES.get(self.ru_model_name, "auto")
        self.speed = config.get("speed", 1.0)
        self.models = {}
        self.speakers = {}
        self._load_models()

    def _is_cis_model(self) -> bool:
        """Проверяет, является ли модель CIS."""
        return "cis" in self.ru_model_name

    def _lang_from_en_model(self) -> str:
        """Определяет язык для en_model (en/de/fr/es)."""
        return LANG_FROM_MODEL.get(self.en_model_name, "en")

    def _get_default_speaker(self, lang: str) -> str:
        """Возвращает голос по умолчанию для указанного языка."""
        if lang != "ru":
            available = self.speakers.get(lang, [])
            return available[0] if available else f"{lang}_0"
        user_speaker = self.config.get("ru_speaker")
        if user_speaker:
            return user_speaker
        available_speakers = self.speakers.get("ru", [])
        if not available_speakers:
            return "kseniya" if not self._is_cis_model() else "ru_dmitriy"
        if self._is_cis_model():
            candidates = [s for s in available_speakers if s.startswith("ru_")]
            return candidates[0] if candidates else available_speakers[0]
        return available_speakers[0] if available_speakers else "kseniya"

    def _get_default_ru_speaker(self) -> str:
        """Возвращает русский голос по умолчанию."""
        return self._get_default_speaker("ru")

    def _get_service_speaker(self) -> str:
        """Возвращает служебный голос."""
        if self.config.get("service_speaker"):
            return self.config["service_speaker"]
        return self._get_default_speaker("ru")

    def get_ru_speakers(self) -> List[str]:
        """Возвращает список доступных русских голосов."""
        if self._is_cis_model():
            return [s for s in self.speakers.get("ru", []) if s.startswith("ru_")]
        return self.speakers.get("ru", RU_SPEAKERS_STANDARD)
    
    def _load_models(self):
        """Загружает TTS модели."""
        # Загрузка русской модели
        actual_model_name = self.ru_model_name
        if actual_model_name == "v5_5_ru_manual":
            # v5_5_ru_manual использует модель v5_5_ru
            actual_model_name = "v5_5_ru"
        
        result = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=actual_model_name,
        )
        if isinstance(result, tuple):
            self.models["ru"] = result[0]
            if len(result) >= 3:
                try:
                    obj = result[2]
                    if isinstance(obj, dict):
                        self.speakers["ru"] = list(obj.keys())
                    elif isinstance(obj, list):
                        self.speakers["ru"] = obj
                except:
                    pass
        if not self.speakers.get("ru"):
            self.speakers["ru"] = (
                RU_SPEAKERS_CIS if self._is_cis_model() else RU_SPEAKERS_STANDARD
            )
        self.models["ru"].to(self.device)
        if hasattr(self.models["ru"], "eval"):
            self.models["ru"].eval()

        # Загрузка дополнительной модели (en/de/fr/es)
        ext_lang = self._lang_from_en_model()
        try:
            result = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=ext_lang,
                speaker=self.en_model_name,
            )
            if isinstance(result, tuple):
                self.models[ext_lang] = result[0]
                if len(result) >= 3:
                    try:
                        obj = result[2]
                        if isinstance(obj, dict):
                            self.speakers[ext_lang] = list(obj.keys())
                        elif isinstance(obj, list):
                            self.speakers[ext_lang] = obj
                    except:
                        pass
            if not self.speakers.get(ext_lang):
                self.speakers[ext_lang] = [f"{ext_lang}_{i}" for i in range(118)]
            self.models[ext_lang].to(self.device)
            if hasattr(self.models[ext_lang], "eval"):
                self.models[ext_lang].eval()
        except Exception:
            self.models[ext_lang] = None
            logger.warning(
                f"Не удалось загрузить модель {self.en_model_name} ({ext_lang})"
            )

    def synthesize(
        self, text: str, language: str = "ru", speaker: str = None
    ) -> np.ndarray:
        """Синтезирует речь из текста."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        try:
            model = self.models.get(language)
            if not model:
                # ищем любую нерусскую модель
                for key in self.models:
                    if key != "ru" and self.models[key] is not None:
                        model = self.models[key]
                        break
                if not model and self.models.get("ru"):
                    model = self.models["ru"]
                    language = "ru"
                    speaker = self._get_default_ru_speaker()
                    text = transliterate_to_cyrillic(text)
                if not model:
                    return np.array([], dtype=np.float32)
            if speaker is None:
                if language != "ru":
                    speaker = self._get_default_speaker(language)
                else:
                    speaker = self._get_default_ru_speaker()

            processed_text = text
            if language == "ru":
                processed_text = TextCleaner.convert_stress_for_model(
                    text, self.ru_model_type
                )

                # Настройки ударений из конфига
                put_accent = self.config.get("put_accent", True)
                put_yo = self.config.get("put_yo", True)
                put_stress_homo = self.config.get("put_stress_homo", True)
                put_yo_homo = self.config.get("put_yo_homo", True)

            # Подготавливаем параметры для apply_tts
            apply_tts_params = {
                "text": processed_text,
                "speaker": speaker,
                "sample_rate": self.sample_rate,
            }
            
            # Добавляем флаги для моделей v5
            if language == "ru" and self.ru_model_name.startswith("v5"):
                apply_tts_params.update({
                    "put_accent": put_accent,
                    "put_yo": put_yo,
                })
                
                # Для v5_5_ru добавляем дополнительные флаги
                if self.ru_model_name in ["v5_5_ru", "v5_5_ru_manual"]:
                    apply_tts_params.update({
                        "put_stress_homo": put_stress_homo,
                        "put_yo_homo": put_yo_homo,
                    })

            try:
                audio = model.apply_tts(**apply_tts_params)
            except TypeError as e:
                # Если модель не поддерживает флаги, пробуем без них
                logger.warning(f"Модель не поддерживает все флаги: {e}. Пробуем без флагов.")
                apply_tts_params.pop("put_stress_homo", None)
                apply_tts_params.pop("put_yo_homo", None)
                try:
                    audio = model.apply_tts(**apply_tts_params)
                except TypeError:
                    apply_tts_params.pop("put_accent", None)
                    apply_tts_params.pop("put_yo", None)
                    audio = model.apply_tts(**apply_tts_params)

            if audio is None:
                return np.array([], dtype=np.float32)
            if torch.is_tensor(audio):
                if audio.numel() == 0:
                    return np.array([], dtype=np.float32)
                audio = audio.cpu().numpy()
            if isinstance(audio, np.ndarray) and audio.size == 0:
                return np.array([], dtype=np.float32)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            if self.speed != 1.0:
                audio = change_speed_audio(audio, self.sample_rate, self.speed)

            return audio
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("Недостаточно памяти GPU. Попробуйте использовать --cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return np.array([], dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)

    def synthesize_ssml(
        self, ssml_text: str, language: str = "ru", speaker: str = None
    ) -> np.ndarray:
        """Синтезирует речь из SSML разметки."""
        if not ssml_text or not ssml_text.strip():
            return np.array([], dtype=np.float32)
        try:
            model = self.models.get(language)
            if not model:
                return np.array([], dtype=np.float32)
            
            if speaker is None:
                speaker = self._get_default_speaker(language)
            
            # Конвертируем Unicode ударения в +гласная для русской модели
            if language == "ru":
                ssml_text = TextCleaner.convert_stress_for_model(
                    ssml_text, self.ru_model_type
                )
            
            # Подготавливаем параметры
            apply_tts_params = {
                "ssml_text": ssml_text,
                "speaker": speaker,
                "sample_rate": self.sample_rate,
            }
            
            # Добавляем флаги ударений для русской модели v5
            if language == "ru" and self.ru_model_name.startswith("v5"):
                apply_tts_params.update({
                    "put_accent": self.config.get("put_accent", True),
                    "put_yo": self.config.get("put_yo", True),
                })
                
                if self.ru_model_name in ["v5_5_ru", "v5_5_ru_manual"]:
                    apply_tts_params.update({
                        "put_stress_homo": self.config.get("put_stress_homo", True),
                        "put_yo_homo": self.config.get("put_yo_homo", True),
                    })
            
            audio = model.apply_tts(**apply_tts_params)
            
            if audio is None:
                return np.array([], dtype=np.float32)
            if torch.is_tensor(audio):
                if audio.numel() == 0:
                    return np.array([], dtype=np.float32)
                audio = audio.cpu().numpy()
            if isinstance(audio, np.ndarray) and audio.size == 0:
                return np.array([], dtype=np.float32)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            return audio
        except Exception:
            return np.array([], dtype=np.float32)

    def is_model_available(self, language: str) -> bool:
        """Проверяет доступность модели."""
        if language == "ru":
            return "ru" in self.models and self.models["ru"] is not None
        if language in self.models and self.models[language] is not None:
            return True
        # fallback: ищем любую нерусскую модель
        for key in self.models:
            if key != "ru" and self.models[key] is not None:
                return True
        return False

    def fetch_model(self, model_name: str) -> bool:
        """Принудительно загружает указанную модель."""
        try:
            # Определяем язык модели
            lang = LANG_FROM_MODEL.get(model_name, "ru")
            
            # Если модель уже загружена
            if lang in self.models and self.models[lang] is not None:
                logger.info(f"Модель {model_name} уже загружена")
                return True
            
            logger.info(f"Загрузка модели {model_name} ({lang})...")
            
            result = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=lang,
                speaker=model_name,
            )
            
            if isinstance(result, tuple):
                self.models[lang] = result[0]
                if len(result) >= 3:
                    try:
                        obj = result[2]
                        if isinstance(obj, dict):
                            self.speakers[lang] = list(obj.keys())
                        elif isinstance(obj, list):
                            self.speakers[lang] = obj
                    except:
                        pass
            
            self.models[lang].to(self.device)
            if hasattr(self.models[lang], "eval"):
                self.models[lang].eval()
            
            logger.info(f"Модель {model_name} загружена успешно")
            return True
        except Exception as e:
            logger.error(f"Не удалось загрузить модель {model_name}: {e}")
            return False


def transliterate_to_cyrillic(text: str) -> str:
    """Транслитерация латиницы в кириллицу для fallback."""
    # Названия букв для аббревиатур (без гласных)
    letter_names = {
        "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и",
        "f": "эф", "g": "джи", "h": "эйч", "i": "ай", "j": "джей",
        "k": "кей", "l": "эль", "m": "эм", "n": "эн", "o": "оу",
        "p": "пи", "q": "кью", "r": "эр", "s": "эс", "t": "ти",
        "u": "ю", "v": "ви", "w": "дабл ю", "x": "экс", "y": "уай",
        "z": "зэд",
        "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и",
        "F": "эф", "G": "джи", "H": "эйч", "I": "ай", "J": "джей",
        "K": "кей", "L": "эль", "M": "эм", "N": "эн", "O": "оу",
        "P": "пи", "Q": "кью", "R": "эр", "S": "эс", "T": "ти",
        "U": "ю", "V": "ви", "W": "дабл ю", "X": "экс", "Y": "уай",
        "Z": "зэд",
    }
    
    # Обычная транслитерация (для слов с гласными)
    mapping = {
        "a": "а",
        "b": "б",
        "c": "к",
        "ce": "се",
        "ch": "ч",
        "ci": "си",
        "cy": "си",
        "d": "д",
        "e": "е",
        "f": "ф",
        "g": "г",
        "ge": "джэ",
        "gi": "джи",
        "gy": "джи",
        "h": "х",
        "i": "и",
        "j": "дж",
        "k": "к",
        "kh": "х",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "ph": "ф",
        "q": "к",
        "qu": "кв",
        "r": "р",
        "s": "с",
        "sh": "ш",
        "sсh": "ск",
        "t": "т",
        "tch": "ч",
        "th": "з",
        "u": "у",
        "v": "в",
        "w": "в",
        "wh": "в",
        "x": "кс",
        "y": "и",
        "z": "з",
    }
    
    # Разбиваем текст на латинские слова и не-латиницу
    parts = re.findall(r'[A-Za-z]+|[^A-Za-z]+', text)
    result_parts = []
    
    for part in parts:
        if re.match(r'^[A-Za-z]+$', part):
            # Нормализуем к нижнему регистру для транслитерации
            part_lower = part.lower()
            
            # Проверяем, есть ли гласные
            has_vowels = any(ch in 'aeiou' for ch in part_lower)
            
            if has_vowels:
                # Обычная транслитерация (слово с гласными)
                result = []
                i = 0
                while i < len(part_lower):
                    ch = part_lower[i]
                    digraph = part_lower[i : i + 2]
                    if digraph in mapping:
                        result.append(mapping[digraph])
                        i += 2
                    else:
                        result.append(mapping.get(ch, ch))
                        i += 1
                result_parts.append(''.join(result))
            else:
                # Аббревиатура — произносим по буквам
                letters = []
                for ch in part:
                    if ch.isalpha():
                        letters.append(letter_names.get(ch, ch))
                result_parts.append(' '.join(letters))
        else:
            result_parts.append(part)
    
    return ''.join(result_parts)


class TTSProcessor:
    """Процессор TTS с калибровкой и обработкой блоков."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config["sample_rate"]
        self.tts = SileroTTS(config)
        self.ru_speaker = self.tts._get_default_ru_speaker()
        self.service_speaker = self.tts._get_service_speaker()
        self.ext_lang = self.tts._lang_from_en_model()
        self.ext_speaker = config.get("en_speaker") or self.tts._get_default_speaker(
            self.ext_lang
        )
        self.pause_between_paragraphs = config.get("pause_between_paragraphs", 0.2)
        self.pause_between_sentences = config.get("pause_between_sentences", 0.05)
        self.footnote_prefix = config.get("footnote_prefix", "Сноска")
        self.footnote_suffix = config.get("footnote_suffix", "Конец сноски")
        self.max_chunk_length = config.get("max_chunk_length", 900)
        self.skip_footnotes = config.get("skip_footnotes", False)
        # Настройки emphasis (курсив)
        self.emphasis_pause_before = config.get("emphasis_pause_before", 0.0)
        self.emphasis_pause_after = config.get("emphasis_pause_after", 0.0)
        self.emphasis_speaker = config.get("emphasis_speaker")
        self.emphasis_speed = config.get("emphasis_speed", 1.0)
        # Настройки английского
        self.en_pause_before = config.get("en_pause_before", 0.0)
        self.en_pause_after = config.get("en_pause_after", 0.0)
        self.en_speed = config.get("en_speed", 1.0)

        self.speaker_gains = {
            "ru_main": 1.0,
            "ru_service": 1.0,
            self.ext_lang: 1.0,
        }

        if config.get("loudness_target") is not None:
            self._calibrate_speakers()

    def _calibrate_speakers(self):
        """Калибрует громкость голосов по тестовой фразе."""
        target_lufs = self.config.get("loudness_target", -23.0)
        target_rms = 10 ** (target_lufs / 20) * 0.5

        test_phrases = [
            (
                "ru_main",
                "ru",
                self.ru_speaker,
                "тестовая фраза для калибровки громкости.",
            ),
            (
                "ru_service",
                "ru",
                self.service_speaker,
                "тестовая фраза для калибровки громкости.",
            ),
            (
                self.ext_lang,
                self.ext_lang,
                self.ext_speaker,
                f"test phrase for volume calibration in {self.ext_lang}.",
            ),
        ]

        logger.info("Калибровка громкости голосов:")

        for name, lang, speaker, phrase in test_phrases:
            if lang == self.ext_lang and not self.tts.is_model_available(self.ext_lang):
                self.speaker_gains[name] = self.speaker_gains["ru_main"]
                logger.info(
                    f"  {name}: модель недоступна, используется gain русского голоса (×{self.speaker_gains['ru_main']:.3f})"
                )
                continue

            audio = self.tts.synthesize(phrase, lang, speaker)
            if len(audio) == 0:
                logger.warning(
                    f"  {name}: не удалось синтезировать тестовую фразу, используется gain=1.0"
                )
                continue

            current_rms = measure_rms(audio)
            if current_rms > 0:
                gain = target_rms / current_rms
                self.speaker_gains[name] = gain
                current_lufs_approx = 20 * np.log10(current_rms / 0.5)
                logger.info(
                    f"  {name}: RMS={current_rms:.4f} (~{current_lufs_approx:.0f} LUFS) → gain=×{gain:.3f}"
                )
            else:
                logger.warning(
                    f"  {name}: нулевая громкость тестовой фразы, используется gain=1.0"
                )

        logger.info(f"  Целевой RMS: {target_rms:.4f} (~{target_lufs} LUFS)")

    def synthesize_text(self, text: str) -> np.ndarray:
        """Синтезирует основной текст с определением языка."""
        if not text or not text.strip():
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        try:
            # Заменяем скобки на запятые (для естественных пауз Silero)
            text = text.replace('(', ', ').replace(')', ', ')
            
            # Если включен SSML — синтезируем с SSML-паузами
            if self.config.get("ssml_enabled", False):
                return self._synthesize_text_with_ssml(text)
            
            # При force_punctuation: разбиваем текст по знакам препинания
            if self.config.get("force_punctuation", False):
                return self._synthesize_with_forced_pauses(text)
            
            segments = LanguageDetector.split_by_language(text)
            all_audio = []
            for lang, seg_text in segments:
                # Заменяем <dash> на паузу
                seg_text = seg_text.replace('<dash>', '')
                cleaned = TextCleaner.clean_for_tts(seg_text)
                if not cleaned.strip() or len(cleaned.strip()) < 2:
                    continue
                chunks = split_long_text(cleaned, self.max_chunk_length)
                for chunk in chunks:
                    if lang == self.ext_lang and self.tts.is_model_available(
                        self.ext_lang
                    ):
                        # Пауза перед английским
                        if self.en_pause_before > 0:
                            all_audio.append(self.add_silence(self.en_pause_before))
                        audio = self.tts.synthesize(
                            chunk, self.ext_lang, self.ext_speaker
                        )
                        audio = audio * self.speaker_gains[self.ext_lang]
                        if self.en_speed != 1.0:
                            audio = change_speed_audio(
                                audio, self.sample_rate, self.en_speed
                            )
                        if len(audio) > 0:
                            all_audio.append(audio)
                        if self.en_pause_after > 0:
                            all_audio.append(self.add_silence(self.en_pause_after))
                    else:
                        # Если русский чанк содержит латиницу — транслитерируем
                        chunk_to_synth = chunk
                        if LanguageDetector.has_latin(chunk):
                            chunk_to_synth = transliterate_to_cyrillic(chunk)
                        audio = self.tts.synthesize(
                            chunk_to_synth, "ru", self.ru_speaker
                        )
                        audio = audio * self.speaker_gains["ru_main"]
                        if len(audio) > 0:
                            all_audio.append(audio)
                    # Добавляем паузу после предложения, если текст заканчивается знаком препинания
                    if chunk.rstrip().endswith(('.', '!', '?', '…')):
                        all_audio.append(
                            self.add_silence(self.config.get("pause_between_sentences", 0.05))
                        )
                    elif chunk.rstrip().endswith(';'):
                        all_audio.append(
                            self.add_silence(self.config.get("pause_semicolon", 0.1))
                        )
                    elif chunk.rstrip().endswith(':'):
                        all_audio.append(
                            self.add_silence(self.config.get("pause_colon", 0.12))
                        )
                    elif chunk.rstrip().endswith(','):
                        all_audio.append(
                            self.add_silence(self.config.get("pause_comma", 0.05))
                        )
                    elif '<dash>' in chunk.rstrip()[-10:]:
                        all_audio.append(
                            self.add_silence(self.config.get("pause_dash", 0.2))
                        )
            if all_audio:
                if len(all_audio) > 1 and all_audio[-1].size < int(
                    self.sample_rate * 0.5
                ):
                    all_audio = all_audio[:-1]
                return np.concatenate(all_audio)
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        except Exception:
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)

    def synthesize_service(self, text: str) -> np.ndarray:
        """Синтезирует служебный текст."""
        audio = self.tts.synthesize(text, "ru", self.service_speaker)
        return audio * self.speaker_gains["ru_service"]

    def _synthesize_text_with_ssml(self, text: str) -> np.ndarray:
        """Синтезирует текст с паузами через SSML."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Определяем паузы из конфига
        pause_sentence_ms = int(self.config.get("pause_between_sentences", 0.05) * 1000)
        pause_semicolon_ms = int(self.config.get("pause_semicolon", 0.1) * 1000)
        pause_colon_ms = int(self.config.get("pause_colon", 0.12) * 1000)
        pause_comma_ms = int(self.config.get("pause_comma", 0.05) * 1000)
        pause_dash_ms = int(self.config.get("pause_dash", 0.2) * 1000)
        
        # Разбиваем текст на сегменты по языкам
        segments = LanguageDetector.split_by_language(text)
        all_audio = []
        
        for lang, seg_text in segments:
            # Заменяем скобки на запятые
            seg_text = seg_text.replace('(', ', ').replace(')', ', ')
            cleaned = TextCleaner.clean_for_tts(seg_text)
            if not cleaned.strip() or len(cleaned.strip()) < 2:
                continue
            
            if lang == self.ext_lang and self.tts.is_model_available(self.ext_lang):
                # Английский: без SSML, обычный синтез
                chunks = split_long_text(cleaned, self.max_chunk_length)
                for chunk in chunks:
                    audio = self.tts.synthesize(chunk, self.ext_lang, self.ext_speaker)
                    audio = audio * self.speaker_gains[self.ext_lang]
                    if len(audio) > 0:
                        all_audio.append(audio)
            else:
                # Русский: вставляем SSML-паузы
                ssml_text = cleaned
                ssml_text = ssml_text.replace('; ', f';<break time="{pause_semicolon_ms}ms"/> ')
                ssml_text = ssml_text.replace(': ', f':<break time="{pause_colon_ms}ms"/> ')
                ssml_text = ssml_text.replace(', ', f',<break time="{pause_comma_ms}ms"/> ')
                ssml_text = ssml_text.replace('<dash>', f'<break time="{pause_dash_ms}ms"/>')
                
                # Транслитерация латиницы
                if LanguageDetector.has_latin(ssml_text):
                    ssml_text = transliterate_to_cyrillic(ssml_text)
                
                # Разбиваем на чанки
                chunks = self._split_ssml_text(ssml_text, self.max_chunk_length)
                
                for chunk in chunks:
                    chunk_ssml = f'<speak>{chunk}</speak>'
                    audio = self.tts.synthesize_ssml(chunk_ssml, "ru", self.ru_speaker)
                    audio = audio * self.speaker_gains["ru_main"]
                    if len(audio) > 0:
                        all_audio.append(audio)
        
        if all_audio:
            return np.concatenate(all_audio)
        return np.array([], dtype=np.float32)

    def _split_ssml_text(self, ssml_text: str, max_length: int) -> List[str]:
        """Разбивает SSML-текст на чанки, не разрезая теги."""
        if len(ssml_text) <= max_length:
            return [ssml_text] if ssml_text.strip() else []
        
        chunks = []
        remaining = ssml_text
        
        while len(remaining) > max_length:
            chunk = remaining[:max_length]
            
            # Ищем последний закрытый тег </break> в чанке
            last_break_end = chunk.rfind('</break>')
            
            if last_break_end != -1:
                # Разрезаем после закрытого тега
                split_pos = last_break_end + len('</break>')
            else:
                # Ищем конец предложения
                split_pos = _find_best_split_position(chunk, max_length)
                if split_pos <= 0:
                    split_pos = max_length
            
            part = remaining[:split_pos].strip()
            if part and len(part) >= 2:
                chunks.append(part)
            
            remaining = remaining[split_pos:].strip()
        
        if remaining and len(remaining) >= 2:
            chunks.append(remaining)
        
        return chunks

    def _synthesize_with_forced_pauses(self, text: str) -> np.ndarray:
        """Синтезирует текст с принудительными паузами по знакам препинания."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Заменяем скобки на запятые (для пауз)
        text = text.replace('(', ', ').replace(')', ', ')
        
        # Разбиваем текст на сегменты по знакам препинания
        segments = []
        current = ""
        
        for char in text:
            current += char
            
            if char == ',':
                segments.append((current, self.config.get("pause_comma", 0.05)))
                current = ""
            elif char == '<dash>':
                segments.append((current, self.config.get("pause_dash", 0.2)))
                current = ""
            elif char == ';':
                segments.append((current, self.config.get("pause_semicolon", 0.1)))
                current = ""
            elif char == ':':
                segments.append((current, self.config.get("pause_colon", 0.12)))
                current = ""
            elif char in '.!?…':
                # Проверяем, что это действительно конец предложения
                if (len(current) >= 2 and 
                    (current[-2].isalpha() or current[-2].isdigit())):
                    segments.append((current, self.config.get("pause_between_sentences", 0.05)))
                    current = ""
        
        if current.strip():
            segments.append((current, 0))
        
        # Устраняем накопление пауз: оставляем только самую длинную
        cleaned_segments = []
        for seg_text, pause_duration in segments:
            if not seg_text.strip():
                # Пустой сегмент — только пауза. Проверяем предыдущий.
                if cleaned_segments and cleaned_segments[-1][1] > 0:
                    # Предыдущий тоже пустой с паузой — оставляем большую
                    prev_text, prev_pause = cleaned_segments[-1]
                    if pause_duration > prev_pause:
                        cleaned_segments[-1] = (prev_text, pause_duration)
                    continue
                else:
                    # Предыдущий не пустой — добавляем паузу к нему
                    if cleaned_segments:
                        prev_text, prev_pause = cleaned_segments[-1]
                        cleaned_segments[-1] = (prev_text, max(prev_pause, pause_duration))
                    continue
            else:
                # Непустой сегмент
                cleaned_segments.append((seg_text, pause_duration))
        
        segments = cleaned_segments
        
        # Синтезируем каждый сегмент
        all_audio = []
        
        for seg_text, pause_duration in segments:
            if not seg_text.strip():
                continue
            
            # Определяем язык сегмента
            lang_segments = LanguageDetector.split_by_language(seg_text)
            
            for lang, lang_text in lang_segments:
                cleaned = TextCleaner.clean_for_tts(lang_text)
                if not cleaned.strip() or len(cleaned.strip()) < 2:
                    continue
                
                chunks = split_long_text(cleaned, self.max_chunk_length)
                for chunk in chunks:
                    if lang == self.ext_lang and self.tts.is_model_available(self.ext_lang):
                        if self.en_pause_before > 0:
                            all_audio.append(self.add_silence(self.en_pause_before))
                        audio = self.tts.synthesize(chunk, self.ext_lang, self.ext_speaker)
                        audio = audio * self.speaker_gains[self.ext_lang]
                        if self.en_speed != 1.0:
                            audio = change_speed_audio(audio, self.sample_rate, self.en_speed)
                        if len(audio) > 0:
                            all_audio.append(audio)
                        if self.en_pause_after > 0:
                            all_audio.append(self.add_silence(self.en_pause_after))
                    else:
                        chunk_to_synth = chunk
                        if LanguageDetector.has_latin(chunk):
                            chunk_to_synth = transliterate_to_cyrillic(chunk)
                        audio = self.tts.synthesize(chunk_to_synth, "ru", self.ru_speaker)
                        audio = audio * self.speaker_gains["ru_main"]
                        if len(audio) > 0:
                            all_audio.append(audio)
            
            # Добавляем принудительную паузу после сегмента
            if pause_duration > 0:
                all_audio.append(self.add_silence(pause_duration))
        
        if all_audio:
            return np.concatenate(all_audio)
        return np.array([], dtype=np.float32)

    def synthesize_emphasis(self, text: str) -> np.ndarray:
        """Синтезирует курсив с отдельным голосом и скоростью."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Если SSML включен — используем SSML-выделение
        if self.config.get("ssml_enabled", False):
            return self._synthesize_emphasis_ssml(text)
        
        speaker = self.emphasis_speaker or self.ru_speaker
        speed = self.emphasis_speed
        
        try:
            audio = self.tts.synthesize(text, "ru", speaker)
            if len(audio) == 0:
                return np.array([], dtype=np.float32)
            
            audio = audio * self.speaker_gains.get("ru_main", 1.0)
            
            if speed != 1.0:
                audio = change_speed_audio(audio, self.sample_rate, speed)
            
            # Добавляем паузы до и после курсива
            result = []
            if self.emphasis_pause_before > 0:
                result.append(self.add_silence(self.emphasis_pause_before))
            result.append(audio)
            if self.emphasis_pause_after > 0:
                result.append(self.add_silence(self.emphasis_pause_after))
            
            return np.concatenate(result) if len(result) > 1 else audio
        except Exception:
            return np.array([], dtype=np.float32)

    def _synthesize_emphasis_ssml(self, text: str) -> np.ndarray:
        """Синтезирует курсив с выделением через SSML (prosody rate/pitch)."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Определяем скорость для emphasis
        speed = self.emphasis_speed
        rate_map = {
            0.25: "x-slow",
            0.5: "slow",
            0.75: "slow",
            1.0: "medium",
            1.25: "fast",
            1.5: "fast",
            2.0: "x-fast",
        }
        rate = rate_map.get(speed, "medium")
        
        # Определяем тон
        pitch = self.config.get("emphasis_pitch", "medium")
        
        # Определяем паузы
        pause_before_ms = int(self.emphasis_pause_before * 1000)
        pause_after_ms = int(self.emphasis_pause_after * 1000)
        
        # Формируем SSML
        ssml_parts = []
        if pause_before_ms > 0:
            ssml_parts.append(f'<break time="{pause_before_ms}ms"/>')
        ssml_parts.append(f'<prosody rate="{rate}" pitch="{pitch}">{text}</prosody>')
        if pause_after_ms > 0:
            ssml_parts.append(f'<break time="{pause_after_ms}ms"/>')
        
        ssml_text = f'<speak>{"".join(ssml_parts)}</speak>'
        
        # Синтезируем основным голосом
        audio = self.tts.synthesize_ssml(ssml_text, "ru", self.ru_speaker)
        return audio * self.speaker_gains["ru_main"]

    def add_silence(self, duration: float = 0.3) -> np.ndarray:
        """Создаёт тишину заданной длительности."""
        return np.zeros(int(self.sample_rate * duration), dtype=np.float32)

    def process_blocks(self, blocks: List[Dict]) -> List[np.ndarray]:
        """Обрабатывает блоки текста."""
        return self._process_blocks_standard(blocks)

    def _process_blocks_standard(self, blocks: List[Dict]) -> List[np.ndarray]:
        """Стандартная обработка блоков с принудительной запятой перед сноской."""
        blocks = self._merge_short_blocks(blocks)
        result_audio = []
        for i, block in enumerate(blocks):
            if block["type"] == "text":
                text = block["content"]
                if text.strip():
                    # Проверяем, есть ли следующая сноска
                    if (self.config.get("footnote_force_comma", True) and 
                        i + 1 < len(blocks) and 
                        blocks[i + 1]["type"] == "footnote"):
                        # Добавляем запятую если текст не заканчивается знаком препинания
                        if not text.rstrip().endswith(('.', ',', ';', ':', '!', '?', '…')):
                            text = text.rstrip() + ","
                    
                    audio = self.synthesize_text(text)
                    if len(audio) > 0:
                        result_audio.append(audio)
                        result_audio.append(
                            self.add_silence(self.pause_between_paragraphs)
                        )

            elif block["type"] == "emphasis":
                text = block.get("content", "")
                if text.strip():
                    if self.config.get("emphasis_as_text", False):
                        audio = self.synthesize_text(text)
                        if len(audio) > 0:
                            result_audio.append(audio)
                            # Без дополнительной паузы - emphasis как часть текста
                    else:
                        audio = self.synthesize_emphasis(text)
                        if len(audio) > 0:
                            result_audio.append(audio)
                            result_audio.append(
                                self.add_silence(self.pause_between_paragraphs)
                            )
            elif block["type"] == "footnote":
                if self.skip_footnotes:
                    continue
                note_text = block["text"]
                note_title = block.get("title", "")
                
                # Создаем аудио сноски
                footnote_audio_parts = []
                footnote_audio_parts.append(self.add_silence(
                    self.config.get("footnote_pause_before", 0.3)
                ))
                
                if note_title:
                    audio = self.synthesize_service(
                        f"{self.footnote_prefix} {note_title}"
                    )
                else:
                    audio = self.synthesize_service(self.footnote_prefix)
                
                if len(audio) > 0:
                    footnote_audio_parts.append(audio)
                    footnote_audio_parts.append(self.add_silence(0.2))
                
                if note_text:
                    audio = self.synthesize_text(note_text)
                    if len(audio) > 0:
                        footnote_audio_parts.append(audio)
                        footnote_audio_parts.append(self.add_silence(0.2))
                
                audio = self.synthesize_service(self.footnote_suffix)
                if len(audio) > 0:
                    footnote_audio_parts.append(audio)
                    footnote_audio_parts.append(self.add_silence(
                        self.config.get("footnote_pause_after", 0.3)
                    ))
                
                # Стандартная обработка: просто добавляем сноску
                result_audio.extend(footnote_audio_parts)
                
                result_audio.append(self.add_silence(0.2))

            elif block["type"] == "pause":
                result_audio.append(self.add_silence(block.get("duration", 0.3)))

        return result_audio

    def _synthesize_footnote(self, block: Dict) -> List[np.ndarray]:
        """Синтезирует сноску и возвращает список аудио-частей."""
        note_text = block["text"]
        note_title = block.get("title", "")
        
        parts = []
        parts.append(self.add_silence(
            self.config.get("footnote_pause_before", 0.3)
        ))
        
        if note_title:
            audio = self.synthesize_service(
                f"{self.footnote_prefix} {note_title}"
            )
        else:
            audio = self.synthesize_service(self.footnote_prefix)
        
        if len(audio) > 0:
            parts.append(audio)
            parts.append(self.add_silence(0.2))
        
        if note_text:
            audio = self.synthesize_text(note_text)
            if len(audio) > 0:
                parts.append(audio)
                parts.append(self.add_silence(0.2))
        
        audio = self.synthesize_service(self.footnote_suffix)
        if len(audio) > 0:
            parts.append(audio)
            parts.append(self.add_silence(
                self.config.get("footnote_pause_after", 0.3)
            ))
        
        return parts

    def _merge_short_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Объединяет короткие текстовые блоки с соседними для улучшения качества."""
        if not blocks:
            return blocks
        
        merged = []
        i = 0
        last_was_pause = False
        
        while i < len(blocks):
            block = blocks[i]
            
            # Проверяем, начинается ли блок со знака препинания
            starts_with_punct = (
                block["type"] == "text" and 
                block["content"].strip() and
                block["content"].strip()[0] in '.,;:!?…'
            )
            
            # Если это короткий текстовый блок
            if (block["type"] == "text" and 
                (len(block["content"].strip()) < 50 or starts_with_punct) and
                not block["content"].strip().endswith(('.', '!', '?', '…'))):
                
                # Проверяем, можно ли объединить с предыдущим
                if (merged and 
                    merged[-1]["type"] == "text" and
                    not last_was_pause and
                    not merged[-1]["content"].rstrip().endswith(('.', '!', '?', '…'))):
                    
                    # Объединяем с предыдущим текстовым блоком
                    prev_text = merged[-1]["content"].rstrip()
                    curr_text = block["content"].strip()
                    
                    # Определяем, нужен ли пробел
                    if (prev_text.endswith(('.', ',', ';', ':', '!', '?', '…')) or
                        curr_text.startswith(('.', ',', ';', ':', '!', '?', '…'))):
                        separator = ""
                    else:
                        separator = " "
                    
                    merged[-1]["content"] = prev_text + separator + curr_text
                    last_was_pause = False
                    i += 1
                    continue
                
                # Проверяем, можно ли объединить со следующей сноской
                if i + 1 < len(blocks) and blocks[i + 1]["type"] == "footnote":
                    # Оставляем как есть, сноска обработает это
                    merged.append(block)
                    i += 1
                    continue
                
                # Проверяем, можно ли объединить со следующим текстовым блоком
                if (i + 1 < len(blocks) and 
                    blocks[i + 1]["type"] == "text" and
                    (i + 2 >= len(blocks) or blocks[i + 2]["type"] != "pause")):
                    next_block = blocks[i + 1]
                    curr_text = block["content"].strip()
                    next_text = next_block["content"].strip()
                    
                    # Определяем, нужен ли пробел
                    if (curr_text.endswith(('.', ',', ';', ':', '!', '?', '…')) or
                        next_text.startswith(('.', ',', ';', ':', '!', '?', '…'))):
                        separator = ""
                    else:
                        separator = " "
                    
                    merged.append({
                        "type": "text",
                        "content": curr_text + separator + next_text
                    })
                    last_was_pause = False
                    i += 2
                    continue
            
            # Обычный блок
            last_was_pause = (block["type"] == "pause")
            merged.append(block)
            i += 1
        
        return merged

    def _apply_pedalboard(self, audio: np.ndarray) -> np.ndarray:
        """Постобработка через pedalboard: компрессор + эквалайзер + реверб + комнатный шум."""
        try:
            from pedalboard import (
                Compressor,
                Gain,
                HighpassFilter,
                LowpassFilter,
                PeakFilter,
                Pedalboard,
                Reverb,
            )
            from scipy import signal as scipy_signal
        except ImportError:
            logger.warning(
                "pedalboard или scipy не установлены. pip install pedalboard scipy"
            )
            return audio
        cfg = self.config
        try:
            # Подготавливаем аудио
            audio_processed = audio.copy()
            
            board = Pedalboard(
                [
                    HighpassFilter(cutoff_frequency_hz=cfg.get("pb_highpass_hz", 85)),
                    PeakFilter(
                        cutoff_frequency_hz=cfg.get("pb_deharsh_hz", 5500),
                        gain_db=cfg.get("pb_deharsh_db", -2.5),
                        q=cfg.get("pb_deharsh_q", 2.0),
                    ),
                    PeakFilter(
                        cutoff_frequency_hz=cfg.get("pb_deharsh2_hz", 7800),
                        gain_db=cfg.get("pb_deharsh2_db", -3.0),
                        q=cfg.get("pb_deharsh2_q", 2.0),
                    ),
                    LowpassFilter(cutoff_frequency_hz=cfg.get("pb_lowpass_hz", 11500)),
                    PeakFilter(
                        cutoff_frequency_hz=cfg.get("pb_warmth_hz", 280),
                        gain_db=cfg.get("pb_warmth_db", 1.8),
                        q=0.9,
                    ),
                    PeakFilter(
                        cutoff_frequency_hz=cfg.get("pb_clarity_hz", 3200),
                        gain_db=cfg.get("pb_clarity_db", 1.4),
                        q=1.1,
                    ),
                    Compressor(
                        threshold_db=cfg.get("pb_comp_threshold", -18),
                        ratio=cfg.get("pb_comp_ratio", 2.4),
                        attack_ms=cfg.get("pb_comp_attack", 18),
                        release_ms=cfg.get("pb_comp_release", 120),
                    ),
                    Reverb(
                        room_size=cfg.get("pb_reverb_room", 0.22),
                        damping=cfg.get("pb_reverb_damping", 0.55),
                        wet_level=cfg.get("pb_reverb_wet", 0.09),
                        dry_level=1.0 - cfg.get("pb_reverb_wet", 0.09),
                        width=cfg.get("pb_reverb_width", 0.6),
                    ),
                    Gain(gain_db=cfg.get("pb_gain_db", 0.3)),
                ]
            )
            processed = board(audio_processed[np.newaxis, :], self.sample_rate)
            result = processed[0]
            room_level = cfg.get("pedalboard_room_tone", 0)
            if room_level < 0:
                white = np.random.randn(len(result)).astype(np.float32)
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                pink = scipy_signal.lfilter(b, a, white)
                pink = pink / (np.max(np.abs(pink)) + 1e-10) * (10 ** (room_level / 20))
                result = result + pink.astype(np.float32)
            return result.astype(np.float32)
        except Exception as e:
            logger.warning(f"Pedalboard error: {e}")
            return audio


class FB2ToAudioConverter:
    """Основной конвертер FB2 в аудио."""

    def __init__(self, config: dict):
        self.config = config
        self.tts = TTSProcessor(config)
        self.sample_rate = config["sample_rate"]
        self.pause_between_chapters = config.get("pause_between_chapters", 1.5)
        self.blocks_per_part = config.get("blocks_per_part", 100)

    def convert_file_streaming(
        self, input_file: str, output_file: str, work_dir: str = None
    ) -> bool:
        """Конвертирует файл в потоковом режиме."""
        start_time = time.time()
        logger.info(f"Обработка (потоковая): {input_file}")

        parser = FB2Parser(input_file)
        parser.force_punctuation = self.config.get("force_punctuation", False)
        parser.pause_between_subsections = self.config.get("pause_between_subsections", 0.8)
        parser.annotation_prefix = self.config.get("annotation_prefix", "Аннотация")
        parser.annotation_suffix = self.config.get("annotation_suffix", "Конец аннотации")
        chapters = parser.parse()
        if not chapters:
            logger.error("Не удалось извлечь текст")
            return False

        if work_dir is None:
            work_dir = Path(input_file).stem + "_parts"
        work_path = Path(work_dir)
        work_path.mkdir(exist_ok=True)

        all_blocks = []
        for chapter_blocks in chapters:
            all_blocks.extend(chapter_blocks)
            all_blocks.append(
                {"type": "pause", "duration": self.pause_between_chapters}
            )

        total_parts = (
            len(all_blocks) + self.blocks_per_part - 1
        ) // self.blocks_per_part
        part_files = []

        logger.info(f"Всего блоков: {len(all_blocks)}, частей: {total_parts}")

        try:
            for part_idx in range(total_parts):
                start = part_idx * self.blocks_per_part
                end = min(start + self.blocks_per_part, len(all_blocks))
                part_blocks = all_blocks[start:end]

                logger.info(
                    f"Часть {part_idx + 1}/{total_parts}: блоки {start + 1}-{end}"
                )

                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed // 3600):02}:{int((elapsed % 3600) // 60):02}:{int(elapsed % 60):02}"
                if part_idx > 0:
                    eta = (elapsed / (part_idx + 1)) * (total_parts - part_idx - 1)
                    eta_str = f"{int(eta // 3600):02}:{int((eta % 3600) // 60):02}:{int(eta % 60):02}"
                else:
                    eta_str = "--:--:--"

                sys.stderr.write(
                    f"\r{GREEN}Часть:{RESET} {YELLOW}{part_idx + 1}/{total_parts}{RESET} "
                    f"{GREEN}Прошло:{RESET} {YELLOW}{elapsed_str}{RESET} "
                    f"{GREEN}Осталось:{RESET} {YELLOW}{eta_str}{RESET} "
                    f"{GREEN}Блоки:{RESET} {YELLOW}{start + 1}-{end}{RESET}/{YELLOW}{len(all_blocks)}{RESET}  "
                )
                sys.stderr.flush()

                part_audio_parts = self.tts.process_blocks(part_blocks)
                try:
                    if part_audio_parts:
                        non_empty_parts = [p for p in part_audio_parts if len(p) > 0]
                        if non_empty_parts:
                            part_audio = np.concatenate(non_empty_parts)
                            if self.config.get("pedalboard_enabled"):
                                part_audio = self.tts._apply_pedalboard(part_audio)
                            part_file = work_path / f"part_{part_idx + 1:04d}.ogg"
                            if save_audio(part_audio, str(part_file), self.sample_rate):
                                part_files.append(part_file)
                            del part_audio
                        del non_empty_parts
                finally:
                    del part_audio_parts

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            sys.stderr.write("\n")
            sys.stderr.flush()
            final_phrase = self.config.get("final_phrase", "")
            if final_phrase:
                logger.info("Финальная фраза...")
                audio = self.tts.synthesize_service(final_phrase)
                if len(audio) > 0:
                    silence = self.tts.add_silence(1.0)
                    final_audio = np.concatenate([silence, audio])
                    final_file = work_path / "part_final.ogg"
                    if save_audio(final_audio, str(final_file), self.sample_rate):
                        part_files.append(final_file)
                    del final_audio
                del audio

            if not part_files:
                logger.error("Не удалось создать ни одной части")
                return False

            logger.info(f"Склейка {len(part_files)} OGG-частей...")
            ffmpeg_filter = self.config.get("ffmpeg_filter")
            if not concat_ogg_files(
                part_files,
                output_file,
                ffmpeg_filter,
                self.config.get("vorbis_quality", 6),
            ):
                logger.error("Не удалось склеить части")
                return False

            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 3600):02}:{int((elapsed % 3600) // 60):02}:{int(elapsed % 60):02}"
            sys.stderr.write(
                f"\r{GREEN}Готово:{RESET} {YELLOW}{elapsed_str}{RESET} "
                f"{GREEN}Частей:{RESET} {YELLOW}{len(part_files)}{RESET} "
                f"{GREEN}Блоков:{RESET} {YELLOW}{len(all_blocks)}{RESET} "
                f"{GREEN}Файл:{RESET} {YELLOW}{os.path.getsize(output_file) / 1024 / 1024:.0f} MB{RESET}"
                f"{' ' * 20}\n"
            )
            sys.stderr.flush()

        finally:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                for f in part_files:
                    try:
                        os.remove(f)
                    except:
                        pass
                try:
                    work_path.rmdir()
                except:
                    pass
            else:
                logger.warning(
                    f"Выходной файл не создан, части сохранены в {work_path}"
                )

        return True


def print_header(
    config: dict,
    input_file: str,
    output_file: str,
    ru_speaker: str,
    service_speaker: str,
):
    """Выводит заголовок с информацией о конвертации."""
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    threads = config.get("parallel_threads", 1)
    loudness = config.get("loudness_target")
    speed = config.get("speed", 1.0)
    ext_model = config.get("en_model", "v3_en")
    ext_speaker = config.get("en_speaker", "en_0")
    # emp_spk = config.get("emphasis_speaker") or ru_speaker
    # emp_speed = config.get("emphasis_speed", 1.0)
    en_speed = config.get("en_speed", 1.0)
    # pp = config.get("pause_between_paragraphs", 0.2)
    # ps = config.get("pause_between_sentences", 0.05)
    # psc = config.get("pause_semicolon", 0.0)
    # pcl = config.get("pause_colon", 0.0)
    skip_fn = config.get("skip_footnotes", False)
    emphasis_as_text = config.get("emphasis_as_text", False)

    def v(text):
        return f"{YELLOW}{text}{RESET}"

    device_name = device.upper()
    if device == "cuda":
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device == "cpu":
        device_name = f"CPU ({threads} потоков)"

    sys.stderr.write(
        f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}║{RESET} {BOLD}FB2 to Audio Converter v{VERSION}{RESET}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════╣{RESET}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}║{RESET} {GREEN}Вход:{RESET} {v(Path(input_file).name):<30} {GREEN}Выход:{RESET} {v(Path(output_file).name)}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}║{RESET} {GREEN}RU:{RESET} {v(config.get('ru_model','?'))} \x7b {v(ru_speaker):<15} \x7d  {GREEN}Служ.:{RESET} {v(service_speaker)}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}║{RESET} {GREEN}EX:{RESET} {v(ext_model)} \x7b {v(ext_speaker):<12} \x7d {GREEN}Скор.:{RESET} {v(str(en_speed)+'x')}\n"
    )
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}Устр-во:{RESET} {v(device_name)}\n")
    sys.stderr.write(
        f"{BOLD}{CYAN}║{RESET} {GREEN}Частота:{RESET} {v(str(config.get('sample_rate', 48000))+' Гц')}  {GREEN}Скор.:{RESET} {v(str(speed)+'x')}   {GREEN}Громк.:{RESET} {v(str(loudness)+' LUFS' if loudness else 'без')}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}║{RESET} {GREEN}SSML:{RESET} {v('вкл' if config.get('ssml_enabled', False) else 'выкл')}    {GREEN}Сноски:{RESET} {v('пропущены' if skip_fn else 'озвучены')}"
    )
    sys.stderr.write(
        f"    {GREEN}Курсив:{RESET} {v('не выделен' if emphasis_as_text else 'выделен')}\n"
    )
    sys.stderr.write(
        f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}\n"
    )
    sys.stderr.flush()


def list_speakers(config: dict):
    """Выводит список доступных голосов."""
    list_config = config.copy()
    list_config["device"] = "cpu"
    tts = SileroTTS(list_config)
    print(f"\nРусские голоса для '{config['ru_model']}':")
    for i, s in enumerate(tts.get_ru_speakers(), 1):
        print(f"  {i:2d}. {s}")
    ext_lang = tts._lang_from_en_model()
    if tts.is_model_available("en"):
        print(f"\nГолоса для '{config['en_model']}' ({ext_lang}):")
        speakers_list = tts.speakers.get(ext_lang, [])
        for i, s in enumerate(speakers_list, 1):
            print(f"  {i:3d}. {s}")
            if i % 20 == 0:
                print()
    else:
        print(f"\nМодель '{config['en_model']}' не загружена")


def list_models():
    """Выводит список доступных моделей."""
    print(
        """
Русские (авто-ударения): v5_5_ru, v5_4_ru, v5_3_ru, v5_2_ru, v5_1_ru, v5_ru
Русские (ручные ударения, v5_5_ru): v5_5_ru_manual
Русские (ручные ударения): v5_cis_base, v5_cis_base_nostress, v5_cis_ext
Русские (v4/v3): v4_ru, v3_1_ru, ru_v3
Дополнительные: v3_en, v3_de, v3_fr, v3_es, v3_en_indic
"""
    )


def setup_logging(config: dict, args):
    """Настраивает логирование."""
    log_level = logging.DEBUG if args.debug else logging.INFO

    logger.handlers.clear()
    logger.setLevel(log_level)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = str(log_dir / Path(args.input).with_suffix(".log").name)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    if args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(console_handler)

    if args.config:
        logger.info(f"Конфиг: {args.config}")

    logger.debug(f"Версия: {VERSION}")
    logger.debug(f"Python: {sys.version}")
    logger.debug(f"Torch: {torch.__version__}")
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")


def dry_run(input_file: str, config: dict):
    """Проверяет текст без синтеза."""
    print(f"\n{BOLD}{CYAN}DRY RUN: проверка текста{RESET}\n")

    parser = FB2Parser(input_file)
    parser.force_punctuation = config.get("force_punctuation", False)
    parser.pause_between_subsections = config.get("pause_between_subsections", 0.8)
    chapters = parser.parse()

    if not chapters:
        print("❌ Не удалось извлечь текст")
        return

    total_blocks = 0
    total_text_blocks = 0
    total_footnotes = 0
    total_chars = 0
    en_chars = 0

    for i, chapter in enumerate(chapters, 1):
        print(f"\n{BOLD}Глава {i}:{RESET} {len(chapter)} блоков")
        if i > 1:
            print(f"  ⏸  Пауза между главами: {config.get('pause_between_chapters', 1.5)}с")
        for block in chapter:
            total_blocks += 1
            if block["type"] == "text":
                total_text_blocks += 1
                text = block["content"]
                total_chars += len(text)
                # Проверяем наличие tts_en тегов
                if "<tts_en>" in text:
                    print(f"  🌍 EN текст: {text[:100]}...")
                if LanguageDetector.has_latin(text):
                    en_chars += len(text)
                if len(text) > 50:
                    print(f"  📝 {text[:50]}...")
                else:
                    print(f"  📝 {text}")
            elif block["type"] == "emphasis":
                print(f"  🖊️  {block.get('content', '')[:50]}")
            elif block["type"] == "footnote":
                total_footnotes += 1
                print(f"  📌 Сноска: {block.get('title', '')[:50]}")
            elif block["type"] == "pause":
                print(f"  ⏸️  Пауза: {block.get('duration', 0)}с")

    blocks_per_part = config.get("blocks_per_part", 100)
    total_parts = (total_blocks + blocks_per_part - 1) // blocks_per_part

    print(f"\n{BOLD}{CYAN}═══ Статистика ═══{RESET}")
    print(f"Всего глав:          {len(chapters)}")
    print(f"Всего блоков:        {total_blocks}")
    print(f"Текстовых блоков:    {total_text_blocks}")
    print(f"Сносок:              {total_footnotes}")
    print(f"Всего символов:      {total_chars}")
    print(f"Латиницы:            {en_chars} ({100*en_chars/max(1,total_chars):.1f}%)")
    print(
        f"Кириллицы:           {total_chars - en_chars} ({100*(total_chars-en_chars)/max(1,total_chars):.1f}%)"
    )
    print(f"Частей для синтеза:  {total_parts}")
    print(f"Блоков в части:      {blocks_per_part}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="FB2 в аудио через Silero TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="FB2 файл (нормализованный)")
    parser.add_argument("-o", "--output", help="Выходной файл")
    parser.add_argument("--config", help="Конфиг (YAML или JSON)")
    parser.add_argument("--ru-model", help="Русская модель")
    parser.add_argument("--en-model", help="Дополнительная модель (en/de/fr/es)")
    parser.add_argument("--ru-speaker", help="Основной голос")
    parser.add_argument("--en-speaker", help="Голос для дополнительного языка")
    parser.add_argument("--service-speaker", help="Служебный голос")
    parser.add_argument("--sample-rate", type=int, help="Частота")
    parser.add_argument("--format", choices=["ogg", "wav"], help="Формат")
    parser.add_argument("--speed", type=float, help="Скорость (0.25-4.0)")
    parser.add_argument(
        "--fetch-models",
        nargs="+",
        help="Принудительно загрузить модели (например: v5_ru v5_5_ru v3_en v3_de)",
    )
    parser.add_argument(
        "--loudness", type=float, help="Целевая громкость LUFS для калибровки"
    )
    parser.add_argument(
        "--no-calibrate", action="store_true", help="Отключить калибровку громкости"
    )
    parser.add_argument(
        "--skip-footnotes", action="store_true", help="Не озвучивать сноски"
    )
    parser.add_argument("--cpu", action="store_true", help="CPU")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Устройство")
    parser.add_argument("--parallel", type=int, default=None, help="Число потоков")
    parser.add_argument("--list-ru-speakers", action="store_true")
    parser.add_argument("--list-en-speakers", action="store_true")
    parser.add_argument(
        "--list-de-speakers", action="store_true", help="Показать немецкие голоса"
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Проверить текст без синтеза"
    )
    parser.add_argument("--debug", action="store_true", help="Расширенный вывод")

    args = parser.parse_args()

    if not args.config:
        for default_config_name in ["config.yaml", "config.yml", "config.json"]:
            if os.path.exists(default_config_name):
                args.config = default_config_name
                break

    config = load_config(args.config)

    for key in [
        "ru_model",
        "en_model",
        "ru_speaker",
        "en_speaker",
        "service_speaker",
        "sample_rate",
        "format",
        "speed",
    ]:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val

    if args.loudness is not None:
        config["loudness_target"] = args.loudness
    if args.no_calibrate:
        config["loudness_target"] = None
    if args.skip_footnotes:
        config["skip_footnotes"] = True
    if args.cpu:
        config["device"] = "cpu"
    if args.device:
        config["device"] = args.device
    if args.parallel is not None:
        config["parallel_threads"] = args.parallel

    speed = config.get("speed", 1.0)
    if speed <= 0:
        print("Скорость должна быть > 0")
        return 1
    if speed > 4.0:
        config["speed"] = 4.0
    elif speed < 0.25:
        config["speed"] = 0.25

    if args.list_models:
        list_models()
        return 0
    if args.list_ru_speakers or args.list_en_speakers or args.list_de_speakers:
        if args.list_de_speakers:
            config["en_model"] = "v3_de"
        list_speakers(config)
        return 0
    if not args.input:
        parser.print_help()
        return 0
    if not os.path.exists(args.input):
        print(f"Файл не найден: {args.input}")
        return 1

    setup_logging(config, args)

    output_file = args.output or str(
        Path(args.input).with_suffix(f".{config.get('format', 'ogg')}")
    )

    if args.dry_run:
        dry_run(args.input, config)
        return 0

    # Принудительная загрузка моделей
    if args.fetch_models:
        logger.info(f"Принудительная загрузка моделей: {args.fetch_models}")
        temp_config = config.copy()
        temp_config["device"] = "cpu"  # загружаем на CPU для скорости
        temp_tts = SileroTTS(temp_config)
        
        for model_name in args.fetch_models:
            success = temp_tts.fetch_model(model_name)
            if success:
                print(f"{GREEN}✓{RESET} Модель {model_name} загружена")
            else:
                print(f"{RED}✗{RESET} Не удалось загрузить модель {model_name}")
        
        # Очищаем временную модель
        del temp_tts
        gc.collect()

    try:
        converter = FB2ToAudioConverter(config)
        print_header(
            config,
            args.input,
            output_file,
            converter.tts.ru_speaker,
            converter.tts.service_speaker,
        )
        return 0 if converter.convert_file_streaming(args.input, output_file) else 1
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
        logger.info("Прервано пользователем")
        return 1
    except Exception as e:
        print(f"Ошибка: {e}")
        logger.exception("Критическая ошибка")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
