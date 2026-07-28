#!/usr/bin/env python3
"""
TTS конвертер FB2 в аудио (OGG/WAV) с поддержкой сносок и ударений
Использует Silero TTS для озвучивания текста

Версия: 1.0.0

ТРЕБОВАНИЯ:
  pip install torch torchaudio silero-tts numpy pyyaml
  sudo apt install ffmpeg

  Перед использованием обязательно нормализовать FB2:
  python normalize_fb2.py book.fb2
"""

VERSION = "1.0.0"

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

import numpy as np

try:
    import torch
    import torchaudio
except ImportError as e:
    print(f"Не установлены необходимые пакеты: {e}")
    print("Установите: pip install torch torchaudio")
    exit(1)

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "ru_model": "v5_cis_base_nostress",
    "en_model": "v3_en",
    "ru_speaker": None,
    "en_speaker": "en_0",
    "service_speaker": None,
    "sample_rate": 48000,
    "format": "ogg",
    "speed": 1.0,
    "device": "auto",
    "pause_between_paragraphs": 0.2,
    "pause_between_sentences": 0.05,
    "pause_between_chapters": 1.5,
    "footnote_prefix": "Сноска",
    "footnote_suffix": "Конец сноски",
    "final_phrase": "Конец озвученного текста.",
    "skip_footnotes": False,
    "loudness_target": -23.0,
    "parallel_threads": 1,
    "max_chunk_length": 900,
    "blocks_per_part": 100,
    # Настройки для выделения курсива/emphasis
    "emphasis_pause_before": 0.0,
    "emphasis_pause_after": 0.0,
    "emphasis_speaker": None,
    "emphasis_speed": 1.0,
    # Настройки для английского текста
    "en_pause_before": 0.0,
    "en_pause_after": 0.0,
    "en_speed": 1.0,
    # Постобработка ffmpeg (эквалайзер/компрессор)
    "ffmpeg_filter": None,
    "vorbis_quality": 6,  # при ffmpeg_filter: битрейт opus = vorbis_quality * 32 kbps
    "filter_threads": 4,  # потоков для аудиофильтров ffmpeg
    # Постобработка pedalboard (рекомендуется)
    "pedalboard_enabled": False,
    "pedalboard_room_tone": -48,      # уровень комнатного шума dB (0 = выкл)
    "pb_highpass_hz": 85,             # обрезка инфраниза
    "pb_lowpass_hz": 11500,           # обрезка высоких (песок)
    "pb_warmth_hz": 280,              # частота "тепла"
    "pb_warmth_db": 1.8,              # усиление тепла
    "pb_clarity_hz": 3200,            # частота ясности
    "pb_clarity_db": 1.4,             # усиление ясности
    "pb_comp_threshold": -18,         # порог компрессора dB
    "pb_comp_ratio": 2.4,             # ratio компрессора
    "pb_comp_attack": 18,             # атака компрессора ms
    "pb_comp_release": 120,           # релиз компрессора ms
    "pb_reverb_room": 0.22,           # размер комнаты (0-1)
    "pb_reverb_damping": 0.55,        # damping реверба (0-1)
    "pb_reverb_wet": 0.09,            # уровень wet реверба (0-1)
    "pb_reverb_width": 0.6,           # ширина реверба (0-1)
    "pb_gain_db": 0.3,                # финальный гейн dB
    "pb_deharsh_hz": 5500,            # частота подавления резонансов
    "pb_deharsh_db": -2.5,            # ослабление резонансов dB
    "pb_deharsh2_hz": 7800,           # вторая частота подавления
    "pb_deharsh2_db": -3.0,           # ослабление dB
}

STRESS_MARK = "\u0301"
STRESS_PATTERN = re.compile(r"([аеёиоуыэюяАЕЁИОУЫЭЮЯ])" + STRESS_MARK, re.IGNORECASE)

MODEL_TYPES = {
    "v5_5_ru": "auto", "v5_4_ru": "auto", "v5_3_ru": "auto",
    "v5_2_ru": "auto", "v5_1_ru": "auto", "v5_ru": "auto",
    "v5_cis_base": "manual", "v5_cis_base_nostress": "manual",
    "v5_cis_ext": "manual",
    "v4_ru": "manual", "ru_v3": "manual", "v3_1_ru": "manual",
}

LANG_FROM_MODEL = {"v3_en": "en", "v3_de": "de", "v3_fr": "fr", "v3_es": "es", "v3_en_indic": "en"}

RU_SPEAKERS_STANDARD = ["aidar", "baya", "kseniya", "xenia", "eugene"]
RU_SPEAKERS_CIS = [
    "ru_aigul", "ru_albina", "ru_alexandr", "ru_bogdan", "ru_dmitriy",
    "ru_ekaterina", "ru_eduard", "ru_gamat", "ru_igor", "ru_karina",
    "ru_kejilgan", "ru_kermen", "ru_larisa", "ru_marat", "ru_miyau",
    "ru_nurgul", "ru_oksana", "ru_onaoy", "ru_ramilia", "ru_roman",
    "ru_safarhuja", "ru_saida", "ru_sibday", "ru_vika", "ru_zara",
    "ru_zhadyra", "ru_zhazira", "ru_zinaida"
]

def save_audio(audio: np.ndarray, filepath: str, sample_rate: int):
    """Сохраняет аудио в файл."""
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
    return config


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


def concat_ogg_files(input_files: List[Path], output_file: str, ffmpeg_filter: str = None) -> bool:
    """Склеивает OGG файлы через ffmpeg с опциональной постобработкой."""
    concat_file = Path(output_file).with_suffix(".concat.txt")
    with open(concat_file, "w") as f:
        for filepath in input_files:
            path = Path(filepath) if not isinstance(filepath, Path) else filepath
            f.write(f"file '{path.absolute()}'\n")
    if ffmpeg_filter:
        # Склейка + фильтр + перекодирование в ogg
        q = str(config.get("vorbis_quality", 6)) if 'config' in dir() else "6"
        args = ["-f", "concat", "-safe", "0", "-i", str(concat_file), "-af", ffmpeg_filter, "-c:a", "libvorbis", "-q:a", q, output_file]
        desc = f"Склеено {len(input_files)} OGG-файлов с фильтром: {ffmpeg_filter}"
    else:
        args = ["-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", output_file]
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
        split_pos = -1
        for pattern in [r"[.!?]\s+", r"[.!?…]+\s*", r":\s+", r";\s+", r",\s+", r"\s+"]:
            matches = list(re.finditer(pattern, chunk))
            if matches:
                split_pos = matches[-1].end()
                break
        if split_pos == -1 or split_pos == 0:
            split_pos = max_length
        part = remaining[:split_pos].strip()
        if part and len(part) >= 2:
            parts.append(part)
        remaining = remaining[split_pos:].strip()
    if remaining and len(remaining) >= 2:
        parts.append(remaining)
    return parts

class TextCleaner:
    """Очистка и подготовка текста для TTS."""

    @staticmethod
    def clean_for_tts(text: str) -> str:
        """Очищает текст от неподдерживаемых символов."""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        replacements = {
            "…": "...", "—": "-", "–": "-",
            "\u2019": "'", "\u2018": "'",
            "\u201c": '"', "\u201d": '"',
            "\u00ab": '"', "\u00bb": '"',
            "&nbsp;": " ", "&mdash;": "-", "&laquo;": '"', "&raquo;": '"',
            "&amp;": "&", "&lt;": "<", "&gt;": ">",
            "//": " ", "/*": " ", "*/": " ",  # убираем шипящие комбинации
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        result = []
        for ch in text:
            cat = unicodedata.category(ch)
            if cat.startswith("L") or cat.startswith("N") or cat.startswith("Z") or cat.startswith("M"):
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
                    annotation = title_info.find(f"{{{self.namespace}}}annotation")
                    if annotation is not None:
                        anno_blocks = []
                        for elem in annotation:
                            self._process_element(elem, anno_blocks)
                        if anno_blocks:
                            chapters.append(anno_blocks)
            for body in root.findall(f"{{{self.namespace}}}body"):
                if body.get("name") == "notes":
                    continue
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
        if element.tag in [f"{{{self.namespace}}}binary", f"{{{self.namespace}}}description"]:
            return
        if element.tag == f"{{{self.namespace}}}epigraph":
            full_text = self._extract_text(element)
            if full_text.strip():
                blocks.append({"type": "text", "content": self._norm(full_text.strip())})
            return
        if element.tag in [
            f"{{{self.namespace}}}p", f"{{{self.namespace}}}title",
            f"{{{self.namespace}}}subtitle", f"{{{self.namespace}}}cite",
            f"{{{self.namespace}}}emphasis", f"{{{self.namespace}}}strong",
            f"{{{self.namespace}}}annotation", f"{{{self.namespace}}}text-author",
            f"{{{self.namespace}}}poem", f"{{{self.namespace}}}stanza",
            f"{{{self.namespace}}}v",
        ]:
            self._extract_text_with_footnotes(element, blocks)
        elif element.tag == f"{{{self.namespace}}}section":
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
            if child.tag == "tts_en" or (child.tag.startswith("{") and child.tag.endswith("}tts_en")):
                inner = self._extract_text(child).strip()
                if inner:
                    blocks.append({"type": "text", "content": f"<tts_en>{inner}</tts_en>"})
                # tail добавляем к предыдущему блоку если он текстовый, иначе отдельно
                if child.tail and child.tail.strip():
                    tail_text = self._norm(child.tail)
                    if blocks and blocks[-1]["type"] == "text":
                        blocks[-1]["content"] += tail_text
                    else:
                        blocks.append({"type": "text", "content": tail_text})
                continue
            if child.tag == f"{{{self.namespace}}}a":
                link_type = child.get("type")
                if link_type == "note":
                    href = child.get(f"{{{self.xlink_namespace}}}href", "")
                    note_id = href[1:] if href.startswith("#") else href
                    if note_id in self.footnotes_map:
                        note_title = self._extract_text(child).strip()
                        blocks.append({
                            "type": "footnote",
                            "id": note_id,
                            "text": self.footnotes_map[note_id],
                            "title": self._norm(note_title),
                        })
                else:
                    link_text = self._extract_text(child)
                    if link_text.strip():
                        blocks.append({"type": "text", "content": self._norm(link_text)})
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
            if child.tag == "tts_en" or (child.tag.startswith("{") and child.tag.endswith("}tts_en")):
                inner = self._extract_text(child).strip()
                if inner:
                    parts.append(f"<tts_en>{inner}</tts_en>")
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
                inner = part[len("<tts_en>"):-len("</tts_en>")]
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
        result = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker=self.ru_model_name,
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
            self.models["en"].to(self.device)
            if hasattr(self.models["en"], "eval"):
                self.models["en"].eval()
        except Exception:
            self.models["en"] = None
            logger.warning(f"Не удалось загрузить модель {self.en_model_name} ({ext_lang})")

    def synthesize(self, text: str, language: str = "ru", speaker: str = None) -> np.ndarray:
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
                processed_text = TextCleaner.convert_stress_for_model(text, self.ru_model_type)

            audio = model.apply_tts(
                text=processed_text, speaker=speaker, sample_rate=self.sample_rate
            )

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

def transliterate_to_cyrillic(text: str) -> str:
    """Транслитерация латиницы в кириллицу для fallback."""
    mapping = {
        'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф',
        'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й', 'k': 'к', 'l': 'л',
        'm': 'м', 'n': 'н', 'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
        's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс',
        'y': 'ы', 'z': 'з',
        'A': 'А', 'B': 'Б', 'C': 'К', 'D': 'Д', 'E': 'Е', 'F': 'Ф',
        'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Й', 'K': 'К', 'L': 'Л',
        'M': 'М', 'N': 'Н', 'O': 'О', 'P': 'П', 'Q': 'К', 'R': 'Р',
        'S': 'С', 'T': 'Т', 'U': 'У', 'V': 'В', 'W': 'В', 'X': 'Кс',
        'Y': 'Ы', 'Z': 'З',
        'sh': 'ш', 'ch': 'ч', 'th': 'з', 'ph': 'ф', 'kh': 'х',
        'Sh': 'Ш', 'Ch': 'Ч', 'Th': 'З', 'Ph': 'Ф', 'Kh': 'Х',
    }
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isascii() and ch.isalpha():
            digraph = text[i:i+2]
            if digraph in mapping:
                result.append(mapping[digraph])
                i += 2
            else:
                result.append(mapping.get(ch, ch))
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


class TTSProcessor:
    """Процессор TTS с калибровкой и обработкой блоков."""

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config["sample_rate"]
        self.tts = SileroTTS(config)
        self.ru_speaker = self.tts._get_default_ru_speaker()
        self.service_speaker = self.tts._get_service_speaker()
        self.ext_lang = self.tts._lang_from_en_model()
        self.ext_speaker = config.get("en_speaker") or self.tts._get_default_speaker(self.ext_lang)
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
            ("ru_main", "ru", self.ru_speaker, "тестовая фраза для калибровки громкости."),
            ("ru_service", "ru", self.service_speaker, "тестовая фраза для калибровки громкости."),
            (self.ext_lang, self.ext_lang, self.ext_speaker, f"test phrase for volume calibration in {self.ext_lang}."),
        ]

        logger.info("Калибровка громкости голосов:")

        for name, lang, speaker, phrase in test_phrases:
            if lang == self.ext_lang and not self.tts.is_model_available(self.ext_lang):
                self.speaker_gains[name] = self.speaker_gains["ru_main"]
                logger.info(f"  {name}: модель недоступна, используется gain русского голоса (×{self.speaker_gains['ru_main']:.3f})")
                continue

            audio = self.tts.synthesize(phrase, lang, speaker)
            if len(audio) == 0:
                logger.warning(f"  {name}: не удалось синтезировать тестовую фразу, используется gain=1.0")
                continue

            current_rms = measure_rms(audio)
            if current_rms > 0:
                gain = target_rms / current_rms
                self.speaker_gains[name] = gain
                current_lufs_approx = 20 * np.log10(current_rms / 0.5)
                logger.info(f"  {name}: RMS={current_rms:.4f} (~{current_lufs_approx:.0f} LUFS) → gain=×{gain:.3f}")
            else:
                logger.warning(f"  {name}: нулевая громкость тестовой фразы, используется gain=1.0")

        logger.info(f"  Целевой RMS: {target_rms:.4f} (~{target_lufs} LUFS)")

    def synthesize_text(self, text: str) -> np.ndarray:
        """Синтезирует основной текст с определением языка."""
        if not text or not text.strip():
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        try:
            segments = LanguageDetector.split_by_language(text)
            all_audio = []
            for lang, seg_text in segments:
                cleaned = TextCleaner.clean_for_tts(seg_text)
                if not cleaned.strip() or len(cleaned.strip()) < 2:
                    continue
                chunks = split_long_text(cleaned, self.max_chunk_length)
                for chunk in chunks:
                    if lang == self.ext_lang and self.tts.is_model_available(self.ext_lang):
                        # Пауза перед английским
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
                        # Если русский чанк содержит латиницу — транслитерируем
                        chunk_to_synth = chunk
                        if LanguageDetector.has_latin(chunk):
                            chunk_to_synth = transliterate_to_cyrillic(chunk)
                        audio = self.tts.synthesize(chunk_to_synth, "ru", self.ru_speaker)
                        audio = audio * self.speaker_gains["ru_main"]
                        if len(audio) > 0:
                            all_audio.append(audio)
                    all_audio.append(np.zeros(int(self.sample_rate * 0.05), dtype=np.float32))
            if all_audio:
                if len(all_audio) > 1 and all_audio[-1].size < int(self.sample_rate * 0.5):
                    all_audio = all_audio[:-1]
                return np.concatenate(all_audio)
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        except Exception:
            return np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)

    def synthesize_service(self, text: str) -> np.ndarray:
        """Синтезирует служебный текст."""
        audio = self.tts.synthesize(text, "ru", self.service_speaker)
        return audio * self.speaker_gains["ru_service"]

    def add_silence(self, duration: float = 0.3) -> np.ndarray:
        """Создаёт тишину заданной длительности."""
        return np.zeros(int(self.sample_rate * duration), dtype=np.float32)

    def process_blocks(self, blocks: List[Dict]) -> List[np.ndarray]:
        """Обрабатывает блоки текста."""
        result_audio = []
        for block in blocks:
            if block["type"] == "text":
                text = block["content"]
                if text.strip():
                    audio = self.synthesize_text(text)
                    if len(audio) > 0:
                        result_audio.append(audio)
                        result_audio.append(self.add_silence(self.pause_between_paragraphs))

            elif block["type"] == "emphasis":
                print(f"  🖊️  {block.get('content', '')[:50]}")
            elif block["type"] == "footnote":
                if self.skip_footnotes:
                    continue
                note_text = block["text"]
                note_title = block.get("title", "")

                result_audio.append(self.add_silence(0.3))

                if note_title:
                    audio = self.synthesize_service(f"{self.footnote_prefix} {note_title}")
                else:
                    audio = self.synthesize_service(self.footnote_prefix)

                if len(audio) > 0:
                    result_audio.append(audio)
                    result_audio.append(self.add_silence(0.2))

                if note_text:
                    audio = self.synthesize_text(note_text)
                    if len(audio) > 0:
                        result_audio.append(audio)
                        result_audio.append(self.add_silence(0.2))

                audio = self.synthesize_service(self.footnote_suffix)
                if len(audio) > 0:
                    result_audio.append(audio)
                    result_audio.append(self.add_silence(0.3))

                result_audio.append(self.add_silence(0.2))

            elif block["type"] == "pause":
                result_audio.append(self.add_silence(block.get("duration", 0.3)))

        return result_audio

    def _apply_pedalboard(self, audio: np.ndarray) -> np.ndarray:
        """Постобработка через pedalboard: компрессор + эквалайзер + реверб + комнатный шум."""
        try:
            from pedalboard import Pedalboard, HighpassFilter, LowpassFilter, PeakFilter, Compressor, Reverb, Gain
            from scipy import signal as scipy_signal
        except ImportError:
            logger.warning("pedalboard или scipy не установлены. pip install pedalboard scipy")
            return audio
        cfg = self.config
        try:
            board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=cfg.get("pb_highpass_hz", 85)),
                PeakFilter(cutoff_frequency_hz=cfg.get("pb_deharsh_hz", 5500), gain_db=cfg.get("pb_deharsh_db", -2.5), q=2.0),
                PeakFilter(cutoff_frequency_hz=cfg.get("pb_deharsh2_hz", 7800), gain_db=cfg.get("pb_deharsh2_db", -3.0), q=2.0),
                LowpassFilter(cutoff_frequency_hz=cfg.get("pb_lowpass_hz", 11500)),
                PeakFilter(cutoff_frequency_hz=cfg.get("pb_warmth_hz", 280), gain_db=cfg.get("pb_warmth_db", 1.8), q=0.9),
                PeakFilter(cutoff_frequency_hz=cfg.get("pb_clarity_hz", 3200), gain_db=cfg.get("pb_clarity_db", 1.4), q=1.1),
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
            ])
            processed = board(audio[np.newaxis, :], self.sample_rate)
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

    def convert_file_streaming(self, input_file: str, output_file: str, work_dir: str = None) -> bool:
        """Конвертирует файл в потоковом режиме."""
        start_time = time.time()
        logger.info(f"Обработка (потоковая): {input_file}")

        parser = FB2Parser(input_file)
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
            all_blocks.append({"type": "pause", "duration": self.pause_between_chapters})

        total_parts = (len(all_blocks) + self.blocks_per_part - 1) // self.blocks_per_part
        part_files = []

        logger.info(f"Всего блоков: {len(all_blocks)}, частей: {total_parts}")

        try:
            for part_idx in range(total_parts):
                start = part_idx * self.blocks_per_part
                end = min(start + self.blocks_per_part, len(all_blocks))
                part_blocks = all_blocks[start:end]

                logger.info(f"Часть {part_idx + 1}/{total_parts}: блоки {start + 1}-{end}")

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
                del part_audio_parts

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
            if not concat_ogg_files(part_files, output_file, ffmpeg_filter):
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
                logger.warning(f"Выходной файл не создан, части сохранены в {work_path}")

        return True

def print_header(config: dict, input_file: str, output_file: str, ru_speaker: str, service_speaker: str):
    """Выводит заголовок с информацией о конвертации."""
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    threads = config.get("parallel_threads", 1)
    loudness = config.get("loudness_target")
    speed = config.get("speed", 1.0)
    ext_model = config.get("en_model", "v3_en")
    ext_speaker = config.get("en_speaker", "en_0")
    emp_spk = config.get("emphasis_speaker") or ru_speaker
    emp_speed = config.get("emphasis_speed", 1.0)
    en_speed = config.get("en_speed", 1.0)
    pp = config.get("pause_between_paragraphs", 0.2)
    ps = config.get("pause_between_sentences", 0.05)
    psc = config.get("pause_semicolon", 0.0)
    pcl = config.get("pause_colon", 0.0)
    skip_fn = config.get("skip_footnotes", False)

    def v(text):
        return f"{YELLOW}{text}{RESET}"

    device_name = device.upper()
    if device == "cuda":
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device == "cpu":
        device_name = f"CPU ({threads} потоков)"

    sys.stderr.write(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {BOLD}FB2 to Audio Converter v{VERSION}{RESET}\n")
    sys.stderr.write(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════╣{RESET}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}Вход:{RESET} {v(Path(input_file).name):<30} {GREEN}Выход:{RESET} {v(Path(output_file).name)}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}RU:{RESET} {v(config.get('ru_model','?'))} \x7b {v(ru_speaker):<15} \x7d  {GREEN}Служ.:{RESET} {v(service_speaker)}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}EX:{RESET} {v(ext_model)} \x7b {v(ext_speaker):<12} \x7d {GREEN}Скор.:{RESET} {v(str(en_speed)+'x')}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}Устр-во:{RESET} {v(device_name)}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}Частота:{RESET} {v(str(config.get('sample_rate', 48000))+' Гц')}  {GREEN}Скор.:{RESET} {v(str(speed)+'x')}   {GREEN}Громк.:{RESET} {v(str(loudness)+' LUFS' if loudness else 'без')}\n")
    sys.stderr.write(f"{BOLD}{CYAN}║{RESET} {GREEN}Чанк:{RESET} {v(str(config.get('max_chunk_length', 900))+' симв.')}    {GREEN}Сноски:{RESET} {v('пропущены' if skip_fn else 'озвучены')}\n")
    sys.stderr.write(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}\n")
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
    print("""
Русские (авто-ударения): v5_5_ru, v5_4_ru, v5_3_ru, v5_2_ru, v5_1_ru, v5_ru
Русские (ручные ударения): v5_cis_base, v5_cis_base_nostress, v5_cis_ext
Русские (v4/v3): v4_ru, v3_1_ru, ru_v3
Дополнительные: v3_en, v3_de, v3_fr, v3_es, v3_en_indic
""")


def setup_logging(config: dict, args):
    """Настраивает логирование."""
    log_level = logging.DEBUG if args.debug else logging.INFO

    logger.handlers.clear()
    logger.setLevel(log_level)

    log_file = str(Path(args.input).with_suffix(".log"))
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
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
        for block in chapter:
            total_blocks += 1
            if block["type"] == "text":
                total_text_blocks += 1
                text = block["content"]
                total_chars += len(text)
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
    print(f"Кириллицы:           {total_chars - en_chars} ({100*(total_chars-en_chars)/max(1,total_chars):.1f}%)")
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
    parser.add_argument("--loudness", type=float, help="Целевая громкость LUFS для калибровки")
    parser.add_argument("--no-calibrate", action="store_true", help="Отключить калибровку громкости")
    parser.add_argument("--skip-footnotes", action="store_true", help="Не озвучивать сноски")
    parser.add_argument("--cpu", action="store_true", help="CPU")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="Устройство")
    parser.add_argument("--parallel", type=int, default=None, help="Число потоков")
    parser.add_argument("--list-ru-speakers", action="store_true")
    parser.add_argument("--list-en-speakers", action="store_true")
    parser.add_argument("--list-de-speakers", action="store_true", help="Показать немецкие голоса")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Проверить текст без синтеза")
    parser.add_argument("--debug", action="store_true", help="Расширенный вывод")

    args = parser.parse_args()

    if not args.config:
        for default_config_name in ["config.yaml", "config.yml", "config.json"]:
            if os.path.exists(default_config_name):
                args.config = default_config_name
                break

    config = load_config(args.config)

    for key in [
        "ru_model", "en_model", "ru_speaker", "en_speaker", "service_speaker",
        "sample_rate", "format", "speed",
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

    output_file = args.output or str(Path(args.input).with_suffix(f".{config.get('format', 'ogg')}"))

    if args.dry_run:
        dry_run(args.input, config)
        return 0

    try:
        converter = FB2ToAudioConverter(config)
        print_header(
            config, args.input, output_file,
            converter.tts.ru_speaker, converter.tts.service_speaker,
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
