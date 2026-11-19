"""
Загрузка/сохранение конфигурации приложения.

Путь - стандартное место ОС:
  Linux:   ~/.config/voice-transcriber/config.json
  Windows: %APPDATA%\\voice-transcriber\\config.json

Доступ потокобезопасный (RLock), запись атомарная (через временный файл).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QStandardPaths


APP_NAME = 'voice-transcriber'
CONFIG_FILENAME = 'config.json'



@dataclass
class AppConfig:
    # выбранные источники
    mic_source_id: Optional[str] = None
    mic_source_name: Optional[str] = None
    sys_source_id: Optional[str] = None
    sys_source_name: Optional[str] = None

    mic_speaker_name: str = 'user'

    # оверлей. -1 значит "посчитать от размера экрана", см. overlay.apply_geometry
    overlay_font_size: int = 22
    overlay_x: int = -1
    overlay_y: int = -1
    overlay_width: int = -1
    overlay_height: int = 280
    overlay_ttl_sec: int = 10   # 0 = не скрывать
    overlay_bg_opacity: int = 78   # подложка карточек, проценты
    overlay_align: str = 'left'    # left | center
    overlay_max_lines: int = 3
    overlay_hover_opacity: int = 20    # % видимости под курсором; 100 - не гасить     # строк в карточке, старые уезжают вверх

    history_enabled: bool = True   # сохранять расшифровки (папка history рядом с настройками)

    # fast | balanced | stable
    recognition_preset: str = 'balanced'
    use_gpu: bool = False              # распознавание на видеокарте (DirectML)
    english: bool = False              # английская речь: многоязычная Parakeet вместо GigaAM
    translate_view: str = 'both'       # что из неё показывать: orig | both | trans

    # Discord-бот: токен лежит отдельно, зашифрованный (secret_store)
    discord_bot_enabled: bool = True
    discord_follow_user_id: Optional[str] = None
    discord_hear_me: bool = False      # бот берёт и ваш голос (по умолчанию - только остальных)
    discord_setup_done: bool = False   # бот хоть раз был на сервере - шаги настройки не нужны

    @classmethod
    def from_dict(cls, d: dict) -> 'AppConfig':
        # неизвестные ключи игнорим - старые конфиги не ломают апгрейд
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known}
        if 'english' not in d and d.get('translate'):
            clean['english'] = True     # 2.2: «Перевод» включал и английскую модель
        return cls(**clean)

# ============================================================
# ПУТЬ
# ============================================================

def config_dir() -> Path:
    """Папка конфига. Создаётся при первом обращении."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
    # если вызвались до setApplicationName - Qt вернёт путь без имени приложения, дописываем сами
    p = Path(base) if base else Path.home() / '.config'
    if p.name != APP_NAME:
        p = p / APP_NAME
    p.mkdir(parents=True, exist_ok=True)
    return p



def config_path() -> Path:
    return config_dir() / CONFIG_FILENAME


# ============================================================
# IO
# ============================================================

_lock = threading.RLock()


def load_config() -> AppConfig:
    """Читает конфиг. Если файла нет или он битый, возвращает дефолт."""
    path = config_path()
    with _lock:
        if not path.exists():
            return AppConfig()
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return AppConfig.from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return AppConfig()


def save_config(cfg: AppConfig) -> None:
    """Атомарная запись: пишем во временный файл, потом os.replace."""
    path = config_path()
    data = json.dumps(asdict(cfg), ensure_ascii=False, indent=2)
    with _lock:
        # Временный файл в той же папке (для atomic rename на одной FS)
        fd, tmp_path = tempfile.mkstemp(
            prefix='.config.', suffix='.tmp', dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(data)
            os.replace(tmp_path, path)
        except Exception:
            # если запись упала на полпути - временный файл не оставляем
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
