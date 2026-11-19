"""
Логирование: пишем в консоль и в файл рядом с конфигом.

Файл: ~/.config/voice-transcriber/voice-transcriber.log (Linux)
      %APPDATA%\\voice-transcriber\\voice-transcriber.log (Windows)

При старте старый лог переименовывается в .log.1 (одна ротация назад).
sys.stdout и sys.stderr перехватываются - print() из любых модулей
тоже попадёт в файл.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path

import version


def _log_dir() -> Path:
    if platform.system() == 'Windows':
        base = os.environ.get('APPDATA', str(Path.home()))
        return Path(base) / 'voice-transcriber'
    return Path.home() / '.config' / 'voice-transcriber'


def log_path() -> Path:
    return _log_dir() / 'voice-transcriber.log'


class _NullStream:
    """Заглушка для сборок --windowed, где sys.stdout и sys.stderr равны None."""
    encoding = 'utf-8'
    errors = 'replace'

    def write(self, s):
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError('no fileno')

    def close(self):
        pass


class _TeeStream:
    """Пишет одновременно в оригинальный поток и в файл."""
    def __init__(self, original, file_handle):
        self._orig = original
        self._file = file_handle

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        try:
            self._file.write(s)
            self._file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._orig, name)


def setup_logging() -> Path:
    """
    Настраивает логирование. Возвращает путь к лог-файлу.
    Зовётся один раз в начале main().
    """
    d = _log_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = log_path()

    # ротация: предыдущий .log -> .log.1 (затирает старый .log.1)
    if p.exists():
        backup = p.with_suffix('.log.1')
        try:
            if backup.exists():
                backup.unlink()
            p.rename(backup)
        except OSError:
            pass

    # Открываем новый файл с line-buffering
    f = open(p, 'w', encoding='utf-8', buffering=1)

    # шапка
    f.write('=== voice-transcriber log ===\n')
    f.write(f'version:  {version.TAG}\n')
    f.write(f'platform: {platform.platform()}\n')
    f.write(f'python:   {sys.version}\n')
    f.write(f'cwd:      {os.getcwd()}\n')
    f.write('=============================\n\n')
    f.flush()


    # все print() из проекта и сторонних либ пойдут и в консоль, и в файл
    sys.stdout = _TeeStream(sys.stdout or _NullStream(), f)
    sys.stderr = _TeeStream(sys.stderr or _NullStream(), f)

    # logging пишет в stderr (он уже Tee-нутый), так модульные логи тоже попадают в файл
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    ))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # если какая-то либа уже навесила свои хендлеры при импорте - сносим
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    logging.info(f'Logging initialized -> {p}')
    return p
