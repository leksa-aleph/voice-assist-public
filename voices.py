"""
База голосовых эталонов: voices.npz в папке пользователя.

Под ключом name лежит нормированный центроид, под name__count - сколько
записей в него усреднено. Файл всегда закрывается сразу после чтения:
открытый NpzFile на Windows не даёт его перезаписать.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from paths import user_data_path

COUNT_SUFFIX = '__count'


def voices_file() -> Path:
    return user_data_path('voices.npz', legacy='voices/voices.npz')


def _read_all() -> dict[str, np.ndarray]:
    path = voices_file()
    if not path.exists():
        return {}
    try:
        with np.load(path) as data:
            return {k: np.array(data[k]) for k in data.files}
    except Exception:
        return {}


def _write_all(items: dict[str, np.ndarray]) -> None:
    path = voices_file()
    if not items:
        try:
            path.unlink()
        except OSError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.voices.', suffix='.npz', dir=str(path.parent))
    os.close(fd)
    try:
        np.savez(tmp, **items)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_centroids() -> dict[str, np.ndarray]:
    return {k: v.astype(np.float32) for k, v in _read_all().items()
            if not k.endswith(COUNT_SUFFIX)}


def load_counts() -> dict[str, int]:
    raw = _read_all()
    out = {}
    for k in raw:
        if k.endswith(COUNT_SUFFIX):
            continue
        c = raw.get(k + COUNT_SUFFIX)
        out[k] = int(c) if c is not None else 1
    return out


def add_sample(name: str, emb: np.ndarray, replace: bool = False) -> int:
    """Добавляет запись к голосу. Возвращает число записей в центроиде."""
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    items = _read_all()
    key_n = name + COUNT_SUFFIX
    if not replace and name in items:
        old_n = int(items.get(key_n, 1))
        merged = items[name].astype(np.float32) * old_n + emb
        items[name] = (merged / (np.linalg.norm(merged) + 1e-9)).astype(np.float32)
        items[key_n] = np.array(old_n + 1, dtype=np.int32)
    else:
        items[name] = emb
        items[key_n] = np.array(1, dtype=np.int32)
    _write_all(items)
    return int(items[key_n])


def delete(name: str) -> None:
    items = _read_all()
    items.pop(name, None)
    items.pop(name + COUNT_SUFFIX, None)
    _write_all(items)


def rename(old: str, new: str) -> None:
    items = _read_all()
    if old not in items or not new or new == old:
        return
    items[new] = items.pop(old)
    if old + COUNT_SUFFIX in items:
        items[new + COUNT_SUFFIX] = items.pop(old + COUNT_SUFFIX)
    _write_all(items)
