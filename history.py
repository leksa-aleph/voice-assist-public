"""
История расшифровок: один сеанс - от «слушаю» до «стоп».

Каждый сеанс - файл <id>.jsonl в папке history рядом с настройками,
по строке на законченную фразу: {"t": unix-время, "who": имя, "src": источник,
"text": текст}. Строки дописываются сразу, поэтому падение программы
ничего не теряет. Файл появляется только с первой фразой: пустые сеансы
не копятся. Названия, которые дал пользователь, лежат в index.json.

Всё остаётся на этом компьютере.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def history_dir() -> Path:
    from config import config_dir
    d = config_dir() / 'history'
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Session:
    id: str
    path: Path
    title: str = ''
    started: float = 0.0
    ended: float = 0.0
    lines: int = 0
    speakers: list = field(default_factory=list)
    preview: str = ''           # начало разговора - чтобы узнать сеанс в списке
    file: bool = False          # расшифровка файла: время в тексте - от начала записи
    base: float = 0.0           # для файла: какое unix-время соответствует началу записи

    @property
    def duration(self) -> float:
        return max(0.0, self.ended - self.started)


_lock = threading.Lock()
_cache: dict[str, tuple[float, Session]] = {}   # id -> (mtime, сводка)


def _index_path() -> Path:
    return history_dir() / 'index.json'


def _read_index() -> dict:
    try:
        with open(_index_path(), encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_index(data: dict):
    path = _index_path()
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_lines(path: Path) -> list[dict]:
    return read_from(path, 0)[0]


def read_from(path: Path, offset: int) -> tuple[list[dict], int]:
    """Строки, дописанные после offset, и новое смещение. Недописанная
    последняя строка остаётся на следующий раз."""
    out = []
    try:
        with open(path, 'rb') as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return out, offset
    end = data.rfind(b'\n') + 1
    for line in data[:end].splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line.decode('utf-8')))
        except ValueError:
            continue            # оборванная строка после сбоя
    return out, offset + end


def _summary(path: Path, titles: dict) -> Optional[Session]:
    sid = path.stem
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _cache.get(sid)
    if hit and hit[0] == mtime:
        s = hit[1]
    else:
        lines = read_lines(path)
        if not lines:
            return None
        who = []
        for ln in lines:
            name = ln.get('who') or '?'
            if name not in who:
                who.append(name)
        preview = ''
        for ln in lines:
            preview = (preview + ' ' + ln.get('text', '')).strip()
            if len(preview) > 90:
                break
        s = Session(sid, path, started=float(lines[0].get('t', 0)),
                    ended=float(lines[-1].get('t', 0)), lines=len(lines), speakers=who,
                    preview=preview)
        _cache[sid] = (mtime, s)
    meta = titles.get(sid) if isinstance(titles.get(sid), dict) else {}
    s.title = meta.get('title', '')
    s.file = bool(meta.get('file'))
    s.base = float(meta.get('base', s.started))
    return s


def list_sessions() -> list[Session]:
    """Сеансы, новые сверху. Файл, который ещё считается, виден сразу."""
    titles = _read_index()
    out = []
    seen = set()
    for p in history_dir().glob('*.jsonl'):
        s = _summary(p, titles)
        if s is not None:
            out.append(s)
            seen.add(s.id)
    for sid, job in list(jobs.items()):
        if sid not in seen:
            meta = titles.get(sid) if isinstance(titles.get(sid), dict) else {}
            out.append(Session(sid, history_dir() / f'{sid}.jsonl', title=meta.get('title', ''),
                               started=job.base, ended=job.base, file=True, base=job.base))
    out.sort(key=lambda s: s.started, reverse=True)
    return out


# Файлы, которые сейчас расшифровываются: id сеанса -> задание (file_transcribe.FileJob)
jobs: dict = {}


def new_file_session(name: str) -> tuple[str, float]:
    """Сеанс для файла: название - имя файла, время в тексте - от начала записи."""
    import uuid
    base = round(time.time(), 2)
    sid = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:4]
    with _lock:
        data = _read_index()
        data[sid] = {'title': name, 'file': True, 'base': base}
        _write_index(data)
    return sid, base


def stamp(t: float, base: Optional[float] = None) -> str:
    """Время строки: часы и минуты, а у файла - место в записи (1:23, 1:02:03)."""
    if base is None:
        return time.strftime('%H:%M', time.localtime(t))
    s = max(0, int(t - base))
    if s >= 3600:
        return f'{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}'
    return f'{s // 60}:{s % 60:02d}'


def append(sid: str, lines: list[dict]):
    """Дописать готовые строки в сеанс (для файлов)."""
    with open(history_dir() / f'{sid}.jsonl', 'a', encoding='utf-8') as f:
        for rec in lines:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def load(sid: str) -> list[dict]:
    return read_lines(history_dir() / f'{sid}.jsonl')


def rename(sid: str, title: str):
    with _lock:
        data = _read_index()
        meta = data.get(sid) if isinstance(data.get(sid), dict) else {}
        title = title.strip()
        if title:
            meta['title'] = title
        else:
            meta.pop('title', None)
        if meta:
            data[sid] = meta
        else:
            data.pop(sid, None)
        _write_index(data)


def delete(sid: str):
    job = jobs.pop(sid, None)
    if job is not None:
        job.cancel()
    with _lock:
        try:
            (history_dir() / f'{sid}.jsonl').unlink()
        except FileNotFoundError:
            pass
        data = _read_index()
        if data.pop(sid, None) is not None:
            _write_index(data)
        _cache.pop(sid, None)


GROUP_GAP_S = 20.0      # после такой паузы - новый заголовок со временем
GROUP_MAX_S = 90.0      # длинный монолог делится на абзацы


class Grouper:
    """Делит фразы на реплики: сменился говорящий, была пауза или
    реплика слишком длинная - начинается новая."""

    def __init__(self):
        self.who: Optional[str] = None
        self.start = 0.0
        self.last = 0.0

    def feed(self, who: str, t: float) -> bool:
        new = (who != self.who or t - self.last > GROUP_GAP_S
               or t - self.start > GROUP_MAX_S)
        if new:
            self.who, self.start = who, t
        self.last = t
        return new


def as_text(sid: str) -> str:
    """Расшифровка простым текстом - для копирования."""
    out: list[str] = []
    g = Grouper()
    meta = _read_index().get(sid)
    base = float(meta['base']) if isinstance(meta, dict) and meta.get('file') and 'base' in meta else None
    for ln in load(sid):
        who = ln.get('who') or '?'
        t = float(ln.get('t', 0))
        if g.feed(who, t):
            at = stamp(t, base)
            out.append(f'\n\n[{at}] {who}:\n' if who != '?' else f'\n\n[{at}]\n')
        elif out:
            out.append(' ')
        out.append(ln.get('text', ''))
        if ln.get('tr'):
            out.append(f"\n→ {ln['tr']}\n")
    return ''.join(out).strip()


class Recorder:
    """Пишет текущий сеанс. Зовётся из потока распознавания."""

    def __init__(self):
        self.enabled = True
        self._sid: Optional[str] = None
        self._file = None
        self._lock = threading.Lock()

    @property
    def live_id(self) -> Optional[str]:
        return self._sid if self._file is not None else None

    @property
    def session_id(self) -> Optional[str]:
        """Текущий сеанс, даже если файла ещё нет (первой фразы не было)."""
        return self._sid

    def begin(self):
        with self._lock:
            self._close()
            self._sid = time.strftime('%Y%m%d-%H%M%S')

    def end(self):
        with self._lock:
            self._close()
            self._sid = None

    def add(self, src: str, who: str, text: str, t: Optional[float] = None, tr: str = '',
            sid: Optional[str] = None):
        """tr - перевод фразы; sid - сеанс, если перевод пришёл уже после «стоп»."""
        if not self.enabled or not text:
            return
        rec = {'t': round(t or time.time(), 2), 'who': who, 'src': src, 'text': text}
        if tr:
            rec['tr'] = tr
        with self._lock:
            if sid is not None and sid != self._sid:
                try:
                    append(sid, [rec])
                except OSError:
                    pass
                return
            if self._sid is None:
                return
            if self._file is None:
                try:
                    self._file = open(history_dir() / f'{self._sid}.jsonl', 'a', encoding='utf-8')
                except OSError:
                    return
            self._file.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self._file.flush()

    def _close(self):
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
