"""
Расшифровка файлов: перетащили видео или запись в окно - в «Истории»
появляется сеанс, и текст в нём растёт по мере распознавания.

Звук достаёт PyAV (внутри FFmpeg): mp4, mkv, webm, mov, mp3, m4a, ogg,
opus, wav, flac и прочее. Файл читается кусками - час видео не занимает
гигабайт памяти. Фразы режет Silero VAD, каждая фраза распознаётся
целиком (до 10 с - это окно видеокарты): точнее, чем потоково, и с
пунктуацией по всей фразе.
"""
from __future__ import annotations

import collections
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

import history

SR = 16000
BLOCK = 512
VAD_ON, VAD_OFF = 0.5, 0.35
END_SILENCE = 16                # ~0.5 с тишины - конец фразы
PAD = 6                         # ~0.2 с звука до и после речи
MAX_PHRASE = int(10.0 * SR / BLOCK)     # длиннее - режем в самом тихом месте
CUT_SEARCH = int(2.0 * SR / BLOCK)
BATCH = 4

EXTENSIONS = ('.mp3', '.wav', '.m4a', '.aac', '.ogg', '.oga', '.opus', '.flac', '.wma', '.amr',
              '.mp4', '.mkv', '.webm', '.mov', '.avi', '.wmv', '.flv', '.m4v', '.3gp', '.ts')


def decode(path: Path):
    """(длительность в секундах или 0, генератор кусков 16 кГц моно float32)."""
    import av
    c = av.open(str(path))
    st = next((s for s in c.streams if s.type == 'audio'), None)
    if st is None:
        c.close()
        raise ValueError('в файле нет звука')
    dur = c.duration / 1e6 if c.duration else 0.0

    def chunks():
        res = av.AudioResampler(format='flt', layout='mono', rate=SR)
        try:
            for frame in c.decode(st):
                for f in res.resample(frame):
                    yield f.to_ndarray().reshape(-1)
            for f in res.resample(None):
                yield f.to_ndarray().reshape(-1)
        finally:
            c.close()

    return dur, chunks()


class FileJob:
    """Один файл. Работает в своём потоке, строки сразу пишет в историю."""

    def __init__(self, path, engine, slow_when_live: bool = True):
        self.path = Path(path)
        self.engine = engine
        self.slow_when_live = slow_when_live
        self.sid, self.base = history.new_file_session(self.path.name)
        self.progress = 0.0
        self.done = False
        self.error = ''
        self._cancel = threading.Event()
        history.jobs[self.sid] = self

    def start(self):
        threading.Thread(target=self._run, daemon=True, name='file').start()

    def cancel(self):
        self._cancel.set()

    # ---------- работа ----------

    def _run(self):
        try:
            self._transcribe()
        except Exception as e:
            self.error = str(e)[:200] or 'error'
            print(f'файл {self.path.name}: {e}')
        finally:
            self.done = True
            self.progress = 1.0
            history.jobs.pop(self.sid, None)

    def _transcribe(self):
        dur, chunks = decode(self.path)
        vad = self.engine.vad
        state = vad.new_state()
        pending: list[tuple[float, np.ndarray]] = []    # (начало, звук фразы)
        pre: collections.deque = collections.deque(maxlen=PAD)
        phrase: list[np.ndarray] = []
        probs: list[float] = []
        start_block = 0
        speaking = False
        silence = 0
        n = 0                       # блоков прочитано
        buf = np.zeros(0, dtype=np.float32)

        def close(upto: Optional[int] = None):
            nonlocal phrase, probs, start_block
            cut = len(phrase) if upto is None else upto
            body = phrase[:cut]
            if upto is None:
                # хвост тишины длиннее запаса не нужен
                tail = 0
                for p in reversed(probs):
                    if p >= VAD_OFF:
                        break
                    tail += 1
                body = phrase[:max(1, len(phrase) - max(0, tail - PAD))]
            if body:
                pending.append((start_block * BLOCK / SR, np.concatenate(body)))
            phrase, probs = phrase[cut:], probs[cut:]
            start_block += cut

        for chunk in chunks:
            if self._cancel.is_set():
                return
            buf = np.concatenate([buf, chunk])
            m = len(buf) // BLOCK
            for i in range(m):
                b = buf[i * BLOCK:(i + 1) * BLOCK]
                p = float(vad([b], [state])[0])
                if not speaking:
                    if p >= VAD_ON:
                        speaking = True
                        phrase = list(pre) + [b]
                        probs = [0.0] * len(pre) + [p]
                        start_block = n - len(pre)
                        silence = 0
                    else:
                        pre.append(b)
                else:
                    phrase.append(b)
                    probs.append(p)
                    silence = silence + 1 if p < VAD_OFF else 0
                    if silence >= END_SILENCE:
                        close()
                        speaking = False
                        pre = collections.deque(phrase[-PAD:], maxlen=PAD)
                        phrase, probs = [], []
                    elif len(phrase) >= MAX_PHRASE:
                        # режем в самом тихом месте последних двух секунд: не посреди слова
                        lo = len(probs) - CUT_SEARCH
                        cut = lo + int(np.argmin(probs[lo:])) + 1
                        close(cut)
                n += 1
            buf = buf[m * BLOCK:]
            if dur > 0:
                self.progress = min(0.99, n * BLOCK / SR / dur)
            if len(pending) >= BATCH:
                self._recognize(pending)
                pending = []
        if speaking and phrase:
            close()
        if pending:
            self._recognize(pending)

    def _recognize(self, items: list[tuple[float, np.ndarray]]):
        stt = self.engine.stt
        if stt is None or self._cancel.is_set():
            return
        t0 = time.perf_counter()
        hyps = stt.recognize([a for _, a in items])
        worker = getattr(self.engine, '_tr', None)
        lines = []
        for (start, _), hyp in zip(items, hyps):
            text = ' '.join(w.text for w in hyp.words).strip()
            if not text:
                continue
            rec = {'t': round(self.base + start, 2), 'who': '?', 'src': 'file', 'text': text}
            if worker is not None:
                # режим «Перевод»: английские фразы - сразу с переводом
                import translate_onnx
                if not translate_onnx.is_russian(text):
                    rec['tr'] = worker.translator.translate(text)
            lines.append(rec)
        if lines and not self._cancel.is_set():
            history.append(self.sid, lines)
        if self.slow_when_live and self.engine.state.name == 'RUNNING':
            # идут живые субтитры - уступаем им процессор
            time.sleep(time.perf_counter() - t0)
