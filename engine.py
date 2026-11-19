"""
Движок распознавания речи.

Модели - на ONNX Runtime (asr_onnx.py), без PyTorch: один поток на сессию
и никакого холостого ожидания. Источники звука независимы:

  mic       имя из настроек;
  mon       звук приложения: Discord RPC, иначе голосовые эталоны;
  dc:<id>   Discord-бот: у каждого участника отдельная дорожка,
            имя известно точно.

Путь звука:
  блок 32 мс -> Silero VAD (все источники одним вызовом) -> фраза
  -> раз в 0.3-0.7 с распознаётся только окно после подтверждённого текста
  (streaming.Transcript) -> оверлей получает подтверждённое и черновик.
По тишине фраза закрывается: последнее окно подтверждается целиком.

Распознавание идёт в одном потоке. Закрытие фраз обслуживается раньше
черновиков, черновики одного источника заменяют друг друга, окна близкой
длины считаются одним батчем.
"""
from __future__ import annotations

import collections
import itertools
import queue
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

import asr_onnx
import audio_source
import history
import metrics
import overlay
import voices
from asr_onnx import Word
from resample import Rebuffer, StreamResampler
from streaming import LEFT_CTX_S, Transcript, ends_sentence, join

try:
    import discord_rpc
except Exception:
    discord_rpc = None


# ============================================================
# КОНСТАНТЫ
# ============================================================
SAMPLE_RATE = 16000
BLOCK_SIZE = 512
BLOCK_S = BLOCK_SIZE / SAMPLE_RATE

# VAD с гистерезисом: начать фразу труднее, чем продолжить.
# Так короткий шум не открывает фразу, а тихий конец слова её не рвёт.
VAD_ON = 0.5
VAD_OFF = 0.35
MIN_SILENCE_MS = 500        # короткая реплика: закрываем быстро
LONG_SILENCE_MS = 850       # длинная фраза: даём договорить
LONG_PHRASE_S = 1.5
MIN_SPEECH_MS = 150         # короче этого - шум
MAX_UTT_S = 120             # страховка: дальше фраза закрывается и идёт следующей
TAIL_KEEP_MS = 200          # сколько тишины оставлять в конце фразы
PREROLL_MS = 500
# Discord перестаёт присылать пакеты уже в короткой паузе посреди фразы.
# Раньше это сразу закрывало фразу: мысль резалась на кусочки, модель теряла
# контекст на стыках, карточки дёргались. Теперь пропуск дописывается
# тишиной в реальном времени, и конец фразы решает VAD - как у микрофона.
BOT_GAP_S = 0.3             # пакетов нет дольше - это пауза, а не сетевой джиттер
BOT_GIVE_UP_S = 3.0         # страховка: совсем пропал - закрываем фразу
KEEP_AUDIO_S = 10.0         # звук до границы подтверждённого - для проверки голоса и финала
# Фраза закрылась - её последние FINAL_CTX_S секунд распознаются ещё раз
# одним окном, и подтверждённым словам достаются знаки оттуда (Transcript.refine).
# Только для окон от границы подтверждённого (GigaAM): у Parakeet окно и так
# от начала предложения.
FINAL_CTX_S = 10.0
EARLY_PARTIALS_S = 2.5      # начало фразы: черновики чаще

# Чаще распознаём - раньше видно слово, но дороже. Замер на монологе
# (tools/subtitle_jumps.py, Ryzen 7500F), слово от конца в звуке до экрана:
#   fast      ~0.45 с, ~40% одного ядра
#   balanced  ~0.58 с, ~29%
#   stable    ~0.8 с,  ~20%, правок меньше всего
RECOGNITION_PRESETS = {
    'fast':     {'partial_interval_ms': 300, 'partial_min_s': 0.3, 'draft_guard_s': 0.15},
    'balanced': {'partial_interval_ms': 450, 'partial_min_s': 0.35, 'draft_guard_s': 0.2},
    'stable':   {'partial_interval_ms': 700, 'partial_min_s': 0.5, 'draft_guard_s': 0.3},
}
PAUSE_PARTIAL_MS = 190      # человек замолчал посреди фразы - сразу дораспознать хвост

SIM_THRESHOLD = 0.30
SIM_MARGIN = 0.05
EMBED_TRIM_MS = 400
EMBED_MIN_S = 0.7
SPEAKER_MAX_S = 8.0         # голос узнаём по первым секундам куска
LAST_SPEAKER_S = 5.0

# Смена говорящего внутри фразы: двое говорят без паузы, и VAD склеивает
# их реплики. Подтверждённый звук последних секунд смотрим окнами и
# начинаем новый кусок там, где один знакомый голос уверенно сменился другим.
SPLIT_MIN_S = 4.0
SPLIT_EVERY_S = 2.0
SPLIT_REGION_S = 6.0
SPLIT_PROBE_S = 2.0
SPLIT_WIN_S = 1.5
SPLIT_HOP_S = 0.5
SPLIT_MIN_RUN = 4           # окон подряд - примерно 2 с одного голоса
SPLIT_SIM = SIM_THRESHOLD + 0.07

DISCORD_MATCH_PAD_S = 0.4

LINK_CHECK_INTERVAL_S = 5
AUDIO_QUEUE_MAX_BLOCKS = 512
SOURCE_IDLE_DROP_S = 120

MAX_BATCH = 6
BATCH_MAX_WASTE = 1.6       # паддинг не больше чем в 1.6 раза от полезного звука

LOAD_STEPS = ('vad', 'embedder', 'stt', 'voices', 'warmup')

TR_CLAUSE_WORDS = 14        # фраза без точки длиннее - переводим по запятой
# Недоговорённое предложение переводится черновиком - вместе с ещё не
# подтверждёнными словами, которые уже видно в оригинале (TR_DRAFT_ASR):
# не раньше чем с TR_DRAFT_MIN_WORDS слов и не чаще раза в TR_DRAFT_EVERY_S.
# Замер на 120 с лекции (tools/subtitle_jumps.py, видеокарта): раньше было
# раз в 0.9 с и только подтверждённое - перевод слова на экране через
# 0.96 / 1.33 с (медиана / 90%), прибавлялся по 4.5 слова, у 10% - по 10+.
# Так - 0.62 / 0.95 с, по 3 слова, у 90% не больше 4. Процессора +6% ядра.
TR_DRAFT_MIN_WORDS = 2
TR_DRAFT_EVERY_S = 0.4
TR_DRAFT_ASR = True
# Черновик на экране - только то, в чём сошлись два черновика подряд, без
# последних TR_AGREE_HOLD слов (их следующее слово оригинала ещё переставит).
# Показанное больше не меняется: следующие черновики и чистовой перевод
# продолжают его. Замер на 150 фразах FLORES (оригинал растёт по слову):
# раньше черновик переводился заново каждый раз - 90% показанных слов потом
# менялись, 60% слов появлялись только в чистовом; так - меняется 0%, слово
# видно в среднем на 1.6 с раньше, chrF к человеческому переводу 60.0
# против 60.7.
TR_AGREE_HOLD = 1
# Законченное предложение продолжает показанное. Если оригинал успел
# поменяться (распознавание поправило слова черновика), продолжение остаётся,
# только когда модель оценивает его не хуже свободного перевода на
# TR_KEEP_MARGIN (средний логарифм вероятности токена).
TR_FINAL_FORCE = True
TR_KEEP_MARGIN = 0.12
VIEWS = ('orig', 'both', 'trans')

_DEBUG_STREAM = bool(__import__('os').environ.get('VA_DEBUG_STREAM'))


def _load_discord_app():
    """
    Discord RPC (кто говорит, без бота) - для разработки, в интерфейсе его нет.
    Своё приложение в Discord Developer Portal; в
    %APPDATA%\\voice-transcriber\\discord_secret.txt две строки:
    Application ID и Client Secret. Нет файла или строк - RPC выключен.
    """
    import os
    p = os.path.join(os.environ.get('APPDATA', ''), 'voice-transcriber', 'discord_secret.txt')
    try:
        with open(p, encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return None
    if len(lines) < 2 or not lines[0].isdigit():
        print('discord rpc: в discord_secret.txt нужны две строки - '
              'Application ID и Client Secret', file=sys.stderr)
        return None
    return lines[0], lines[1]


def _blocks(ms: float) -> int:
    return max(1, int(round(ms / 1000 / BLOCK_S)))


# ============================================================
# СОСТОЯНИЯ
# ============================================================

class EngineState(Enum):
    IDLE = auto()
    LOADING = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()


@dataclass
class _Piece:
    """Кусок фразы одного говорящего. Обычно кусок на всю фразу."""
    first_word: int
    start_s: float
    speaker: Optional[str] = None
    sim: float = 0.0
    exact: bool = False
    used_s: float = 0.0         # по скольким секундам узнан голос
    checked_s: float = 0.0      # граница подтверждённого при прошлой проверке смены голоса
    tr_upto: int = 0            # столько слов куска уже отдано переводчику
    tr_draft: int = 0           # столько слов было в последнем черновике перевода
    tr_draft_t: float = 0.0
    tr_draft_text: str = ''


class _Utt:
    """
    Одна фраза источника. Звук дописывает поток VAD (только append),
    распознанное и куски меняет только поток STT.
    """

    def __init__(self, key: str, kind: str, n: int, speaker: Optional[str], t0: float,
                 draft_guard_s: float = 0.2, sentence_windows: bool = False):
        self.key = key
        self.kind = kind
        self.n = n
        self.speaker = speaker          # фиксированное имя (mic, bot)
        self.t0 = t0                    # monotonic начала первого блока
        self.blocks: list = []
        # поток VAD
        self.probs: list = []
        self.speech_blocks = 0
        self.silence = 0
        self.since_partial = 0
        self.partial_sent = False
        # поток STT
        self.tr = Transcript(draft_guard_s, sentence_windows)
        self.pieces: list[_Piece] = [_Piece(0, 0.0)]
        self.sent: dict[int, tuple] = {}
        self.done = False
        self.dropped = 0                # блоки раньше этого уже освобождены
        self.run = 0                    # номер запуска записи (для перевода)

    def audio(self, b0: int, b1: int) -> np.ndarray:
        b0 = max(b0, self.dropped)
        parts = [b for b in self.blocks[b0:b1] if b is not None]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def seg(self, i: int):
        return self.n if i == 0 else f'{self.n}/{i}'


@dataclass
class _Source:
    key: str
    kind: str                       # mic | mon | bot
    vad_state: list
    preroll: collections.deque
    speaker: Optional[str] = None
    utt: Optional[_Utt] = None
    n_utt: int = 0
    last_block_t: float = 0.0
    filled_t: float = 0.0           # бот: до какого момента пропуск уже дописан тишиной


@dataclass
class _Job:
    utt: _Utt
    upto: int                       # сколько блоков фразы видно этому заданию
    final: bool
    submitted: float

    @property
    def key(self) -> str:
        return self.utt.key

    def est_samples(self) -> int:
        start = self.utt.tr.window_start(self.upto * BLOCK_S)
        return max(BLOCK_SIZE, self.upto * BLOCK_SIZE - int(start * SAMPLE_RATE))


# ============================================================
# ПЕРЕВОД
# ============================================================

class _TranslateWorker:
    """
    Свой поток: переводит предложения по очереди и шлёт оверлею перевод
    куска целиком. Закрытый кусок пишет в историю вместе с переводом - в
    порядке очереди, поэтому порядок фраз в истории не путается.

    Задания (run - номер запуска записи):
      ('sent', run, key, seg, who, text, end, t)   законченное предложение;
      ('draft', run, key, seg, who, text, n, end, t)  недоговорённое - перевод
            черновиком, пока предложение не кончится. Из очереди берётся
            только самый свежий черновик куска;
      end - конец последнего слова в звуке, t - когда отдано (для замеров);
      ('close', run, key, seg, who, text, started, sid)  кусок закрыт;
      ('end', run)                         запись остановлена.

    Русское не переводится. Без оригинала на экране (вид «Перевод») оно
    идёт в ленту как есть, иначе там его уже видно.
    Черновик продолжает начало прошлого черновика (без последнего слова) -
    перевод растёт, а не переписывается. Законченное предложение
    переводится начисто.
    Оверлей получает только события текущего запуска: после «стоп» и
    нового старта номера фраз начинаются заново, и старый перевод лёг бы
    под чужую фразу.
    """

    def __init__(self, translator, engine: 'TranscriberEngine'):
        self.translator = translator
        self.engine = engine
        self._q: queue.Queue = queue.Queue()
        self._segs: dict[tuple, dict] = {}         # (run, key, seg) -> части и черновик
        self._latest: dict[tuple, int] = {}         # самый свежий черновик куска
        self._tickets = itertools.count(1)
        self._thread = threading.Thread(target=self._loop, daemon=True, name='translate')
        self._thread.start()

    def submit(self, job: tuple):
        self._q.put(job)

    def draft(self, run: int, key: str, seg, who: str, text: str, end: float = 0.0):
        n = next(self._tickets)
        self._latest[(run, key, seg)] = n
        self._q.put(('draft', run, key, seg, who, text, n, end, time.monotonic()))

    def close(self):
        self._q.put(None)

    def _translate(self, text: str, keep: tuple, submitted: float) -> tuple[str, list, float]:
        t0 = time.monotonic()
        metrics.add('tr_queue', t0 - submitted)
        out = self.translator.translate_scored(text, keep)
        metrics.add('tr_infer', time.monotonic() - t0)
        return out

    def _seg(self, k: tuple) -> dict:
        st = self._segs.get(k)
        if st is None:
            st = self._segs[k] = {'parts': [], 'draft': '', 'ids': [], 'prev': None, 'ticket': 0,
                                  'sent': ('', ''), 'end': 0.0}
        return st

    def _push(self, run: int, key: str, seg, who: str, st: dict):
        eng = self.engine
        if run != eng._run:
            return
        own = eng.view == 'trans'       # без оригинала русское показываем здесь
        text = _join_parts([p for p in st['parts'] if p[1] or own])
        msg = (text, st['draft'])
        if msg != st['sent']:
            st['sent'] = msg
            overlay.push_trans(key, seg, who, text, st['draft'], st['end'])

    def _loop(self):
        import translate_onnx
        while True:
            job = self._q.get()
            if job is None:
                return
            try:
                kind, run = job[0], job[1]
                if kind == 'end':
                    for k in [k for k in self._segs if k[0] == run]:
                        del self._segs[k]
                    for k in [k for k in self._latest if k[0] == run]:
                        del self._latest[k]
                    continue
                k = (run, job[2], job[3])
                if kind == 'sent':
                    _, _, key, seg, who, text, end, t = job
                    st = self._seg(k)
                    st['end'] = max(st['end'], end)
                    if translate_onnx.is_russian(text):
                        st['parts'].append((text, False, text))
                    else:
                        src, keep = st.get('draft_src', ''), tuple(st['ids'])
                        if TR_FINAL_FORCE and keep and (text + ' ').startswith(src + ' '):
                            out = self._translate(text, keep, t)[0]
                        else:
                            out, _, lp = self._translate(text, (), t)
                            if TR_FINAL_FORCE and keep:
                                kept, _, lk = self._translate(text, keep, time.monotonic())
                                if lk >= lp - TR_KEEP_MARGIN:
                                    out = kept
                        st['parts'].append((out, True, text))
                    st['draft_src'] = ''
                    st['draft'], st['ids'], st['prev'] = '', [], None
                    if self._latest.get(k, 0) <= st['ticket']:
                        self._push(run, key, seg, who, st)
                    # иначе за ним в очереди черновик продолжения - покажем вместе,
                    # без мигания «черновик пропал - черновик появился»
                elif kind == 'draft':
                    _, _, key, seg, who, text, n, end, t = job
                    if self._latest.get(k) != n or run != self.engine._run:
                        continue                    # есть свежее или запись уже другая
                    st = self._seg(k)
                    st['ticket'] = n
                    st['end'] = max(st['end'], end)
                    if translate_onnx.is_russian(text):
                        st['draft'], st['ids'], st['prev'] = (text if self.engine.view == 'trans' else ''), [], None
                    else:
                        new = self._translate(text, tuple(st['ids']), t)[1]
                        if st['prev'] is not None:
                            agreed = _agreed(st['prev'], new, self.translator.target_vocab)
                            if len(agreed) > len(st['ids']):
                                st['ids'] = agreed
                                st['draft'] = self.translator.decode(agreed)
                        st['prev'] = new
                    st['draft_src'] = text
                    self._push(run, key, seg, who, st)
                else:
                    _, _, key, seg, who, text, started, sid = job
                    st = self._segs.pop(k, None) or {'parts': []}
                    self._latest.pop(k, None)
                    tr = _join_parts([p for p in st['parts'] if p[1]])
                    self.engine.history.add(key, who, text, started, tr=tr, sid=sid)
            except Exception as e:
                print(f'перевод: {e}', file=sys.stderr)


def _join_parts(parts: list) -> str:
    """
    Части перевода куска - (перевод, переведено ли, исходник) - в одну строку.
    Длинную фразу переводчик получает кусками по запятой, и каждый кусок
    он оформляет как целое предложение: «…средних классов. И мы рады…».
    Если исходник разрезан по запятой, стык снова становится запятой, а
    заглавная - строчной (если в исходнике там было строчное слово, не имя).
    """
    out: list[str] = []
    prev_src = ''
    for text, _translated, src in parts:
        if not text:
            continue
        if out and prev_src.rstrip().endswith((',', ';', ':')) and src[:1].islower():
            if out[-1].endswith('.') and not out[-1].endswith('..'):
                out[-1] = out[-1][:-1] + ','
            text = text[:1].lower() + text[1:]
        out.append(text)
        prev_src = src
    return ' '.join(out)


def _agreed(prev: list, new: list, vocab: list, hold: int = TR_AGREE_HOLD) -> list:
    """
    Токены, в которых сошлись два черновика: общее начало до границы слова,
    без недостроенного слова и ещё hold последних.
    """
    k = 0
    while k < min(len(prev), len(new)) and prev[k] == new[k]:
        k += 1
    starts = [i for i in range(len(new)) if vocab[new[i]].startswith('▁')]
    words = sum(1 for i in starts if i < k)
    if k < len(new) and not vocab[new[k]].startswith('▁'):
        words -= 1                  # слово не дописано одинаково
    words -= hold
    if words <= 0:
        return []
    return new[:starts[words]] if words < len(starts) else list(new)


# ============================================================
# ПЛАНИРОВЩИК
# ============================================================

class _SttScheduler:
    """
    Одна очередь распознавания. Свой объект на каждый запуск записи:
    поток старого запуска не может проснуться в новом.
    """

    def __init__(self, run_batch):
        self._run_batch = run_batch
        self._cond = threading.Condition()
        self._finals: collections.deque[_Job] = collections.deque()
        self._partials: dict[str, _Job] = {}
        self.stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name='stt')

    def start(self):
        self._thread.start()

    def stop(self):
        with self._cond:
            self.stopped = True
            self._finals.clear()
            self._partials.clear()
            self._cond.notify_all()

    def submit(self, job: _Job):
        with self._cond:
            if self.stopped:
                return
            if job.final:
                p = self._partials.get(job.key)
                if p is not None and p.utt is job.utt:
                    del self._partials[job.key]
                self._finals.append(job)
            else:
                self._partials[job.key] = job
            self._cond.notify()

    def _take(self) -> list[_Job]:
        cands = list(self._finals) + sorted(self._partials.values(),
                                            key=lambda j: j.submitted)
        batch = [cands[0]]
        real = longest = cands[0].est_samples()
        for job in cands[1:]:
            if len(batch) >= MAX_BATCH:
                break
            n = job.est_samples()
            new_longest = max(longest, n)
            if new_longest * (len(batch) + 1) > BATCH_MAX_WASTE * (real + n):
                continue
            batch.append(job)
            real += n
            longest = new_longest

        for job in batch:
            if job.final:
                self._finals.remove(job)
            elif self._partials.get(job.key) is job:
                del self._partials[job.key]
        return batch

    def _loop(self):
        while True:
            with self._cond:
                while not self.stopped and not self._finals and not self._partials:
                    self._cond.wait()
                if self.stopped:
                    return
                batch = self._take()
            try:
                self._run_batch(batch, self)
            except Exception as e:
                import traceback
                print(f'STT batch error: {e}\n{traceback.format_exc()}', file=sys.stderr)


# ============================================================
# ENGINE
# ============================================================

class TranscriberEngine(QObject):
    state_changed = pyqtSignal(object)            # EngineState
    load_progress = pyqtSignal(str, int, int)     # step, done, total
    error_occurred = pyqtSignal(str)
    status_message = pyqtSignal(str)
    device_changed = pyqtSignal(bool, str)        # на видеокарте ли, ошибка видеокарты
    mode_changed = pyqtSignal(bool, str)          # английская ли модель, ошибка

    def __init__(self, mic_speaker_name: str = 'user'):
        super().__init__()
        self.mic_speaker_name = mic_speaker_name
        self._state = EngineState.IDLE

        # модели
        self.vad: Optional[asr_onnx.SileroVad] = None
        self.embedder: Optional[asr_onnx.SpeakerEmbedder] = None
        self.stt = None                 # asr_onnx.GigaAM или asr_parakeet.Parakeet
        self.voice_db: dict[str, np.ndarray] = {}
        self.device_name = ''
        self.use_gpu = False
        # Английская речь: многоязычная Parakeet вместо GigaAM. Что из неё
        # показывать - view: оригинал, оригинал и перевод, только перевод.
        self.multilang = False
        self.view = 'both'
        self._tr: Optional[_TranslateWorker] = None
        # смена модели (язык, видеокарта) - по одной за раз и под последние настройки
        self._cfg_lock = threading.Lock()
        self._cfg_gen = 0
        self._run = 0                   # номер запуска записи: старые переводы в новый не попадают

        # запуск
        self._audio_q: Optional[queue.Queue] = None
        self._sched: Optional[_SttScheduler] = None
        self._stop_event = threading.Event()
        self._vad_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._streams: list = []
        self._streams_lock = threading.Lock()
        self._sys_source_id: Optional[str] = None
        self._mic_source_id: Optional[str] = None
        self.discord = None

        # Discord-бот
        self._bot_streams: dict[str, tuple[StreamResampler, Rebuffer]] = {}
        self._bot_in_voice = False

        # история расшифровок: сеанс - от start() до stop()
        self.history = history.Recorder()

        # последний узнанный голос по источнику (трогает только поток STT)
        self._last_spk: dict[str, tuple[str, float]] = {}

        # уровни звука и статистика для интерфейса
        self._levels: dict[str, tuple[float, float]] = {}
        self._stt_hist: collections.deque = collections.deque(maxlen=60)
        self._stt_hist_all: collections.deque = collections.deque(maxlen=20000)  # для tools/subtitle_jumps.py
        self._stt_load = 0.0
        self._stt_last_end = time.perf_counter()

        self.apply_recognition_preset('balanced')

    # ---------- состояние ----------

    @property
    def state(self) -> EngineState:
        return self._state

    def _set_state(self, s: EngineState):
        if self._state != s:
            self._state = s
            self.state_changed.emit(s)

    # ---------- данные для интерфейса ----------

    def levels(self) -> dict[str, float]:
        """Уровни сигнала по источникам, 0..1, с плавным спадом."""
        now = time.monotonic()
        out = {}
        for key, (rms, t) in list(self._levels.items()):
            age = now - t
            if age > 2.0:
                continue
            # речь в -30..-10 dBFS превращаем в 0..1
            db = 20 * np.log10(max(rms, 1e-6))
            v = min(1.0, max(0.0, (db + 52) / 40))
            out[key] = v * float(np.exp(-age / 0.12))
        return out

    def stats(self) -> dict:
        hist = list(self._stt_hist)
        if not hist:
            return {'infer_ms': None, 'rtf': None, 'history': [], 'device': self.device_name,
                    'stages': metrics.summary(), 'view': self.view if self.multilang else 'orig'}
        recent = hist[-20:]
        lat = sorted(h[1] for h in recent)[len(recent) // 2] * 1000
        audio = sum(h[0] for h in recent)
        proc = sum(h[1] for h in recent)
        return {
            'infer_ms': lat,             # медиана одного распознавания, не задержка субтитров
            'rtf': proc / audio if audio > 0 else None,
            'history': [h[1] * 1000 for h in hist],
            'device': self.device_name,
            'stages': metrics.summary(),
            'view': self.view if self.multilang else 'orig',   # в «RU» на экране только перевод
        }

    # ---------- загрузка моделей ----------

    def load_models(self):
        """Блокирующая загрузка. Вызывать из рабочего потока."""
        if self._state in (EngineState.READY, EngineState.RUNNING, EngineState.LOADING):
            return
        self._set_state(EngineState.LOADING)
        total = len(LOAD_STEPS)
        try:
            md = asr_onnx.model_dir()
            self.load_progress.emit('vad', 0, total)
            self.vad = asr_onnx.SileroVad(md / 'silero_vad_v6.onnx')
            self.load_progress.emit('vad', 1, total)

            self.embedder = asr_onnx.SpeakerEmbedder(md)
            self.load_progress.emit('embedder', 2, total)

            try:
                self.stt = self._make_stt()
            except Exception as e:
                if not self.multilang:
                    raise
                # английская модель не встала - субтитры всё равно нужны
                print(f'английская модель: {e}', file=sys.stderr)
                self.multilang = False
                self.mode_changed.emit(False, str(e)[:200])
                self.stt = self._make_stt()
            self._set_device(self.stt)
            self.load_progress.emit('stt', 3, total)

            self.voice_db = voices.load_centroids()
            self.load_progress.emit('voices', 4, total)

            self._warmup()
            self._tune_threads(self.stt)
            if self._need_translator():
                try:
                    self._start_translator(self._make_translator())
                except Exception as e:
                    print(f'переводчик: {e}', file=sys.stderr)
                    self.view = 'orig'
                    self.mode_changed.emit(self.multilang, str(e)[:200])
            self.load_progress.emit('warmup', 5, total)

            self._set_state(EngineState.READY)
            self.status_message.emit('Готов')
        except Exception as e:
            import traceback
            print(f'=== Ошибка загрузки моделей ===\n{traceback.format_exc()}',
                  file=sys.stderr)
            self._set_state(EngineState.ERROR)
            self.error_occurred.emit(f'Ошибка загрузки моделей: {e}')

    def _warmup(self):
        """Первые прогоны выделяют память под тензоры - пусть это будет до первой фразы."""
        rng = np.random.default_rng(0)

        def noise(sec):
            return (rng.standard_normal(int(sec * SAMPLE_RATE)) * 0.01).astype(np.float32)

        t0 = time.perf_counter()
        try:
            self._warm(self.stt)
            self.embedder.embed(noise(2.0))
        except Exception as e:
            print(f'warmup: {e}', file=sys.stderr)
        print(f'warmup: {time.perf_counter() - t0:.2f}s')

    @staticmethod
    def _warm(stt):
        rng = np.random.default_rng(0)

        def noise(sec):
            return (rng.standard_normal(int(sec * SAMPLE_RATE)) * 0.01).astype(np.float32)

        stt.recognize([noise(1.0)])
        stt.recognize([noise(2.5), noise(2.0)])

    def _make_stt(self):
        if self.multilang:
            import asr_parakeet
            stt = asr_parakeet.Parakeet(asr_parakeet.model_dir(), gpu=self.use_gpu)
        else:
            stt = asr_onnx.GigaAM(asr_onnx.model_dir(), gpu=self.use_gpu)
        stt.want_gpu = self.use_gpu     # просили видеокарту - не пересобирать, если она не подошла
        return stt

    def _need_translator(self) -> bool:
        if not self.multilang or self.view == 'orig':
            return False
        import translate_onnx
        return translate_onnx.available()

    @staticmethod
    def _make_translator():
        import translate_onnx
        return translate_onnx.Translator(translate_onnx.model_dir())

    def _start_translator(self, translator):
        old, self._tr = self._tr, _TranslateWorker(translator, self)
        if old is not None:
            old.close()

    def set_language(self, multilang: bool):
        """
        Английская речь: распознаёт многоязычная Parakeet (английский, русский
        и ещё 23 языка), законченные английские предложения переводятся на
        русский. Модели грузятся рядом, старые работают, пока новые не готовы.
        """
        self.multilang = bool(multilang)
        self._reconfigure()

    def set_view(self, view: str):
        """Что показывать в английском режиме: 'orig', 'both' или 'trans'."""
        self.view = view if view in VIEWS else 'both'
        self._reconfigure()

    def set_gpu(self, on: bool):
        """Переключить распознавание между процессором и видеокартой на ходу."""
        self.use_gpu = bool(on)
        self._reconfigure()

    def _reconfigure(self):
        """
        Привести модели к настройкам (язык, вид, видеокарта). Сборка идёт в
        своём потоке, по одной за раз; если пока она шла, настройки снова
        поменялись, результат выбрасывается и собирается под новые.
        """
        if self.stt is None:
            return
        self._cfg_gen += 1
        gen = self._cfg_gen
        threading.Thread(target=self._apply_config, args=(gen,), daemon=True, name='reconfigure').start()

    def _apply_config(self, gen: int):
        with self._cfg_lock:
            if gen != self._cfg_gen:
                return              # есть запрос новее - он и соберёт, и ответит
            if self.stt is None:
                self._config_idle()
                return
            import asr_parakeet
            multilang, use_gpu = self.multilang, self.use_gpu
            want = asr_parakeet.Parakeet if multilang else asr_onnx.GigaAM
            stt = self.stt
            error = ''
            try:
                if not isinstance(stt, want) or getattr(stt, 'want_gpu', None) != use_gpu:
                    stt = self._make_stt()
                    self._warm(stt)
                    self._tune_threads(stt)
            except Exception as e:
                print(f'смена модели: {e}', file=sys.stderr)
                error = str(e)[:200]
                stt = self.stt
                if multilang and not isinstance(stt, asr_parakeet.Parakeet):
                    self.multilang = multilang = False
            translator = None
            need = self._need_translator()
            if need and self._tr is None:
                try:
                    translator = self._make_translator()
                except Exception as e:
                    print(f'переводчик: {e}', file=sys.stderr)
                    error = error or str(e)[:200]
                    self.view = 'orig'
                    need = False
            if gen != self._cfg_gen or self.stt is None:
                if stt is not self.stt:
                    asr_onnx.release_gpu(stt)
                if self.stt is None:
                    self._config_idle()     # пока собирали, модели выгрузили
                return                      # иначе настройки поменялись - соберёт следующий
            if stt is not self.stt:
                old, self.stt = self.stt, stt
                self._stt_hist.clear()
                # сессия видеокарты выгружается в очереди к ней, а не когда
                # сборщик мусора доберётся - посреди окна новой модели
                asr_onnx.release_gpu(old)
            if translator is not None:
                self._start_translator(translator)
            elif not need and self._tr is not None:
                self._tr.close()
                self._tr = None
            self._set_device(stt)
            self.mode_changed.emit(self.multilang, error)

    def _config_idle(self):
        """Моделей нет - собирать нечего; интерфейс ждёт ответа, чтобы отпустить переключатели."""
        self.mode_changed.emit(self.multilang, '')
        self.device_changed.emit(False, '')

    def _set_device(self, stt):
        self.device_name = 'Видеокарта' if stt.gpu else 'Процессор'
        error = '' if stt.gpu or not self.use_gpu else (stt.gpu_error or 'error')
        if error and asr_onnx.gpu_lost:
            # видеокарту сбросили, пока модель собиралась на ней - она не «не подошла»
            error = 'lost'
        elif error:
            print(f'GPU недоступна: {error}', file=sys.stderr)
        self.device_changed.emit(stt.gpu, error)

    @staticmethod
    def _tune_threads(stt):
        """
        Медленному процессору - два-три потока на кодировщик. Работы у
        процессора столько же (потоки не крутятся вхолостую), но окно
        готово в 1.5 раза раньше: иначе на ×4-×8 окна распознавания растут
        и субтитры отстают на секунды. У модели свой масштаб порогов (tune_scale).
        """
        if stt.gpu:
            return
        import os
        cores = os.cpu_count() or 2
        k = getattr(stt, 'tune_scale', 1.0)
        a = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        a[::97] = 0.01
        stt.recognize([a])
        t0 = time.perf_counter()
        stt.recognize([a])
        speed = 4.0 / max(1e-3, time.perf_counter() - t0)
        threads = 1
        if speed < 16 * k and cores >= 4:
            threads = 2
        if speed < 9 * k and cores >= 8:
            threads = 3
        stt.set_threads(threads)
        print(f'распознавание: x{speed:.0f} на одном потоке, потоков: {threads}')

    def unload_models(self):
        if self._state == EngineState.RUNNING:
            self.stop()
        self.vad = None
        self.embedder = None
        old, self.stt = self.stt, None
        if old is not None:
            asr_onnx.release_gpu(old)
        if self._tr is not None:
            self._tr.close()
            self._tr = None
        self.voice_db = {}
        overlay.push_clear()
        self._set_state(EngineState.IDLE)
        self.status_message.emit('Модели выгружены')

    def reload_voices(self):
        self.voice_db = voices.load_centroids()

    def apply_recognition_preset(self, preset: str):
        cfg = RECOGNITION_PRESETS.get(preset, RECOGNITION_PRESETS['balanced'])
        self.partial_interval_ms = cfg['partial_interval_ms']
        self.partial_min_s = cfg['partial_min_s']
        self.draft_guard_s = cfg['draft_guard_s']

    # ---------- помощники для записи эталонов ----------

    def voiced_only(self, audio: np.ndarray) -> np.ndarray:
        """Оставляет только блоки с речью (своё состояние VAD, чужое не трогаем)."""
        audio = np.asarray(audio, np.float32)
        out = []
        speaking = False
        for i, prob in enumerate(self.vad.probs(audio)):
            speaking = prob >= (VAD_OFF if speaking else VAD_ON)
            if speaking:
                out.append(audio[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        return self.embedder.embed(audio)

    # ---------- start / stop ----------

    def start(self, mic_source_id: Optional[str], sys_source_id: Optional[str],
              allow_no_sources: bool = False):
        """
        Если источник сейчас не виден (Discord ещё не запущен), watchdog
        подхватит его, когда появится. allow_no_sources - для режима, где
        звук приходит только от Discord-бота.
        """
        if self._state != EngineState.READY:
            self.error_occurred.emit('Модели не готовы')
            return
        if not mic_source_id and not sys_source_id and not allow_no_sources:
            self.error_occurred.emit('Не выбран ни один источник')
            return

        self._mic_source_id = mic_source_id
        self._sys_source_id = sys_source_id
        self._stop_event = stop_event = threading.Event()
        self._audio_q = q = queue.Queue(maxsize=AUDIO_QUEUE_MAX_BLOCKS)
        self._sched = sched = _SttScheduler(self._run_batch)
        self._bot_streams = {}
        self._last_spk.clear()
        self._run += 1
        metrics.reset()                 # задержка по этапам - за этот запуск
        self.history.begin()
        sched.start()

        rpc_app = _load_discord_app() if discord_rpc is not None and self.discord is None else None
        if rpc_app:
            try:
                self.discord = discord_rpc.DiscordSpeakers(*rpc_app)
                self.discord.start()
            except Exception as e:
                print(f'discord rpc: {e}', file=sys.stderr)
                self.discord = None

        self._vad_thread = threading.Thread(
            target=self._vad_loop, args=(q, sched, stop_event),
            daemon=True, name='vad')
        self._vad_thread.start()

        self._open_available_streams()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, args=(stop_event,), daemon=True, name='watchdog')
        self._watchdog_thread.start()

        self._set_state(EngineState.RUNNING)
        self.status_message.emit('Слушаю')

    def stop(self):
        if self._state != EngineState.RUNNING:
            return
        # Всё, что нужно очистке, забираем в локальные переменные:
        # быстрый повторный старт создаст новые объекты, и очистка
        # старого запуска не должна их задеть.
        stop_event = self._stop_event
        stop_event.set()
        run, self._run = self._run, self._run + 1
        if self._tr is not None:
            self._tr.submit(('end', run))
        q, self._audio_q = self._audio_q, None
        sched, self._sched = self._sched, None
        discord, self.discord = self.discord, None
        vad_thread, self._vad_thread = self._vad_thread, None
        self._watchdog_thread = None
        with self._streams_lock:
            streams, self._streams = self._streams, []
        if sched is not None:
            sched.stop()
        self.history.end()
        self._bot_streams = {}
        self._levels.clear()

        def _cleanup():
            if discord is not None:
                try:
                    discord.stop()
                except Exception:
                    pass
            for s in streams:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
            if q is not None:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
            if vad_thread is not None:
                vad_thread.join(timeout=1.0)
            overlay.push_clear()

        threading.Thread(target=_cleanup, daemon=True, name='stop-cleanup').start()
        self._set_state(EngineState.READY)
        self.status_message.emit('Остановлено')

    # ---------- Discord-бот ----------

    def set_bot_in_voice(self, in_voice: bool):
        """Бот в голосовом канале: захват процесса Discord дублировал бы его."""
        self._bot_in_voice = bool(in_voice)

    def bot_audio(self, user_id: str, name: str, pcm48: np.ndarray):
        """Звук одного участника: float32, моно, 48 кГц."""
        if self._audio_q is None:
            return
        st = self._bot_streams.get(user_id)
        if st is None:
            st = (StreamResampler(48000, SAMPLE_RATE), Rebuffer(BLOCK_SIZE))
            self._bot_streams[user_id] = st
        y = st[0].process(pcm48)
        for block in st[1].push(y):
            self._enqueue('dc:' + user_id, 'bot', name, block)

    def bot_user_left(self, user_id: str):
        self._bot_streams.pop(user_id, None)

    def _sys_is_discord(self) -> bool:
        sid = (self._sys_source_id or '').lower()
        return 'discord' in sid

    # ---------- потоки звука ----------

    def _open_available_streams(self):
        mic_ids = {m.id for m in audio_source.list_microphones()}
        sys_ids = {s.id for s in audio_source.list_system_audio()}

        if self._sys_source_id and self._sys_source_id in sys_ids:
            try:
                self._open_stream('mon')
            except Exception as e:
                self.error_occurred.emit(f'Системный звук: {e}')

        if self._mic_source_id and self._mic_source_id in mic_ids:
            try:
                self._open_stream('mic')
            except Exception as e:
                self.error_occurred.emit(f'Микрофон: {e}')

    def _open_stream(self, kind: str):
        if kind == 'mic':
            stream = audio_source.open_microphone(
                self._mic_source_id, SAMPLE_RATE, BLOCK_SIZE, self._make_callback('mic'))
        else:
            stream = audio_source.open_system_audio(
                self._sys_source_id, SAMPLE_RATE, BLOCK_SIZE, self._make_callback('mon'))
        stream._vt_kind = kind
        stream.start()
        self._register_stream(stream)

    def _register_stream(self, stream) -> bool:
        with self._streams_lock:
            if not self._stop_event.is_set():
                self._streams.append(stream)
                return True
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        return False

    def change_microphone(self, new_id: Optional[str]):
        self._mic_source_id = new_id
        if self._state == EngineState.RUNNING:
            self._reopen_kind('mic')

    def change_system_audio(self, new_id: Optional[str]):
        self._sys_source_id = new_id
        if self._state == EngineState.RUNNING:
            self._reopen_kind('mon')

    def _reopen_kind(self, kind: str):
        with self._streams_lock:
            to_close = [s for s in self._streams if getattr(s, '_vt_kind', None) == kind]
            self._streams = [s for s in self._streams
                             if getattr(s, '_vt_kind', None) != kind]
        for s in to_close:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

        new_id = self._mic_source_id if kind == 'mic' else self._sys_source_id
        if not new_id:
            return
        try:
            if kind == 'mic':
                ids = {m.id for m in audio_source.list_microphones()}
            else:
                ids = {s.id for s in audio_source.list_system_audio()}
            if new_id not in ids:
                return  # подхватит watchdog
            self._open_stream(kind)
        except Exception as e:
            self.error_occurred.emit(f'Переключение источника: {e}')

    def _watchdog_loop(self, stop_event: threading.Event):
        """Раз в несколько секунд дооткрывает источники, которых не было при старте."""
        def have_stream(kind):
            with self._streams_lock:
                return any(getattr(s, '_vt_kind', None) == kind for s in self._streams)

        reported = {'mon': None, 'mic': None}
        names = {'mon': 'Системный звук', 'mic': 'Микрофон'}

        while not stop_event.is_set():
            try:
                if self._sys_source_id and self._sys_source_id.startswith('app:'):
                    try:
                        audio_source.ensure_app_linked(self._sys_source_id[4:])
                    except Exception:
                        pass

                ids = {
                    'mon': {s.id for s in audio_source.list_system_audio()},
                    'mic': {m.id for m in audio_source.list_microphones()},
                }
                for kind in ('mon', 'mic'):
                    want = self._sys_source_id if kind == 'mon' else self._mic_source_id
                    if not want or have_stream(kind) or want not in ids[kind]:
                        continue
                    try:
                        self._open_stream(kind)
                        reported[kind] = None
                        self.status_message.emit(f'{names[kind]} подключён')
                    except Exception as e:
                        msg = f'{names[kind]}: {e}'
                        if reported[kind] != msg:
                            reported[kind] = msg
                            self.error_occurred.emit(msg)
            except Exception as e:
                print(f'watchdog: {e}', file=sys.stderr)
            stop_event.wait(LINK_CHECK_INTERVAL_S)

    def _make_callback(self, kind: str):
        stop_event = self._stop_event

        def cb(indata, frames, time_info, status):
            if status:
                print(f'[audio {kind}] {status}', file=sys.stderr)
            if stop_event.is_set():
                return
            chunk = indata[:, 0] if indata.ndim > 1 else indata
            chunk = np.array(chunk, dtype=np.float32, copy=True)
            if kind == 'mon' and self._bot_in_voice and self._sys_is_discord():
                # тот же звук уже идёт от бота по дорожкам участников
                self._levels[kind] = (float(np.sqrt(np.mean(chunk * chunk))), time.monotonic())
                return
            speaker = self.mic_speaker_name if kind == 'mic' else None
            self._enqueue(kind, kind, speaker, chunk)

        return cb

    def _enqueue(self, key: str, kind: str, speaker: Optional[str], chunk: np.ndarray):
        q = self._audio_q
        if q is None:
            return
        t = time.monotonic()
        self._levels[key] = (float(np.sqrt(np.mean(chunk * chunk))), t)
        item = (key, kind, speaker, chunk, t)
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    # ---------- говорящий ----------

    def _identify_speaker(self, audio: np.ndarray) -> tuple[str, float, Optional[str]]:
        """(имя или '?', сходство, ближайший эталон даже если не уверены)."""
        if not self.voice_db or self.embedder is None:
            return '?', 0.0, None
        # по краям preroll и хвост тишины - уводят эмбеддинг в шум
        trim = int(EMBED_TRIM_MS / 1000 * SAMPLE_RATE)
        min_samples = int(EMBED_MIN_S * SAMPLE_RATE)
        clip = audio
        if len(clip) - 2 * trim >= min_samples:
            clip = clip[trim:len(clip) - trim]
        if len(clip) < min_samples:
            return '?', 0.0, None

        emb = self.embedder.embed(clip)
        scored = sorted(((float(np.dot(emb, ref)), name)
                         for name, ref in self.voice_db.items()), reverse=True)
        best_sim, best_name = scored[0]
        if best_sim < SIM_THRESHOLD:
            return '?', best_sim, best_name
        # два похожих голоса: честнее '?', чем уверенно чужое имя
        if len(scored) > 1 and best_sim - scored[1][0] < SIM_MARGIN:
            return '?', best_sim, None
        return best_name, best_sim, best_name

    def _resolve_speaker(self, u: _Utt, end_s: float, final: bool):
        """Имя для последнего куска фразы. Точно - от бота, микрофона или RPC."""
        pc = u.pieces[-1]
        if u.speaker is not None:
            pc.speaker, pc.sim, pc.exact = u.speaker, 1.0, True
            return
        if pc.exact:
            return
        now = time.monotonic()
        if self.discord is not None:
            try:
                rpc = self.discord.who_spoke(u.t0 + pc.start_s - DISCORD_MATCH_PAD_S,
                                             u.t0 + end_s + DISCORD_MATCH_PAD_S)
            except Exception:
                rpc = None
            if rpc:
                pc.speaker, pc.sim, pc.exact = rpc, 1.0, True
                self._last_spk[u.key] = (rpc, now)
                return

        last = self._last_spk.get(u.key)
        recent = last if last and now - last[1] < LAST_SPEAKER_S else None
        avail = min(end_s, pc.start_s + SPEAKER_MAX_S) - pc.start_s
        # По короткому звуку голос узнаётся хуже: пересчитываем, когда звука
        # стало вдвое больше, и на закрытии фразы.
        due = (pc.used_s == 0.0 or avail >= 2 * pc.used_s
               or (final and avail >= 1.4 * pc.used_s))
        if self.voice_db and due and avail >= EMBED_MIN_S:
            b0 = int(pc.start_s / BLOCK_S)
            clip = u.audio(b0, b0 + int(avail / BLOCK_S))
            name, sim, nearest = self._identify_speaker(clip)
            pc.used_s = avail
            # Неуверенный ответ отдаём тому, кто только что говорил, лишь когда
            # ближайший эталон - он же. Иначе в диалоге слова собеседника
            # приписывались бы предыдущему говорящему.
            if (name == '?' and recent and nearest == recent[0]
                    and sim >= SIM_THRESHOLD - 0.08):
                name = recent[0]
            if name != '?' or pc.speaker in (None, '?'):
                pc.speaker, pc.sim = name, sim
            if name != '?':
                self._last_spk[u.key] = (name, now)
        if pc.speaker is None:
            # голос ещё не узнан - пока подписываем тем, кто говорил только что
            pc.speaker = recent[0] if recent else '?'
        self._maybe_split(u, now)

    def _maybe_split(self, u: _Utt, now: float):
        """Голос в подтверждённом звуке уверенно сменился - новый кусок."""
        pc = u.pieces[-1]
        tr = u.tr
        if (len(self.voice_db) < 2 or pc.exact or pc.speaker in (None, '?')
                or tr.commit_t - pc.start_s < SPLIT_MIN_S
                or tr.commit_t - pc.checked_s < SPLIT_EVERY_S):
            return
        pc.checked_s = tr.commit_t
        b1 = int(tr.commit_t / BLOCK_S)
        try:
            # Дешёвая проба: один эмбеддинг последних секунд. Полный разбор
            # окнами (~0.4 с процессора) - только если там уверенно другой голос.
            name, sim, _ = self._identify_speaker(u.audio(b1 - int(SPLIT_PROBE_S / BLOCK_S), b1))
            if name in ('?', pc.speaker):
                return
            b0 = int(max(pc.start_s, tr.commit_t - SPLIT_REGION_S) / BLOCK_S)
            parts = self._split_by_speaker(u.audio(b0, b1))
        except Exception as e:
            print(f'split: {e}', file=sys.stderr)
            return
        if not parts:
            return
        a, b, name, sim = parts[-1]
        if name == pc.speaker or b - a < 2.0 * SAMPLE_RATE:
            return
        cut = b0 * BLOCK_S + a / SAMPLE_RATE
        k = next((i for i in range(pc.first_word, len(tr.words)) if tr.words[i].start >= cut),
                 len(tr.words))
        if k == pc.first_word:
            pc.speaker, pc.sim = name, sim
        else:
            u.pieces.append(_Piece(k, cut, name, sim, used_s=(b - a) / SAMPLE_RATE,
                                   checked_s=tr.commit_t))
        self._last_spk[u.key] = (name, now)

    def _split_by_speaker(self, audio: np.ndarray) -> Optional[list[tuple[int, int, str, float]]]:
        """
        Режет звук на куски по смене знакомого голоса.
        None - резать не нужно. Окна считаются одним батчем эмбеддера.
        """
        if len(self.voice_db) < 2 or self.embedder is None:
            return None
        sr = SAMPLE_RATE
        win = int(SPLIT_WIN_S * sr)
        hop = int(SPLIT_HOP_S * sr)
        starts = list(range(0, len(audio) - win + 1, hop))
        if len(starts) < 2 * SPLIT_MIN_RUN:
            return None

        chunks = np.stack([audio[s:s + win] for s in starts]).astype(np.float32)
        rms = np.sqrt(np.mean(chunks ** 2, axis=1))
        embs = self.embedder.embed_batch(chunks)
        names = list(self.voice_db.keys())
        refs = np.stack([self.voice_db[n] for n in names]).astype(np.float32)
        sims = embs @ refs.T

        labels: list[Optional[int]] = []
        quiet = max(1e-4, float(np.median(rms)) * 0.25)
        for i in range(len(starts)):
            order = np.argsort(sims[i])[::-1]
            best = sims[i, order[0]]
            second = sims[i, order[1]] if len(order) > 1 else -1.0
            ok = rms[i] > quiet and best >= SPLIT_SIM and best - second >= SIM_MARGIN
            labels.append(int(order[0]) if ok else None)

        # одиночное окно посреди чужих - выброс
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
                labels[i] = labels[i - 1]

        runs: list[list] = []   # [label, first, last]
        for i, lab in enumerate(labels):
            if lab is None:
                continue
            if runs and runs[-1][0] == lab:
                runs[-1][2] = i
            else:
                runs.append([lab, i, i])
        runs = [r for r in runs if r[2] - r[1] + 1 >= SPLIT_MIN_RUN]
        merged: list[list] = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(r)
        if len(merged) < 2:
            return None

        # граница - посередине между серией одного и серией другого голоса,
        # уточняем по самому тихому месту рядом
        frame = int(0.02 * sr)
        cuts = [0]
        for a, b in zip(merged, merged[1:]):
            end_a = starts[a[2]] + win // 2
            start_b = starts[b[1]] + win // 2
            mid = (end_a + start_b) // 2
            lo = max(cuts[-1] + frame, mid - int(0.35 * sr))
            hi = min(len(audio) - frame, mid + int(0.35 * sr))
            best_at = mid
            if hi - lo > frame:
                seg = audio[lo:hi]
                n = len(seg) // frame
                if n > 0:
                    en = np.sqrt(np.mean(seg[:n * frame].reshape(n, frame) ** 2, axis=1))
                    best_at = lo + int(np.argmin(en)) * frame + frame // 2
            cuts.append(best_at)
        cuts.append(len(audio))

        parts = []
        for k, r in enumerate(merged):
            sim = float(np.mean([sims[i, r[0]] for i in range(r[1], r[2] + 1)]))
            parts.append((cuts[k], cuts[k + 1], names[r[0]], sim))
        return parts

    # ---------- распознавание ----------

    def _run_batch(self, batch: list[_Job], sched: _SttScheduler):
        started = time.monotonic()
        for job in batch:
            metrics.add('stt_wait', started - job.submitted)
        work = []
        for job in batch:
            u = job.utt
            if u.done:
                continue
            end_s = job.upto * BLOCK_S
            b0 = max(u.dropped, int(u.tr.window_start(end_s) / BLOCK_S))
            if job.upto - b0 < 4:
                if job.final:
                    self._close(u)
                continue
            work.append((job, b0, end_s))
        if not work:
            return

        audios = [job.utt.audio(b0, job.upto) for job, b0, _ in work]
        stt = self.stt
        gpu = stt.gpu
        t0 = time.perf_counter()
        hyps = stt.recognize(audios)
        dt = time.perf_counter() - t0
        metrics.add('infer', dt)
        if gpu and not stt.gpu:
            # видеокарту отобрали на ходу (сброс драйвера) - модель сама перешла
            # на процессор; показываем это, но настройку не трогаем: при
            # следующем запуске видеокарта снова попробуется
            print(f'видеокарта отвалилась: {stt.gpu_error}', file=sys.stderr)
            self.device_name = 'Процессор'
            self.device_changed.emit(False, 'lost')
        now_pc = time.perf_counter()
        span = max(dt, now_pc - self._stt_last_end)
        self._stt_last_end = now_pc
        # доля времени, которое поток распознавания занят (сглаженная)
        self._stt_load += 0.25 * (min(1.0, dt / span if span > 0 else 1.0) - self._stt_load)
        rec = (sum(len(a) for a in audios) / SAMPLE_RATE, dt, len(work))
        self._stt_hist.append(rec)
        self._stt_hist_all.append(rec)
        if sched.stopped:
            return

        for (job, b0, end_s), hyp in zip(work, hyps):
            u = job.utt
            off = b0 * BLOCK_S
            words = [Word(w.text, w.start + off, w.end + off) for w in hyp.words]
            commit_before = u.tr.commit_t
            u.tr.update(words, end_s, job.final, off)
            if _DEBUG_STREAM:
                print(f'~ {u.key}#{u.n} {"F" if job.final else "P"} [{off:.2f}-{end_s:.2f}] '
                      f'commit {commit_before:.2f}->{u.tr.commit_t:.2f} '
                      f'words: {len(words)} recognized, {len(u.tr.words)} committed, '
                      f'{len(u.tr.draft)} draft')
            if job.final and not u.tr.sentence_windows and FINAL_CTX_S > 0:
                self._refine_final(u, job.upto)
            self._resolve_speaker(u, end_s, job.final)
            if job.final:
                u.done = True
            self._emit(u, job.final)
            self._trim(u)

    def _refine_final(self, u: _Utt, upto: int):
        """Конец закрытой фразы одним окном - пунктуация для подтверждённого (FINAL_CTX_S)."""
        b0 = max(u.dropped, upto - int(FINAL_CTX_S / BLOCK_S))
        audio = u.audio(b0, upto)
        if len(audio) < SAMPLE_RATE // 2:
            return
        try:
            hyp = self.stt.recognize([audio])[0]
        except Exception as e:
            print(f'финал фразы: {e}', file=sys.stderr)
            return
        off = b0 * BLOCK_S
        u.tr.refine([Word(w.text, w.start + off, w.end + off) for w in hyp.words], off)

    def _close(self, u: _Utt):
        """Закрыть фразу без нового распознавания: черновик становится итогом."""
        u.tr.words.extend(u.tr.draft)
        u.tr.draft = []
        u.done = True
        self._emit(u, True)

    def _emit(self, u: _Utt, final: bool):
        transcript = u.tr
        worker = self._tr               # переводчик: один на весь проход, может смениться между проходами
        for i, pc in enumerate(u.pieces):
            last = i == len(u.pieces) - 1
            end = len(transcript.words) if last else u.pieces[i + 1].first_word
            done_words = transcript.words[pc.first_word:end]
            done = join(done_words)
            draft = transcript.shown_draft() if last and not final else ''
            closed = final or not last
            msg = (pc.speaker or '?', done, draft, closed)
            if u.sent.get(i) == msg:
                continue
            had = i in u.sent
            u.sent[i] = msg
            if not done and not draft:
                if had:
                    overlay.push_drop(u.key, u.seg(i))
                continue
            # когда каждое слово кончилось в звуке - для замера задержки показа
            ends = [u.t0 + w.end for w in done_words]
            if draft:
                ends += [u.t0 + w.end for w in transcript.shown_draft_words()]
            overlay.push_text(u.key, u.seg(i), msg[0], done, draft, closed, ends)
            if worker is not None:
                self._queue_translation(worker, u, i, pc, done_words, closed,
                                        transcript.shown_draft_words() if draft else [])
            if closed and done:
                started = time.time() - (time.monotonic() - (u.t0 + pc.start_s))
                if worker is not None:
                    # в историю - вместе с переводом, когда он будет готов
                    worker.submit(('close', u.run, u.key, u.seg(i), msg[0], done, started,
                                   self.history.session_id))
                else:
                    self.history.add(u.key, msg[0], done, started)

    @staticmethod
    def _queue_translation(tr, u: _Utt, i: int, pc: _Piece, words: list, closed: bool,
                           tail: Optional[list] = None):
        """
        Отдаёт переводчику законченные предложения куска - подтверждённые и
        с точкой, за которой уже есть следующее слово (знак у последнего слова
        ещё может поменяться). Длинную фразу без точки - по запятой.
        Недоговорённый хвост - черновиком, чтобы перевод не ждал точки.
        """
        start, n = pc.tr_upto, len(words)
        who = pc.speaker or '?'
        if n > start:
            cut = start
            if closed:
                cut = n
            else:
                for j in range(start, n - 1):
                    if ends_sentence(words[j].text):
                        cut = j + 1
                if cut == start and n - start >= TR_CLAUSE_WORDS:
                    for j in range(n - 2, start + TR_CLAUSE_WORDS // 2, -1):
                        if words[j].text.endswith((',', ';', ':')):
                            cut = j + 1
                            break
                    else:
                        if n - start >= 2 * TR_CLAUSE_WORDS:
                            cut = n - 1
            if cut > start:
                pc.tr_upto = cut
                pc.tr_draft, pc.tr_draft_text = 0, ''
                end = u.t0 + words[cut - 1].end
                metrics.add('tr_confirm', time.monotonic() - end)
                tr.submit(('sent', u.run, u.key, u.seg(i), who, join(words[start:cut]), end,
                           time.monotonic()))
        open_words = words[pc.tr_upto:] + (list(tail or []) if TR_DRAFT_ASR else [])
        rest = len(open_words)
        text = join(open_words)
        if closed or rest < TR_DRAFT_MIN_WORDS or text == pc.tr_draft_text:
            return
        now = time.monotonic()
        if now - pc.tr_draft_t < TR_DRAFT_EVERY_S and rest - pc.tr_draft < 4:
            return
        pc.tr_draft, pc.tr_draft_t, pc.tr_draft_text = rest, now, text
        tr.draft(u.run, u.key, u.seg(i), who, text, u.t0 + open_words[-1].end)

    def _trim(self, u: _Utt):
        """Освобождает звук, который больше не понадобится."""
        if u.done:
            u.blocks = []
            return
        keep_s = u.tr.commit_t - max(KEEP_AUDIO_S, LEFT_CTX_S)
        pc = u.pieces[-1]
        if not pc.exact and pc.used_s < SPEAKER_MAX_S:
            keep_s = min(keep_s, pc.start_s)
        kb = int(keep_s / BLOCK_S)
        for i in range(u.dropped, min(kb, len(u.blocks))):
            u.blocks[i] = None
        u.dropped = max(u.dropped, kb)

    # ---------- VAD ----------

    def _vad_loop(self, q: queue.Queue, sched: _SttScheduler, stop_event: threading.Event):
        short_sil = _blocks(MIN_SILENCE_MS)
        long_sil = _blocks(LONG_SILENCE_MS)
        long_phrase = _blocks(LONG_PHRASE_S * 1000)
        min_speech = _blocks(MIN_SPEECH_MS)
        max_blocks = _blocks(MAX_UTT_S * 1000)
        tail_keep = _blocks(TAIL_KEEP_MS)
        preroll_n = _blocks(PREROLL_MS)
        early_blocks = preroll_n + _blocks(EARLY_PARTIALS_S * 1000)
        pause_partial = _blocks(PAUSE_PARTIAL_MS)
        vad = self.vad

        sources: dict[str, _Source] = {}
        last_maintenance = time.monotonic()

        def submit(u: _Utt, upto: int, final: bool, now: float):
            sched.submit(_Job(utt=u, upto=upto, final=final, submitted=now))

        def begin(src: _Source, first_blocks: list, first_probs: list, t_audio: float,
                  announce: bool = True):
            """t_audio - когда драйвер отдал последний из first_blocks (конец его звука)."""
            src.n_utt += 1
            u = _Utt(src.key, src.kind, src.n_utt, src.speaker,
                     t_audio - len(first_blocks) * BLOCK_S, self.draft_guard_s,
                     getattr(self.stt, 'sentence_windows', False))
            u.run = self._run
            u.blocks = first_blocks
            u.probs = first_probs
            u.speech_blocks = sum(1 for p in first_probs if p >= VAD_OFF)
            u.since_partial = len(first_blocks)
            src.utt = u
            if announce and src.kind == 'mon':
                overlay.push_speaking(src.key, None, True)

        def finish(src: _Source, now: float, audio_stopped: bool = False,
                   announce: bool = True):
            u = src.utt
            if u.speech_blocks >= min_speech or u.partial_sent:
                # лишняя тишина в конце только тратит процессор
                trailing = 0
                for p in reversed(u.probs):
                    if p >= VAD_OFF:
                        break
                    trailing += 1
                upto = max(1, len(u.blocks) - max(0, trailing - tail_keep))
                if audio_stopped:
                    # звук оборвался на слове: немного тишины помогает
                    # модели дописать последний звук
                    u.blocks.extend([np.zeros(BLOCK_SIZE, dtype=np.float32)] * tail_keep)
                    upto = len(u.blocks)
                submit(u, upto, True, now)
            else:
                u.done = True
            if audio_stopped:
                # в хвосте речь, а не тишина: в preroll она задвоила бы слова
                src.preroll = collections.deque(maxlen=preroll_n)
            else:
                src.preroll = collections.deque(u.blocks[-preroll_n:], maxlen=preroll_n)
            src.utt = None
            if announce and src.kind == 'mon':
                overlay.push_speaking(src.key, None, False)

        def get_source(key: str, kind: str, speaker: Optional[str]) -> _Source:
            src = sources.get(key)
            if src is None:
                src = _Source(key=key, kind=kind, vad_state=vad.new_state(),
                              preroll=collections.deque(maxlen=preroll_n))
                sources[key] = src
            src.speaker = speaker
            return src

        silence_block = np.zeros(BLOCK_SIZE, dtype=np.float32)

        def maintenance(now: float):
            for key, src in list(sources.items()):
                idle = now - src.last_block_t
                if src.kind == 'bot':
                    # Бот присылает звук, только пока человек говорит: пропуск
                    # дописываем тишиной, конец фразы решит VAD.
                    if src.utt is not None and idle > BOT_GAP_S:
                        start = max(src.filled_t, src.last_block_t)
                        n = int((now - start) / BLOCK_S)
                        for _ in range(n):
                            if src.utt is None:
                                break
                            step(src, silence_block, 0.0, now)
                        src.filled_t = start + n * BLOCK_S
                    if src.utt is not None and idle > BOT_GIVE_UP_S:
                        finish(src, now, audio_stopped=True)
                    if src.utt is None and idle > SOURCE_IDLE_DROP_S:
                        del sources[key]
                elif src.utt is not None and idle > 1.0:
                    # микрофон и приложение шлют поток всегда: пауза у них - обрыв
                    finish(src, now, audio_stopped=True)

        def step(src: _Source, chunk: np.ndarray, prob: float, now: float,
                 t_audio: Optional[float] = None):
            interval = max(1, int(self.partial_interval_ms / 1000 / BLOCK_S))
            partial_min = max(1, int(self.partial_min_s / BLOCK_S))
            u = src.utt
            if u is None:
                if prob >= VAD_ON:
                    pre = list(src.preroll)
                    src.preroll.clear()
                    begin(src, pre + [chunk], [0.0] * len(pre) + [prob], t_audio or now)
                else:
                    src.preroll.append(chunk)
                return

            u.blocks.append(chunk)
            u.probs.append(prob)
            u.since_partial += 1
            if prob >= VAD_OFF:
                u.speech_blocks += 1
                u.silence = 0
            else:
                u.silence += 1

            need = long_sil if u.speech_blocks >= long_phrase else short_sil
            if u.silence >= need:
                finish(src, now)
                return
            if u.silence == pause_partial and u.since_partial >= u.silence and u.partial_sent:
                # Пауза посреди фразы: черновиков во время тишины нет, и последнее
                # слово (его держат у края окна) висело бы до продолжения речи.
                # Одно распознавание сейчас - слово уже договорено, его видно.
                submit(u, len(u.blocks), False, now)
                u.since_partial = 0
                return
            if len(u.blocks) >= max_blocks:
                # оверлей продолжит ту же карточку: тот же источник, без паузы
                finish(src, now, announce=False)
                begin(src, [], [], t_audio or now, announce=False)
                return
            # в начале фразы окно короткое и дешёвое - обновляем чаще,
            # чтобы первые слова появились быстрее
            iv = interval if len(u.blocks) > early_blocks else max(1, int(interval * 0.55))
            # распознавание не успевает (говорят несколько человек сразу) -
            # черновики реже, но ровно, вместо рывков с выброшенными черновиками
            if self._stt_load > 0.7:
                iv = int(iv * min(3.0, self._stt_load / 0.7))
            if (u.silence == 0 and u.since_partial >= iv
                    and len(u.blocks) >= partial_min):
                submit(u, len(u.blocks), False, now)
                u.since_partial = 0
                u.partial_sent = True

        stopping = False
        while not stopping:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                now = time.monotonic()
                maintenance(now)
                last_maintenance = now
                continue
            if item is None:
                break
            # всё, что накопилось, - одним проходом: блоки разных источников
            # идут в VAD одним вызовом
            items = [item]
            while len(items) < 64:
                try:
                    it = q.get_nowait()
                except queue.Empty:
                    break
                if it is None:
                    stopping = True
                    break
                items.append(it)

            now = time.monotonic()
            if now - last_maintenance > 0.1:
                maintenance(now)
                last_maintenance = now

            pending: dict[str, list] = {}
            for key, kind, speaker, chunk, t_audio in items:
                if len(chunk) != BLOCK_SIZE:
                    continue
                metrics.add('audio_queue', now - t_audio)
                src = get_source(key, kind, speaker)
                src.last_block_t = now
                src.filled_t = now
                pending.setdefault(key, []).append((chunk, t_audio))
            while pending:
                keys = list(pending)
                got = [pending[k].pop(0) for k in keys]
                blocks = [b for b, _ in got]
                srcs = [sources[k] for k in keys]
                try:
                    probs = vad(blocks, [s.vad_state for s in srcs])
                except Exception as e:
                    print(f'VAD error: {e}', file=sys.stderr)
                    break
                for s, (b, t_audio), p in zip(srcs, got, probs):
                    step(s, b, float(p), now, t_audio)
                pending = {k: v for k, v in pending.items() if v}
