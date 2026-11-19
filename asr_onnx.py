"""
Модели без PyTorch: GigaAM v3 e2e RNNT, Silero VAD v6 и WeSpeaker ResNet34
на ONNX Runtime + numpy.

Почему так. PyTorch держит пул потоков на все ядра и между вызовами
крутит их вхолостую - на 6-ядерном Ryzen это ~500% CPU ради 0.4 мс работы
VAD. Здесь у каждой сессии один поток и холостое ожидание выключено:
распознавание стоит ~6% одного ядра на секунду звука, VAD на все
источники - ~1.5%.

Кодировщик GigaAM квантован в int8 (только матричные умножения): в 1.5
раза быстрее fp32 при расхождении текста ~1.5% слов. Декодер RNNT и
joint-сеть маленькие (LSTM 320 и три матрицы) - их считает numpy сразу
для всего батча, без сотен мелких вызовов ORT на фразу.

Видеокарта (по желанию, DirectML - любая карта под Windows): кодировщик
в fp16 (веса в файле - int8 со шкалой по каналу, разворачиваются при
загрузке: файл вдвое меньше), окно всегда одной длины, 10.24 с. Так DirectML считает окно за
~20-40 мс и почти не трогает процессор; с меняющейся длиной он был
медленнее процессора.

Файлы моделей - models/onnx/, готовит tools/export_onnx.py.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
FRAME_S = 0.04                  # кадр кодировщика: шаг 10 мс × прореживание 4
BLANK = 1024
MAX_TOKENS_PER_STEP = 3
# графы для процессора и видеокарты, веса общие - int8 по каналам в WEIGHTS (tools/export_gpu.py)
CPU_MODEL = 'v3_e2e_rnnt_encoder.cpu.onnx'
GPU_MODEL = 'v3_e2e_rnnt_encoder.gpu.onnx'
WEIGHTS = 'v3_e2e_rnnt_encoder.npz'
GPU_FRAMES = 1024               # окно видеокарты: 10.24 с звука, больше окна стриминга
GPU_SAMPLES = (GPU_FRAMES - 1) * 160 + 320


def session_options(threads: int = 1) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # главное: потоки засыпают сразу после работы, а не крутятся в ожидании
    so.add_session_config_entry('session.intra_op.allow_spinning', '0')
    so.add_session_config_entry('session.inter_op.allow_spinning', '0')
    so.log_severity_level = 3
    return so


def _preload_directml():
    """
    DirectML.dll - свою, из пакета onnxruntime. В Windows есть и системная
    копия другой версии; собранная программа подхватывала её, и кодировщик
    выдавал NaN. Уже загруженную библиотеку Windows больше не ищет.
    """
    p = Path(ort.__file__).parent / 'capi' / 'DirectML.dll'
    if not p.exists():
        return
    try:
        import ctypes
        ctypes.WinDLL(str(p))
    except OSError:
        pass


_preload_directml()


def available_gpu() -> bool:
    return 'DmlExecutionProvider' in ort.get_available_providers()


def gpu_models_present() -> bool:
    """Модели для видеокарты есть (в версии «без видеокарты» их нет)."""
    from paths import resource_path
    return ((model_dir() / GPU_MODEL).exists()
            or (Path(resource_path('models/parakeet')) / 'encoder-model.gpu.onnx').exists())


DML_OPTIONS = {'performance_preference': 'high_performance', 'device_filter': 'gpu'}


# Окно с NaN/inf от видеокарты пересчитывается процессором; после стольких
# таких окон распознавание насовсем уходит на процессор (до перезапуска).
GPU_BAD_LIMIT = 3

# Видеокарту сбросили посреди работы (DXGI_ERROR_DEVICE_REMOVED: драйвер,
# другая программа). ORT держит мёртвое D3D12-устройство до конца процесса,
# и новые сессии DirectML не создаются - до перезапуска только процессор.
gpu_lost = ''


def err_text(e: BaseException) -> str:
    """
    Текст ошибки ORT. Сообщения Windows в нём на языке системы (cp1251),
    а pybind читает их как utf-8 и вместо ошибки бросает UnicodeDecodeError.
    """
    if isinstance(e, UnicodeDecodeError) and isinstance(e.object, bytes):
        return e.object.decode('mbcs', 'replace')[-300:].strip()
    return str(e)


DEVICE_LOST = ('887A0005', '887A0006', '887A0007', '887A0020')     # DXGI: removed, hung, reset, driver


def gpu_open_failed(model, e: BaseException):
    """Сессия на видеокарте не собралась. Если устройство сброшено - это насовсем, как gpu_failed."""
    global gpu_lost
    model._gpu = None
    model.gpu_error = err_text(e)[-200:] or 'error'
    if any(c in model.gpu_error.upper() for c in DEVICE_LOST):
        gpu_lost = model.gpu_error


def gpu_failed(model, e: BaseException):
    """Видеокарта отказала на ходу - модель насовсем уходит на процессор."""
    global gpu_lost
    model.gpu = False
    model._gpu = None
    model.gpu_error = gpu_lost = err_text(e)[-200:] or 'error'


def directml_loaded() -> str:
    """Какие DirectML.dll загружены в процесс - для журнала."""
    try:
        import psutil
        return ', '.join(sorted({m.path for m in psutil.Process().memory_maps()
                                 if m.path.lower().endswith('directml.dll')})) or 'нет'
    except Exception as e:
        return f'не узнать ({e})'


def gpu_selfcheck_twice(check):
    """
    Самопроверка видеокарты; не прошла - ещё раз, с подробностями в журнал.
    Один раз NaN поймали в отдельном процессе из исходников и не смогли
    повторить (35 процессов, с нагрузкой на видеокарту и без): если это
    случайность, второй прогон её отличит, если нет - откат на процессор.
    """
    try:
        check()
    except RuntimeError as e:
        print(f'самопроверка видеокарты: {e}; DirectML: {directml_loaded()}; повтор', file=sys.stderr)
        check()


def gpu_bad_window(model, what: str) -> bool:
    """Видеокарта выдала NaN/inf на окне. True - пора насовсем на процессор."""
    model._gpu_bad = getattr(model, '_gpu_bad', 0) + 1
    print(f'{what}: NaN от видеокарты ({model._gpu_bad}/{GPU_BAD_LIMIT}), окно - процессором', file=sys.stderr)
    return model._gpu_bad >= GPU_BAD_LIMIT


# Одно D3D12-устройство на процесс, и DirectML не терпит, когда с ним
# работают из двух потоков сразу: сборка сессии Parakeet (переключили на
# английский), пока GigaAM считает окно, - и Windows сбрасывает видеокарту
# (DXGI_ERROR_DEVICE_REMOVED, воспроизводится с первого раза). Поэтому всё,
# что касается видеокарты - распознавание, сборка, самопроверка, выгрузка
# сессии, - строго по очереди, под этим замком.
GPU_LOCK = threading.RLock()


class GpuReleased(Exception):
    """Сессию видеокарты выгрузили (сменили модель), пока окно ждало очереди."""


@contextmanager
def gpu_turn():
    """Очередь к видеокарте: сколько ждали (живые субтитры и файл - отдельно, metrics.py)."""
    import metrics
    t = time.monotonic()
    with GPU_LOCK:
        live = threading.current_thread().name == 'stt'
        metrics.add('gpu_wait' if live else 'gpu_wait_file', time.monotonic() - t)
        yield


def release_gpu(model):
    """Выгрузить сессию видеокарты - тоже в очереди: деструктор работает с устройством."""
    with GPU_LOCK:
        model.gpu = False
        model._gpu = None


def shared_weights(npz: Path, cpu: bool) -> tuple[list, list]:
    """
    Веса из .npz (tools/export_gpu.py): int8 со шкалой по выходному каналу.
    Видеокарте - всё в fp16. Процессору - матрицы как есть, int8 и шкала
    (его граф считает их в int8), остальное (свёртки) - в fp32.
    """
    names, vals = [], []
    with np.load(npz) as d:
        raw = set(str(x) for x in d['raw']) if 'raw' in d.files else set()
        for k, name in enumerate(str(n) for n in d['names']):
            q, sc = d[f'q{k}'], d[f's{k}']
            if cpu and name in raw:
                names += [name + '_quantized', name + '_scale']
                vals += [q, sc.reshape(-1).astype(np.float32)]
            else:
                names.append(name)
                vals.append((q.astype(np.float32) * sc).astype(np.float32 if cpu else np.float16))
    return names, [ort.OrtValue.ortvalue_from_numpy(v) for v in vals]


def open_gpu_session(path: Path, npz: Path = None) -> ort.InferenceSession:
    """
    Сессия DirectML. Если есть .npz (по умолчанию - рядом с графом), большие
    веса в файле модели пустые: разворачиваем их из int8 в fp16 и отдаём ORT
    вместо внешних данных. Файл вдвое меньше fp16, считает видеокарта ту же
    fp16-модель.
    """
    so = session_options(1)
    so.enable_mem_pattern = False
    npz = Path(npz) if npz else Path(path).with_suffix('.npz')
    keep = []
    if npz.exists():
        names, keep = shared_weights(npz, cpu=False)
        so.add_external_initializers(names, keep)
    sess = ort.InferenceSession(str(path), so, providers=[('DmlExecutionProvider', DML_OPTIONS),
                                                          'CPUExecutionProvider'])
    del keep        # ORT скопировал веса в граф - свои копии больше не нужны
    if 'DmlExecutionProvider' not in sess.get_providers():
        raise RuntimeError('DirectML недоступен')
    return sess


def open_cpu_session(path: Path, npz: Path, threads: int) -> ort.InferenceSession:
    """Сессия процессора для графа с весами в общем .npz (те же, что у видеокарты)."""
    so = session_options(threads)
    names, keep = shared_weights(npz, cpu=True)
    so.add_external_initializers(names, keep)
    sess = ort.InferenceSession(str(path), so, providers=['CPUExecutionProvider'])
    del keep
    return sess


@dataclass
class Word:
    text: str
    start: float        # секунды от начала окна: кадр первого токена
    end: float          # кадр последнего токена + 1


@dataclass
class Hypothesis:
    words: list[Word]

    @property
    def text(self) -> str:
        return ' '.join(w.text for w in self.words)


# ============================================================
# ПРИЗНАКИ
# ============================================================

class _GigaamFeatures:
    """Лог-мел GigaAM v3: окно 20 мс, шаг 10 мс, 64 полосы, без паддинга."""
    N_FFT = 320
    HOP = 160

    def __init__(self, fbank: np.ndarray, window: np.ndarray):
        self.fbank = fbank.astype(np.float32)
        self.window = window.astype(np.float32)

    def __call__(self, wav: np.ndarray, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frames = np.lib.stride_tricks.sliding_window_view(wav, self.N_FFT, axis=1)[:, ::self.HOP]
        spec = np.abs(np.fft.rfft(frames * self.window, self.N_FFT)).astype(np.float32) ** 2
        mel = np.log(np.clip(spec @ self.fbank, 1e-9, 1e9))
        feats = np.ascontiguousarray(mel.transpose(0, 2, 1), dtype=np.float32)
        return feats, ((lens - self.N_FFT) // self.HOP + 1).astype(np.int64)


def kaldi_fbank(wav: np.ndarray, fbank: np.ndarray) -> np.ndarray:
    """Kaldi fbank как у WeSpeaker (snip_edges, hamming, 80 полос) + вычитание среднего."""
    win, hop, n_fft = 400, 160, 512
    x = wav.astype(np.float32) * 32768.0
    if x.shape[-1] < win:
        x = np.pad(x, ((0, 0), (0, win - x.shape[-1])))
    frames = np.lib.stride_tricks.sliding_window_view(x, win, axis=1)[:, ::hop]
    frames = frames - frames.mean(axis=-1, keepdims=True)
    prev = np.concatenate([frames[..., :1], frames[..., :-1]], axis=-1)
    frames = frames - 0.97 * prev
    frames = frames * np.hamming(win).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, n_fft)).astype(np.float32) ** 2
    feats = np.log(np.maximum(spec @ fbank, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=1, keepdims=True)).astype(np.float32)


# ============================================================
# РАСПОЗНАВАНИЕ
# ============================================================

class GigaAM:
    def __init__(self, model_dir: Path, gpu: bool = False, threads: int = 1):
        model_dir = Path(model_dir)
        with np.load(model_dir / 'fbanks.npz') as d:
            self._features = _GigaamFeatures(d['gigaam_v3'], d['gigaam_v3_window'])
        self.vocab: list[str] = [''] * 1025
        with open(model_dir / 'v3_e2e_rnnt_vocab.txt', encoding='utf-8') as f:
            for line in f:
                tok, idx = line.rstrip('\n').rsplit(' ', 1)
                self.vocab[int(idx)] = tok
        self._dir = model_dir
        self.threads = threads
        self.gpu = False
        self.gpu_error = ''
        self._enc = None            # процессор: int8, любая длина
        self._gpu = None            # видеокарта: fp16, окно постоянной длины
        if gpu:
            self._open_gpu()
        if not self.gpu:
            self._enc = self._open_cpu(threads)
        self._load_head(model_dir)

    def _open_cpu(self, threads: int) -> ort.InferenceSession:
        return open_cpu_session(self._dir / CPU_MODEL, self._dir / WEIGHTS, threads)

    def _open_gpu(self):
        # DirectML не умеет int8-умножения - для видеокарты модель в fp16.
        # И длина окна у неё одна на всё: с меняющейся длиной DirectML
        # не собирает граф целиком и работает медленнее процессора.
        path = self._dir / GPU_MODEL
        if not available_gpu():
            self.gpu_error = 'no-gpu'
            return
        if gpu_lost:
            self.gpu_error = 'lost'
            return
        if not path.exists():
            self.gpu_error = 'no-model'
            return
        # Длину не фиксируем в настройках сессии: со статической формой DirectML
        # собирает граф, который выдаёт мусор. Достаточно всегда подавать одну.
        try:
            with GPU_LOCK:
                self._gpu = open_gpu_session(path, self._dir / WEIGHTS)
                gpu_selfcheck_twice(self._gpu_selfcheck)
            self.gpu = True
        except Exception as e:
            gpu_open_failed(self, e)

    def _gpu_selfcheck(self):
        """
        Видеокарта должна слышать то же, что процессор. На некоторых
        сочетаниях драйвера и DirectML граф собирается неверно и выдаёт
        мусор - тогда остаёмся на процессоре.
        """
        t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
        tone = 150 + 80 * np.sin(2 * np.pi * 3 * t)
        a = (0.1 * np.sin(2 * np.pi * tone * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(np.float32)
        wav = np.zeros((1, GPU_SAMPLES), dtype=np.float32)
        wav[0, :len(a)] = a
        feats, _ = self._features(wav, np.array([GPU_SAMPLES], dtype=np.int64))
        n = (len(a) - self._features.N_FFT) // self._features.HOP + 1
        g, gl = self._gpu.run(['encoded', 'encoded_len'], {'audio_signal': feats, 'length': np.array([n])})
        cpu = self._open_cpu(1)
        f2, l2 = self._features(a[None], np.array([len(a)], dtype=np.int64))
        c, _ = cpu.run(['encoded', 'encoded_len'], {'audio_signal': f2, 'length': l2})
        k = min(int(gl[0]), c.shape[2])
        x, y = g[0, :, :k].ravel(), c[0, :, :k].ravel()
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if not cos > 0.9:
            raise RuntimeError(f'видеокарта считает неверно (сходство {cos:.2f})')

    def set_threads(self, threads: int):
        """Потоков процессора на кодировщик. Работы столько же, окно готово раньше."""
        if threads != self.threads and self._enc is not None:
            self._enc = self._open_cpu(threads)
        self.threads = threads

    def _load_head(self, model_dir: Path):
        """Веса LSTM-декодера и joint из npz (готовит tools/export_onnx.py)."""
        with np.load(model_dir / 'rnnt_head.npz') as d:
            h = {k: d[k].astype(np.float32) for k in d.files}
        self._embed = h['embed']                    # 1025×320
        self._lstm_w = h['lstm_w'].T.copy()         # 320×1280, порядок гейтов i o f c
        self._lstm_r = h['lstm_r'].T.copy()
        self._lstm_b = h['lstm_b']                  # 1280: Wb + Rb
        self._enc_w, self._enc_b = h['enc_w'], h['enc_b']       # 768×320
        self._pred_w, self._pred_b = h['pred_w'], h['pred_b']   # 320×320
        self._out_w, self._out_b = h['out_w'], h['out_b']       # 320×1025
        self.hidden = self._embed.shape[1]

    def _lstm(self, tokens: np.ndarray, h: np.ndarray, c: np.ndarray):
        g = self._embed[tokens] @ self._lstm_w + h @ self._lstm_r + self._lstm_b
        H = self.hidden
        i = 1 / (1 + np.exp(-g[:, :H]))
        o = 1 / (1 + np.exp(-g[:, H:2 * H]))
        f = 1 / (1 + np.exp(-g[:, 2 * H:3 * H]))
        c = f * c + i * np.tanh(g[:, 3 * H:])
        h = o * np.tanh(c)
        return h, c

    def recognize(self, audios: list[np.ndarray]) -> list[Hypothesis]:
        """Пачка фраз за один проход кодировщика. Короткие дополняются нулями."""
        if self.gpu:
            try:
                return self._recognize_gpu(audios)
            except GpuReleased:
                pass
            except Exception as e:
                # видеокарту отобрали (сброс драйвера, игра) - дальше на процессоре
                gpu_failed(self, e)
        return self._recognize_cpu(audios)

    def _encode_cpu(self, audios: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self._enc is None:
            self._enc = self._open_cpu(self.threads)
        lens = np.array([max(len(a), self._features.N_FFT) for a in audios], dtype=np.int64)
        wav = np.zeros((len(audios), int(lens.max())), dtype=np.float32)
        for i, a in enumerate(audios):
            wav[i, :len(a)] = a
        feats, feat_lens = self._features(wav, lens)
        enc, enc_lens = self._enc.run(['encoded', 'encoded_len'],
                                      {'audio_signal': feats, 'length': feat_lens})
        return enc.transpose(0, 2, 1), enc_lens.astype(np.int64)      # B×T×768

    def _recognize_cpu(self, audios: list[np.ndarray]) -> list[Hypothesis]:
        enc, enc_lens = self._encode_cpu(audios)
        proj = enc @ self._enc_w + self._enc_b             # B×T×320 - один раз на всё
        return self._greedy(proj, np.minimum(enc_lens, proj.shape[1]))

    def _recognize_gpu(self, audios: list[np.ndarray]) -> list[Hypothesis]:
        projs, lens = [], []
        for a in audios:
            if len(a) > GPU_SAMPLES:
                # длиннее окна видеокарты (бывает только у файлов) - процессором
                enc, el = self._encode_cpu([a])
            else:
                wav = np.zeros((1, GPU_SAMPLES), dtype=np.float32)
                wav[0, :len(a)] = a
                feats, _ = self._features(wav, np.array([GPU_SAMPLES], dtype=np.int64))
                n = (max(len(a), self._features.N_FFT) - self._features.N_FFT) // self._features.HOP + 1
                with gpu_turn():
                    if self._gpu is None:
                        raise GpuReleased()
                    enc, el = self._gpu.run(['encoded', 'encoded_len'],
                                            {'audio_signal': feats, 'length': np.array([n], dtype=np.int64)})
                if np.isfinite(enc[0, :, :int(el[0])]).all():
                    enc, el = enc.transpose(0, 2, 1), el.astype(np.int64)
                elif gpu_bad_window(self, 'GigaAM'):
                    raise RuntimeError('видеокарта выдаёт NaN')     # recognize() перейдёт на процессор
                else:
                    enc, el = self._encode_cpu([a])
            projs.append(enc[0].astype(np.float32) @ self._enc_w + self._enc_b)
            lens.append(min(int(el[0]), enc.shape[1]))
        T = max(p.shape[0] for p in projs)
        proj = np.zeros((len(projs), T, projs[0].shape[1]), dtype=np.float32)
        for i, p in enumerate(projs):
            proj[i, :p.shape[0]] = p
        return self._greedy(proj, np.array(lens, dtype=np.int64))

    def _greedy(self, proj: np.ndarray, lens: np.ndarray) -> list[Hypothesis]:
        """Жадный RNNT сразу для всего батча: на шаге одна матрица на всех."""
        B = proj.shape[0]
        H = self.hidden
        h = np.zeros((B, H), np.float32)
        c = np.zeros((B, H), np.float32)
        h, c = self._lstm(np.full(B, BLANK), h, c)
        dec = h @ self._pred_w + self._pred_b
        t = np.zeros(B, np.int64)
        emitted = np.zeros(B, np.int64)
        tokens: list[list[tuple[int, int]]] = [[] for _ in range(B)]
        active = np.flatnonzero(t < lens)
        while active.size:
            x = np.maximum(proj[active, t[active]] + dec[active], 0.0)
            best = (x @ self._out_w + self._out_b).argmax(axis=1)
            emit = best != BLANK
            if emit.any():
                idx = active[emit]
                for b, tok in zip(idx, best[emit]):
                    tokens[b].append((int(tok), int(t[b])))
                nh, nc = self._lstm(best[emit], h[idx], c[idx])
                h[idx], c[idx] = nh, nc
                dec[idx] = nh @ self._pred_w + self._pred_b
                emitted[idx] += 1
            step = ~emit | (emitted[active] >= MAX_TOKENS_PER_STEP)
            adv = active[step]
            t[adv] += 1
            emitted[adv] = 0
            active = active[t[active] < lens[active]]
        return [self._words(tk) for tk in tokens]

    def _words(self, tokens: list[tuple[int, int]]) -> Hypothesis:
        words: list[Word] = []
        for tok, frame in tokens:
            piece = self.vocab[tok]
            if piece.startswith('▁') or not words:
                words.append(Word(piece.lstrip('▁'), frame * FRAME_S, (frame + 1) * FRAME_S))
            else:
                w = words[-1]
                w.text += piece
                w.end = (frame + 1) * FRAME_S
        out: list[Word] = []
        for w in words:
            if not w.text:
                continue
            if out and w.text not in '—–' and not any(ch.isalnum() for ch in w.text):
                # отдельно стоящий знак («▁,») - к предыдущему слову; тире - отдельно
                out[-1].text += w.text
                out[-1].end = w.end
            else:
                out.append(w)
        return Hypothesis(out)


# ============================================================
# VAD
# ============================================================

class SileroVad:
    """
    Silero VAD v6, одна сессия на все источники. Состояние каждого
    источника - строка в state; блоки разных источников, пришедшие
    одновременно, считаются одним вызовом.
    """
    BLOCK = 512
    CONTEXT = 64

    def __init__(self, model_path: Path):
        self._sess = ort.InferenceSession(str(model_path), session_options(1),
                                          providers=['CPUExecutionProvider'])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def new_state(self) -> list:
        return [np.zeros((2, 1, 128), np.float32), np.zeros((1, self.CONTEXT), np.float32)]

    def __call__(self, blocks: list[np.ndarray], states: list[list]) -> np.ndarray:
        x = np.concatenate([np.concatenate([s[1], b[None]], axis=1) for b, s in zip(blocks, states)])
        st = np.concatenate([s[0] for s in states], axis=1)
        out, st = self._sess.run(['output', 'stateN'], {'input': x, 'state': st, 'sr': self._sr})
        for i, s in enumerate(states):
            s[0] = st[:, i:i + 1]
            s[1] = x[i:i + 1, -self.CONTEXT:]
        return out[:, 0]

    def probs(self, audio: np.ndarray) -> np.ndarray:
        """Вероятности речи по блокам целого файла (для записи эталонов)."""
        st = self.new_state()
        n = len(audio) // self.BLOCK
        res = np.zeros(n, np.float32)
        for i in range(n):
            res[i] = self([audio[i * self.BLOCK:(i + 1) * self.BLOCK]], [st])[0]
        return res


# ============================================================
# ГОЛОСА
# ============================================================

class SpeakerEmbedder:
    """WeSpeaker ResNet34 (тот же чекпоинт, что был в pyannote): 256-мерный вектор."""

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        with np.load(model_dir / 'fbanks.npz') as d:
            self._fbank = d['wespeaker'].astype(np.float32)
        self._sess = ort.InferenceSession(str(model_dir / 'wespeaker_resnet34.onnx'),
                                          session_options(1), providers=['CPUExecutionProvider'])

    def embed_batch(self, audios: np.ndarray) -> np.ndarray:
        """audios: N×samples одной длины. Возвращает нормированные N×256."""
        feats = kaldi_fbank(np.atleast_2d(audios), self._fbank)
        (embs,) = self._sess.run(['embs'], {'feats': feats})
        embs = embs.astype(np.float32)
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    def embed(self, audio: np.ndarray) -> np.ndarray:
        return self.embed_batch(np.asarray(audio, np.float32)[None])[0]


def model_dir() -> Path:
    from paths import resource_path
    return Path(resource_path('models/onnx'))
