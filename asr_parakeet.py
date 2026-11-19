"""
Parakeet TDT 0.6B v3 (NVIDIA, CC-BY-4.0) - распознавание 25 европейских
языков одной моделью, включая английский и русский. Язык определяет сама,
пунктуацию и заглавные ставит сама. Нужна для английской речи: лекция на
английском, вопросы на русском.

ONNX-версия - istupakov/parakeet-tdt-0.6b-v3-onnx: признаки (nemo128.onnx),
кодировщик int8, предсказатель+joint одним графом. Интерфейс как у GigaAM:
recognize(звуки) -> гипотезы со словами и временем, поэтому потоковая
часть (streaming.py) работает без изменений.

TDT - тот же трансдьюсер, но на каждом шаге модель говорит ещё и сколько
кадров пропустить (0-4): кадров 80 мс, шагов вдвое-втрое меньше.

Целым файлом модель держится одного языка и выкидывает фразы на другом;
на окнах по несколько секунд, как в потоке, переключается сама.

Видеокарта: кодировщик в fp16 (веса в файле - int8, см. tools/export_gpu.py),
окно всегда одной длины, 12.8 с - как у GigaAM, иначе DirectML медленный.
Окно в 12 с считается за ~30 мс; признаки, предсказатель и joint - на
процессоре.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import onnxruntime as ort

import asr_onnx
from asr_onnx import (GPU_LOCK, GpuReleased, Hypothesis, Word, available_gpu, gpu_bad_window, gpu_failed,
                      gpu_open_failed, gpu_selfcheck_twice, gpu_turn, open_cpu_session, open_gpu_session,
                      session_options)

SAMPLE_RATE = 16000
FRAME_S = 0.08
MAX_TOKENS_PER_STEP = 10
# Короткое окно от начала фразы модель часто оставляет пустым: учена на
# целых фразах, а у окна конец обрезан посреди слова. 0.3 с тишины справа -
# как будто фраза договорена. Замер на 40 фразах LibriSpeech, окна 0.75-4 с
# от начала фразы: пустых 18 -> 12, верных слов 2485 -> 2510, лишних 204 -> 182.
# Тишина слева до 3 с (как было) - хуже всего: пустых 37.
RIGHT_PAD_S = 0.3
DURATIONS = (0, 1, 2, 3, 4)
# Графы для процессора и видеокарты, веса у них общие - WEIGHTS (tools/export_gpu.py)
CPU_MODEL = 'encoder-model.cpu.onnx'
GPU_MODEL = 'encoder-model.gpu.onnx'
WEIGHTS = 'encoder-model.npz'
HOP = 160
GPU_SAMPLES = int(12.8 * SAMPLE_RATE)     # окно видеокарты: длиннее любого окна стриминга и фразы файла
GPU_FRAMES = 1 + GPU_SAMPLES // HOP
SELFCHECK_TEXT = 'The weather is nice today, so we will go to the park.'


def _norm(w: str) -> str:
    return w.strip('.,!?;:').lower()


def model_dir() -> Path:
    from paths import resource_path
    return Path(resource_path('models/parakeet'))


def available() -> bool:
    d = model_dir()
    return (d / WEIGHTS).exists() and ((d / CPU_MODEL).exists() or (d / GPU_MODEL).exists())


def available_cpu() -> bool:
    return (model_dir() / CPU_MODEL).exists()


class Parakeet:
    sentence_windows = True     # окна от начала предложения (см. streaming.py)
    # Модель крупнее GigaAM: её ×15 на пробе - уже быстрый процессор (Ryzen 5
    # 7500F), второй поток там дал бы -0.05 с задержки за +30% нагрузки.
    # Пороги потоков (engine._tune_threads) - в 0.75 раза ниже.
    tune_scale = 0.75

    def __init__(self, model_dir: Path, gpu: bool = False, threads: int = 1):
        model_dir = Path(model_dir)
        self._dir = model_dir
        self.threads = threads
        self.gpu = False
        self.gpu_error = ''
        self._enc = None            # процессор: int8, любая длина
        self._gpu = None            # видеокарта: fp16, окно постоянной длины
        self.vocab: list[str] = []
        with open(model_dir / 'vocab.txt', encoding='utf-8') as f:
            for line in f:
                tok, idx = line.rstrip('\n').rsplit(' ', 1)
                i = int(idx)
                if i >= len(self.vocab):
                    self.vocab += [''] * (i + 1 - len(self.vocab))
                self.vocab[i] = tok
        self.blank = len(self.vocab) - 1
        self.V = len(self.vocab)
        so1 = session_options(1)
        self._pre = ort.InferenceSession(str(model_dir / 'nemo128.onnx'), so1, providers=['CPUExecutionProvider'])
        self._joint = ort.InferenceSession(str(model_dir / 'decoder_joint-model.int8.onnx'), so1,
                                           providers=['CPUExecutionProvider'])
        if gpu:
            self._open_gpu()
        if not self.gpu:
            self._enc = self._open_cpu(threads)

    def _open_cpu(self, threads: int) -> ort.InferenceSession:
        path = self._dir / CPU_MODEL
        if not path.exists():
            raise RuntimeError(f'видеокарта не подошла ({self.gpu_error}), а модели для процессора нет')
        return open_cpu_session(path, self._dir / WEIGHTS, threads)

    def _open_gpu(self):
        path = self._dir / GPU_MODEL
        if not available_gpu():
            self.gpu_error = 'no-gpu'
            return
        if not path.exists():
            self.gpu_error = 'no-model'
            return
        if asr_onnx.gpu_lost:
            self.gpu_error = 'lost'
            return
        try:
            with GPU_LOCK:
                self._gpu = open_gpu_session(path, self._dir / WEIGHTS)
                gpu_selfcheck_twice(self._gpu_selfcheck)
            self.gpu = True
        except Exception as e:
            gpu_open_failed(self, e)

    def _gpu_selfcheck(self):
        """
        Видеокарта должна расслышать фразу так же, как процессор. На некоторых
        сочетаниях драйвера и DirectML граф собирается неверно и выдаёт мусор -
        тогда остаёмся на процессоре. Фраза - selfcheck.npy (синтезатор речи
        Windows, tools/export_gpu.py); сравниваем текст, а не числа: модель
        для процессора квантована грубее, и её выход заметно другой.
        """
        a = np.load(self._dir / 'selfcheck.npy').astype(np.float32) / 32768
        feats, flens = self._features([a])
        g, gl = self._encode_gpu(feats, flens)
        if not np.isfinite(g).all():
            raise RuntimeError('видеокарта выдаёт NaN')
        got = [_norm(w.text) for w in self._greedy(g, np.minimum(gl, g.shape[2]))[0].words]
        want = [_norm(w) for w in SELFCHECK_TEXT.split()]
        sim = SequenceMatcher(None, got, want).ratio()
        if sim < 0.7:
            raise RuntimeError(f'видеокарта слышит не то ({sim:.2f}): {" ".join(got)[:80]}')

    def set_threads(self, threads: int):
        """Потоков процессора на кодировщик. Работы столько же, окно готово раньше."""
        if threads != self.threads and self._enc is not None:
            self._enc = self._open_cpu(threads)
        self.threads = threads

    def _features(self, audios: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        lens = np.array([len(a) for a in audios], dtype=np.int64)
        wav = np.zeros((len(audios), int(lens.max())), dtype=np.float32)
        for i, a in enumerate(audios):
            wav[i, :len(a)] = a
        feats, flens = self._pre.run(['features', 'features_lens'], {'waveforms': wav, 'waveforms_lens': lens})
        return feats, flens.astype(np.int64)

    def _encode_cpu(self, feats: np.ndarray, flens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._enc is None:
            self._enc = self._open_cpu(self.threads)
        enc, elens = self._enc.run(['outputs', 'encoded_lengths'], {'audio_signal': feats, 'length': flens})
        return enc, elens.astype(np.int64)

    def _encode_gpu(self, feats: np.ndarray, flens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """По одной фразе, окно всегда GPU_FRAMES: признаки за концом фразы - нули, как у NeMo."""
        outs, lens = [], []
        for i in range(feats.shape[0]):
            n = int(flens[i])
            if n > GPU_FRAMES:
                # длиннее окна видеокарты - процессором (в потоке и в файлах не бывает)
                e, el = self._encode_cpu(feats[i:i + 1, :, :n], flens[i:i + 1])
            else:
                x = np.zeros((1, feats.shape[1], GPU_FRAMES), dtype=np.float32)
                x[0, :, :n] = feats[i, :, :n]
                with gpu_turn():
                    if self._gpu is None:
                        raise GpuReleased()
                    e, el = self._gpu.run(['outputs', 'encoded_lengths'],
                                          {'audio_signal': x, 'length': flens[i:i + 1]})
                if not np.isfinite(e[0, :, :int(el[0])]).all():
                    if gpu_bad_window(self, 'Parakeet'):
                        raise RuntimeError('видеокарта выдаёт NaN')     # recognize() перейдёт на процессор
                    e, el = self._encode_cpu(feats[i:i + 1, :, :n], flens[i:i + 1])
            outs.append(e[0].astype(np.float32))
            lens.append(int(el[0]))
        T = max(o.shape[1] for o in outs)
        enc = np.zeros((len(outs), outs[0].shape[0], T), dtype=np.float32)
        for i, o in enumerate(outs):
            enc[i, :, :o.shape[1]] = o
        return enc, np.array(lens, dtype=np.int64)

    def recognize(self, audios: list[np.ndarray]) -> list[Hypothesis]:
        pad = np.zeros(int(RIGHT_PAD_S * SAMPLE_RATE), dtype=np.float32)
        feats, flens = self._features([np.concatenate([a, pad]) for a in audios])
        enc = None
        if self.gpu:
            try:
                enc, elens = self._encode_gpu(feats, flens)
            except GpuReleased:
                pass
            except Exception as e:
                # видеокарту отобрали (сброс драйвера, игра) - дальше на процессоре
                gpu_failed(self, e)
        if enc is None:
            enc, elens = self._encode_cpu(feats, flens)
        hyps = self._greedy(enc, np.minimum(elens, enc.shape[2]))
        for h, a in zip(hyps, audios):
            end = len(a) / SAMPLE_RATE
            h.words = [w for w in h.words if w.start < end]     # целиком в добавленной тишине - не речь
        return hyps

    def _greedy(self, enc: np.ndarray, lens: np.ndarray) -> list[Hypothesis]:
        """Жадный TDT сразу для всего батча: на шаге один вызов joint на всех."""
        B = enc.shape[0]
        s1 = np.zeros((2, B, 640), np.float32)
        s2 = np.zeros((2, B, 640), np.float32)
        last = np.full(B, self.blank, np.int32)
        t = np.zeros(B, np.int64)
        emitted = np.zeros(B, np.int64)
        tokens: list[list[tuple[int, int, int]]] = [[] for _ in range(B)]     # (id, кадр, длительность)
        active = np.flatnonzero(t < lens)
        while active.size:
            frames = enc[active, :, t[active]][:, :, None]                      # A×1024×1
            out, _, n1, n2 = self._joint.run(None, {
                'encoder_outputs': frames,
                'targets': last[active][:, None],
                'target_length': np.ones(active.size, np.int32),
                'input_states_1': s1[:, active], 'input_states_2': s2[:, active]})
            out = out.reshape(active.size, -1)
            tok = out[:, :self.V].argmax(axis=1)
            dur = np.array(DURATIONS)[out[:, self.V:].argmax(axis=1)]
            emit = tok != self.blank
            for k in np.flatnonzero(emit):
                b = active[k]
                tokens[b].append((int(tok[k]), int(t[b]), int(dur[k])))
                last[b] = tok[k]
                s1[:, b] = n1[:, k]
                s2[:, b] = n2[:, k]
                emitted[b] += 1
            step = dur.copy()
            stuck = (step == 0) & (~emit | (emitted[active] >= MAX_TOKENS_PER_STEP))
            step[stuck] = 1
            moved = step > 0
            t[active[moved]] += step[moved]
            emitted[active[moved]] = 0
            active = active[t[active] < lens[active]]
        return [self._words(tk) for tk in tokens]

    def _words(self, tokens: list[tuple[int, int, int]]) -> Hypothesis:
        words: list[Word] = []
        for tok, frame, dur in tokens:
            piece = self.vocab[tok]
            if piece.startswith('<'):
                continue
            end = (frame + max(1, dur)) * FRAME_S
            if piece.startswith('▁') or not words:
                words.append(Word(piece.lstrip('▁'), frame * FRAME_S, end))
            else:
                words[-1].text += piece
                words[-1].end = end
        out: list[Word] = []
        for w in words:
            if not w.text:
                continue
            if out and w.text not in '—–' and not any(ch.isalnum() for ch in w.text):
                out[-1].text += w.text
                out[-1].end = w.end
            else:
                out.append(w)
        return Hypothesis(out)
