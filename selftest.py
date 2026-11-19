"""
Самопроверка без окна: voice-assist.exe --selftest

Ничего не трогает в настройках. Проверяет, что модели грузятся, как быстро
считает процессор, работает ли видеокарта (русская и английская модель),
переводчик и чтение файлов. Отчёт -
%TEMP%\\voice-assist-selftest.txt (у оконной сборки нет консоли).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
import wave
from pathlib import Path


def run() -> int:
    out = Path(tempfile.gettempdir()) / 'voice-assist-selftest.txt'
    lines: list[str] = []
    ok = True

    def say(s: str):
        lines.append(s)
        try:
            print(s)
        except Exception:
            pass

    def step(name, fn):
        nonlocal ok
        t0 = time.perf_counter()
        try:
            res = fn()
            say(f'[ok] {name}: {res} ({time.perf_counter() - t0:.1f} с)')
        except Exception as e:
            ok = False
            say(f'[!!] {name}: {e}')
            lines.append(traceback.format_exc())

    import numpy as np
    import onnxruntime as ort
    import asr_onnx

    say(f'onnxruntime {ort.__version__}, {", ".join(ort.get_available_providers())}')
    say(f'процессор: {os.cpu_count()} потоков')
    md = asr_onnx.model_dir()
    a = np.zeros(4 * asr_onnx.SAMPLE_RATE, dtype=np.float32)
    a[::97] = 0.01

    def speed(m):
        m.recognize([a])
        t0 = time.perf_counter()
        m.recognize([a])
        return f'x{4 / (time.perf_counter() - t0):.0f}'

    step('распознавание на процессоре', lambda: speed(asr_onnx.GigaAM(md)))

    def gpu():
        m = asr_onnx.GigaAM(md, gpu=True)
        if not m.gpu:
            return f'нет ({m.gpu_error})'
        return speed(m)

    step('распознавание на видеокарте', gpu)

    def dml_path():
        import psutil
        paths = {m.path for m in psutil.Process().memory_maps() if m.path.lower().endswith('directml.dll')}
        return ', '.join(sorted(paths)) or 'не загружен'

    step('DirectML', dml_path)

    def english(gpu):
        import asr_parakeet
        if not asr_parakeet.available():
            return 'не установлено'
        m = asr_parakeet.Parakeet(asr_parakeet.model_dir(), gpu=gpu)
        if gpu and not m.gpu:
            return f'нет ({m.gpu_error})'
        clip = np.load(asr_parakeet.model_dir() / 'selfcheck.npy').astype(np.float32) / 32768
        return f'{speed(m)}: {m.recognize([clip])[0].text}'

    step('английская речь на процессоре', lambda: english(False))
    step('английская речь на видеокарте', lambda: english(True))

    def translate():
        import translate_onnx
        if not translate_onnx.available():
            return 'не установлен'
        tr = translate_onnx.Translator(translate_onnx.model_dir())
        tr.translate('Hello.')
        t0 = time.perf_counter()
        text = tr.translate('A recursive function is a function that calls itself.')
        return f'{1000 * (time.perf_counter() - t0):.0f} мс: {text}'

    step('перевод', translate)

    def files():
        import file_transcribe
        p = Path(tempfile.gettempdir()) / 'voice-assist-selftest.wav'
        with wave.open(str(p), 'wb') as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            t = np.arange(44100 * 2) / 44100
            x = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
            w.writeframes(np.repeat(x, 2).tobytes())
        dur, chunks = file_transcribe.decode(p)
        n = sum(len(c) for c in chunks)
        p.unlink()
        return f'{n / 16000:.2f} с звука из wav'

    step('чтение файлов', files)
    out.write_text('\n'.join(lines), encoding='utf-8')
    return 0 if ok else 1
