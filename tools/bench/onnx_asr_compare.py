"""
GigaAM v3 e2e CTC vs RNNT через onnx-asr (int8 и fp32): скорость на одном
потоке без холостого ожидания и расхождение текста между вариантами.

Запуск из окружения build/venv-tools:
  build/venv-tools/Scripts/python tools/bench/onnx_asr_compare.py build/test-audio/монолог.webm
"""
import os
import subprocess
import sys
import time

import numpy as np
import onnx_asr
import onnxruntime as rt
import psutil

SR = 16000
path = sys.argv[1] if len(sys.argv) > 1 else 'samples/test_discord_clean.wav'
start = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 120.0

raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(start), '-t', str(seconds), '-i', path,
                      '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                     capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype=np.float32).copy()
# куски по 6 секунд - типичная длина фразы
chunks = [audio[i:i + 6 * SR] for i in range(0, len(audio) - 6 * SR, 6 * SR)]
print(f'{path}: {len(chunks)} кусков по 6 с')

so = rt.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
so.add_session_config_entry('session.intra_op.allow_spinning', '0')


def norm(t):
    return [w.strip('.,!?;:—–-«»"\'()…').lower() for w in t.split() if w.strip('.,!?;:—–-«»"\'()…')]


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


proc = psutil.Process()
results = {}
for name, q in (('gigaam-v3-e2e-rnnt', None), ('gigaam-v3-e2e-rnnt', 'int8'),
                ('gigaam-v3-e2e-ctc', None), ('gigaam-v3-e2e-ctc', 'int8')):
    try:
        m = onnx_asr.load_model(name, quantization=q, sess_options=so,
                                providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f'{name} {q}: не загрузилась: {e}')
        continue
    m.recognize(chunks[0], sample_rate=SR)
    c0 = proc.cpu_times()
    t0 = time.perf_counter()
    texts = [m.recognize(c, sample_rate=SR) for c in chunks]
    wall = time.perf_counter() - t0
    c1 = proc.cpu_times()
    cpu = (c1.user - c0.user) + (c1.system - c0.system)
    key = f'{name.split("-")[-1]}-{q or "fp32"}'
    results[key] = texts
    print(f'{key:12s} RTFx {len(chunks) * 6 / wall:5.1f}  CPU {100 * cpu / wall:4.0f}% одного ядра')

ref = results.get('rnnt-fp32')
if ref:
    for key, texts in results.items():
        if key == 'rnnt-fp32':
            continue
        e = w = 0
        for a, b in zip(ref, texts):
            ra, hb = norm(a), norm(b)
            e += edit(ra, hb)
            w += len(ra)
        print(f'{key:12s} расхождение с rnnt-fp32: {100 * e / max(1, w):.1f}% слов')
    for i in range(min(4, len(ref))):
        print(f'\n[{i}] rnnt-fp32: {ref[i]}')
        for key, texts in results.items():
            if key != 'rnnt-fp32':
                print(f'    {key:9s}: {texts[i]}')
