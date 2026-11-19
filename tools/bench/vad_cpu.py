"""Сколько процессора ест Silero VAD в реальном темпе: потоки torch по умолчанию vs 1."""
import os
import sys
import time

import numpy as np
import psutil
import soundfile as sf
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sources = int(sys.argv[2]) if len(sys.argv) > 2 else 2
if threads:
    torch.set_num_threads(threads)
print('torch threads:', torch.get_num_threads(), '| sources:', sources, '| cores:', psutil.cpu_count())

audio, sr = sf.read(os.path.join(PROJ, 'samples/test_discord_clean.wav'), dtype='float32')
models = [torch.jit.load(os.path.join(PROJ, 'models/silero_vad.jit')).eval() for _ in range(sources)]
proc = psutil.Process()
blocks = [audio[i:i + 512] for i in range(0, len(audio) - 512, 512)]

# прогрев
for b in blocks[:50]:
    for m in models:
        m(torch.from_numpy(b), sr)

SECONDS = 10
n = int(SECONDS * sr / 512)
proc.cpu_percent(None)
t0 = time.perf_counter()
cpu0 = proc.cpu_times()
call_ms = []
for i in range(n):
    b = blocks[i % len(blocks)]
    for m in models:
        c0 = time.perf_counter()
        with torch.inference_mode():
            m(torch.from_numpy(b), sr).item()
        call_ms.append((time.perf_counter() - c0) * 1000)
    target = t0 + (i + 1) * 512 / sr
    d = target - time.perf_counter()
    if d > 0:
        time.sleep(d)
wall = time.perf_counter() - t0
cpu1 = proc.cpu_times()
cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
print(f'wall {wall:.1f}s | cpu time {cpu:.2f}s | avg load {100 * cpu / wall:.0f}% of one core '
      f'| per call {np.median(call_ms):.2f} ms median, p95 {np.percentile(call_ms, 95):.2f} ms')
