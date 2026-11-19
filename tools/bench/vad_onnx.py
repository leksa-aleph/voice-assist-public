"""
Silero VAD v6 в ONNX: нагрузка в реальном темпе, один поток без холостого
ожидания. Источники считаются одним батчем: состояние каждого - отдельная
строка в state, вызов один на блок 32 мс.

  build/venv-tools/Scripts/python tools/bench/vad_onnx.py [источников]
"""
import os
import sys
import time

import numpy as np
import onnxruntime as rt
import psutil
import soundfile as sf

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sources = int(sys.argv[1]) if len(sys.argv) > 1 else 2
so = rt.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.add_session_config_entry('session.intra_op.allow_spinning', '0')
sess = rt.InferenceSession(os.path.join(PROJ, 'build/onnx/silero-v6/silero_vad.onnx'), so,
                           providers=['CPUExecutionProvider'])
audio, sr = sf.read(os.path.join(PROJ, 'samples/test_discord_clean.wav'), dtype='float32')
CTX = 64                     # v5/v6 ждут 64 отсчёта прошлого блока перед новым
state = np.zeros((2, sources, 128), np.float32)
context = np.zeros((sources, CTX), np.float32)
sr_t = np.array(sr, dtype=np.int64)
proc = psutil.Process()
SECONDS = 10
n = int(SECONDS * sr / 512)
probs = []
t0 = time.perf_counter()
c0 = proc.cpu_times()
for i in range(n):
    blk = np.stack([audio[(i * 512 + k * 7777) % (len(audio) - 512):][:512] for k in range(sources)])
    x = np.concatenate([context, blk], axis=1)
    out, state = sess.run(None, {'input': x, 'state': state, 'sr': sr_t})
    context = blk[:, -CTX:]
    probs.append(out[:, 0])
    d = t0 + (i + 1) * 512 / sr - time.perf_counter()
    if d > 0:
        time.sleep(d)
wall = time.perf_counter() - t0
c1 = proc.cpu_times()
cpu = (c1.user - c0.user) + (c1.system - c0.system)
p = np.array(probs)
print(f'{sources} источн.: CPU {100 * cpu / wall:.1f}% одного ядра | речь в {100 * (p[:, 0] > 0.5).mean():.0f}% блоков')
