"""
Сверка asr_onnx.GigaAM (numpy-декодер, батч) с onnx-asr: текст должен
совпасть, плюс скорость и нагрузка на одном потоке.

  build/venv-tools/Scripts/python tools/bench/asr_onnx_check.py build/test-audio/монолог.webm
"""
import os
import subprocess
import sys
import time

import numpy as np
import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import asr_onnx

SR = 16000
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'samples/test_discord_clean.wav')
raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '120', '-t', '120', '-i', path,
                      '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype=np.float32).copy()
chunks = [audio[i:i + 6 * SR] for i in range(0, len(audio) - 6 * SR, 6 * SR)]

m = asr_onnx.GigaAM(os.path.join(ROOT, 'models/onnx'))
m.recognize(chunks[:1])
proc = psutil.Process()
c0 = proc.cpu_times(); t0 = time.perf_counter()
mine = [m.recognize([c])[0] for c in chunks]
wall = time.perf_counter() - t0; c1 = proc.cpu_times()
cpu = (c1.user - c0.user) + (c1.system - c0.system)
print(f'asr_onnx по одному: RTFx {len(chunks) * 6 / wall:.1f}, CPU {100 * cpu / wall:.0f}% ядра')
t0 = time.perf_counter()
batched = []
for i in range(0, len(chunks), 4):
    batched += m.recognize(chunks[i:i + 4])
print(f'asr_onnx батч 4:    RTFx {len(chunks) * 6 / (time.perf_counter() - t0):.1f}')

try:
    import onnx_asr
    import onnxruntime as rt
    so = asr_onnx.session_options(1)
    ref = onnx_asr.load_model('gigaam-v3-e2e-rnnt', quantization='int8', sess_options=so,
                              providers=['CPUExecutionProvider'])
    same = 0
    for c, h, hb in zip(chunks, mine, batched):
        r = ref.recognize(c, sample_rate=SR)
        if r.strip() == h.text and h.text == hb.text:
            same += 1
        else:
            print(f'  ref : {r}\n  mine: {h.text}\n  bat : {hb.text}')
    print(f'совпало с onnx-asr: {same}/{len(chunks)}')
except ImportError:
    pass
print(mine[3].text)
print([(w.text, round(w.start, 2), round(w.end, 2)) for w in mine[3].words[:8]])
