"""Расхождение текста: PyTorch fp32 vs ONNX int8 кодировщик (декодер тот же)."""
import os
import time

import numpy as np
import soundfile as sf
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJ, 'build', 'onnx')  # build/ в .gitignore: файлы по 300-800 МБ
torch.set_num_threads(2)
orig = torch.load
torch.load = lambda *a, **k: orig(*a, **{**k, 'weights_only': False})
import gigaam
import onnxruntime as ort

model = gigaam.load_model('v3_e2e_rnnt', download_root=os.path.join(PROJ, 'models/gigaam')).eval()
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.add_session_config_entry('session.intra_op.allow_spinning', '0')
sess = ort.InferenceSession(os.path.join(OUT, 'v3_e2e_rnnt_encoder.int8.onnx'), so,
                            providers=['CPUExecutionProvider'])


def edit(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def norm(t):
    return [w.strip('.,!?;:—–-«»"\'()…').lower() for w in t.split() if w.strip('.,!?;:—–-«»"\'()…')]


errs = words = 0
for name in ('samples/test_discord_clean.wav', 'samples/test_discord_7min.wav'):
    audio, sr = sf.read(os.path.join(PROJ, name), dtype='float32')
    for start in range(0, min(len(audio) // sr - 8, 240), 8):
        x = torch.from_numpy(audio[sr * start: sr * (start + 8)]).unsqueeze(0)
        ln = torch.tensor([x.shape[1]])
        with torch.inference_mode():
            f, fl = model.preprocessor(x, ln)
            enc, el = model.encoder(f, fl)
            ref = model._decode(enc, el, ln, False)[0][0]
            o = sess.run(None, {sess.get_inputs()[0].name: f.numpy(), sess.get_inputs()[1].name: fl.numpy()})
            hyp = model._decode(torch.from_numpy(o[0]), torch.from_numpy(o[1]), ln, False)[0][0]
        r, h = norm(ref), norm(hyp)
        if not r:
            continue
        e = edit(r, h)
        errs += e
        words += len(r)
        if e:
            print(f'{name[8:20]} {start:3d}s  fp32: {ref}\n{"":17s}int8: {hyp}')
print(f'\nрасхождение int8 vs fp32: {100 * errs / max(1, words):.1f}% слов ({errs}/{words})')
