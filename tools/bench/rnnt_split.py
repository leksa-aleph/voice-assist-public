"""Сколько времени GigaAM RNNT тратит на кодировщик и сколько на пошаговый декодер."""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
threads = int(sys.argv[1]) if len(sys.argv) > 1 else 2
torch.set_num_threads(threads)
orig = torch.load
torch.load = lambda *a, **k: orig(*a, **{**k, 'weights_only': False})
import gigaam

model = gigaam.load_model('v3_e2e_rnnt', download_root=os.path.join(PROJ, 'models/gigaam')).eval()
audio, sr = sf.read(os.path.join(PROJ, 'samples/test_discord_clean.wav'), dtype='float32')
print('threads', torch.get_num_threads())
with torch.inference_mode():
    for sec in (1, 3, 6, 12):
        x = torch.from_numpy(audio[sr * 20: sr * (20 + sec)]).unsqueeze(0)
        ln = torch.tensor([x.shape[1]])
        model.forward(x, ln)  # прогрев формы
        reps = 3
        te = td = 0.0
        for _ in range(reps):
            t0 = time.perf_counter()
            enc, enc_len = model.forward(x, ln)
            t1 = time.perf_counter()
            out = model._decode(enc, enc_len, ln, False)
            t2 = time.perf_counter()
            te += t1 - t0
            td += t2 - t1
        te /= reps
        td /= reps
        print(f'{sec:2d}s audio: encoder {te * 1000:6.0f} ms | rnnt decode {td * 1000:6.0f} ms '
              f'| decode share {100 * td / (te + td):3.0f}% | RTFx {sec / (te + td):5.1f} | {out[0][0][:50]}')
