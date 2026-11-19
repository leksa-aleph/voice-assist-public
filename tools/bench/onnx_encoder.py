"""GigaAM v3 e2e RNNT: кодировщик в PyTorch vs ONNX fp32 vs ONNX int8, CPU 1/2 потока."""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJ, 'build', 'onnx')  # build/ в .gitignore: файлы по 300-800 МБ
os.makedirs(OUT, exist_ok=True)
orig = torch.load
torch.load = lambda *a, **k: orig(*a, **{**k, 'weights_only': False})
import gigaam
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

model = gigaam.load_model('v3_e2e_rnnt', download_root=os.path.join(PROJ, 'models/gigaam')).eval()
enc_fp32 = os.path.join(OUT, 'v3_e2e_rnnt_encoder.onnx')
if not os.path.exists(enc_fp32):
    t0 = time.time()
    model.to_onnx(OUT)
    print(f'export {time.time() - t0:.0f}s', os.listdir(OUT))
enc_int8 = os.path.join(OUT, 'v3_e2e_rnnt_encoder.int8.onnx')
if not os.path.exists(enc_int8):
    t0 = time.time()
    quantize_dynamic(enc_fp32, enc_int8, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'])
    print(f'quantize {time.time() - t0:.0f}s')
print('sizes MB:', {f: round(os.path.getsize(os.path.join(OUT, f)) / 2**20) for f in os.listdir(OUT)})

audio, sr = sf.read(os.path.join(PROJ, 'samples/test_discord_clean.wav'), dtype='float32')


def feats(sec):
    x = torch.from_numpy(audio[sr * 20: sr * (20 + sec)]).unsqueeze(0)
    ln = torch.tensor([x.shape[1]])
    with torch.inference_mode():
        f, fl = model.preprocessor(x, ln)
    return f, fl


sess_cache = {}


def ort_sess(path, threads):
    key = (path, threads)
    if key not in sess_cache:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_cache[key] = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    return sess_cache[key]


for threads in (1, 2):
    torch.set_num_threads(threads)
    print(f'\n--- threads {threads} ---')
    for sec in (3, 6, 12):
        f, fl = feats(sec)
        res = {}
        with torch.inference_mode():
            model.encoder(f, fl)
            t0 = time.perf_counter()
            for _ in range(3):
                enc_t, _ = model.encoder(f, fl)
            res['torch'] = (time.perf_counter() - t0) / 3
        for name, path in (('onnx fp32', enc_fp32), ('onnx int8', enc_int8)):
            s = ort_sess(path, threads)
            inp = {s.get_inputs()[0].name: f.numpy(), s.get_inputs()[1].name: fl.numpy()}
            out = s.run(None, inp)
            t0 = time.perf_counter()
            for _ in range(3):
                out = s.run(None, inp)
            res[name] = (time.perf_counter() - t0) / 3
            if name == 'onnx int8':
                diff = np.abs(out[0] - enc_t.numpy()).mean() / (np.abs(enc_t.numpy()).mean() + 1e-9)
                res['int8 rel.err'] = diff
        print(f'{sec:2d}s: ' + ' | '.join(
            f'{k} {v * 1000:.0f} ms (x{sec / v:.0f})' if 'err' not in k else f'{k} {v:.3f}'
            for k, v in res.items()))
