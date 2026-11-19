"""
Готовит models/onnx/ для рантайма без PyTorch. Нужен только разработчику,
запускается один раз из окружения с torch + pyannote:

  <окружение с torch>/Scripts/python tools/export_onnx.py

Что делает:
  * WeSpeaker ResNet34 из нашего чекпоинта pyannote -> wespeaker_resnet34.onnx
    (вход - готовые fbank-признаки, их считает numpy в asr_onnx.py);
  * копирует GigaAM v3 e2e RNNT int8 (istupakov/gigaam-v3-onnx, MIT)
    и Silero VAD v6 из build/onnx;
  * кладёт мел-фильтры и окно в fbanks.npz (взяты из onnx-asr, MIT);
  * проверяет, что эмбеддинги совпадают с pyannote: старые эталоны
    голосов должны остаться рабочими.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'models' / 'onnx'
SRC_RNNT = ROOT / 'build' / 'onnx' / 'istupakov-v3-e2e-rnnt'
SRC_VAD = ROOT / 'build' / 'onnx' / 'silero-v6' / 'silero_vad.onnx'
FBANKS = ROOT / 'build' / 'venv-tools' / 'Lib' / 'site-packages' / 'onnx_asr' / 'preprocessors' / 'data' / 'fbanks.npz'


def copy_models():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('v3_e2e_rnnt_encoder.int8.onnx', 'v3_e2e_rnnt_decoder.onnx',
                 'v3_e2e_rnnt_joint.onnx', 'v3_e2e_rnnt_vocab.txt'):
        shutil.copy2(SRC_RNNT / name, OUT / name)
    shutil.copy2(SRC_VAD, OUT / 'silero_vad_v6.onnx')
    with np.load(FBANKS) as d:
        np.savez(OUT / 'fbanks.npz', gigaam_v3=d['gigaam_v3'],
                 gigaam_v3_window=d['gigaam_v3_window'], wespeaker=d['wespeaker'])
    export_head()
    print('модели скопированы')


def export_head():
    """Веса LSTM-декодера и joint в npz: их считает numpy, батчем."""
    import onnx
    from onnx import numpy_helper

    def inits(name):
        m = onnx.load(str(SRC_RNNT / name))
        return m, {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    dec, w = inits('v3_e2e_rnnt_decoder.onnx')
    lstm = next(n for n in dec.graph.node if n.op_type == 'LSTM')
    W, R, B = (w[lstm.input[i]] for i in (1, 2, 3))
    H = W.shape[1] // 4
    jm, j = inits('v3_e2e_rnnt_joint.onnx')
    mats = [n for n in jm.graph.node if n.op_type == 'MatMul']
    enc_w, pred_w, out_w = (next(j[i] for i in n.input if i in j) for n in mats)
    assert enc_w.shape == (768, 320) and pred_w.shape == (320, 320) and out_w.shape[0] == 320
    np.savez(OUT / 'rnnt_head.npz', embed=w['embed.weight'],
             lstm_w=W[0], lstm_r=R[0], lstm_b=B[0, :4 * H] + B[0, 4 * H:],
             enc_w=enc_w, enc_b=j['enc.bias'], pred_w=pred_w, pred_b=j['pred.bias'],
             out_w=out_w, out_b=j['joint_net.1.bias'])


def export_wespeaker():
    import torch
    import torchaudio
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    if not hasattr(torchaudio, 'set_audio_backend'):
        torchaudio.set_audio_backend = lambda *_a, **_k: None
    orig = torch.load
    torch.load = lambda *a, **k: orig(*a, **{**k, 'weights_only': False})
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(str(ROOT / 'models' / 'pyannote_wespeaker-voxceleb-resnet34-LM.bin')).eval()

    class Net(torch.nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet

        def forward(self, feats):
            return self.resnet(feats)[1]

    net = Net(model.resnet).eval()
    dummy = torch.randn(1, 200, 80)
    path = OUT / 'wespeaker_resnet34.onnx'
    torch.onnx.export(net, (dummy,), str(path), input_names=['feats'], output_names=['embs'],
                      dynamic_axes={'feats': {0: 'batch', 1: 'time'}, 'embs': {0: 'batch'}},
                      opset_version=17, dynamo=False)
    print(f'экспорт: {path} ({path.stat().st_size / 1e6:.1f} МБ)')

    # сверка: pyannote целиком против numpy-fbank + ONNX
    sys.path.insert(0, str(ROOT))
    import asr_onnx
    import soundfile as sf
    spk = asr_onnx.SpeakerEmbedder(OUT)
    inf = Inference(model, window='whole')
    wavs = sorted((ROOT / 'voices' / 'wav').rglob('*.wav'))[:6] or [ROOT / 'samples' / 'test_discord_clean.wav']
    sims = []
    for w in wavs:
        audio, sr = sf.read(w, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[:sr * 8]
        ref = np.asarray(inf({'waveform': torch.from_numpy(audio)[None], 'sample_rate': sr})).squeeze()
        ref /= np.linalg.norm(ref)
        got = spk.embed(audio)
        sims.append(float(ref @ got))
    print('косинус pyannote vs onnx:', ' '.join(f'{s:.4f}' for s in sims))


def export_fp16():
    """
    Кодировщик для видеокарты: fp32 из build/onnx -> fp16 (в два раза меньше,
    DirectML считает его быстрее). Нужны onnx и onnxconverter-common:

      build/venv-tools/Scripts/python tools/export_onnx.py fp16

    Конвертер оставляет у узлов Cast to=float, хотя выход уже fp16, -
    такую модель ORT не загружает. Чиним эти Cast.
    """
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16
    m = onnx.load(str(ROOT / 'build' / 'onnx' / 'v3_e2e_rnnt_encoder.onnx'))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(m16.graph.value_info) + list(m16.graph.output)}
    for n in m16.graph.node:
        if n.op_type == 'Cast':
            to = next(a for a in n.attribute if a.name == 'to')
            if to.i == TensorProto.FLOAT and types.get(n.output[0]) == TensorProto.FLOAT16:
                to.i = TensorProto.FLOAT16
    onnx.checker.check_model(m16)
    onnx.save(m16, str(OUT / 'v3_e2e_rnnt_encoder.fp16.onnx'))
    print('fp16 готов')


if __name__ == '__main__':
    if sys.argv[1:] == ['fp16']:
        export_fp16()
        sys.exit()
    copy_models()
    export_wespeaker()
