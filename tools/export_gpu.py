"""
Кодировщик для видеокарты, вдвое меньше fp16: веса матриц хранятся в int8
(своя шкала на каждый выходной канал), считается всё в fp16.

  build/venv-tools/Scripts/python tools/export_gpu.py parakeet
  build/venv-tools/Scripts/python tools/export_gpu.py gigaam

Граф - обычная fp16-модель, только большие веса в нём помечены внешними.
Их int8 и шкалы лежат рядом в .npz; при загрузке они разворачиваются в
fp16 и подставляются через add_external_initializers. DirectML получает
ровно ту же fp16-модель, что и раньше. Узлы DequantizeLinear в графе он
исполнял с ошибкой (device removed), поэтому разворачиваем сами.
Ошибка такого округления весов - доли процента: в отличие от int8 для
процессора, активации не квантуются.

Where с вещественным выходом (маска пустого хвоста окна во внимании и
свёртках) заменяется арифметикой: a*m + b*(1-m). DirectML в этом графе
считает Where неверно, хотя каждый его вход совпадает с процессором (одна
такая операция отдельно - верно). Из-за этого окно, дополненное тишиной
до 10.24 с, давало GigaAM на FLEURS 4.1-5.7% ошибок вместо 2.9%.

Parakeet: fp32 из istupakov/parakeet-tdt-0.6b-v3-onnx (encoder-model.onnx
+ .data, 2.4 ГБ) -> models/parakeet/encoder-model.gpu.onnx + .npz (~0.6 ГБ),
и фраза для самопроверки видеокарты selfcheck.npy - синтезатор речи Windows
(голос Zira), 16 кГц int16.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MIN_ELEMS = 1 << 14         # мелкие тензоры (нормировки, смещения) оставляем в fp16


def tensor_types(g) -> dict:
    """Тип каждого тензора графа: по входам, весам и типам операций (без shape_inference)."""
    from onnx import TensorProto
    dt = {i.name: i.type.tensor_type.elem_type for i in g.input}
    dt.update({t.name: t.data_type for t in g.initializer})
    for n in g.node:
        if n.op_type in ('Shape', 'Size', 'NonZero'):
            t = TensorProto.INT64
        elif n.op_type == 'Cast':
            t = next(a.i for a in n.attribute if a.name == 'to')
        elif n.op_type == 'Constant':
            t = n.attribute[0].t.data_type if n.attribute[0].HasField('t') else TensorProto.FLOAT
        elif n.op_type in ('Equal', 'Less', 'Greater', 'Not', 'And', 'Or', 'Xor', 'LessOrEqual', 'GreaterOrEqual'):
            t = TensorProto.BOOL
        elif n.op_type == 'Where':
            t = dt.get(n.input[1], TensorProto.FLOAT)
        elif n.op_type in ('ConstantOfShape', 'Range'):
            t = n.attribute[0].t.data_type if n.attribute else dt.get(n.input[0], TensorProto.FLOAT)
        else:
            t = next((dt[i] for i in n.input if i in dt), TensorProto.FLOAT)
        for o in n.output:
            dt.setdefault(o, t)
    return dt


def no_where(src: Path) -> Path:
    """Where(c, a, b) с вещественным выходом -> a*m + b*(1-m), m = Cast(c). Пишет рядом *.nowhere.onnx."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(str(src))
    g = m.graph
    dt = tensor_types(g)
    g.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), 'nowhere_one'))
    nodes, fixed = [], 0
    for n in g.node:
        if n.op_type == 'Where' and dt.get(n.output[0]) == TensorProto.FLOAT:
            c, a, b = n.input
            p = (n.name or n.output[0]) + '_nw'
            nodes += [
                helper.make_node('Cast', [c], [p + '_m'], to=TensorProto.FLOAT, name=p + '_cast'),
                helper.make_node('Sub', ['nowhere_one', p + '_m'], [p + '_im'], name=p + '_sub'),
                helper.make_node('Mul', [a, p + '_m'], [p + '_am'], name=p + '_mul_a'),
                helper.make_node('Mul', [b, p + '_im'], [p + '_bm'], name=p + '_mul_b'),
                helper.make_node('Add', [p + '_am', p + '_bm'], [n.output[0]], name=p + '_add'),
            ]
            fixed += 1
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    dst = src.with_name(src.stem + '.nowhere.onnx')
    big = sum(t.ByteSize() for t in g.initializer) > 1.8e9
    onnx.save(m, str(dst), save_as_external_data=big, location=dst.name + '.data' if big else None)
    print(f'Where -> арифметика: {fixed}')
    return dst


def fp16(src: Path):
    """fp32 -> fp16 с починкой Cast (см. export_onnx.export_fp16)."""
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16
    # типы промежуточных тензоров нужны конвертеру; у модели больше 2 ГБ
    # shape_inference работает только через файлы
    tmp = src.with_name(src.stem + '.shapes.onnx')
    onnx.shape_inference.infer_shapes_path(str(src), str(tmp))
    m = onnx.load(str(tmp))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
    tmp.unlink()
    for f in tmp.parent.glob(tmp.name + '*'):
        f.unlink()
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(m16.graph.value_info) + list(m16.graph.output)}
    for n in m16.graph.node:
        if n.op_type == 'Cast':
            to = next(a for a in n.attribute if a.name == 'to')
            if to.i == TensorProto.FLOAT and types.get(n.output[0]) == TensorProto.FLOAT16:
                to.i = TensorProto.FLOAT16
    return m16


def mark_external(t, ext_name: str, nbytes: int):
    """Тензор в графе пустой, его значение подставит загрузчик (add_external_initializers)."""
    from onnx import TensorProto
    t.ClearField('raw_data')
    t.data_location = TensorProto.EXTERNAL
    del t.external_data[:]
    for key, val in (('location', ext_name), ('offset', '0'), ('length', str(nbytes))):
        e = t.external_data.add()
        e.key, e.value = key, val


def weight_int8(m, npz: Path, ext_name: str, given: dict = None, raw: list = ()):
    """
    Большие веса MatMul/Conv/Gemm -> int8 + шкала по выходному каналу в npz.
    В модели они остаются fp16, но помечены как внешние данные: файла
    ext_name нет, при загрузке их подставляет asr_onnx.open_gpu_session.
    """
    from onnx import TensorProto, numpy_helper
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    users: dict[str, list] = {}
    for n in g.node:
        for k, i in enumerate(n.input):
            if i in inits:
                users.setdefault(i, []).append((n, k))
    arrays, names = {}, []
    saved = 0
    for name, uses in users.items():
        t = inits[name]
        if t.data_type != TensorProto.FLOAT16 or np.prod(t.dims) < MIN_ELEMS or len(t.dims) < 2:
            continue
        axes = set()
        for n, k in uses:
            if n.op_type == 'MatMul' and k == 1:
                axes.add(len(t.dims) - 1)
            elif n.op_type == 'Conv' and k == 1:
                axes.add(0)
            elif n.op_type == 'Gemm' and k == 1:
                tb = next((a.i for a in n.attribute if a.name == 'transB'), 0)
                axes.add(0 if tb else 1)
            else:
                axes.add(None)
        if len(axes) != 1 or None in axes:
            continue
        axis = axes.pop()
        if given and name in given:
            q, scale = given[name]      # те же int8, что у модели для процессора
        else:
            w = numpy_helper.to_array(t).astype(np.float32)
            red = tuple(i for i in range(w.ndim) if i != axis)
            scale = np.abs(w).max(axis=red, keepdims=True) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        k = len(names)
        names.append(name)
        arrays[f'q{k}'] = q
        arrays[f's{k}'] = scale.astype(np.float32)
        mark_external(t, ext_name, q.size * 2)
        saved += q.size
    np.savez(npz, names=np.array(names), raw=np.array(list(raw), dtype=str), **arrays)
    print(f'в int8: {saved / 1e6:.0f} млн весов')
    return m


def build(src: Path, dst: Path):
    import onnx
    npz = dst.with_suffix('.npz')
    fixed = no_where(src)
    m = weight_int8(fp16(fixed), npz, dst.stem + '.fp16')
    for f in fixed.parent.glob(fixed.name + '*'):
        f.unlink()
    onnx.save(m, str(dst))
    print(f'{dst.name}: {dst.stat().st_size / 1e6:.0f} МБ, {npz.name}: {npz.stat().st_size / 1e6:.0f} МБ')


def build_shared(src: Path, out: Path, stem: str):
    """
    Одна копия весов на процессор и видеокарту: <stem>.npz (int8 со шкалой
    по выходному каналу), <stem>.cpu.onnx и <stem>.gpu.onnx - только графы.
    Процессор: матрицы считаются в int8 (DynamicQuantizeLinear+MatMulInteger,
    те же int8 из npz), свёртки - fp32, развёрнутые из int8 при загрузке
    (в ConvInteger модель теряла точность и замедлялась в 8 раз).
    Видеокарта: всё разворачивается в fp16, как раньше.
    Шкала по каналу, а не одна на матрицу (как в готовом int8 Parakeet):
    дальняя речь Golos 15.2% -> 8.2% ошибок, FLEURS ru 7.1% -> 5.5%.
    """
    import onnx
    from onnx import numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic
    tmp = out / (stem + '.pc8.onnx')
    quantize_dynamic(str(src), str(tmp), weight_type=QuantType.QInt8, per_channel=True,
                     use_external_data_format=True, op_types_to_quantize=['MatMul'])
    cpu = onnx.load(str(tmp))
    inits = {t.name: t for t in cpu.graph.initializer}
    given, raw = {}, []
    for name in list(inits):
        if name.endswith('_quantized') and name[:-10] + '_scale' in inits:
            base = name[:-10]
            q = numpy_helper.to_array(inits[name])
            sc = numpy_helper.to_array(inits[base + '_scale']).astype(np.float32)
            given[base] = (q, sc.reshape([1] * (q.ndim - 1) + [-1]))
            raw.append(base)
    ext = stem + '.weights'
    fixed = no_where(src)
    gpu = weight_int8(fp16(fixed), out / (stem + '.npz'), ext, given, raw)
    for f in fixed.parent.glob(fixed.name + '*'):
        f.unlink()
    onnx.save(gpu, str(out / (stem + '.gpu.onnx')))
    # граф для процессора: всё, что лежит в npz, - пустое
    d = np.load(out / (stem + '.npz'))
    names = set(str(x) for x in d['names'])
    for t in cpu.graph.initializer:
        if t.name.endswith(('_quantized', '_scale')) and t.name.rsplit('_', 1)[0] in raw:
            mark_external(t, ext, int(np.prod(t.dims)) * 4)
        elif t.name in names and t.name not in raw:
            mark_external(t, ext, int(np.prod(t.dims)) * 4)
    onnx.save(cpu, str(out / (stem + '.cpu.onnx')))
    for f in out.glob(tmp.name + '*'):
        f.unlink()
    for f in (stem + '.cpu.onnx', stem + '.gpu.onnx', stem + '.npz'):
        print(f, f'{(out / f).stat().st_size / 1e6:.0f} МБ')


def selfcheck_clip(dst: Path):
    import subprocess
    import tempfile
    import wave
    sys.path.insert(0, str(ROOT))
    from asr_parakeet import SELFCHECK_TEXT
    wav = Path(tempfile.gettempdir()) / 'va-selfcheck.wav'
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$s.SelectVoice('Microsoft Zira Desktop'); "
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{wav}', $f); $s.Speak('{SELFCHECK_TEXT}'); $s.Dispose()")
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=True)
    with wave.open(str(wav)) as w:
        np.save(dst, np.frombuffer(w.readframes(w.getnframes()), np.int16))
    wav.unlink()


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'parakeet'
    if what == 'parakeet':
        build_shared(ROOT / 'build' / 'onnx' / 'parakeet-v3' / 'encoder-model.onnx',
                     ROOT / 'models' / 'parakeet', 'encoder-model')
        selfcheck_clip(ROOT / 'models' / 'parakeet' / 'selfcheck.npy')
    elif what == 'gigaam':
        build_shared(ROOT / 'build' / 'onnx' / 'v3_e2e_rnnt_encoder.onnx',
                     ROOT / 'models' / 'onnx', 'v3_e2e_rnnt_encoder')
