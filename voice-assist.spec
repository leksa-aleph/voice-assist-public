# -*- mode: python ; coding: utf-8 -*-
# Одна сборка для любых видеокарт: распознавание на ONNX Runtime (процессор,
# по желанию видеокарта через DirectML), без PyTorch/ROCm/CUDA.
# Перед сборкой: npm ci в discord_bridge.
#   venv\Scripts\pyinstaller voice-assist.spec --noconfirm
import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

ROOT = SPECPATH
# кодировщик: графы для процессора и видеокарты, веса общие в .npz (tools/export_gpu.py)
MODELS = ['v3_e2e_rnnt_encoder.cpu.onnx', 'v3_e2e_rnnt_encoder.gpu.onnx', 'v3_e2e_rnnt_encoder.npz',
          'v3_e2e_rnnt_vocab.txt', 'rnnt_head.npz', 'fbanks.npz', 'silero_vad_v6.onnx',
          'wespeaker_resnet34.onnx']
datas = [(os.path.join(ROOT, 'models', 'onnx', m), 'models/onnx') for m in MODELS]
# английская речь: Parakeet (распознавание 25 языков) и Marian en->ru;
# в установщике это отдельный компонент, модели для видеокарты - ещё один
for _sub, _files in (('parakeet', ['nemo128.onnx', 'encoder-model.cpu.onnx', 'encoder-model.gpu.onnx',
                                   'encoder-model.npz', 'selfcheck.npy',
                                   'decoder_joint-model.int8.onnx', 'vocab.txt']),
                     ('translate', ['encoder.int8.onnx', 'decoder.int8.onnx', 'translate.json'])):
    for _f in _files:
        _p = os.path.join(ROOT, 'models', _sub, _f)
        if os.path.exists(_p):
            datas.append((_p, f'models/{_sub}'))
        else:
            print(f'WARNING: нет {_p}')

# Discord-бот: мост на Node + node.exe рядом, чтобы у пользователя
# не было отдельной установки Node.js.
_bridge = os.path.join(ROOT, 'discord_bridge')
datas += [
    (os.path.join(_bridge, 'bridge.mjs'), 'discord_bridge'),
    (os.path.join(_bridge, 'package.json'), 'discord_bridge'),
    (os.path.join(_bridge, 'node_modules'), 'discord_bridge/node_modules'),
]
_node = os.path.join(_bridge, 'node.exe')
if not os.path.exists(_node):
    _node = shutil.which('node')
if _node:
    datas.append((_node, 'discord_bridge'))
    _node_license = Path(_node).with_name('LICENSE.node.txt')
    if _node_license.is_file():
        datas.append((str(_node_license), 'licenses/Node.js'))
else:
    print('WARNING: node.exe не найден - бот в сборке работать не будет')

# PyInstaller обычно исключает метаданные пакетов, поэтому берём тексты
# лицензий и уведомления из установленных wheel явно.
for _name in (
    'av', 'cffi', 'coloredlogs', 'comtypes', 'flatbuffers', 'humanfriendly',
    'numpy', 'onnxruntime-directml', 'packaging', 'proc-tap', 'protobuf', 'psutil',
    'pycaw', 'PyInstaller', 'PyQt6', 'PyQt6-Qt6', 'PyQt6-sip', 'sounddevice',
):
    try:
        _dist = distribution(_name)
    except PackageNotFoundError:
        continue
    for _file in _dist.files or ():
        if _file.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE', 'THIRD')):
            _path = _dist.locate_file(_file)
            if _path.is_file() and '..' not in _file.parts:
                datas.append((str(_path), f'licenses/{_name}/{_file.parent.as_posix()}'))

_python_license = Path(sys.base_prefix) / 'LICENSE.txt'
if _python_license.is_file():
    datas.append((str(_python_license), 'licenses/Python'))

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['pycaw.pycaw', 'comtypes.stream'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # onnxruntime тянет их для своих утилит, приложению они не нужны
    excludes=['torch', 'torchaudio', 'scipy', 'sympy', 'mpmath', 'matplotlib', 'pandas',
              'tkinter', 'IPython', 'PyQt6.QtWebEngineCore', 'PyQt6.QtQml', 'PyQt6.QtQuick',
              'PyQt6.Qt3DCore', 'PyQt6.QtMultimedia', 'PyQt6.QtPdf', 'PyQt6.QtBluetooth'],
    noarchive=False,
    optimize=0,
)
# Windows 11 supplies the UCRT. A private copy found on the build machine's
# PATH can override it and break imports such as QtCore on startup.
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() != 'ucrtbase.dll']
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='voice-assist',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, 'installer', 'icon.ico'),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='voice-assist',
)
