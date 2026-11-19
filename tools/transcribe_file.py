"""
Расшифровка файлов (видео, аудио) в текст: фразы по Silero VAD, каждая
целиком в GigaAM. Нужен ffmpeg в PATH.

  python tools/transcribe_file.py запись.mp4 [ещё.m4a ...]

Рядом с каждым файлом появляется .txt.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import asr_onnx

SR = 16000
BLOCK = 512
VAD_ON, VAD_OFF = 0.5, 0.35
SILENCE_BLOCKS = 20          # 0.64 с тишины - конец фразы
MAX_PHRASE_S = 20


def load(path: Path) -> np.ndarray:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vn', '-ac', '1',
                          '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def phrases(probs: np.ndarray):
    start, speaking, sil = 0, False, 0
    for i, p in enumerate(probs):
        if not speaking:
            if p >= VAD_ON:
                speaking, start, sil = True, max(0, i - 8), 0
            continue
        sil = sil + 1 if p < VAD_OFF else 0
        if sil >= SILENCE_BLOCKS or (i - start) * BLOCK >= MAX_PHRASE_S * SR:
            yield start, i
            speaking = False
    if speaking:
        yield start, len(probs)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    files = [Path(p) for p in sys.argv[1:]]
    for f in files:
        if not f.exists():
            sys.exit(f'Не найден: {f}')
    md = Path(__file__).resolve().parent.parent / 'models' / 'onnx'
    stt = asr_onnx.GigaAM(md)
    vad = asr_onnx.SileroVad(md / 'silero_vad_v6.onnx')
    for f in files:
        t0 = time.time()
        audio = load(f)
        lines = []
        for a, b in phrases(vad.probs(audio)):
            text = stt.recognize([audio[a * BLOCK:b * BLOCK]])[0].text
            if text:
                s = a * BLOCK / SR
                lines.append(f'[{int(s // 60):02d}:{s % 60:04.1f}] {text}')
        out = f.with_suffix('.txt')
        out.write_text('\n'.join(lines), encoding='utf-8')
        dur = len(audio) / SR
        print(f'{f.name}: {dur / 60:.1f} мин за {time.time() - t0:.0f} с -> {out.name}')


if __name__ == '__main__':
    main()
