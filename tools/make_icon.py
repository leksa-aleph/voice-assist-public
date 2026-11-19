"""
Иконка exe и установщика из того же логотипа, что рисует приложение:
  venv\\Scripts\\python tools\\make_icon.py  ->  installer\\icon.ico
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

app = QApplication(sys.argv)
from ui.window import paint_logo

out = ROOT / 'installer' / 'icon.ico'
sizes = (16, 20, 24, 32, 40, 48, 64, 128, 256)
frames = []
for size in sizes:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    m = size * 0.04
    paint_logo(p, QRectF(m, m, size - 2 * m, size - 2 * m))
    p.end()
    frames.append(img)

# ICO с несколькими размерами: каждый кадр - PNG внутри (поддерживается с Vista)
import struct
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice

blobs = []
for img in frames:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, 'PNG')
    blobs.append(bytes(ba))
header = struct.pack('<HHH', 0, 1, len(frames))
offset = 6 + 16 * len(frames)
entries = b''
for img, blob in zip(frames, blobs):
    w = img.width() if img.width() < 256 else 0
    entries += struct.pack('<BBBBHHII', w, w, 0, 0, 1, 32, len(blob), offset)
    offset += len(blob)
out.write_bytes(header + entries + b''.join(blobs))
print(out, out.stat().st_size, 'байт')
