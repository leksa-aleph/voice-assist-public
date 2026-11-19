"""
Векторные иконки: 24x24, линия 2, скруглённые концы.
Рисуются через QSvgRenderer прямо в QPainter - чёткие на любом масштабе.
"""
from __future__ import annotations

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

_P = {
    'power': '<path d="M12 3v9"/><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/>',
    'mic': '<rect x="9" y="2.5" width="6" height="11.5" rx="3"/>'
           '<path d="M18.5 10.5v.5a6.5 6.5 0 0 1-13 0v-.5"/><path d="M12 17.5V21"/>',
    'speaker': '<path d="M11 5 6.5 9H3v6h3.5L11 19z"/><path d="M15.5 9a4.2 4.2 0 0 1 0 6"/>'
               '<path d="M18.5 6a8.5 8.5 0 0 1 0 12"/>',
    'headphones': '<path d="M3.5 15v-3a8.5 8.5 0 0 1 17 0v3"/>'
                  '<rect x="2.5" y="14" width="5" height="7" rx="2"/>'
                  '<rect x="16.5" y="14" width="5" height="7" rx="2"/>',
    'users': '<circle cx="9" cy="8" r="3.6"/><path d="M2.5 20.5v-.6A5.4 5.4 0 0 1 7.9 14.5h2.2a5.4 5.4 0 0 1 5.4 5.4v.6"/>'
             '<path d="M15.6 4.6a3.6 3.6 0 0 1 0 6.8"/><path d="M21.5 20.5v-.6a5.4 5.4 0 0 0-3.4-5"/>',
    'user': '<circle cx="12" cy="8" r="4"/><path d="M4.5 21v-.7A6.3 6.3 0 0 1 10.8 14h2.4a6.3 6.3 0 0 1 6.3 6.3v.7"/>',
    'user-plus': '<circle cx="9.5" cy="8" r="3.6"/><path d="M3 20.5v-.6a5.4 5.4 0 0 1 5.4-5.4h2.2a5.4 5.4 0 0 1 5.4 5.4v.6"/>'
                 '<path d="M19 8v6"/><path d="M16 11h6"/>',
    'captions': '<rect x="2.5" y="4.5" width="19" height="15" rx="3.5"/>'
                '<path d="M6.5 10.5h6"/><path d="M15 10.5h2.5"/><path d="M6.5 14.5h3"/><path d="M12 14.5h5.5"/>',
    'sliders': '<path d="M4 7h9"/><path d="M19 7h1"/><circle cx="16" cy="7" r="2.4"/>'
               '<path d="M4 17h2"/><path d="M12 17h8"/><circle cx="9" cy="17" r="2.4"/>',
    'waves': '<path d="M3 10v4"/><path d="M7 6.5v11"/><path d="M11 3.5v17"/>'
             '<path d="M15 7.5v9"/><path d="M19 5v14"/>',
    'trash': '<path d="M3.5 6.5h17"/><path d="M9 6.5V4.5h6v2"/>'
             '<path d="M18.5 6.5 17.6 19a2 2 0 0 1-2 1.9H8.4a2 2 0 0 1-2-1.9L5.5 6.5"/>',
    'x': '<path d="M6.5 6.5l11 11"/><path d="M17.5 6.5l-11 11"/>',
    'check': '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    'plus': '<path d="M12 5v14"/><path d="M5 12h14"/>',
    'minus': '<path d="M6 12h12"/>',
    'chevron-left': '<path d="M14.5 6 8.5 12l6 6"/>',
    'chevron-right': '<path d="M9.5 6l6 6-6 6"/>',
    'chevron-down': '<path d="M6 9.5l6 6 6-6"/>',
    'clock': '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    'letter': '<path d="M5.5 19 12 5l6.5 14"/><path d="M8 13.8h8"/>',
    'infinity': '<path d="M6.5 15.5C4.6 15.5 3 13.9 3 12s1.6-3.5 3.5-3.5c3.8 0 7.2 7 11 7 1.9 0 3.5-1.6 3.5-3.5s-1.6-3.5-3.5-3.5c-3.8 0-7.2 7-11 7z"/>',
    'zap': '<path d="M13 2.5 4.5 13.5H11l-1 8 8.5-11H12z"/>',
    'gauge': '<path d="M12 14l3.8-3.8"/><path d="M3.6 18.5a9.5 9.5 0 1 1 16.8 0"/>',
    'anchor': '<circle cx="12" cy="5" r="2.3"/><path d="M12 7.3V21"/>'
              '<path d="M5.5 12.5H3a9 9 0 0 0 18 0h-2.5"/>',
    'file': '<path d="M14 2.5H6.5a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V8z"/>'
            '<path d="M14 2.5V8h5.5"/><path d="M8.5 13h7"/><path d="M8.5 17h4.5"/>',
    'external': '<path d="M14.5 3.5h6v6"/><path d="M10 14 20.5 3.5"/>'
                '<path d="M18.5 13.5v5a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-11a2 2 0 0 1 2-2h5"/>',
    'paste': '<rect x="8" y="2.5" width="8" height="4" rx="1.2"/>'
             '<path d="M16 4.5h1.5a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2H8"/>',
    'key': '<circle cx="7.5" cy="15.5" r="4.5"/><path d="M10.7 12.3 20.5 2.5"/>'
           '<path d="M16.5 6.5l3 3"/><path d="M14 9l2.2 2.2"/>',
    'bot': '<rect x="4" y="8" width="16" height="12" rx="3.5"/><path d="M12 4.5V8"/>'
           '<circle cx="12" cy="3.3" r="1.2"/><path d="M9 13v2"/><path d="M15 13v2"/>'
           '<path d="M1.5 13v3"/><path d="M22.5 13v3"/>',
    'expand': '<path d="M8 3.5H5.5a2 2 0 0 0-2 2V8"/><path d="M20.5 8V5.5a2 2 0 0 0-2-2H16"/>'
              '<path d="M3.5 16v2.5a2 2 0 0 0 2 2H8"/><path d="M16 20.5h2.5a2 2 0 0 0 2-2V16"/>',
    'sparkles': '<path d="M11 3.5l1.8 4.9 4.9 1.8-4.9 1.8L11 16.9l-1.8-4.9-4.9-1.8 4.9-1.8z"/>'
                '<path d="M18.5 14.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/>',
    'lines': '<path d="M4 6.5h16"/><path d="M4 12h16"/><path d="M4 17.5h9"/>',
    'leaf': '<path d="M5 19c0-8 5-13 15-14-1 10-6 15-14 15"/><path d="M5 19l7-7"/>',
    'history': '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1"/><path d="M3.5 4.5v4h4"/><path d="M12 7.5V12l3 2"/>',
    'copy': '<rect x="8.5" y="8.5" width="12" height="12" rx="2.5"/><path d="M15.5 8.5V5.5a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"/>',
    'pencil': '<path d="M15.5 4.5l4 4L8 20H4v-4z"/>',
    'folder': '<path d="M3.5 6.5a2 2 0 0 1 2-2h4l2 2.5h7a2 2 0 0 1 2 2v8.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    'lock': '<rect x="4.5" y="10.5" width="15" height="10" rx="2.5"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/>',
    'align-left': '<path d="M4 6.5h16"/><path d="M4 12h10"/><path d="M4 17.5h13"/>',
    'align-center': '<path d="M4 6.5h16"/><path d="M7 12h10"/><path d="M5.5 17.5h13"/>',
    'droplet': '<path d="M12 3s6.5 6.6 6.5 11.5a6.5 6.5 0 0 1-13 0C5.5 9.6 12 3 12 3z"/>',
    'refresh': '<path d="M20.5 12a8.5 8.5 0 1 1-2.5-6"/><path d="M20.5 3.5v5h-5"/>',
    'stop': '<rect x="6.5" y="6.5" width="11" height="11" rx="2.5" fill="currentColor"/>',
    'play': '<path d="M8 5.2v13.6a.8.8 0 0 0 1.2.7l10.4-6.8a.8.8 0 0 0 0-1.4L9.2 4.5A.8.8 0 0 0 8 5.2z" fill="currentColor"/>',
    'record': '<circle cx="12" cy="12" r="6" fill="currentColor"/>',
    'radio': '<circle cx="12" cy="12" r="2"/><path d="M16.2 7.8a6 6 0 0 1 0 8.4"/>'
             '<path d="M7.8 16.2a6 6 0 0 1 0-8.4"/><path d="M19.1 4.9a10 10 0 0 1 0 14.2"/>'
             '<path d="M4.9 19.1a10 10 0 0 1 0-14.2"/>',
    'logout': '<path d="M9 20.5H5.5a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2H9"/><path d="M15.5 16.5 20 12l-4.5-4.5"/>'
              '<path d="M20 12H9"/>',
    'eye': '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/>'
           '<circle cx="12" cy="12" r="2.8"/>',
    'eye-off': '<path d="M10.6 5.6A9.6 9.6 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a16 16 0 0 1-2.6 3.3"/>'
               '<path d="M6.4 6.9C3.9 8.6 2.5 12 2.5 12S6 18.5 12 18.5a9.3 9.3 0 0 0 4.7-1.3"/>'
               '<path d="M3 3l18 18"/><path d="M9.9 10a2.8 2.8 0 0 0 4 4"/>',
    'cpu': '<rect x="5" y="5" width="14" height="14" rx="2.5"/><rect x="9" y="9" width="6" height="6" rx="1"/>'
           '<path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
    'languages': '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/>'
                 '<path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',
    'gpu': '<rect x="2.5" y="6" width="19" height="11.5" rx="2.5"/><circle cx="9" cy="11.75" r="2.6"/>'
           '<circle cx="15.5" cy="11.75" r="2.6"/><path d="M6 17.5v2.5M18 17.5v2.5"/>',
    'activity': '<path d="M3 12h4l3 7.5 4-15 3 7.5h4"/>',
    'monitor': '<rect x="2.5" y="3.5" width="19" height="13" rx="2.5"/><path d="M8 20.5h8"/><path d="M12 16.5v4"/>',
    'move': '<path d="M12 3v18"/><path d="M3 12h18"/><path d="M9 6l3-3 3 3"/><path d="M9 18l3 3 3-3"/>'
            '<path d="M6 9l-3 3 3 3"/><path d="M18 9l3 3-3 3"/>',
    'link': '<path d="M10 13.5a4.5 4.5 0 0 0 6.4.5l3-3a4.5 4.5 0 0 0-6.4-6.4l-1.2 1.2"/>'
            '<path d="M14 10.5a4.5 4.5 0 0 0-6.4-.5l-3 3a4.5 4.5 0 0 0 6.4 6.4l1.2-1.2"/>',
    'alert': '<path d="M12 3.5 2.5 20h19z"/><path d="M12 10v4.5"/><path d="M12 17.3v.2"/>',
    'help': '<circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.3a2.5 2.5 0 1 1 3.3 2.4c-.6.3-.9.8-.9 1.4v.6"/>'
            '<path d="M12 16.6v.2"/>',
    'hash': '<path d="M5 9h15"/><path d="M4 15h15"/><path d="M10 3.5 8 20.5"/><path d="M16 3.5l-2 17"/>',
}

_cache: dict[tuple, QSvgRenderer] = {}


def _svg(name: str, color: str, stroke: float) -> bytes:
    body = _P.get(name, _P['help']).replace('currentColor', color)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
            f'stroke-linejoin="round">{body}</svg>').encode()


def renderer(name: str, color, stroke: float = 2.0) -> QSvgRenderer:
    c = QColor(color)
    hexc = c.name(QColor.NameFormat.HexRgb)
    key = (name, hexc, round(stroke, 2))
    r = _cache.get(key)
    if r is None:
        r = QSvgRenderer(QByteArray(_svg(name, hexc, stroke)))
        _cache[key] = r
    return r


def draw(p: QPainter, name: str, rect: QRectF, color, opacity: float = 1.0,
         stroke: float = 2.0):
    if opacity <= 0.003:
        return
    c = QColor(color)
    a = c.alphaF() * opacity
    if a <= 0.003:
        return
    old = p.opacity()
    p.setOpacity(old * a)
    renderer(name, c, stroke).render(p, QRectF(rect))
    p.setOpacity(old)


def pixmap(name: str, color, size: int, dpr: float = 2.0, stroke: float = 2.0) -> QPixmap:
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw(p, name, QRectF(0, 0, size, size), color, 1.0, stroke)
    p.end()
    return pm


def icon(name: str, color, size: int = 20) -> QIcon:
    return QIcon(pixmap(name, color, size))
