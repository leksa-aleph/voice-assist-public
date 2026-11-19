"""
Палитра, шрифты и мелкие помощники для рисования.

Два цвета, у каждого свой смысл:
  PRIMARY - всё, что можно нажать или выбрать: кнопка действия, выбранный
            вариант, ползунок, переключатель, текущий экран;
  ACCENT  - «живое»: слушаю, уровень звука, бот в канале, кто говорит.
RED - только ошибки и удаление. Основа - графит с холодным оттенком,
цвета говорящих - чтобы различать людей в субтитрах.
"""
from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtGui import QColor, QFont, QFontDatabase

# ---- поверхности ----
BG0 = '#0E0F14'        # фон окна
BG1 = '#16171F'        # карточки
BG2 = '#1D1F29'        # приподнятое / наведение
BG3 = '#272A37'
LINE = '#262935'       # разделители строк в карточке

# ---- текст ----
FG = '#E9EAF0'
FG_DIM = '#9196A8'
FG_FAINT = '#5E6376'

# ---- два цвета ----
PRIMARY = '#8E9CFF'    # барвинок: нажать / выбрать
ACCENT = '#67D3AE'     # мята: живое, слушаю
RED = '#E9737B'

# приглушённые, но различимые - для имён в субтитрах и аватаров
SPEAKER_COLORS = (
    '#8FB4D6', '#D6A1B6', '#9DC8AF', '#B6A9D6', '#D4BE8E',
    '#8DC4C4', '#D3A893', '#ADC593', '#A2ABCB', '#C7A2C7',
)
UNKNOWN_SPEAKER_COLOR = '#8C8F98'


def _pick_family(*names: str) -> str:
    try:
        fams = set(QFontDatabase.families())
    except Exception:
        return names[-1]
    for n in names:
        if n in fams:
            return n
    return names[-1]


FONT_TEXT = 'Segoe UI Variable Text'
FONT_DISPLAY = 'Segoe UI Variable Display'


def init_fonts():
    """Вызывать после создания QApplication: до него база шрифтов пуста."""
    global FONT_TEXT, FONT_DISPLAY
    FONT_TEXT = _pick_family('Segoe UI Variable Text', 'Segoe UI', 'Inter', 'Sans Serif')
    FONT_DISPLAY = _pick_family('Segoe UI Variable Display', 'Segoe UI', 'Inter', 'Sans Serif')
    import overlay
    overlay.FONT_DISPLAY = FONT_DISPLAY


def font(size: float, weight: QFont.Weight = QFont.Weight.Normal,
         display: bool = False, spacing: float = 0.0) -> QFont:
    f = QFont(FONT_DISPLAY if display else FONT_TEXT)
    f.setPixelSize(max(1, int(round(size))))
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def qc(hex_or_color, alpha: Optional[float] = None) -> QColor:
    c = QColor(hex_or_color)
    if alpha is not None:
        c.setAlphaF(max(0.0, min(1.0, alpha)))
    return c


def white(alpha: float) -> QColor:
    return qc('#FFFFFF', alpha)


def mix(a, b, t: float) -> QColor:
    a, b = QColor(a), QColor(b)
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ---- цвета говорящих: одинаковые в окне и в оверлее ----

_speaker_slots: dict[str, int] = {}


def speaker_color(name: Optional[str]) -> str:
    name = (name or '').strip()
    if not name or name == '?':
        return UNKNOWN_SPEAKER_COLOR
    idx = _speaker_slots.get(name)
    if idx is not None:
        return SPEAKER_COLORS[idx]
    pref = sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % len(SPEAKER_COLORS)
    used = set(_speaker_slots.values())
    idx = pref
    for off in range(len(SPEAKER_COLORS)):
        cand = (pref + off) % len(SPEAKER_COLORS)
        if cand not in used:
            idx = cand
            break
    _speaker_slots[name] = idx
    return SPEAKER_COLORS[idx]


def initials(name: str) -> str:
    name = (name or '').strip()
    if not name:
        return '?'
    parts = [p for p in name.replace('_', ' ').replace('.', ' ').split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return name[:2].upper() if len(name) > 1 and not name[1].isspace() else name[0].upper()


def plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


TAU = math.tau
