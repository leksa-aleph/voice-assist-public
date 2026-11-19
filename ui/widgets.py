"""
Виджеты интерфейса. Всё рисуется вручную через QPainter.

Правило: виджет рисует только внутри своего прямоугольника с запасом,
иначе Qt срезает края по прямой. Без свечений и градиентов - глубина
тонкими границами и оттенками серого.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QFontMetricsF, QGuiApplication, QLinearGradient, QPainter,
    QPainterPath, QPen, QPixmap,
)
from PyQt6.QtWidgets import QHBoxLayout, QLineEdit, QSizePolicy, QVBoxLayout, QWidget

from ui import icons
from ui.anim import Anim, Spring, clock
from ui.theme import (
    ACCENT, BG0, BG1, BG2, BG3, FG, FG_DIM, FG_FAINT, LINE, PRIMARY, RED,
    ease_out, font, initials, lerp, mix, qc, speaker_color, white,
)

AA = QPainter.RenderHint.Antialiasing


# ============================================================
# БАЗА
# ============================================================

class AWidget(QWidget):
    """Виджет с анимированными значениями. Кадры берёт у общих часов."""

    idle_fps = 60.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._anims: list = []
        self._acc = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def anim(self, value: float = 0.0, rate: float = 14.0) -> Anim:
        a = Anim(value, rate)
        self._anims.append(a)
        return a

    def spring(self, value: float = 0.0, stiffness: float = 260.0, damping: float = 22.0) -> Spring:
        s = Spring(value, stiffness, damping)
        self._anims.append(s)
        return s

    def kick(self):
        clock().wake(self)

    def continuous(self) -> bool:
        return False

    def on_frame(self, dt: float) -> bool:
        # свёрнутое окно Qt считает видимым - не рисуем его впустую;
        # MainWindow будит виджеты, когда окно разворачивают
        if not self.isVisible() or self.window().isMinimized():
            return False
        moving = False
        for a in self._anims:
            if a.step(dt):
                moving = True
        cont = self.continuous()
        if moving:
            self._acc = 0.0
            self.update()
        elif cont:
            self._acc += dt
            if self._acc >= 1.0 / self.idle_fps:
                self._acc = 0.0
                self.update()
        return moving or cont

    def showEvent(self, e):
        super().showEvent(e)
        self.kick()

    def event(self, e):
        t = e.type()
        if t == QEvent.Type.HoverEnter:
            self.hover_changed(True)
        elif t == QEvent.Type.HoverLeave:
            self.hover_changed(False)
        return super().event(e)

    def hover_changed(self, inside: bool):
        pass


def elide(text: str, f: QFont, width: float) -> str:
    return QFontMetricsF(f).elidedText(text, Qt.TextElideMode.ElideRight, width)


def paint_window_bg(p: QPainter, r: QRectF):
    """Ровный тёмный фон, чуть светлее сверху."""
    p.fillRect(r, qc(BG0))
    g = QLinearGradient(0, r.top(), 0, r.top() + 220)
    g.setColorAt(0, white(0.022))
    g.setColorAt(1, white(0.0))
    p.fillRect(QRectF(r.left(), r.top(), r.width(), 220), g)


# ============================================================
# КНОПКИ
# ============================================================

class IconButton(AWidget):
    clicked = pyqtSignal()

    def __init__(self, icon_name: str, size: int = 34, icon_size: int = 18,
                 color: str = FG_DIM, hover_color: str = FG, accent: Optional[str] = None,
                 tooltip: str = '', parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.icon_size = icon_size
        self.color = color
        self.hover_color = hover_color
        self.danger = False
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            self.setToolTip(tooltip)
        self.h = self.anim(0, 16)
        self.p = self.anim(0, 26)
        self.on = self.anim(0, 12)

    def set_icon(self, name: str):
        self.icon_name = name
        self.update()

    def set_on(self, on: bool):
        self.on.set(1.0 if on else 0.0)
        self.kick()

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.kick()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.p.set(1.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.p.set(0.0)
            self.kick()
            if self.rect().contains(e.position().toPoint()):
                self.clicked.emit()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == QEvent.Type.EnabledChange:
            self.setCursor(Qt.CursorShape.PointingHandCursor if self.isEnabled()
                           else Qt.CursorShape.ArrowCursor)
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        if not self.isEnabled():
            p.setOpacity(0.35)
        r = QRectF(self.rect())
        c = r.center()
        k = 1.0 - 0.07 * self.p.value
        rad = (r.width() / 2 - 1) * k
        bg = mix(white(0.055 * self.h.value), qc(PRIMARY, 0.20), self.on.value)
        if self.danger:
            bg = qc(RED, 0.14 * self.h.value)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawEllipse(c, rad, rad)
        col = mix(mix(self.color, self.hover_color, self.h.value), PRIMARY, self.on.value)
        if self.danger:
            col = mix(col, RED, self.h.value)
        s = self.icon_size * k
        icons.draw(p, self.icon_name, QRectF(c.x() - s / 2, c.y() - s / 2, s, s), col)
        p.end()


class PillButton(AWidget):
    clicked = pyqtSignal()

    def __init__(self, text: str = '', icon_name: Optional[str] = None,
                 primary: bool = False, accent: Optional[str] = None, height: int = 38, parent=None):
        super().__init__(parent)
        self.text = text
        self.icon_name = icon_name
        self.primary = primary
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(height)
        self.h = self.anim(0, 16)
        self.p = self.anim(0, 26)
        self.enabled_ = self.anim(1, 10)
        self._f = font(13, QFont.Weight.DemiBold)
        self._refit()

    def _refit(self):
        fm = QFontMetricsF(self._f)
        w = fm.horizontalAdvance(self.text) + (26 if self.icon_name else 0) + 36
        if not self.text:
            w = self.height()
        self.setMinimumWidth(int(w))

    def set_text(self, text: str, icon_name: Optional[str] = None):
        self.text = text
        if icon_name is not None:
            self.icon_name = icon_name
        self._refit()
        self.update()

    def setEnabled(self, on: bool):
        super().setEnabled(on)
        self.enabled_.set(1.0 if on else 0.4)
        self.setCursor(Qt.CursorShape.PointingHandCursor if on else Qt.CursorShape.ArrowCursor)
        self.kick()

    def hover_changed(self, inside):
        self.h.set(1.0 if inside and self.isEnabled() else 0.0)
        self.kick()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.p.set(1.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.p.set(0.0)
            self.kick()
            if self.isEnabled() and self.rect().contains(e.position().toPoint()):
                self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setOpacity(self.enabled_.value)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        k = 1.0 - 0.03 * self.p.value
        c = r.center()
        r = QRectF(c.x() - r.width() * k / 2, c.y() - r.height() * k / 2,
                   r.width() * k, r.height() * k)
        rad = r.height() / 2
        if self.primary:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(mix(PRIMARY, '#B9C2FF', self.h.value))
            p.drawRoundedRect(r, rad, rad)
            fg = qc(BG0)
        else:
            p.setPen(QPen(white(0.11 + 0.06 * self.h.value), 1))
            p.setBrush(white(0.02 + 0.04 * self.h.value))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), rad, rad)
            fg = mix(FG_DIM, FG, 0.35 + 0.65 * self.h.value)
        fm = QFontMetricsF(self._f)
        tw = fm.horizontalAdvance(self.text) if self.text else 0
        iw = 17 if self.icon_name else 0
        gap = 8 if (self.text and self.icon_name) else 0
        x = c.x() - (tw + iw + gap) / 2
        if self.icon_name:
            icons.draw(p, self.icon_name, QRectF(x, c.y() - 8.5, 17, 17), fg)
            x += iw + gap
        if self.text:
            p.setFont(self._f)
            p.setPen(fg)
            p.drawText(QPointF(x, c.y() + fm.ascent() / 2 - fm.descent() / 2 + 0.5), self.text)
        p.end()


# ============================================================
# ПЕРЕКЛЮЧАТЕЛЬ
# ============================================================

class Toggle(AWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, on: bool = False, accent: str = PRIMARY, parent=None):
        super().__init__(parent)
        self.setFixedSize(44, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.accent = accent
        self._on = on
        self.k = self.spring(1.0 if on else 0.0, 420, 30)
        self.h = self.anim(0, 16)

    def is_on(self) -> bool:
        return self._on

    def set_on(self, on: bool, emit: bool = False):
        if on == self._on:
            return
        self._on = on
        self.k.set(1.0 if on else 0.0)
        self.kick()
        if emit:
            self.toggled.emit(on)

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.set_on(not self._on, emit=True)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        r = QRectF(self.rect()).adjusted(2, 3, -2, -3)
        t = max(0.0, min(1.0, self.k.value))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(mix(white(0.12 + 0.04 * self.h.value), qc(self.accent), t))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        kr = r.height() / 2 - 3
        x = lerp(r.left() + 3 + kr, r.right() - 3 - kr, max(0.0, min(1.0, self.k.value)))
        p.setBrush(mix(qc('#C9CBD1'), qc(BG0), t))
        p.drawEllipse(QPointF(x, r.center().y()), kr, kr)
        p.end()


# ============================================================
# СЛАЙДЕР
# ============================================================

class Slider(AWidget):
    """Слайдер с шагами; над ручкой подпись значения."""
    changed = pyqtSignal(object)

    def __init__(self, values: Sequence, value, fmt: Callable = str,
                 icon_left: Optional[str] = None, icon_right: Optional[str] = None,
                 accent: Optional[str] = None, show_label: bool = True, parent=None):
        super().__init__(parent)
        self.values = list(values)
        self.fmt = fmt
        self.icon_left = icon_left
        self.icon_right = icon_right
        self.show_label = show_label
        self.setFixedHeight(52 if show_label else 30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._idx = self._nearest(value)
        self.pos_ = self.spring(self._frac(self._idx), 520, 40)
        self.h = self.anim(0, 14)
        self.drag = self.anim(0, 14)
        self._dragging = False

    def _nearest(self, value) -> int:
        try:
            return self.values.index(value)
        except ValueError:
            try:
                return min(range(len(self.values)), key=lambda i: abs(self.values[i] - value))
            except TypeError:
                return 0

    def _frac(self, idx: int) -> float:
        return idx / max(1, len(self.values) - 1)

    def value(self):
        return self.values[self._idx]

    def set_value(self, value):
        self._idx = self._nearest(value)
        self.pos_.set(self._frac(self._idx))
        self.kick()

    def _track(self) -> QRectF:
        left = 32 if self.icon_left else 10
        right = 32 if self.icon_right else 10
        y = self.height() - 17 if self.show_label else self.height() / 2 - 1.5
        return QRectF(left, y, self.width() - left - right, 3)

    def _set_from_x(self, x: float):
        tr = self._track()
        f = max(0.0, min(1.0, (x - tr.left()) / max(1.0, tr.width())))
        idx = int(round(f * (len(self.values) - 1)))
        if idx != self._idx:
            self._idx = idx
            self.changed.emit(self.values[idx])
        self.pos_.set(self._frac(idx))
        self.kick()

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.kick()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.drag.set(1.0)
            self._set_from_x(e.position().x())

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._set_from_x(e.position().x())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.drag.set(0.0)
            self.kick()

    def wheelEvent(self, e):
        self._step(1 if e.angleDelta().y() > 0 else -1)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self._step(1)
        elif e.key() in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self._step(-1)
        else:
            super().keyPressEvent(e)

    def _step(self, d: int):
        idx = max(0, min(len(self.values) - 1, self._idx + d))
        if idx != self._idx:
            self._idx = idx
            self.pos_.set(self._frac(idx))
            self.changed.emit(self.values[idx])
            self.kick()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        tr = self._track()
        cy = tr.center().y()
        act = max(self.h.value, self.drag.value)
        icol = mix(FG_FAINT, FG_DIM, act)
        if self.icon_left:
            icons.draw(p, self.icon_left, QRectF(6, cy - 8, 16, 16), icol)
        if self.icon_right:
            icons.draw(p, self.icon_right, QRectF(self.width() - 25, cy - 9.5, 19, 19), icol)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(white(0.08))
        p.drawRoundedRect(tr, 1.5, 1.5)
        f = max(0.0, min(1.0, self.pos_.value))
        kx = tr.left() + tr.width() * f
        p.setBrush(qc(PRIMARY, 0.85 + 0.15 * act))
        p.drawRoundedRect(QRectF(tr.left(), tr.top(), kx - tr.left(), tr.height()), 1.5, 1.5)

        kr = 6.5 + 1.0 * self.h.value + 1.0 * self.drag.value
        if act > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(PRIMARY, 0.18 * act))
            p.drawEllipse(QPointF(kx, cy), kr + 5, kr + 5)
        p.setPen(QPen(qc(PRIMARY), 2))
        p.setBrush(qc(BG0))
        p.drawEllipse(QPointF(kx, cy), kr, kr)
        if not self.show_label:
            p.end()
            return

        label = self.fmt(self.values[self._idx])
        f12 = font(12.5, QFont.Weight.Medium)
        fm = QFontMetricsF(f12)
        tw = fm.horizontalAdvance(label)
        x = max(0.0, min(self.width() - tw, kx - tw / 2))
        p.setFont(f12)
        p.setPen(mix(FG_DIM, FG, act))
        p.drawText(QPointF(x, cy - 14 - 2 * self.drag.value), label)
        p.end()


# ============================================================
# ВЫБОР ВАРИАНТА
# ============================================================

class Segmented(AWidget):
    """Несколько круглых вариантов. Подложка выбранного плавно переезжает."""
    changed = pyqtSignal(str)

    def __init__(self, options: Sequence[tuple[str, str, str]], current: str,
                 accent: Optional[str] = None, size: int = 60, parent=None):
        super().__init__(parent)
        self.options = list(options)        # (key, icon, caption)
        self.size_ = size
        self.setFixedHeight(size + 40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        keys = [o[0] for o in self.options]
        self._idx = keys.index(current) if current in keys else min(1, len(keys) - 1)
        self.x = self.spring(self._idx, 300, 28)
        self.cap = self.anim(1.0, 10)
        self._cap_text = self.options[self._idx][2]
        self._old_cap = ''
        self.hov = [self.anim(0, 14) for _ in self.options]
        self.setMouseTracking(True)

    def current(self) -> str:
        return self.options[self._idx][0]

    def set_current(self, key: str, emit: bool = False):
        keys = [o[0] for o in self.options]
        if key not in keys:
            return
        i = keys.index(key)
        if i == self._idx:
            return
        self._idx = i
        self.x.set(i)
        self._old_cap = self._cap_text
        self._cap_text = self.options[i][2]
        self.cap.set(0.0, instant=True)
        self.cap.set(1.0)
        self.kick()
        if emit:
            self.changed.emit(key)

    def _centers(self) -> list[float]:
        n = len(self.options)
        step = self.width() / n
        return [step * (i + 0.5) for i in range(n)]

    def _hit(self, x: float) -> int:
        n = len(self.options)
        return max(0, min(n - 1, int(x / (self.width() / n))))

    def mouseMoveEvent(self, e):
        i = self._hit(e.position().x())
        for j, a in enumerate(self.hov):
            a.set(1.0 if j == i else 0.0)
        self.kick()

    def hover_changed(self, inside):
        if not inside:
            for a in self.hov:
                a.set(0.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.set_current(self.options[self._hit(e.position().x())][0], emit=True)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        cs = self._centers()
        r = self.size_ / 2 - 2
        cy = self.size_ / 2 + 2
        n = len(cs)
        fx = max(0.0, min(n - 1.0, self.x.value))
        i0 = int(fx)
        i1 = min(n - 1, i0 + 1)
        sel_x = lerp(cs[i0], cs[i1], fx - i0)

        for i, _o in enumerate(self.options):
            p.setPen(QPen(white(0.07 + 0.07 * self.hov[i].value), 1))
            p.setBrush(white(0.015 + 0.025 * self.hov[i].value))
            p.drawEllipse(QPointF(cs[i], cy), r, r)

        stretch = min(1.0, abs(self.x.vel) * 0.04)
        rect = QRectF(sel_x - r - stretch * 14, cy - r, 2 * r + stretch * 28, 2 * r)
        p.setPen(QPen(qc(PRIMARY, 0.55), 1))
        p.setBrush(qc(PRIMARY, 0.16))
        p.drawRoundedRect(rect, r, r)

        for i, (_key, icon_name, _cap) in enumerate(self.options):
            sel = max(0.0, 1.0 - abs(fx - i))
            col = mix(mix(FG_FAINT, FG_DIM, self.hov[i].value), PRIMARY, sel)
            s = 24
            icons.draw(p, icon_name, QRectF(cs[i] - s / 2, cy - s / 2, s, s), col, stroke=1.8)

        f = font(12.5, QFont.Weight.Normal)
        p.setFont(f)
        ty = self.size_ + 12
        t = ease_out(self.cap.value)
        if t < 1.0 and self._old_cap:
            p.setPen(qc(FG_DIM, 1.0 - t))
            p.drawText(QRectF(0, ty - 5 * t, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self._old_cap)
        p.setPen(qc(FG, t))
        p.drawText(QRectF(0, ty + 5 * (1 - t), self.width(), 20), Qt.AlignmentFlag.AlignCenter, self._cap_text)
        p.end()


# ============================================================
# ТЕКСТ С ПЕРЕХОДОМ
# ============================================================

class FadeLabel(AWidget):
    """Подпись, которая меняет текст с плавным перетеканием."""

    def __init__(self, text: str = '', size: float = 13, color: str = FG_DIM,
                 weight=QFont.Weight.Medium, align=Qt.AlignmentFlag.AlignCenter,
                 display: bool = False, parent=None):
        super().__init__(parent)
        self._text = text
        self._old = ''
        self._f = font(size, weight, display)
        self.color = color
        self._old_color = color
        self.align = align
        self.t = self.anim(1.0, 9)
        fm = QFontMetricsF(self._f)
        self.setFixedHeight(int(fm.height() + 12))
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def text(self) -> str:
        return self._text

    def set_text(self, text: str, color: Optional[str] = None):
        if text == self._text and (color is None or color == self.color):
            return
        self._old = self._text
        self._old_color = self.color
        self._text = text
        if color:
            self.color = color
        self.t.set(0.0, instant=True)
        self.t.set(1.0)
        self.kick()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setFont(self._f)
        t = ease_out(self.t.value)
        r = QRectF(self.rect())
        w = r.width() - 4
        if t < 1.0 and self._old:
            p.setPen(qc(self._old_color, 1 - t))
            p.drawText(r.translated(0, -5 * t), self.align, elide(self._old, self._f, w))
        p.setPen(qc(self.color, t))
        p.drawText(r.translated(0, 5 * (1 - t)), self.align, elide(self._text, self._f, w))
        p.end()


# ============================================================
# ВСПЛЫВАЮЩЕЕ УВЕДОМЛЕНИЕ
# ============================================================

class Toast(AWidget):
    KINDS = {
        'info': ('radio', FG_DIM),
        'ok': ('check', ACCENT),
        'warn': ('alert', FG),
        'error': ('alert', RED),
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.show_ = self.spring(0.0, 300, 28)
        self._text = ''
        self._kind = 'info'
        self._queue: list[tuple[str, str]] = []
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._dismiss)
        self._f = font(12.5, QFont.Weight.Medium)
        self.hide()

    def push(self, text: str, kind: str = 'info'):
        if not text:
            return
        if self.isVisible() and self.show_.target > 0.5:
            if text == self._text:
                self._hide_timer.start(2600)
                return
            self._queue = [(text, kind)]
            self._hide_timer.start(350)
            return
        self._present(text, kind)

    def _present(self, text: str, kind: str):
        self._text = text
        self._kind = kind
        fm = QFontMetricsF(self._f)
        w = int(min(self.parentWidget().width() - 40, fm.horizontalAdvance(text) + 62))
        self.setFixedSize(w, 48)
        self.move((self.parentWidget().width() - w) // 2, 48)
        self.show()
        self.raise_()
        self.show_.set(1.0)
        self.kick()
        self._hide_timer.start(2600)

    def _dismiss(self):
        self.show_.set(0.0)
        self.kick()

    def on_frame(self, dt):
        alive = super().on_frame(dt)
        if self.show_.target == 0.0 and self.show_.value < 0.02:
            self.hide()
            if self._queue:
                text, kind = self._queue.pop(0)
                QTimer.singleShot(60, lambda: self._present(text, kind))
            return False
        return alive

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        v = max(0.0, min(1.0, self.show_.value))
        p.setOpacity(v)
        r = QRectF(self.rect()).adjusted(6, 6, -6, -8)
        r.translate(0, -8 * (1 - v))
        icon_name, color = self.KINDS.get(self._kind, self.KINDS['info'])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc('#000000', 0.30))
        p.drawRoundedRect(r.translated(0, 2), r.height() / 2, r.height() / 2)
        p.setBrush(qc(BG2))
        p.setPen(QPen(white(0.08), 1))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        icons.draw(p, icon_name, QRectF(r.left() + 14, r.center().y() - 8, 16, 16), color)
        p.setFont(self._f)
        p.setPen(qc(FG))
        tr = QRectF(r.left() + 40, r.top(), r.width() - 52, r.height())
        p.drawText(tr, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   elide(self._text, self._f, tr.width()))
        p.end()


# ============================================================
# ВСПЛЫВАЮЩИЙ СПИСОК
# ============================================================

class Popover(QWidget):
    """
    Список выбора поверх окна. Открывается с лёгким подъёмом,
    подсветка строки едет за курсором.
    items: [{'id', 'text', 'sub', 'icon', 'disabled'}]

    Держите ссылку на объект (или передайте parent) - иначе Python
    удалит окно сразу после показа.
    """
    picked = pyqtSignal(object)

    ROW = 44
    PAD = 6
    MARGIN = 16
    MAX_ROWS = 7

    def __init__(self, items: list[dict], selected=None, width: int = 290, parent=None):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMouseTracking(True)
        self.items = items
        self.selected = selected
        rows = max(1, min(self.MAX_ROWS, len(items)))
        self._panel_h = rows * self.ROW + 2 * self.PAD
        self.setFixedSize(width + 2 * self.MARGIN, self._panel_h + 2 * self.MARGIN)
        self._scroll = 0.0
        self._max_scroll = max(0, len(items) * self.ROW - rows * self.ROW)
        self.appear = Anim(0.0, 18)
        self.hl = Spring(-1, 520, 40)
        self._hover = -1
        self._closing = False
        self._anims = [self.appear, self.hl]

    def on_frame(self, dt):
        moving = False
        for a in self._anims:
            if a.step(dt):
                moving = True
        self.update()
        if self._closing and self.appear.value < 0.03:
            self.close()
            return False
        return moving

    def popup_at(self, global_pos: QPoint):
        """global_pos - левый верхний угол самой панели."""
        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        x = global_pos.x() - self.MARGIN
        y = global_pos.y() - self.MARGIN
        x = max(geo.left(), min(x, geo.right() - self.width()))
        y = max(geo.top(), min(y, geo.bottom() - self.height()))
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()
        self.appear.set(1.0)
        if self.selected is not None:
            for i, it in enumerate(self.items):
                if it.get('id') == self.selected:
                    self.hl.set(i, instant=True)
                    self._hover = i
        clock().wake(self)

    def _panel(self) -> QRectF:
        return QRectF(self.MARGIN, self.MARGIN, self.width() - 2 * self.MARGIN, self._panel_h)

    def _row_at(self, pos: QPointF) -> int:
        pr = self._panel()
        if not pr.contains(pos):
            return -1
        i = int((pos.y() - pr.top() - self.PAD + self._scroll) // self.ROW)
        return i if 0 <= i < len(self.items) else -1

    def mouseMoveEvent(self, e):
        i = self._row_at(e.position())
        if i != self._hover:
            self._hover = i
            if i >= 0:
                if self.hl.value < 0:
                    self.hl.set(i, instant=True)
                self.hl.set(i)
            clock().wake(self)

    def wheelEvent(self, e):
        self._scroll = max(0.0, min(self._max_scroll, self._scroll - e.angleDelta().y() / 2))
        self.update()

    def mouseReleaseEvent(self, e):
        i = self._row_at(e.position())
        if i < 0:
            self._start_close()
            return
        it = self.items[i]
        if it.get('disabled'):
            return
        self.selected = it.get('id')
        self.picked.emit(it)
        self._start_close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._start_close()

    def _start_close(self):
        self._closing = True
        self.appear.set(0.0)
        clock().wake(self)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        a = ease_out(self.appear.value)
        p.setOpacity(a)
        pr = self._panel()
        p.translate(0, 6 * (1 - a))

        for i, spread in enumerate((10, 4)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc('#000000', 0.12 + i * 0.10))
            p.drawRoundedRect(pr.adjusted(-spread, -spread + 4, spread, spread + 4), 14 + spread, 14 + spread)
        p.setBrush(qc(BG2))
        p.setPen(QPen(white(0.08), 1))
        p.drawRoundedRect(pr.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)

        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(pr.adjusted(1, 1, -1, -1), 13, 13)
        p.setClipPath(clip)

        if self.hl.value >= 0 and self._hover >= 0:
            hy = pr.top() + self.PAD + self.hl.value * self.ROW - self._scroll
            hr = QRectF(pr.left() + 6, hy + 2, pr.width() - 12, self.ROW - 4)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(white(0.055))
            p.drawRoundedRect(hr, 9, 9)

        tf = font(13, QFont.Weight.Normal)
        sf = font(11, QFont.Weight.Normal)
        for i, it in enumerate(self.items):
            y = pr.top() + self.PAD + i * self.ROW - self._scroll
            if y + self.ROW < pr.top() or y > pr.bottom():
                continue
            sel = it.get('id') == self.selected and it.get('id') is not None
            dis = it.get('disabled', False)
            col = qc(FG_FAINT if dis else FG)
            icon_col = PRIMARY if sel else (FG_FAINT if dis else FG_DIM)
            x = pr.left() + 16
            if it.get('icon'):
                icons.draw(p, it['icon'], QRectF(x, y + self.ROW / 2 - 8.5, 17, 17), icon_col)
                x += 30
            right = pr.right() - (40 if sel else 16)
            p.setFont(tf)
            p.setPen(col)
            sub = it.get('sub')
            if sub:
                p.drawText(QRectF(x, y + 4, right - x, 20),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           elide(it.get('text', ''), tf, right - x))
                p.setFont(sf)
                p.setPen(qc(FG_DIM))
                p.drawText(QRectF(x, y + 22, right - x, 16),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           elide(sub, sf, right - x))
            else:
                p.drawText(QRectF(x, y, right - x, self.ROW),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           elide(it.get('text', ''), tf, right - x))
            if sel:
                icons.draw(p, 'check', QRectF(pr.right() - 32, y + self.ROW / 2 - 8.5, 17, 17), PRIMARY)
        p.restore()
        p.end()


# ============================================================
# АВАТАРЫ
# ============================================================

def paint_avatar(p: QPainter, center: QPointF, r: float, name: str,
                 image: Optional[QPixmap] = None, speaking: float = 0.0,
                 alpha: float = 1.0):
    """
    Нейтральный круг, инициалы в цвете человека. Говорит - вокруг тонкое
    кольцо его цвета. Всё внутри r + 5.
    """
    color = qc(speaker_color(name))
    p.save()
    p.setOpacity(p.opacity() * alpha)
    if speaking > 0.01:
        rr = r + 2.5 + 2.0 * speaking
        p.setPen(QPen(qc(color, 0.9 * speaking), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center, rr, rr)
    if image is not None and not image.isNull():
        path = QPainterPath()
        path.addEllipse(center, r, r)
        p.setClipPath(path)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r), image,
                     QRectF(image.rect()))
        p.setClipping(False)
    else:
        p.setPen(QPen(white(0.07), 1))
        p.setBrush(qc(BG3))
        p.drawEllipse(center, r - 0.5, r - 0.5)
        f = font(r * 0.72, QFont.Weight.DemiBold, display=True)
        p.setFont(f)
        p.setPen(color)
        p.drawText(QRectF(center.x() - r, center.y() - r, 2 * r, 2 * r),
                   Qt.AlignmentFlag.AlignCenter, initials(name))
    p.restore()


class AvatarStrip(AWidget):
    """Ряд участников; кто говорит - с кольцом своего цвета."""

    def __init__(self, radius: float = 17, parent=None):
        super().__init__(parent)
        self.r = radius
        self.setFixedHeight(int(radius * 2 + 36))
        self.members: list[dict] = []      # {'id', 'name', 'image'}
        self._speak: dict[str, Anim] = {}
        self._appear: dict[str, Anim] = {}
        self._tip = -1
        self.setMouseTracking(True)

    def set_members(self, members: list[dict]):
        ids = {m['id'] for m in members}
        for m in members:
            if m['id'] not in self._appear:
                a = Anim(0.0, 9)
                a.set(1.0)
                self._appear[m['id']] = a
                self._anims.append(a)
            if m['id'] not in self._speak:
                s = Anim(0.0, 12)
                self._speak[m['id']] = s
                self._anims.append(s)
        self.members = list(members)
        for k in list(self._appear):
            if k not in ids:
                self._anims.remove(self._appear.pop(k))
                self._anims.remove(self._speak.pop(k))
        self.kick()

    def set_speaking(self, uid: str, on: bool):
        a = self._speak.get(uid)
        if a is not None:
            a.set(1.0 if on else 0.0)
            self.kick()

    def set_image(self, uid: str, pm: QPixmap):
        for m in self.members:
            if m['id'] == uid:
                m['image'] = pm
        self.update()

    def _cy(self) -> float:
        return self.r + 7

    def _positions(self) -> list[float]:
        n = len(self.members)
        if not n:
            return []
        step = self.r * 2 + 14
        total = step * (n - 1)
        x0 = self.width() / 2 - total / 2
        return [x0 + i * step for i in range(n)]

    def mouseMoveEvent(self, e):
        cy = self._cy()
        hit = -1
        for i, x in enumerate(self._positions()):
            d = QPointF(x, cy) - e.position()
            if math.hypot(d.x(), d.y()) < self.r + 4:
                hit = i
        if hit != self._tip:
            self._tip = hit
            self.update()

    def hover_changed(self, inside):
        if not inside:
            self._tip = -1
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        xs = self._positions()
        cy = self._cy()
        many = len(self.members) > 4
        f = font(11, QFont.Weight.Normal)
        for i, (m, x) in enumerate(zip(self.members, xs)):
            sp = self._speak[m['id']].value if m['id'] in self._speak else 0.0
            ap = self._appear[m['id']].value if m['id'] in self._appear else 1.0
            r = self.r * (0.7 + 0.3 * ease_out(ap))
            paint_avatar(p, QPointF(x, cy), r, m['name'], m.get('image'), sp, ap)
            show = max(sp, 1.0 if i == self._tip else 0.0)
            if many and show <= 0.02:
                continue
            p.setFont(f)
            col = mix(FG_DIM, speaker_color(m['name']), sp)
            col.setAlphaF(show if many else 0.6 + 0.4 * show)
            p.setPen(col)
            p.drawText(QRectF(x - self.r - 16, cy + self.r + 7, self.r * 2 + 32, 16),
                       Qt.AlignmentFlag.AlignCenter, elide(m['name'], f, self.r * 2 + 28))
        p.end()


# ============================================================
# ПОЛЕ ВВОДА
# ============================================================

class TextField(QLineEdit):
    """Поле без рамки: снизу тонкая линия, при фокусе светлеет от центра."""

    def __init__(self, text: str = '', placeholder: str = '', size: float = 15,
                 accent: Optional[str] = None, center: bool = False, parent=None):
        super().__init__(text, parent)
        self.setPlaceholderText(placeholder)
        self.setFont(font(size, QFont.Weight.Normal))
        self.setFrame(False)
        self.setFixedHeight(int(size * 2.4))
        if center:
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f'QLineEdit {{ background: transparent; color: {FG}; border: none;'
            f' padding: 0 2px; selection-background-color: #3B4170; selection-color: {FG}; }}')
        self._focus = Anim(0.0, 12)
        self._hov = Anim(0.0, 14)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def on_frame(self, dt):
        a = self._focus.step(dt)
        b = self._hov.step(dt)
        self.update()
        return a or b

    def event(self, e):
        t = e.type()
        if t == QEvent.Type.HoverEnter:
            self._hov.set(1.0)
            clock().wake(self)
        elif t == QEvent.Type.HoverLeave:
            self._hov.set(0.0)
            clock().wake(self)
        return super().event(e)

    def focusInEvent(self, e):
        super().focusInEvent(e)
        self._focus.set(1.0)
        clock().wake(self)

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        self._focus.set(0.0)
        clock().wake(self)

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(AA)
        y = self.height() - 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(white(0.10 + 0.06 * self._hov.value))
        p.drawRect(QRectF(0, y, self.width(), 1))
        f = self._focus.value
        if f > 0.01:
            w = self.width() * ease_out(f)
            p.setBrush(qc(PRIMARY, 0.9))
            p.drawRect(QRectF((self.width() - w) / 2, y - 0.5, w, 1.5))
        p.end()


# ============================================================
# ГРАФИК ЗАДЕРЖКИ
# ============================================================

class Sparkline(AWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: list[float] = []
        self.setFixedHeight(44)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_values(self, values: list[float]):
        self.values = list(values)[-48:]
        self.update()

    def paintEvent(self, _):
        if len(self.values) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(AA)
        r = QRectF(self.rect()).adjusted(4, 6, -6, -4)
        hi = max(max(self.values) * 1.15, 50.0)
        n = len(self.values)
        pts = [QPointF(r.left() + r.width() * i / (n - 1),
                       r.bottom() - r.height() * min(1.0, v / hi)) for i, v in enumerate(self.values)]
        path = QPainterPath(pts[0])
        for a, b in zip(pts, pts[1:]):
            mx = (a.x() + b.x()) / 2
            path.cubicTo(QPointF(mx, a.y()), QPointF(mx, b.y()), b)
        fill = QPainterPath(path)
        fill.lineTo(pts[-1].x(), r.bottom())
        fill.lineTo(pts[0].x(), r.bottom())
        fill.closeSubpath()
        g = QLinearGradient(0, r.top(), 0, r.bottom())
        g.setColorAt(0, white(0.08))
        g.setColorAt(1, white(0.0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(g)
        p.drawPath(fill)
        p.setPen(QPen(white(0.6), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(FG))
        p.drawEllipse(pts[-1], 2.5, 2.5)
        p.end()


# ============================================================
# КАРТОЧКА
# ============================================================

def paint_surface(p: QPainter, r: QRectF, hover: float = 0.0, radius: float = 16,
                  accent: Optional[str] = None, accent_amt: float = 0.0):
    p.setPen(QPen(white(0.06 + 0.03 * hover), 1))
    p.setBrush(mix(qc(BG1), qc(BG2), hover))
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


class Surface(AWidget):
    """Карточка: ровный тон и тонкая граница. Контент - обычный layout."""

    def __init__(self, radius: float = 16, hoverable: bool = False, parent=None):
        super().__init__(parent)
        self.radius = radius
        self.hoverable = hoverable
        self.h = self.anim(0, 14)

    def hover_changed(self, inside):
        if self.hoverable:
            self.h.set(1.0 if inside else 0.0)
            self.kick()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        paint_surface(p, QRectF(self.rect()), self.h.value, self.radius)
        p.end()


class SectionHeader(QWidget):
    """Иконка + короткий заголовок вместо абзацев пояснений."""

    def __init__(self, icon_name: str, text: str, color: str = FG_DIM, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.text = text
        self.color = color
        self.setFixedHeight(24)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        icons.draw(p, self.icon_name, QRectF(1, 4, 15, 15), FG_FAINT)
        p.setFont(font(12, QFont.Weight.Medium))
        p.setPen(qc(self.color))
        p.drawText(QRectF(24, 0, self.width() - 24, 24),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text)
        p.end()


# ============================================================
# КАРТОЧКА СО СТРОКАМИ (как в настройках Windows 11)
# ============================================================
#
# Группа - это карточка, без заголовков. Строка - цветная иконка,
# одно-два слова и элемент справа. Пояснений нет: что делает настройка,
# видно по иконке и по тому, как сразу меняется результат.

class IconTile(QWidget):
    """Иконка в скруглённом квадрате основного цвета - якорь строки."""

    def __init__(self, icon_name: str, color: str = PRIMARY, size: int = 30, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.color = color
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def set_color(self, color: str):
        self.color = color
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        r = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(self.color, 0.15))
        p.drawRoundedRect(r, 9, 9)
        s = r.width() * 0.56
        icons.draw(p, self.icon_name, QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s),
                   qc(self.color), stroke=1.9)
        p.end()


class RowCard(Surface):
    """Карточка, в которой строки разделены тонкими линиями."""

    def __init__(self, parent=None):
        super().__init__(16, parent=parent)
        # по высоте - ровно по строкам: не растягиваем и не сжимаем
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.rows = QVBoxLayout(self)
        self.rows.setContentsMargins(0, 2, 0, 2)
        self.rows.setSpacing(0)

    def add(self, w: QWidget) -> QWidget:
        self.rows.addWidget(w)
        return w

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setPen(QPen(qc(LINE), 1))
        shown = [self.rows.itemAt(i).widget() for i in range(self.rows.count())]
        shown = [w for w in shown if w is not None and w.isVisible()]
        for w in shown[:-1]:
            y = w.geometry().bottom() + 0.5
            p.drawLine(QPointF(58, y), QPointF(self.width() - 14, y))
        p.end()


class Row(QWidget):
    """Иконка, короткое название, справа - переключатель, значение или кнопка."""

    def __init__(self, icon_name: str, title: str, control: Optional[QWidget] = None,
                 color: str = PRIMARY, height: int = 54, tooltip: str = '', parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        if tooltip:
            self.setToolTip(tooltip)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 12, 0)
        lay.setSpacing(14)
        self.icon = IconTile(icon_name, color)
        lay.addWidget(self.icon)
        self.title = FadeLabel(title, size=14, color=FG, weight=QFont.Weight.Medium,
                               align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.title, 1)
        self.control = control
        if control is not None:
            lay.addWidget(control)


class ValueText(FadeLabel):
    """Значение справа в строке - основным цветом, моноширинные цифры."""

    def __init__(self, text: str = '', color: str = PRIMARY, parent=None):
        super().__init__(text, size=14, color=color, weight=QFont.Weight.DemiBold,
                         align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, parent=parent)
        self._f.setFeature(QFont.Tag('tnum'), 1)
        self._fit(text)

    def _fit(self, text: str):
        self.setMinimumWidth(int(max(40.0, QFontMetricsF(self._f).horizontalAdvance(text) + 8)))

    def set_text(self, text: str, color: Optional[str] = None):
        self._fit(text)
        super().set_text(text, color)


class SliderRow(QWidget):
    """Строка с ползунком: иконка, название и значение сверху, ползунок под ними."""
    changed = pyqtSignal(object)

    def __init__(self, icon_name: str, title: str, values: Sequence, value, fmt: Callable = str,
                 tooltip: str = '', parent=None):
        super().__init__(parent)
        self.fmt = fmt
        if tooltip:
            self.setToolTip(tooltip)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 8)
        lay.setSpacing(0)
        self.value_text = ValueText(fmt(value))
        self.head = Row(icon_name, title, self.value_text, height=44)
        lay.addWidget(self.head)
        sl = QHBoxLayout()
        sl.setContentsMargins(52, 0, 6, 0)
        self.slider = Slider(values, value, fmt, show_label=False)
        self.slider.changed.connect(self._changed)
        sl.addWidget(self.slider)
        lay.addLayout(sl)

    def _changed(self, v):
        self.value_text.set_text(self.fmt(v))
        self.changed.emit(v)

    def set_value(self, v):
        self.slider.set_value(v)
        self.value_text.set_text(self.fmt(v))


class Pills(AWidget):
    """Несколько коротких вариантов в ряд; выбранный подсвечен и плавно переезжает."""
    changed = pyqtSignal(object)

    def __init__(self, options: Sequence[tuple], current, cell: int = 38, parent=None):
        super().__init__(parent)
        self.options = list(options)      # (key, текст или None, иконка или None, подсказка)
        self.cell = cell
        self.setFixedSize(cell * len(self.options) + 6, 34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        keys = [o[0] for o in self.options]
        self._idx = keys.index(current) if current in keys else 0
        self.x = self.spring(self._idx, 320, 30)
        self.hov = [self.anim(0, 14) for _ in self.options]

    def current(self):
        return self.options[self._idx][0]

    def set_current(self, key, emit: bool = False):
        keys = [o[0] for o in self.options]
        if key not in keys or keys.index(key) == self._idx:
            return
        self._idx = keys.index(key)
        self.x.set(self._idx)
        self.kick()
        if emit:
            self.changed.emit(key)

    def _hit(self, x: float) -> int:
        return max(0, min(len(self.options) - 1, int((x - 3) / self.cell)))

    def mouseMoveEvent(self, e):
        i = self._hit(e.position().x())
        for j, a in enumerate(self.hov):
            a.set(1.0 if j == i else 0.0)
        tip = self.options[i][3] if len(self.options[i]) > 3 else ''
        self.setToolTip(tip)
        self.kick()

    def hover_changed(self, inside):
        if not inside:
            for a in self.hov:
                a.set(0.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.set_current(self.options[self._hit(e.position().x())][0], emit=True)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        r = QRectF(self.rect())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(white(0.04))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        fx = max(0.0, min(len(self.options) - 1.0, self.x.value))
        sel = QRectF(3 + fx * self.cell, 3, self.cell, r.height() - 6)
        p.setBrush(qc(PRIMARY, 0.20))
        p.drawRoundedRect(sel, sel.height() / 2, sel.height() / 2)
        f = font(13, QFont.Weight.DemiBold)
        f.setFeature(QFont.Tag('tnum'), 1)
        p.setFont(f)
        for i, opt in enumerate(self.options):
            text, icon_name = opt[1], opt[2]
            k = max(0.0, 1.0 - abs(fx - i))
            col = mix(mix(FG_FAINT, FG_DIM, self.hov[i].value), PRIMARY, k)
            cell = QRectF(3 + i * self.cell, 3, self.cell, r.height() - 6)
            if icon_name:
                s = 17
                icons.draw(p, icon_name, QRectF(cell.center().x() - s / 2, cell.center().y() - s / 2, s, s),
                           col, stroke=1.9)
            else:
                p.setPen(col)
                p.drawText(cell, Qt.AlignmentFlag.AlignCenter, str(text))
        p.end()


class Chevron(IconButton):
    """Кнопка-стрелка справа в строке: «открыть»."""

    def __init__(self, icon_name: str = 'chevron-right', tooltip: str = '', parent=None):
        super().__init__(icon_name, 32, 16, tooltip=tooltip, parent=parent)
