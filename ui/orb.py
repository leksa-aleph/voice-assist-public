"""Главная кнопка и круглые кнопки источников."""
from __future__ import annotations

import math
import time
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen

from engine import EngineState
from ui import icons
from ui.theme import (
    ACCENT, BG1, BG2, BG3, FG, FG_DIM, FG_FAINT, RED,
    ease_out, font, mix, qc, white,
)
from ui.widgets import AA, AWidget, elide

RING_COLORS = {
    EngineState.IDLE: white(0.0),
    EngineState.LOADING: white(0.0),
    EngineState.READY: white(0.16),
    EngineState.RUNNING: qc(ACCENT, 0.85),
    EngineState.ERROR: qc(RED, 0.7),
}


class Orb(AWidget):
    """
    Клик - старт/стоп (или загрузка, если моделей нет).
    Удержание - выгрузка моделей: по кольцу набегает красная дуга,
    отпустил раньше - ничего не случилось.
    Всё рисуется в радиусе R + 26, виджет больше с запасом.
    """
    clicked = pyqtSignal()
    long_pressed = pyqtSignal()

    HOLD_DELAY = 0.30
    HOLD_TIME = 0.90
    R = 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 240)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.state = EngineState.IDLE
        self._from = QColor(RING_COLORS[self.state])
        self._to = QColor(RING_COLORS[self.state])
        self.ct = self.anim(1.0, 6)
        self.h = self.anim(0, 12)
        self.pr = self.anim(0, 24)
        self.progress = self.anim(0, 6)
        self.run = self.anim(0, 5)
        self.loading = self.anim(0, 6)
        self.err = self.anim(0, 6)
        self.level = self.anim(0, 14)
        self.slow = self.anim(0, 3)
        self.shake = self.spring(0, 900, 14)
        self._t = 0.0
        self._ripples: list[float] = []
        self._hold_t0: Optional[float] = None
        self._hold = 0.0
        self._hold_fired = False

    # ---------- API ----------

    def set_state(self, state: EngineState):
        prev = self.state
        self.state = state
        self._from = mix(self._from, self._to, self.ct.value)
        self._to = QColor(RING_COLORS.get(state, white(0.1)))
        self.ct.set(0.0, instant=True)
        self.ct.set(1.0)
        self.run.set(1.0 if state == EngineState.RUNNING else 0.0)
        self.loading.set(1.0 if state == EngineState.LOADING else 0.0)
        self.err.set(1.0 if state == EngineState.ERROR else 0.0)
        if state == EngineState.LOADING and prev != EngineState.LOADING:
            self.progress.set(0.0, instant=True)
        if state == EngineState.ERROR and prev != EngineState.ERROR:
            self.shake.vel = 300.0
        if state == EngineState.RUNNING and prev != state:
            self._ripples.append(self._t)
        # медленное «дыхание» во время записи - 30 к/с хватает
        self.idle_fps = 30.0
        self.kick()

    def set_progress(self, done: int, total: int):
        self.progress.set(max(0.0, min(1.0, done / max(1, total))))
        self.kick()

    def set_levels(self, sys_level: float, mic_level: float):
        v = max(sys_level, mic_level * 0.85)
        if abs(v - self.level.target) > 0.01:
            self.level.set(v)
            self.slow.set(v)
            self.kick()

    # ---------- ввод ----------

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.kick()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._inside(e.position()):
            self.pr.set(1.0)
            self._hold_t0 = time.perf_counter()
            self._hold_fired = False
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self.pr.set(0.0)
        fired = self._hold_fired
        held = self._hold
        self._hold_t0 = None
        self._hold = 0.0
        self.kick()
        if not fired and held < 0.05 and self._inside(e.position()):
            self._ripples.append(self._t)
            self.clicked.emit()

    def _inside(self, pos: QPointF) -> bool:
        d = pos - QPointF(self.width() / 2, self.height() / 2)
        return math.hypot(d.x(), d.y()) < self.R + 10

    # ---------- кадры ----------

    def continuous(self) -> bool:
        # в тишине кнопка неподвижна и не стоит процессору ни кадра;
        # шевелится, только когда слышит звук
        heard = self.state == EngineState.RUNNING and max(self.level.value, self.slow.value) > 0.02
        return (self.state == EngineState.LOADING or heard
                or bool(self._ripples) or self._hold_t0 is not None)

    def on_frame(self, dt):
        self._t += dt
        if self._hold_t0 is not None and not self._hold_fired:
            el = time.perf_counter() - self._hold_t0
            self._hold = max(0.0, min(1.0, (el - self.HOLD_DELAY) / self.HOLD_TIME))
            if self._hold >= 1.0:
                self._hold_fired = True
                self._hold = 0.0
                self.shake.vel = -240.0
                self.long_pressed.emit()
        self._ripples = [r for r in self._ripples if self._t - r < 0.8]
        return super().on_frame(dt)

    # ---------- рисование ----------

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        c = QPointF(self.width() / 2 + self.shake.value * 0.05, self.height() / 2)
        R = self.R * (1 + 0.02 * self.h.value - 0.035 * self.pr.value)
        ring = mix(self._from, self._to, self.ct.value)
        run = self.run.value
        lvl = self.level.value
        ring_r = R + 7

        # дыхание под звук: два тонких кольца, второе запаздывает
        if run > 0.01:
            for val, a in ((self.level.value, 0.30), (self.slow.value, 0.14)):
                rr = ring_r + 3 + 15 * val
                p.setPen(QPen(qc(ACCENT, a * run * (0.35 + 0.65 * val)), 1.2))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, rr, rr)

        for r0 in self._ripples:
            k = (self._t - r0) / 0.8
            rr = ring_r + 18 * ease_out(k)
            p.setPen(QPen(white(0.16 * (1 - k)), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rr, rr)

        # кольцо состояния
        p.setBrush(Qt.BrushStyle.NoBrush)
        if ring.alphaF() > 0.01:
            p.setPen(QPen(ring, 1.5))
            p.drawEllipse(c, ring_r, ring_r)
        lo = self.loading.value
        if lo > 0.01:
            rect = QRectF(c.x() - ring_r, c.y() - ring_r, 2 * ring_r, 2 * ring_r)
            p.setPen(QPen(white(0.06 * lo), 1.5))
            p.drawEllipse(c, ring_r, ring_r)
            span = self.progress.value * 360
            if span > 0.5:
                p.setPen(QPen(white(0.75 * lo), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawArc(rect, 90 * 16, -int(span * 16))
        if self._hold > 0.001:
            rect = QRectF(c.x() - ring_r, c.y() - ring_r, 2 * ring_r, 2 * ring_r)
            p.setPen(QPen(qc(RED, 0.95), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 90 * 16, -int(self._hold * 360 * 16))

        # тело
        p.setPen(QPen(white(0.07 + 0.05 * self.h.value), 1))
        p.setBrush(mix(qc(BG1), qc(BG2), 0.4 + 0.6 * self.h.value))
        p.drawEllipse(c, R, R)

        self._paint_icon(p, c, R, lvl)
        p.end()

    def _paint_icon(self, p: QPainter, c: QPointF, R: float, lvl: float):
        run = self.run.value
        h = self.h.value
        err = self.err.value
        t = self._t

        pw = (1 - run) * (1 - err)
        if pw > 0.01:
            if self.state == EngineState.LOADING:
                ic = white(0.35 + 0.25 * (0.5 + 0.5 * math.sin(t * 3.5)))
            elif self.state == EngineState.IDLE:
                ic = mix(FG_DIM, FG, h)
            else:
                ic = mix(qc(FG, 0.85), qc('#FFFFFF'), h)
            ic.setAlphaF(ic.alphaF() * pw)
            r = R * 0.28
            p.setPen(QPen(ic, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRectF(c.x() - r, c.y() - r + 2, 2 * r, 2 * r)
            path = QPainterPath()
            path.arcMoveTo(rect, 120)
            path.arcTo(rect, 120, 300)
            p.drawPath(path)
            p.drawLine(QPointF(c.x(), c.y() - r * 1.2 + 2), QPointF(c.x(), c.y() - r * 0.2 + 2))

        if err > 0.01:
            s = R * 0.55
            icons.draw(p, 'refresh', QRectF(c.x() - s / 2, c.y() - s / 2, s, s),
                       mix(qc(RED, 0.85), qc(RED), h), err, stroke=1.8)

        if run > 0.01:
            # три тихих полоски; при наведении - «стоп»
            eq = run * (1 - h)
            if eq > 0.01:
                bw, gap = 4.0, 5.0
                x0 = c.x() - (3 * bw + 2 * gap) / 2
                for i in range(3):
                    ph = t * (3.2 + i * 0.7) + i * 1.9
                    k = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(ph)) * min(1.0, lvl * 1.5)
                    if i != 1:
                        k *= 0.7
                    bh = 6 + 22 * k
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(qc(FG, 0.9 * eq))
                    p.drawRoundedRect(QRectF(x0 + i * (bw + gap), c.y() - bh / 2, bw, bh), bw / 2, bw / 2)
            st = run * h
            if st > 0.01:
                s = R * 0.62
                icons.draw(p, 'stop', QRectF(c.x() - s / 2, c.y() - s / 2, s, s), qc(FG), st)


class SourceChip(AWidget):
    """
    Круглая кнопка источника. Пунктир - не выбран или сейчас не виден,
    сплошная граница - выбран, тонкое кольцо акцента - идёт звук.
    """
    clicked = pyqtSignal()

    def __init__(self, icon_name: str, placeholder: str, accent: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setFixedSize(112, 104)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.icon_name = icon_name
        self.placeholder = placeholder
        self.status = 'empty'
        self.name = ''
        self.h = self.anim(0, 14)
        self.pr = self.anim(0, 24)
        self.level = self.anim(0, 14)
        self.ok = self.anim(0, 8)
        self.act = self.anim(0, 6)
        self.bounce = self.spring(0, 520, 16)
        self._t = 0.0

    def nudge(self):
        """Подпрыгнуть: «выбери меня»."""
        self.bounce.vel = -220.0
        self.kick()

    def set_status(self, status: str, name: str = ''):
        self.status = status
        self.name = name
        self.ok.set(1.0 if status == 'ok' else 0.0)
        self.kick()

    def set_active(self, on: bool):
        self.act.set(1.0 if on else 0.0)
        self.kick()

    def set_level(self, v: float):
        if abs(v - self.level.target) > 0.01:
            self.level.set(v)
            self.kick()

    def on_frame(self, dt):
        self._t += dt
        return super().on_frame(dt)

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.kick()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.pr.set(1.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.pr.set(0.0)
            self.kick()
            if self.rect().contains(e.position().toPoint()):
                self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        h = self.h.value
        c = QPointF(self.width() / 2, 40 + self.bounce.value * 0.03)
        r = 26 * (1 + 0.03 * h - 0.05 * self.pr.value)
        ok = self.ok.value
        act = self.act.value
        lvl = self.level.value

        if act > 0.01:
            p.setPen(QPen(qc(ACCENT, act * (0.25 + 0.7 * lvl)), 1.4))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, r + 5, r + 5)

        p.setBrush(mix(qc(BG1), qc(BG2), 0.3 + 0.7 * h))
        if ok > 0.5:
            p.setPen(QPen(white(0.13 + 0.07 * h), 1))
        else:
            pen = QPen(white(0.16 + 0.08 * h), 1.1, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.RoundCap)
            pen.setDashPattern([2.2, 3.0])
            p.setPen(pen)
        p.drawEllipse(c, r, r)

        if self.status == 'empty':
            ic = mix(FG_FAINT, FG_DIM, h)
        elif self.status == 'waiting':
            ic = mix(FG_DIM, FG, h * 0.5)
        else:
            ic = mix(qc(FG, 0.9), qc('#FFFFFF'), h)
        s = 22
        icons.draw(p, self.icon_name, QRectF(c.x() - s / 2, c.y() - s / 2, s, s), ic, stroke=1.8)

        if self.status == 'empty':
            bc = QPointF(c.x() + r * 0.72, c.y() + r * 0.72)
            p.setPen(QPen(white(0.08), 1))
            p.setBrush(mix(qc(BG2), qc(BG3), h))
            p.drawEllipse(bc, 8, 8)
            icons.draw(p, 'plus', QRectF(bc.x() - 5.5, bc.y() - 5.5, 11, 11), mix(FG_DIM, FG, h), stroke=2.2)

        f = font(11.5, QFont.Weight.Normal)
        p.setFont(f)
        label = self.name or self.placeholder
        col = mix(qc(FG_DIM if self.status == 'ok' else FG_FAINT), qc(FG), h * 0.7)
        p.setPen(col)
        p.drawText(QRectF(4, 76, self.width() - 8, 20), Qt.AlignmentFlag.AlignCenter,
                   elide(label, f, self.width() - 10))
        p.end()
