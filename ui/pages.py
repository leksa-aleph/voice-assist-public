"""Экраны приложения."""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRect, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QColor, QDesktopServices, QFont, QFontMetricsF, QGuiApplication, QLinearGradient,
    QPainter, QPainterPath, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout,
    QWidget,
)

import asr_onnx
import audio_source
import overlay
import secret_store
import version
import voices
from engine import EngineState
from ui import icons
from ui.anim import Anim, clock
from ui.orb import Orb, SourceChip
from ui.theme import (
    ACCENT, BG0, BG1, BG2, BG3, FG, FG_DIM, FG_FAINT, PRIMARY, RED,
    ease_out, font, initials, lerp, mix, plural, qc, smooth, speaker_color, white, TAU,
)
from ui.widgets import (
    AA, AWidget, AvatarStrip, Chevron, FadeLabel, IconButton, Pills, PillButton, Row, RowCard,
    Segmented, SliderRow, Sparkline, TextField, Toggle, ValueText, elide, paint_avatar,
)


def _page_layout(w: QWidget, top: int = 6) -> QVBoxLayout:
    lay = QVBoxLayout(w)
    lay.setContentsMargins(20, top, 20, 12)
    lay.setSpacing(12)
    return lay


def _hcenter(widget: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    row.addWidget(widget)
    row.addStretch(1)
    return row


# ============================================================
# ГЛАВНЫЙ
# ============================================================

class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 6, 14, 6)
        lay.setSpacing(0)

        self.members = AvatarStrip(16)
        self.members.setVisible(False)
        lay.addWidget(self.members)
        lay.addStretch(2)

        self.orb = Orb()
        lay.addLayout(_hcenter(self.orb))
        self.status = FadeLabel('', size=15, color=FG_DIM, weight=QFont.Weight.DemiBold,
                                display=True)
        lay.addSpacing(-8)
        lay.addWidget(self.status)
        lay.addStretch(3)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.mic_chip = SourceChip('mic', 'Микрофон')
        self.sys_chip = SourceChip('speaker', 'Приложение')
        self.bot_chip = SourceChip('headphones', 'Discord-бот')
        for c in (self.mic_chip, self.sys_chip, self.bot_chip):
            row.addStretch(1)
            row.addWidget(c)
        row.addStretch(1)
        lay.addLayout(row)


# ============================================================
# ГОЛОСА
# ============================================================

class VoiceTile(AWidget):
    """Голос: аватар, имя, точки - сколько записей. Удаление - в два клика."""
    delete_requested = pyqtSignal(str)

    def __init__(self, name: str, count: int, parent=None):
        super().__init__(parent)
        self.name = name
        self.count = count
        self.setFixedSize(104, 112)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.h = self.anim(0, 14)
        self.arm = self.anim(0, 12)
        self.appear = self.anim(0, 8)
        self.appear.set(1.0)
        self.gone = self.anim(0, 10)
        self._x_hover = False
        self._disarm = QTimer(self)
        self._disarm.setSingleShot(True)
        self._disarm.timeout.connect(lambda: (self.arm.set(0.0), self.kick()))
        self.setMouseTracking(True)

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        if not inside:
            self._x_hover = False
        self.kick()

    def _x_rect(self) -> QRectF:
        return QRectF(self.width() / 2 + 14, 4, 22, 22)

    def mouseMoveEvent(self, e):
        xh = self._x_rect().adjusted(-4, -4, 4, 4).contains(e.position())
        if xh != self._x_hover:
            self._x_hover = xh
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if self.arm.target > 0.5:
            # второй клик - удаляем
            self.gone.set(1.0)
            self.kick()
            QTimer.singleShot(260, lambda: self.delete_requested.emit(self.name))
            return
        if self._x_rect().adjusted(-4, -4, 4, 4).contains(e.position()) and self.h.value > 0.3:
            self.arm.set(1.0)
            self.kick()
            self._disarm.start(2600)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        ap = ease_out(self.appear.value) * (1 - self.gone.value)
        p.setOpacity(ap)
        h, arm = self.h.value, self.arm.value
        c = QPointF(self.width() / 2, 40 - 1.5 * h)
        r = 28 * (0.8 + 0.2 * ap)
        paint_avatar(p, c, r, self.name, speaking=0.6 * h * (1 - arm))
        if arm > 0.01:
            # второй клик удалит: тихий красный, без заливки
            p.setPen(QPen(qc(RED, 0.8 * arm), 1.4))
            p.setBrush(qc(BG2, arm))
            p.drawEllipse(c, r, r)
            s = 22
            icons.draw(p, 'trash', QRectF(c.x() - s / 2, c.y() - s / 2, s, s), qc(RED), arm, stroke=1.8)
        xa = h * (1 - arm)
        if xa > 0.01:
            xr = self._x_rect()
            p.setPen(QPen(white(0.08 * xa), 1))
            p.setBrush(qc(BG3, xa))
            p.drawEllipse(xr.center(), 9.5, 9.5)
            icons.draw(p, 'x', xr.adjusted(6, 6, -6, -6), qc(RED if self._x_hover else FG_DIM), xa, stroke=2.4)

        f = font(12.5, QFont.Weight.Medium)
        p.setFont(f)
        p.setPen(mix(qc(FG_DIM), qc(FG), 0.6 + 0.4 * h))
        p.drawText(QRectF(0, 76, self.width(), 18), Qt.AlignmentFlag.AlignCenter,
                   elide(self.name, f, self.width() - 6))
        # точки: сколько записей в эталоне
        n = max(1, min(5, self.count))
        dx = 6
        x0 = self.width() / 2 - (n - 1) * dx / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(white(0.28))
        for i in range(n):
            p.drawEllipse(QPointF(x0 + i * dx, 100), 1.8, 1.8)
        p.end()


class AddTile(AWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(104, 112)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.h = self.anim(0, 14)
        self.rot = self.spring(0, 200, 16)
        self._t = 0.0

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.rot.set(90.0 if inside else 0.0)
        self.kick()

    def continuous(self):
        return self.h.value > 0.05

    def on_frame(self, dt):
        self._t += dt
        return super().on_frame(dt)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        h = self.h.value
        c = QPointF(self.width() / 2, 40 - 1.5 * h)
        r = 28
        p.setBrush(white(0.02 + 0.03 * h))
        pen = QPen(white(0.16 + 0.12 * h), 1.1, Qt.PenStyle.CustomDashLine, Qt.PenCapStyle.RoundCap)
        pen.setDashPattern([2.4, 3.0])
        pen.setDashOffset(self._t * 3)
        p.setPen(pen)
        p.drawEllipse(c, r - 0.5, r - 0.5)
        p.save()
        p.translate(c)
        p.rotate(self.rot.value)
        icons.draw(p, 'plus', QRectF(-11, -11, 22, 22), mix(FG_DIM, FG, h), stroke=1.8)
        p.restore()
        f = font(12.5, QFont.Weight.Medium)
        p.setFont(f)
        p.setPen(mix(FG_FAINT, FG_DIM, 0.5 + 0.5 * h))
        p.drawText(QRectF(0, 76, self.width(), 18), Qt.AlignmentFlag.AlignCenter, 'Новый')
        p.end()


class RecordRing(AWidget):
    """
    Большая круглая кнопка записи эталона: кольцо отсчёта, живая волна
    внутри, потом спиннер обработки и галочка.
    """
    clicked = pyqtSignal()

    def __init__(self, seconds: int = 30, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.seconds = seconds
        self.mode = 'idle'      # idle | rec | proc | done | error | locked
        self.lock_icon = 'power'
        self.progress = 0.0
        self.remaining = seconds
        self.h = self.anim(0, 14)
        self.pr = self.anim(0, 24)
        self.level = self.anim(0, 20)
        self.done_t = self.anim(0, 4)
        self.shake = self.spring(0, 900, 12)
        self.hist = collections.deque([0.0] * 64, maxlen=64)
        self._t = 0.0

    def set_mode(self, mode: str):
        self.mode = mode
        if mode == 'done':
            self.done_t.set(0.0, instant=True)
            self.done_t.set(1.0)
        if mode == 'error':
            self.shake.vel = 500.0
        if mode == 'idle':
            self.progress = 0.0
            self.remaining = self.seconds
            self.hist.extend([0.0] * 64)
        self.kick()

    def push_level(self, v: float):
        self.level.set(v)
        self.hist.append(v)

    def continuous(self):
        return self.mode in ('rec', 'proc')

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
        t = self._t
        c = QPointF(self.width() / 2 + self.shake.value * 0.05, self.height() / 2)
        h = self.h.value
        R = 58 * (1 + 0.02 * h - 0.035 * self.pr.value)
        mode = self.mode

        # кольцо прогресса - всё внутри R + 16, виджет 200x200
        ring = R + 14
        rect = QRectF(c.x() - ring, c.y() - ring, 2 * ring, 2 * ring)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(qc(RED, 0.45) if mode == 'error' else white(0.06), 2))
        p.drawEllipse(c, ring, ring)
        if mode == 'rec':
            p.setPen(QPen(white(0.8), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 90 * 16, -int(self.progress * 360 * 16))
        elif mode == 'proc':
            head = (t * 240) % 360
            p.setPen(QPen(white(0.6), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, int((90 - head) * 16), -int(60 * 16))
        elif mode == 'done':
            p.setPen(QPen(qc(ACCENT, 0.9), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 90 * 16, -int(360 * ease_out(self.done_t.value) * 16))

        # тонкая круговая волна - уровень за последние секунды
        if mode == 'rec':
            n = len(self.hist)
            for i, v in enumerate(self.hist):
                a = i / n * TAU - t * 0.35
                L = 2 + 9 * v
                base = R + 4
                p.setPen(QPen(white(0.10 + 0.40 * (i / n)), 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                dx, dy = math.sin(a), -math.cos(a)
                p.drawLine(QPointF(c.x() + dx * base, c.y() + dy * base),
                           QPointF(c.x() + dx * (base + L), c.y() + dy * (base + L)))

        p.setPen(QPen(white(0.07 + 0.05 * h), 1))
        p.setBrush(mix(qc(BG1), qc(BG2), 0.4 + 0.6 * h))
        p.drawEllipse(c, R, R)

        if mode == 'rec':
            f = font(36, QFont.Weight.Normal, display=True)
            p.setFont(f)
            p.setPen(qc(FG))
            p.drawText(QRectF(c.x() - R, c.y() - R, 2 * R, 2 * R - 10), Qt.AlignmentFlag.AlignCenter,
                       str(max(0, int(math.ceil(self.remaining)))))
            # точка записи; при наведении - «стоп»
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 4))
            if h < 0.5:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qc(RED, pulse * (1 - h * 2)))
                p.drawEllipse(QPointF(c.x(), c.y() + 26), 3.5, 3.5)
            else:
                icons.draw(p, 'stop', QRectF(c.x() - 7, c.y() + 19, 14, 14), qc(FG), (h - 0.5) * 2)
        elif mode == 'proc':
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(3):
                k = 0.3 + 0.7 * (0.5 + 0.5 * math.sin(t * 5 - i * 0.8))
                p.setBrush(white(0.25 + 0.6 * k))
                p.drawEllipse(QPointF(c.x() - 12 + i * 12, c.y()), 3, 3)
        elif mode == 'done':
            k = ease_out(self.done_t.value)
            path = QPainterPath()
            path.moveTo(QPointF(c.x() - 16, c.y() + 1))
            path.lineTo(QPointF(c.x() - 5, c.y() + 12))
            path.lineTo(QPointF(c.x() + 17, c.y() - 11))
            pen = QPen(qc(ACCENT), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            pen.setDashPattern([path.length() * k / 3, 1000])
            p.setPen(pen)
            p.drawPath(path)
        else:
            name = {'idle': 'mic', 'error': 'mic', 'locked': self.lock_icon}[mode]
            s = 34
            col = mix(FG_DIM, FG, h) if mode != 'locked' else mix(FG_FAINT, FG_DIM, h)
            icons.draw(p, name, QRectF(c.x() - s / 2, c.y() - s / 2, s, s), col, stroke=1.6)
        p.end()


class EnrollSheet(AWidget):
    """
    Лист записи эталона, выезжает снизу поверх экрана голосов.
    Имя, большая кнопка записи, одна строка состояния.
    """
    finished = pyqtSignal()
    _call = pyqtSignal(object)

    SECONDS = 30
    MIN_VOICED_S = 3.0

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self._call.connect(lambda fn: fn())
        self.open_ = self.spring(0.0, 240, 26)
        self.hide()

        self.panel = QWidget(self)
        self._fade = QGraphicsOpacityEffect(self.panel)
        self.panel.setGraphicsEffect(self._fade)
        lay = QVBoxLayout(self.panel)
        lay.setContentsMargins(18, 12, 12, 16)
        lay.setSpacing(6)

        top = QHBoxLayout()
        self.src = FadeLabel('', size=12, color=FG_DIM, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.src, 1)
        self.close_btn = IconButton('x', 32, 16, tooltip='Закрыть')
        self.close_btn.clicked.connect(self.close_sheet)
        top.addWidget(self.close_btn)
        lay.addLayout(top)

        self.name = TextField('', 'Имя', size=20, center=True)
        self.name.textChanged.connect(self._on_name)
        lay.addWidget(self.name)

        row = QHBoxLayout()
        row.addStretch(1)
        self.replace_lbl = FadeLabel('', size=12, color=FG_DIM, align=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.replace_lbl.setFixedWidth(150)
        row.addWidget(self.replace_lbl)
        self.replace = Toggle(False)
        self.replace.toggled.connect(lambda _: self._on_name(self.name.text()))
        row.addWidget(self.replace)
        row.addStretch(1)
        self.replace_row = QWidget()
        self.replace_row.setLayout(row)
        self.replace_row.setFixedHeight(30)
        self.replace_row.setVisible(False)
        lay.addWidget(self.replace_row)

        lay.addSpacing(4)
        self.ring = RecordRing(self.SECONDS)
        self.ring.clicked.connect(self._on_ring)
        lay.addLayout(_hcenter(self.ring))
        self.caption = FadeLabel('', size=13, color=FG_DIM)
        lay.addWidget(self.caption)

        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._t0 = 0.0
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._on_tick)
        self._lvl = 0.0

    # ---------- открытие ----------

    def open_sheet(self):
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.open_.set(1.0)
        self.kick()
        self._refresh_lock()
        self.name.setText('')
        self.replace.set_on(False)
        self.replace_row.setVisible(False)
        self.caption.set_text('')
        QTimer.singleShot(120, self.name.setFocus)

    def close_sheet(self):
        self._abort()
        self.open_.set(0.0)
        self.kick()

    def on_frame(self, dt):
        alive = super().on_frame(dt)
        self._layout_panel()
        if self.open_.target == 0.0 and self.open_.value < 0.01:
            self.hide()
            return False
        return alive

    def resizeEvent(self, e):
        self._layout_panel()

    def _layout_panel(self):
        # карточка целиком внутри экрана: края не срезаются нижней панелью
        ph = min(self.height() - 24, 420)
        final_y = self.height() - ph - 10
        v = max(0.0, min(1.0, self.open_.value))
        y = final_y + (self.height() - final_y) * (1 - v) * 0.35
        self.panel.setGeometry(10, int(y), self.width() - 20, ph)
        if hasattr(self, '_fade'):
            self._fade.setOpacity(v)

    def mouseReleaseEvent(self, e):
        if not self.panel.geometry().contains(e.position().toPoint()):
            self.close_sheet()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        v = max(0.0, min(1.0, self.open_.value))
        p.fillRect(self.rect(), qc(BG0, 0.6 * v))
        p.setOpacity(v)
        r = QRectF(self.panel.geometry())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc('#000000', 0.25))
        p.drawRoundedRect(r.adjusted(-4, 0, 4, 8), 22, 22)
        p.setBrush(qc(BG1))
        p.setPen(QPen(white(0.08), 1))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 20, 20)
        p.end()

    # ---------- логика ----------

    def _refresh_lock(self):
        eng = self.win.engine
        sys_item = self.win.current_sys()
        if eng.state not in (EngineState.READY, EngineState.RUNNING):
            self.ring.lock_icon = 'power'
            self.ring.set_mode('locked')
            self.src.set_text('Модели не загружены')
            self.caption.set_text('Нажмите, чтобы загрузить')
        elif not sys_item:
            self.ring.lock_icon = 'speaker'
            self.ring.set_mode('locked')
            self.src.set_text('')
            self.caption.set_text('Выберите, чей звук слушать')
        else:
            self.ring.set_mode('idle')
            self.src.set_text(f'Звук: {sys_item.name}')
            self.caption.set_text(f'{self.SECONDS} секунд речи')

    def _on_name(self, text: str):
        name = text.strip()
        exists = name in self.win.engine.voice_db
        self.replace_row.setVisible(exists)
        if exists:
            n = voices.load_counts().get(name, 1)
            self.replace_lbl.set_text('Записать заново' if self.replace.is_on()
                                      else f'Дополнить · {n} {plural(n, "запись", "записи", "записей")}')

    def _on_ring(self):
        mode = self.ring.mode
        if mode == 'locked':
            if self.ring.lock_icon == 'power':
                self.win.load_models()
                self.caption.set_text('Загружаю…')
                QTimer.singleShot(800, self._wait_models)
            else:
                self.win.pick_system_source(self.ring, after=self._refresh_lock)
            return
        if mode == 'rec':
            self._finish_recording()
            return
        if mode in ('idle', 'error', 'done'):
            self._start()

    def _wait_models(self):
        if not self.isVisible():
            return
        if self.win.engine.state == EngineState.LOADING:
            QTimer.singleShot(500, self._wait_models)
            return
        self._refresh_lock()

    def _start(self):
        name = self.name.text().strip()
        if not name:
            self.name.setFocus()
            self.ring.set_mode('error')
            self.caption.set_text('Сначала имя', RED)
            QTimer.singleShot(1200, lambda: self.ring.mode == 'error' and self.ring.set_mode('idle'))
            return
        sys_item = self.win.current_sys()
        if sys_item is None:
            self._refresh_lock()
            return
        self._chunks = []
        try:
            self._stream = audio_source.open_system_audio(sys_item.id, 16000, 512, self._audio_cb)
            self._stream.start()
        except Exception as e:
            self._stream = None
            self.ring.set_mode('error')
            self.caption.set_text(str(e)[:80], RED)
            return
        self._name = name
        self._t0 = time.monotonic()
        self.ring.set_mode('rec')
        self.caption.set_text('Пусть человек говорит', FG_DIM)
        self.name.setEnabled(False)
        self._tick.start(33)

    def _audio_cb(self, indata, frames, time_info, status):
        chunk = np.array(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32, copy=True)
        self._chunks.append(chunk)
        rms = float(np.sqrt(np.mean(chunk * chunk)) + 1e-9)
        self._lvl = max(0.0, min(1.0, (20 * math.log10(rms) + 52) / 40))

    def _on_tick(self):
        el = time.monotonic() - self._t0
        self.ring.progress = min(1.0, el / self.SECONDS)
        self.ring.remaining = self.SECONDS - el
        self.ring.push_level(self._lvl)
        self._lvl *= 0.85
        if el >= self.SECONDS:
            self._finish_recording()

    def _close_stream(self):
        self._tick.stop()
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.stop()
                s.close()
            except Exception:
                pass

    def _abort(self):
        self._close_stream()
        self.name.setEnabled(True)

    def _finish_recording(self):
        self._close_stream()
        self.name.setEnabled(True)
        if not self._chunks:
            self.ring.set_mode('error')
            self.caption.set_text('Тишина. Ещё раз?', RED)
            return
        audio = np.concatenate(self._chunks)
        self.ring.set_mode('proc')
        self.caption.set_text('Запоминаю голос…', FG_DIM)
        replace = self.replace.is_on()
        name = self._name
        threading.Thread(target=self._process, args=(audio, name, replace), daemon=True).start()

    def _process(self, audio, name, replace):
        eng = self.win.engine
        try:
            voiced = eng.voiced_only(audio)
            secs = len(voiced) / 16000
            if secs < self.MIN_VOICED_S:
                self._ui(lambda: self._fail(f'Голоса всего {secs:.1f} с. Ещё раз?'))
                return
            emb = eng.embed(voiced)
            voices.add_sample(name, emb, replace=replace)
            eng.reload_voices()
            self._ui(self._ok)
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)[:80]
            self._ui(lambda: self._fail(msg))

    def _ui(self, fn):
        # из рабочего потока в GUI: сигнал доставится очередью
        self._call.emit(fn)

    def _fail(self, msg: str):
        self.ring.set_mode('error')
        self.caption.set_text(msg, RED)

    def _ok(self):
        self.ring.set_mode('done')
        self.caption.set_text('Готово', ACCENT)
        self.finished.emit()
        QTimer.singleShot(1300, self.close_sheet)


class VoicesPage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        lay = _page_layout(self)

        me = RowCard()
        self.me_name = TextField(win.config.mic_speaker_name, 'Ваше имя', size=15)
        self.me_name.setFixedWidth(170)
        self.me_name.editingFinished.connect(self._save_me)
        me.add(Row('mic', 'Это я', self.me_name, tooltip='Так подписывается ваш микрофон'))
        lay.addWidget(me)
        lay.addSpacing(6)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(6)
        lay.addWidget(self.grid_host)
        lay.addStretch(1)

        self.sheet = EnrollSheet(win, self)
        self.sheet.finished.connect(self.refresh)
        self._tiles: list[QWidget] = []
        self.refresh()

    def resizeEvent(self, e):
        if self.sheet.isVisible():
            self.sheet.setGeometry(self.rect())

    def _save_me(self):
        name = self.me_name.text().strip() or 'user'
        self.win.set_mic_name(name)

    def refresh(self):
        for t in self._tiles:
            self.grid.removeWidget(t)
            t.deleteLater()
        self._tiles = []
        counts = voices.load_counts()
        names = sorted(counts.keys(), key=str.lower)
        add = AddTile()
        add.clicked.connect(self.sheet.open_sheet)
        tiles = [add]
        for n in names:
            t = VoiceTile(n, counts[n])
            t.delete_requested.connect(self._delete)
            tiles.append(t)
        for i, t in enumerate(tiles):
            self.grid.addWidget(t, i // 3, i % 3, Qt.AlignmentFlag.AlignHCenter)
        self._tiles = tiles
        self.me_name.setText(self.win.config.mic_speaker_name)

    def _delete(self, name: str):
        voices.delete(name)
        self.win.engine.reload_voices()
        self.win.toast(f'«{name}» удалён', 'info')
        self.refresh()


class _AvatarDot(AWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setFixedSize(52, 52)

    def set_name(self, name: str):
        self.name = name
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        paint_avatar(p, QPointF(26, 26), 22, self.name)
        p.end()


class _IconBadge(QWidget):
    def __init__(self, icon_name: str, color: str = FG_DIM, parent=None):
        super().__init__(parent)
        self.icon_name = icon_name
        self.color = color
        self.setFixedSize(34, 34)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setPen(QPen(white(0.07), 1))
        p.setBrush(white(0.03))
        p.drawEllipse(QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5))
        icons.draw(p, self.icon_name, QRectF(9, 9, 16, 16), self.color, stroke=1.8)
        p.end()


# ============================================================
# DISCORD
# ============================================================

class BotHero(AWidget):
    """Аватар бота с кольцом состояния. В канале от него расходятся волны."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(180, 144)
        self.state = 'off'
        self.image: Optional[QPixmap] = None
        self.col_from = white(0.08)
        self.col_to = white(0.08)
        self.ct = self.anim(1.0, 5)
        self.live = self.anim(0, 4)
        self.spin = self.anim(0, 6)
        self.speaking = self.anim(0, 10)
        self._t = 0.0

    def set_state(self, state: str):
        self.state = state
        col = {
            'online': white(0.22), 'in_voice': qc(ACCENT, 0.85),
            'error': qc(RED, 0.6), 'bad-token': qc(RED, 0.6),
        }.get(state, white(0.08))
        self.col_from = mix(self.col_from, self.col_to, self.ct.value)
        self.col_to = QColor(col)
        self.ct.set(0.0, instant=True)
        self.ct.set(1.0)
        self.live.set(1.0 if state == 'in_voice' else 0.0)
        self.spin.set(1.0 if state in ('connecting', 'joining') else 0.0)
        self.kick()

    def set_speaking(self, on: bool):
        self.speaking.set(1.0 if on else 0.0)
        self.kick()

    def continuous(self):
        return self.live.value > 0.01 or self.spin.value > 0.01

    def on_frame(self, dt):
        self._t += dt
        return super().on_frame(dt)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        c = QPointF(self.width() / 2, self.height() / 2)
        col = mix(self.col_from, self.col_to, self.ct.value)
        R = 40
        t = self._t
        live = self.live.value
        ring_r = R + 6
        # в канале: одна медленная волна; всё внутри ring_r + 20 < 72
        if live > 0.01:
            ph = (t * 0.4) % 1.0
            rr = ring_r + ph * 20
            p.setPen(QPen(qc(ACCENT, (1 - ph) * 0.35 * live), 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rr, rr)
        p.setPen(QPen(col, 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, ring_r, ring_r)
        if self.spin.value > 0.01:
            rect = QRectF(c.x() - ring_r, c.y() - ring_r, 2 * ring_r, 2 * ring_r)
            head = (t * 220) % 360
            p.setPen(QPen(white(0.6 * self.spin.value), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, int((90 - head) * 16), -int(55 * 16))
        if self.image is not None and not self.image.isNull():
            paint_avatar(p, c, R, 'bot', self.image)
        else:
            p.setPen(QPen(white(0.07), 1))
            p.setBrush(qc(BG1))
            p.drawEllipse(c, R, R)
            s = 32
            icons.draw(p, 'bot', QRectF(c.x() - s / 2, c.y() - s / 2, s, s),
                       mix(FG_DIM, FG, 0.5 * self.speaking.value), stroke=1.6)
        p.end()


class _StepTile(AWidget):
    """Номер шага: впереди - тусклый, текущий - основным цветом, пройденный - галочка."""

    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.n = n
        self.state = 'todo'
        self.setFixedSize(30, 30)
        self.now = self.anim(0, 10)
        self.ok = self.anim(0, 10)

    def set_state(self, state: str):
        self.state = state
        self.now.set(1.0 if state == 'now' else 0.0)
        self.ok.set(1.0 if state == 'done' else 0.0)
        self.kick()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        now, ok = self.now.value, self.ok.value
        p.setPen(QPen(mix(white(0.12), qc(PRIMARY, 0.0), max(now, ok)), 1.2))
        p.setBrush(mix(mix(white(0.0), qc(PRIMARY, 0.18), now), qc(ACCENT, 0.16), ok))
        p.drawRoundedRect(r, 9, 9)
        if ok > 0.5:
            icons.draw(p, 'check', r.adjusted(7, 7, -7, -7), qc(ACCENT), stroke=2.4)
        else:
            p.setFont(font(13, QFont.Weight.Bold))
            p.setPen(mix(FG_FAINT, PRIMARY, now))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(self.n))
        p.end()


class StepRow(QWidget):
    def __init__(self, n: int, title: str, control: Optional[QWidget] = None,
                 tooltip: str = '', parent=None):
        super().__init__(parent)
        self.setFixedHeight(54)
        if tooltip:
            self.setToolTip(tooltip)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 12, 0)
        lay.setSpacing(14)
        self.tile = _StepTile(n)
        lay.addWidget(self.tile)
        self.title = FadeLabel(title, size=14, color=FG, weight=QFont.Weight.Medium,
                               align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if title:
            lay.addWidget(self.title, 1)
        self.control = control
        if control is not None:
            lay.addWidget(control, 0 if title else 1)
        elif not title:
            lay.addStretch(1)

    def set_state(self, state: str):
        self.tile.set_state(state)
        col = {'todo': FG_FAINT, 'now': FG, 'done': FG_DIM}[state]
        self.title.set_text(self.title.text(), col)


class DiscordPage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        lay = _page_layout(self, top=0)
        lay.setSpacing(10)

        self.hero = BotHero()
        lay.addLayout(_hcenter(self.hero))
        self.status = FadeLabel('', size=15, color=FG, weight=QFont.Weight.DemiBold, display=True)
        lay.addWidget(self.status)
        self.sub = FadeLabel('', size=12, color=FG_DIM)
        lay.addWidget(self.sub)
        lay.addSpacing(4)

        # ---- первый запуск: четыре шага, потом обычная карточка бота ----
        self.setup = RowCard()
        self.step_make = Chevron('external', tooltip='Discord Developer Portal:\n'
                                 'New Application - любое имя\n'
                                 'Bot - Reset Token - Copy')
        self.step_make.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl('https://discord.com/developers/applications')))
        self.steps = [self.setup.add(StepRow(1, 'Создать бота', self.step_make))]
        tok = QWidget()
        trow = QHBoxLayout(tok)
        trow.setContentsMargins(0, 0, 0, 0)
        trow.setSpacing(2)
        self.token = TextField('', 'Токен', size=14)
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.returnPressed.connect(self._save_token)
        self.token.textChanged.connect(self._on_token_text)
        trow.addWidget(self.token, 1)
        self.eye = IconButton('eye', 32, 16, tooltip='Показать')
        self.eye.clicked.connect(self._toggle_eye)
        trow.addWidget(self.eye)
        self.paste = IconButton('paste', 32, 16, tooltip='Вставить')
        self.paste.clicked.connect(self._paste)
        trow.addWidget(self.paste)
        self.steps.append(self.setup.add(StepRow(2, '', tok)))
        self.step_invite = Chevron('external', tooltip='Выбрать сервер, куда позвать бота')
        self.step_invite.clicked.connect(self._invite)
        self.steps.append(self.setup.add(StepRow(3, 'Позвать на сервер', self.step_invite)))
        self.steps.append(self.setup.add(StepRow(4, 'Зайти в голосовой', None,
                                                 tooltip='Бот сам придёт в ваш канал')))
        lay.addWidget(self.setup)

        # ---- бот подключён ----
        self.bot_box = QWidget()
        bl = QVBoxLayout(self.bot_box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)
        card = self.bot_card = RowCard()
        self.enabled = Toggle(win.config.discord_bot_enabled)
        self.enabled.toggled.connect(win.set_bot_enabled)
        self.bot_row = card.add(Row('bot', 'Бот', self.enabled))
        self.bot_name = self.bot_row.title
        self.follow = _FollowRow()
        self.follow.reset.connect(win.reset_follow)
        card.add(self.follow)
        self.hear_me = Toggle(win.config.discord_hear_me)
        self.hear_me.toggled.connect(win.set_hear_me)
        card.add(Row('mic', 'Мой голос', self.hear_me,
                     tooltip='Субтитры и для вашей речи. Если выбран микрофон, ваш голос и так идёт с него'))
        self.invite = Chevron('external', tooltip='Пригласить бота на сервер')
        self.invite.clicked.connect(self._invite)
        card.add(Row('plus', 'На сервер', self.invite))
        bl.addWidget(card)

        self.members = AvatarStrip(18)
        bl.addWidget(self.members)

        out = RowCard()
        self.forget = Chevron('logout', tooltip='Забыть токен')
        self.forget.danger = True
        self.forget.clicked.connect(self._forget)
        forget_row = out.add(Row('key', 'Забыть токен', self.forget, color=RED, height=48))
        forget_row.title.color = FG_DIM
        bl.addSpacing(8)
        bl.addWidget(out)
        lay.addWidget(self.bot_box)

        # голос режет сеть: путь к node.exe - добавить в обход блокировок
        self.net_hint = RowCard()
        self.copy_node = Chevron('copy', tooltip='Скопировать путь к node.exe -\n'
                                 'добавьте его в обход блокировок (OneXray и т.п.)\n'
                                 'так же, как Discord.exe')
        self.copy_node.clicked.connect(self._copy_node)
        self.net_hint.add(Row('alert', 'node.exe в обход', self.copy_node, color=RED, height=50))
        self.net_hint.hide()
        lay.insertWidget(lay.indexOf(self.setup), self.net_hint)

        # Discord не пускает бота в канал: права канала
        self.perm_hint = RowCard()
        perm_row = self.perm_hint.add(Row('lock', 'Доступ к каналу', None, color=RED, height=50,
                                          tooltip='Настройки канала - Права доступа - добавить роль бота\n'
                                                  '(или самого бота) и разрешить «Просматривать канал»\n'
                                                  'и «Подключаться». Делает владелец сервера или админ.'))
        perm_row.setToolTip(perm_row.toolTip())
        self.perm_hint.hide()
        lay.insertWidget(lay.indexOf(self.setup), self.perm_hint)
        lay.addStretch(1)

        self.refresh()

    # ---------- токен ----------

    def _on_token_text(self, text: str):
        if secret_store.looks_like_token(text.strip()):
            self._save_token()

    def _toggle_eye(self):
        pw = self.token.echoMode() == QLineEdit.EchoMode.Password
        self.token.setEchoMode(QLineEdit.EchoMode.Normal if pw else QLineEdit.EchoMode.Password)
        self.eye.set_icon('eye-off' if pw else 'eye')

    def _paste(self):
        text = (QGuiApplication.clipboard().text() or '').strip()
        if text:
            self.token.setText(text)
            if secret_store.looks_like_token(text):
                self._save_token()

    def _save_token(self):
        text = self.token.text().strip()
        if not secret_store.looks_like_token(text):
            self.win.toast('Это не похоже на токен бота', 'warn')
            return
        secret_store.save_token(text)
        self.token.clear()
        self.win.set_bot_enabled(True)
        self.enabled.set_on(True)
        self.refresh()

    def _copy_node(self):
        import discord_bot
        path = discord_bot.find_node() or ''
        QGuiApplication.clipboard().setText(path)
        self.copy_node.set_on(True)
        QTimer.singleShot(900, lambda: self.copy_node.set_on(False))
        self.win.toast('Путь скопирован', 'ok')

    def _forget(self):
        self.win.bridge.stop()
        secret_store.forget_token()
        self.hero.image = None
        self.refresh()

    def _invite(self):
        bot_id = self.win.bridge.bot.get('id') or secret_store.bot_id_from_token(secret_store.load_token())
        if bot_id:
            QDesktopServices.openUrl(QUrl(secret_store.invite_url(bot_id)))

    # ---------- состояние ----------

    def refresh(self):
        has_token = bool(secret_store.load_token())
        br = self.win.bridge
        state = br.state if has_token else 'no-token'
        if state == 'bad-token':
            has_token = False
        if has_token and br.guilds > 0 and not self.win.config.discord_setup_done:
            self.win.config.discord_setup_done = True
            self.win.save()
        setup = not (has_token and self.win.config.discord_setup_done)
        self.setup.setVisible(setup)
        self.bot_box.setVisible(has_token)
        self.bot_card.setVisible(not setup)
        self.members.setVisible(not setup and state == 'in_voice')
        done = [has_token, has_token, br.guilds > 0 or self.win.config.discord_setup_done,
                state == 'in_voice']
        current = next((i for i, d in enumerate(done) if not d), 4)
        for i, step in enumerate(self.steps):
            step.set_state('done' if done[i] else ('now' if i == current else 'todo'))
        self.step_invite.setEnabled(has_token)
        self.hero.set_state(state)

        titles = {
            'no-token': ('Бот не подключён', ''),
            'bad-token': ('Токен не подошёл', 'Скопируйте его заново'),
            'off': ('Выключен', ''),
            'connecting': ('Подключение', ''),
            'online': ('Онлайн', 'Зайдите в голосовой канал'),
            'joining': ('Захожу в канал', br.detail),
            'error': ('Ошибка', self._error_text(br.detail)),
        }
        if state == 'in_voice' and br.channel:
            ch = br.channel.get('channel', {}).get('name', '')
            guild = br.channel.get('guild', {}).get('name', '')
            title, sub = f'# {ch}', guild
        else:
            title, sub = titles.get(state, (state, ''))
        if state == 'online' and br.guilds == 0:
            sub = 'Пригласите бота на сервер'
        self.status.set_text(title)
        self.sub.set_text(sub)
        self.net_hint.setVisible(state == 'error' and br.detail == 'voice-net')
        self.perm_hint.setVisible(state == 'error' and br.detail in ('no-connect', 'voice-denied'))
        self.bot_name.set_text(br.bot.get('name') or 'Бот')
        self.enabled.set_on(self.win.config.discord_bot_enabled)
        self._refresh_follow()

    def _refresh_follow(self):
        br = self.win.bridge
        uid = br.follow_id
        name = ''
        if uid:
            m = br.members.get(uid)
            name = (m or {}).get('name') or getattr(br, 'follow_name', '')
        self.follow.set_target(uid, name, br.members.get(uid, {}).get('image') if uid else None)

    @staticmethod
    def _error_text(detail: str) -> str:
        return {
            'no-node': 'Нужен Node.js 22+',
            'old-node': 'Обновите Node.js до 22+',
            'no-modules': 'Не установлены модули моста',
            'no-bridge': 'Нет файлов моста',
            'voice-timeout': 'Канал не ответил',
            'voice-denied': 'Discord не пустил в канал',
            'no-connect': 'Боту нельзя в этот канал',
            'channel-full': 'Канал заполнен',
            'voice-net': 'Голос не доходит до Discord',
        }.get(detail, detail[:60])


class _FollowRow(AWidget):
    """За кем ходит бот. Крестик сбрасывает - следующий /join назначит заново."""
    reset = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(54)
        self.uid = None
        self.name = ''
        self.image = None
        self.h = self.anim(0, 14)
        self.setMouseTracking(True)
        self._x_hover = False

    def set_target(self, uid, name, image=None):
        self.uid, self.name, self.image = uid, name, image
        self.setCursor(Qt.CursorShape.PointingHandCursor if uid else Qt.CursorShape.ArrowCursor)
        self.setToolTip('' if uid else 'Напишите /join в Discord')
        self.update()

    def _x_rect(self):
        return QRectF(self.width() - 42, 13, 28, 28)

    def mouseMoveEvent(self, e):
        xh = self._x_rect().contains(e.position())
        if xh != self._x_hover:
            self._x_hover = xh
            self.h.set(1.0 if xh else 0.0)
            self.kick()

    def hover_changed(self, inside):
        if not inside:
            self._x_hover = False
            self.h.set(0.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if self.uid and self._x_rect().contains(e.position()):
            self.reset.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        tile = QRectF(14, 12, 30, 30)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(PRIMARY, 0.15))
        p.drawRoundedRect(tile, 9, 9)
        icons.draw(p, 'user', tile.adjusted(7, 7, -7, -7), qc(PRIMARY), stroke=1.9)
        f = font(14, QFont.Weight.Medium)
        p.setFont(f)
        x = 58
        if self.uid:
            label = self.name or 'выбран'
            paint_avatar(p, QPointF(x + 13, 27), 13, label, self.image)
            p.setPen(qc(FG))
            p.drawText(QRectF(x + 34, 0, self.width() - x - 84, 54),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       elide(label, f, self.width() - x - 84))
            xr = self._x_rect()
            p.setBrush(qc(RED, 0.15 * self.h.value))
            p.drawEllipse(xr.center(), 13, 13)
            icons.draw(p, 'x', xr.adjusted(7, 7, -7, -7), mix(FG_FAINT, RED, self.h.value), stroke=2.4)
        else:
            p.setPen(qc(FG_FAINT))
            p.drawText(QRectF(x, 0, self.width() - x - 14, 54),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, '/join')
        p.end()


# ============================================================
# ОВЕРЛЕЙ
# ============================================================

class ScreenPreview(AWidget):
    """
    Маленький экран. Зону субтитров на нём можно таскать и тянуть за края -
    настоящий оверлей двигается вместе с ней.
    """
    geometry_changed = pyqtSignal(int, int, int, int)

    EDGE = 9

    def __init__(self, overlay_win, parent=None):
        super().__init__(parent)
        self.ov = overlay_win
        self.setFixedHeight(206)
        self.setMouseTracking(True)
        self.h = self.anim(0, 12)
        self.drag = self.anim(0, 12)
        self._kind = None
        self._start = QPointF()
        self._start_zone = QRectF()
        self._hover_kind = None
        self.renderer = overlay.CardRenderer()
        self.cards = overlay.demo_cards('p')
        self._t = 0.0

    def on_frame(self, dt):
        self._t += dt
        return super().on_frame(dt)

    def _screen(self) -> QRect:
        return QApplication.primaryScreen().geometry()

    def _monitor(self) -> QRectF:
        scr = self._screen()
        aspect = scr.width() / max(1, scr.height())
        h = self.height() - 34
        w = h * aspect
        if w > self.width() - 8:
            w = self.width() - 8
            h = w / aspect
        return QRectF((self.width() - w) / 2, 6, w, h)

    def _scale(self) -> float:
        return self._monitor().width() / max(1, self._screen().width())

    def _zone_widget(self) -> QRectF:
        g = self.ov.geometry()
        scr = self._screen()
        m = self._monitor()
        s = self._scale()
        return QRectF(m.left() + (g.x() - scr.x()) * s, m.top() + (g.y() - scr.y()) * s,
                      g.width() * s, g.height() * s)

    def _hit(self, pos: QPointF) -> Optional[str]:
        z = self._zone_widget()
        e = self.EDGE
        if not z.adjusted(-e, -e, e, e).contains(pos):
            return None
        l, r = abs(pos.x() - z.left()) < e, abs(pos.x() - z.right()) < e
        t, b = abs(pos.y() - z.top()) < e, abs(pos.y() - z.bottom()) < e
        if t and l: return 'tl'
        if t and r: return 'tr'
        if b and l: return 'bl'
        if b and r: return 'br'
        if l: return 'left'
        if r: return 'right'
        if t: return 'top'
        if b: return 'bottom'
        return 'move' if z.contains(pos) else None

    _CURSORS = {
        'move': Qt.CursorShape.SizeAllCursor,
        'left': Qt.CursorShape.SizeHorCursor, 'right': Qt.CursorShape.SizeHorCursor,
        'top': Qt.CursorShape.SizeVerCursor, 'bottom': Qt.CursorShape.SizeVerCursor,
        'tl': Qt.CursorShape.SizeFDiagCursor, 'br': Qt.CursorShape.SizeFDiagCursor,
        'tr': Qt.CursorShape.SizeBDiagCursor, 'bl': Qt.CursorShape.SizeBDiagCursor,
    }

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._kind is None:
            k = self._hit(pos)
            if k != self._hover_kind:
                self._hover_kind = k
                self.h.set(1.0 if k else 0.0)
                self.setCursor(self._CURSORS.get(k, Qt.CursorShape.ArrowCursor))
                self.kick()
            return
        s = self._scale()
        d = (pos - self._start) / s
        z = QRectF(self._start_zone)
        k = self._kind
        min_w, min_h = overlay.EDIT_MIN_W, overlay.EDIT_MIN_H
        if k == 'move':
            z.translate(d)
        else:
            if k in ('left', 'tl', 'bl'):
                z.setLeft(min(z.left() + d.x(), z.right() - min_w))
            if k in ('right', 'tr', 'br'):
                z.setRight(max(z.right() + d.x(), z.left() + min_w))
            if k in ('top', 'tl', 'tr'):
                z.setTop(min(z.top() + d.y(), z.bottom() - min_h))
            if k in ('bottom', 'bl', 'br'):
                z.setBottom(max(z.bottom() + d.y(), z.top() + min_h))
        scr = QRectF(self._screen())
        if z.left() < scr.left(): z.moveLeft(scr.left())
        if z.top() < scr.top(): z.moveTop(scr.top())
        if z.right() > scr.right(): z.moveRight(scr.right())
        if z.bottom() > scr.bottom(): z.moveBottom(scr.bottom())
        # прилипание к центру по горизонтали
        if k == 'move' and abs(z.center().x() - scr.center().x()) < 24:
            z.moveCenter(QPointF(scr.center().x(), z.center().y()))
        g = z.toRect()
        self.ov.apply_geometry(g.x(), g.y(), g.width(), g.height())
        self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        k = self._hit(e.position())
        if not k:
            return
        self._kind = k
        self._start = e.position()
        self._start_zone = QRectF(self.ov.geometry())
        self.drag.set(1.0)
        self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._kind:
            self._kind = None
            self.drag.set(0.0)
            self.kick()
            g = self.ov.geometry()
            self.geometry_changed.emit(g.x(), g.y(), g.width(), g.height())

    def hover_changed(self, inside):
        if not inside and self._kind is None:
            self._hover_kind = None
            self.h.set(0.0)
            self.kick()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        m = self._monitor()
        t = self._t

        # подставка
        stand = QPainterPath()
        cx = m.center().x()
        stand.moveTo(cx - 14, m.bottom())
        stand.lineTo(cx + 14, m.bottom())
        stand.lineTo(cx + 22, m.bottom() + 18)
        stand.lineTo(cx - 22, m.bottom() + 18)
        stand.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(BG3))
        p.drawPath(stand)
        p.drawRoundedRect(QRectF(cx - 40, m.bottom() + 16, 80, 6), 3, 3)

        # рамка и нейтральный «рабочий стол»
        p.setBrush(qc('#0A0B0D'))
        p.setPen(QPen(white(0.09), 1))
        p.drawRoundedRect(m.adjusted(-5, -5, 5, 5), 11, 11)
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(m, 5, 5)
        p.setClipPath(clip)
        g = QLinearGradient(m.topLeft(), m.bottomRight())
        g.setColorAt(0, qc('#22242A'))
        g.setColorAt(1, qc('#17181C'))
        p.fillRect(m, g)
        p.fillRect(QRectF(m.left(), m.bottom() - 7, m.width(), 7), qc('#000000', 0.30))

        z = self._zone_widget()
        s = self._scale()
        hv = max(self.h.value, self.drag.value)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(white(0.03 + 0.04 * hv))
        p.drawRoundedRect(z, 3, 3)
        p.save()
        p.translate(z.topLeft())
        p.scale(s, s)
        st = self.ov.style_
        style = overlay.OverlayStyle(st.font_size, st.bg_opacity, st.align, 0, st.max_lines)
        overlay.paint_stack(p, QRectF(0, 0, z.width() / s, z.height() / s), self.renderer,
                            self.cards, [], style, overlay._now(), 0.0)
        p.restore()
        pen = QPen(white(0.25 + 0.45 * hv), 1.0, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([3, 3])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(z, 3, 3)
        if hv > 0.01:
            for hx, hy in ((z.left(), z.top()), (z.right(), z.top()), (z.left(), z.bottom()), (z.right(), z.bottom())):
                p.setPen(QPen(qc('#0A0B0D', hv), 1.5))
                p.setBrush(white(0.9 * hv))
                p.drawEllipse(QPointF(hx, hy), 3, 3)
        p.restore()

        # направляющая центра при перетаскивании
        if self._kind == 'move' and abs(z.center().x() - m.center().x()) < 1.5:
            p.setPen(QPen(white(0.45), 1, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(m.center().x(), m.top()), QPointF(m.center().x(), m.bottom()))
        p.end()


class OverlayPage(QWidget):
    def __init__(self, win, overlay_win, parent=None):
        super().__init__(parent)
        self.win = win
        self.ov = overlay_win
        cfg = win.config
        lay = _page_layout(self, top=2)
        lay.setSpacing(10)

        self.preview = ScreenPreview(overlay_win)
        self.preview.setFixedHeight(116)
        self.preview.geometry_changed.connect(win.save_overlay_geometry)
        lay.addWidget(self.preview)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.align = Pills([('left', None, 'align-left', 'По левому краю'),
                            ('center', None, 'align-center', 'По центру')],
                           cfg.overlay_align if cfg.overlay_align == 'center' else 'left')
        self.align.changed.connect(self._align)
        row.addWidget(self.align)
        row.addStretch(1)
        self.edit = IconButton('expand', 36, 18, tooltip='Двигать прямо на экране')
        self.edit.clicked.connect(self._edit_on_screen)
        row.addWidget(self.edit)
        self.demo = IconButton('sparkles', 36, 18, tooltip='Показать пример')
        self.demo.clicked.connect(self._demo)
        row.addWidget(self.demo)
        lay.addLayout(row)

        text = RowCard()
        sizes = [12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40, 48]
        self.size = text.add(SliderRow('letter', 'Размер', sizes, cfg.overlay_font_size, str))
        self.size.changed.connect(self._size)
        self.lines = Pills([(n, str(n), None, f'{n} строки') for n in (2, 3, 4, 5)],
                           cfg.overlay_max_lines, cell=34)
        self.lines.changed.connect(self._lines)
        text.add(Row('lines', 'Строки', self.lines, tooltip='Сколько строк видно в карточке'))
        lay.addWidget(text)

        look = RowCard()
        alphas = list(range(0, 101, 10))
        self.alpha = look.add(SliderRow('droplet', 'Фон', alphas,
                                        int(round(cfg.overlay_bg_opacity / 10) * 10), lambda v: f'{v}%'))
        self.alpha.changed.connect(self._alpha)
        hovers = [0, 10, 20, 30, 40, 50, 60, 80, 100]
        hv = min(hovers, key=lambda v: abs(v - cfg.overlay_hover_opacity))
        self.hover = look.add(SliderRow('eye', 'Под курсором', hovers, hv,
                                        lambda v: 'Выкл' if v >= 100 else f'{v}%',
                                        tooltip='Навели мышь на субтитры - они становятся прозрачными'))
        self.hover.changed.connect(self._hover)
        ttls = [3, 5, 8, 10, 15, 20, 30, 60, 120, 300, 0]
        self.ttl = look.add(SliderRow('clock', 'Держать', ttls, cfg.overlay_ttl_sec, self._fmt_ttl,
                                      tooltip='Сколько текст висит после того, как человек замолчал'))
        self.ttl.changed.connect(self._ttl)
        lay.addWidget(look)
        lay.addStretch(1)

        self._editor = None

    @staticmethod
    def _fmt_ttl(v: int) -> str:
        if v == 0:
            return '∞'
        if v < 60:
            return f'{v} с'
        return f'{v // 60} мин'

    def _align(self, a: str):
        self.win.config.overlay_align = a
        self.ov.set_align(a)
        self.win.save()
        self.preview.update()

    def _lines(self, n: int):
        self.win.config.overlay_max_lines = n
        self.ov.set_max_lines(n)
        self.win.save()
        self.preview.update()

    def _size(self, v):
        self.win.config.overlay_font_size = v
        self.ov.set_font_size(v)
        if self._editor is not None:
            self._editor.set_font_size(v)
        self.win.save()
        self.preview.update()

    def _ttl(self, v):
        self.win.config.overlay_ttl_sec = v
        self.ov.set_ttl_seconds(v)
        self.win.save()

    def _alpha(self, v):
        self.win.config.overlay_bg_opacity = v
        self.ov.set_bg_opacity(v / 100)
        self.win.save()
        self.preview.update()

    def _hover(self, v):
        self.win.config.overlay_hover_opacity = v
        self.ov.set_hover_opacity(v / 100)
        if v < 100:
            self.ov.preview_hover()
        self.win.save()

    def _demo(self):
        self.ov.show_demo()

    def _edit_on_screen(self):
        if self._editor is not None:
            self._editor.close()
            return
        self.ov.hide_overlay()
        self._editor = self.ov.open_editor()
        self.edit.set_on(True)
        self._editor.geometry_changed.connect(lambda *_: self.preview.update())
        self._editor.closed.connect(self._editor_closed)

    def _editor_closed(self):
        ed, self._editor = self._editor, None
        if ed is not None:
            g = ed.geometry()
            self.ov.apply_geometry(g.x(), g.y(), g.width(), g.height())
            self.win.save_overlay_geometry(g.x(), g.y(), g.width(), g.height())
        self.ov.show_overlay()
        self.edit.set_on(False)
        self.preview.update()

    def leave(self):
        if self._editor is not None:
            self._editor.close()


# ============================================================
# ОТКЛИК
# ============================================================

_STAGES = [
    ('word_total', 'от конца слова до экрана'),
    ('audio_queue', 'звук в очереди'),
    ('stt_wait', 'окно ждало распознавания'),
    ('infer', 'распознавание окна'),
    ('word_arrive', 'слово распознано (от конца слова)'),
    ('word_pace', 'слово ждало в темпе чтения'),
    ('tr_total', 'перевод на экране (от конца слова)'),
    ('tr_confirm', 'фраза отдана переводчику'),
    ('tr_queue', 'перевод ждал очереди'),
    ('tr_infer', 'перевод'),
]


def _delay_tip(stages: dict) -> str:
    """Подсказка «Задержки»: медиана и 90% по этапам (metrics.py). Этапы не складываются."""
    def fmt(v):
        return f'{v * 1000:.0f} мс' if v < 1 else f'{v:.2f} с'

    lines = ['От конца слова в звуке до появления на экране.', 'Медиана · у 90% не дольше:']
    for key, name in _STAGES:
        if key in stages:
            p50, p90, _n = stages[key]
            lines.append(f'  {name}: {fmt(p50)} · {fmt(p90)}')
    return '\n'.join(lines)


class TunePage(QWidget):
    PRESETS = [
        ('fast', 'zap', 'Быстро'),
        ('balanced', 'gauge', 'Баланс'),
        ('stable', 'leaf', 'Экономно'),
    ]

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        lay = _page_layout(self, top=10)
        lay.setSpacing(12)
        self.presets = Segmented(self.PRESETS, win.config.recognition_preset)
        self.presets.setToolTip('Быстро - слово видно раньше, но процессор работает больше.\n'
                                'Экономно - чуть позже, зато нагрузка меньше всего.')
        self.presets.changed.connect(win.set_preset)
        lay.addWidget(self.presets)

        card = RowCard()
        import asr_parakeet
        import translate_onnx
        eng = win.engine
        self.eng = Toggle(eng.multilang)
        self.eng.toggled.connect(self._language)
        self.eng_row = card.add(Row('languages', 'Английский', self.eng,
                                    tooltip='Английская речь: распознаёт другая модель, она понимает\n'
                                            'и английский, и русский, и ещё 23 языка - сама\n'
                                            'разбирает, на каком говорят'))
        self.eng_row.setVisible(asr_parakeet.available())
        self.view = Pills([('orig', 'EN', None, 'Только английский текст'),
                           ('both', 'EN+RU', None, 'Английский и перевод под ним'),
                           ('trans', 'RU', None, 'Только перевод на русский')],
                          eng.view, cell=54)
        self.view.changed.connect(win.set_view)
        self.view_row = card.add(Row('captions', 'Показ', self.view))
        self._can_translate = translate_onnx.available()
        self.view_row.setVisible(eng.multilang and self._can_translate)
        self.gpu = Toggle(win.config.use_gpu)
        self.gpu.toggled.connect(self._gpu)
        self.gpu_row = card.add(Row('gpu', 'Видеокарта', self.gpu,
                                    tooltip='Распознавать на видеокарте: процессор почти свободен,\n'
                                            'субтитры не отстают даже на слабом процессоре'))
        self.gpu_row.setVisible(asr_onnx.available_gpu() and asr_onnx.gpu_models_present())
        self.delay = ValueText('')
        self.delay_row = card.add(Row('activity', 'Задержка', self.delay,
                                      tooltip='От конца слова в звуке до появления на экране'))
        self.infer = ValueText('')
        card.add(Row('cpu', 'Распознавание', self.infer, tooltip='Сколько считается одно окно звука'))
        self.speed = ValueText('', color=ACCENT)
        card.add(Row('zap', 'Скорость', self.speed, color=ACCENT,
                     tooltip='Во сколько раз быстрее, чем говорят'))
        self.spark = Sparkline()
        self.spark_box = QWidget()
        sb = QHBoxLayout(self.spark_box)
        sb.setContentsMargins(52, 0, 14, 8)
        sb.addWidget(self.spark)
        card.add(self.spark_box)
        self.spark_box.hide()
        lay.addWidget(card)

        logs = RowCard()
        self.log_btn = Chevron(tooltip='Открыть журнал')
        self.log_btn.clicked.connect(win.open_log)
        # версия - в той же строке: журнал и номер версии нужны вместе, когда что-то сломалось
        log_ctl = QWidget()
        lc = QHBoxLayout(log_ctl)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.setSpacing(6)
        lc.addWidget(ValueText(version.LABEL, color=FG_FAINT))
        lc.addWidget(self.log_btn)
        logs.add(Row('file', 'Журнал', log_ctl, color=FG_DIM, height=48))
        lay.addStretch(1)
        lay.addWidget(logs)

    def _language(self, on: bool):
        if self.win.engine.stt is not None:
            self.set_language(on, busy=True)
        self.win.set_language(on)

    def set_language(self, on: bool, busy: bool = False):
        self.eng.setEnabled(not busy)
        self.eng_row.icon.set_color(ACCENT if on and not busy else PRIMARY)
        if not busy and self.eng.is_on() != on:
            self.eng.set_on(on)
        self.view_row.setVisible(on and self._can_translate)

    def set_view(self, view: str):
        self.view.set_current(view)

    def _gpu(self, on: bool):
        if self.win.engine.stt is not None:
            self.set_device(on, busy=True)
        self.win.set_gpu(on)

    def set_device(self, gpu: bool, busy: bool = False):
        """Переключатель показывает, где распознавание идёт на самом деле."""
        self.gpu.setEnabled(not busy)
        self.gpu_row.icon.set_color(ACCENT if gpu and not busy else PRIMARY)
        if not busy and self.gpu.is_on() != gpu:
            self.gpu.set_on(gpu)

    def update_stats(self, st: dict):
        infer = st.get('infer_ms')
        rtf = st.get('rtf')
        stages = st.get('stages') or {}
        total = stages.get('tr_total' if st.get('view') == 'trans' else 'word_total')
        self.delay.set_text('—' if total is None else f'{total[0]:.2f} с')
        self.delay_row.setToolTip(_delay_tip(stages))
        self.infer.set_text('—' if infer is None else f'{infer:.0f} мс')
        self.speed.set_text(f'×{1 / rtf:.0f}' if rtf and rtf > 0 else '—')
        hist = st.get('history') or []
        self.spark.set_values(hist)
        self.spark_box.setVisible(len(hist) > 1)
