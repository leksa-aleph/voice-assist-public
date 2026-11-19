"""Главное окно: рамка, живой фон, нижняя панель, переходы, трей."""
from __future__ import annotations

import ctypes
import sys
from typing import Optional

from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QDesktopServices, QFont, QFontMetricsF, QIcon, QLinearGradient, QPainter,
    QPainterPath, QPen, QPixmap, QRegion,
)
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMenu, QSystemTrayIcon, QVBoxLayout, QWidget

import audio_source
import overlay
import secret_store
import version
from config import AppConfig, load_config, save_config
from discord_bot import DiscordBridge
from engine import EngineState, TranscriberEngine
from logging_setup import log_path
from ui import icons
from ui.anim import Anim, Spring, clock
from ui.history_page import HistoryPage
from ui.pages import DiscordPage, HomePage, OverlayPage, TunePage, VoicesPage
from ui.theme import (
    ACCENT, BG0, BG1, BG2, BG3, FG, FG_DIM, FG_FAINT, PRIMARY, RED,
    ease_out, font, lerp, mix, qc, white,
)
from ui.widgets import AA, AWidget, FadeLabel, IconButton, Popover, Toast, paint_window_bg

W, H = 400, 664


# ============================================================
# ЛОГОТИП
# ============================================================

def paint_logo(p: QPainter, r: QRectF, tint: Optional[str] = None, gray: bool = False):
    """Квадрат основного цвета, три тёмные полоски. tint - точка состояния в углу."""
    s = r.width()
    rad = s * 0.28
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(qc('#2A2C36' if gray else PRIMARY))
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), rad, rad)
    ink = qc(FG_DIM if gray else BG0)
    bars = (0.30, 0.52, 0.38)
    bw = s * 0.11
    gap = s * 0.08
    x0 = r.center().x() - (3 * bw + 2 * gap) / 2
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(ink)
    for i, k in enumerate(bars):
        bh = s * k
        p.drawRoundedRect(QRectF(x0 + i * (bw + gap), r.center().y() - bh / 2, bw, bh), bw / 2, bw / 2)
    if tint:
        d = s * 0.30
        c = QPointF(r.right() - d * 0.38, r.bottom() - d * 0.38)
        p.setBrush(qc(BG0))
        p.drawEllipse(c, d * 0.6, d * 0.6)
        p.setBrush(qc(tint))
        p.drawEllipse(c, d * 0.42, d * 0.42)


def app_icon(tint: Optional[str] = None, gray: bool = False) -> QIcon:
    ic = QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(AA)
        m = size * 0.06
        paint_logo(p, QRectF(m, m, size - 2 * m, size - 2 * m), tint, gray)
        p.end()
        ic.addPixmap(pm)
    return ic


class Logo(QWidget):
    def __init__(self, size: int = 24, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        paint_logo(p, QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
        p.end()


# ============================================================
# ЗАГОЛОВОК, ПАНЕЛЬ, СТЕК
# ============================================================

class StageBadge(QWidget):
    """Плашка «бета» в заголовке, пока выпуск не обычный."""

    TEXT = {'beta': 'бета', 'alpha': 'альфа'}

    def __init__(self, stage: str, parent=None):
        super().__init__(parent)
        self._text = self.TEXT.get(stage, stage)
        self._f = font(10.5, QFont.Weight.DemiBold)
        fm = QFontMetricsF(self._f)
        self.setFixedSize(int(fm.horizontalAdvance(self._text) + 16), 20)
        self.setToolTip(f'Версия {version.LABEL}. Бета: основное работает, '
                        f'но возможны ошибки - если что-то сломалось, '
                        f'пригодится журнал (вкладка «Отклик»).')

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(qc(FG_FAINT), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setFont(self._f)
        p.setPen(qc(FG_DIM))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()


class TitleBar(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.setFixedHeight(48)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 6, 10, 0)
        lay.setSpacing(10)
        lay.addWidget(Logo(24))
        self.title = FadeLabel('', size=16, color=FG, weight=QFont.Weight.DemiBold, display=True,
                               align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.title, 1)
        if version.STAGE:
            lay.addWidget(StageBadge(version.STAGE), 0, Qt.AlignmentFlag.AlignVCenter)
        self.min_btn = IconButton('minus', 32, 16, tooltip='Свернуть')
        self.min_btn.clicked.connect(win.showMinimized)
        lay.addWidget(self.min_btn)
        self.close_btn = IconButton('x', 32, 16, tooltip='Выход')
        self.close_btn.danger = True
        self.close_btn.clicked.connect(win.close)
        lay.addWidget(self.close_btn)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            h = self.window().windowHandle()
            if h is not None:
                h.startSystemMove()


class NavDock(AWidget):
    changed = pyqtSignal(int)

    def __init__(self, items: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.items = items
        self.setFixedHeight(74)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.idx = 0
        self.x = self.spring(0.0, 340, 27)
        self.hov = [self.anim(0, 14) for _ in items]
        self.badges: dict[int, Optional[str]] = {}

    def set_index(self, i: int, emit: bool = False):
        if i == self.idx:
            return
        self.idx = i
        self.x.set(float(i))
        self.kick()
        if emit:
            self.changed.emit(i)

    def set_badge(self, i: int, color: Optional[str]):
        if self.badges.get(i) != color:
            self.badges[i] = color
            self.update()

    def _bar(self) -> QRectF:
        return QRectF(16, 10, self.width() - 32, 54)

    def _centers(self) -> list[float]:
        b = self._bar()
        n = len(self.items)
        step = b.width() / n
        return [b.left() + step * (i + 0.5) for i in range(n)]

    def _hit(self, x: float) -> int:
        b = self._bar()
        n = len(self.items)
        return max(0, min(n - 1, int((x - b.left()) / (b.width() / n))))

    def mouseMoveEvent(self, e):
        i = self._hit(e.position().x())
        for j, a in enumerate(self.hov):
            a.set(1.0 if j == i else 0.0)
        self.setToolTip(self.items[i][1])
        self.kick()

    def hover_changed(self, inside):
        if not inside:
            for a in self.hov:
                a.set(0.0)
            self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.set_index(self._hit(e.position().x()), emit=True)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        b = self._bar()
        p.setBrush(qc(BG1))
        p.setPen(QPen(white(0.06), 1))
        p.drawRoundedRect(b.adjusted(0.5, 0.5, -0.5, -0.5), 26, 26)

        cs = self._centers()
        n = len(cs)
        fx = max(0.0, min(n - 1.0, self.x.value))
        i0 = int(fx)
        i1 = min(n - 1, i0 + 1)
        sx = lerp(cs[i0], cs[i1], fx - i0)
        stretch = min(1.0, abs(self.x.vel) * 0.05)
        pill = QRectF(sx - 25 - 10 * stretch, b.top() + 8, 50 + 20 * stretch, b.height() - 16)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(PRIMARY, 0.16))
        p.drawRoundedRect(pill, pill.height() / 2, pill.height() / 2)

        for i, (icon_name, _tip) in enumerate(self.items):
            sel = max(0.0, 1.0 - abs(fx - i))
            h = self.hov[i].value
            cy = b.center().y()
            s = 21
            col = mix(mix(FG_FAINT, FG_DIM, h), PRIMARY, sel)
            icons.draw(p, icon_name, QRectF(cs[i] - s / 2, cy - s / 2, s, s), col, stroke=1.8)
            badge = self.badges.get(i)
            if badge:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(qc(BG1))
                p.drawEllipse(QPointF(cs[i] + 10, cy - 9), 4.2, 4.2)
                p.setBrush(qc(badge))
                p.drawEllipse(QPointF(cs[i] + 10, cy - 9), 2.8, 2.8)
        p.end()


class PageStack(QWidget):
    """
    Переход между экранами: оба экрана снимаются в картинку и плавно
    разъезжаются со сдвигом и проявлением. Живые виджеты на время
    перехода спрятаны - так анимация ровная на любом железе.
    """
    DURATION = 0.34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pages: list[QWidget] = []
        self.index = 0
        self._old: Optional[QPixmap] = None
        self._new: Optional[QPixmap] = None
        self._dir = 1
        self._t = 1.0

    def add(self, page: QWidget):
        page.setParent(self)
        page.setGeometry(self.rect())
        page.setVisible(not self.pages)
        self.pages.append(page)

    def current(self) -> QWidget:
        return self.pages[self.index]

    def resizeEvent(self, e):
        for p in self.pages:
            p.setGeometry(self.rect())

    def _grab(self, w: QWidget) -> QPixmap:
        dpr = self.devicePixelRatioF()
        pm = QPixmap(int(self.width() * dpr), int(self.height() * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.GlobalColor.transparent)
        w.setGeometry(self.rect())
        if w.layout() is not None:
            w.layout().activate()
        w.render(pm, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
        return pm

    @staticmethod
    def _settle(page: QWidget):
        # Спрятанные виджеты не получают кадров, их анимации стоят на месте.
        # Для снимка доводим всё до конечного состояния, иначе подписи
        # в переходе пустые и появляются рывком после него.
        for w in [page] + page.findChildren(QWidget):
            for a in getattr(w, '_anims', ()):
                a.set(a.target, instant=True)

    def go(self, i: int):
        if i == self.index or not (0 <= i < len(self.pages)):
            return
        old, new = self.pages[self.index], self.pages[i]
        self._old = self._grab(old)
        new.setVisible(True)
        self._settle(new)
        self._new = self._grab(new)
        old.setVisible(False)
        new.setVisible(False)
        self._dir = 1 if i > self.index else -1
        self.index = i
        self._t = 0.0
        clock().wake(self)
        self.update()

    def on_frame(self, dt):
        self._t = min(1.0, self._t + dt / self.DURATION)
        self.update()
        if self._t >= 1.0:
            self._old = self._new = None
            self.pages[self.index].setVisible(True)
            return False
        return True

    def paintEvent(self, _):
        if self._old is None or self._new is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        e = ease_out(self._t)
        w = self.width()
        shift = w * 0.22
        p.setOpacity(max(0.0, 1 - e * 1.4))
        p.drawPixmap(QPointF(-self._dir * shift * e, 0), self._old)
        p.setOpacity(min(1.0, e * 1.2))
        p.drawPixmap(QPointF(self._dir * shift * (1 - e), 0), self._new)
        p.end()


# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================

class ModelLoader(QThread):
    def __init__(self, engine: TranscriberEngine):
        super().__init__()
        self.engine = engine

    def run(self):
        self.engine.load_models()


# ============================================================
# ОКНО
# ============================================================

class MainWindow(QWidget):
    HOME, VOICES, DISCORD, OVERLAY, HISTORY, TUNE = range(6)
    TITLES = ['Voice Assist', 'Голоса', 'Discord', 'Субтитры', 'История', 'Отклик']
    QUIET = {'Готов', 'Слушаю', 'Остановлено'}

    def __init__(self):
        super().__init__()
        self.config: AppConfig = load_config()
        self.engine = TranscriberEngine(mic_speaker_name=self.config.mic_speaker_name)
        self.engine.use_gpu = self.config.use_gpu
        self._gpu_lost_told = False
        import asr_parakeet
        import translate_onnx
        self.engine.multilang = self.config.english and asr_parakeet.available()
        view = self.config.translate_view
        self.engine.view = view if view in ('orig', 'both', 'trans') and translate_onnx.available() else 'orig'
        self.setAcceptDrops(True)       # файлы для расшифровки - перетаскиванием
        self.bridge = DiscordBridge(self.engine, self.config)
        self.loader: Optional[ModelLoader] = None
        self.overlay_win = None
        self._mic: Optional[audio_source.AudioInput] = None
        self._sys: Optional[audio_source.AudioInput] = None
        self._levels: dict[str, float] = {}
        self._avatars: dict[str, QPixmap] = {}
        self._net = QNetworkAccessManager(self)
        self._quitting = False

        self.setWindowTitle(f'Voice Assist {version.LABEL}')
        self.setWindowIcon(app_icon())
        # без MinimizeButtonHint рамочное окно не сворачивается кликом по панели задач
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
                            | Qt.WindowType.WindowMinimizeButtonHint)
        self.setFixedSize(W, H)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)


        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.titlebar = TitleBar(self)
        root.addWidget(self.titlebar)
        self.stack = PageStack()
        root.addWidget(self.stack, 1)
        self.dock = NavDock([
            ('waves', 'Главная'),
            ('users', 'Голоса'),
            ('headphones', 'Discord'),
            ('captions', 'Субтитры'),
            ('history', 'История'),
            ('sliders', 'Отклик'),
        ])
        self.dock.changed.connect(self.navigate)
        root.addWidget(self.dock)

        self.home = HomePage()
        self.voices = VoicesPage(self)
        self.discord = DiscordPage(self)
        self.tune = TunePage(self)
        self.history = HistoryPage(self)
        self.engine.history.enabled = self.config.history_enabled
        self.stack.add(self.home)
        self.stack.add(self.voices)
        self.stack.add(self.discord)
        self._overlay_slot = QWidget()
        self.stack.add(self._overlay_slot)
        self.stack.add(self.history)
        self.stack.add(self.tune)
        self.titlebar.title.set_text(self.TITLES[0])

        self.toast_w = Toast(self)

        # ---- сигналы ----
        h = self.home
        h.orb.clicked.connect(self.on_power)
        h.orb.long_pressed.connect(self.on_power_hold)
        h.mic_chip.clicked.connect(lambda: self._pick_source('mic', h.mic_chip))
        h.sys_chip.clicked.connect(lambda: self._pick_source('sys', h.sys_chip))
        h.bot_chip.clicked.connect(lambda: self.navigate(self.DISCORD))

        self.engine.state_changed.connect(self._on_engine_state)
        self.engine.load_progress.connect(self._on_load_progress)
        self.engine.error_occurred.connect(lambda m: self.toast(m, 'error'))
        self.engine.device_changed.connect(self._on_device)
        self.engine.mode_changed.connect(self._on_mode)
        self.engine.status_message.connect(self._on_status)

        br = self.bridge
        br.state_changed.connect(self._on_bridge_state)
        br.channel_changed.connect(self._on_channel)
        br.members_changed.connect(self._on_members)
        br.speaking_changed.connect(self._on_speaking)
        br.follow_changed.connect(self._on_follow)
        br.bot_ready.connect(self._on_bot_ready)
        br.guilds_changed.connect(lambda _n: self.discord.refresh())

        self._restore_sources()
        self._on_engine_state(self.engine.state)

        self._lvl_timer = QTimer(self)
        self._lvl_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._lvl_timer.timeout.connect(self._poll_levels)
        self._lvl_timer.start(33)
        self._avail_timer = QTimer(self)
        self._avail_timer.timeout.connect(self._refresh_availability)
        self._avail_timer.start(1500)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(lambda: self.tune.update_stats(self.engine.stats())
                                          if self.stack.index == self.TUNE else None)
        self._stats_timer.start(500)

        if secret_store.load_token() and self.config.discord_bot_enabled:
            QTimer.singleShot(400, self.bridge.start)
        self._update_bot_chip()
        self.load_models()

    # ---------- рамка Windows 11 ----------

    def showEvent(self, e):
        super().showEvent(e)
        if sys.platform == 'win32':
            try:
                hwnd = int(self.winId())
                dwm = ctypes.windll.dwmapi
                for attr, val in ((33, 2), (20, 1), (34, 0x00332C2A)):
                    v = ctypes.c_int(val)
                    dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
            except Exception:
                pass

    def changeEvent(self, e):
        super().changeEvent(e)
        # в свёрнутом окне анимации стоят - при разворачивании будим их
        if e.type() == QEvent.Type.WindowStateChange and not self.isMinimized():
            for w in self.findChildren(AWidget):
                w.kick()

    def paintEvent(self, _):
        p = QPainter(self)
        paint_window_bg(p, QRectF(self.rect()))
        p.end()

    # ---------- публичное для экранов ----------

    def toast(self, text: str, kind: str = 'info'):
        self.toast_w.push(text, kind)

    def save(self):
        try:
            save_config(self.config)
        except Exception as e:
            print(f'save config: {e}', file=sys.stderr)

    def current_sys(self):
        return self._sys

    def current_mic(self):
        return self._mic

    def load_models(self):
        if self.engine.state in (EngineState.LOADING, EngineState.READY, EngineState.RUNNING):
            return
        self.loader = ModelLoader(self.engine)
        self.loader.start()

    def set_mic_name(self, name: str):
        if name != self.config.mic_speaker_name:
            self.config.mic_speaker_name = name
            self.engine.mic_speaker_name = name
            self.save()

    # ---------- файлы ----------

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            e.acceptProposedAction()
            self.navigate(self.HISTORY)
            self.history.add_files(paths)

    def set_language(self, english: bool):
        self.config.english = bool(english)
        self.save()
        self.engine.set_language(english)

    def set_view(self, view: str):
        self.config.translate_view = view
        self.save()
        self.engine.set_view(view)
        self._apply_view()

    def _apply_view(self):
        if self.overlay_win is not None:
            self.overlay_win.set_split_segments(self.engine.multilang)
            self.overlay_win.set_view(self.engine.view if self.engine.multilang else 'orig')

    def _on_mode(self, english: bool, error: str):
        self.tune.set_language(english, busy=False)
        self.tune.set_view(self.engine.view)
        self._apply_view()
        if error:
            # не вышло - не пробуем при каждом запуске
            self.config.english = english
            self.config.translate_view = self.engine.view
            self.save()
            self.toast('Английская модель не загрузилась' if not english else 'Перевод не загрузился',
                       'error')

    def set_gpu(self, on: bool):
        self.config.use_gpu = bool(on)
        self.save()
        self.engine.set_gpu(on)

    def _on_device(self, gpu: bool, error: str):
        self.tune.set_device(gpu, busy=False)
        if error == 'lost':
            # сброс драйвера на ходу: видеокарта исправна, вернётся после перезапуска
            if not self._gpu_lost_told:
                self._gpu_lost_told = True
                self.toast('Видеокарта сбросилась. До перезапуска программы распознаёт процессор', 'error')
        elif error and self.config.use_gpu:
            # не вышло - остаёмся на процессоре и не пробуем при каждом запуске
            self.config.use_gpu = False
            self.engine.use_gpu = False
            self.save()
            self.toast('Видеокарта не подошла, распознаёт процессор', 'error')

    def set_preset(self, key: str):
        self.config.recognition_preset = key
        self.engine.apply_recognition_preset(key)
        if self.overlay_win is not None:
            self.overlay_win.set_fast_reveal(key == 'fast')
        self.save()

    def save_overlay_geometry(self, x, y, w, h):
        self.config.overlay_x, self.config.overlay_y = x, y
        self.config.overlay_width, self.config.overlay_height = w, h
        self.save()

    def open_log(self):
        p = log_path()
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _update_skip_self(self):
        # Свой голос бот не берёт: его либо не нужно (субтитры - про других),
        # либо он уже идёт с микрофона. Включить можно переключателем «Мой голос».
        self.bridge.set_skip_self(self._mic is not None or not self.config.discord_hear_me)

    def set_hear_me(self, on: bool):
        self.config.discord_hear_me = bool(on)
        self.save()
        self._update_skip_self()

    def set_bot_enabled(self, on: bool):
        self.config.discord_bot_enabled = bool(on)
        self.save()
        if on:
            self.bridge.start()
        else:
            self.bridge.stop()
        self.discord.refresh()
        self._update_bot_chip()

    def reset_follow(self):
        self.bridge.set_follow(None)
        self.save()
        self.discord.refresh()

    def pick_system_source(self, anchor: QWidget, after=None):
        self._pick_source('sys', anchor, after)

    def set_overlay(self, overlay_win):
        self.overlay_win = overlay_win
        c = self.config
        overlay_win.apply_geometry(c.overlay_x, c.overlay_y, c.overlay_width, c.overlay_height)
        overlay_win.set_font_size(c.overlay_font_size)
        overlay_win.set_ttl_seconds(c.overlay_ttl_sec)
        overlay_win.set_bg_opacity(c.overlay_bg_opacity / 100)
        overlay_win.set_align(c.overlay_align)
        overlay_win.set_max_lines(c.overlay_max_lines)
        overlay_win.set_hover_opacity(c.overlay_hover_opacity / 100)
        overlay_win.set_fast_reveal(c.recognition_preset == 'fast')
        overlay_win.set_split_segments(self.engine.multilang)
        overlay_win.set_view(self.engine.view if self.engine.multilang else 'orig')
        overlay_win.level_fn = lambda s: self._levels.get(s, 0.5)
        self.overlay_page = OverlayPage(self, overlay_win)
        idx = self.stack.pages.index(self._overlay_slot)
        self._overlay_slot.deleteLater()
        self.overlay_page.setParent(self.stack)
        self.overlay_page.setGeometry(self.stack.rect())
        self.overlay_page.setVisible(self.stack.index == idx)
        self.stack.pages[idx] = self.overlay_page

    # ---------- навигация ----------

    def navigate(self, i: int):
        if self.stack.index == self.OVERLAY and i != self.OVERLAY and hasattr(self, 'overlay_page'):
            self.overlay_page.leave()
        if self.stack.index == self.HISTORY and i != self.HISTORY:
            self.history.leave()
        # обновляем до перехода: в снимок должно попасть свежее состояние
        if i == self.VOICES:
            self.voices.refresh()
        elif i == self.DISCORD:
            self.discord.refresh()
        elif i == self.TUNE:
            self.tune.update_stats(self.engine.stats())
        elif i == self.HISTORY:
            self.history.refresh()
        self.dock.set_index(i)
        self.stack.go(i)
        self.titlebar.title.set_text(self.TITLES[i])

    # ---------- источники ----------

    def _restore_sources(self):
        c = self.config
        try:
            mic_list = audio_source.list_microphones()
        except Exception:
            mic_list = []
        try:
            apps = {s.id: s for s in audio_source.list_system_audio()}
        except Exception:
            apps = {}
        if c.mic_source_id:
            self._mic = audio_source.AudioInput(c.mic_source_id, c.mic_source_name or c.mic_source_id, '')
            self._follow_mic(mic_list)
        if c.sys_source_id:
            self._sys = apps.get(c.sys_source_id) or audio_source.AudioInput(
                c.sys_source_id, c.sys_source_name or c.sys_source_id, '')
        self._update_skip_self()
        self._apply_chips({m.id for m in mic_list}, apps.keys())

    def _follow_mic(self, mic_list):
        """Номер устройства сменился, а имя то же - переезжаем на новый номер."""
        if self._mic is None:
            return
        found = audio_source.resolve_microphone(self._mic.id, self.config.mic_source_name, mic_list)
        if found is not None and found.id != self._mic.id:
            print(f'Микрофон «{found.name}» переехал: {self._mic.id} -> {found.id}')
            self._mic = found
            self.config.mic_source_id = found.id
            self.save()
            if self.engine.state == EngineState.RUNNING:
                self.engine.change_microphone(found.id)
        elif found is not None:
            self._mic = found

    def _apply_chips(self, mic_ids, app_ids):
        h = self.home
        if self._mic is None:
            h.mic_chip.set_status('empty')
        else:
            h.mic_chip.set_status('ok' if self._mic.id in mic_ids else 'waiting', self._short(self._mic.name))
        if self._sys is None:
            h.sys_chip.set_status('empty')
        else:
            h.sys_chip.set_status('ok' if self._sys.id in app_ids else 'waiting', self._sys.name)

    @staticmethod
    def _short(name: str) -> str:
        # «Микрофон (Fifine Microphone)» -> «Fifine Microphone»
        if '(' in name and name.endswith(')'):
            inner = name[name.index('(') + 1:-1].strip()
            if inner:
                return inner
        return name

    def _refresh_availability(self):
        if not self.isVisible() or self.isMinimized():
            return
        try:
            mic_list = audio_source.list_microphones()
            app_ids = {s.id for s in audio_source.list_system_audio()}
        except Exception:
            return
        self._follow_mic(mic_list)
        self._apply_chips({m.id for m in mic_list}, app_ids)

    def _pick_source(self, kind: str, anchor: QWidget, after=None):
        try:
            items = audio_source.list_microphones() if kind == 'mic' else audio_source.list_system_audio()
        except Exception as e:
            items = []
            print(f'list sources: {e}', file=sys.stderr)
        cur = self._mic if kind == 'mic' else self._sys
        rows = [{'id': None, 'text': 'Не слушать', 'icon': 'x', 'data': None}]
        for it in items:
            rows.append({
                'id': it.id,
                'text': self._short(it.name) if kind == 'mic' else it.name,
                'icon': 'mic' if kind == 'mic' else 'speaker',
                'data': it,
            })
        if cur is not None and cur.id not in {it.id for it in items}:
            rows.append({'id': cur.id, 'text': cur.name, 'sub': 'сейчас не запущено',
                         'icon': 'alert', 'data': cur})
        if len(rows) == 1:
            rows.append({'id': '__none', 'text': 'Нет звучащих приложений' if kind == 'sys'
                         else 'Микрофонов не найдено', 'icon': 'help', 'disabled': True})
        # Родитель и ссылка обязательны: без них Python удалял окно списка
        # сразу после показа, и выбрать источник было нельзя.
        pop = Popover(rows, cur.id if cur else None, width=300, parent=self)
        self._popover = pop

        def picked(row):
            self._set_source(kind, row.get('data'))
            if after:
                after()

        pop.picked.connect(picked)
        panel_w = pop.width() - 2 * Popover.MARGIN
        panel_h = pop.height() - 2 * Popover.MARGIN
        top = anchor.mapToGlobal(QPoint(anchor.width() // 2, 0))
        pop.popup_at(QPoint(top.x() - panel_w // 2, top.y() - panel_h - 6))

    def _set_source(self, kind: str, item):
        c = self.config
        if kind == 'mic':
            self._mic = item
            c.mic_source_id = item.id if item else None
            c.mic_source_name = item.name if item else None
            self._update_skip_self()
            if self.engine.state == EngineState.RUNNING:
                self.engine.change_microphone(c.mic_source_id)
        else:
            self._sys = item
            c.sys_source_id = item.id if item else None
            c.sys_source_name = item.name if item else None
            if self.engine.state == EngineState.RUNNING:
                self.engine.change_system_audio(c.sys_source_id)
        self.save()
        self._refresh_availability()

    # ---------- кнопка ----------

    def on_power(self):
        st = self.engine.state
        if st in (EngineState.IDLE, EngineState.ERROR):
            self.load_models()
        elif st == EngineState.LOADING:
            self.toast('Модели ещё загружаются', 'info')
        elif st == EngineState.READY:
            bot = self.bridge.running()
            if not self._mic and not self._sys and not bot:
                self.toast('Выберите, кого слушать', 'warn')
                for i, chip in enumerate((self.home.mic_chip, self.home.sys_chip, self.home.bot_chip)):
                    QTimer.singleShot(i * 70, chip.nudge)
                return
            self.engine.start(self._mic.id if self._mic else None,
                              self._sys.id if self._sys else None,
                              allow_no_sources=bot)
        elif st == EngineState.RUNNING:
            self.engine.stop()

    def on_power_hold(self):
        if self.engine.state in (EngineState.RUNNING, EngineState.READY, EngineState.ERROR):
            self.engine.unload_models()
            self.toast('Модели выгружены из памяти', 'info')

    # ---------- движок ----------

    def _on_engine_state(self, state: EngineState):
        self.home.orb.set_state(state)
        words = {
            EngineState.IDLE: ('Выключено', FG_DIM),
            EngineState.LOADING: ('Загрузка', FG_DIM),
            EngineState.READY: ('Готов', FG),
            EngineState.RUNNING: ('Слушаю', ACCENT),
            EngineState.ERROR: ('Ошибка', RED),
        }
        text, col = words.get(state, ('', FG_DIM))
        self.home.status.set_text(text, col)
        if state == EngineState.READY:
            self.engine.apply_recognition_preset(self.config.recognition_preset)
        running = state == EngineState.RUNNING
        self.home.mic_chip.set_active(running and self._mic is not None)
        self.home.sys_chip.set_active(running and self._sys is not None)
        self._update_bot_chip()
        if hasattr(self, 'tray'):
            self.tray.on_state(state)

    def _on_load_progress(self, step: str, done: int, total: int):
        self.home.orb.set_progress(done, total)
        self.home.status.set_text(f'Загрузка  {done}/{total}', FG_DIM)

    def _on_status(self, msg: str):
        if msg not in self.QUIET:
            self.toast(msg, 'ok' if 'подключ' in msg.lower() else 'info')

    def _poll_levels(self):
        if self.engine.state != EngineState.RUNNING:
            if self._levels:
                self._levels = {}
                h = self.home
                h.orb.set_levels(0, 0)
                for c in (h.mic_chip, h.sys_chip, h.bot_chip):
                    c.set_level(0)
            return
        lv = self.engine.levels()
        self._levels = lv
        bot = max((v for k, v in lv.items() if k.startswith('dc:')), default=0.0)
        mon = lv.get('mon', 0.0)
        mic = lv.get('mic', 0.0)
        h = self.home
        if self.stack.index == self.HOME and self.isVisible():
            h.orb.set_levels(max(mon, bot), mic)
            h.mic_chip.set_level(mic)
            h.sys_chip.set_level(mon)
            h.bot_chip.set_level(bot)

    # ---------- Discord ----------

    def _update_bot_chip(self):
        br = self.bridge
        chip = self.home.bot_chip
        has_token = bool(secret_store.load_token())
        if not has_token or not self.config.discord_bot_enabled:
            chip.set_status('empty')
            chip.set_active(False)
            self.dock.set_badge(self.DISCORD, None)
            return
        st = br.state
        if st == 'in_voice' and br.channel:
            chip.set_status('ok', '# ' + br.channel.get('channel', {}).get('name', ''))
            self.dock.set_badge(self.DISCORD, ACCENT)
        elif st == 'online':
            chip.set_status('ok', 'Онлайн')
            self.dock.set_badge(self.DISCORD, None)
        elif st in ('connecting', 'joining'):
            chip.set_status('waiting', 'Подключение')
            self.dock.set_badge(self.DISCORD, None)
        else:
            chip.set_status('waiting', 'Ошибка')
            self.dock.set_badge(self.DISCORD, RED)
        chip.set_active(self.engine.state == EngineState.RUNNING and st == 'in_voice')

    def _on_bridge_state(self, state: str, detail: str):
        self._update_bot_chip()
        self.discord.refresh()
        if state == 'bad-token':
            self.toast('Discord не принял токен', 'error')
        elif state == 'error' and detail in ('no-node', 'old-node', 'no-modules', 'no-bridge'):
            self.toast(self.discord._error_text(detail), 'error')

    def _on_channel(self, ch):
        self.home.members.setVisible(bool(ch))
        self._update_bot_chip()
        self.discord.refresh()
        if ch:
            self.toast(f'Бот в канале # {ch.get("channel", {}).get("name", "")}', 'ok')

    def _on_members(self, members: list):
        rows = []
        for m in members:
            row = {'id': m['id'], 'name': m['name'], 'image': self._avatars.get(m.get('avatar') or '')}
            rows.append(row)
            if m.get('avatar') and m['avatar'] not in self._avatars:
                self._load_avatar(m['avatar'], m['id'])
        self.home.members.set_members(rows)
        self.discord.members.set_members([dict(r) for r in rows])
        self.discord._refresh_follow()

    def _on_speaking(self, uid: str, active: bool):
        self.home.members.set_speaking(uid, active)
        self.discord.members.set_speaking(uid, active)
        self.discord.hero.set_speaking(bool(self.bridge.speaking))

    def _on_follow(self, uid: str, name: str):
        self.save()
        self.discord.refresh()
        if name:
            self.toast(f'Хожу за {name}', 'ok')

    def _on_bot_ready(self, bot: dict):
        self.discord.refresh()
        url = bot.get('avatar')
        if url:
            self._load_avatar(url, '__bot__')

    def _load_avatar(self, url: str, uid: str):
        req = QNetworkRequest(QUrl(url))
        reply = self._net.get(req)

        def done():
            data = reply.readAll()
            reply.deleteLater()
            pm = QPixmap()
            if not pm.loadFromData(bytes(data)):
                return
            self._avatars[url] = pm
            if uid == '__bot__':
                self.discord.hero.image = pm
                self.discord.hero.update()
                return
            self.home.members.set_image(uid, pm)
            self.discord.members.set_image(uid, pm)
            m = self.bridge.members.get(uid)
            if m is not None:
                m['image'] = pm
            self.discord._refresh_follow()

        reply.finished.connect(done)

    # ---------- выход ----------

    def closeEvent(self, e):
        self.shutdown()
        e.accept()
        QApplication.quit()

    def shutdown(self):
        if self._quitting:
            return
        self._quitting = True
        try:
            if self.engine.state == EngineState.RUNNING:
                self.engine.stop()
        except Exception:
            pass
        try:
            self.bridge.stop()
        except Exception:
            pass
        loader = self.loader
        if loader is not None and loader.isRunning():
            loader.wait(5000)


# ============================================================
# ТРЕЙ
# ============================================================

class Tray:
    def __init__(self, win: MainWindow, app: QApplication):
        self.win = win
        self.app = app
        self.tray = QSystemTrayIcon(app_icon(gray=True), parent=win)
        self.tray.setToolTip(f'Voice Assist {version.LABEL}')
        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{ background: {BG2}; color: {FG}; border: 1px solid #2A303C;
                     border-radius: 10px; padding: 6px; }}
            QMenu::item {{ padding: 7px 22px 7px 12px; border-radius: 6px; }}
            QMenu::item:selected {{ background: #262C38; }}
            QMenu::icon {{ padding-left: 8px; }}
            QMenu::separator {{ height: 1px; background: #2A303C; margin: 4px 8px; }}
        """)
        self.show_act = QAction(icons.icon('expand', FG_DIM), 'Показать', menu)
        self.show_act.triggered.connect(self._show)
        menu.addAction(self.show_act)
        self.toggle_act = QAction(icons.icon('play', FG), 'Слушать', menu)
        self.toggle_act.triggered.connect(win.on_power)
        menu.addAction(self.toggle_act)
        menu.addSeparator()
        quit_act = QAction(icons.icon('x', RED), 'Выход', menu)
        quit_act.triggered.connect(self._quit)
        menu.addAction(quit_act)
        self.menu = menu
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._activated)
        win.tray = self
        self.on_state(win.engine.state)
        self.tray.show()

    def _show(self):
        self.win.showNormal()
        self.win.raise_()
        self.win.activateWindow()

    def _activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.win.isVisible() and not self.win.isMinimized():
                self.win.hide()
            else:
                self._show()

    def on_state(self, state: EngineState):
        tint = {
            EngineState.RUNNING: ACCENT, EngineState.READY: None,
            EngineState.ERROR: RED, EngineState.LOADING: None,
        }.get(state)
        gray = state in (EngineState.IDLE, EngineState.LOADING)
        self.tray.setIcon(app_icon(tint, gray))
        running = state == EngineState.RUNNING
        self.toggle_act.setText('Остановить' if running else 'Слушать')
        self.toggle_act.setIcon(icons.icon('stop' if running else 'play', FG))
        self.toggle_act.setEnabled(state in (EngineState.READY, EngineState.RUNNING, EngineState.ERROR, EngineState.IDLE))

    def _quit(self):
        self.win.shutdown()
        self.app.quit()
