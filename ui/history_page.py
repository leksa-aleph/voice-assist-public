"""
Экран «История»: сохранённые расшифровки.

Сверху - переключатель «Сохранять». Ниже сеансы по дням, новые сверху:
участники аватарами, начало разговора, время и длительность; идущий
сейчас отмечен мятной точкой на аватаре. Нажатие открывает расшифровку -
как переписку: аватар, имя цветом человека, время плашкой, ниже текст.
Новые фразы дописываются в конец и не сбивают место, где вы читаете;
если были внизу - лента едет за ними, если выше - снизу появляется
кнопка со счётчиком новых.
"""
from __future__ import annotations

import datetime
import os
import time
from typing import Optional

from PyQt6.QtCore import QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QFont, QFontMetricsF, QGuiApplication, QLinearGradient, QPainter, QPen, QTextBlockFormat,
    QTextCharFormat, QTextCursor,
)
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QScrollArea, QTextBrowser, QVBoxLayout, QWidget

import file_transcribe
import history
from ui import icons
from ui.theme import (
    ACCENT, BG1, FG, FG_DIM, FG_FAINT, PRIMARY, font, mix, plural, qc, speaker_color, white,
)
from ui.widgets import (
    AA, AWidget, IconButton, Row, RowCard, Surface, TextField, Toggle, elide, paint_avatar,
    paint_window_bg,
)

MONTHS = ['янв', 'фев', 'мар', 'апр', 'мая', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
MONTHS_FULL = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа',
               'сентября', 'октября', 'ноября', 'декабря']

SCROLL_CSS = '''
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: rgba(255,255,255,0.10); border-radius: 3px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: rgba(255,255,255,0.18); }
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
    background: none; height: 0; }
'''


def _duration(sec: float) -> str:
    m = int(round(sec / 60))
    if m < 1:
        return '<1 мин'
    if m < 60:
        return f'{m} мин'
    return f'{m // 60} ч {m % 60:02d} мин'


def day_label(ts: float, now: Optional[float] = None) -> str:
    d = datetime.date.fromtimestamp(ts)
    today = datetime.date.fromtimestamp(time.time() if now is None else now)
    days = (today - d).days
    if days == 0:
        return 'Сегодня'
    if days == 1:
        return 'Вчера'
    s = f'{d.day} {MONTHS_FULL[d.month - 1]}'
    return s if d.year == today.year else f'{s} {d.year}'


def _named(speakers) -> list[str]:
    return [n for n in speakers if n and n != '?']


# ============================================================
# СПИСОК СЕАНСОВ
# ============================================================

class SessionRow(AWidget):
    opened = pyqtSignal(str)

    def __init__(self, s: history.Session, live: bool, parent=None):
        super().__init__(parent)
        self.s = s
        self.live = live
        self.setFixedHeight(64)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.h = self.anim(0, 14)
        self._time = time.strftime('%H:%M', time.localtime(s.started))
        self._named = _named(s.speakers)
        self.setToolTip(f'{s.lines} {plural(s.lines, "фраза", "фразы", "фраз")}')

    def hover_changed(self, inside):
        self.h.set(1.0 if inside else 0.0)
        self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.opened.emit(self.s.id)

    def _paint_avatars(self, p: QPainter, c: QPointF, bg):
        names = self._named
        if self.s.file:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(PRIMARY, 0.14))
            p.drawEllipse(c, 19, 19)
            icons.draw(p, 'file', QRectF(c.x() - 9, c.y() - 9, 18, 18), qc(PRIMARY), stroke=2.0)
            job = history.jobs.get(self.s.id)
            if job is not None:
                # файл ещё считается - кольцо прогресса
                r = QRectF(c.x() - 21, c.y() - 21, 42, 42)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(white(0.08), 2.4))
                p.drawEllipse(r)
                pen = QPen(qc(ACCENT), 2.4)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                p.drawArc(r, 90 * 16, -int(360 * 16 * max(0.02, job.progress)))
            return
        if not names:
            # имён нет - нейтральный круг со звуковой волной
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(PRIMARY, 0.14))
            p.drawEllipse(c, 19, 19)
            icons.draw(p, 'waves', QRectF(c.x() - 9, c.y() - 9, 18, 18), qc(PRIMARY), stroke=2.0)
        elif len(names) == 1:
            paint_avatar(p, c, 19, names[0])
        else:
            # двое и больше - стопкой, у второго вырез цветом фона
            paint_avatar(p, QPointF(c.x() - 8, c.y() - 8), 12, names[0])
            c2 = QPointF(c.x() + 8, c.y() + 8)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawEllipse(c2, 14.5, 14.5)
            if len(names) == 2:
                paint_avatar(p, c2, 12, names[1])
            else:
                p.setBrush(qc('#272A37'))
                p.setPen(QPen(white(0.07), 1))
                p.drawEllipse(c2, 11.5, 11.5)
                p.setPen(qc(FG_DIM))
                p.setFont(font(10, QFont.Weight.DemiBold))
                p.drawText(QRectF(c2.x() - 12, c2.y() - 12, 24, 24), Qt.AlignmentFlag.AlignCenter,
                           f'+{len(names) - 1}')
        if self.live:
            # идёт запись - мятная точка, как «в сети»
            d = QPointF(c.x() + 15, c.y() + 15)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(bg)
            p.drawEllipse(d, 6.5, 6.5)
            p.setBrush(qc(ACCENT))
            p.drawEllipse(d, 4.5, 4.5)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        h = self.h.value
        bg = mix(qc(BG1), white(1.0), 0.03 * h)
        if h > 0.01:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(white(0.03 * h))
            p.drawRoundedRect(QRectF(self.rect()).adjusted(4, 2, -4, -2), 12, 12)
        self._paint_avatars(p, QPointF(34, 32), bg)

        x = 68
        right = self.width() - 16
        ft = font(12, QFont.Weight.Medium)
        p.setFont(ft)
        job = history.jobs.get(self.s.id)
        corner = f'{int(job.progress * 100)}%' if job is not None else self._time
        tw = QFontMetricsF(ft).horizontalAdvance(corner)
        p.setPen(qc(ACCENT) if self.live or job is not None else qc(FG_FAINT))
        p.drawText(QRectF(right - tw, 11, tw + 1, 22),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, corner)

        f1 = font(14, QFont.Weight.Medium)
        title = self.s.title or self.s.preview or self._time
        p.setFont(f1)
        p.setPen(qc(FG))
        w1 = right - x - tw - 12
        p.drawText(QRectF(x, 11, w1, 22), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   elide(title, f1, w1))

        f2 = font(12.5)
        sub = _duration(self.s.duration)
        if self._named:
            sub += '  ·  ' + ', '.join(self._named)
        p.setFont(f2)
        p.setPen(qc(FG_DIM))
        w2 = right - x - 20
        p.drawText(QRectF(x, 33, w2, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   elide(sub, f2, w2))
        icons.draw(p, 'chevron-right', QRectF(right - 13, 36, 14, 14), mix(FG_FAINT, FG_DIM, h))
        p.end()


class DayHeader(QWidget):
    def __init__(self, text: str, first: bool, parent=None):
        super().__init__(parent)
        self.text = text
        self.setFixedHeight(26 if first else 38)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        p.setFont(font(12.5, QFont.Weight.DemiBold))
        p.setPen(qc(FG_DIM))
        p.drawText(QRectF(8, 0, self.width() - 8, self.height() - 7),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, self.text)
        p.end()


class SessionList(QWidget):
    """Сеансы по дням: подпись дня и карточка со строками."""
    opened = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(0)
        self.lay.addStretch(1)
        self._items: list[QWidget] = []
        self._rows: dict[str, SessionRow] = {}

    def update_sessions(self, sessions: list):
        """Те же сеансы, поменялись только цифры - без перестройки."""
        for s in sessions:
            row = self._rows.get(s.id)
            if row is not None:
                row.s = s
                row.update()

    def set_sessions(self, sessions: list, live_id: Optional[str]):
        for w in self._items:
            self.lay.removeWidget(w)
            w.deleteLater()
        self._items = []
        self._rows = {}
        days: list[tuple[str, list]] = []
        for s in sessions:
            label = day_label(s.started)
            if not days or days[-1][0] != label:
                days.append((label, []))
            days[-1][1].append(s)
        for i, (label, items) in enumerate(days):
            head = DayHeader(label, i == 0)
            card = RowCard()
            for s in items:
                row = SessionRow(s, s.id == live_id)
                row.opened.connect(self.opened)
                card.add(row)
                self._rows[s.id] = row
            for w in (head, card):
                self.lay.insertWidget(self.lay.count() - 1, w)
                self._items.append(w)


class EmptyState(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        c = QPointF(self.width() / 2, self.height() / 2 - 16)
        p.setPen(QPen(white(0.06), 1))
        p.setBrush(qc(BG1))
        p.drawEllipse(c, 38, 38)
        icons.draw(p, 'history', QRectF(c.x() - 17, c.y() - 17, 34, 34), FG_FAINT, stroke=1.6)
        p.setFont(font(13))
        p.setPen(qc(FG_FAINT))
        p.drawText(QRectF(0, c.y() + 50, self.width(), 20), Qt.AlignmentFlag.AlignCenter, 'Пока пусто')
        p.end()


# ============================================================
# РАСШИФРОВКА
# ============================================================

AVATAR_R = 15
GUTTER = 2 * AVATAR_R + 13       # место под аватар слева от текста
PILL_PAD = 6                     # поля плашки со временем
FADE = 14                        # затухание текста у краёв карточки


def _x(v) -> float:
    return v[0] if isinstance(v, tuple) else v


class Transcript(QTextBrowser):
    """
    Текст сеанса. Имена и время - часть текста (копируются вместе с ним),
    аватары и плашки под временем дорисовываются только для видимых строк.
    Документ только растёт с конца, поэтому прокрутка не сбивается.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setOpenLinks(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet(
            f'QTextBrowser {{ background: transparent; border: none; color: {FG};'
            f' selection-background-color: #3B4170; selection-color: {FG}; }}' + SCROLL_CSS)
        self.viewport().setAutoFillBackground(False)
        self.document().setDocumentMargin(16)
        self._f_text = QTextCharFormat()
        self._f_text.setFont(font(14))
        self._f_text.setForeground(qc(FG))
        tf = font(11.5, QFont.Weight.DemiBold)
        self._f_time = QTextCharFormat()
        self._f_time.setFont(tf)
        self._f_time.setForeground(qc(FG_DIM))
        self._time_cap = QFontMetricsF(tf).capHeight()
        self._f_tr = QTextCharFormat()        # перевод - тем же цветом, что в субтитрах
        self._f_tr.setFont(font(14))
        self._f_tr.setForeground(qc('#B2BEFF'))
        self.reset(False)

    def reset(self, avatars: bool, base: Optional[float] = None):
        self.document().clear()
        self.base = base            # у файла время - место в записи
        self.avatars = avatars
        self._gutter = GUTTER if avatars else 0
        self._meta: list[tuple[str, int, int]] = []    # заголовки: (кто, начало времени, длина)
        self._group = history.Grouper()
        self._after_tr = False
        self._empty = True
        self.phrases = 0

    def park_cursor(self):
        """Курсор виджета - в начало, не трогая прокрутку. Иначе он стоит
        в конце, едет вместе с дописанным текстом и тянет экран за собой."""
        bar = self.verticalScrollBar()
        v = bar.value()
        self.setTextCursor(QTextCursor(self.document()))
        bar.setValue(v)

    def append_lines(self, lines: list[dict]) -> int:
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.MoveOperation.End)
        cur.beginEditBlock()
        n = 0
        for ln in lines:
            text = (ln.get('text') or '').strip()
            if not text:
                continue
            who = ln.get('who') or '?'
            t = float(ln.get('t', 0))
            tr = (ln.get('tr') or '').strip()
            if self._group.feed(who, t):
                self._header(cur, who, t)
                self._para(cur)
                cur.insertText(text, self._f_text)
            elif self._after_tr or tr:
                # фраза с переводом и фраза после перевода - новым абзацем:
                # оригинал - перевод - оригинал
                self._para(cur, top=6)
                cur.insertText(text, self._f_text)
            else:
                cur.insertText(' ' + text, self._f_text)
            self._after_tr = False
            if tr:
                self._para(cur, top=2)
                cur.insertText(tr, self._f_tr)
                self._after_tr = True
            n += 1
        cur.endEditBlock()
        self.phrases += n
        return n

    def _para(self, cur: QTextCursor, top: int = 0):
        bf = QTextBlockFormat()
        bf.setLeftMargin(self._gutter)
        bf.setTopMargin(top)
        bf.setLineHeight(145.0, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
        cur.insertBlock(bf, self._f_text)

    def _header(self, cur: QTextCursor, who: str, t: float):
        named = who != '?'
        bf = QTextBlockFormat()
        bf.setTopMargin(0 if self._empty else 20)
        bf.setBottomMargin(3)
        # без имени время стоит первым: плашке нужно поле слева
        bf.setLeftMargin(self._gutter + (0 if named else PILL_PAD))
        if self._empty:
            cur.setBlockFormat(bf)
            self._empty = False
        else:
            cur.insertBlock(bf)
        stamp = history.stamp(t, self.base)
        off = 0
        if named:
            nf = QTextCharFormat()
            nf.setFont(font(14, QFont.Weight.DemiBold))
            nf.setForeground(qc(speaker_color(who)))
            cur.insertText(who, nf)
            cur.insertText(' ', nf)
            off = len(who) + 1
        cur.insertText(stamp, self._f_time)
        cur.block().setUserState(len(self._meta))
        self._meta.append((who, off, len(stamp)))

    def scrollContentsBy(self, dx, dy):
        # без сдвига картинки: края с затуханием стоят на месте, текст едет под ними
        super().scrollContentsBy(dx, dy)
        self.viewport().update()

    def paintEvent(self, e):
        p = QPainter(self.viewport())
        p.setRenderHint(AA)
        self._paint_marks(p)
        p.end()
        super().paintEvent(e)
        bar = self.verticalScrollBar()
        p = QPainter(self.viewport())
        w, h = self.viewport().width(), self.viewport().height()
        for edge, on in ((0, bar.value() > 0), (h, bar.value() < bar.maximum())):
            if not on:
                continue
            g = QLinearGradient(0, edge, 0, edge + (FADE if edge == 0 else -FADE))
            g.setColorAt(0, qc(BG1))
            g.setColorAt(1, qc(BG1, 0.0))
            p.fillRect(QRectF(0, edge if edge == 0 else edge - FADE, w, FADE), g)
        p.end()

    def _paint_marks(self, p: QPainter):
        vy = self.verticalScrollBar().value()
        h = self.viewport().height()
        block = self.cursorForPosition(QPoint(0, 0)).block()
        # аватар заголовка чуть выше экрана торчит вниз - берём на блок больше
        for _ in range(2):
            if block.previous().isValid():
                block = block.previous()
        p.setPen(Qt.PenStyle.NoPen)
        while block.isValid():
            lay = block.layout()
            pos = lay.position()
            top = pos.y() - vy
            if top > h:
                break
            idx = block.userState()
            if 0 <= idx < len(self._meta) and lay.lineCount():
                who, off, n = self._meta[idx]
                line = lay.lineAt(0)
                ltop = top + line.y()
                base = ltop + line.ascent()
                x0 = pos.x() + _x(line.cursorToX(off))
                x1 = pos.x() + _x(line.cursorToX(off + n))
                cy = base - self._time_cap / 2
                p.setBrush(white(0.06))
                p.drawRoundedRect(QRectF(x0 - PILL_PAD, cy - 9, x1 - x0 + 2 * PILL_PAD, 18), 9, 9)
                if self.avatars:
                    paint_avatar(p, QPointF(pos.x() + AVATAR_R, ltop + AVATAR_R - 2), AVATAR_R, who)
                    p.setPen(Qt.PenStyle.NoPen)
            block = block.next()


class JumpDown(AWidget):
    """Кнопка «к новым»: круг со стрелкой и счётчиком пропущенных фраз."""
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 52)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.v = self.anim(0, 14)
        self.hv = self.anim(0, 16)
        self.count = 0
        self.setVisible(False)

    def show_it(self, on: bool, count: int = 0):
        if count != self.count:
            self.count = count
            self.update()
        self.v.set(1.0 if on else 0.0)
        if on:
            self.setVisible(True)
            self.raise_()
        self.kick()

    def on_frame(self, dt):
        r = super().on_frame(dt)
        if self.v.value < 0.01 and self.v.target == 0.0 and self.isVisible():
            self.setVisible(False)
        return r

    def hover_changed(self, inside):
        self.hv.set(1.0 if inside else 0.0)
        self.kick()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.v.target > 0:
            self.clicked.emit()

    def paintEvent(self, _):
        v = self.v.value
        if v < 0.01:
            return
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setOpacity(v)
        c = QPointF(26, 30 + 6 * (1 - v))
        p.setPen(QPen(white(0.10 + 0.06 * self.hv.value), 1))
        p.setBrush(mix(qc('#272A37'), qc('#323646'), self.hv.value))
        p.drawEllipse(c, 18, 18)
        icons.draw(p, 'chevron-down', QRectF(c.x() - 9, c.y() - 8, 18, 18), qc(FG), stroke=2.2)
        if self.count > 0:
            txt = str(self.count) if self.count < 100 else '99+'
            f = font(10.5, QFont.Weight.Bold)
            w = max(18.0, QFontMetricsF(f).horizontalAdvance(txt) + 10)
            r = QRectF(c.x() - w / 2, c.y() - 30, w, 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(PRIMARY))
            p.drawRoundedRect(r, 9, 9)
            p.setFont(f)
            p.setPen(qc('#11131C'))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, txt)
        p.end()


class LiveDot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 36)
        self.setToolTip('Идёт запись')

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(AA)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qc(ACCENT, 0.22))
        p.drawEllipse(QPointF(7, 18), 6.5, 6.5)
        p.setBrush(qc(ACCENT))
        p.drawEllipse(QPointF(7, 18), 3.5, 3.5)
        p.end()


class TranscriptView(QWidget):
    """Расшифровка одного сеанса поверх списка."""
    closed = pyqtSignal()
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sid: Optional[str] = None
        self._armed = False
        self._offset = 0
        self._unseen = 0
        self._follow = False
        self._base = None
        self.live = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 2, 12, 10)
        lay.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(4)
        self.back = IconButton('chevron-left', 36, 18, tooltip='Назад')
        self.back.clicked.connect(self.close_view)
        head.addWidget(self.back)
        self.dot = LiveDot()
        self.dot.hide()
        head.addWidget(self.dot)
        self.title = TextField('', '', size=15)
        self.title.setToolTip('Название - можно изменить')
        self.title.editingFinished.connect(self._rename)
        head.addWidget(self.title, 1)
        self.copy = IconButton('copy', 36, 17, tooltip='Скопировать текст')
        self.copy.clicked.connect(self._copy)
        head.addWidget(self.copy)
        self.trash = IconButton('trash', 36, 17, tooltip='Удалить')
        self.trash.danger = True
        self.trash.clicked.connect(self._delete)
        head.addWidget(self.trash)
        lay.addLayout(head)

        self.card = Surface(16)
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(1, 4, 3, 4)
        self.text = Transcript()
        cl.addWidget(self.text)
        lay.addWidget(self.card, 1)

        self.jump = JumpDown(self)
        self.jump.clicked.connect(self.to_bottom)
        bar = self.text.verticalScrollBar()
        bar.valueChanged.connect(self._scrolled)
        bar.rangeChanged.connect(self._range)
        self._scroll_anim = QPropertyAnimation(bar, b'value', self)
        self._scroll_anim.setDuration(260)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._disarm = QTimer(self)
        self._disarm.setSingleShot(True)
        self._disarm.timeout.connect(self._reset_trash)

    def paintEvent(self, _):
        # непрозрачный фон окна: список под расшифровкой не просвечивает
        p = QPainter(self)
        off = self.mapTo(self.window(), self.rect().topLeft())
        p.translate(-off.x(), -off.y())
        paint_window_bg(p, QRectF(self.window().rect()))
        p.end()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_jump()

    def _place_jump(self):
        g = self.card.geometry()
        self.jump.move(g.center().x() - self.jump.width() // 2, g.bottom() - self.jump.height() - 6)

    # ---------- данные ----------

    def _path(self):
        return history.history_dir() / f'{self.sid}.jsonl'

    def open(self, sid: str, live: bool = False):
        self.sid = sid
        self._reset_trash()
        s = next((x for x in history.list_sessions() if x.id == sid), None)
        lt = time.localtime(s.started if s else time.time())
        default = f'{lt.tm_mday} {MONTHS[lt.tm_mon - 1]}, {time.strftime("%H:%M", lt)}'
        self.title.setPlaceholderText(default)
        self.title.setText(s.title if s else '')
        self._base = s.base if s is not None and s.file else None
        self.set_live(live)
        self._unseen = 0
        self.jump.show_it(False)
        self._load_all()
        # идущий сеанс открывается на последних фразах, прошлый - с начала
        self._follow = live
        bar = self.text.verticalScrollBar()
        bar.setValue(bar.maximum() if live else 0)
        self._update_jump()

    def _load_all(self):
        lines, self._offset = history.read_from(self._path(), 0)
        self.text.reset(any((ln.get('who') or '?') != '?' for ln in lines), self._base)
        self.text.append_lines(lines)
        self.text.park_cursor()

    def set_live(self, live: bool):
        self.live = live
        self.dot.setVisible(live)

    def poll(self):
        """Дописать новые фразы идущего сеанса."""
        if not self.sid:
            return
        try:
            size = os.path.getsize(self._path())
        except OSError:
            return
        if size <= self._offset:
            return
        bar = self.text.verticalScrollBar()
        lines, offset = history.read_from(self._path(), self._offset)
        if not lines:
            self._offset = offset
            return
        if not self.text.avatars and any((ln.get('who') or '?') != '?' for ln in lines):
            # появилось первое имя - нужна колонка под аватары, перестраиваем
            keep, follow = bar.value(), self._follow
            self._load_all()
            self._follow = follow
            bar.setValue(bar.maximum() if follow else keep)
            return
        self._offset = offset
        n = self.text.append_lines(lines)
        if self._follow:
            bar.setValue(bar.maximum())
        else:
            self._unseen += n
            self._update_jump()

    # ---------- прокрутка ----------

    def _at_bottom(self) -> bool:
        bar = self.text.verticalScrollBar()
        return bar.value() >= bar.maximum() - 6

    def _scrolled(self, _v):
        if self._scroll_anim.state() == QPropertyAnimation.State.Running:
            return
        self._follow = self._at_bottom()
        if self._follow:
            self._unseen = 0
        self._update_jump()

    def _range(self, _lo, hi):
        # документ раскладывается частями - держимся низа, пока следим
        if self._follow:
            self.text.verticalScrollBar().setValue(hi)
        self._update_jump()

    def _update_jump(self):
        bar = self.text.verticalScrollBar()
        far = bar.maximum() - bar.value() > 40
        self._place_jump()
        self.jump.show_it(far and not self._follow, self._unseen)

    def to_bottom(self):
        bar = self.text.verticalScrollBar()
        self._follow = True
        self._unseen = 0
        self.jump.show_it(False)
        self._scroll_anim.stop()
        self._scroll_anim.setStartValue(bar.value())
        self._scroll_anim.setEndValue(bar.maximum())
        self._scroll_anim.start()

    # ---------- действия ----------

    def _rename(self):
        if self.sid:
            history.rename(self.sid, self.title.text())
            self.changed.emit()

    def _copy(self):
        if self.sid:
            QGuiApplication.clipboard().setText(history.as_text(self.sid))
            self.copy.set_on(True)
            QTimer.singleShot(900, lambda: self.copy.set_on(False))

    def _delete(self):
        if not self.sid:
            return
        if not self._armed:
            # первое нажатие только взводит: кнопка краснеет
            self._armed = True
            self.trash.set_icon('alert')
            self.trash.h.set(1.0)
            self.trash.kick()
            self._disarm.start(2600)
            return
        history.delete(self.sid)
        self.sid = None
        self.changed.emit()
        self.close_view()

    def _reset_trash(self):
        self._armed = False
        self.trash.set_icon('trash')
        self.trash.h.set(0.0)
        self.trash.kick()

    def close_view(self):
        self.hide()
        self.closed.emit()


# ============================================================
# ЭКРАН
# ============================================================

class HistoryPage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 6, 20, 12)
        lay.setSpacing(4)

        top = RowCard()
        self.enabled = Toggle(win.config.history_enabled)
        self.enabled.toggled.connect(self._toggle)
        top.add(Row('history', 'Сохранять', self.enabled, tooltip='Записывать расшифровки на этот компьютер'))
        self.folder = IconButton('external', 32, 16, tooltip='Открыть папку')
        self.folder.clicked.connect(self._open_folder)
        self.add_file = IconButton('plus', 32, 17, tooltip='Расшифровать запись или видео.\n'
                                                           'Файлы можно и просто перетащить в окно')
        self.add_file.clicked.connect(self._pick_files)
        top.add(Row('file', 'Файл', self.add_file, height=48))
        top.add(Row('folder', 'Папка', self.folder, color=FG_DIM, height=48))
        lay.addWidget(top)
        self._waiting: list[str] = []       # файлы ждут загрузки моделей
        self._watch: dict[str, object] = {}  # запущенные задания - сообщить об ошибке

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(SCROLL_CSS)
        self.list = SessionList()
        self.list.opened.connect(self.open_session)
        self.scroll.setWidget(self.list)
        lay.addWidget(self.scroll, 1)
        self.empty = EmptyState()
        lay.addWidget(self.empty, 1)

        self.view = TranscriptView(self)
        self.view.hide()
        self.view.closed.connect(self.refresh)
        self.view.changed.connect(self.refresh)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._live_tick)
        self._tick.start(1000)
        self._ticks = 0
        self._sig = None

    def resizeEvent(self, e):
        self.view.setGeometry(self.rect())

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()

    def refresh(self):
        sessions = history.list_sessions()
        live = self.win.engine.history.live_id
        sig = tuple((s.id, s.title) for s in sessions) + (live, day_label(time.time()))
        if sig != self._sig:
            self._sig = sig
            bar = self.scroll.verticalScrollBar()
            keep = bar.value()
            self.list.set_sessions(sessions, live)
            self.list.lay.activate()
            bar.setValue(keep)
        else:
            self.list.update_sessions(sessions)
        self.scroll.setVisible(bool(sessions))
        self.empty.setVisible(not sessions)

    def _live_tick(self):
        self._start_waiting()
        self._check_jobs()
        # пока экран открыт, идущий сеанс дописывается сам
        if not self.isVisible() or self.window().isMinimized():
            return
        live = self.win.engine.history.live_id
        if self.view.isVisible():
            if self.view.sid and self.view.live:
                self.view.poll()
                if self.view.sid != live and self.view.sid not in history.jobs:
                    self.view.poll()
                    self.view.set_live(False)
        elif history.jobs:
            self.refresh()
        elif live or self._sig is None:
            # список не обязан успевать за каждой фразой - раз в три секунды
            self._ticks += 1
            if self._ticks % 3 == 0 or self._sig is None:
                self.refresh()

    def open_session(self, sid: str):
        self.view.setGeometry(self.rect())
        self.view.show()
        self.view.raise_()
        self.view.open(sid, live=sid == self.win.engine.history.live_id or sid in history.jobs)

    # ---------- файлы ----------

    def _pick_files(self):
        from PyQt6.QtWidgets import QFileDialog
        exts = ' '.join('*' + e for e in file_transcribe.EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(self, 'Расшифровать', '',
                                                f'Запись или видео ({exts});;Все файлы (*)')
        self.add_files(paths)

    def add_files(self, paths: list[str]):
        paths = [p for p in paths if os.path.isfile(p)]
        if not paths:
            return
        self._waiting += paths
        if self.win.engine.stt is None:
            self.win.load_models()
        self._start_waiting()

    def _start_waiting(self):
        eng = self.win.engine
        if not self._waiting or eng.stt is None or eng.vad is None:
            return
        for path in self._waiting:
            job = file_transcribe.FileJob(path, eng)
            self._watch[job.sid] = job
            job.start()
        self._waiting = []
        self._sig = None
        self.refresh()

    def _check_jobs(self):
        for sid, job in list(self._watch.items()):
            if not job.done:
                continue
            del self._watch[sid]
            if job.error:
                if not history.load(sid):
                    history.delete(sid)
                self.win.toast(f'{job.path.name}: не получилось прочитать', 'error')
            self._sig = None
        if not self._watch and not history.jobs:
            return
        if self.isVisible():
            self.refresh()

    def _toggle(self, on: bool):
        self.win.config.history_enabled = on
        self.win.engine.history.enabled = on
        self.win.save()

    def _open_folder(self):
        os.startfile(str(history.history_dir()))

    def leave(self):
        self.view.hide()
