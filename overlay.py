"""
Оверлей субтитров.

Прозрачное окно поверх всех, мышь проходит насквозь. События приходят
из движка через потокобезопасную очередь и привязаны к паре
(источник, кусок фразы), а не к имени: имя может уточниться позже,
и строка от этого не должна теряться.

Что видит читающий - одну ленту, как в чате:
  * реплики идут подряд на общей подложке; имя цветом - только когда
    говорящий сменился, у каждой реплики слева полоска его цвета;
  * новое добавляется снизу, вся лента плавно уезжает вверх и растворяется
    у верхнего края; ничего не появляется и не исчезает по отдельности,
    лента гаснет целиком после тишины;
  * слова выходят по одному в темпе чтения, черновой хвост тусклее;
  * кто сейчас говорит - мягко пульсирует его полоска. Отдельная строка
    с именем и полосками звука появляется, только если человек заговорил,
    а его реплики внизу ещё нет (и только если говорит дольше 0.3 с).

Английская речь с переводом - две ленты: перевод внизу, оригинал над
ним. Перевод приходит, когда предложение договорено (до этого - черновик
перевода, тусклее), и в своей ленте ничего не сдвигает в оригинале.
Вид «Перевод» - только нижняя лента, «Оригинал» - только верхняя.
"""
from __future__ import annotations

import math
import queue
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QColor, QCursor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen,
)
from PyQt6.QtWidgets import QApplication, QWidget

try:
    import ctypes
    from ctypes import wintypes

    class _CURSORINFO(ctypes.Structure):
        _fields_ = [('cbSize', wintypes.DWORD), ('flags', wintypes.DWORD),
                    ('hCursor', wintypes.HANDLE), ('ptScreenPos', wintypes.POINT)]

    def _cursor_shown() -> bool:
        ci = _CURSORINFO()
        ci.cbSize = ctypes.sizeof(_CURSORINFO)
        if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):
            return True
        return bool(ci.flags & 1)       # CURSOR_SHOWING
except (ImportError, AttributeError, OSError):
    def _cursor_shown() -> bool:
        return True

from ui.theme import FONT_DISPLAY, speaker_color

event_q: "queue.Queue[dict]" = queue.Queue()

# ---- поведение ----
CARD_GAP_MS = 12000          # тот же человек снова говорит раньше - продолжаем его реплику
STALE_MS = 20000             # черновик без обновлений столько - считаем законченным
MAX_CARDS = 40
KEEP_LINES = 8               # прокрученные строки сверх этого выбрасываются

# ---- темп показа ----
REVEAL_BASE = 5.0            # слов в секунду, когда очередь пуста
REVEAL_GAIN = 2.5            # +слов/с на каждое слово в очереди
REVEAL_MAX = 12.0
# темп для пресета «Быстро»: (база, прибавка за слово в очереди, потолок)
REVEAL_FAST = (8.0, 4.0, 20.0)
# перевод приходит предложением сразу - выходит быстрее, но не разом
TR_REVEAL_BASE = 10.0
TR_REVEAL_GAIN = 3.0
TR_REVEAL_MAX = 24.0

# ---- анимация ----
# Оверлей висит поверх игры всё время, пока кто-то говорит, и каждый кадр
# прозрачного окна Windows копирует в композитор целиком. Проявление
# слова (220 мс) на 30 к/с - это 7 кадров: на глаз плавно, а стоит вдвое
# дешевле 60 к/с.
FRAME_MS = 33
TRANS_RGB = (178, 190, 255)    # перевод - светлый барвинок, отличим от белой речи
HOVER_TAU_S = 0.09          # как быстро лента гаснет под курсором
WORD_ANIM_MS = 220
WORD_RISE_PX = 5
DONE_ANIM_MS = 300
DRAFT_ALPHA = 0.58
CARD_APPEAR_MS = 280
CARD_FADE_MS = 700
POS_TAU_S = 0.08
SIZE_TAU_S = 0.07
SCROLL_TAU_S = 0.12
FEED_SCROLL_TAU_S = 0.16     # лента уплывает вверх спокойно, но не отстаёт от чтения
FEED_SIZE_TAU_S = 0.12
WORD_SLIDE_TAU_S = 0.09      # строка по центру сдвигается на полслова - слова доезжают
GHOST_FADE_MS = 260
GHOST_DELAY_MS = 300         # короткие всплески «начал говорить» строку не показывают
GHOST_LINGER_MS = 1200       # замолчал - индикатор ещё немного держится: без мигания
PULSE_MS = 1600              # период пульса полоски говорящего

# ---- геометрия ----
MARGIN = 10
CARD_GAP = 8
ACCENT_W = 3
ZONE_GAP = 6                 # между лентой оригинала и лентой перевода
RAISE_INTERVAL_MS = 1500

EDIT_BORDER_PX = 10
EDIT_MIN_W = 220
EDIT_MIN_H = 90

DEMO = [
    ('Дима', 'Идём на босса, он уже появился.'),
    ('Маша', 'Погоди, я ещё квест не сдала.'),
    ('Дима', 'Окей, жду. Только не тяни, через две минуты он уйдёт.'),
    ('?', 'Кто-нибудь слышит меня?'),
]
DEMO_EN = [
    ('Okay everyone, today we are going to talk about recursion.',
     'Итак, сегодня мы поговорим о рекурсии.'),
    ('A recursive function is a function that calls itself.',
     'Рекурсивная функция — это функция, которая вызывает саму себя.'),
    ('Without a base case it never stops.', 'Без базового случая она никогда не останавливается.'),
]


def _now() -> int:
    return time.monotonic_ns() // 1_000_000


_EDGE_PUNCT = '.,!?;:—–-«»"\'()…'


def _norm(word: str) -> str:
    # GigaAM расставляет запятые и заглавные в уже сказанном по мере роста
    # контекста. Без нормализации это выглядело бы как новое слово.
    return word.strip(_EDGE_PUNCT).lower().replace('ё', 'е')


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _approach(cur: float, target: float, dt: float, tau: float,
              eps: float = 0.3) -> tuple[float, bool]:
    """Экспоненциально тянет cur к target. Второе значение - ещё движется."""
    if abs(target - cur) < eps:
        return target, False
    k = 1.0 - math.exp(-dt / tau) if dt > 0 else 1.0
    return cur + (target - cur) * k, True


# ============================================================
# МОДЕЛЬ
# ============================================================

@dataclass(eq=False)
class W:
    text: str
    done: bool                  # подтверждено движком
    shown: int = 0              # когда показано; 0 - ждёт в очереди
    done_t: int = 0             # когда черновик стал подтверждённым
    end: Optional[float] = None # monotonic конца слова в звуке (для замеров)
    created: int = 0            # когда слово пришло в оверлей (для замеров)
    cx: Optional[float] = None  # где слово нарисовано сейчас (плавно едет к месту)
    cline: int = -1
    tr: bool = False            # слово перевода


@dataclass(eq=False)
class Segment:
    stream: str
    seg: object
    speaker: str
    words: list = field(default_factory=list)
    final: bool = False
    updated: int = 0


@dataclass(eq=False)
class Card:
    speaker: str
    stream: str
    segments: list = field(default_factory=list)
    born: int = 0
    updated: int = 0
    y: Optional[float] = None
    vis: float = 1.0                # 0, если не помещается в окно
    gone_t: int = 0                 # принудительное исчезновение (drop/clear)
    w: float = 0.0                  # анимированные размеры фона
    h: float = 0.0
    top: float = 0.0                # анимированная первая видимая строка
    trim: int = 0                   # столько первых слов уже уехало навсегда
    last_reveal: int = 0
    tr: bool = False                # карточка ленты перевода
    _layout_key: tuple = ()
    _layout: object = None

    def words(self) -> list:
        return [w for s in self.segments for w in s.words]

    def text(self) -> str:
        return ' '.join(w.text for w in self.words())

    def live(self) -> bool:
        return any(not s.final for s in self.segments)


@dataclass
class Ghost:
    stream: str
    speaker: Optional[str]
    born: int
    active: bool = True
    ended: int = 0
    y: Optional[float] = None
    vis: float = 1.0


@dataclass
class FeedState:
    """Анимированные размеры ленты."""
    w: float = 0.0
    h: float = 0.0
    scroll: float = 0.0         # насколько содержимое уехало вверх
    born: int = 0
    ready: bool = False         # первый кадр ставит размеры сразу, дальше - плавно
    tops: dict = field(default_factory=dict)
    rect: Optional[QRectF] = None   # где лента на последнем кадре - для наведения курсора
    hold_h: float = 0.0             # лента перевода не сжимается до следующего затухания


@dataclass
class OverlayStyle:
    font_size: int = 22
    bg_opacity: float = 0.78
    align: str = 'left'
    ttl_ms: int = 10_000
    max_lines: int = 4          # строк текста видно в ленте
    hover_opacity: float = 0.2  # под курсором лента почти прозрачная; 1 - не менять
    split_segments: bool = False    # английская речь: каждая фраза с новой строки
    view: str = 'orig'              # orig - оригинал, both - оригинал и перевод, trans - перевод
    reveal: tuple = (REVEAL_BASE, REVEAL_GAIN, REVEAL_MAX)    # темп выхода слов


@dataclass
class _Layout:
    words: list          # (W, x, line)
    lines: int
    content_w: float
    name: str
    name_w: float
    pad_x: float
    pad_y: float
    name_h: float
    line_h: float
    first_word_of_line: list
    align: str = 'left'


def demo_cards(prefix: str, n: int = 3) -> list:
    """Готовые карточки из DEMO - для превью и редактора положения."""
    now = _now()
    cards = []
    for i, (spk, text) in enumerate(DEMO[:n]):
        c = Card(speaker=spk, stream=f'{prefix}{i}', born=now - 10_000, updated=now)
        s = Segment(c.stream, i, spk, [W(t, True, now - 10_000) for t in text.split()], True, now)
        c.segments.append(s)
        cards.append(c)
    return cards


# ============================================================
# ОТРИСОВКА КАРТОЧЕК (общая для оверлея, редактора и превью)
# ============================================================

class CardRenderer:
    def __init__(self):
        self._fonts: dict[int, tuple] = {}

    def fonts(self, size: int):
        f = self._fonts.get(size)
        if f is None:
            text = QFont(FONT_DISPLAY)
            text.setPointSizeF(size)
            text.setWeight(QFont.Weight.DemiBold)
            text.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
            name = QFont(FONT_DISPLAY)
            name.setPointSizeF(max(7.0, size * 0.56))
            name.setWeight(QFont.Weight.Bold)
            name.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, max(0.4, size * 0.03))
            f = (text, name, QFontMetricsF(text), QFontMetricsF(name))
            self._fonts[size] = f
        return f

    @staticmethod
    def named(speaker: Optional[str]) -> bool:
        return bool(speaker) and speaker != '?'

    def layout(self, card: Card, max_w: float, size: int, align: str = 'left',
               split: bool = False) -> _Layout:
        shown = [w for w in card.words()[card.trim:] if w.shown]
        key = (tuple((id(w), w.text) for w in shown), card.speaker, int(max_w), size, align, split)
        if card._layout_key == key and card._layout is not None:
            return card._layout
        text_f, name_f, tm, nm = self.fonts(size)
        pad_x = max(12.0, size * 0.62)
        pad_y = max(7.0, size * 0.38)
        avail = max(40.0, max_w - 2 * pad_x - ACCENT_W - 4)

        # Поток слов реплики; split - каждый кусок с новой строки. None - перенос.
        flow: list = []
        k = 0
        for seg in card.segments:
            vis = [w for j, w in enumerate(seg.words, k) if j >= card.trim and w.shown]
            k += len(seg.words)
            if split and vis and flow:
                flow.append(None)
            flow += vis

        words = []
        firsts = [0]                # сколько слов реплики выше строки
        space = tm.horizontalAdvance(' ')
        x = 0.0
        line = 0
        used = 0.0
        orig = 0
        brk = False
        for w in flow:
            if w is None:
                brk = True
                continue
            ww = tm.horizontalAdvance(w.text)
            if x > 0 and (brk or x + ww > avail):
                x = 0.0
                line += 1
                firsts.append(orig)
            brk = False
            words.append((w, x, line))
            used = max(used, x + ww)
            x += ww + space
            orig += 1

        name = card.speaker if self.named(card.speaker) else ''
        name_w = nm.horizontalAdvance(name.upper()) if name else 0.0
        name_h = (nm.height() + size * 0.12) if name else 0.0
        content_w = max(used, name_w)
        if align == 'center' and words:
            # каждая строка по центру карточки, как в кино
            ends: dict[int, float] = {}
            for w, x, ln in words:
                ends[ln] = x + tm.horizontalAdvance(w.text)
            words = [(w, x + (content_w - ends[ln]) / 2, ln) for w, x, ln in words]
        lay = _Layout(words, line + 1 if words else 0, content_w, name, name_w,
                      pad_x, pad_y, name_h, tm.height(), firsts, align)
        card._layout_key = key
        card._layout = lay
        return lay

    def metrics(self, size: int) -> dict:
        text_f, name_f, tm, nm = self.fonts(size)
        return {
            'pad_x': max(14.0, size * 0.7), 'pad_y': max(9.0, size * 0.45),
            'line_h': tm.height(), 'name_h': nm.height() + size * 0.18,
            'gap': size * 0.5, 'radius': min(16.0, max(9.0, size * 0.6)),
            'indent': ACCENT_W + 8,
        }

    def slide_words(self, lay: _Layout, dt: float) -> bool:
        """Слова плавно доезжают на место (строка по центру растёт). True - ещё едут."""
        moving = False
        k = 1.0 - math.exp(-dt / WORD_SLIDE_TAU_S) if dt > 0 else 1.0
        for word, wx, line in lay.words:
            if word.cx is None or word.cline != line or dt <= 0:
                word.cx, word.cline = wx, line
            elif abs(wx - word.cx) > 0.3:
                word.cx += (wx - word.cx) * k
                moving = True
            else:
                word.cx = wx
        return moving

    def paint_words(self, p: QPainter, lay: _Layout, tx: float, ty: float, alpha: float,
                    now: int, size: int, row_alpha, tr_color: bool = False):
        """Текст реплики. row_alpha(y) - прозрачность строки у края ленты."""
        text_f, name_f, tm, nm = self.fonts(size)
        p.setFont(text_f)
        for word, wx, line in lay.words:
            y = ty + line * lay.line_h
            a = alpha * row_alpha(y)
            if a <= 0.01:
                continue
            dy = 0.0
            t = (now - word.shown) / WORD_ANIM_MS
            if t < 1.0:
                e = _ease_out(t)
                a *= 0.05 + 0.95 * e
                dy = WORD_RISE_PX * (1 - e)
            if word.done:
                k = 1.0
                if word.done_t:
                    k = DRAFT_ALPHA + (1 - DRAFT_ALPHA) * _smooth((now - word.done_t) / DONE_ANIM_MS)
            else:
                k = DRAFT_ALPHA
            a *= k
            if a <= 0.01:
                continue
            if word.tr and tr_color:
                p.setPen(QColor(*TRANS_RGB, int(250 * a)))
            else:
                p.setPen(QColor(236, 237, 240, int(250 * a)))
            p.drawText(QPointF(tx + word.cx, y + tm.ascent() + dy), word.text)

    def paint_bars(self, p: QPainter, x: float, cy: float, h: float, alpha: float, now: int,
                   size: int, level: float):
        bw = max(2.0, size * 0.12)
        gap = bw * 0.9
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(4):
            ph = now / 140.0 + i * 1.7
            k = 0.25 + 0.75 * abs(math.sin(ph)) * (0.55 + 0.45 * level)
            bh = max(bw, h * 0.5 * k)
            p.setBrush(QColor(236, 237, 240, int(150 * alpha)))
            p.drawRoundedRect(QRectF(x + i * (bw + gap), cy - bh / 2, bw, bh), bw / 2, bw / 2)
        return 4 * (bw + gap)


# ============================================================
# ЛОГИКА (без окна - её же гоняет tools/subtitle_jumps.py)
# ============================================================

class SubtitleModel:
    def __init__(self, style: Optional[OverlayStyle] = None, **kwargs):
        # PyQt зовёт конструкторы по цепочке наследования, поэтому сюда
        # можно прийти и из QWidget.__init__ - без аргументов
        super().__init__(**kwargs)
        self.style_ = style or OverlayStyle()
        self.cards: list[Card] = []
        self.ghosts: dict[str, Ghost] = {}
        self._index: dict[tuple, tuple[Card, Segment]] = {}
        self.feed = FeedState()
        # лента перевода: свои карточки, по тем же (источник, кусок)
        self.tcards: list[Card] = []
        self._tindex: dict[tuple, tuple[Card, Segment]] = {}
        self.tfeed = FeedState()
        # время модели - настоящие мс monotonic: можно мерить задержку показа
        # (в tools/subtitle_jumps.py модель гоняется на своём времени - там нельзя)
        self.measure = False

    def clear(self):
        self.cards = []
        self.ghosts = {}
        self._index = {}
        self.feed = FeedState()
        self.tcards = []
        self._tindex = {}
        self.tfeed = FeedState()

    def all_cards(self) -> list:
        return self.cards + self.tcards

    def speaking(self, now: int) -> set:
        """Источники, которые сейчас говорят (с задержкой на исчезновение)."""
        return {g.stream for g in self.ghosts.values()
                if g.active or now - g.ended < GHOST_LINGER_MS}

    def _unindex(self, card: Card):
        idx = self._tindex if card.tr else self._index
        for seg in card.segments:
            idx.pop((seg.stream, seg.seg), None)

    def _apply(self, ev: dict, now: int):
        t = ev.get('type')
        if t == 'clear':
            self.clear()
            return
        stream = ev.get('stream', '')
        if t == 'speaking':
            g = self.ghosts.get(stream)
            if ev.get('active'):
                if g is None:
                    self.ghosts[stream] = Ghost(stream, ev.get('speaker'), now)
                else:
                    # снова заговорил, пока индикатор ещё держится, - тот же
                    # индикатор, без повторного появления
                    g.active = True
                    g.speaker = ev.get('speaker') or g.speaker
            elif g is not None and g.active:
                g.active = False
                g.ended = now
            return

        key = (stream, ev.get('seg'))
        if t == 'trans':
            self._apply_trans(key, ev, now)
            return
        if t == 'drop':
            for idx in (self._index, self._tindex):
                found = idx.pop(key, None)
                if found:
                    self._remove_seg(*found, now)
            return
        if t != 'text':
            return

        speaker = ev.get('speaker') or '?'
        target = [(w, True) for w in (ev.get('done') or '').split()]
        target += [(w, False) for w in (ev.get('draft') or '').split()]
        ends = ev.get('ends') or []
        if len(ends) == len(target):
            target = [(t, d, e) for (t, d), e in zip(target, ends)]
        found = self._index.get(key)
        if found is None:
            if not target:
                return
            card = self._card_for(stream, speaker, now)
            seg = Segment(stream, ev.get('seg'), speaker, [], False, now)
            card.segments.append(seg)
            self._index[key] = (card, seg)
        else:
            card, seg = found
            if speaker != '?' and speaker != seg.speaker:
                seg.speaker = speaker
                if card.speaker != speaker:
                    if len(card.segments) == 1:
                        card.speaker = speaker
                    else:
                        # имя уточнилось и оказалось чужим - кусок уходит в свою
                        # карточку, и слова выходят там заново в общем темпе
                        card.segments.remove(seg)
                        card.updated = now
                        for w in seg.words:
                            w.shown = 0
                        card = Card(speaker, stream, [seg], born=now, updated=now)
                        self.cards.append(card)
                        self._index[key] = (card, seg)
        self._merge_words(seg, target, now)
        seg.final = bool(ev.get('final'))
        seg.updated = now
        card.updated = now

    @staticmethod
    def _remove_seg(card: Card, seg: Segment, now: int):
        card.segments.remove(seg)
        card.updated = now
        if not card.segments:
            card.gone_t = max(now, 1)

    def _apply_trans(self, key: tuple, ev: dict, now: int):
        """
        Перевод куска: text - переведённые законченные предложения, draft -
        перевод недоговорённого (тусклее, потом заменяется). Слова совмещаются
        как у оригинала: совпавшие остаются на месте, новые выходят по очереди.
        """
        target = [(w, True) for w in (ev.get('text') or '').split()]
        target += [(w, False) for w in (ev.get('draft') or '').split()]
        found = self._tindex.get(key)
        if found is None:
            if not target:
                return
            speaker = ev.get('speaker') or '?'
            card = self._card_for(key[0], speaker, now, tr=True)
            seg = Segment(key[0], key[1], speaker, [], False, now)
            card.segments.append(seg)
            self._tindex[key] = (card, seg)
        else:
            card, seg = found
            if not target:
                del self._tindex[key]
                self._remove_seg(card, seg, now)
                return
        self._merge_words(seg, target, now, ev.get('end') or None, tr=True)
        seg.final = not ev.get('draft')
        seg.updated = now
        card.updated = now

    def _card_for(self, stream: str, speaker: str, now: int, tr: bool = False) -> Card:
        """
        Продолжаем карточку этого источника, если она ещё живая или стоит
        последней и пауза короткая. Неуверенное '?' считаем тем же человеком.
        """
        cards = self.tcards if tr else self.cards
        bottom = next((c for c in reversed(cards) if not c.gone_t), None)
        for c in reversed(cards):
            if c.gone_t or c.stream != stream:
                continue
            fresh = c.live() or (c is bottom and now - c.updated < CARD_GAP_MS)
            if fresh and (c.speaker == speaker or speaker == '?'):
                return c
            break
        card = Card(speaker=speaker, stream=stream, born=now, updated=now, tr=tr)
        cards.append(card)
        return card

    def _merge_words(self, seg: Segment, target: list, now: int, end: Optional[float] = None,
                     tr: bool = False):
        """
        Совмещает новый текст куска со старым по выравниванию слов: если
        модель передумала про одно слово («Нет» -> «Ну»), меняется только
        оно, а не весь хвост за ним.
        """
        cur = seg.words
        sm = SequenceMatcher(None, [_norm(w.text) for w in cur],
                             [_norm(t[0]) for t in target], autojunk=False)
        out: list[W] = [None] * len(target)
        last_shown = -1
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op != 'equal':
                continue
            for k in range(a1 - a0):
                w = cur[a0 + k]
                text, done = target[b0 + k][:2]
                w.text = text                     # запятая или заглавная догнали
                if done and not w.done:
                    w.done = True
                    w.done_t = now
                out[b0 + k] = w
                if w.shown:
                    last_shown = b0 + k
        for i, t in enumerate(target):
            if out[i] is None:
                # слово внутри уже показанного текста видно сразу,
                # новые в конце ждут своей очереди
                out[i] = W(t[0], t[1], now if i < last_shown else 0, created=now, tr=tr)
                out[i].end = t[2] if len(t) > 2 else end
                if out[i].shown:
                    self._seen(out[i], now)
        # переписанный хвост на месте показанного тоже не прячем
        shown_before = sum(1 for w in cur if w.shown)
        for i in range(last_shown + 1, min(len(out), shown_before)):
            if not out[i].shown:
                out[i].shown = now
                self._seen(out[i], now)
        seg.words = out

    def pace(self, now: int) -> bool:
        """Выпускает слова из очереди в темпе чтения. True - очередь не пуста."""
        waiting = False
        for card in self.all_cards():
            if card.gone_t:
                continue
            words = card.words()
            hidden = [w for w in words if not w.shown]
            if not hidden:
                continue
            waiting = True
            if card.tr:
                rate = min(TR_REVEAL_MAX, TR_REVEAL_BASE + TR_REVEAL_GAIN * len(hidden))
            else:
                base, gain, top = self.style_.reveal
                rate = min(top, base + gain * len(hidden))
            if card.last_reveal == 0 or now - card.last_reveal >= 1000.0 / rate:
                hidden[0].shown = now
                card.last_reveal = now
                self._seen(hidden[0], now)
        return waiting

    def _seen(self, w: W, now: int):
        """Слово вышло в ленту - сколько оно шло от конца в звуке (metrics.py)."""
        if not self.measure or w.end is None:
            return
        import metrics
        if w.tr:
            metrics.add('tr_total', now / 1000 - w.end)
            return
        metrics.add('word_arrive', w.created / 1000 - w.end)
        metrics.add('word_pace', (now - w.created) / 1000)
        metrics.add('word_total', now / 1000 - w.end)

    def housekeeping(self, now: int) -> bool:
        changed = False
        for card in self.all_cards():
            for seg in card.segments:
                if not seg.final and now - seg.updated > STALE_MS:
                    seg.final = True
                    changed = True
        for tr in (False, True):
            cards = self.tcards if tr else self.cards
            keep = []
            for card in cards:
                if card.gone_t and now - card.gone_t > CARD_FADE_MS:
                    self._unindex(card)
                    continue
                keep.append(card)
            if len(keep) != len(cards):
                changed = True
            for card in keep[:-MAX_CARDS]:
                self._unindex(card)
            keep = keep[-MAX_CARDS:]
            if tr:
                self.tcards = keep
            else:
                self.cards = keep
        # Лента гаснет целиком: когда всё сказанное показано и прошло ttl.
        ttl = self.style_.ttl_ms
        if ttl > 0 and self.all_cards() and self.idle_since(now) > ttl + CARD_FADE_MS:
            self.cards, self.tcards = [], []
            self._index, self._tindex = {}, {}
            self.feed, self.tfeed = FeedState(), FeedState()
            changed = True
        for key, g in list(self.ghosts.items()):
            if not g.active and now - g.ended > GHOST_LINGER_MS + GHOST_FADE_MS:
                del self.ghosts[key]
                changed = True
        return changed

    def idle_since(self, now: int) -> float:
        """Сколько мс ничего не происходит (0 - кто-то говорит или слова ещё выходят)."""
        cards = self.all_cards()
        if any(not self._idle(c) for c in cards if not c.gone_t):
            return 0.0
        if any(g.active for g in self.ghosts.values()):
            return 0.0
        last = max((c.updated for c in cards), default=now)
        return max(0.0, now - last)

    @staticmethod
    def _idle(card: Card) -> bool:
        return not card.live() and all(w.shown for s in card.segments for w in s.words)

    def drop_card(self, card: Card):
        """Реплика целиком уехала за верхний край - больше не нужна."""
        self._unindex(card)
        (self.tcards if card.tr else self.cards).remove(card)

    def trim_lines(self, card: Card, lay: _Layout, lines: int):
        """Первые строки длинной реплики уехали наверх - выбрасываем, перенос остальных тот же."""
        lines = min(lines, lay.lines - 1)
        if lines <= 0:
            return
        card.trim += lay.first_word_of_line[lines]
        idx = self._tindex if card.tr else self._index
        while card.segments and len(card.segments[0].words) <= card.trim and len(card.segments) > 1:
            seg = card.segments.pop(0)
            card.trim -= len(seg.words)
            idx.pop((seg.stream, seg.seg), None)


# ============================================================
# ОВЕРЛЕЙ
# ============================================================

class OverlayWindow(QWidget, SubtitleModel):
    geometry_changed = pyqtSignal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.X11BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self.renderer = CardRenderer()
        self.measure = True             # время - настоящие мс monotonic
        self._last_frame = _now()
        self._layer: Optional[QImage] = None
        self._moving = False
        self.level_fn = None        # функция stream -> уровень звука 0..1

        screen = QApplication.primaryScreen().geometry()
        w = int(screen.width() * 0.5)
        h = 280
        self.setGeometry((screen.width() - w) // 2, int(screen.height() * 0.85) - h, w, h)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._on_poll)
        self._poll.start(40)

        self._frame = QTimer(self)
        self._frame.setTimerType(Qt.TimerType.PreciseTimer)
        self._frame.timeout.connect(self._on_frame)

        self._raise = QTimer(self)
        self._raise.timeout.connect(self.raise_)
        self._raise.start(RAISE_INTERVAL_MS)

        self._hover = 1.0           # множитель прозрачности: 1 - курсора над лентой нет
        self._hover_target = 1.0
        self._hover_force_until = 0

        self.show()
        self.raise_()

    # ---------- API ----------

    def apply_geometry(self, x: int, y: int, width: int, height: int):
        """-1 у координаты - посчитать от экрана."""
        screen = QApplication.primaryScreen().geometry()
        if width <= 0:
            width = int(screen.width() * 0.5)
        if height <= 0:
            height = 280
        width = max(EDIT_MIN_W, min(width, screen.width()))
        height = max(EDIT_MIN_H, min(height, screen.height()))
        if x < 0:
            x = screen.x() + (screen.width() - width) // 2
        if y < 0:
            y = screen.y() + int(screen.height() * 0.85) - height
        x = max(screen.x(), min(x, screen.x() + screen.width() - width))
        y = max(screen.y(), min(y, screen.y() + screen.height() - height))
        self.setGeometry(x, y, width, height)
        self._kick()

    def set_font_size(self, size: int):
        self.style_.font_size = max(10, min(60, int(size)))
        self._kick()

    def font_size(self) -> int:
        return self.style_.font_size

    def set_ttl_seconds(self, seconds: int):
        self.style_.ttl_ms = max(0, int(seconds) * 1000)
        self._kick()

    def set_bg_opacity(self, value: float):
        self.style_.bg_opacity = max(0.0, min(1.0, float(value)))
        self._kick()

    def set_align(self, align: str):
        self.style_.align = 'center' if align == 'center' else 'left'
        self._kick()

    def set_max_lines(self, n: int):
        self.style_.max_lines = max(1, min(8, int(n)))
        self._kick()

    def set_split_segments(self, on: bool):
        self.style_.split_segments = bool(on)
        self._kick()

    def set_fast_reveal(self, fast: bool):
        """Пресет «Быстро»: слова из очереди выходят быстрее."""
        self.style_.reveal = REVEAL_FAST if fast else (REVEAL_BASE, REVEAL_GAIN, REVEAL_MAX)

    def set_view(self, view: str):
        """orig - только оригинал, both - оригинал и перевод под ним, trans - только перевод."""
        self.style_.view = view if view in ('orig', 'both', 'trans') else 'orig'
        self._kick()

    def set_hover_opacity(self, value: float):
        self.style_.hover_opacity = max(0.0, min(1.0, float(value)))
        self._kick()

    def preview_hover(self, ms: int = 1600):
        """Показать, как лента выглядит под курсором: пока двигают ползунок."""
        self._hover_force_until = _now() + ms
        if not self.all_cards():
            self.show_demo()
        self._kick()

    def hide_overlay(self):
        self.hide()

    def show_overlay(self):
        self.show()
        self.raise_()

    def clear(self):
        SubtitleModel.clear(self)
        self._kick()

    def show_demo(self):
        """Пример субтитров - посмотреть, как выглядит текущий стиль."""
        now = _now()
        if self.style_.view != 'orig':
            for i, (text, tr) in enumerate(DEMO_EN):
                event_q.put({'type': 'text', 'stream': 'demo', 'seg': now + i, 'speaker': '?',
                             'done': text, 'draft': '', 'final': True, 'delay': i * 2200})
                event_q.put({'type': 'trans', 'stream': 'demo', 'seg': now + i, 'speaker': '?',
                             'text': tr, 'draft': '', 'delay': i * 2200 + 900})
            return
        for i, (spk, text) in enumerate(DEMO):
            event_q.put({'type': 'text', 'stream': f'demo{i % 2}', 'seg': now + i,
                         'speaker': spk, 'done': text, 'draft': '', 'final': True,
                         'delay': i * 1400})

    def open_editor(self):
        editor = OverlayEditor(self.style_)
        editor.setGeometry(self.geometry())
        editor.show()
        editor.raise_()
        editor.activateWindow()
        return editor

    # ---------- события ----------

    def _kick(self):
        if not self._frame.isActive():
            self._last_frame = _now()
            self._frame.start(FRAME_MS)
        self.update()

    def _on_poll(self):
        changed = False
        now = _now()
        delayed = []
        while True:
            try:
                ev = event_q.get_nowait()
            except queue.Empty:
                break
            if ev.get('delay'):
                ev['due'] = now + ev.pop('delay')
                delayed.append(ev)
                continue
            if ev.get('due', 0) > now:
                delayed.append(ev)
                continue
            self._apply(ev, now)
            changed = True
        for ev in delayed:
            event_q.put(ev)
        if self.pace(now):
            changed = True
        if self.housekeeping(now):
            changed = True
        self._hover_target = self._hover_wanted(now)
        if changed or self._needs_animation(now):
            self._kick()

    def _hover_wanted(self, now: int) -> float:
        k = self.style_.hover_opacity
        if k >= 0.999:
            return 1.0
        if now < self._hover_force_until:
            return k
        rects = [r for r in (self.feed.rect, self.tfeed.rect) if r is not None]
        if not rects or not self.isVisible() or not _cursor_shown():
            # курсор спрятан (игра держит его в центре) - не гасим
            return 1.0
        pos = QPointF(self.mapFromGlobal(QCursor.pos()))
        return k if any(r.adjusted(-6, -6, 6, 6).contains(pos) for r in rects) else 1.0

    # ---------- анимация ----------

    def _needs_animation(self, now: int) -> bool:
        if self.ghosts or self._moving or abs(self._hover - self._hover_target) > 0.005:
            return True
        ttl = self.style_.ttl_ms
        idle = self.idle_since(now)
        if self.all_cards() and (idle == 0.0 or (ttl > 0 and idle > ttl - 50)):
            return True
        for card in self.all_cards():
            if card.gone_t or now - card.born < CARD_APPEAR_MS:
                return True
            for seg in card.segments:
                for w in seg.words:
                    if not w.shown or now - w.shown < WORD_ANIM_MS:
                        return True
                    if w.done_t and now - w.done_t < DONE_ANIM_MS:
                        return True
        return False

    def _on_frame(self):
        now = _now()
        self.pace(now)
        self.update()
        if not self._needs_animation(now):
            self._frame.stop()

    # ---------- рисование ----------

    def paintEvent(self, _e):
        if not self.cards and not self.tcards and not self.ghosts:
            self._moving = False
            self.feed.rect = self.tfeed.rect = None
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        now = _now()
        dt = max(0.0, min(0.1, (now - self._last_frame) / 1000.0))
        self._last_frame = now
        if dt > 0:
            self._hover += (self._hover_target - self._hover) * (1 - math.exp(-dt / HOVER_TAU_S))
        if abs(self._hover - self._hover_target) <= 0.005:
            self._hover = self._hover_target
        p.setOpacity(self._hover)
        self._moving = paint_overlay(p, QRectF(self.rect()), self.renderer, self, self.style_, now, dt,
                                     self.level_fn)
        p.end()


def paint_overlay(p: QPainter, area: QRectF, renderer: 'CardRenderer', model: SubtitleModel,
                  style: OverlayStyle, now: int, dt: float, level_fn=None) -> bool:
    """
    Оригинал и перевод - две ленты: перевод внизу (его высота не сжимается,
    пока лента не погаснет), оригинал - над ним. Пока перевода нет, оригинал
    стоит внизу один. Вид 'trans' - только перевод, 'orig' - только оригинал.
    """
    ghosts = list(model.ghosts.values())
    if style.view == 'trans':
        model.feed.rect = None
        return paint_feed(p, area, renderer, model.tcards, ghosts, style, now, dt, level_fn, model,
                          feed=model.tfeed)
    if style.view == 'both' and any(w.shown for c in model.tcards for w in c.words()):
        # высота окна делится поровну: оригиналу не меньше, чем переводу
        half = (area.height() - 2 * MARGIN - ZONE_GAP) / 2
        moving = paint_feed(p, area, renderer, model.tcards, [], style, now, dt, level_fn, model,
                            feed=model.tfeed, hold=True, tr_color=True, names=False, max_h=half)
        r = model.tfeed.rect
        upper = area
        if r is not None:
            upper = QRectF(area.left(), area.top(), area.width(),
                           max(0.0, r.top() - ZONE_GAP + MARGIN - area.top()))
        return paint_feed(p, upper, renderer, model.cards, ghosts, style, now, dt, level_fn, model) or moving
    model.tfeed.rect = None
    return paint_feed(p, area, renderer, model.cards, ghosts, style, now, dt, level_fn, model)


def paint_feed(p: QPainter, area: QRectF, renderer: CardRenderer, cards: list,
               ghosts: list, style: OverlayStyle, now: int, dt: float,
               level_fn=None, model: Optional[SubtitleModel] = None,
               feed: Optional[FeedState] = None, hold: bool = False, tr_color: bool = False,
               names: bool = True, max_h: Optional[float] = None) -> bool:
    """
    Лента реплик снизу области. Размеры и прокрутка плавно тянутся к целевым.
    Возвращает True, если что-то ещё движется. dt=0 - без анимации (превью).
    feed - состояние этой ленты (у перевода своё); hold - высота не
    сжимается; tr_color - перевод своим цветом; names - имена говорящих;
    max_h - потолок высоты ленты вместе с отступами.
    """
    size = style.font_size
    m = renderer.metrics(size)
    pad_x, pad_y, line_h, name_h, gap = m['pad_x'], m['pad_y'], m['line_h'], m['name_h'], m['gap']
    text_f, name_f, tm, nm = renderer.fonts(size)
    if feed is None:
        feed = model.feed if model is not None else FeedState()
    feed.rect = None
    max_w = area.width() - 2 * MARGIN
    moving = False

    def follow(cur, target, tau, eps=0.3):
        nonlocal moving
        if dt <= 0 or not feed.ready:
            return target
        v, mv = _approach(cur, target, dt, tau, eps)
        moving = moving or mv
        return v

    speaking = model.speaking(now) if model is not None else set()

    # ---- реплики ----
    blocks = []                     # (card, lay, show_name, height)
    prev = None
    for card in cards:
        lay = renderer.layout(card, max_w, size, style.align, style.split_segments or card.tr)
        if not lay.words:
            continue
        if renderer.slide_words(lay, dt):
            moving = True
        show_name = names and bool(lay.name) and (prev is None or prev.speaker != card.speaker)
        blocks.append([card, lay, show_name, (name_h if show_name else 0.0) + lay.lines * line_h])
        prev = card

    # ---- кто заговорил, а его реплики внизу ещё нет ----
    bottom = blocks[-1][0] if blocks else None
    rows = []
    for g in sorted(ghosts, key=lambda g: g.born):
        if now - g.born < GHOST_DELAY_MS:
            continue
        if bottom is not None and bottom.stream == g.stream:
            continue                # его реплика внизу - пульсирует её полоска
        if any(c.stream == g.stream and c.live() for c in cards):
            continue
        a = _smooth((now - g.born - GHOST_DELAY_MS) / GHOST_FADE_MS)
        if not g.active:
            a *= 1.0 - _smooth((now - g.ended - GHOST_LINGER_MS) / GHOST_FADE_MS)
        if a > 0.01:
            rows.append((g, a))
        moving = True

    if not blocks and not rows:
        return moving

    content_h = sum(b[3] for b in blocks) + gap * max(0, len(blocks) - 1)
    ghost_h = line_h
    if rows:
        content_h += (gap if blocks else 0.0) + len(rows) * ghost_h
    avail_h = area.height() - 2 * MARGIN - 2 * pad_y
    if max_h is not None:
        avail_h = min(avail_h, max_h - 2 * pad_y)
    view_max = max(line_h, min(avail_h, style.max_lines * line_h * 1.25 + name_h))
    target_h = min(content_h, view_max)
    if hold:
        target_h = max(target_h, min(feed.hold_h, view_max))
        feed.hold_h = target_h
    target_scroll = content_h - target_h

    # ширина - по самой широкой видимой реплике
    widest = 0.0
    for card, lay, show_name, h in blocks:
        widest = max(widest, lay.content_w if show_name else max(
            (x + tm.horizontalAdvance(w.text) for w, x, ln in lay.words), default=0.0))
    for g, _a in rows:
        widest = max(widest, nm.horizontalAdvance((g.speaker or '').upper()) + size * 2.2)
    target_w = min(max_w, widest + 2 * pad_x + m['indent'])

    feed.h = follow(feed.h, target_h, FEED_SIZE_TAU_S)
    feed.w = follow(feed.w, target_w, FEED_SIZE_TAU_S)
    feed.scroll = follow(feed.scroll, target_scroll, FEED_SCROLL_TAU_S)
    feed.ready = True
    if feed.born == 0:
        feed.born = now

    # прозрачность ленты: появление и затухание после тишины
    alpha = _smooth((now - feed.born) / CARD_APPEAR_MS) if dt > 0 else 1.0
    ttl = style.ttl_ms
    if model is not None and ttl > 0:
        idle = model.idle_since(now)
        if idle > ttl:
            alpha *= 1.0 - _smooth((idle - ttl) / CARD_FADE_MS)
    if alpha <= 0.005:
        return moving

    w = feed.w
    h = feed.h + 2 * pad_y
    x = area.left() + (area.width() - w) / 2 if style.align == 'center' else area.left() + MARGIN
    y = area.bottom() - MARGIN - h
    rect = QRectF(x, y, w, h)
    feed.rect = rect
    bg_a = style.bg_opacity
    radius = m['radius']
    if bg_a > 0.15:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, int(40 * bg_a * alpha)))
        p.drawRoundedRect(rect.adjusted(-2, 0, 2, 3), radius + 2, radius + 2)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(16, 17, 20, int(255 * bg_a * alpha)))
    p.drawRoundedRect(rect, radius, radius)
    p.setPen(QPen(QColor(255, 255, 255, int(16 * alpha)), 1.0))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)

    inner_top = y + pad_y
    inner_bottom = y + h - pad_y
    fade = line_h * 0.9

    def row_alpha(row_y: float) -> float:
        # строка уходит за верхний край ленты - растворяется
        top = min(1.0, max(0.0, 1.0 + (row_y - inner_top) / fade))
        bot = min(1.0, max(0.0, (inner_bottom - row_y) / line_h))
        return top * bot

    p.save()
    p.setClipRect(QRectF(x, y + 1, w, h - 2))
    cy = inner_top - feed.scroll
    tx = x + pad_x + m['indent']
    pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi / PULSE_MS)
    drop_above = []
    feed.tops = {}
    for i, (card, lay, show_name, bh) in enumerate(blocks):
        top = cy
        feed.tops[id(card)] = top           # для замеров плавности
        ca = alpha * _smooth((now - card.born) / CARD_APPEAR_MS)
        if card.gone_t:
            ca *= 1.0 - _smooth((now - card.gone_t) / CARD_FADE_MS)
        color = QColor(speaker_color(card.speaker))
        # полоса цвета говорящего; пульсирует, пока он говорит
        talking = card.stream in speaking
        bar_top = max(top, inner_top - 2)
        bar_bot = min(top + bh, inner_bottom + 2)
        if bar_bot - bar_top > 2:
            k = 0.55 + 0.45 * pulse if talking else 0.75
            ac = QColor(color)
            ac.setAlpha(int(255 * k * ca))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ac)
            p.drawRoundedRect(QRectF(x + pad_x * 0.55, bar_top + 2, ACCENT_W, bar_bot - bar_top - 4),
                              ACCENT_W / 2, ACCENT_W / 2)
        if show_name:
            na = ca * row_alpha(cy)
            if na > 0.01:
                p.setFont(name_f)
                nc = QColor(color)
                nc.setAlpha(int(235 * na))
                p.setPen(nc)
                nx = tx + (lay.content_w - lay.name_w) / 2 if style.align == 'center' else tx
                p.drawText(QPointF(nx, cy + nm.ascent()), lay.name.upper())
            cy += name_h
        renderer.paint_words(p, lay, tx, cy, ca, now, size, row_alpha, tr_color)
        cy += lay.lines * line_h
        if model is not None and not card.live() and cy < inner_top - line_h * KEEP_LINES:
            drop_above.append((card, cy - top + gap))
        elif model is not None and lay.lines > 2:
            # длинная реплика: строки, давно ушедшие наверх, выбрасываем
            gone = int((inner_top - (top + (name_h if show_name else 0.0))) / line_h) - KEEP_LINES
            if gone > 0:
                model.trim_lines(card, lay, gone)
                feed.scroll -= gone * line_h
        cy += gap

    for g, ga in rows:
        a = alpha * ga * row_alpha(cy)
        gx = tx
        color = QColor(speaker_color(g.speaker))
        p.setPen(Qt.PenStyle.NoPen)
        dc = QColor(color)
        dc.setAlpha(int(255 * (0.55 + 0.45 * pulse) * a))
        p.setBrush(dc)
        p.drawRoundedRect(QRectF(x + pad_x * 0.55, cy + line_h * 0.2, ACCENT_W, line_h * 0.6),
                          ACCENT_W / 2, ACCENT_W / 2)
        if renderer.named(g.speaker):
            p.setFont(name_f)
            nc = QColor(color)
            nc.setAlpha(int(235 * a))
            p.setPen(nc)
            p.drawText(QPointF(gx, cy + line_h / 2 + nm.ascent() / 2 - nm.descent() / 2),
                       g.speaker.upper())
            gx += nm.horizontalAdvance(g.speaker.upper()) + size * 0.5
        lvl = level_fn(g.stream) if level_fn else 0.6
        renderer.paint_bars(p, gx, cy + line_h / 2, line_h, a, now, size, lvl)
        cy += ghost_h
    p.restore()

    # реплики, давно уехавшие наверх, выбрасываем - прокрутку сдвигаем на их высоту
    for card, height in drop_above:
        model.drop_card(card)
        feed.scroll -= height
    return moving


# превью и редактор зовут под старым именем
paint_stack = paint_feed


# ============================================================
# РЕДАКТОР ПОЛОЖЕНИЯ
# ============================================================

class OverlayEditor(QWidget):
    """
    Обычное окно без click-through: его двигают и тянут за края.
    Внутри - примерные субтитры в текущем стиле.
    """
    geometry_changed = pyqtSignal(int, int, int, int)
    closed = pyqtSignal()

    def __init__(self, style: OverlayStyle):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.style_ = style
        self.renderer = CardRenderer()
        self._drag = False
        self._kind = None
        self._start_pos = QPoint()
        self._start_geom = QRect()
        self._hover = None
        self._cards = demo_cards('d')

    def set_font_size(self, size: int):
        self.style_.font_size = size
        self.update()

    def _hit(self, pos: QPoint) -> str:
        b = EDIT_BORDER_PX
        l, r = pos.x() < b, pos.x() > self.width() - b
        t, bt = pos.y() < b, pos.y() > self.height() - b
        if t and l: return 'tl'
        if t and r: return 'tr'
        if bt and l: return 'bl'
        if bt and r: return 'br'
        if l: return 'left'
        if r: return 'right'
        if t: return 'top'
        if bt: return 'bottom'
        return 'move'

    _CURSORS = {
        'move': Qt.CursorShape.SizeAllCursor,
        'left': Qt.CursorShape.SizeHorCursor, 'right': Qt.CursorShape.SizeHorCursor,
        'top': Qt.CursorShape.SizeVerCursor, 'bottom': Qt.CursorShape.SizeVerCursor,
        'tl': Qt.CursorShape.SizeFDiagCursor, 'br': Qt.CursorShape.SizeFDiagCursor,
        'tr': Qt.CursorShape.SizeBDiagCursor, 'bl': Qt.CursorShape.SizeBDiagCursor,
    }

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = True
        self._kind = self._hit(e.position().toPoint())
        self._start_pos = e.globalPosition().toPoint()
        self._start_geom = self.geometry()

    def mouseDoubleClickEvent(self, e):
        self.close()

    def mouseMoveEvent(self, e):
        if not self._drag:
            self._hover = self._hit(e.position().toPoint())
            self.setCursor(self._CURSORS.get(self._hover, Qt.CursorShape.ArrowCursor))
            return
        screen = QApplication.primaryScreen().geometry()
        d = e.globalPosition().toPoint() - self._start_pos
        g = QRect(self._start_geom)
        k = self._kind
        if k == 'move':
            g.translate(d)
        else:
            if k in ('left', 'tl', 'bl'):
                g.setLeft(min(g.left() + d.x(), g.right() - EDIT_MIN_W))
            if k in ('right', 'tr', 'br'):
                g.setRight(max(g.right() + d.x(), g.left() + EDIT_MIN_W))
            if k in ('top', 'tl', 'tr'):
                g.setTop(min(g.top() + d.y(), g.bottom() - EDIT_MIN_H))
            if k in ('bottom', 'bl', 'br'):
                g.setBottom(max(g.bottom() + d.y(), g.top() + EDIT_MIN_H))
        if g.left() < screen.left(): g.moveLeft(screen.left())
        if g.top() < screen.top(): g.moveTop(screen.top())
        if g.right() > screen.right(): g.moveRight(screen.right())
        if g.bottom() > screen.bottom(): g.moveBottom(screen.bottom())
        self.setGeometry(g)
        self.geometry_changed.emit(g.x(), g.y(), g.width(), g.height())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = False
            self._kind = None

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.close()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        now = _now()

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(12, 13, 16, 130))
        p.drawRoundedRect(r, 12, 12)

        pen = QPen(QColor(255, 255, 255, 150), 1.2, Qt.PenStyle.DashLine)
        pen.setDashPattern([5, 4])
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 12, 12)

        # ручки по углам - внутри окна, чтобы не срезались краем
        for cx, cy in ((r.left() + 8, r.top() + 8), (r.right() - 8, r.top() + 8),
                       (r.left() + 8, r.bottom() - 8), (r.right() - 8, r.bottom() - 8)):
            p.setPen(QPen(QColor(12, 13, 16), 1.5))
            p.setBrush(QColor(236, 237, 240))
            p.drawEllipse(QPointF(cx, cy), 4, 4)

        paint_stack(p, QRectF(self.rect()), self.renderer, self._cards, [],
                    OverlayStyle(self.style_.font_size, self.style_.bg_opacity,
                                 self.style_.align, 0, self.style_.max_lines), now, 0.0)

        # значок перемещения по центру
        c = QPointF(r.center())
        p.setPen(QPen(QColor(255, 255, 255, 30), 1))
        p.setBrush(QColor(22, 23, 27, 220))
        p.drawEllipse(c, 21, 21)
        pen = QPen(QColor(236, 237, 240), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                   Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        s = 9
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tip = QPointF(c.x() + dx * s, c.y() + dy * s)
            p.drawLine(c, tip)
            path = QPainterPath()
            path.moveTo(tip.x() - dy * 3 - dx * 3, tip.y() - dx * 3 - dy * 3)
            path.lineTo(tip)
            path.lineTo(tip.x() + dy * 3 - dx * 3, tip.y() + dx * 3 - dy * 3)
            p.drawPath(path)
        p.end()

    def closeEvent(self, e):
        self.closed.emit()
        super().closeEvent(e)


# ============================================================
# ОЧЕРЕДЬ СОБЫТИЙ (вызывается из любых потоков)
# ============================================================

def drain_queue():
    try:
        while True:
            event_q.get_nowait()
    except queue.Empty:
        pass


def push_text(stream: str, seg, speaker: str, done: str, draft: str, final: bool,
              ends: Optional[list] = None):
    """done - подтверждённый текст куска, draft - черновой хвост после него."""
    event_q.put({'type': 'text', 'stream': stream, 'seg': seg, 'speaker': speaker,
                 'done': done, 'draft': draft, 'final': bool(final), 'ends': ends})


def push_trans(stream: str, seg, speaker: str, text: str, draft: str = '', end: float = 0.0):
    """
    Перевод куска: законченные предложения (растёт) и черновик недоговорённого.
    end - конец в звуке последнего переведённого слова (для замеров).
    """
    event_q.put({'type': 'trans', 'stream': stream, 'seg': seg, 'speaker': speaker,
                 'text': text, 'draft': draft, 'end': end})


def push_drop(stream: str, seg):
    event_q.put({'type': 'drop', 'stream': stream, 'seg': seg})


def push_speaking(stream: str, speaker: Optional[str], active: bool):
    event_q.put({'type': 'speaking', 'stream': stream, 'speaker': speaker,
                 'active': bool(active)})


def push_clear():
    drain_queue()
    event_q.put({'type': 'clear'})
