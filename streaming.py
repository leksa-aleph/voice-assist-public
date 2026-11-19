"""
Потоковое распознавание поверх нестримовой модели (LocalAgreement).

GigaAM видит фразу целиком, поэтому раньше каждый черновик заново
распознавал весь буфер: нагрузка росла квадратично от длины фразы,
а модель с новым контекстом переписывала уже показанный хвост -
«+17 слов» за одно обновление.

Здесь каждое обновление распознаёт только окно: 1.5 с уже подтверждённого
звука для контекста плюс всё, что после. Слово подтверждается, когда
две гипотезы подряд согласны (или у него уже 2.5 с правого контекста -
дальше модель его не меняет). Подтверждённое больше не пересчитывается,
граница сдвигается вслед за ним, текст растёт только слева направо.

Время везде - секунды от начала фразы.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from asr_onnx import Word

LEFT_CTX_S = 1.5        # подтверждённый звук перед окном - контекст для модели
MAX_WINDOW_S = 10.0     # потолок окна, даже если подтверждать нечего
TAIL_GUARD_S = 0.65     # подтверждаем, когда модель уже слышала продолжение (точка, запятая)
FORCE_CTX_S = 2.5       # с таким правым контекстом слово подтверждается и без согласия
MATCH_TOL_S = 0.45      # время одного слова в разных окнах гуляет до ~0.3 с
DRIFT_S = 0.25
STALL_S = 3.0           # столько звука без слов (музыка, шум) - окно не растим
DRAFT_GUARD_S = 0.2     # последнее слово черновика у края окна обычно обрезано
DRAFT_HOLD = 2          # столько окон подряд держим хвост черновика, который модель «проглотила»
# Окна от начала предложения (для Parakeet): окно, начатое посреди фразы, эта
# модель оставляет пустым или выкидывает его начало - она учена на целых
# предложениях. Поэтому окно начинается сразу после последней точки.
SENT_MAX_WINDOW_S = 12.0
SENT_STALL_S = 6.0
_SENTENCE_END = ('.', '?', '!', '…')
_CLAUSE_END = (',', ';', ':')
# Точка после сокращения - не конец предложения: иначе «Mr.» уходит
# переводчику отдельно («Мистер.»), а окно начинается посреди фразы.
_ABBREV = {'mr', 'mrs', 'ms', 'dr', 'prof', 'st', 'jr', 'sr', 'vs', 'mt', 'fig', 'approx', 'dept', 'inc',
           'др', 'им', 'ул', 'проф', 'стр', 'см', 'тыс', 'млн', 'млрд', 'руб', 'пр'}

_EDGE_PUNCT = '.,!?;:—–-«»"\'()…'


def ends_sentence(word: str) -> bool:
    """Слово кончает предложение: точка, вопрос, восклицание - но не «Mr.» и не «J.»."""
    w = word.rstrip('»"\')')
    if not w.endswith(_SENTENCE_END):
        return False
    if w.endswith('.') and not w.endswith('..'):
        core = w[:-1].lstrip('«"\'(').lower()
        if core in _ABBREV:
            return False
        if len(core) == 1 and core.isalpha():
            return False                    # инициал: J. R. R. Tolkien
        if '.' in core and len(core) <= 5:
            return False                    # e.g. i.e. U.S. т.е.
    return True


def norm(word: str) -> str:
    return word.strip(_EDGE_PUNCT).lower().replace('ё', 'е')


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def join(words: list[Word]) -> str:
    return ' '.join(w.text for w in words)


def _core(word: str) -> tuple[str, str, str]:
    """(знаки до, само слово, знаки после)."""
    a = len(word) - len(word.lstrip(_EDGE_PUNCT))
    b = len(word.rstrip(_EDGE_PUNCT))
    return word[:a], word[a:b], word[b:]


class Transcript:
    """Подтверждённые слова фразы, черновой хвост и граница окна."""

    def __init__(self, draft_guard_s: float = DRAFT_GUARD_S, sentence_windows: bool = False):
        self.draft_guard_s = draft_guard_s
        self.sentence_windows = sentence_windows
        self.sent_t = 0.0               # начало текущего предложения (для окон от предложения)
        self.clause_t = 0.0             # начало текущей части после запятой
        self.words: list[Word] = []     # подтверждённые
        self.draft: list[Word] = []     # последняя гипотеза после границы
        self.commit_t = 0.0             # граница: всё раньше уже подтверждено
        self.end_s = 0.0                # конец окна последней гипотезы
        self._prev: list[str] = []
        self._held = 0

    def shown_draft(self) -> str:
        """
        Черновик для экрана. Слово у самого края окна обычно обрезано
        («завязыва», «пери») - его не показываем, пока не договорено.
        Точку в конце модель ставит просто потому, что звук кончился, -
        её тоже не показываем, иначе она мигает.
        """
        words = self.shown_draft_words()
        if not words:
            return ''
        text = join(words)
        return text.rstrip('.,…') or text

    def shown_draft_words(self) -> list[Word]:
        return [w for w in self.draft if w.end <= self.end_s - self.draft_guard_s]

    def window_start(self, end_s: float) -> float:
        if self.sentence_windows:
            # предложение длиннее окна - от последней запятой: с начала части
            # фразы модель слышит лучше, чем с середины
            for t in (self.sent_t, self.clause_t):
                if end_s - t <= SENT_MAX_WINDOW_S:
                    return max(0.0, t)
        return max(0.0, self.commit_t - LEFT_CTX_S, end_s - MAX_WINDOW_S)

    def _new_part(self, hyp: list[Word], start_s: float = 0.0) -> tuple[list[Word], bool]:
        """
        Отрезает от гипотезы слова контекста. Время одного и того же слова
        в разных окнах гуляет до 0.3 с, поэтому стык ищем по тексту:
        по последнему подтверждённому слову (с допуском на распознанное
        чуть иначе: «френрейт» / «фрейнрейт»), затем по предпоследнему.
        Не нашли - по времени, с запасом, и отсекаем повтор хвоста.
        Окно, начатое за подтверждённым (новое предложение у Parakeet),
        контекста не содержит: «...you prefer. | You have» - оба you нужны.
        """
        if not self.words:
            return [w for w in hyp if w.start >= self.commit_t - 0.02], False
        if start_s >= self.commit_t - 0.05:
            return list(hyp), False
        changed = False
        last = self.words[-1]
        i = self._find(hyp, last)
        if i is not None:
            same = hyp[i]
            # знак препинания после слова модель ставит, только увидев
            # следующее: «отпустило» -> «отпустило.» - даём ему догнать.
            # Само слово не трогаем: у начала окна модель ставит заглавную
            # без причины («окно Открываешь»).
            if norm(same.text) == norm(last.text):
                pre, core, _ = _core(last.text)
                text = pre + core + _core(same.text)[2]
                if text != last.text:
                    last.text = text
                    changed = True
            return hyp[i + 1:], changed
        if len(self.words) > 1:
            i = self._find(hyp, self.words[-2])
            if i is not None:
                rest = hyp[i + 1:]
                # следующее за ним - искажённое последнее подтверждённое
                if rest and rest[0].start < self.commit_t - 0.1:
                    rest = rest[1:]
                return rest, changed
        rest = [w for w in hyp if w.start >= self.commit_t - DRIFT_S]
        # повтор - это хвост подтверждённого в том же порядке, а не любое похожее слово
        done = [norm(w.text) for w in self.words[-2:]]
        for k in (2, 1):
            head = [norm(w.text) for w in rest[:k]]
            if len(head) == k and len(done) >= k and all(
                    _similar(a, b) >= 0.6 for a, b in zip(head, done[-k:])):
                return rest[k:], changed
        return rest, changed

    @staticmethod
    def _find(hyp: list[Word], ref: Word):
        """Индекс того же слова в гипотезе или None."""
        key = norm(ref.text)
        best, best_score = None, 0.0
        for i, w in enumerate(hyp):
            dt = abs(w.start - ref.start)
            if w.start > ref.start + MATCH_TOL_S:
                break
            if dt > MATCH_TOL_S:
                continue
            sim = _similar(norm(w.text), key)
            if sim >= 0.6 and sim - dt > best_score:
                best, best_score = i, sim - dt
        return best

    def refine(self, hyp: list[Word], start_s: float) -> bool:
        """
        Фраза закрыта: hyp - её конец (с start_s), распознанный одним окном.
        Подтверждённым словам оттуда берём только знаки и заглавные: окна
        потока видят фразу кусками, и точку в конце она теряла (F1 точек
        75% против 95% у распознавания целиком). Сами слова не меняем -
        их уже прочитали. True - что-то поменялось.
        """
        first = next((i for i, w in enumerate(self.words) if w.start >= start_s - 0.05), len(self.words))
        old = self.words[first:]
        sm = SequenceMatcher(None, [norm(w.text) for w in old], [norm(w.text) for w in hyp], autojunk=False)
        changed = False
        for blk in sm.get_matching_blocks():
            for k in range(blk.size):
                w, h = old[blk.a + k], hyp[blk.b + k]
                if w.text != h.text:
                    w.text = h.text
                    changed = True
        return changed

    def update(self, hyp: list[Word], end_s: float, final: bool, start_s: float = 0.0) -> bool:
        """Новая гипотеза окна [start_s, end_s]. True - текст изменился."""
        new, changed = self._new_part(hyp, start_s)
        self.end_s = end_s
        texts = [norm(w.text) for w in new]
        if final:
            n = len(new)
        else:
            agree = 0
            for a, b in zip(texts, self._prev):
                if a != b:
                    break
                agree += 1
            n = 0
            for i, w in enumerate(new):
                if (i < agree and w.end <= end_s - TAIL_GUARD_S) or w.end <= end_s - FORCE_CTX_S:
                    n = i + 1
                else:
                    break
        if n:
            self.words.extend(new[:n])
            if n < len(new):
                edge = (new[n - 1].end + new[n].start) / 2
            else:
                edge = new[n - 1].end + 0.04
            self.commit_t = max(self.commit_t, edge)
            changed = True
            for k in range(n - 1, -1, -1):
                nxt = new[k + 1].start if k + 1 < len(new) else new[k].end + 0.08
                at = min(new[k].end + 0.12, (new[k].end + nxt) / 2)
                if new[k].text.endswith(_CLAUSE_END):
                    self.clause_t = max(self.clause_t, at)
                if ends_sentence(new[k].text):
                    self.sent_t = max(self.sent_t, at)
                    self.clause_t = max(self.clause_t, at)
                    break
        draft = new[n:]
        old = [norm(w.text) for w in self.draft]
        rest = self.draft[n:]
        if (not final and self._held < DRAFT_HOLD and len(draft) < len(rest)
                and texts == old[:len(texts)]):
            # Окно потеряло хвост черновика, который прошлое окно слышало
            # (Parakeet так делает на стыке слов). Держим прежний: иначе слова
            # на экране пропадают и через окно появляются снова.
            draft = rest
            self._held += 1
        else:
            self._held = 0
        if [w.text for w in draft] != [w.text for w in self.draft]:
            changed = True
        self.draft = draft
        self._prev = texts[n:]
        stall = SENT_STALL_S if self.sentence_windows else STALL_S
        if not final and not new and end_s - self.commit_t > stall:
            self.commit_t = end_s - LEFT_CTX_S
            self.sent_t = max(self.sent_t, self.commit_t)
            self.clause_t = max(self.clause_t, self.commit_t)
        return changed
