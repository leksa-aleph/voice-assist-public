"""
Общие часы анимации.

Один точный таймер на всё окно, не чаще 60 кадров в секунду: на экране
165 Гц глаз разницы почти не видит, а процессор - видит.
Виджет просит кадры через clock().wake(self) и получает on_frame(dt),
пока возвращает True. Когда никто не анимируется, таймер стоит.
"""
from __future__ import annotations

import math
import time
import weakref

from PyQt6.QtCore import QObject, Qt, QTimer
from PyQt6.QtWidgets import QApplication

MAX_FPS = 60.0


class Anim:
    """Значение, которое экспоненциально тянется к цели."""

    __slots__ = ('value', 'target', 'rate')

    def __init__(self, value: float = 0.0, rate: float = 14.0):
        self.value = float(value)
        self.target = float(value)
        self.rate = rate

    def set(self, target: float, instant: bool = False):
        self.target = float(target)
        if instant:
            self.value = self.target

    def step(self, dt: float) -> bool:
        d = self.target - self.value
        if abs(d) < 1e-4:
            self.value = self.target
            return False
        self.value += d * (1.0 - math.exp(-self.rate * dt))
        return True

    @property
    def moving(self) -> bool:
        return abs(self.target - self.value) >= 1e-4


class Spring:
    """Пружина с перелётом - для живых индикаторов."""

    __slots__ = ('value', 'target', 'vel', 'k', 'c')

    def __init__(self, value: float = 0.0, stiffness: float = 260.0, damping: float = 22.0):
        self.value = float(value)
        self.target = float(value)
        self.vel = 0.0
        self.k = stiffness
        self.c = damping

    def set(self, target: float, instant: bool = False):
        self.target = float(target)
        if instant:
            self.value = self.target
            self.vel = 0.0

    def step(self, dt: float) -> bool:
        # полунеявный Эйлер, шагами не больше 4 мс - устойчиво на любом fps
        n = max(1, int(math.ceil(dt / 0.004)))
        h = dt / n
        for _ in range(n):
            a = self.k * (self.target - self.value) - self.c * self.vel
            self.vel += a * h
            self.value += self.vel * h
        if abs(self.target - self.value) < 1e-4 and abs(self.vel) < 1e-3:
            self.value = self.target
            self.vel = 0.0
            return False
        return True


class _Clock(QObject):
    def __init__(self):
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        screen = QApplication.primaryScreen()
        rate = screen.refreshRate() if screen else 60
        self._interval = max(16, int(1000 / min(MAX_FPS, max(30.0, rate or 60))))
        self._subs: list[weakref.ref] = []
        self._last = time.perf_counter()
        self.t = 0.0

    def wake(self, w):
        for r in self._subs:
            if r() is w:
                break
        else:
            self._subs.append(weakref.ref(w))
        if not self._timer.isActive():
            self._last = time.perf_counter()
            self._timer.start(self._interval)

    def _tick(self):
        now = time.perf_counter()
        dt = min(0.05, now - self._last)
        self._last = now
        self.t += dt
        alive = []
        for r in self._subs:
            w = r()
            if w is None:
                continue
            try:
                if w.on_frame(dt):
                    alive.append(r)
            except RuntimeError:
                # C++-объект уже удалён
                continue
        self._subs = alive
        if not alive:
            self._timer.stop()


_clock = None


def clock() -> _Clock:
    global _clock
    if _clock is None:
        _clock = _Clock()
    return _clock


def now() -> float:
    return time.perf_counter()
