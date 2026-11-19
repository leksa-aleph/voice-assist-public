"""
Потоковый ресемплер без артефактов на стыках блоков.

scipy.signal.resample_poly на каждом чанке по отдельности дополняет края
нулями: на стыке каждого блока (у ProcTap это каждые 10 мс) амплитуда
проседает, и в звуке появляется дребезг. STT от этого хуже различает
согласные. Здесь фильтр помнит хвост предыдущего блока, поэтому поток
режется на любые куски и результат совпадает с обработкой целиком.
"""
from __future__ import annotations

from math import gcd

import numpy as np

# Сколько входных отсчётов покрывает фильтр. 96 даёт подавление
# наложения около 60 дБ при переходе 48 кГц -> 16 кГц.
TAPS_PER_PHASE = 96


def _lowpass(numtaps: int, cutoff: float, fs: float, beta: float) -> np.ndarray:
    """ФНЧ с окном Кайзера, как scipy.signal.firwin - без scipy в зависимостях."""
    fc = cutoff / (fs / 2)
    m = np.arange(numtaps) - (numtaps - 1) / 2
    h = fc * np.sinc(fc * m) * np.kaiser(numtaps, beta)
    return h / h.sum()


class StreamResampler:
    def __init__(self, src_rate: int, dst_rate: int):
        self.src = int(src_rate)
        self.dst = int(dst_rate)
        self.passthrough = self.src == self.dst
        if self.passthrough:
            return

        g = gcd(self.src, self.dst)
        self.up = self.dst // g
        self.down = self.src // g

        L = TAPS_PER_PHASE
        cutoff = 0.45 * min(self.src, self.dst)
        h = _lowpass(L * self.up, cutoff, self.src * self.up, 8.0) * self.up
        # H[p, j] = h[p + j*up]: фильтр для фазы p
        self._H = h.reshape(L, self.up).T.astype(np.float32).copy()
        self._L = L
        self._taps = np.arange(L)

        self._hist = np.zeros(L - 1, dtype=np.float32)
        self._n_in = 0       # сколько входных отсчётов получено всего
        self._M = 0          # позиция следующего выхода в повышенной частоте

    def reset(self):
        if self.passthrough:
            return
        self._hist[:] = 0.0
        self._n_in = 0
        self._M = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.passthrough or x.size == 0:
            return x

        ext = np.concatenate([self._hist, x])
        ext_base = self._n_in - self._hist.size
        self._n_in += x.size

        last_m = self.up * self._n_in - 1
        if last_m < self._M:
            self._hist = ext[-(self._L - 1):]
            return np.zeros(0, dtype=np.float32)

        count = (last_m - self._M) // self.down + 1
        ms = self._M + self.down * np.arange(count, dtype=np.int64)
        ns = ms // self.up
        ps = ms % self.up

        idx = (ns - ext_base)[:, None] - self._taps[None, :]
        y = np.einsum('ij,ij->i', ext[idx], self._H[ps])

        self._M += self.down * count
        self._hist = ext[-(self._L - 1):]
        return y.astype(np.float32, copy=False)


class Rebuffer:
    """Режет поток на блоки ровно по block_size отсчётов."""

    def __init__(self, block_size: int):
        self.block = int(block_size)
        self._buf = np.zeros(0, dtype=np.float32)

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        if x.size:
            self._buf = np.concatenate([self._buf, x.astype(np.float32, copy=False)])
        n = self._buf.size // self.block
        if n == 0:
            return []
        out = [self._buf[i * self.block:(i + 1) * self.block].copy()
               for i in range(n)]
        self._buf = self._buf[n * self.block:]
        return out
