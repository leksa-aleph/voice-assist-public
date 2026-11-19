"""
Перевод английского на русский без PyTorch: Marian (Helsinki-NLP
opus-mt-tc-big-en-zle, Apache-2.0) в ONNX int8 и свой токенизатор
SentencePiece Unigram на Python.

Фраза из 15 слов переводится за ~80-160 мс на одном ядре. Законченное
предложение переводится начисто. Недоговорённое - черновиком, и каждый
следующий черновик продолжает начало прошлого (translate_ids с force):
иначе перевод начала фразы менялся бы с каждым новым словом.

Файлы - models/translate/ (готовит tools/export_translate.py):
encoder.int8.onnx, decoder.int8.onnx, translate.json (словарь и номера
служебных слов).
"""
from __future__ import annotations

import json
import threading
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

from asr_onnx import session_options

MAX_LEN = 200


class _Unigram:
    """SentencePiece Unigram: разбиение слова на куски с наибольшей суммой весов."""

    def __init__(self, source: list, unk: int):
        self.pieces = {p: (i, s) for p, s, i in source}
        self.unk = unk
        self.max_len = max(len(p) for p, _, _ in source)

    def encode_word(self, w: str) -> list[int]:
        n = len(w)
        best: list[tuple[float, int, int]] = [(-1e18, -1, -1)] * (n + 1)   # (вес, начало, id)
        best[0] = (0.0, -1, -1)
        for end in range(1, n + 1):
            for start in range(max(0, end - self.max_len), end):
                if best[start][0] <= -1e18:
                    continue
                hit = self.pieces.get(w[start:end])
                if hit is None:
                    if end - start == 1:
                        sc = best[start][0] - 20.0      # неизвестный символ
                        if sc > best[end][0]:
                            best[end] = (sc, start, self.unk)
                    continue
                sc = best[start][0] + hit[1]
                if sc > best[end][0]:
                    best[end] = (sc, start, hit[0])
        ids = []
        end = n
        while end > 0:
            _, start, i = best[end]
            ids.append(i)
            end = start
        return ids[::-1]

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize('NFKC', text)
        ids = []
        for w in text.split():
            ids += self.encode_word('▁' + w)
        return ids


class Translator:
    def __init__(self, model_dir: Path, threads: int = 1):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / 'translate.json').read_text(encoding='utf-8'))
        self.tok = _Unigram(meta['source'], meta['unk'])
        self.target_vocab: list[str] = meta['target_vocab']
        self.pad, self.eos, self.start = meta['pad'], meta['eos'], meta['start']
        self.lang = meta['target']
        self.heads, self.head_dim = meta['heads'], meta['head_dim']
        self._skip = {self.pad, self.eos, meta['unk']}
        so = session_options(threads)
        self._enc = ort.InferenceSession(str(model_dir / 'encoder.int8.onnx'), so,
                                         providers=['CPUExecutionProvider'])
        self._dec = ort.InferenceSession(str(model_dir / 'decoder.int8.onnx'), so,
                                         providers=['CPUExecutionProvider'])
        self._past = [i.name for i in self._dec.get_inputs() if i.name.startswith('past_key_values')]
        self._outs = [o.name for o in self._dec.get_outputs()]
        self._lock = threading.Lock()

    def translate(self, text: str) -> str:
        return self.translate_ids(text)[0]

    def translate_ids(self, text: str, force: tuple = ()) -> tuple[str, list[int]]:
        """
        Перевод и его токены. force - начало перевода, которое уже на экране:
        модель продолжает его, а не переводит заново. Так черновик перевода
        недоговорённой фразы растёт, а не переписывается с каждым словом.
        """
        return self.translate_scored(text, force)[:2]

    def translate_scored(self, text: str, force: tuple = ()) -> tuple[str, list[int], float]:
        """То же и средний логарифм вероятности токена (вместе с концом): насколько модель согласна."""
        text = text.strip()
        if not text:
            return '', [], 0.0
        src = [self.lang] + self.tok.encode(text)[:MAX_LEN] + [self.eos]
        ids = np.array([src], dtype=np.int64)
        mask = np.ones_like(ids)
        with self._lock:
            (enc,) = self._enc.run(['last_hidden_state'], {'input_ids': ids, 'attention_mask': mask})
            past = {n: np.zeros((1, self.heads, 0, self.head_dim), dtype=np.float32) for n in self._past}
            out: list[int] = []
            logp = 0.0
            token = self.start
            for step in range(min(MAX_LEN, max(2 * len(src) + 10, len(force) + 10))):
                feeds = {'encoder_attention_mask': mask, 'input_ids': np.array([[token]], dtype=np.int64),
                         'encoder_hidden_states': enc, 'use_cache_branch': np.array([step > 0])}
                feeds.update(past)
                res = self._dec.run(None, feeds)
                logits = res[0][0, -1]
                logits[self.pad] = -np.inf
                token = force[step] if step < len(force) else int(logits.argmax())
                top = float(logits.max())
                logp += float(logits[token]) - top - float(np.log(np.exp(logits - top).sum()))
                if token == self.eos:
                    break
                out.append(token)
                for name, val in zip(self._outs[1:], res[1:]):
                    key = name.replace('present', 'past_key_values')
                    if step > 0 and '.encoder.' in key:
                        continue            # перекрёстное внимание считается один раз
                    past[key] = val
        return self.decode(out), out, logp / (len(out) + 1)

    def decode(self, ids: list[int]) -> str:
        text = ''.join(self.target_vocab[i] for i in ids if i not in self._skip)
        return text.replace('▁', ' ').strip()


def model_dir() -> Path:
    from paths import resource_path
    return Path(resource_path('models/translate'))


def available() -> bool:
    return (model_dir() / 'translate.json').exists()


def is_russian(text: str) -> bool:
    """Фраза на русском (кириллицы не меньше латиницы) - переводить не нужно."""
    cyr = sum(1 for ch in text if 'Ѐ' <= ch <= 'ӿ')
    lat = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    return cyr >= lat
