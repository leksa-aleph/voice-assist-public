"""
Быстрые тесты без моделей и GPU:
    python -m unittest tests.test_core -v
"""
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


class ResamplerTest(unittest.TestCase):
    def test_chunking_matches_whole_signal(self):
        from resample import StreamResampler
        rng = np.random.default_rng(1)
        x = rng.standard_normal(48000).astype(np.float32)
        whole = StreamResampler(48000, 16000).process(x)
        r = StreamResampler(48000, 16000)
        parts, i = [], 0
        while i < len(x):
            n = int(rng.integers(1, 3000))
            parts.append(r.process(x[i:i + n]))
            i += n
        self.assertTrue(np.allclose(np.concatenate(parts), whole, atol=1e-5))
        self.assertEqual(len(whole), 16000)

    def test_passband_and_alias_rejection(self):
        from resample import StreamResampler

        def amp(y, f, sr):
            t = np.arange(y.size) / sr
            return 2 * np.hypot((y * np.cos(2 * np.pi * f * t)).mean(),
                                (y * np.sin(2 * np.pi * f * t)).mean())

        for src in (48000, 44100):
            t = np.arange(src) / src
            y = StreamResampler(src, 16000).process(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
            self.assertAlmostEqual(amp(y[300:], 1000, 16000), 1.0, delta=0.02)
            y = StreamResampler(src, 16000).process(np.sin(2 * np.pi * 10000 * t).astype(np.float32))
            self.assertLess(amp(y[300:], 6000, 16000), 0.01)

    def test_rebuffer(self):
        from resample import Rebuffer
        rb = Rebuffer(512)
        out = rb.push(np.ones(700, np.float32)) + rb.push(np.ones(400, np.float32))
        self.assertEqual([len(b) for b in out], [512, 512])


class NativePcmTest(unittest.TestCase):
    def test_int16_and_float_stereo_to_mono(self):
        from audio_source import decode_pcm
        st = np.array([[0.5, -0.5], [0.25, 0.25]], np.float32)
        self.assertTrue(np.allclose(decode_pcm(st.tobytes(), 2, 4), [0.0, 0.25]))
        i16 = (st * 32767).astype(np.int16)
        self.assertTrue(np.allclose(decode_pcm(i16.tobytes(), 2, 2), [0.0, 0.25], atol=1e-3))

    def test_converter_is_replaced(self):
        import audio_source
        audio_source._patch_proctap()
        import proctap.backends.windows as w
        self.assertIs(w.AudioConverter, audio_source._NativePCM)
        c = w.AudioConverter(src_rate=44100, src_channels=2, src_width=2, src_format='int16',
                             dst_rate=48000, dst_channels=2, dst_width=4, dst_format='float32')
        self.assertEqual((c.rate, c.channels, c.width), (44100, 2, 2))
        self.assertEqual(c.convert(b'abcd'), b'abcd')


class StreamingTest(unittest.TestCase):
    """Подтверждение слов поверх окон (streaming.Transcript)."""

    @staticmethod
    def hyp(*items):
        from asr_onnx import Word
        return [Word(t, s, s + 0.3) for t, s in items]

    def test_sentence_windows_start_after_last_full_stop(self):
        from streaming import Transcript
        tr = Transcript(sentence_windows=True)
        self.assertEqual(tr.window_start(5.0), 0.0)    # до первой точки - от начала фразы
        h = self.hyp(('Hello', 0.2), ('there.', 0.6), ('Next', 1.4), ('one', 1.8), ('goes', 2.2))
        tr.update(h, 4.0, False)
        tr.update(h, 4.4, False)
        self.assertEqual([w.text for w in tr.words][:2], ['Hello', 'there.'])
        self.assertGreater(tr.window_start(5.0), 0.85)   # после «there.», а не за 1.5 с до границы
        self.assertLess(tr.window_start(5.0), 1.45)
        plain = Transcript()
        plain.update(h, 4.0, False)
        plain.update(h, 4.4, False)
        self.assertEqual(plain.window_start(5.0), max(0.0, plain.commit_t - 1.5))

    def test_translation_queue_takes_finished_sentences(self):
        from asr_onnx import Word
        import engine
        sent, drafts = [], []

        class Tr:
            def submit(self, job):
                sent.append(job[5])

            def draft(self, run, key, seg, who, text, end=0.0):
                drafts.append(text)

        u = engine._Utt('mon', 'mon', 1, None, 0.0)
        pc = u.pieces[0]
        words = [Word(t, i * 0.3, i * 0.3 + 0.2) for i, t in enumerate(
            'Okay everyone. Today Mr. Smith talks about recursion. And'.split())]
        engine.TranscriberEngine._queue_translation(Tr(), u, 0, pc, words, False)
        # «Mr.» - не конец предложения: перевод не режется на «Мистер.»
        self.assertEqual(sent, ['Okay everyone. Today Mr. Smith talks about recursion.'])
        engine.TranscriberEngine._queue_translation(Tr(), u, 0, pc, words, False)
        self.assertEqual(len(sent), 1)                   # уже отдано - второй раз не шлём
        engine.TranscriberEngine._queue_translation(Tr(), u, 0, pc, words, True)
        self.assertEqual(sent[-1], 'And')               # кусок закрылся - остаток тоже
        import translate_onnx
        self.assertTrue(translate_onnx.is_russian('Если нет базового случая.'))
        self.assertFalse(translate_onnx.is_russian('Exactly, a stack overflow.'))

    def test_unfinished_sentence_goes_as_draft(self):
        from asr_onnx import Word
        import engine
        drafts = []

        class Tr:
            def submit(self, job):
                pass

            def draft(self, run, key, seg, who, text, end=0.0):
                drafts.append(text)

        u = engine._Utt('mon', 'mon', 1, None, 0.0)
        pc = u.pieces[0]
        ws = 'So a recursive function is a function that calls itself'.split()
        words = [Word(t, i * 0.3, i * 0.3 + 0.2) for i, t in enumerate(ws)]
        q = engine.TranscriberEngine._queue_translation
        q(Tr(), u, 0, pc, words[:1], False)
        self.assertEqual(drafts, [])                     # одно слово - рано
        q(Tr(), u, 0, pc, words[:2], False, words[2:3])  # с неподтверждённым хвостом
        self.assertEqual(drafts, ['So a recursive'])
        q(Tr(), u, 0, pc, words[:3], False, words[3:4])
        self.assertEqual(len(drafts), 1)                 # слишком часто - ждём
        pc.tr_draft_t -= 5
        q(Tr(), u, 0, pc, words[:3], False, words[3:4])
        self.assertEqual(drafts[-1], 'So a recursive function')
        pc.tr_draft_t -= 5
        q(Tr(), u, 0, pc, words[:3], False, words[3:4])
        self.assertEqual(len(drafts), 2)                 # тот же текст - не переводим снова

    def test_sentence_end_skips_abbreviations(self):
        from streaming import ends_sentence
        for w in ('end.', 'today?', 'wow!', '"Yes."', 'so...', 'дела.'):
            self.assertTrue(ends_sentence(w), w)
        for w in ('Mr.', 'Dr.', 'J.', 'e.g.', 'U.S.', 'т.е.', 'and'):
            self.assertFalse(ends_sentence(w), w)

    def test_long_sentence_window_starts_at_last_comma(self):
        from streaming import Transcript
        tr = Transcript(sentence_windows=True)
        words = [(f'w{i}' + (',' if i == 20 else ''), i * 0.5) for i in range(30)]
        h = self.hyp(*words)
        tr.update(h, 16.0, False)
        tr.update(h, 16.3, False)
        self.assertEqual(tr.sent_t, 0.0)
        start = tr.window_start(16.3)            # от начала предложения - больше 12 с
        self.assertGreater(start, 10.3)          # после «w20,» (10.0-10.3)
        self.assertLess(start, 10.5)

    def test_new_sentence_keeps_word_like_previous_tail(self):
        from streaming import Transcript
        tr = Transcript(sentence_windows=True)
        h = self.hyp(('whatever', 0.2), ('you', 0.6), ('prefer.', 0.9))
        tr.update(h, 2.0, False)
        tr.update(h, 2.3, False)
        self.assertEqual([w.text for w in tr.words][-1], 'prefer.')
        start = tr.window_start(4.0)
        # новое окно - от начала предложения: «You» в нём не повтор
        tr.update(self.hyp(('You', 1.3), ('have', 1.6), ('ten', 1.9), ('minutes.', 2.2)), 4.0, True, start)
        self.assertEqual(' '.join(w.text for w in tr.words), 'whatever you prefer. You have ten minutes.')
        # а повтор хвоста по порядку отсекается и без окна от предложения
        plain = Transcript()
        plain.update(self.hyp(('one', 0.1), ('two', 0.5)), 2.0, False)
        plain.update(self.hyp(('one', 0.1), ('two', 0.5)), 2.4, False)
        plain.commit_t = 3.0
        plain.update(self.hyp(('tw', 2.9), ('three', 3.2)), 4.0, True, 1.5)
        self.assertEqual(' '.join(w.text for w in plain.words), 'one two three')

    def test_refine_takes_punctuation_not_words(self):
        from streaming import Transcript
        tr = Transcript()
        tr.words = self.hyp(('все', 0.2), ('состоит', 0.6), ('из', 1.0), ('атомов', 1.3))
        whole = self.hyp(('Всё', 0.2), ('состоит', 0.6), ('из', 1.0), ('атомов.', 1.3))
        self.assertTrue(tr.refine(whole, 0.0))
        self.assertEqual(' '.join(w.text for w in tr.words), 'Всё состоит из атомов.')
        other = self.hyp(('Всё', 0.2), ('стоит', 0.6), ('из', 1.0), ('атомов.', 1.3))
        tr.words[1].text = 'состоит'
        tr.refine(other, 0.0)
        self.assertEqual(tr.words[1].text, 'состоит')     # другое слово не подменяем

    def test_swallowed_draft_tail_is_held(self):
        from streaming import Transcript
        tr = Transcript()
        tr.update(self.hyp(('Nor', 0.1), ('is', 0.5), ('mister', 0.9)), 2.0, False)
        self.assertEqual(tr.shown_draft(), 'Nor is mister')
        tr.update(self.hyp(('Nor', 0.1)), 2.3, False)          # окно «проглотило» хвост
        self.assertEqual(' '.join([w.text for w in tr.words] + [tr.shown_draft()]), 'Nor is mister')
        tr.update(self.hyp(('Nor', 0.1), ('is', 0.5), ('mister', 0.9), ('Quilter', 1.4)), 2.6, False)
        self.assertIn('Quilter', [w.text for w in tr.draft])
        for _ in range(3):                                      # но не вечно
            tr.update(self.hyp(('Nor', 0.1)), 2.9, False)
        self.assertNotIn('Quilter', [w.text for w in tr.draft + tr.words])

    def test_agreement_commits_and_keeps_tail_as_draft(self):
        from streaming import Transcript
        tr = Transcript()
        tr.update(self.hyp(('Ну,', 0.2), ('что', 0.6), ('поделать', 1.0)), 1.5, False)
        self.assertEqual(tr.words, [])                  # одна гипотеза - ещё не повод
        tr.update(self.hyp(('Ну,', 0.2), ('что', 0.6), ('поделать,', 1.0), ('пришлось', 1.6)), 2.2, False)
        self.assertEqual([w.text for w in tr.words], ['Ну,', 'что', 'поделать,'])
        self.assertEqual([w.text for w in tr.draft], ['пришлось'])
        self.assertGreater(tr.commit_t, 1.3)

    def test_seam_drops_context_and_fixes_trailing_punct(self):
        from streaming import Transcript
        tr = Transcript()
        h = self.hyp(('окно', 0.2), ('открываешь', 0.8), ('не', 1.8))
        tr.update(h, 2.4, False)
        tr.update(h, 2.4, False)
        self.assertEqual([w.text for w in tr.words], ['окно', 'открываешь'])
        # новое окно начинается с контекста; у начала окна модель ставит
        # заглавную - само слово не трогаем, точку после него берём
        tr.update(self.hyp(('Открываешь.', 0.85), ('Не', 1.8), ('дует.', 2.1)), 2.6, True)
        self.assertEqual(' '.join(w.text for w in tr.words), 'окно открываешь. Не дует.')

    def test_word_time_drift_does_not_lose_or_repeat_words(self):
        from streaming import Transcript
        tr = Transcript()
        h = self.hyp(('Что', 10.9), ('делать?', 11.3), ('Когда', 11.6))
        tr.update(h, 12.6, False)
        tr.update(h, 12.6, False)
        self.assertEqual(tr.words[-1].text, 'Когда')
        # «когда» сдвинулось на 0.3 с, «делать» распозналось иначе
        tr.update(self.hyp(('делаешь?', 11.25), ('Когда', 11.9), ('что?', 12.3)), 13.0, True)
        self.assertEqual(' '.join(w.text for w in tr.words), 'Что делать? Когда что?')

    def test_incomplete_edge_word_is_not_shown(self):
        from streaming import Transcript
        tr = Transcript()
        tr.update(self.hyp(('макароны', 0.2), ('свар', 1.1)), 1.45, False)
        self.assertEqual(tr.shown_draft(), 'макароны')


class SchedulerTest(unittest.TestCase):
    def _job(self, key, final, sec=1.0, t=0.0, utt=None):
        import engine
        if utt is None:
            utt = engine._Utt(key, 'mon', 1, None, 0.0)
        return engine._Job(utt=utt, upto=int(sec / engine.BLOCK_S), final=final, submitted=t)

    def test_priorities_and_replacement(self):
        import engine
        s = engine._SttScheduler(lambda b, s: None)
        ua = engine._Utt('a', 'mon', 1, None, 0.0)
        s.submit(self._job('a', False, t=1, utt=ua))
        s.submit(self._job('a', False, t=2, utt=ua))     # заменяет черновик
        s.submit(self._job('b', True, sec=1.0, t=3))
        self.assertEqual(len(s._partials), 1)
        batch = s._take()
        self.assertTrue(batch[0].final)
        # финал b и черновик a близкой длины - в одном батче
        self.assertEqual({(j.key, j.final) for j in batch}, {('b', True), ('a', False)})

    def test_final_drops_pending_partial(self):
        import engine
        s = engine._SttScheduler(lambda b, s: None)
        ua = engine._Utt('a', 'mon', 3, None, 0.0)
        s.submit(self._job('a', False, utt=ua))
        s.submit(self._job('a', True, utt=ua))
        self.assertNotIn('a', s._partials)

    def test_window_not_whole_phrase(self):
        import engine
        u = engine._Utt('a', 'mon', 1, None, 0.0)
        u.tr.commit_t = 30.0                              # тридцать секунд уже подтверждены
        job = self._job('a', False, sec=31.0, utt=u)
        self.assertLess(job.est_samples(), 3 * engine.SAMPLE_RATE)

    def test_no_wasteful_padding(self):
        import engine
        s = engine._SttScheduler(lambda b, s: None)
        s.submit(self._job('a', True, sec=9.0))
        s.submit(self._job('b', False, sec=0.6, t=1))
        self.assertEqual(len(s._take()), 1)

    def test_stop_wakes_thread(self):
        import engine
        s = engine._SttScheduler(lambda b, s: None)
        s.start()
        s.stop()
        s._thread.join(1.0)
        self.assertFalse(s._thread.is_alive())


class EmitTest(unittest.TestCase):
    """Фраза из нескольких кусков (сменился голос посреди фразы) - в оверлей и историю."""

    def setUp(self):
        import engine
        import overlay
        from asr_onnx import Word
        self.engine, self.overlay = engine, overlay
        overlay.drain_queue()
        self.u = engine._Utt('mon', 'mon', 1, None, time.monotonic())
        self.u.tr.words = [Word('первый', 0, .3), Word('кусок.', .3, .6), Word('второй', .9, 1.2)]
        self.u.pieces = [engine._Piece(0, 0, speaker='A'), engine._Piece(2, .9, speaker='B')]
        self.history = []
        self.jobs = []

    def fake(self, with_worker):
        from types import SimpleNamespace
        jobs = self.jobs

        class Worker:
            def submit(self, job):
                jobs.append(job)

            def draft(self, *a):
                jobs.append(('draft',) + a)

        return SimpleNamespace(
            _tr=Worker() if with_worker else None,
            history=SimpleNamespace(add=lambda *a, **k: self.history.append(a[2]), session_id='s'),
            _queue_translation=self.engine.TranscriberEngine._queue_translation)

    def emit(self, eng, final):
        self.engine.TranscriberEngine._emit(eng, self.u, final)
        out = []
        while not self.overlay.event_q.empty():
            e = self.overlay.event_q.get_nowait()
            out.append((e['seg'], e['speaker'], e['done'], e['final']))
        return out

    def test_two_pieces_without_translator(self):
        eng = self.fake(False)
        self.assertEqual(self.emit(eng, False), [(1, 'A', 'первый кусок.', True),
                                                 ('1/1', 'B', 'второй', False)])
        self.assertEqual(self.history, ['первый кусок.'])
        self.assertEqual(self.emit(eng, False), [])          # ничего не поменялось - ничего не шлём
        self.assertEqual(self.emit(eng, True), [('1/1', 'B', 'второй', True)])
        self.assertEqual(self.history, ['первый кусок.', 'второй'])   # каждый кусок - один раз

    def test_history_disabled_does_not_log_recognized_text(self):
        from history import Recorder

        eng = self.fake(False)
        eng.history = Recorder()
        eng.history.enabled = False
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertTrue(self.emit(eng, True))
        self.assertEqual(captured.getvalue(), '')

    def test_two_pieces_with_translator(self):
        eng = self.fake(True)
        self.assertEqual(len(self.emit(eng, False)), 2)
        self.assertEqual([j[0] for j in self.jobs], ['sent', 'close'])
        self.assertEqual(self.jobs[0][5], 'первый кусок.')
        self.emit(eng, True)
        self.assertEqual([(j[0], j[3]) for j in self.jobs[2:]], [('sent', '1/1'), ('close', '1/1')])
        self.assertEqual(self.history, [])                  # историю пишет переводчик


class EngineModeTest(unittest.TestCase):
    """Смена языка и видеокарты на ходу: модели-заглушки, без файлов моделей."""

    def test_quick_switches_end_in_last_requested_state(self):
        import engine
        import asr_onnx
        import asr_parakeet
        built = []

        def fake(cls):
            class Fake(cls):
                def __init__(self, gpu):
                    self.gpu = gpu
                    self.gpu_error = ''
                    self.threads = 1

                def recognize(self, audios):
                    return []

                def set_threads(self, n):
                    self.threads = n
            return Fake

        FakeG, FakeP = fake(asr_onnx.GigaAM), fake(asr_parakeet.Parakeet)
        eng = engine.TranscriberEngine('me')
        eng.view = 'orig'

        def make():
            time.sleep(0.2)             # грузится не мгновенно
            stt = (FakeP if eng.multilang else FakeG)(eng.use_gpu)
            stt.want_gpu = eng.use_gpu
            built.append(stt)
            return stt

        eng._make_stt = make
        eng.stt = FakeG(False)
        eng.stt.want_gpu = False
        modes = []
        eng.mode_changed.connect(lambda on, err: modes.append(on))
        eng.set_gpu(True)
        eng.set_language(True)
        eng.set_gpu(False)
        eng.set_language(True)
        for _ in range(100):
            time.sleep(0.05)
            if not eng._cfg_lock.locked() and built and isinstance(eng.stt, FakeP):
                break
        time.sleep(0.3)
        self.assertIsInstance(eng.stt, FakeP)           # английская, как просили последним
        self.assertFalse(eng.stt.want_gpu)              # и без видеокарты
        self.assertLessEqual(len(built), 2)             # промежуточные состояния не собирались все
        self.assertIsNone(eng._tr)                      # вид «EN» - переводчик не нужен
        eng.stt = None
        eng._cfg_gen += 1
        eng._apply_config(eng._cfg_gen)                 # моделей нет - всё равно отвечает интерфейсу

    def test_gpu_loss_text_and_no_retry(self):
        """Сброс видеокарты: текст Windows (cp1251) читается, новые сессии не пробуются до перезапуска."""
        import asr_onnx
        raw = 'FAIL : '.encode() + 'Экземпляр устройства GPU приостановлен.'.encode('cp1251')
        e = UnicodeDecodeError('utf-8', raw, 7, 8, 'invalid continuation byte')
        self.assertIn('Экземпляр устройства GPU', asr_onnx.err_text(e))

        class M:
            gpu, _gpu, gpu_error = True, object(), ''
        m = M()
        try:
            asr_onnx.gpu_failed(m, e)
            self.assertFalse(m.gpu)
            self.assertTrue(asr_onnx.gpu_lost)
            g = asr_onnx.GigaAM.__new__(asr_onnx.GigaAM)
            g._dir = Path(tempfile.gettempdir())
            g.gpu_error = ''
            g._open_gpu()
            self.assertEqual(g.gpu_error, 'lost' if asr_onnx.available_gpu() else 'no-gpu')
        finally:
            asr_onnx.gpu_lost = ''

    def test_old_translate_flag_means_english(self):
        from config import AppConfig
        c = AppConfig.from_dict({'translate': True, 'overlay_font_size': 30})
        self.assertTrue(c.english)
        self.assertEqual(c.translate_view, 'both')
        self.assertFalse(AppConfig.from_dict({'translate': False}).english)


class TranslateWorkerTest(unittest.TestCase):
    """Поток перевода без модели: переводчик-заглушка."""

    def setUp(self):
        import engine
        import overlay
        self.overlay = overlay
        overlay.drain_queue()

        class Fake:
            """Перевод - «RU(текст)», токен на слово."""
            def __init__(self):
                self.target_vocab = []

            def translate(self, text):
                return 'RU(' + text + ')'

            def _id(self, w):
                if '▁' + w not in self.target_vocab:
                    self.target_vocab.append('▁' + w)
                return self.target_vocab.index('▁' + w)

            def decode(self, ids):
                return ' '.join(self.target_vocab[i][1:] for i in ids)

            def translate_scored(self, text, force=()):
                ids = [self._id(w) for w in self.translate(text).split()]
                ids = list(force) + ids[len(force):]
                return self.decode(ids), ids, 0.0

        class History:
            def __init__(self):
                self.rows = []

            def add(self, key, who, text, started, tr='', sid=None):
                self.rows.append((text, tr))

        class Eng:
            _run = 1
            view = 'both'
            history = History()

        self.eng = Eng()
        self.w = engine._TranslateWorker(Fake(), self.eng)

    def tearDown(self):
        self.w.close()
        self.w._thread.join(2)

    def flush(self):
        self.w._q.put(('sent', -1, 'x', 0, '?', '', 0.0, time.monotonic()))   # не наш запуск - только чтобы дождаться очереди
        self.w._q.put(('end', -1))
        while not self.w._q.empty():
            time.sleep(0.01)
        time.sleep(0.05)
        out = []
        while not self.overlay.event_q.empty():
            out.append(self.overlay.event_q.get_nowait())
        return out

    def test_stale_run_never_reaches_overlay(self):
        self.w.submit(('sent', 1, 'mon', 3, '?', 'Hello.', 0.0, time.monotonic()))
        evs = self.flush()
        self.assertEqual([(e['seg'], e['text']) for e in evs], [(3, 'RU(Hello.)')])
        # остановили: перевод, досчитанный после «стоп», на экран не идёт
        self.eng._run = 2
        self.w.submit(('end', 1))
        self.w.submit(('sent', 1, 'mon', 4, '?', 'Late one.', 0.0, time.monotonic()))
        self.assertEqual(self.flush(), [])
        # новый запуск, тот же номер фразы - старые части к нему не липнут
        self.eng._run = 3
        self.w.submit(('sent', 3, 'mon', 3, '?', 'Fresh.', 0.0, time.monotonic()))
        self.assertEqual([e['text'] for e in self.flush()], ['RU(Fresh.)'])

    def test_clause_parts_join_with_comma(self):
        import engine
        parts = [('Мистер Квилтер - апостол средних классов.', True, 'Mr. Quilter is the apostle of the middle classes,'),
                 ('Мы рады его видеть.', True, 'and we are glad to see him.'),
                 ('Фред пришёл.', True, 'Fred came.')]
        self.assertEqual(engine._join_parts(parts),
                         'Мистер Квилтер - апостол средних классов, мы рады его видеть. Фред пришёл.')
        # исходник с заглавной (имя) - не трогаем
        self.assertEqual(engine._join_parts([('Раз.', True, 'One,'), ('Фред тут.', True, 'Fred is here.')]),
                         'Раз. Фред тут.')

    def test_close_writes_history_with_translation(self):
        self.w.submit(('sent', 1, 'mon', 1, '?', 'Hello.', 0.0, time.monotonic()))
        self.w.submit(('sent', 1, 'mon', 1, '?', 'Привет.', 0.0, time.monotonic()))
        self.w.submit(('close', 1, 'mon', 1, '?', 'Hello. Привет.', 0.0, None))
        self.flush()
        self.assertEqual(self.eng.history.rows, [('Hello. Привет.', 'RU(Hello.)')])
        self.assertEqual(self.w._segs, {})

    def test_russian_shown_only_without_original(self):
        self.w.submit(('sent', 1, 'mon', 1, '?', 'Привет всем.', 0.0, time.monotonic()))
        self.assertEqual(self.flush(), [])               # оригинал на экране - не дублируем
        self.eng.view = 'trans'
        self.w.submit(('sent', 1, 'mon', 2, '?', 'Привет всем.', 0.0, time.monotonic()))
        self.assertEqual([e['text'] for e in self.flush()], ['Привет всем.'])

    def test_only_newest_draft_is_translated(self):
        calls = []
        orig = self.w.translator.translate
        self.w.translator.translate = lambda t: (calls.append(t), time.sleep(0.05), orig(t))[2]
        self.w.submit(('sent', 1, 'mon', 9, '?', 'Block.', 0.0, time.monotonic()))     # занимает поток
        for n in range(1, 6):
            self.w.draft(1, 'mon', 1, '?', ' '.join(['word'] * (n + 2)))
        self.flush()
        self.assertEqual(calls, ['Block.', ' '.join(['word'] * 7)])

    def test_draft_shows_only_agreed_words(self):
        """На экран идёт только то, в чём сошлись два черновика подряд, без последнего слова; дальше не меняется."""
        self.w.translator.translate = lambda t: 'RU ' + t
        drafts = []
        for text in ['the cat', 'the cat sat', 'the cat sat on', 'the dog sat on the']:
            self.w.draft(1, 'mon', 1, '?', text)
            drafts += [e['draft'] for e in self.flush()]
        # первый черновик сравнить не с чем; «RU the cat» / «RU the cat sat» - сошлись
        # три слова, без последнего - «RU the»
        self.assertEqual(drafts, ['RU the', 'RU the cat', 'RU the cat sat'])
        # оригинал поправился (cat -> dog), но показанное продолжается, а не переписывается
        self.w.submit(('sent', 1, 'mon', 1, '?', 'the dog sat on the mat.', 0.0, time.monotonic()))
        evs = self.flush()
        self.assertEqual((evs[-1]['text'], evs[-1]['draft']), ('RU the cat sat on the mat.', ''))

    def test_agreed_prefix(self):
        import engine
        v = ['▁a', '▁b', 'x', '▁c', '▁d', 'y']
        self.assertEqual(engine._agreed([0, 1, 2, 3], [0, 1, 2, 3, 4], v), [0, 1, 2])      # a bx c | без c
        self.assertEqual(engine._agreed([0, 1, 5], [0, 1, 2, 3], v), [])                   # b недописано одинаково
        self.assertEqual(engine._agreed([0, 1, 2, 3, 4], [0, 1, 2, 3, 4], v, hold=0), [0, 1, 2, 3, 4])


class OverlayModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        import overlay
        cls.overlay = overlay

    def setUp(self):
        self.m = self.overlay.SubtitleModel(self.overlay.OverlayStyle())
        self.t = 0

    def ev(self, dt=0, **kw):
        self.t += dt
        kw.setdefault('type', 'text')
        kw.setdefault('done', '')
        kw.setdefault('draft', '')
        kw.setdefault('final', False)
        self.m._apply(kw, self.t)

    def run_for(self, ms):
        shown = []
        for _ in range(ms // 16):
            self.t += 16
            self.m.pace(self.t)
            shown.append(sum(1 for c in self.m.cards for w in c.words() if w.shown))
        return shown

    def trans(self, dt=0, **kw):
        kw.setdefault('stream', 'mon')
        kw.setdefault('speaker', '?')
        kw.setdefault('text', '')
        kw.setdefault('draft', '')
        self.ev(dt, type='trans', **kw)

    def test_translation_has_its_own_feed(self):
        self.ev(stream='mon', seg=1, speaker='?', done='Hello there.', final=True)
        self.ev(dt=10, stream='mon', seg=2, speaker='?', done='Привет.', final=True)
        self.run_for(1500)
        before = [w for c in self.m.cards for w in c.words()]
        self.trans(seg=1, text='Привет всем.')
        # оригинал не тронут: ни новых слов, ни сдвига
        self.assertEqual([w for c in self.m.cards for w in c.words()], before)
        self.assertEqual(len(self.m.tcards), 1)
        self.assertEqual(self.m.tcards[0].text(), 'Привет всем.')
        self.assertTrue(all(w.tr for w in self.m.tcards[0].words()))
        self.trans(seg=2, text='Ещё одна.')
        r = self.overlay.CardRenderer()
        self.run_for(1000)
        lay = r.layout(self.m.tcards[0], 2000, 20, 'left', split=True)
        rows = {}
        for w, x, line in lay.words:
            rows.setdefault(line, []).append(w.text)
        self.assertEqual(rows, {0: ['Привет', 'всем.'], 1: ['Ещё', 'одна.']})   # каждая фраза с новой строки

    def test_draft_translation_is_replaced_in_place(self):
        self.trans(seg=1, draft='Рекурсивная функция')
        self.run_for(1000)
        seg = self.m._tindex[('mon', 1)][1]
        first = list(seg.words)
        self.assertFalse(any(w.done for w in first))     # черновик - тусклый
        self.assertFalse(seg.final)
        self.trans(seg=1, text='Рекурсивная функция вызывает себя.')
        self.assertIs(seg.words[0], first[0])            # совпавшие слова остались на месте
        self.assertTrue(all(w.done for w in seg.words))
        self.assertTrue(seg.final)
        self.trans(seg=1)                                # перевода не стало - кусок уходит
        self.assertNotIn(('mon', 1), self.m._tindex)

    def test_translation_feeds_fade_together_and_clear(self):
        self.m.style_.ttl_ms = 2000
        self.ev(stream='mon', seg=1, speaker='?', done='Hello.', final=True)
        self.trans(seg=1, text='Привет.')
        self.run_for(1000)
        self.t += 2000 + 800
        self.m.housekeeping(self.t)
        self.assertEqual((self.m.cards, self.m.tcards), ([], []))
        self.ev(stream='mon', seg=1, speaker='?', done='Hello.', final=True)
        self.trans(seg=1, text='Привет.')
        self.ev(type='clear')
        self.assertEqual((self.m.cards, self.m.tcards, self.m._tindex), ([], [], {}))

    def paint_both(self, frames=60, w=700, h=320, view='both'):
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QImage, QPainter
        self.m.style_.view = view
        self.m.style_.split_segments = True
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        r = self.overlay.CardRenderer()
        for _ in range(frames):
            self.t += 16
            self.m.pace(self.t)
            self.overlay.paint_overlay(p, QRectF(0, 0, w, h), r, self.m, self.m.style_, self.t, 0.016)
        p.end()

    def test_both_view_original_above_translation(self):
        self.m.style_.max_lines = 2
        self.ev(stream='mon', seg=1, speaker='?', done='Okay everyone, today we talk about recursion.', final=True)
        self.paint_both()
        self.assertIsNone(self.m.tfeed.rect)              # перевода ещё нет - оригинал один
        self.trans(seg=1, text='Итак, сегодня поговорим о рекурсии.')
        self.paint_both(frames=90)
        top, bottom = self.m.feed.rect, self.m.tfeed.rect
        self.assertLessEqual(top.bottom(), bottom.top() - self.overlay.ZONE_GAP + 0.5)
        held = self.m.tfeed.h
        for i in range(2, 9):
            self.ev(stream='mon', seg=i, speaker='?', done=f'Sentence number {i} goes here.', final=True)
            self.trans(seg=i, text=f'Предложение номер {i} здесь.')
            self.paint_both(frames=40)
            self.assertGreaterEqual(self.m.tfeed.h, held - 0.5)   # лента перевода не сжимается
            held = max(held, self.m.tfeed.h)
        m = self.overlay.CardRenderer().metrics(self.m.style_.font_size)
        self.assertLessEqual(self.m.tfeed.h, 2 * m['line_h'] * 1.25 + m['name_h'] + 1)

    def test_translation_only_view(self):
        self.ev(stream='mon', seg=1, speaker='?', done='Hello there.', final=True)
        self.trans(seg=1, text='Привет.')
        self.paint_both(view='trans')
        self.assertIsNone(self.m.feed.rect)               # оригинала на экране нет
        self.assertIsNotNone(self.m.tfeed.rect)

    def test_hover_fades_feed(self):
        win = self.overlay.OverlayWindow()
        try:
            win.set_hover_opacity(0.2)
            win.preview_hover(5000)             # как будто двигают ползунок
            now = self.overlay._now()
            self.assertAlmostEqual(win._hover_wanted(now), 0.2)
            win._hover_force_until = 0
            win.feed.rect = win.tfeed.rect = None   # ленты нет - гасить нечего
            self.assertEqual(win._hover_wanted(now), 1.0)
            win.set_hover_opacity(1.0)          # «Выкл»
            win.preview_hover(5000)
            self.assertEqual(win._hover_wanted(self.overlay._now()), 1.0)
        finally:
            win.close()

    def test_words_come_out_at_reading_pace(self):
        self.ev(stream='mon', seg=1, speaker='?', done='раз два три четыре пять шесть семь восемь')
        shown = self.run_for(300)
        self.assertLessEqual(shown[-1], 4)               # не всё предложение разом
        self.run_for(2000)
        self.assertTrue(all(w.shown for w in self.m.cards[0].words()))

    def test_partial_then_final_same_card(self):
        self.ev(stream='mon', seg=1, speaker='?', draft='привет')
        self.ev(stream='mon', seg=1, speaker='Маша', done='Привет, как дела?', final=True)
        self.assertEqual(len(self.m.cards), 1)
        card = self.m.cards[0]
        self.assertEqual(card.speaker, 'Маша')          # имя уточнилось
        self.assertFalse(card.live())

    def test_one_changed_word_does_not_reshow_tail(self):
        self.ev(stream='mon', seg=1, speaker='?', draft='Нет обычно так тоже делаю')
        self.run_for(3000)
        old = self.m.cards[0].words()
        self.ev(stream='mon', seg=1, speaker='?', draft='Ну я обычно так тоже делаю')
        new = self.m.cards[0].words()
        kept = sum(1 for w in new if any(w is o for o in old))
        self.assertEqual(kept, 4)                        # «обычно так тоже делаю» - те же слова

    def test_continuous_speech_stays_in_one_card(self):
        self.ev(stream='dc:1', seg=1, speaker='Дима', done='раз', final=True)
        self.ev(dt=1500, stream='dc:1', seg=2, speaker='Дима', done='два', final=True)
        self.ev(dt=500, stream='dc:2', seg=1, speaker='Маша', done='три', final=True)
        self.ev(dt=6000, stream='dc:2', seg=2, speaker='Маша', done='четыре', final=True)
        self.ev(dt=13000, stream='dc:2', seg=3, speaker='Маша', done='пять', final=True)
        self.assertEqual([c.text() for c in self.m.cards], ['раз два', 'три четыре', 'пять'])

    def test_unsure_speaker_continues_card(self):
        self.ev(stream='mon', seg=1, speaker='Маша', done='раз', final=True)
        self.ev(dt=800, stream='mon', seg=2, speaker='?', draft='два')
        self.assertEqual(len(self.m.cards), 1)

    def test_drop_removes_stuck_partial(self):
        self.ev(stream='mon', seg=2, speaker='?', draft='эм')
        self.ev(type='drop', stream='mon', seg=2)
        self.assertTrue(self.m.cards[0].gone_t > 0)

    def paint(self, frames=60, w=600, h=160):
        from PyQt6.QtCore import QRectF
        from PyQt6.QtGui import QImage, QPainter
        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        p = QPainter(img)
        r = self.overlay.CardRenderer()
        for _ in range(frames):
            self.t += 16
            self.m.pace(self.t)
            self.overlay.paint_feed(p, QRectF(0, 0, w, h), r, self.m.cards,
                                    list(self.m.ghosts.values()), self.m.style_, self.t, 0.016,
                                    model=self.m)
        p.end()
        return r

    def test_feed_scrolls_as_one_block(self):
        for i, who in enumerate(('a', 'b', 'c', 'd', 'e')):
            self.ev(dt=300, stream=who, seg=i, speaker=who, done='строка номер %d' % i, final=True)
        r = self.paint(frames=120)
        m = r.metrics(self.m.style_.font_size)
        feed = self.m.feed
        self.assertGreater(feed.scroll, 0.0)            # старое уехало вверх
        self.assertLessEqual(feed.h, 160)
        self.assertLessEqual(feed.h, self.m.style_.max_lines * m['line_h'] * 1.25 + m['name_h'] + 1)
        self.assertEqual(self.m.cards[-1].text(), 'строка номер 4')

    def test_speaking_indicator_does_not_blink(self):
        self.ev(type='speaking', stream='dc:1', speaker='Маша', active=True)
        self.ev(dt=200, type='speaking', stream='dc:1', active=False)
        self.ev(dt=150, type='speaking', stream='dc:1', active=True)
        g = self.m.ghosts['dc:1']
        self.assertTrue(g.active)
        self.assertEqual(len(self.m.ghosts), 1)          # тот же индикатор, без повторного появления
        self.assertIn('dc:1', self.m.speaking(self.t))
        self.ev(dt=100, type='speaking', stream='dc:1', active=False)
        self.assertIn('dc:1', self.m.speaking(self.t + 500))      # держится после паузы
        self.assertNotIn('dc:1', self.m.speaking(self.t + 3000))

    def test_feed_fades_as_a_whole(self):
        self.m.style_.ttl_ms = 2000
        self.ev(stream='a', seg=1, speaker='a', done='раз', final=True)
        self.ev(dt=500, stream='b', seg=1, speaker='b', done='два', final=True)
        self.run_for(1000)
        self.assertEqual(len(self.m.cards), 2)
        self.t += 2000 + 800
        self.m.housekeeping(self.t)
        self.assertEqual(self.m.cards, [])              # ушли вместе, а не по одной

    def test_center_align_centers_lines(self):
        self.ev(stream='mon', seg=1, speaker='?', done='короткая строка')
        for w in self.m.cards[0].words():
            w.shown = 1
        r = self.overlay.CardRenderer()
        left = r.layout(self.m.cards[0], 600, 22, 'left')
        center = r.layout(self.m.cards[0], 600, 22, 'center')
        self.assertEqual(left.words[0][1], 0.0)
        # карточка по ширине текста: единственная строка и так по центру
        self.assertAlmostEqual(center.words[0][1], 0.0, places=3)
        self.ev(stream='mon', seg=1, speaker='?',
                done='это первая строка которая получилась заметно длиннее второй а вот и вторая')
        for w in self.m.cards[0].words():
            w.shown = 1
        center = r.layout(self.m.cards[0], 400, 22, 'center')
        last_line = [x for w, x, ln in center.words if ln == center.lines - 1]
        self.assertGreater(last_line[0], 1.0)           # короткая последняя строка сдвинута к центру

    def test_long_monologue_scrolls_inside_feed(self):
        text = ' '.join(f'слово{i}' for i in range(200))
        self.ev(stream='mon', seg=1, speaker='?', done=text)
        for w in self.m.cards[0].words():
            w.shown = 1
        self.paint(frames=5, w=500, h=200)
        card = self.m.cards[0]
        self.assertGreater(card.trim, 0)                 # уехавшее наверх выброшено
        self.assertTrue(card.words()[-1].text.endswith('199'))


class HistoryTest(unittest.TestCase):
    def test_record_list_rename_delete(self):
        import history
        with tempfile.TemporaryDirectory() as d:
            orig = history.history_dir
            history.history_dir = lambda: Path(d)
            try:
                rec = history.Recorder()
                rec.begin()
                self.assertIsNone(rec.live_id)            # пустой сеанс файла не создаёт
                rec.add('dc:1', 'Дима', 'привет', 1000.0)
                rec.add('dc:2', 'Маша', 'привет, как дела?', 1003.0)
                self.assertIsNotNone(rec.live_id)
                rec.end()
                with open(next(Path(d).glob('*.jsonl')), 'a', encoding='utf-8') as f:
                    f.write('{"t": 10')                # оборванная строка после сбоя
                ss = history.list_sessions()
                self.assertEqual(len(ss), 1)
                self.assertEqual(ss[0].speakers, ['Дима', 'Маша'])
                self.assertEqual(ss[0].lines, 2)
                history.rename(ss[0].id, 'Рейд')
                self.assertEqual(history.list_sessions()[0].title, 'Рейд')
                self.assertIn('Маша', history.as_text(ss[0].id))
                history.delete(ss[0].id)
                self.assertEqual(history.list_sessions(), [])
            finally:
                history.history_dir = orig

    def test_read_from_waits_for_whole_line(self):
        import history
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 's.jsonl'
            p.write_bytes(b'{"t": 1, "who": "a", "text": "\xd1\x80\xd0\xb0\xd0\xb7"}\n{"t": 2, "who"')
            lines, off = history.read_from(p, 0)
            self.assertEqual([ln['text'] for ln in lines], ['раз'])
            with open(p, 'ab') as f:
                f.write(b': "b", "text": "two"}\n')
            more, off2 = history.read_from(p, off)
            self.assertEqual([ln['text'] for ln in more], ['two'])
            self.assertEqual(history.read_from(p, off2), ([], off2))

    def test_grouper_splits_on_speaker_pause_and_length(self):
        import history
        g = history.Grouper()
        feed = [('a', 0), ('a', 5), ('b', 7), ('b', 40), ('b', 60), ('b', 80), ('b', 100), ('b', 120),
                ('b', 135)]
        # смена говорящего, пауза 33 с, затем монолог дольше 90 с без пауз
        self.assertEqual([g.feed(w, t) for w, t in feed],
                         [True, False, True, True, False, False, False, False, True])

    def test_transcript_append_keeps_scroll(self):
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from ui.history_page import Transcript
        tr = Transcript()
        tr.resize(360, 300)
        tr.show()
        tr.append_lines([{'t': i * 30.0, 'who': 'ab'[i % 2], 'text': 'фраза номер %d ' % i * 4}
                         for i in range(60)])
        tr.park_cursor()
        app.processEvents()
        bar = tr.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        bar.setValue(bar.maximum() // 2)
        v = bar.value()
        tr.append_lines([{'t': 9999.0, 'who': 'c', 'text': 'новая'}])
        app.processEvents()
        self.assertEqual(bar.value(), v)
        self.assertFalse(tr.textCursor().hasSelection())
        self.assertTrue(tr.toPlainText().endswith('новая'))
        tr.close()

    def test_file_session_times_are_offsets(self):
        import history
        with tempfile.TemporaryDirectory() as d:
            orig = history.history_dir
            history.history_dir = lambda: Path(d)
            try:
                sid, base = history.new_file_session('лекция.mp4')

                class Job:
                    pass
                job = Job()
                job.base = base
                job.cancel = lambda: None
                history.jobs[sid] = job
                s = history.list_sessions()[0]         # виден ещё до первой фразы
                self.assertTrue(s.file)
                self.assertEqual(s.title, 'лекция.mp4')
                history.append(sid, [{'t': base + 5, 'who': '?', 'text': 'раз'},
                                     {'t': base + 3725, 'who': '?', 'text': 'два'}])
                history.jobs.pop(sid)
                history.rename(sid, 'Пара')
                s = history.list_sessions()[0]
                self.assertTrue(s.file)                 # переименование не теряет «файл»
                self.assertEqual(s.title, 'Пара')
                text = history.as_text(sid)
                self.assertIn('[0:05]', text)
                self.assertIn('[1:02:05]', text)
                self.assertEqual(history.stamp(base + 61, base), '1:01')
            finally:
                history.history_dir = orig

    def test_translation_in_text_export(self):
        import history
        with tempfile.TemporaryDirectory() as d:
            orig = history.history_dir
            history.history_dir = lambda: Path(d)
            try:
                rec = history.Recorder()
                rec.begin()
                rec.add('mon', '?', 'Hello there.', 1000.0, tr='Привет.')
                sid = rec.session_id
                rec.end()
                rec.add('mon', '?', 'Late one.', 1001.0, tr='Поздняя.', sid=sid)    # перевод после «стоп»
                text = history.as_text(sid)
                self.assertIn('→ Привет.', text)
                self.assertIn('→ Поздняя.', text)
            finally:
                history.history_dir = orig

    def test_disabled_writes_nothing(self):
        import history
        with tempfile.TemporaryDirectory() as d:
            orig = history.history_dir
            history.history_dir = lambda: Path(d)
            try:
                rec = history.Recorder()
                rec.enabled = False
                rec.begin()
                rec.add('mon', '?', 'текст')
                rec.end()
                self.assertEqual(list(Path(d).iterdir()), [])
            finally:
                history.history_dir = orig


class SecretStoreTest(unittest.TestCase):
    def test_token_parsing(self):
        import base64
        import secret_store
        bot_id = '123456789012345678'
        head = base64.urlsafe_b64encode(bot_id.encode()).decode().rstrip('=')
        token = f'{head}.GxXxXx.' + 'a' * 38
        self.assertEqual(secret_store.bot_id_from_token(token), bot_id)
        self.assertTrue(secret_store.looks_like_token(token))
        self.assertFalse(secret_store.looks_like_token('hello'))
        self.assertIn(bot_id, secret_store.invite_url(bot_id))


class VoicesTest(unittest.TestCase):
    def test_add_merge_delete(self):
        import voices
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'voices.npz'
            orig = voices.voices_file
            voices.voices_file = lambda: path
            try:
                a = np.zeros(4, np.float32); a[0] = 1
                b = np.zeros(4, np.float32); b[1] = 1
                self.assertEqual(voices.add_sample('x', a), 1)
                self.assertEqual(voices.add_sample('x', b), 2)
                c = voices.load_centroids()['x']
                self.assertAlmostEqual(float(np.linalg.norm(c)), 1.0, places=5)
                self.assertEqual(voices.load_counts(), {'x': 2})
                voices.delete('x')
                self.assertFalse(path.exists())
            finally:
                voices.voices_file = orig


class VersionTest(unittest.TestCase):
    def test_installer_matches_version_py(self):
        import re
        import version
        iss = (Path(__file__).resolve().parent.parent / 'installer' / 'voice-assist.iss').read_text(encoding='utf-8-sig')
        self.assertEqual(re.search(r'#define AppVersion "([^"]*)"', iss).group(1), version.VERSION)
        self.assertEqual(re.search(r'#define Stage "([^"]*)"', iss).group(1), version.STAGE)
        self.assertEqual(version.TAG, f'{version.VERSION}-{version.STAGE}' if version.STAGE else version.VERSION)


if __name__ == '__main__':
    unittest.main()
