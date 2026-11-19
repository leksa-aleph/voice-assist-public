"""
Ловец скачков субтитров.

Гоняет движок на аудиофайле в реальном темпе (как будто звук идёт из
Discord), пропускает события через настоящую логику оверлея с её темпом
показа (кадр за кадром, 60 к/с) и меряет то, что видит читающий:

  burst_300ms   сколько слов появилось на экране за любые 300 мс.
                «Целое предложение в лицо» - это 6+;
  first_show    сколько слов в карточке в момент её появления;
  rewrite       сколько уже показанных слов потом исчезло или сменилось
                (запятая или заглавная не считаются);
  mid_break     новая карточка того же источника, хотя прошлая не
                закончила предложение;
  latency       от начала речи до первого слова на экране;
  word_lag      от конца слова в звуке до его появления на экране;
  pacing_lag    сколько слово ждёт в очереди показа.

И качество: итоговый текст сравнивается с распознаванием тех же фраз
целиком (как делала старая версия) - расхождение слов и знаков препинания.
Плюс нагрузка: процессор за прогон и скорость распознавания.

Запуск:
  python tools/subtitle_jumps.py samples/разговор.wav --as-bot
  python tools/subtitle_jumps.py build/test-audio/монолог.webm --start 300 --seconds 240

Рядом с отчётом пишется таймлайн: что видел бы человек.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('QT_QPA_FONTDIR', 'C:/Windows/Fonts')
os.environ['VOICE_ASSIST_NO_DISCORD'] = '1'
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np

SR = 16000
END_PUNCT = ('.', '?', '!', '…')
TICK_MS = 16
BURST_WINDOW_MS = 300


def load_audio(path: str, start: float, seconds: float) -> np.ndarray:
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', path]
    if seconds > 0:
        cmd += ['-t', str(seconds)]
    cmd += ['-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def norm(w: str) -> str:
    return w.strip('.,!?;:—–-«»"\'()…').lower().replace('ё', 'е')


def punct_class(w: str) -> str:
    w = w.rstrip('»"\')')
    if w.endswith(END_PUNCT):
        return 'end'
    if w.endswith((',', ';', ':')):
        return 'comma'
    return ''


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[-1]


def compare_text(ref: list[str], hyp: list[str]) -> dict:
    """Расхождение слов и F1 точек/запятых по выровненным словам."""
    rn, hn = [norm(w) for w in ref], [norm(w) for w in hyp]
    wer = edit_distance(rn, hn) / max(1, len(rn))
    stats = {'end': [0, 0, 0], 'comma': [0, 0, 0]}   # tp, fp, fn
    sm = SequenceMatcher(None, rn, hn, autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            r, h = punct_class(ref[a + i]), punct_class(hyp[b + i])
            for cls in stats:
                if r == cls and h == cls:
                    stats[cls][0] += 1
                elif h == cls:
                    stats[cls][1] += 1
                elif r == cls:
                    stats[cls][2] += 1

    def f1(tp, fp, fn):
        return round(2 * tp / max(1, 2 * tp + fp + fn), 3)

    return {'word_diff_pct': round(100 * wer, 1),
            'punct_end_f1': f1(*stats['end']), 'punct_comma_f1': f1(*stats['comma'])}


def reference_text(eng, audio: np.ndarray) -> list[str]:
    """Старый способ: фраза по VAD целиком в модель."""
    import engine as E
    probs = eng.vad.probs(audio)
    segs, start, speaking, sil = [], 0, False, 0
    for i, p in enumerate(probs):
        if not speaking:
            if p >= E.VAD_ON:
                speaking, start, sil = True, max(0, i - 15), 0
            continue
        sil = sil + 1 if p < E.VAD_OFF else 0
        if sil >= 26 or i - start >= int(20 / E.BLOCK_S):
            segs.append((start, i))
            speaking = False
    if speaking:
        segs.append((start, len(probs)))
    words = []
    for a, b in segs:
        hyp = eng.stt.recognize([audio[a * 512:b * 512]])[0]
        words += hyp.text.split()
    return words


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=0.0)
    ap.add_argument('--preset', default='balanced')
    ap.add_argument('--out', default='')
    ap.add_argument('--no-reference', action='store_true')
    ap.add_argument('--as-bot', action='store_true',
                    help='подавать как дорожку Discord-бота: в паузах пакетов нет')
    ap.add_argument('--gpu', action='store_true', help='распознавать на видеокарте')
    ap.add_argument('--translate', action='store_true', help='английская речь (Parakeet) с переводом')
    ap.add_argument('--slow-reveal', action='store_true', help='для «fast» - прежний темп показа слов')
    ap.add_argument('--with-file', default='', help='параллельно расшифровывать этот файл (как перетащили в окно)')
    ap.add_argument('--view', default='both', help='orig | both | trans - что показывать в английском режиме')
    ap.add_argument('--slow', type=float, default=1.0,
                    help='изобразить процессор в N раз медленнее: распознавание ждёт (N-1)×своё время')
    args = ap.parse_args()

    audio = load_audio(args.audio, args.start, args.seconds)
    dur = len(audio) / SR
    print(f'звук: {dur:.1f} с')

    import psutil
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName('voice-transcriber')
    from ui import theme
    theme.init_fonts()
    import engine
    import history
    import overlay
    import tempfile

    # история прогона - во временную папку, не в историю пользователя
    tmp_hist = Path(tempfile.mkdtemp(prefix='va-jumps-'))
    history.history_dir = lambda: tmp_hist

    eng = engine.TranscriberEngine('me')
    eng.use_gpu = args.gpu
    eng.multilang = args.translate
    eng.view = args.view
    t0 = time.time()
    eng.load_models()
    print(f'модели: {time.time() - t0:.1f} с, {eng.device_name}')
    eng.apply_recognition_preset(args.preset)
    if args.slow > 1.0:
        real = eng.stt.recognize

        def slow_recognize(audios, _real=real, k=args.slow):
            t = time.perf_counter()
            out = _real(audios)
            time.sleep((time.perf_counter() - t) * (k - 1))   # кодировщик отпускает GIL - как и сон
            return out
        eng.stt.recognize = slow_recognize
    eng.start(None, None, allow_no_sources=True)
    file_job = None
    if args.with_file:
        import file_transcribe
        file_job = file_transcribe.FileJob(args.with_file, eng)
        file_job.start()
    qmax = [0]

    def qwatch():
        while not stop.is_set():
            q = eng._audio_q
            if q is not None:
                qmax[0] = max(qmax[0], q.qsize())
            time.sleep(0.05)

    events: list[tuple[float, dict]] = []
    stop = threading.Event()
    clock0 = [0.0]

    def collector():
        while not stop.is_set() or not overlay.event_q.empty():
            try:
                ev = overlay.event_q.get(timeout=0.02)
            except queue.Empty:
                continue
            events.append((time.monotonic() - clock0[0], ev))

    proc = psutil.Process()
    th = threading.Thread(target=collector, daemon=True)
    clock0[0] = time.monotonic()
    th.start()
    threading.Thread(target=qwatch, daemon=True).start()
    c0 = proc.cpu_times()
    block = 512
    quiet = 0
    for i in range(0, len(audio) - block, block):
        chunk = audio[i:i + block].copy()
        if args.as_bot:
            # Discord не шлёт пакеты в тишине (после ~100 мс «хвоста»)
            quiet = quiet + 1 if float(np.sqrt(np.mean(chunk * chunk))) < 0.004 else 0
            if quiet <= 3:
                eng._enqueue('dc:1', 'bot', 'Маша', chunk)
        else:
            eng._enqueue('mon', 'mon', None, chunk)
        d = clock0[0] + (i + block) / SR - time.monotonic()
        if d > 0:
            time.sleep(d)
    wall = time.monotonic() - clock0[0]
    c1 = proc.cpu_times()
    if os.environ.get('VA_THREADS'):
        names = {t.native_id: t.name for t in threading.enumerate()}
        for t in sorted(proc.threads(), key=lambda t: -(t.user_time + t.system_time))[:8]:
            print(f'  поток {names.get(t.id, t.id)}: {100 * (t.user_time + t.system_time) / wall:.1f}%')
    time.sleep(4)
    stop.set()
    th.join()
    eng.stop()
    cpu = (c1.user - c0.user) + (c1.system - c0.system)
    st = eng.stats()
    stages = st.get('stages') or {}
    hist = list(eng._stt_hist_all)

    # ---- проигрываем события через логику оверлея кадр за кадром ----
    model = overlay.SubtitleModel(overlay.OverlayStyle(ttl_ms=0))
    if args.preset == 'fast' and not args.slow_reveal:
        model.style_.reveal = overlay.REVEAL_FAST
    timeline = []
    speak_start: dict[str, float] = {}
    first_word_lat = []
    first_show = []
    new_cards = mid_breaks = 0
    rewritten = 0
    shown_prev: dict = {}       # id(W) -> (норма, W, источник)
    reveal_times: list[int] = []
    queued_at: dict[int, int] = {}
    lags = []
    word_lags = []
    word_lag_at = []
    lag_words = []
    tr_seen: set = set()
    tr_prev: dict = {}
    tr_chunks = []
    tr_dropped = [0]
    tr_lags = []
    final_text: dict[tuple, str] = {}
    translations: dict[tuple, str] = {}
    order: list[tuple] = []
    ei = 0
    end_ms = int((events[-1][0] if events else 0) * 1000) + 3000
    now = 0
    while now <= end_ms:
        while ei < len(events) and events[ei][0] * 1000 <= now:
            t, ev = events[ei]
            ei += 1
            typ = ev.get('type')
            if typ == 'speaking' and ev.get('active'):
                speak_start[ev['stream']] = t
            if typ == 'text':
                key = (ev['stream'], ev['seg'])
                if key not in final_text:
                    order.append(key)
                final_text[key] = ' '.join(x for x in (ev['done'], ev['draft']) if x)
            before = list(model.cards)
            last = before[-1] if before else None
            model._apply(ev, now)
            if typ == 'trans':
                # сколько слов перевода прибавилось одним событием («пачка»)
                cur = (ev['text'] + ' ' + ev.get('draft', '')).split()
                prev = tr_prev.get((ev['stream'], ev['seg']), [])
                same = 0
                while same < min(len(cur), len(prev)) and cur[same] == prev[same]:
                    same += 1
                if len(cur) > same:
                    tr_chunks.append(len(cur) - same)
                tr_dropped[0] += len(prev) - same
                tr_prev[(ev['stream'], ev['seg'])] = cur
                timeline.append(f'[{int(t // 60):02d}:{t % 60:05.2f}] T {ev["stream"]}#{ev["seg"]} '
                                f'{ev["text"]} ‹{ev.get("draft", "")}›')
                translations[(ev['stream'], ev['seg'])] = ev['text']
            if typ == 'text':
                timeline.append(f'[{int(t // 60):02d}:{t % 60:05.2f}] {"F" if ev["final"] else "P"} '
                                f'{ev["stream"]}#{ev["seg"]} {ev["done"]} ‹{ev["draft"]}›')
            if len(model.cards) > len(before):
                new_cards += 1
                if (last is not None and last.stream == model.cards[-1].stream
                        and not last.text().rstrip().endswith(END_PUNCT)):
                    mid_breaks += 1
                    timeline.append('        ^ новая карточка посреди фразы')
        model.pace(now)
        # что видно на экране
        visible = {}
        for card in model.cards:
            ws = card.words()
            shown_n = sum(1 for w in ws if w.shown)
            for w in ws:
                if not w.shown:
                    queued_at.setdefault(id(w), now)
                    continue
                visible[id(w)] = (norm(w.text), w, card.stream)
            if shown_n and not getattr(card, '_seen', False):
                card._seen = True
                first_show.append(shown_n)
                if shown_n > 2:
                    timeline.append(f'        ^ карточка появилась сразу с {shown_n} словами')
        for card in model.tcards:
            for w in card.words():
                if w.shown and id(w) not in tr_seen:
                    tr_seen.add(id(w))
                    if w.end:
                        tr_lags.append(now / 1000 - (w.end - clock0[0]))
        for wid, (nt, w, stream) in visible.items():
            if wid not in shown_prev:
                reveal_times.append(now)
                if wid in queued_at:
                    lags.append(now - queued_at.pop(wid))
                s0 = speak_start.pop(stream, None)
                if s0 is not None:
                    first_word_lat.append(now / 1000 - s0)
                if w.end is not None:
                    word_lags.append(now / 1000 - (w.end - clock0[0]))
                    word_lag_at.append(now / 1000)
                    lag_words.append(w.text)
        for wid in shown_prev:
            if wid not in visible:
                rewritten += 1
        shown_prev = visible
        now += TICK_MS

    burst = 0
    burst_at = 0
    j = 0
    for i, t in enumerate(reveal_times):
        while reveal_times[j] < t - BURST_WINDOW_MS:
            j += 1
        if i - j + 1 > burst:
            burst, burst_at = i - j + 1, t

    hyp_words = ' '.join(final_text[k] for k in order).split()
    report = {
        'audio_s': round(dur, 1),
        'device': eng.device_name,
        'preset': args.preset,
        'cpu_pct_one_core': round(100 * cpu / wall, 1),
        'stt_rtfx': round(1 / st['rtf'], 1) if st['rtf'] else None,
        'stt_call_ms_median': round(st['infer_ms'] or 0),
        'stt_audio_per_s': round(sum(h[0] for h in hist) / wall, 2),
        'stt_window_s_mean': round(sum(h[0] for h in hist) / max(1, sum(h[2] for h in hist)), 2),
        'stt_busy_pct': round(100 * sum(h[1] for h in hist) / wall, 1),
        'words_total': len(hyp_words),
        'cards': new_cards,
        'cards_mid_sentence': mid_breaks,
        'first_show_words_max': max(first_show, default=0),
        'burst_300ms_max': burst,
        'burst_at_s': round(burst_at / 1000, 2),
        'rewritten_words': rewritten,
        'rewrite_pct': round(100 * rewritten / max(1, len(hyp_words)), 1),
        'latency_first_word_s_median': round(float(np.median(first_word_lat)), 2) if first_word_lat else None,
        'latency_first_word_s_p90': round(float(np.percentile(first_word_lat, 90)), 2) if first_word_lat else None,
        'word_lag_s_median': round(float(np.median(word_lags)), 2) if word_lags else None,
        'word_lag_s_p90': round(float(np.percentile(word_lags, 90)), 2) if word_lags else None,
        'word_lag_s_max': round(max(word_lags), 2) if word_lags else None,
        'word_lag_s_max_at': round(word_lag_at[int(np.argmax(word_lags))], 2) if word_lags else None,
        'word_lag_s_max_word': lag_words[int(np.argmax(word_lags))] if word_lags else None,
        'word_lag_s_by_quarter': [round(float(np.median([l for l, t in zip(word_lags, word_lag_at)
                                                          if q * wall / 4 <= t < (q + 1) * wall / 4] or [0])), 2)
                                  for q in range(4)],
        'audio_queue_max_blocks': qmax[0],
        'slow': args.slow,
        'stages_ms_p50_p90': {k: [round(v[0] * 1000), round(v[1] * 1000)]
                              for k, v in stages.items() if not k.startswith('word_')},
        'tr_lag_s_median': round(float(np.median(tr_lags)), 2) if tr_lags else None,
        'tr_lag_s_p90': round(float(np.percentile(tr_lags, 90)), 2) if tr_lags else None,
        'tr_chunk_words_mean': round(float(np.mean(tr_chunks)), 2) if tr_chunks else None,
        'tr_chunk_words_p90': round(float(np.percentile(tr_chunks, 90)), 1) if tr_chunks else None,
        'tr_updates': len(tr_chunks),
        'tr_words_replaced': tr_dropped[0],
        'file_progress': round(file_job.progress, 3) if file_job else None,
        'pacing_lag_ms_median': int(np.median(lags)) if lags else 0,
        'pacing_lag_ms_p90': int(np.percentile(lags, 90)) if lags else 0,
    }
    if not args.no_reference:
        ref = reference_text(eng, audio)
        report.update(compare_text(ref, hyp_words))
        report['reference_words'] = len(ref)
        if translations:
            timeline += ['', '=== перевод ===', ' '.join(translations[k] for k in order if k in translations)]
        timeline += ['', '=== целыми фразами ===', ' '.join(ref),
                     '', '=== потоково ===', ' '.join(hyp_words)]

    print('\n=== отчёт ===')
    for k, v in report.items():
        print(f'  {k:30s} {v}')
    out = Path(args.out) if args.out else Path(args.audio).with_suffix('.jumps.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    tl = out.with_suffix('.timeline.txt')
    tl.write_text('\n'.join(timeline), encoding='utf-8')
    print(f'\nотчёт: {out}\nтаймлайн: {tl}')


if __name__ == '__main__':
    main()
