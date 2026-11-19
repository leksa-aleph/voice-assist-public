"""
Кроссплатформенный выбор и открытие аудиоисточников.

Четыре публичные функции:
  list_microphones, list_system_audio - перечисление,
  open_microphone, open_system_audio - открытие потока.

Linux: PulseAudio/PipeWire через sounddevice. Для захвата конкретного
приложения - роутинг в виртуальный sink RecordSink через pw-link.

Windows: WASAPI. Микрофон - обычный input через sounddevice.
Системный звук - per-process через proc-tap (Windows Process Loopback API,
требует Win 10 20H1+). Список приложений с активной аудио-сессией берём
через pycaw.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from resample import Rebuffer, StreamResampler

log = logging.getLogger('audio_source')


IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"


# ============================================================
# Публичные структуры
# ============================================================

@dataclass
class AudioInput:
    """Описание источника для показа в UI и последующего открытия."""
    id: str            # внутренний идентификатор для open_*
    name: str          # короткое имя для отображения
    description: str   # подсказка (тип устройства, приложение и т.п.)


# ============================================================
# LINUX РЕАЛИЗАЦИЯ
# ============================================================

LINK_SINK = "RecordSink"  # виртуальный sink, создан null-sink модулем PipeWire


def _pactl(args: list[str], timeout: float = 2.0) -> str:
    """Запустить pactl и вернуть stdout. Пустая строка при ошибке."""
    try:
        r = subprocess.run(
            ["pactl"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _pw_link(args: list[str], timeout: float = 2.0) -> str:
    try:
        r = subprocess.run(
            ["pw-link"] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout if r.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _list_microphones_linux() -> list[AudioInput]:
    """
    Парсит `pactl list sources` и возвращает реальные физические/виртуальные
    источники ввода. Отфильтровываем monitor-источники (.monitor) - это loopback'и sink'ов,
    для микрофонов они не нужны.
    """
    out = _pactl(["list", "sources"])
    if not out:
        return []

    sources = []
    current = {}
    for line in out.splitlines():
        if line.startswith("Source #"):
            if current:
                sources.append(current)
            current = {"id": "", "description": "", "monitor_of": ""}
            continue
        if not current:
            continue
        s = line.strip()
        if s.startswith("Name: "):
            current["id"] = s.split(": ", 1)[1]
        elif s.startswith("Description: "):
            current["description"] = s.split(": ", 1)[1]
        elif s.startswith("Monitor of Sink: "):
            current["monitor_of"] = s.split(": ", 1)[1]
    if current:
        sources.append(current)

    result = []
    for src in sources:
        # Убираем мониторы и пустые
        if src["monitor_of"] and src["monitor_of"] != "n/a":
            continue
        if not src["id"]:
            continue
        result.append(AudioInput(
            id=src["id"],
            name=src["description"] or src["id"],
            description="Микрофон",
        ))
    return result


def _list_system_audio_linux() -> list[AudioInput]:
    """
    Возвращает список приложений, которые сейчас играют звук
    (parsing `pactl list sink-inputs`). Группирует по node.name.
    Плюс опция "весь системный звук" (default sink monitor) как fallback.
    """
    out = _pactl(["list", "sink-inputs"])
    apps_by_node: dict[str, dict] = {}

    if out:
        current = None
        for line in out.splitlines():
            if line.startswith("Sink Input #"):
                if current and current.get("node"):
                    node = current["node"]
                    if current["app"] == "RecordSink":
                        current = None
                        continue
                    if node not in apps_by_node:
                        apps_by_node[node] = {
                            "node": node,
                            "app": current["app"],
                            "media": current["media"],
                            "count": 1,
                        }
                    else:
                        apps_by_node[node]["count"] += 1
                current = {"app": "?", "media": "", "node": ""}
                continue
            if current is None:
                continue
            s = line.strip()
            if s.startswith('application.name = '):
                current["app"] = s.split(" = ", 1)[1].strip().strip('"')
            elif s.startswith('media.name = '):
                current["media"] = s.split(" = ", 1)[1].strip().strip('"')
            elif s.startswith('node.name = '):
                current["node"] = s.split(" = ", 1)[1].strip().strip('"')
        # последний sink-input - в цикле не успел попасть в apps_by_node
        if current and current.get("node") and current.get("app") != "RecordSink":
            node = current["node"]
            if node not in apps_by_node:
                apps_by_node[node] = {
                    "node": node,
                    "app": current["app"],
                    "media": current["media"],
                    "count": 1,
                }

    result = []
    for node, info in sorted(apps_by_node.items()):
        media = info["media"]
        if len(media) > 60:
            media = media[:57] + "..."
        suffix = f" (×{info['count']})" if info["count"] > 1 else ""
        result.append(AudioInput(
            id=f"app:{node}",
            name=f"{info['app']}{suffix}",
            description=media or "-",
        ))

    # Fallback: вся звуковая карта (monitor дефолтного sink)
    result.append(AudioInput(
        id="system:default",
        name="Весь системный звук",
        description="Всё что играет в этот момент",
    ))
    return result


# --- Роутинг приложения в RecordSink ---

def _get_pw_outputs(app_node: str) -> list[str]:
    out = _pw_link(["-o"])
    return [
        ln.strip() for ln in out.splitlines()
        if ln.strip().startswith(f"{app_node}:output_")
    ]


def _get_recordsink_inputs() -> dict[str, list[str]]:
    out = _pw_link(["-l", LINK_SINK])
    result: dict[str, list[str]] = {}
    current_channel: Optional[str] = None
    for line in out.splitlines():
        m = re.match(rf"^{re.escape(LINK_SINK)}:playback_(\w+)\s*$", line)
        if m:
            current_channel = m.group(1)
            result.setdefault(current_channel, [])
            continue
        if re.match(rf"^{re.escape(LINK_SINK)}:", line):
            current_channel = None
            continue
        if current_channel and line.lstrip().startswith("|<-"):
            src = line.split("|<-", 1)[1].strip()
            result[current_channel].append(src)
        elif not line.startswith(" ") and not line.startswith("\t"):
            current_channel = None
    return result


def _app_linked(app_node: str) -> bool:
    inputs = _get_recordsink_inputs()
    for sources in inputs.values():
        for src in sources:
            if src.startswith(f"{app_node}:output_"):
                return True
    return False


def _link_app_to_recordsink(app_node: str) -> bool:
    """Линкует все output_* приложения в playback_* RecordSink."""
    ports = _get_pw_outputs(app_node)
    if not ports:
        return False
    ok = True
    for port in ports:
        suffix = port.rsplit("_", 1)[-1]
        target = f"{LINK_SINK}:playback_{suffix}"
        try:
            r = subprocess.run(
                ["pw-link", port, target],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode != 0 and "exists" not in (r.stderr or "").lower():
                ok = False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ok = False
    return ok


def ensure_app_linked(app_node: str) -> bool:
    """Гарантирует что приложение залинковано в RecordSink. True если связано."""
    if _app_linked(app_node):
        return True
    if not _get_pw_outputs(app_node):
        return False  # приложение не играет звук сейчас
    _link_app_to_recordsink(app_node)
    return _app_linked(app_node)


# --- Открытие потоков ---

def _open_pulse_stream(pulse_source: str, samplerate: int, blocksize: int,
                      callback: Callable) -> sd.InputStream:
    """
    Открывает sounddevice.InputStream через pulse-backend с указанным
    PULSE_SOURCE. sounddevice читает переменную в момент open(), поэтому
    переключаем её прямо перед созданием стрима.
    """
    prev = os.environ.get("PULSE_SOURCE")
    os.environ["PULSE_SOURCE"] = pulse_source
    try:
        stream = sd.InputStream(
            device="pulse",
            channels=1,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype="float32",
            callback=callback,
        )
    finally:
        # Восстанавливаем (хотя следующие open уже выбрали что им надо)
        if prev is None:
            os.environ.pop("PULSE_SOURCE", None)
        else:
            os.environ["PULSE_SOURCE"] = prev
    return stream


def _open_microphone_linux(source_id: str, samplerate: int, blocksize: int,
                          callback: Callable) -> sd.InputStream:
    return _open_pulse_stream(source_id, samplerate, blocksize, callback)


def _open_system_audio_linux(source_id: str, samplerate: int, blocksize: int,
                            callback: Callable) -> sd.InputStream:
    """
    source_id:
      "app:<node_name>"  - линкуем приложение в RecordSink, читаем RecordSink.monitor
      "system:default"   - читаем @DEFAULT_MONITOR@ (мониторы дефолтного sink)
    """
    if source_id.startswith("app:"):
        app_node = source_id[len("app:"):]
        ensure_app_linked(app_node)  # best effort, не падаем если не залинковалось
        pulse_source = f"{LINK_SINK}.monitor"
    elif source_id == "system:default":
        pulse_source = "@DEFAULT_MONITOR@"
    else:
        raise ValueError(f"Unknown system audio source_id: {source_id}")
    return _open_pulse_stream(pulse_source, samplerate, blocksize, callback)


# ============================================================
# WINDOWS РЕАЛИЗАЦИЯ
# ============================================================

def _find_wasapi_host_api_index() -> Optional[int]:
    """Возвращает индекс host API 'Windows WASAPI' в sounddevice, или None."""
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if 'WASAPI' in api.get('name', ''):
                return i
    except Exception as e:
        log.error(f'query_hostapis failed: {e}')
    return None


def _list_microphones_windows() -> list[AudioInput]:
    """
    Микрофоны через WASAPI host API в sounddevice.
    id = строковый индекс устройства (например, "12").
    """
    wasapi_idx = _find_wasapi_host_api_index()
    if wasapi_idx is None:
        log.error('WASAPI host API не найден')
        return []

    result: list[AudioInput] = []
    try:
        devices = sd.query_devices()
    except Exception as e:
        log.error(f'query_devices failed: {e}')
        return []

    for i, dev in enumerate(devices):
        if dev.get('hostapi') != wasapi_idx:
            continue
        if dev.get('max_input_channels', 0) <= 0:
            continue
        name = dev.get('name', f'device {i}')
        # Loopback-устройства WASAPI тоже выглядят как input - отсекаем по суффиксу
        if '[Loopback]' in name:
            continue
        result.append(AudioInput(
            id=str(i),
            name=name,
            description='Микрофон (WASAPI)',
        ))
    log.debug(f'Найдено микрофонов (Windows): {len(result)}')
    return result



def _list_system_audio_windows() -> list[AudioInput]:
    """
    Список процессов, имеющих активную аудио-сессию (pycaw).
    id = "proc:<exe_name>" - имя exe, т.к. PID меняется между запусками.
    Если одно имя на несколько сессий - берём первое.
    """
    try:
        from pycaw.pycaw import AudioUtilities
    except Exception as e:
        log.error(f'pycaw import failed: {e}')
        return []

    try:
        sessions = AudioUtilities.GetAllSessions()
    except Exception as e:
        log.error(f'GetAllSessions failed: {e}')
        return []

    seen: dict[str, AudioInput] = {}
    for s in sessions:
        proc = s.Process
        if proc is None:
            # системные звуки пропускаем
            continue
        try:
            name = proc.name() or ''
            pid = proc.pid
        except Exception:
            continue
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        # Чистое имя для отображения (без .exe)
        display = name[:-4] if name.lower().endswith('.exe') else name
        seen[key] = AudioInput(
            id=f'proc:{name}',
            name=display,
            description=f'PID {pid}',
        )

    result = list(seen.values())
    log.debug(f'Найдено приложений с аудио-сессией: {len(result)}')
    return result


def _open_microphone_windows(source_id: str, samplerate: int, blocksize: int,
                            callback: Callable):
    """
    Открывает микрофон на его нативной частоте (WASAPI редко поддерживает
    16kHz напрямую) и ресемплит в callback до запрошенной частоты.
    """
    try:
        device_idx = int(source_id)
    except ValueError:
        raise RuntimeError(f'Invalid Windows mic source_id: {source_id!r}')

    dev_info = sd.query_devices(device_idx)
    native_sr = int(dev_info.get('default_samplerate') or 48000)

    log.info(f'Open microphone (Windows): device={device_idx}, '
             f'native_sr={native_sr}, target_sr={samplerate}, block={blocksize}')

    if native_sr == samplerate:
        # частоты совпадают - обычный путь без ресемпла
        return sd.InputStream(
            device=device_idx,
            channels=1,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype='float32',
            callback=callback,
        )

    # нужен ресемпл - оборачиваем callback. Ресемплер потоковый: помнит
    # хвост прошлого блока, поэтому на стыках нет щелчков.
    resampler = StreamResampler(native_sr, samplerate)
    rebuf = Rebuffer(blocksize)
    lock = threading.Lock()

    native_block = max(1, int(round(blocksize * native_sr / samplerate)))

    def wrapped_cb(indata, frames, time_info, status):
        try:
            with lock:
                blocks = rebuf.push(resampler.process(indata[:, 0]))
            for chunk in blocks:
                callback(chunk.reshape(-1, 1), blocksize, time_info, status)
        except Exception as e:
            log.error(f'mic resample callback error: {e}')

    return sd.InputStream(
        device=device_idx,
        channels=1,
        samplerate=native_sr,
        blocksize=native_block,
        dtype='float32',
        callback=wrapped_cb,
    )


class _NativePCM:
    """
    Подменяет конвертер proc-tap. Если карта работает не в 48 кГц float,
    proc-tap пересчитывает звук через scipy и кусками - на стыках кусков
    появляются щелчки. Мы забираем звук как есть, запоминаем формат и
    пересчитываем своим потоковым ресемплером.
    """

    def __init__(self, src_rate: int = 48000, src_channels: int = 2, src_width: int = 4, **_):
        self.rate = int(src_rate)
        self.channels = max(1, int(src_channels))
        self.width = int(src_width)

    def convert(self, pcm: bytes) -> bytes:
        return pcm


def _patch_proctap():
    try:
        import proctap.backends.windows as w
        w.AudioConverter = _NativePCM
    except Exception as e:
        log.warning(f'proctap: не удалось подменить конвертер: {e}')


def decode_pcm(pcm: bytes, channels: int, width: int) -> np.ndarray:
    """Байты WASAPI -> моно float32. 4 байта - float32, 2 - int16."""
    if width == 2:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        x = np.frombuffer(pcm, dtype=np.float32)
    if channels > 1:
        n = x.size // channels
        x = x[:n * channels].reshape(n, channels).mean(axis=1)
    return x

class _ProcTapStream:
    """
    Адаптер ProcessAudioCapture под интерфейс sd.InputStream.

    Discord (и не только) держит несколько процессов с одним именем: часть
    служебные и отдают тишину, голос идёт лишь в одном. Какой именно -
    заранее неизвестно и меняется между запусками. Поэтому открываем все
    сразу и передаём наружу тот поток, в котором появился звук.

    На вход от ProcTap идут чанки 48kHz / stereo / float32 (bytes).
    Конвертируем: stereo -> mono (среднее каналов), 48000 -> samplerate.
    Затем отдаём в callback в формате sounddevice: (frames, 1), float32.
    """
    NATIVE_SR = 48000
    NATIVE_CH = 2

    SILENCE_PEAK = 1e-4    # ниже этого считаем блок тишиной (~-80 dBFS)
    SWITCH_AFTER_S = 3.0   # молчит столько - разрешаем переключиться на другой

    def __init__(self, exe_name: str, samplerate: int, blocksize: int,
                 callback: Callable):
        self._exe_name = exe_name
        self._target_sr = samplerate
        self._blocksize = blocksize
        self._user_cb = callback
        self._taps: list = []
        self._rebuf = Rebuffer(blocksize)
        # у каждого процесса свой поток и свой ресемплер: состояние фильтра
        # одного процесса не должно попадать в другой
        self._resamplers: dict[int, StreamResampler] = {}
        self._formats: dict[int, tuple[int, int, int]] = {}   # pid -> (частота, каналы, байт на отсчёт)
        self._lock = threading.Lock()

        # Выбор живого процесса
        self._active_pid: Optional[int] = None
        self._last_signal: dict[int, float] = {}

    def _resolve_pids(self) -> list[int]:
        """Все PID с таким именем, у которых есть аудио-сессия."""
        # pycaw только на Windows - отложенный импорт
        from pycaw.pycaw import AudioUtilities
        sessions = AudioUtilities.GetAllSessions()
        pids: list[int] = []
        for s in sessions:
            proc = s.Process
            if proc is None:
                continue
            try:
                if (proc.name() or '').lower() != self._exe_name.lower():
                    continue
                pid = int(proc.pid)
            except Exception:
                continue
            if pid not in pids:
                pids.append(pid)
        if not pids:
            raise RuntimeError(
                f'Процесс {self._exe_name} не имеет активной аудио-сессии'
            )
        return pids

    def _on_data(self, pid: int, pcm: bytes, frames: int):
        try:
            sr, ch, width = self._formats.get(pid, (self.NATIVE_SR, self.NATIVE_CH, 4))
            samples = decode_pcm(pcm, ch, width)

            peak = float(np.abs(samples).max()) if samples.size else 0.0
            now = time.monotonic()

            with self._lock:
                rs = self._resamplers.get(pid)
                if rs is None:
                    rs = StreamResampler(sr, self._target_sr)
                    self._resamplers[pid] = rs
                samples = rs.process(samples)

                if peak >= self.SILENCE_PEAK:
                    self._last_signal[pid] = now

                    # Первый заговоривший становится активным. Позже сменить
                    # его может только тот, кто звучит, когда активный молчит
                    # дольше SWITCH_AFTER_S (например, Discord пересоздал
                    # процесс голосового движка).
                    if self._active_pid is None:
                        self._active_pid = pid
                        self._rebuf.reset()
                        log.info(f'Активный аудио-процесс: pid={pid}')
                    elif pid != self._active_pid:
                        quiet = now - self._last_signal.get(self._active_pid, 0.0)
                        if quiet >= self.SWITCH_AFTER_S:
                            log.info(f'Переключение аудио-процесса: '
                                     f'pid={self._active_pid} -> {pid}')
                            self._active_pid = pid
                            self._rebuf.reset()

                # До первого звука не отдаём ничего: VAD всё равно молчит,
                # а лишние источники смешивать нельзя.
                if pid != self._active_pid:
                    return

                # Отдаём ровно по blocksize, как делает sd.InputStream
                ready = self._rebuf.push(samples)

            for chunk in ready:
                try:
                    self._user_cb(chunk.reshape(-1, 1), self._blocksize,
                                  None, None)
                except Exception as e:
                    log.error(f'ProcTap user callback error: {e}')

        except Exception as e:
            log.error(f'ProcTap _on_data error: {e}')

    @staticmethod
    def _check_process_specific(tap, pid: int):
        # ProcTap при неудаче молча переходит на весь системный звук.
        # None - проверка недоступна, не мешаем работе.
        native = getattr(getattr(tap, "_backend", None), "_native", None)
        checker = getattr(native, "is_process_specific", None)
        if checker is None:
            return None
        try:
            ok = bool(checker())
        except Exception as e:
            log.warning(f"is_process_specific() недоступен для pid={pid}: {e}")
            return None
        if not ok:
            reason = ""
            getter = getattr(native, "get_last_error", None)
            if getter is not None:
                try:
                    reason = (getter() or "").strip()
                except Exception:
                    reason = ""
            log.error(f"ProcTap pid={pid}: нет захвата отдельного процесса, "
                      f"взят весь системный звук. {reason}")
        return ok

    def start(self):
        # proctap только на Windows - импорт тут чтобы файл подгружался на Linux
        from proctap import ProcessAudioCapture
        _patch_proctap()
        pids = self._resolve_pids()
        log.info(f'Open system audio (Windows): proc={self._exe_name} '
                 f'pids={pids} target_sr={self._target_sr}')

        errors = []
        not_specific = []
        for pid in pids:
            def on_data(pcm, frames, _pid=pid):
                self._on_data(_pid, pcm, frames)
            try:
                tap = ProcessAudioCapture(pid, on_data=on_data)
                conv = getattr(getattr(tap, '_backend', None), '_converter', None)
                if isinstance(conv, _NativePCM):
                    self._formats[pid] = (conv.rate, conv.channels, conv.width)
                    log.info(f'pid={pid}: формат карты {conv.rate} Гц, {conv.channels} кан., '
                             f'{conv.width * 8} бит - пересчитываем сами')
                tap.start()
            except Exception as e:
                errors.append(f'pid={pid}: {e}')
                continue
            self._taps.append(tap)
            if self._check_process_specific(tap, pid) is False:
                not_specific.append(pid)

        if not_specific:
            for t in self._taps:
                try:
                    t.close()
                except Exception:
                    pass
            self._taps = []
            raise RuntimeError(
                "Звук отдельного приложения недоступен: система отдаёт "
                "весь звук сразу. Нужен Windows 11 (сборка 22000 или новее)."
            )

        if not self._taps:
            raise RuntimeError(
                f'Не удалось открыть {self._exe_name}: {"; ".join(errors)}'
            )
        if errors:
            log.warning(f'Часть процессов не открылась: {"; ".join(errors)}')

    def stop(self):
        for tap in self._taps:
            try:
                tap.stop()
            except Exception as e:
                log.warning(f'ProcTap stop error: {e}')

    def close(self):
        for tap in self._taps:
            try:
                tap.close()
            except Exception as e:
                log.warning(f'ProcTap close error: {e}')
        self._taps = []
        self._active_pid = None
        self._resamplers = {}
        self._rebuf.reset()



def _open_system_audio_windows(source_id: str, samplerate: int, blocksize: int,
                              callback: Callable):
    """
    source_id: "proc:<exe_name>" - захватываем аудио конкретного процесса
    через ProcTap (WASAPI process loopback). Имя процесса (а не PID), потому
    что PID меняется между запусками приложения.
    """
    if not source_id.startswith('proc:'):
        raise RuntimeError(f'Invalid Windows system audio source_id: {source_id!r}')
    exe_name = source_id[len('proc:'):]
    if not exe_name:
        raise RuntimeError('Empty process name in source_id')
    return _ProcTapStream(exe_name, samplerate, blocksize, callback)

# ============================================================
# ПУБЛИЧНЫЙ API
# ============================================================

def list_microphones() -> list[AudioInput]:
    if IS_LINUX:
        return _list_microphones_linux()
    if IS_WINDOWS:
        return _list_microphones_windows()
    raise RuntimeError(f"Unsupported platform: {platform.system()}")


def resolve_microphone(source_id: Optional[str], name: Optional[str],
                       mics: Optional[list[AudioInput]] = None) -> Optional[AudioInput]:
    """
    На Windows id микрофона - номер устройства в sounddevice, и он меняется,
    когда подключают или отключают другое устройство. Поэтому сначала
    сверяем имя: тот же номер с другим именем - это уже чужой микрофон.
    """
    if mics is None:
        mics = list_microphones()
    by_id = {m.id: m for m in mics}
    if source_id in by_id and (not name or by_id[source_id].name == name):
        return by_id[source_id]
    if name:
        for m in mics:
            if m.name == name:
                return m
    return None


def list_system_audio() -> list[AudioInput]:
    if IS_LINUX:
        return _list_system_audio_linux()
    if IS_WINDOWS:
        return _list_system_audio_windows()
    raise RuntimeError(f"Unsupported platform: {platform.system()}")


def open_microphone(source_id: str, samplerate: int, blocksize: int,
                   callback: Callable):
    if IS_LINUX:
        return _open_microphone_linux(source_id, samplerate, blocksize, callback)
    if IS_WINDOWS:
        return _open_microphone_windows(source_id, samplerate, blocksize, callback)
    raise RuntimeError(f"Unsupported platform: {platform.system()}")


def open_system_audio(source_id: str, samplerate: int, blocksize: int,
                     callback: Callable):
    if IS_LINUX:
        return _open_system_audio_linux(source_id, samplerate, blocksize, callback)
    if IS_WINDOWS:
        return _open_system_audio_windows(source_id, samplerate, blocksize, callback)
    raise RuntimeError(f"Unsupported platform: {platform.system()}")


# ============================================================
# CLI для отладки
# ============================================================

if __name__ == "__main__":
    print(f'Платформа: {platform.system()}')
    print()

    print("Микрофоны:")
    try:
        for m in list_microphones():
            print(f"  [{m.id}]")
            print(f"     {m.name}  - {m.description}")
    except Exception as e:
        print(f'  ОШИБКА: {e}')

    print("\nСистемный звук:")
    try:
        for s in list_system_audio():
            print(f"  [{s.id}]")
            print(f"     {s.name}  - {s.description}")
    except Exception as e:
        print(f'  ОШИБКА: {e}')
