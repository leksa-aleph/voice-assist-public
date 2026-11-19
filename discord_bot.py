"""
Discord-бот: запускает Node-мост (discord_bridge/bridge.mjs) и разбирает
его поток. Звук каждого участника уходит в движок отдельной дорожкой
с точным именем, события - в интерфейс через Qt-сигналы.

Почему Node, а не Python: с марта 2026 Discord пускает в голосовые каналы
только клиентов с шифрованием DAVE. Приём звука с DAVE стабильно работает
в официальной @discordjs/voice (0.19.2+), у Python-библиотек - только форки.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import threading
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

import overlay
import secret_store
from paths import resource_path

MIN_NODE_MAJOR = 22


def _bridge_dir():
    return resource_path('discord_bridge')


def find_node() -> Optional[str]:
    bundled = _bridge_dir() / ('node.exe' if sys.platform == 'win32' else 'node')
    if bundled.exists():
        return str(bundled)
    return shutil.which('node')


def node_ok(node: str) -> bool:
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        out = subprocess.run([node, '--version'], capture_output=True, text=True,
                             timeout=10, creationflags=flags).stdout.strip()
        major = int(out.lstrip('v').split('.')[0])
        return major >= MIN_NODE_MAJOR
    except Exception:
        return False


def bridge_ready() -> tuple[bool, str]:
    """(готов ли мост, причина если нет)."""
    d = _bridge_dir()
    if not (d / 'bridge.mjs').exists():
        return False, 'no-bridge'
    if not (d / 'node_modules' / '@discordjs' / 'voice').exists():
        return False, 'no-modules'
    node = find_node()
    if not node:
        return False, 'no-node'
    if not node_ok(node):
        return False, 'old-node'
    return True, ''


class DiscordBridge(QObject):
    state_changed = pyqtSignal(str, str)        # state, detail
    bot_ready = pyqtSignal(dict)
    channel_changed = pyqtSignal(object)        # {'guild', 'channel'} | None
    members_changed = pyqtSignal(list)
    speaking_changed = pyqtSignal(str, bool)
    follow_changed = pyqtSignal(str, str)       # user_id, name
    guilds_changed = pyqtSignal(int)

    def __init__(self, engine, config):
        super().__init__()
        self.engine = engine
        self.config = config
        self.state = 'off'
        self.detail = ''
        self.bot: dict = {}
        self.channel: Optional[dict] = None
        self.members: dict[str, dict] = {}
        self.speaking: set[str] = set()
        self.follow_id: Optional[str] = config.discord_follow_user_id
        self.guilds = 0
        self.follow_name = ''
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._skip_self = False

    # ---------- управление ----------

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        if self.running():
            return True
        token = secret_store.load_token()
        if not token:
            self._set_state('no-token', '')
            return False
        ok, why = bridge_ready()
        if not ok:
            self._set_state('error', why)
            return False

        env = dict(os.environ)
        env['DISCORD_TOKEN'] = token
        if self.follow_id:
            env['FOLLOW_USER_ID'] = self.follow_id
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        try:
            self._proc = subprocess.Popen(
                [find_node(), str(_bridge_dir() / 'bridge.mjs')],
                cwd=str(_bridge_dir()), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=flags, bufsize=0,
            )
        except OSError as e:
            self._set_state('error', str(e))
            return False
        proc = self._proc
        threading.Thread(target=self._read_stdout, args=(proc,), daemon=True,
                         name='discord-bridge').start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True,
                         name='discord-bridge-log').start()
        self._set_state('connecting', '')
        self._update_skip()
        return True

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                self._send_to(proc, {'cmd': 'quit'})
                proc.stdin.close()
            except Exception:
                pass

            def reap():
                try:
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            threading.Thread(target=reap, daemon=True).start()
        self._clear_voice()
        self._set_state('off', '')

    def restart(self):
        self.stop()
        self.start()

    def set_follow(self, user_id: Optional[str]):
        self.follow_id = user_id
        self.config.discord_follow_user_id = user_id
        self._send({'cmd': 'follow', 'user_id': user_id})

    def set_skip_self(self, skip: bool):
        """Свой голос и так идёт с микрофона - от бота его не берём."""
        self._skip_self = skip
        self._update_skip()

    def _update_skip(self):
        self._send({'cmd': 'skip', 'user_id': self.follow_id if self._skip_self else None})

    def _send(self, msg: dict):
        proc = self._proc
        if proc is not None:
            self._send_to(proc, msg)

    @staticmethod
    def _send_to(proc, msg: dict):
        try:
            proc.stdin.write((json.dumps(msg) + '\n').encode('utf-8'))
            proc.stdin.flush()
        except Exception:
            pass

    def _set_state(self, state: str, detail: str):
        self.state = state
        self.detail = detail
        self.state_changed.emit(state, detail)

    def _clear_voice(self):
        for uid in list(self.speaking):
            overlay.push_speaking('dc:' + uid, None, False)
        self.speaking.clear()
        with self._lock:
            self.members = {}
        self.channel = None
        self.engine.set_bot_in_voice(False)
        self.channel_changed.emit(None)
        self.members_changed.emit([])

    # ---------- чтение моста ----------

    def _read_stdout(self, proc: subprocess.Popen):
        out = proc.stdout
        head = struct.Struct('<BI')
        try:
            while True:
                h = self._read_exact(out, 5)
                if h is None:
                    break
                kind, n = head.unpack(h)
                payload = self._read_exact(out, n)
                if payload is None:
                    break
                if kind == 2:
                    self._on_audio(payload)
                elif kind == 1:
                    try:
                        self._on_event(json.loads(payload.decode('utf-8')))
                    except Exception as e:
                        print(f'[discord] bad event: {e}', file=sys.stderr)
        finally:
            code = proc.poll()
            if proc is self._proc or self._proc is None:
                if self.state not in ('off', 'error'):
                    self._set_state('error', f'exit-{code}')
                self._clear_voice()

    @staticmethod
    def _read_exact(f, n: int) -> Optional[bytes]:
        buf = b''
        while len(buf) < n:
            chunk = f.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _read_stderr(self, proc: subprocess.Popen):
        for line in iter(proc.stderr.readline, b''):
            text = line.decode('utf-8', 'replace').rstrip()
            if text:
                print(f'[discord] {text}', flush=True)

    def _on_audio(self, payload: bytes):
        uid = str(struct.unpack_from('<Q', payload, 0)[0])
        if self._skip_self and uid == self.follow_id:
            return
        pcm = np.frombuffer(payload, dtype='<i2', offset=8).astype(np.float32) / 32768.0
        with self._lock:
            name = (self.members.get(uid) or {}).get('name') or 'Discord'
        self.engine.bot_audio(uid, name, pcm)

    def _on_event(self, ev: dict):
        e = ev.get('evt')
        if e == 'status':
            state = ev.get('state', '')
            if state == 'error' and ev.get('detail') == 'bad-token':
                self._set_state('bad-token', '')
            else:
                self._set_state(state, ev.get('detail', '') or '')
        elif e == 'ready':
            self.bot = ev.get('bot') or {}
            self.guilds = int(ev.get('guilds') or 0)
            fid = ev.get('follow_id')
            if fid and not self.follow_id:
                self.follow_id = fid
                self._update_skip()
            if fid and fid == self.follow_id:
                self.follow_name = ev.get('follow_name') or ''
            self.bot_ready.emit(self.bot)
            self.guilds_changed.emit(self.guilds)
        elif e == 'guilds':
            self.guilds = int(ev.get('count') or 0)
            self.guilds_changed.emit(self.guilds)
        elif e == 'channel':
            ch = ev.get('channel')
            if ch:
                self.channel = {'guild': ev.get('guild') or {}, 'channel': ch}
                self.engine.set_bot_in_voice(True)
                self.channel_changed.emit(self.channel)
            else:
                self._clear_voice()
        elif e == 'members':
            members = ev.get('members') or []
            with self._lock:
                self.members = {m['id']: m for m in members}
            self.members_changed.emit(members)
        elif e == 'speaking':
            uid = str(ev.get('user_id'))
            active = bool(ev.get('active'))
            if active:
                self.speaking.add(uid)
            else:
                self.speaking.discard(uid)
            self.speaking_changed.emit(uid, active)
            # «призрак» в оверлее показывает, кто начал говорить, ещё до
            # текста. Себя не показываем: человек и так знает, что говорит.
            if uid != self.follow_id:
                with self._lock:
                    name = (self.members.get(uid) or {}).get('name')
                overlay.push_speaking('dc:' + uid, name, active)
        elif e == 'left':
            uid = str(ev.get('user_id'))
            self.engine.bot_user_left(uid)
        elif e == 'follow':
            uid = str(ev.get('user_id'))
            self.follow_id = uid
            self.follow_name = ev.get('name') or ''
            self.config.discord_follow_user_id = uid
            self._update_skip()
            self.follow_changed.emit(uid, ev.get('name') or '')
