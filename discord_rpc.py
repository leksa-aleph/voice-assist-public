"""
Discord RPC: кто говорит в голосовом канале.

Авторизация:
  - хранится refresh_token и срок жизни, протухший токен обновляется молча;
  - окно авторизации показывается не больше MAX_AUTHORIZE_ATTEMPTS раз
    за запуск процесса, дальше RPC выключается до перезапуска;
  - при наличии rpc_token окно не показывается вообще;
  - выход из голосового канала не рвёт соединение;
  - всё состояние пишется в лог префиксом [discord-rpc].
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from paths import user_data_path

TOKEN_FILE = str(user_data_path('.rpc_token', legacy='tools/.rpc_token'))

HISTORY_SEC = 30.0
MIN_OVERLAP = 0.20

API_BASE = 'https://discord.com/api'
USER_AGENT = 'voice-assist/1.0'
REDIRECT_URI = 'http://localhost'
SCOPES = ['rpc', 'rpc.voice.read']

RECONNECT_S = 5.0
REFRESH_EARLY_S = 24 * 3600      # обновляем за сутки до истечения
MAX_AUTHORIZE_ATTEMPTS = 2       # больше двух окон подряд пользователь не увидит

# Один флаг на процесс. engine создаёт DiscordSpeakers на каждый старт записи,
# без этого счётчик окон сбрасывался бы кнопкой стоп/старт.
_auth_blocked = threading.Event()


class AuthBlocked(RuntimeError):
    """Авторизация не удалась, окно больше не показываем."""


class DiscordSpeakers:
    def __init__(self, client_id, client_secret):
        self._cid = client_id
        self._secret = client_secret
        self._pipe = None
        self._names = {}
        self._self_id = None
        self._segments = []          # [uid, start, end|None]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._nonce = 0
        self._channel_id = None
        self._pending_channel = None
        self._has_pending = False
        self._need_names = False
        self._auth_attempts = 0
        self.status = 'idle'

    def _log(self, text):
        if text != self.status:
            self.status = text
            print('[discord-rpc] %s' % text, flush=True)

    # ---------- транспорт ----------
    def _pack(self, op, payload):
        d = json.dumps(payload).encode()
        return struct.pack('<II', op, len(d)) + d

    def _read_exact(self, n):
        # На небуферизованном пайпе read(n) имеет право вернуть меньше.
        buf = b''
        while len(buf) < n:
            chunk = self._pipe.read(n - len(buf))
            if not chunk:
                raise ConnectionError('пайп закрыт')
            buf += chunk
        return buf

    def _read(self):
        op, ln = struct.unpack('<II', self._read_exact(8))
        return op, json.loads(self._read_exact(ln))

    def _cmd(self, name, args, evt=None):
        self._nonce += 1
        nonce = '%s-%d' % (name, self._nonce)
        msg = {'cmd': name, 'args': args, 'nonce': nonce}
        if evt:
            msg['evt'] = evt
        self._pipe.write(self._pack(1, msg))
        while True:
            _, d = self._read()
            if d.get('nonce') == nonce:
                return d
            # события, пришедшие пока ждём ответ, не теряем
            self._dispatch(d)

    def _close_pipe(self):
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass

    def _connect(self):
        self._close_pipe()
        for n in range(10):
            try:
                self._pipe = open(r'\\?\pipe\discord-ipc-%d' % n, 'r+b', buffering=0)
                return True
            except OSError:
                continue
        return False

    # ---------- токен ----------
    def _post(self, path, fields):
        body = urllib.parse.urlencode(fields).encode()
        req = urllib.request.Request(
            API_BASE + path, data=body,
            headers={'User-Agent': USER_AGENT,
                     'Content-Type': 'application/x-www-form-urlencoded'})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode('utf-8', 'replace')[:200]
            raise RuntimeError('%s -> HTTP %s %s' % (path, e.code, detail)) from None

    def _stored(self):
        try:
            with open(TOKEN_FILE, encoding='utf-8') as f:
                raw = f.read().strip()
        except OSError:
            return {}
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
        # старый формат: в файле лежал голый access_token, срок неизвестен
        return {'access_token': raw}

    def _save(self, data):
        directory = os.path.dirname(TOKEN_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = TOKEN_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, TOKEN_FILE)

    def _forget_token(self):
        try:
            os.remove(TOKEN_FILE)
        except OSError:
            pass

    def _store_grant(self, grant):
        data = {'access_token': grant['access_token']}
        if grant.get('refresh_token'):
            data['refresh_token'] = grant['refresh_token']
        if grant.get('expires_in'):
            data['expires_at'] = time.time() + float(grant['expires_in'])
        self._save(data)
        return data

    def _rpc_token(self):
        """Одноразовый токен, с которым Discord не показывает окно.
        Доступен одобренным приложениям и пользователям из списка тестеров."""
        try:
            answer = self._post('/oauth2/token/rpc',
                                {'client_id': self._cid,
                                 'client_secret': self._secret})
            return answer.get('rpc_token')
        except Exception as e:
            self._log('rpc_token недоступен, будет окно: %s' % e)
            return None

    def _authorize(self):
        if _auth_blocked.is_set():
            raise AuthBlocked('авторизация отключена до перезапуска')
        if self._auth_attempts >= MAX_AUTHORIZE_ATTEMPTS:
            _auth_blocked.set()
            raise AuthBlocked('авторизоваться не удалось, окно больше не показываем')
        self._auth_attempts += 1

        args = {'client_id': self._cid, 'scopes': SCOPES}
        rpc_token = self._rpc_token()
        if rpc_token:
            args['rpc_token'] = rpc_token

        self._log('запрос авторизации (%d из %d)'
                  % (self._auth_attempts, MAX_AUTHORIZE_ATTEMPTS))
        answer = self._cmd('AUTHORIZE', args)
        if answer.get('evt') == 'ERROR':
            raise RuntimeError('AUTHORIZE отклонён: %s' % (answer.get('data'),))

        grant = self._post('/oauth2/token', {
            'client_id': self._cid,
            'client_secret': self._secret,
            'grant_type': 'authorization_code',
            'code': answer['data']['code'],
            'redirect_uri': REDIRECT_URI,
        })
        self._log('авторизация получена, токен сохранён')
        return self._store_grant(grant)

    def _access_token(self):
        data = self._stored()
        token = data.get('access_token')
        expires_at = data.get('expires_at')

        # срок неизвестен (старый файл) - пробуем как есть
        if token and (expires_at is None
                      or time.time() < expires_at - REFRESH_EARLY_S):
            return token

        if data.get('refresh_token'):
            try:
                grant = self._post('/oauth2/token', {
                    'client_id': self._cid,
                    'client_secret': self._secret,
                    'grant_type': 'refresh_token',
                    'refresh_token': data['refresh_token'],
                })
                self._log('токен обновлён без окна')
                return self._store_grant(grant)['access_token']
            except Exception as e:
                self._log('обновить токен не удалось: %s' % e)

        if token:
            return token          # пусть AUTHENTICATE решит, живой он или нет
        return self._authorize()['access_token']

    # ---------- канал и события ----------
    def _load_channel(self):
        answer = self._cmd('GET_SELECTED_VOICE_CHANNEL', {})
        channel = answer.get('data')
        if not channel:
            return None
        with self._lock:
            for vs in channel.get('voice_states', []):
                u = vs['user']
                self._names[u['id']] = (vs.get('nick')
                                        or u.get('global_name')
                                        or u['username'])
        return channel

    def _switch_channel(self, channel_id):
        if channel_id == self._channel_id:
            return
        if self._channel_id:
            for e in ('SPEAKING_START', 'SPEAKING_STOP'):
                try:
                    self._cmd('UNSUBSCRIBE',
                              {'channel_id': self._channel_id}, evt=e)
                except Exception:
                    pass
        self._channel_id = channel_id
        with self._lock:
            self._segments = []
        if not channel_id:
            self._log('не в голосовом канале')
            return
        channel = self._load_channel()
        for e in ('SPEAKING_START', 'SPEAKING_STOP'):
            self._cmd('SUBSCRIBE', {'channel_id': channel_id}, evt=e)
        self._log('канал: %s' % ((channel or {}).get('name') or channel_id))

    def _dispatch(self, frame):
        evt = frame.get('evt')
        if evt in ('SPEAKING_START', 'SPEAKING_STOP'):
            self._on_speaking(evt, frame['data']['user_id'])
        elif evt == 'VOICE_CHANNEL_SELECT':
            self._pending_channel = (frame.get('data') or {}).get('channel_id')
            self._has_pending = True

    def _on_speaking(self, evt, uid):
        now = time.monotonic()
        with self._lock:
            if evt == 'SPEAKING_START':
                self._segments.append([uid, now, None])
            else:
                for seg in reversed(self._segments):
                    if seg[0] == uid and seg[2] is None:
                        seg[2] = now
                        break
            cutoff = now - HISTORY_SEC
            self._segments = [s for s in self._segments
                              if (s[2] or now) >= cutoff]
            unknown = uid not in self._names
        if unknown:
            self._need_names = True   # подтянем из основного цикла

    # ---------- жизненный цикл ----------
    def start(self):
        if os.environ.get('VOICE_ASSIST_NO_DISCORD'):
            self._log('выключен переменной окружения VOICE_ASSIST_NO_DISCORD')
            return
        if _auth_blocked.is_set():
            self._log('отключён до перезапуска приложения')
            return
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()
        self._close_pipe()

    def _run(self):
        while not self._stop.is_set():
            try:
                self._session()
            except AuthBlocked as e:
                self._log('отключён: %s' % e)
                self._close_pipe()
                return
            except Exception as e:
                self._log('ошибка: %s' % e)
            self._close_pipe()
            if _auth_blocked.is_set():
                self._log('отключён до перезапуска приложения')
                return
            if self._stop.wait(RECONNECT_S):
                break

    def _session(self):
        self._channel_id = None
        if not self._connect():
            self._log('Discord не запущен')
            return

        self._pipe.write(self._pack(0, {'v': 1, 'client_id': self._cid}))
        self._read()

        token = self._access_token()
        answer = self._cmd('AUTHENTICATE', {'access_token': token})
        if answer.get('evt') == 'ERROR':
            self._forget_token()
            raise RuntimeError('AUTHENTICATE: %s' % (answer.get('data'),))

        self._self_id = answer['data']['user']['id']
        self._auth_attempts = 0      # рабочая сессия - счётчик окон сбрасываем

        # Пока подписаны на смену канала, выход из войса не рвёт соединение.
        self._cmd('SUBSCRIBE', {}, evt='VOICE_CHANNEL_SELECT')
        current = self._load_channel()
        self._switch_channel(current['id'] if current else None)

        while not self._stop.is_set():
            _, frame = self._read()
            self._dispatch(frame)
            if self._has_pending:
                self._has_pending = False
                self._switch_channel(self._pending_channel)
            elif self._need_names:
                self._need_names = False
                try:
                    self._load_channel()
                except Exception:
                    pass

    # ---------- публичное ----------
    def who_spoke(self, t_start, t_end):
        """Имя, если в интервале говорил ровно один. Иначе None."""
        now = time.monotonic()
        totals = {}
        with self._lock:
            for uid, s, e in self._segments:
                if uid == self._self_id:
                    continue
                e = e if e is not None else now
                ov = min(t_end, e) - max(t_start, s)
                if ov > 0:
                    totals[uid] = totals.get(uid, 0.0) + ov
            active = [u for u, v in totals.items() if v >= MIN_OVERLAP]
            if len(active) == 1:
                return self._names.get(active[0])
        return None
