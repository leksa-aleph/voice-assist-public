"""
Хранение токена Discord-бота.

На Windows токен шифруется DPAPI: расшифровать его может только этот же
пользователь Windows на этом же компьютере. Скопированный файл чужому
бесполезен. На других системах лежит как есть, с правами только владельцу.
"""
from __future__ import annotations

import base64
import ctypes
import os
import sys
from ctypes import wintypes

from paths import user_data_path

TOKEN_FILE = 'discord_bot.token'
_ENTROPY = b'voice-assist/discord-bot'


class _Blob(ctypes.Structure):
    _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _dpapi(data: bytes, protect: bool) -> bytes:
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    src = _blob(data)
    ent = _blob(_ENTROPY)
    out = _Blob()
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(src), None, ctypes.byref(ent), None, None, 0x01, ctypes.byref(out))
    if not ok:
        raise OSError('DPAPI: %d' % ctypes.GetLastError())
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def save_token(token: str) -> None:
    path = user_data_path(TOKEN_FILE)
    raw = token.strip().encode('utf-8')
    if sys.platform == 'win32':
        payload = b'dpapi:' + base64.b64encode(_dpapi(raw, True))
    else:
        payload = b'plain:' + base64.b64encode(raw)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        f.write(payload)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def load_token() -> str:
    path = user_data_path(TOKEN_FILE)
    try:
        payload = path.read_bytes()
    except OSError:
        return ''
    try:
        kind, _, body = payload.partition(b':')
        data = base64.b64decode(body)
        if kind == b'dpapi':
            data = _dpapi(data, False)
        return data.decode('utf-8').strip()
    except Exception:
        return ''


def forget_token() -> None:
    try:
        user_data_path(TOKEN_FILE).unlink()
    except OSError:
        pass


def bot_id_from_token(token: str) -> str:
    """Первая часть токена - base64 от id бота (он же id приложения)."""
    try:
        head = token.strip().split('.')[0]
        head += '=' * (-len(head) % 4)
        value = base64.urlsafe_b64decode(head).decode('ascii')
        return value if value.isdigit() else ''
    except Exception:
        return ''


def looks_like_token(token: str) -> bool:
    parts = token.strip().split('.')
    return len(parts) == 3 and bool(bot_id_from_token(token)) and len(parts[2]) >= 20


def invite_url(bot_id: str) -> str:
    # Видеть каналы + подключаться. Говорить боту не нужно.
    perms = 1024 | 1048576
    return ('https://discord.com/oauth2/authorize?client_id=%s'
            '&scope=bot%%20applications.commands&permissions=%d' % (bot_id, perms))
