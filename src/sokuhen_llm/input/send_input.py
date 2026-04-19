"""Text injection for Windows.

Primary path: PostMessage(WM_CHAR) to the focused window. This delivers
each character directly to the window's message queue as a character-input
event, without going through TSF or the keyboard-queue translation layer.
Electron/Chromium apps (including Claude desktop) were treating SendInput
KEYEVENTF_UNICODE events as IME-composition input, requiring a second
Enter to commit — WM_CHAR avoids that entirely.

Fallback: SendInput with KEYEVENTF_UNICODE for anything that doesn't
respond to WM_CHAR (rare — mostly older games / fullscreen apps).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import sys


log = logging.getLogger(__name__)


WM_CHAR = 0x0102
WM_IME_CHAR = 0x0286


# --- Win32 structures --------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wt.LONG),
        ("dy", wt.LONG),
        ("mouseData", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wt.DWORD),
        ("wParamL", wt.WORD),
        ("wParamH", wt.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wt.DWORD), ("u", _INPUT_UNION)]


from . import _win32


_user32 = _win32.user32


if _user32 is not None:
    # SendInput-specific signature. The rest (GetForegroundWindow,
    # GetGUIThreadInfo, PostMessageW, ...) is declared in _win32 so the
    # struct types don't collide with other modules.
    _user32.SendInput.argtypes = [wt.UINT, ctypes.c_void_p, ctypes.c_int]
    _user32.SendInput.restype = wt.UINT


def _send(inputs: list[_INPUT]) -> int:
    if _user32 is None:
        return 0
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    return _user32.SendInput(n, arr, ctypes.sizeof(_INPUT))


def send_unicode_text(text: str) -> int:
    """Inject a string into the focused window.

    Tries WM_CHAR PostMessage first (works around Electron/Chromium
    treating KEYEVENTF_UNICODE keystrokes as IME composition). Falls back
    to SendInput on failure.
    """
    if not text or _user32 is None:
        return 0

    hwnd = _win32.focused_hwnd()
    if hwnd:
        ok = True
        for ch in text:
            # WM_CHAR delivers one UTF-16 code unit per message. Surrogate
            # pairs need two PostMessage calls.
            utf16 = ch.encode("utf-16-le")
            for i in range(0, len(utf16), 2):
                code = utf16[i] | (utf16[i + 1] << 8)
                if not _user32.PostMessageW(hwnd, WM_CHAR, code, 0):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return len(text)
        log.debug("PostMessage WM_CHAR failed; falling back to SendInput")

    inputs: list[_INPUT] = []
    for ch in text:
        # Every char is emitted as one (or two, for surrogate pairs) UTF-16
        # code units, each as a down+up pair.
        utf16 = ch.encode("utf-16-le")
        for i in range(0, len(utf16), 2):
            scan = utf16[i] | (utf16[i + 1] << 8)
            for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
                ki = _KEYBDINPUT(
                    wVk=0,
                    wScan=scan,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=None,
                )
                u = _INPUT_UNION()
                u.ki = ki
                inputs.append(_INPUT(type=INPUT_KEYBOARD, u=u))

    if not inputs:
        return 0
    sent = _send(inputs)
    if sent != len(inputs):
        log.warning("SendInput sent %s/%s events for text of %s chars", sent, len(inputs), len(text))
    return sent


