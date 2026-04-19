"""Global keyboard hook (Windows WH_KEYBOARD_LL).

Installs a low-level keyboard hook that can inspect and optionally suppress
key events before they reach the foreground window. The hook callback must
run on a thread that pumps the Windows message loop, so we install from the
main (Qt) thread and rely on Qt's event loop for message dispatch.

The consumer registers a callback that returns True to suppress the event
(prevent it from reaching the underlying app) or False to let it through.
Typical use:

    hook = KeyboardHook()
    hook.on_event = lambda ev: ime.handle(ev)  # returns bool
    hook.install()
    # ... runs in message loop ...
    hook.uninstall()
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import sys
from dataclasses import dataclass
from enum import IntFlag
from typing import Callable, Optional


log = logging.getLogger(__name__)


# --- Win32 constants ---------------------------------------------------

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
_KEYDOWN_MSGS = {WM_KEYDOWN, WM_SYSKEYDOWN}
_KEYUP_MSGS = {WM_KEYUP, WM_SYSKEYUP}

# LLKHF flags
LLKHF_EXTENDED = 0x01
LLKHF_INJECTED = 0x10
LLKHF_ALTDOWN = 0x20
LLKHF_UP = 0x80


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wt.DWORD),
        ("scanCode", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wt.ULONG)),
    ]


_LowLevelKeyboardProc = ctypes.WINFUNCTYPE(wt.LPARAM, ctypes.c_int, wt.WPARAM, wt.LPARAM)


# --- Win32 function signatures -----------------------------------------
# ctypes defaults to `int` for return values and argument types, which
# silently truncates 64-bit handles on x64 Python. We must declare every
# call we make. Missing declarations show up as weird "failed" returns.


def _configure_win32() -> None:
    if sys.platform != "win32":
        return
    u = ctypes.windll.user32
    k = ctypes.windll.kernel32

    u.SetWindowsHookExW.argtypes = [
        ctypes.c_int,           # idHook
        _LowLevelKeyboardProc,  # lpfn
        wt.HINSTANCE,           # hmod
        wt.DWORD,               # dwThreadId
    ]
    u.SetWindowsHookExW.restype = wt.HANDLE  # HHOOK is a HANDLE

    u.UnhookWindowsHookEx.argtypes = [wt.HANDLE]
    u.UnhookWindowsHookEx.restype = wt.BOOL

    u.CallNextHookEx.argtypes = [wt.HANDLE, ctypes.c_int, wt.WPARAM, wt.LPARAM]
    u.CallNextHookEx.restype = wt.LPARAM

    k.GetModuleHandleW.argtypes = [wt.LPCWSTR]
    k.GetModuleHandleW.restype = wt.HMODULE

    u.GetKeyboardLayout.argtypes = [wt.DWORD]
    u.GetKeyboardLayout.restype = wt.HKL

    u.GetKeyboardState.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]
    u.GetKeyboardState.restype = wt.BOOL

    u.GetAsyncKeyState.argtypes = [ctypes.c_int]
    u.GetAsyncKeyState.restype = ctypes.c_short

    u.ToUnicodeEx.argtypes = [
        wt.UINT, wt.UINT,
        ctypes.POINTER(ctypes.c_ubyte),
        wt.LPWSTR, ctypes.c_int, wt.UINT, wt.HKL,
    ]
    u.ToUnicodeEx.restype = ctypes.c_int


_configure_win32()


class Modifiers(IntFlag):
    NONE = 0
    SHIFT = 1 << 0
    CTRL = 1 << 1
    ALT = 1 << 2
    WIN = 1 << 3


@dataclass(frozen=True)
class HookEvent:
    """A parsed keyboard event from the low-level hook."""

    vk: int  # virtual key code
    scan: int  # hardware scan code
    pressed: bool  # True on key-down, False on key-up
    modifiers: Modifiers
    char: str  # the mapped unicode char, or "" if not printable


# Virtual keys we care about. Values from WinUser.h.
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_PAUSE = 0x13
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F1 = 0x70
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4
VK_RMENU = 0xA5
VK_OEM_3 = 0xC0  # `~ key on US layout - used for IME toggle (Alt+`)

# --- IME mode-change virtual keys ------------------------------------
# Various VK codes that Microsoft IME (and keyboards on Japanese
# layouts) emit when the user toggles between Japanese and 英数字
# input. Watching for these in our low-level keyboard hook lets
# sokuhen stay in lock-step with the OS IME without having to poll
# IMM32 (which has side effects against TSF-based IMEs).
#
# We group them by the direction they nudge state:
#   TOGGLE  -- flip current state (unknown after the flip until next event)
#   OPEN    -- force open (Japanese mode)
#   CLOSE   -- force closed (英数字 mode)

# Toggle keys (one press = one flip)
VK_KANJI = 0x19                 # 半角/全角 (JIS); also VK_HANJA on Korean

# Explicit "IME ON" / "IME OFF" VKs introduced in Windows 10+. MS IME
# default shortcuts (e.g. Ctrl+Shift for 英数/ひらがな cycle) synthesize
# these via the TSF text service.
VK_IME_ON = 0x16                # Force IME open
VK_IME_OFF = 0x1A               # Force IME closed

# "Input mode" keys -- MS IME interprets these as specific mode changes.
# Default mapping in MS IME config:
#   - 無変換 (NONCONVERT) cycles through / switches to 英数. Treat as CLOSE.
#   - 変換 (CONVERT) initiates kanji conversion; not really a mode key,
#     but some configs use it to re-enable Japanese. Leave alone.
VK_NONCONVERT = 0x1D            # 無変換

# DBE (Double-Byte Encoding) mode keys -- emitted by the
# alphanumeric / hiragana / katakana keys on JIS layouts and by TSF.
VK_DBE_ALPHANUMERIC = 0xF0      # Force alphanumeric (closed)
VK_DBE_KATAKANA = 0xF1          # Force katakana (open)
VK_DBE_HIRAGANA = 0xF2          # Force hiragana (open)
VK_DBE_SBCSCHAR = 0xF3          # Half-width (closed)
VK_DBE_DBCSCHAR = 0xF4          # Full-width / toggle kana (open)
VK_DBE_ROMAN = 0xF5             # Roman mode (closed)
VK_DBE_NOROMAN = 0xF6           # Non-roman mode (open)


# --- character mapping -------------------------------------------------

_user32 = ctypes.windll.user32 if sys.platform == "win32" else None


def _get_key_state(vk: int) -> int:
    if _user32 is None:
        return 0
    return _user32.GetAsyncKeyState(vk) & 0x8000


def _current_modifiers() -> Modifiers:
    mods = Modifiers.NONE
    if _get_key_state(VK_SHIFT):
        mods |= Modifiers.SHIFT
    if _get_key_state(VK_CONTROL):
        mods |= Modifiers.CTRL
    if _get_key_state(VK_MENU):
        mods |= Modifiers.ALT
    if _get_key_state(VK_LWIN) or _get_key_state(VK_RWIN):
        mods |= Modifiers.WIN
    return mods


def _to_unicode(vk: int, scan: int, mods: Modifiers) -> str:
    """Map a VK+scan+modifier combo to a unicode character via ToUnicodeEx.

    Respects the current keyboard layout, so Dvorak / JIS users still see
    their expected characters. Returns "" for non-printable or dead keys.
    """
    if _user32 is None:
        return ""

    # GetKeyboardState wants the full 256-byte state vector.
    state = (ctypes.c_ubyte * 256)()
    if not _user32.GetKeyboardState(state):
        return ""
    # Build the state vector from our parsed modifiers so we're not racing
    # with the foreground window's state.
    state[VK_SHIFT] = 0x80 if Modifiers.SHIFT in mods else 0
    state[VK_CONTROL] = 0x80 if Modifiers.CTRL in mods else 0
    state[VK_MENU] = 0x80 if Modifiers.ALT in mods else 0
    state[VK_CAPITAL] = state[VK_CAPITAL] & 0x01  # preserve toggle

    buf = ctypes.create_unicode_buffer(8)
    hkl = _user32.GetKeyboardLayout(0)
    # Use flag 0x4 (KLF_NOMENU) so Alt+key doesn't produce a char.
    n = _user32.ToUnicodeEx(vk, scan, state, buf, len(buf), 0, hkl)
    if n <= 0:
        return ""
    # When Ctrl is held, ToUnicodeEx returns control codes; we don't want those.
    if Modifiers.CTRL in mods:
        return ""
    return buf[:n]


# --- hook installation -------------------------------------------------


class KeyboardHook:
    """Low-level keyboard hook.

    Callers install this, set ``on_event``, and run a Windows message pump
    (Qt's event loop counts). The callback returns True to suppress the key
    from reaching the foreground app.
    """

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("KeyboardHook currently supports Windows only")
        self.on_event: Optional[Callable[[HookEvent], bool]] = None
        self._hook_id: int = 0
        self._callback = _LowLevelKeyboardProc(self._hook_proc)
        self._kernel32 = ctypes.windll.kernel32
        self._user32 = ctypes.windll.user32

    def install(self) -> None:
        if self._hook_id:
            return
        mod_handle = self._kernel32.GetModuleHandleW(None)
        self._hook_id = self._user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._callback, mod_handle, 0
        )
        if not self._hook_id:
            err = ctypes.GetLastError()
            raise OSError(
                f"SetWindowsHookExW failed (GetLastError={err}). "
                f"mod_handle={mod_handle!r}"
            )
        log.info("Installed low-level keyboard hook (id=0x%x)", self._hook_id)

    def uninstall(self) -> None:
        if self._hook_id:
            self._user32.UnhookWindowsHookEx(self._hook_id)
            self._hook_id = 0

    def _hook_proc(self, n_code: int, w_param: int, l_param: int) -> int:
        if n_code < 0 or self.on_event is None:
            return self._user32.CallNextHookEx(self._hook_id, n_code, w_param, l_param)
        try:
            kb = ctypes.cast(l_param, ctypes.POINTER(_KBDLLHOOKSTRUCT))[0]
            pressed = w_param in _KEYDOWN_MSGS
            # Skip synthetic events (we inject our own chars via SendInput; we
            # must not re-enter our pipeline on those).
            if kb.flags & LLKHF_INJECTED:
                return self._user32.CallNextHookEx(self._hook_id, n_code, w_param, l_param)

            mods = _current_modifiers()
            char = _to_unicode(kb.vkCode, kb.scanCode, mods) if pressed else ""
            event = HookEvent(
                vk=kb.vkCode,
                scan=kb.scanCode,
                pressed=pressed,
                modifiers=mods,
                char=char,
            )
            suppressed = bool(self.on_event(event))
            if suppressed:
                return 1  # non-zero = swallow
        except Exception:  # pragma: no cover — defensive
            log.exception("hook callback raised; passing event through")
        return self._user32.CallNextHookEx(self._hook_id, n_code, w_param, l_param)
