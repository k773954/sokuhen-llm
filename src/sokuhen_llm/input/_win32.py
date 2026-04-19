"""Shared Win32 type declarations + input-anchor positioning helpers.

Both hook.py, send_input.py and composition_window.py need to call
GetGUIThreadInfo. If each module defines its own ``_GUITHREADINFO``
struct, the second-imported one silently replaces the first module's
``ctypes.windll.user32.GetGUIThreadInfo.argtypes`` — calls from the
original module then raise ``TypeError: expected LP__GUITHREADINFO
instance instead of pointer to _GUITHREADINFO``. Single source of
truth for the struct and argtypes avoids that.

``input_anchor_position`` is the main positioning entry point: it
walks a ladder of detection strategies (caret → UIAutomation focused
element → focused window rect) and returns the most specific
location it can for the current input field.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import sys


log = logging.getLogger(__name__)


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wt.LONG),
        ("top", wt.LONG),
        ("right", wt.LONG),
        ("bottom", wt.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD),
        ("flags", wt.DWORD),
        ("hwndActive", wt.HWND),
        ("hwndFocus", wt.HWND),
        ("hwndCapture", wt.HWND),
        ("hwndMenuOwner", wt.HWND),
        ("hwndMoveSize", wt.HWND),
        ("hwndCaret", wt.HWND),
        ("rcCaret", RECT),
    ]


if sys.platform == "win32":
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wt.HWND
    user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    user32.GetWindowThreadProcessId.restype = wt.DWORD
    user32.GetGUIThreadInfo.argtypes = [wt.DWORD, ctypes.POINTER(GUITHREADINFO)]
    user32.GetGUIThreadInfo.restype = wt.BOOL
    user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
    user32.GetCursorPos.restype = wt.BOOL
    user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(POINT)]
    user32.ClientToScreen.restype = wt.BOOL
    user32.SetWindowPos.argtypes = [
        wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wt.UINT,
    ]
    user32.SetWindowPos.restype = wt.BOOL
    user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.PostMessageW.restype = wt.BOOL
    user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(RECT)]
    user32.GetWindowRect.restype = wt.BOOL
    user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(RECT)]
    user32.GetClientRect.restype = wt.BOOL
    user32.BringWindowToTop.argtypes = [wt.HWND]
    user32.BringWindowToTop.restype = wt.BOOL
    # Note: wintypes has no LRESULT alias — it's ssize_t-wide on the
    # target arch. c_ssize_t matches on x64 / ARM64 where Windows typedefs
    # LRESULT to LONG_PTR.
    user32.SendMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.SendMessageW.restype = ctypes.c_ssize_t
    user32.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
    user32.AttachThreadInput.restype = wt.BOOL
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentThreadId.restype = wt.DWORD
    # imm32.dll for IME state queries
    try:
        imm32 = ctypes.windll.imm32
        imm32.ImmGetDefaultIMEWnd.argtypes = [wt.HWND]
        imm32.ImmGetDefaultIMEWnd.restype = wt.HWND
        imm32.ImmGetContext.argtypes = [wt.HWND]
        imm32.ImmGetContext.restype = wt.HANDLE
        imm32.ImmGetOpenStatus.argtypes = [wt.HANDLE]
        imm32.ImmGetOpenStatus.restype = wt.BOOL
        imm32.ImmReleaseContext.argtypes = [wt.HWND, wt.HANDLE]
        imm32.ImmReleaseContext.restype = wt.BOOL
    except (OSError, AttributeError):
        imm32 = None  # type: ignore[assignment]
else:  # pragma: no cover — non-Windows smoke
    user32 = None  # type: ignore[assignment]
    imm32 = None  # type: ignore[assignment]
    kernel32 = None  # type: ignore[assignment]


HWND_TOPMOST = -1
HWND_NOTOPMOST = -2
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_NOSENDCHANGING = 0x0400


# --- focused-window helpers -------------------------------------------


def focused_hwnd() -> int:
    """HWND that owns the caret/focus, falling back to the foreground
    window if the app doesn't expose a focused child."""
    if user32 is None:
        return 0
    fg = user32.GetForegroundWindow()
    if not fg:
        return 0
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    tid = user32.GetWindowThreadProcessId(fg, None)
    if tid and user32.GetGUIThreadInfo(tid, ctypes.byref(info)):
        if info.hwndFocus:
            return info.hwndFocus
    return fg


def caret_screen_position() -> tuple[int, int] | None:
    """Best-effort screen coords of the caret. None if the foreground app
    doesn't expose a caret (common in Electron/Chromium)."""
    if user32 is None:
        return None
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    tid = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
    if not tid:
        return None
    if not user32.GetGUIThreadInfo(tid, ctypes.byref(info)):
        return None
    rc = info.rcCaret
    if rc.left == 0 and rc.top == 0 and rc.right == 0 and rc.bottom == 0:
        return None
    pt = POINT(rc.left, rc.bottom)
    if info.hwndCaret:
        user32.ClientToScreen(info.hwndCaret, ctypes.byref(pt))
    return pt.x, pt.y


def cursor_screen_position() -> tuple[int, int]:
    if user32 is None:
        return (100, 100)
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# --- UIAutomation fallback --------------------------------------------
#
# Many modern apps (Electron/Chromium: Claude desktop, Discord, Slack,
# VSCode; UWP/WinRT apps; Qt apps with WebEngine) don't expose a
# Win32 caret at all — rcCaret returns zeros. But they DO participate
# in UI Automation and report BoundingRectangle for their focused
# element. We query it via COM / ctypes.
#
# The machinery is only set up lazily because COM initialization is
# per-thread and we only want to pay the cost when the cheaper caret
# path fails.

_UIA_INITIALIZED = False
_UIA_INSTANCE: int | None = None  # raw IUIAutomation* as int

# GUIDs
_CLSID_CUIAutomation = "{FF48DBA4-60EF-4201-AA87-54103EEF594E}"
_IID_IUIAutomation = "{30CBE57D-D9D0-452A-AB13-7AC5AC4825EE}"
_IID_IUIAutomationElement = "{D22108AA-8AC5-49A5-837B-37BBB3D7591E}"

COINIT_APARTMENTTHREADED = 0x2
CLSCTX_INPROC_SERVER = 0x1
S_OK = 0


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wt.DWORD),
        ("Data2", wt.WORD),
        ("Data3", wt.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _parse_guid(s: str) -> _GUID:
    """Parse a {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX} string into a GUID."""
    s = s.strip("{}")
    parts = s.split("-")
    d1 = int(parts[0], 16)
    d2 = int(parts[1], 16)
    d3 = int(parts[2], 16)
    d4_bytes = bytes.fromhex(parts[3] + parts[4])
    g = _GUID()
    g.Data1 = d1
    g.Data2 = d2
    g.Data3 = d3
    g.Data4 = (ctypes.c_ubyte * 8)(*d4_bytes)
    return g


def _init_uia() -> int | None:
    """One-shot COM + UIAutomation instantiation. Returns IUIAutomation*.

    Cached in ``_UIA_INSTANCE`` — we never release it (the process
    exits with the IME process so leak is harmless).
    """
    global _UIA_INITIALIZED, _UIA_INSTANCE
    if _UIA_INITIALIZED:
        return _UIA_INSTANCE
    _UIA_INITIALIZED = True
    if sys.platform != "win32":
        return None

    try:
        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
        clsid = _parse_guid(_CLSID_CUIAutomation)
        iid = _parse_guid(_IID_IUIAutomation)
        ppv = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid),
            None,
            CLSCTX_INPROC_SERVER,
            ctypes.byref(iid),
            ctypes.byref(ppv),
        )
        if hr != S_OK or not ppv.value:
            log.debug("CoCreateInstance(UIAutomation) failed: hr=0x%x", hr)
            return None
        _UIA_INSTANCE = ppv.value
        return ppv.value
    except Exception as e:  # pragma: no cover — defensive
        log.debug("UIA init threw: %s", e)
        return None


def _slot(obj_ptr: int, index: int, fntype):
    """Fetch a vtable method as a callable. ``obj_ptr`` is the COM object
    pointer (as int), ``index`` is the vtable slot, ``fntype`` is the
    WINFUNCTYPE. Common helper to keep the COM call-sites tidy."""
    vtbl = ctypes.cast(obj_ptr, ctypes.POINTER(ctypes.c_void_p))[0]
    slots = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))
    return fntype(slots[index])


# IUnknown vtable slots
_SLOT_IU_RELEASE = 2

# IUIAutomation vtable slots (IUnknown 0-2 + UIA methods)
_SLOT_UIA_GET_FOCUSED_ELEMENT = 8

# IUIAutomationElement vtable slots
_SLOT_EL_GET_CURRENT_PATTERN = 16   # GetCurrentPattern(patternId, **ppv)
_SLOT_EL_BOUNDING_RECT = 43         # get_CurrentBoundingRectangle(*RECT)

# IUIAutomationTextPattern vtable (IUnknown 0-2 + 6 methods — see
# UIAutomationClient.h). Order: RangeFromPoint(3), RangeFromChild(4),
# GetSelection(5), GetVisibleRanges(6), get_DocumentRange(7),
# get_SupportedTextSelection(8).
_SLOT_TP_GET_SELECTION = 5

# IUIAutomationTextRangeArray: get_Length(3), GetElement(4)
_SLOT_RA_LENGTH = 3
_SLOT_RA_GET_ELEMENT = 4

# IUIAutomationTextRange methods start at slot 3:
# Clone(3), Compare(4), CompareEndpoints(5), ExpandToEnclosingUnit(6),
# FindAttribute(7), FindText(8), GetAttributeValue(9),
# GetBoundingRectangles(10), GetEnclosingElement(11), …
_SLOT_TR_GET_BOUNDING_RECTS = 10


UIA_TextPatternId = 10014


_HRESULT = ctypes.c_long
_HRESULT_PTR_VOIDP = ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
_HRESULT_INT_PTR_VOIDP = ctypes.WINFUNCTYPE(
    _HRESULT, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
)
_HRESULT_PTR_RECT = ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.POINTER(RECT))
_RELEASE_FN = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)


def _release(obj_ptr: int) -> None:
    """Call IUnknown::Release on the given COM pointer."""
    try:
        _slot(obj_ptr, _SLOT_IU_RELEASE, _RELEASE_FN)(obj_ptr)
    except Exception:  # pragma: no cover
        pass


def _uia_focused_element() -> int | None:
    """Return raw IUIAutomationElement* of the focused control, or None.
    The caller is responsible for Release."""
    uia = _init_uia()
    if uia is None:
        return None
    try:
        GetFocusedElement = _slot(
            uia, _SLOT_UIA_GET_FOCUSED_ELEMENT, _HRESULT_PTR_VOIDP
        )
        element = ctypes.c_void_p()
        hr = GetFocusedElement(uia, ctypes.byref(element))
        if hr != S_OK or not element.value:
            return None
        return element.value
    except Exception as e:  # pragma: no cover
        log.debug("GetFocusedElement threw: %s", e)
        return None


def uia_focus_rect() -> tuple[int, int, int, int] | None:
    """BoundingRectangle of the UIAutomation-focused element. Fallback
    used when TextPattern doesn't give us a caret rect."""
    element = _uia_focused_element()
    if element is None:
        return None
    try:
        GetBoundingRect = _slot(element, _SLOT_EL_BOUNDING_RECT, _HRESULT_PTR_RECT)
        rc = RECT()
        hr = GetBoundingRect(element, ctypes.byref(rc))
        if hr != S_OK or rc.right - rc.left < 1 or rc.bottom - rc.top < 1:
            return None
        return rc.left, rc.top, rc.right, rc.bottom
    except Exception as e:  # pragma: no cover
        log.debug("BoundingRectangle threw: %s", e)
        return None
    finally:
        _release(element)


def uia_caret_rect() -> tuple[int, int, int, int] | None:
    """Caret (text-selection) rectangle via UIAutomation TextPattern.

    This is the most accurate "caret location" we can get in
    Electron/Chromium/UWP apps that don't expose a classic Win32 caret.
    Returns the first selection-range bounding rect, which for a
    collapsed cursor is a thin vertical rect at the caret position.
    Returns None if the focused element doesn't support TextPattern
    or has no selection.
    """
    element = _uia_focused_element()
    if element is None:
        return None
    tp_ptr = 0
    ra_ptr = 0
    tr_ptr = 0
    try:
        GetCurrentPattern = _slot(
            element, _SLOT_EL_GET_CURRENT_PATTERN, _HRESULT_INT_PTR_VOIDP
        )
        pattern = ctypes.c_void_p()
        hr = GetCurrentPattern(element, UIA_TextPatternId, ctypes.byref(pattern))
        if hr != S_OK or not pattern.value:
            return None
        tp_ptr = pattern.value

        GetSelection = _slot(tp_ptr, _SLOT_TP_GET_SELECTION, _HRESULT_PTR_VOIDP)
        array = ctypes.c_void_p()
        hr = GetSelection(tp_ptr, ctypes.byref(array))
        if hr != S_OK or not array.value:
            return None
        ra_ptr = array.value

        # Length (IUIAutomationTextRangeArray::get_Length)
        GetLength = _slot(
            ra_ptr, _SLOT_RA_LENGTH,
            ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)),
        )
        length = ctypes.c_int(0)
        hr = GetLength(ra_ptr, ctypes.byref(length))
        if hr != S_OK or length.value < 1:
            return None

        GetElement = _slot(ra_ptr, _SLOT_RA_GET_ELEMENT, _HRESULT_INT_PTR_VOIDP)
        rng = ctypes.c_void_p()
        hr = GetElement(ra_ptr, 0, ctypes.byref(rng))
        if hr != S_OK or not rng.value:
            return None
        tr_ptr = rng.value

        # GetBoundingRectangles returns SAFEARRAY of doubles (4 per rect).
        # We access it by querying the safearray metadata via OleAut32 APIs.
        GetRects = _slot(
            tr_ptr, _SLOT_TR_GET_BOUNDING_RECTS,
            ctypes.WINFUNCTYPE(_HRESULT, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
        )
        sa_ptr = ctypes.c_void_p()
        hr = GetRects(tr_ptr, ctypes.byref(sa_ptr))
        if hr != S_OK or not sa_ptr.value:
            return None

        try:
            return _safearray_first_rect(sa_ptr.value)
        finally:
            ctypes.windll.oleaut32.SafeArrayDestroy(sa_ptr)
    except Exception as e:  # pragma: no cover
        log.debug("TextPattern caret-rect threw: %s", e)
        return None
    finally:
        if tr_ptr:
            _release(tr_ptr)
        if ra_ptr:
            _release(ra_ptr)
        if tp_ptr:
            _release(tp_ptr)
        _release(element)


def _safearray_first_rect(sa_ptr: int) -> tuple[int, int, int, int] | None:
    """Pull the first RECT out of a SAFEARRAY<double> returned by
    GetBoundingRectangles. The SAFEARRAY holds [x, y, w, h, x, y, w, h, …]
    in screen coordinates."""
    oleaut = ctypes.windll.oleaut32
    oleaut.SafeArrayGetDim.argtypes = [ctypes.c_void_p]
    oleaut.SafeArrayGetDim.restype = ctypes.c_uint
    oleaut.SafeArrayGetUBound.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_long),
    ]
    oleaut.SafeArrayGetUBound.restype = _HRESULT
    oleaut.SafeArrayGetLBound.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_long),
    ]
    oleaut.SafeArrayGetLBound.restype = _HRESULT
    oleaut.SafeArrayAccessData.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ]
    oleaut.SafeArrayAccessData.restype = _HRESULT
    oleaut.SafeArrayUnaccessData.argtypes = [ctypes.c_void_p]
    oleaut.SafeArrayUnaccessData.restype = _HRESULT

    if oleaut.SafeArrayGetDim(sa_ptr) < 1:
        return None
    lb = ctypes.c_long(0)
    ub = ctypes.c_long(0)
    if oleaut.SafeArrayGetLBound(sa_ptr, 1, ctypes.byref(lb)) != 0:
        return None
    if oleaut.SafeArrayGetUBound(sa_ptr, 1, ctypes.byref(ub)) != 0:
        return None
    n = ub.value - lb.value + 1
    if n < 4:
        return None
    data = ctypes.c_void_p()
    if oleaut.SafeArrayAccessData(sa_ptr, ctypes.byref(data)) != 0:
        return None
    try:
        values = ctypes.cast(data, ctypes.POINTER(ctypes.c_double))
        x = values[0]
        y = values[1]
        w = values[2]
        h = values[3]
    finally:
        oleaut.SafeArrayUnaccessData(sa_ptr)
    if w < 0.5 and h < 0.5:
        return None
    return int(x), int(y), int(x + w), int(y + h)


# --- main positioning entry point -------------------------------------


def input_anchor_position() -> tuple[int, int]:
    """Best position to anchor the composition window at: as close as we
    can get to the actual caret (text cursor) the user is editing.

    Ladder (top first; each returns if it has a plausible answer):
      1. GetGUIThreadInfo.rcCaret           — classic Win32 caret (Notepad,
                                               Edit, RichEdit, many IDEs).
                                               Returns the caret bottom-left.
      2. UIAutomation TextPattern selection — real caret position in
                                               Electron/Chromium (Claude,
                                               Discord, Slack, VSCode),
                                               UWP, Qt WebEngine. This
                                               is a thin rect at the
                                               insertion point itself.
      3. UIAutomation focused-element rect  — last resort for apps that
                                               expose UIA but no text
                                               pattern; use the left of
                                               the field's vertical
                                               middle so we're near the
                                               text baseline.
      4. Focused-window client area         — we couldn't identify the
                                               element; aim just inside
                                               the foreground window.
      5. Cursor position                    — never preferred; only if
                                               none of the above worked.

    Returns the (x, y) of the caret bottom so the composition window can
    hang just below the current text baseline.
    """
    # 1. Classic Win32 caret
    caret = caret_screen_position()
    if caret is not None:
        return caret

    # 2. UIA TextPattern — actual caret rect
    text_rect = uia_caret_rect()
    if text_rect is not None:
        left, _top, _right, bottom = text_rect
        return left, bottom

    # 3. UIA focused-element rect — approximate caret at first-line
    #    baseline instead of the element's bottom. For most text inputs
    #    (single-line, or multi-line where the user is editing near the
    #    top), the caret is on the first visible line; anchoring at
    #    (left, top + ~line_height) lands much closer to the caret than
    #    (left, bottom) did.
    rect = uia_focus_rect()
    if rect is not None:
        left, top, _right, bottom = rect
        height = bottom - top
        LINE_HEIGHT = 24  # rough text line-height estimate
        # Short fields (single-line chrome/edit): caret ≈ bottom anyway.
        # Taller fields: use top + first-line height. Still beats bottom
        # because the user's caret is almost never at the absolute bottom
        # of a multi-line textarea while typing.
        if height <= LINE_HEIGHT + 8:
            y = bottom
        else:
            y = top + LINE_HEIGHT
        return left + 2, y

    # 4. Focused/foreground window, nudge inside from the bottom-left
    if user32 is not None:
        hwnd = focused_hwnd()
        if hwnd:
            rc = RECT()
            if user32.GetClientRect(hwnd, ctypes.byref(rc)):
                pt = POINT(rc.left + 20, rc.bottom - 40)
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
                return pt.x, pt.y
            if user32.GetWindowRect(hwnd, ctypes.byref(rc)):
                return rc.left + 40, rc.bottom - 60

    # 5. Absolute fallback — cursor
    return cursor_screen_position()


def debug_anchor_source() -> str:
    """Return a human-readable description of which ladder rung fired.
    Not used in production; handy for ``python -c`` diagnostics when a
    user reports the window is in the wrong place."""
    if caret_screen_position() is not None:
        return f"classic caret: {caret_screen_position()}"
    tr = uia_caret_rect()
    if tr is not None:
        return f"UIA TextPattern: rect={tr}"
    fr = uia_focus_rect()
    if fr is not None:
        return f"UIA focused element: rect={fr}"
    if user32 is not None:
        hwnd = focused_hwnd()
        if hwnd:
            return f"focused-hwnd fallback: hwnd=0x{hwnd:x}"
    return f"cursor fallback: {cursor_screen_position()}"


def set_topmost(hwnd: int) -> None:
    """Force the given HWND to the top of the z-order without stealing
    focus. Call repeatedly — some apps (fullscreen games, Electron
    windows with their own always-on-top) race to the top and we have
    to keep reasserting."""
    if user32 is None:
        return
    # First: NOT topmost toggle removes any stale "not topmost" state,
    # then TOPMOST puts us firmly on top. This is more reliable than
    # a single TOPMOST call against apps that fight for the position.
    flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW | SWP_NOSENDCHANGING
    user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
    user32.BringWindowToTop(hwnd)


# --- OS IME state -----------------------------------------------------
#
# The built-in Windows / Microsoft IME has a global on/off toggle (半角/
# 全角 or Alt+`). When it's off, the user has explicitly switched to
# direct ASCII input. sokuhen should defer to that intent — if the OS
# IME is off, we suppress our own conversion.
#
# Detecting this cross-thread is fiddly. Two complementary strategies:
#
#  1. AttachThreadInput(my_tid, foreground_tid, TRUE) + ImmGetContext +
#     ImmGetOpenStatus. Most accurate when the target thread exposes an
#     IME context. Doesn't work if the target app doesn't use IMM32
#     (e.g. TSF-only apps, some Electron frames).
#
#  2. SendMessage(ImmGetDefaultIMEWnd(hwnd), WM_IME_CONTROL,
#     IMC_GETOPENSTATUS). This routes through the IME's own default
#     window message pump. Works for apps without a direct IMM32
#     context.
#
# We try (1) first; fall back to (2); safe-default to True on failure
# (so sokuhen stays functional even when we can't read state — the
# user can still use Alt+` manually).

_WM_IME_CONTROL = 0x0283
_IMC_GETOPENSTATUS = 0x0005


def os_ime_is_open() -> bool:
    """Return True if the foreground app's native Windows IME is in a
    "Japanese input" state (i.e. NOT in 英数字 / direct-ASCII mode).

    Returns True on non-Windows or if the query fails — the safe
    default is "let sokuhen run normally".
    """
    state, _source = _os_ime_state_with_source()
    return state


def _os_ime_state_with_source() -> tuple[bool, str]:
    """Internal variant that returns the state plus the detection
    strategy used.

    IMPORTANT: this must be side-effect-free. An earlier version used
    ``AttachThreadInput(my_tid, foreground_tid, TRUE)`` + ImmGetContext
    + ImmGetOpenStatus, which is the "officially recommended" pattern
    for querying IME state. In practice, calling AttachThreadInput at
    our 250 ms polling cadence caused the Microsoft IME on Windows 10/
    11 to toggle between 日本語 and 英数字 on its own — the attach/
    detach races with IME initialization, and the IME sometimes treats
    the short window of shared input state as a mode-switch trigger.
    So we only use the passive ``SendMessage(WM_IME_CONTROL)`` path
    here. Its return value isn't always reliable (TSF-based IMEs may
    report stale status), which is why app.py ALSO observes the 半角/
    全角 and 英数 keys directly in the keyboard hook — that covers the
    common case at zero lag, while this poll only plays catch-up for
    Language-Bar / PowerToys-style toggles that don't go through a
    physical key.
    """
    if user32 is None or imm32 is None:
        return True, "no-win32"
    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return True, "no-fg-hwnd"
        ime_wnd = imm32.ImmGetDefaultIMEWnd(hwnd)
        if not ime_wnd:
            return True, "no-ime-wnd"
        result = user32.SendMessageW(
            ime_wnd, _WM_IME_CONTROL, _IMC_GETOPENSTATUS, 0
        )
        return bool(result), "wm-ime-control"
    except Exception as e:
        log.debug("os_ime_is_open failed: %r", e)
        return True, "exception"
