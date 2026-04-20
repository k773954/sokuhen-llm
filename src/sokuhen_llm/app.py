"""Main application: wires the engine, the keyboard hook, and the UI.

Lifecycle:
  1. Load dictionaries + learning data (blocking; ~1s for SKK-JISYO.L).
  2. Install global low-level keyboard hook.
  3. Start Qt event loop; the hook dispatches into ``handle_event`` which
     updates the composer and UI.
  4. On quit, save learning data.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from .engine import Converter, Dictionary, LearningStore, LiveComposer
from .input import HookEvent, KeyboardHook, Modifiers, send_unicode_text
from .input.hook import (
    VK_BACK,
    VK_DBE_ALPHANUMERIC,
    VK_DBE_DBCSCHAR,
    VK_DBE_HIRAGANA,
    VK_DBE_KATAKANA,
    VK_DBE_NOROMAN,
    VK_DBE_ROMAN,
    VK_DBE_SBCSCHAR,
    VK_DOWN,
    VK_ESCAPE,
    VK_F7,
    VK_F8,
    VK_F9,
    VK_F10,
    VK_IME_OFF,
    VK_IME_ON,
    VK_KANJI,
    VK_LEFT,
    VK_NONCONVERT,
    VK_OEM_3,
    VK_RETURN,
    VK_RIGHT,
    VK_SPACE,
    VK_TAB,
    VK_UP,
)

# Classify IME-mode VKs by the state transition they imply.
_IME_VK_TOGGLE: frozenset[int] = frozenset({VK_KANJI})
_IME_VK_FORCE_OPEN: frozenset[int] = frozenset({
    VK_IME_ON, VK_DBE_HIRAGANA, VK_DBE_KATAKANA,
    VK_DBE_DBCSCHAR, VK_DBE_NOROMAN,
})
_IME_VK_FORCE_CLOSE: frozenset[int] = frozenset({
    VK_IME_OFF, VK_DBE_ALPHANUMERIC, VK_DBE_SBCSCHAR,
    VK_DBE_ROMAN, VK_NONCONVERT,
})
from .paths import data_dir, learning_file, log_dir, pid_file
from .ui import CompositionWindow, StatusIndicator


log = logging.getLogger("sokuhen-llm")


class ImeCore(QObject):
    """Non-UI orchestrator: holds the composer and brokers key events.

    Signals are emitted when the UI should redraw. We emit on the Qt thread
    (via Qt's queued connections) because the keyboard hook fires on the
    thread that owns the message loop — which happens to be the Qt thread,
    so direct calls are fine. Still, using signals keeps the code symmetric
    in case we later move the hook to a dedicated thread.
    """

    state_changed = pyqtSignal()
    active_changed = pyqtSignal(bool)
    # Emitted once per commit when the LLM rescorer was ready and
    # participated in picking the surface. The UI connects this to
    # StatusIndicator.flash_rescored() for a brief visual pulse.
    llm_rescored = pyqtSignal()
    # Emitted whenever the LLM loader's status transitions
    # ("loading" -> "ready" / "failed" / etc). Carries the new status
    # string so the UI can update the chip.
    llm_status_changed = pyqtSignal(str)

    def __init__(self, composer: LiveComposer, learning: LearningStore) -> None:
        super().__init__()
        self.composer = composer
        self.learning = learning
        # Debounce learning-file writes — flushing on every Enter press
        # rewrites a multi-KB JSON file synchronously, which is wasteful
        # and also amplifies disk wear.
        self._save_timer = QTimer()
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(2_000)  # 2s of idle → flush
        self._save_timer.timeout.connect(self.learning.save)

        # The user may have sokuhen-llm globally enabled but explicitly
        # switched the OS IME to 英数字 mode via 半角/全角. Respect that
        # intent: when the OS IME is off, sokuhen-llm also goes silent.
        #
        # Detection is purely PASSIVE: we watch the keyboard hook for
        # 半角/全角 (VK_KANJI), 英数 (VK_DBE_SBCSCHAR), and カタカナ/
        # ひらがな (VK_DBE_DBCSCHAR) keydowns and flip our cached
        # ``_os_ime_open`` accordingly. See _note_os_ime_toggle_key.
        #
        # We used to also actively poll IMM32 (ImmGetOpenStatus /
        # WM_IME_CONTROL). That turned out to have side effects:
        # queries from another thread confused Microsoft IME into
        # toggling its own state, producing an infinite 英数字 ⇄
        # 日本語 flip-flop. Polling is disabled; the passive hook
        # alone is reliable enough — the uncommon case of toggling
        # via Language Bar / PowerToys is still recoverable via
        # Alt+` to manually re-sync.
        self._os_ime_open = True  # flipped by _note_os_ime_toggle_key

    @property
    def effective_active(self) -> bool:
        """True only when BOTH the user's global toggle is on AND the
        OS IME is in Japanese mode. This is the state the UI displays
        and that key handling checks."""
        return self.composer.state.active and self._os_ime_open

    # --- IME toggle ----------------------------------------------------

    def toggle_active(self) -> None:
        self.composer.set_active(not self.composer.state.active)
        new_state = self.composer.state.active
        log.info("IME %s", "ON" if new_state else "OFF")
        self.active_changed.emit(self.effective_active)
        self.state_changed.emit()

    # --- event entry ---------------------------------------------------

    def _note_os_ime_toggle_key(self, ev: HookEvent) -> None:
        """Observe IME-mode toggle keys passing through the keyboard hook
        and update our cached OS IME state immediately.

        Watched VK groups (see _IME_VK_* sets above):
          - TOGGLE       -- 半角/全角 (VK_KANJI). Flips current state.
          - FORCE_OPEN   -- IME_ON, DBE_HIRAGANA, DBE_KATAKANA,
                            DBE_DBCSCHAR, DBE_NOROMAN.
          - FORCE_CLOSE  -- IME_OFF, DBE_ALPHANUMERIC, DBE_SBCSCHAR,
                            DBE_ROMAN, NONCONVERT.

        These cover the common MS IME "IME OFF" shortcuts -- the
        dedicated 英数 key on JIS keyboards, the 無変換 key, Windows-10+
        VK_IME_OFF synthetic key, etc. The key is NOT suppressed: we let
        it reach the foreground app so the OS IME still switches.
        """
        if not ev.pressed:
            return
        new_state: Optional[bool]
        if ev.vk in _IME_VK_TOGGLE:
            new_state = not self._os_ime_open
        elif ev.vk in _IME_VK_FORCE_OPEN:
            new_state = True
        elif ev.vk in _IME_VK_FORCE_CLOSE:
            new_state = False
        else:
            return
        if new_state == self._os_ime_open:
            return
        self._os_ime_open = new_state
        log.info(
            "OS IME %s (hook-observed: vk=0x%02x)",
            "opened" if new_state else "closed (英数字 mode)",
            ev.vk,
        )
        if not new_state and not self.composer.state.is_empty:
            self.composer.cancel()
        self.active_changed.emit(self.effective_active)
        self.state_changed.emit()

    def handle_event(self, ev: HookEvent) -> bool:
        """Decide what to do with a hook event.

        Returns True to suppress (prevent propagation to the foreground app).
        """
        # Observe IME-mode toggle keys (pass them through afterwards).
        self._note_os_ime_toggle_key(ev)

        # IME toggle: Alt + ` (grave/tilde key). Works whether or not active.
        if (
            ev.pressed
            and ev.vk == VK_OEM_3
            and Modifiers.ALT in ev.modifiers
            and Modifiers.CTRL not in ev.modifiers
        ):
            self.toggle_active()
            return True

        # Hands off when IME is inactive OR the OS IME is in 英数字
        # mode. The latter lets the user toggle native 半角/全角 and
        # have sokuhen-llm defer automatically — no need to also press
        # Alt+` to silence us.
        if not self.effective_active:
            return False

        # Pass through Ctrl/Win combos — the user is doing an app shortcut.
        if (
            Modifiers.CTRL in ev.modifiers
            or Modifiers.WIN in ev.modifiers
            or (Modifiers.ALT in ev.modifiers and ev.vk != VK_OEM_3)
        ):
            return False

        if not ev.pressed:
            # IME is edge-triggered on key-down. Key-up passes through silently,
            # which prevents the OS seeing orphan key-ups for suppressed keys.
            return False

        return self._handle_keydown(ev)

    # --- keydown dispatch ---------------------------------------------

    def _handle_keydown(self, ev: HookEvent) -> bool:
        vk = ev.vk

        if vk == VK_RETURN:
            return self._commit()

        if vk == VK_ESCAPE:
            if self.composer.state.is_empty:
                return False  # nothing to cancel; let Esc reach the app
            # Two-step Esc: first press closes the candidate panel (if it's
            # open), keeping the composition. Second press cancels entirely.
            if self.composer.state.show_candidates:
                self.composer.hide_candidates()
            else:
                self.composer.cancel()
            self.state_changed.emit()
            return True

        if vk == VK_BACK:
            consumed = self.composer.backspace()
            if consumed:
                self.state_changed.emit()
                return True
            return False

        if vk == VK_SPACE:
            if self.composer.state.is_empty:
                # No composition — pass Space through so it types a space.
                return False
            # Cycle to next candidate. On the first Space after starting a
            # composition, this also "reveals" the candidate list in the UI.
            self.composer.next_candidate(+1 if Modifiers.SHIFT not in ev.modifiers else -1)
            self.state_changed.emit()
            return True

        if vk == VK_TAB:
            # Tab commits (implicit confirm).
            if not self.composer.state.is_empty:
                return self._commit()
            return False

        if vk == VK_LEFT:
            if self.composer.state.is_empty:
                return False
            # Shift+Left/Right mirrors Microsoft IME: resizes the active
            # segment's right boundary instead of moving the segment cursor.
            if Modifiers.SHIFT in ev.modifiers:
                self.composer.resize_segment(-1)
            else:
                self.composer.select_segment(-1)
            self.state_changed.emit()
            return True

        if vk == VK_RIGHT:
            if self.composer.state.is_empty:
                return False
            if Modifiers.SHIFT in ev.modifiers:
                self.composer.resize_segment(+1)
            else:
                self.composer.select_segment(+1)
            self.state_changed.emit()
            return True

        # Up/Down navigate within the candidate list for the active segment.
        # Mirror the Space / Shift+Space cycling but without the "reveal on
        # first press" confusion — arrow keys are already unambiguous.
        if vk == VK_UP:
            if self.composer.state.is_empty:
                return False
            self.composer.next_candidate(-1)
            self.state_changed.emit()
            return True
        if vk == VK_DOWN:
            if self.composer.state.is_empty:
                return False
            self.composer.next_candidate(+1)
            self.state_changed.emit()
            return True

        if vk in (VK_F7, VK_F8, VK_F9, VK_F10):
            kind = {
                VK_F7: "katakana",
                VK_F8: "hankaku_kana",
                VK_F9: "zenkaku_ascii",
                VK_F10: "hankaku_ascii",
            }[vk]
            self.composer.force_conversion(kind)
            self.state_changed.emit()
            return True

        # Printable character — feed to composer (lowercase a-z, digits,
        # shifted punctuation, etc.). ev.char comes from ToUnicodeEx so it
        # respects the user's keyboard layout.
        if ev.char and ev.char.isprintable() and all(ord(c) < 0x80 for c in ev.char):
            # Only forward ASCII-printable chars; non-ASCII typed directly
            # via the layout (unlikely on a US/JIS layout when IME is on)
            # passes through unchanged.
            for ch in ev.char:
                self.composer.input_char(ch)
            self.state_changed.emit()
            return True

        return False

    # --- commit --------------------------------------------------------

    def _commit(self) -> bool:
        """Commit the in-flight composition synchronously.

        The big reliability footgun here is Windows'
        ``LowLevelHooksTimeout`` (default 300 ms): if our hook
        callback exceeds it even once, the OS silently unhooks us and
        every subsequent keystroke passes straight to the foreground
        app. The user observes "Enter stopped working". To stay under
        the budget:

          * LLM weights are loaded on a *background* thread (see
            ``_AsyncRescorer``). Until they finish, ``rescore()``
            returns in microseconds, so the very first commit is
            fast.
          * Once loaded, a full rescoring pass takes ~150-250 ms on
            CPU. That's within the 300 ms budget on typical hardware.

        Deferring commit via QTimer.singleShot would sidestep the
        budget entirely, but then a fast Enter+next-character
        sequence races (the next char arrives before the deferred
        commit runs, and gets appended to the still-live composition
        instead of starting a new one). Staying synchronous is
        simpler and correct.
        """
        if self.composer.state.is_empty:
            return False
        text = self.composer.commit()
        llm_fired = (
            getattr(self.composer, "rescorer", None) is not None
            and getattr(self.composer.rescorer, "status", None) == "ready"
        )
        if text:
            # Inject into the foreground app. The hook marks our
            # SendInput events as INJECTED so we don't re-process them.
            injected = send_unicode_text(text)
            if injected == 0:
                log.warning(
                    "Injection of %r produced 0 events. No foreground window "
                    "accepted the text (try clicking the target text field).",
                    text,
                )
        if llm_fired:
            self.llm_rescored.emit()
        self._save_timer.start()
        self.state_changed.emit()
        return True


# --- app bootstrap ------------------------------------------------------


def _find_dictionary_files() -> list[Path]:
    """All SKK-JISYO.* files the user has downloaded, in priority order.

    SKK-JISYO.L is the baseline — load it first so its cost assignments
    win on tie. Everything else (jinmei, geo, user-added mozc-ut dumps)
    is merged on top.
    """
    root = data_dir()
    priority = ["SKK-JISYO.L"]
    files: list[Path] = []
    seen: set[str] = set()
    for name in priority:
        p = root / name
        if p.exists():
            files.append(p)
            seen.add(p.name)
    # Anything else that looks like an SKK dict file. Sorted for stable
    # merge order across runs (matters for tie-breaking within Dictionary.add).
    for p in sorted(root.glob("SKK-JISYO.*")):
        if p.name in seen or p.suffix in {".gz", ".md5", ".bak"}:
            continue
        files.append(p)
    return files


def _build_engine() -> tuple[ImeCore, Converter, LearningStore]:
    dict_files = _find_dictionary_files()
    if not dict_files:
        raise RuntimeError(
            "No dictionary found. Run: python -m sokuhen_llm.scripts.download_dict"
        )
    log.info("Loading dictionaries: %s", [p.name for p in dict_files])
    d = Dictionary.load_skk(dict_files[0])
    for p in dict_files[1:]:
        # Name/place dictionaries are loaded with a +500 cost offset so
        # they're available as manual candidates but don't win over
        # ordinary vocabulary. Without this, e.g. しきの → 敷野 (surname)
        # beats [式][の] in "折りたたみ式の…".
        offset = 500 if any(
            tag in p.name.lower()
            for tag in ("jinmei", "fullname", "geo", "station")
        ) else 0
        d.merge_skk(p, cost_offset=offset)

    # SKK-JISYO.L and edict2 both ship many pure-katakana surfaces whose
    # readings are English (e.g., "monitor /モニター/", "speaker /スピーカー/").
    # merge_skk indexes them under the ASCII reading, which the user never
    # types. Walk the files again, this time keying by the hiragana
    # form of the katakana surface itself — that way when the user types
    # "もにたー" the Viterbi finds モニター at a reasonable cost.
    for p in dict_files:
        # Encoding: edict2 is UTF-8; SKK-JISYO.L and others are EUC-JP.
        enc = "utf-8" if "edict" in p.name.lower() else "euc_jp"
        n = d.merge_katakana_from_edict(p, encoding=enc)
        if n:
            log.info("Extracted %d katakana entries from %s", n, p.name)

    d.ensure_particles()
    log.info("Dictionary ready: %d entries", d.size)

    learning = LearningStore.load(learning_file())
    converter = Converter(d, language_model=learning.lm)

    # Build the LLM rescorer. It's optional — if transformers / torch
    # aren't installed, or the user turned it off, we ship a
    # DummyBackend and commit is unchanged from classical sokuhen.
    rescorer = _build_rescorer()

    composer = LiveComposer(converter, learning, rescorer=rescorer)
    core = ImeCore(composer, learning)
    return core, converter, learning


def _build_rescorer():
    """Instantiate the LLM rescorer, or ``None`` if unavailable/disabled.

    We MUST NOT block startup for model load (15 s on CPU). We also
    must not block any later keyboard-hook callback for longer than
    Windows' LowLevelHooksTimeout (~300 ms) -- exceeding that makes
    Windows silently unhook us, and from then on the IME receives no
    more keys (the original "Enter doesn't commit" bug).

    Solution: a proxy rescorer that spawns a background thread to
    construct ``HFBackend``. The hook callback path stays fast the
    whole time:

      * before the model is ready: rescore() returns the Viterbi
        result unchanged in <1 ms.
      * after load completes: rescore() delegates to a real
        ``Rescorer`` and adds ~150-250 ms on commit, which is fine
        because commit() runs on the Qt main thread, not inside a
        blocking hook callback (the hook returns True immediately
        on Enter-down; the actual commit work is deferred via
        Qt signals).

    Exposes a `status` property ("disabled" / "loading" / "ready" /
    "failed") for the UI to display.
    """
    import os

    if os.environ.get("SOKUHEN_LLM_DISABLE") == "1":
        log.info("LLM rescoring disabled via SOKUHEN_LLM_DISABLE")
        return None

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        log.info("LLM rescoring unavailable (transformers / torch not installed)")
        return None

    model_id = os.environ.get("SOKUHEN_LLM_MODEL", "rinna/japanese-gpt2-small")
    return _AsyncRescorer(model_id=model_id)


class _AsyncRescorer:
    """Non-blocking wrapper around ``Rescorer`` + ``HFBackend``.

    Construction kicks off a Python thread that instantiates the real
    rescorer. Until it's done, ``rescore()`` is a no-op -- commit
    still works, it just skips the LLM pass.

    The ``status`` property lets the UI render ``ロード中... / LLM ON /
    LLM エラー`` in the status bar. ``on_status_changed`` is a single
    callback slot (not a Qt signal to avoid importing Qt into this
    engine-level class; ``app.py`` wires it to Qt).
    """

    _STATUS_LOADING = "loading"
    _STATUS_READY = "ready"
    _STATUS_FAILED = "failed"

    def __init__(self, model_id: str) -> None:
        import threading

        self._model_id = model_id
        self._real = None
        self._status = self._STATUS_LOADING
        self._status_lock = threading.Lock()
        self._status_callback = None  # set by app.py after UI exists
        self._error_message = ""
        self._thread = threading.Thread(target=self._load, daemon=True, name="sokuhen-llm-load")
        self._thread.start()

    # --- public API ------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def error_message(self) -> str:
        return self._error_message

    def set_status_callback(self, callback) -> None:
        """Register a callable ``(status: str) -> None`` to be called
        whenever the load state transitions. Immediately invoked with
        the current status so callers get the initial state too."""
        self._status_callback = callback
        try:
            callback(self._status)
        except Exception:
            pass

    def rescore(self, frozen_prefix, result, initial_overrides=None):
        """Rescore if the backend has finished loading; otherwise
        return the Viterbi result unchanged (no-op, fast)."""
        real = self._real
        if real is None:
            from .llm.rescorer import RescoreResult
            return RescoreResult(
                overrides=dict(initial_overrides or {}),
                surface=result.surface_at(initial_overrides or {}),
                llm_hits=0,
            )
        return real.rescore(frozen_prefix, result, initial_overrides)

    # --- worker thread ---------------------------------------------

    def _load(self) -> None:
        try:
            from .llm.hf_backend import HFBackend
            from .llm.rescorer import Rescorer

            log.info("[LLM] background load starting (%s)", self._model_id)
            backend = HFBackend(model_id=self._model_id)
            if not backend.available:
                self._set_status(self._STATUS_FAILED, "backend unavailable")
                return
            self._real = Rescorer(backend)
            self._set_status(self._STATUS_READY)
            log.info("[LLM] ready -- rescoring will be used from next commit")
        except Exception as e:
            log.warning("[LLM] load failed: %s", e)
            self._set_status(self._STATUS_FAILED, str(e))

    def _set_status(self, status: str, error: str = "") -> None:
        with self._status_lock:
            self._status = status
            self._error_message = error
        cb = self._status_callback
        if cb is not None:
            try:
                cb(status)
            except Exception:
                log.debug("LLM status callback raised", exc_info=True)


def _make_tray_icon(active: bool) -> QIcon:
    """Generate a tiny filled-circle-with-'あ' icon.

    Needed because Qt's empty QIcon() renders invisibly in the Windows
    notification area. We paint a 32x32 pixmap so it looks crisp at any
    DPI scaling. Green for ON, grey for OFF — that's the user's only
    live signal that the IME is loaded.
    """
    pm = QPixmap(32, 32)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    color = QColor(70, 180, 110) if active else QColor(130, 130, 130)
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 30, 30)
    p.setPen(QColor(255, 255, 255))
    font = QFont("Yu Gothic UI", 14)
    font.setBold(True)
    p.setFont(font)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "あ" if active else "A")
    p.end()
    return QIcon(pm)


def _create_tray(app: QApplication, core: ImeCore) -> Optional[QSystemTrayIcon]:
    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning(
            "System tray is not available. The IME will still run but you "
            "won't see a tray icon."
        )
        return None

    tray = QSystemTrayIcon(_make_tray_icon(active=False), parent=app)
    tray.setToolTip("sokuhen-llm (OFF) — Alt+` to toggle")

    menu = QMenu()
    act_status = menu.addAction("sokuhen-llm: OFF")
    act_status.setEnabled(False)
    act_llm_status = menu.addAction("LLM: -")
    act_llm_status.setEnabled(False)
    menu.addSeparator()
    act_toggle = menu.addAction("IMEを切り替え  (Alt + `)")
    act_toggle.triggered.connect(core.toggle_active)
    menu.addSeparator()
    act_quit = menu.addAction("終了")
    act_quit.triggered.connect(app.quit)
    tray.setContextMenu(menu)

    # The tooltip encodes both IME on/off AND LLM state so hovering
    # the tray icon always tells the user what's going on.
    tray_state = {"active": False, "llm": "disabled"}
    _llm_tray_labels = {
        "loading":  "ロード中",
        "ready":    "ON",
        "disabled": "OFF",
        "failed":   "エラー",
    }

    def _refresh_tray() -> None:
        active = tray_state["active"]
        llm_label = _llm_tray_labels.get(tray_state["llm"], "OFF")
        tray.setIcon(_make_tray_icon(active=active))
        tray.setToolTip(
            f"sokuhen-llm (IME {'ON' if active else 'OFF'} / LLM {llm_label}) — Alt+` to toggle"
        )
        act_status.setText(f"sokuhen-llm: {'ON' if active else 'OFF'}")
        act_llm_status.setText(f"LLM: {llm_label}")

    def _on_active(active: bool) -> None:
        tray_state["active"] = active
        _refresh_tray()

    def _on_llm_status(status: str) -> None:
        tray_state["llm"] = status
        _refresh_tray()

    core.active_changed.connect(_on_active)
    core.llm_status_changed.connect(_on_llm_status)

    # Left-click on the tray icon also toggles — matches user expectations
    # from other notification-area apps.
    def _on_activated(reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            core.toggle_active()

    tray.activated.connect(_on_activated)
    tray.show()

    # Show a balloon tip once at startup so the user notices it's running.
    tray.showMessage(
        "sokuhen-llm",
        "起動しました。Alt + ` で IME を ON/OFF",
        QSystemTrayIcon.MessageIcon.Information,
        4000,
    )
    return tray


def _reset_learning() -> int:
    """Delete the learning file (user picks + LM counts). Useful when the
    model has been poisoned by bad commits. New typing starts from scratch."""
    lf = learning_file()
    if lf.exists():
        try:
            lf.unlink()
        except OSError as e:
            print(f"[reset] Failed to remove {lf}: {e}")
            return 1
        print(f"[reset] Removed {lf}")
    else:
        print("[reset] No learning data to remove.")
    return 0


def _stop_running_instance() -> int:
    """Terminate a running sokuhen-llm process, if any.

    Reads the PID file, sends a graceful WM_QUIT via signal, waits briefly,
    then force-kills if still alive. Removes the stale pid file either way.
    """
    pf = pid_file()
    if not pf.exists():
        print("[stop] No sokuhen-llm is running (no pid file).")
        return 0

    try:
        pid = int(pf.read_text().strip())
    except (OSError, ValueError) as e:
        print(f"[stop] Unreadable pid file: {e}. Removing.")
        try:
            pf.unlink(missing_ok=True)
        except OSError:
            pass
        return 1

    if not _pid_alive(pid):
        print(f"[stop] pid {pid} is not alive; cleaning up stale pid file.")
        try:
            pf.unlink(missing_ok=True)
        except OSError:
            pass
        return 0

    print(f"[stop] Stopping sokuhen-llm (pid={pid})...")
    if sys.platform == "win32":
        import ctypes
        import time as _time

        # First try: polite CTRL_BREAK_EVENT. Rarely works for GUI apps but
        # cheap to try.
        kernel32 = ctypes.windll.kernel32
        kernel32.GenerateConsoleCtrlEvent(1, pid)  # CTRL_BREAK_EVENT

        # Wait up to 3s for graceful shutdown.
        for _ in range(30):
            if not _pid_alive(pid):
                break
            _time.sleep(0.1)

        if _pid_alive(pid):
            # Force kill via taskkill (doesn't need admin for own process).
            import subprocess

            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True,
                check=False,
            )
            _time.sleep(0.3)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    try:
        pf.unlink(missing_ok=True)
    except OSError:
        pass

    if _pid_alive(pid):
        print(f"[stop] WARNING: pid {pid} still alive after kill attempt.")
        return 1
    print("[stop] sokuhen-llm stopped.")
    return 0


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is running (Windows)."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    import ctypes
    import ctypes.wintypes as wt

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    # Declare signatures so handles don't truncate on x64 / ARM64.
    kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
    kernel32.OpenProcess.restype = wt.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wt.HANDLE, ctypes.POINTER(wt.DWORD)]
    kernel32.GetExitCodeProcess.restype = wt.BOOL
    kernel32.CloseHandle.argtypes = [wt.HANDLE]
    kernel32.CloseHandle.restype = wt.BOOL

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    exit_code = wt.DWORD(0)
    kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
    kernel32.CloseHandle(handle)
    return exit_code.value == STILL_ACTIVE


def _run_selftest() -> int:
    """Verify setup without installing the keyboard hook.

    Useful when the user isn't sure whether the app is actually working.
    Exercises: dictionary load, a canned conversion, tray icon creation.
    """
    print("--- sokuhen-llm self test ---")
    try:
        app = QApplication([])  # noqa: F841 (Qt needs this for QPixmap etc.)
    except Exception as e:
        print(f"[FAIL] Qt init: {e}")
        return 1

    try:
        core, _, _ = _build_engine()
    except Exception as e:
        print(f"[FAIL] Engine build: {e}")
        return 1
    print(f"[ OK ] Dictionary loaded: {core.composer.converter.dict.size} entries")

    core.composer.set_active(True)
    for ch in "konnnichiha":
        core.composer.input_char(ch)
    out = core.composer.commit()
    print(f"[ OK ] Conversion 'konnnichiha' -> {out}")

    try:
        icon = _make_tray_icon(True)
        assert icon.availableSizes()
    except Exception as e:
        print(f"[FAIL] Tray icon: {e}")
        return 1
    print("[ OK ] Tray icon generated")

    try:
        _ = CompositionWindow()
    except Exception as e:
        print(f"[FAIL] UI window: {e}")
        return 1
    print("[ OK ] Composition window created")

    print("All checks passed. If the real launcher still seems silent,")
    print("look for a green 'あ' tray icon after pressing Alt+`.")
    return 0


def _setup_logging() -> Path:
    """Configure root logger. Writes to stderr (when a console exists) AND
    to a rotating file in the user log dir. Returns the log-file path so
    --selftest / stderr banners can show the user where to look.

    Needed because ``sokuhen-llm.bat`` launches the app with ``pythonw.exe``
    to detach from the CMD window — in that mode there is no console,
    so a file handler is the only way to see logs.
    """
    lf = log_dir() / "sokuhen-llm.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Idempotent — --selftest and main() both call this, and pytest might
    # import us multiple times. Avoid stacking handlers.
    has_file = any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
    )
    if not has_file:
        fh = logging.handlers.RotatingFileHandler(
            lf, maxBytes=500_000, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Only attach a stderr handler if we actually have a stderr stream
    # (pythonw.exe detaches streams — attempting to write there raises).
    if sys.stderr is not None and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in root.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    return lf


def main(argv: Optional[list[str]] = None) -> int:
    log_path = _setup_logging()

    args = list(argv or sys.argv[1:])
    if "--selftest" in args or "--check" in args:
        return _run_selftest()
    if "--stop" in args:
        return _stop_running_instance()
    if "--reset-learning" in args:
        return _reset_learning()

    app = QApplication(argv or sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("sokuhen-llm")

    # Single-instance guard. If a previous process left a stale pid file
    # we overwrite it (taskkill already failed or the process crashed).
    pf = pid_file()
    prev = 0
    if pf.exists():
        try:
            prev = int(pf.read_text().strip())
        except (OSError, ValueError):
            # Unreadable or garbage contents — treat as stale.
            prev = 0
    if prev and _pid_alive(prev):
        log.error(
            "sokuhen-llm already running (pid=%d). Use stop-sokuhen-llm.bat "
            "to quit it first.",
            prev,
        )
        return 1
    try:
        pf.write_text(str(os.getpid()), encoding="ascii")
        log.info("PID file written: %s (pid=%d)", pf, os.getpid())
    except OSError as e:
        # Not fatal — --stop won't work, but the IME can still run.
        log.warning("Could not write pid file %s: %s", pf, e)

    core, _converter, learning = _build_engine()
    window = CompositionWindow()
    indicator = StatusIndicator()

    def _on_state_changed() -> None:
        window.set_state(core.composer.state)

    def _on_active_changed(active: bool) -> None:
        indicator.set_active(active)

    def _on_llm_status_changed(status: str) -> None:
        indicator.set_llm_status(status)

    def _on_llm_rescored() -> None:
        indicator.flash_rescored()

    core.state_changed.connect(_on_state_changed)
    core.active_changed.connect(_on_active_changed)
    core.llm_status_changed.connect(_on_llm_status_changed)
    core.llm_rescored.connect(_on_llm_rescored)

    # Wire the _AsyncRescorer background loader to the UI. The
    # rescorer calls our callback from a worker thread; we bounce it
    # through a Qt signal so the indicator update happens on the
    # main thread.
    rescorer = getattr(core.composer, "rescorer", None)
    if rescorer is not None and hasattr(rescorer, "set_status_callback"):
        # Bounce the worker-thread status updates to the main thread
        # via the llm_status_changed signal (Qt handles the
        # cross-thread marshalling).
        rescorer.set_status_callback(lambda s: core.llm_status_changed.emit(s))
    else:
        # No rescorer (env disabled or deps missing). Show the
        # "disabled" state in the chip from the start.
        core.llm_status_changed.emit("disabled")

    hook = KeyboardHook()
    hook.on_event = core.handle_event
    hook.install()

    tray = _create_tray(app, core)  # noqa: F841 (keep reference alive)

    # Qt swallows Ctrl+C on Windows otherwise.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    # Use a timer tick so the Python signal handler gets serviced in the loop.
    heartbeat = QTimer()
    heartbeat.start(500)
    heartbeat.timeout.connect(lambda: None)

    # Startup banner. Uses stdout when a console is attached; always logged
    # to the file handler so pythonw-launched runs leave a trail.
    banner_lines = [
        "=" * 60,
        "  sokuhen-llm is READY.",
        "  Press  Alt + `  to toggle IME ON/OFF.",
        "  Tray icon: a grey 'A' (OFF) or green 'あ' (ON).",
        f"  Log file: {log_path}",
        "  To stop: run stop-sokuhen-llm.bat",
        "=" * 60,
    ]
    log.info("sokuhen-llm is READY. Log file: %s", log_path)
    if sys.stdout is not None:
        try:
            print()
            for line in banner_lines:
                print(line)
            print()
        except OSError:
            # pythonw-ish: stdout exists but isn't writable. Not fatal.
            pass

    try:
        return app.exec()
    finally:
        hook.uninstall()
        learning.save()
        try:
            pf.unlink(missing_ok=True)
        except OSError:
            pass
        log.info("sokuhen-llm quit. Learning data saved.")


if __name__ == "__main__":
    raise SystemExit(main())
