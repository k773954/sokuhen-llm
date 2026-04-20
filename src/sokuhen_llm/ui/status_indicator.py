"""Always-visible IME status chip with LLM state.

Sits in a fixed bottom-right corner of the primary screen. The chip's
label and colour reflect two orthogonal states:

    IME on/off        -- shown/hidden
    LLM state         -- label + colour band inside the chip
                         * "ロード中..."  (yellow band)  LLM weights
                           are being pulled into memory
                         * "LLM ON"     (green band)    LLM rescoring
                           is active and will re-rank on commit
                         * "LLM OFF"    (grey band)     LLM disabled
                           (SOKUHEN_LLM_DISABLE, missing deps, etc.)
                         * "LLM エラー"  (red band)     backend failed
                           to load; IME still works, rescoring is off.

The chip also flashes for ~600 ms on these transitions so the user
sees the state change without having to look: activating the IME, or
the LLM becoming ready.

Why a big visible chip?
  * Users reported "LLM が動いている気配がない" -- the tray icon isn't
    enough and isn't always visible on Win11 (it hides by default).
  * Making the state obvious prevents the user from retyping because
    they think nothing is happening.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QApplication, QWidget

from ..input import _win32


# Band colours, keyed by LLM status. Pill body stays the same green
# whether LLM is on or not (the pill's purpose is "IME is live"); the
# secondary band on the right communicates LLM.
ON_BG = QColor(40, 170, 90, 240)
ON_FG = QColor(255, 255, 255)
ON_BORDER = QColor(22, 120, 66)
FLASH_BG = QColor(80, 215, 130, 250)

LLM_BAND_COLORS = {
    "ready":    QColor(80, 220, 120, 240),   # bright green
    "loading":  QColor(245, 200, 70, 240),   # amber
    "disabled": QColor(120, 120, 120, 220),  # neutral grey
    "failed":   QColor(220, 90, 80, 240),    # red
    # While a rescore is actively in flight we swap in a cool-blue
    # band with a pulsing dot animation (see _thinking_*). This gives
    # the user an unmistakable "LLM is thinking right now" cue, not
    # just the post-hoc "LLM changed this" flash.
    "thinking": QColor(110, 180, 255, 240),  # cool blue
}

LLM_BAND_LABELS = {
    "ready":    "LLM ON",
    "loading":  "ロード中...",
    "disabled": "LLM OFF",
    "failed":   "LLM エラー",
    "thinking": "LLM 思考中",
}

# Briefly flash the background slightly brighter when LLM adjusted
# the surface on commit -- confirms "the LLM changed something".
RESCORE_FLASH_BG = QColor(120, 200, 255, 250)   # cool blue pulse

PADDING_X = 14
PADDING_Y = 6
BAND_GAP = 8                      # px between main label and the LLM band
BAND_PADDING_X = 10
CORNER_MARGIN = 16


@dataclass
class ChipState:
    """Immutable display state for one paint cycle."""

    visible: bool = False
    flash_activate: bool = False
    flash_rescore: bool = False
    llm_status: str = "disabled"     # "ready" / "loading" / "disabled" / "failed"
    thinking: bool = False           # True while an LLM rescore is in flight
    thinking_phase: int = 0          # 0..2, rotating dot for the animation


class StatusIndicator(QWidget):
    """Floating chip: `あ  IME ON | LLM ON`."""

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.BypassWindowManagerHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._font = QFont("Yu Gothic UI", 11)
        self._font.setWeight(QFont.Weight.Bold)
        self._band_font = QFont("Yu Gothic UI", 10)
        self._band_font.setWeight(QFont.Weight.DemiBold)

        self._state = ChipState()

        # Flash timers. One for the "IME just toggled on" pulse, one
        # for the "LLM just rewrote this commit" pulse.
        self._activate_flash_timer = QTimer(self)
        self._activate_flash_timer.setSingleShot(True)
        self._activate_flash_timer.timeout.connect(self._end_activate_flash)

        self._rescore_flash_timer = QTimer(self)
        self._rescore_flash_timer.setSingleShot(True)
        self._rescore_flash_timer.timeout.connect(self._end_rescore_flash)

        # Drives the rotating "● ○ ○ / ○ ● ○ / ○ ○ ●" dot animation
        # that renders inside the LLM band while a rescore is in flight.
        # 150 ms per phase gives a lively, clearly-alive motion without
        # being noisy.
        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(150)
        self._thinking_timer.timeout.connect(self._advance_thinking_phase)

        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(750)
        self._topmost_timer.timeout.connect(self._force_topmost)

        self._size_to_content()
        self.hide()

    # --- public API ----------------------------------------------------

    def set_active(self, active: bool) -> None:
        self._state.visible = active
        if active:
            self._state.flash_activate = True
            self._activate_flash_timer.start(600)
            self._size_to_content()
            self._position_bottom_right()
            self.show()
            self._force_topmost()
            if not self._topmost_timer.isActive():
                self._topmost_timer.start()
            self.update()
        else:
            self._state.flash_activate = False
            self._topmost_timer.stop()
            self.hide()

    def set_llm_status(self, status: str) -> None:
        """Update the LLM band. ``status`` must be one of the keys in
        ``LLM_BAND_COLORS``; unknown values are normalized to
        ``disabled`` so the chip always renders something sane."""
        if status not in LLM_BAND_COLORS:
            status = "disabled"
        prev = self._state.llm_status
        self._state.llm_status = status
        # If LLM just finished loading, flash the chip so the user
        # sees the transition.
        if status == "ready" and prev != "ready":
            self._state.flash_activate = True
            self._activate_flash_timer.start(800)
        if self._state.visible:
            self._size_to_content()
            self._position_bottom_right()
            self.update()

    def flash_rescored(self) -> None:
        """Briefly pulse the chip to signal the LLM rewrote this
        commit's surface. Called from ImeCore after a non-trivial
        ``llm_hits`` count."""
        if not self._state.visible:
            return
        self._state.flash_rescore = True
        self._rescore_flash_timer.start(350)
        self.update()

    def set_thinking(self, thinking: bool) -> None:
        """Start / stop the in-flight 'LLM is rescoring right now'
        animation. While thinking is True the band switches to cool
        blue with rotating dots and the label reads ``LLM 思考中``;
        on stop we revert to the static ``LLM ON`` display."""
        if thinking == self._state.thinking:
            return
        self._state.thinking = thinking
        if thinking:
            self._state.thinking_phase = 0
            if not self._thinking_timer.isActive():
                self._thinking_timer.start()
        else:
            self._thinking_timer.stop()
        if self._state.visible:
            self._size_to_content()
            self._position_bottom_right()
            self.update()

    # --- painting / layout --------------------------------------------

    def _end_activate_flash(self) -> None:
        self._state.flash_activate = False
        self.update()

    def _end_rescore_flash(self) -> None:
        self._state.flash_rescore = False
        self.update()

    def _advance_thinking_phase(self) -> None:
        """Tick the dot animation forward one step (0 -> 1 -> 2 -> 0)."""
        if not self._state.thinking:
            return
        self._state.thinking_phase = (self._state.thinking_phase + 1) % 3
        self.update()

    def _label_main(self) -> str:
        return "あ  IME ON"

    def _effective_band_status(self) -> str:
        """The status whose colour/label we render. ``thinking``
        takes precedence over the base status (``ready``) because
        it's a transient in-flight state we specifically want the
        user to see."""
        if self._state.thinking and self._state.llm_status == "ready":
            return "thinking"
        return self._state.llm_status

    def _label_band(self) -> str:
        st = self._effective_band_status()
        return LLM_BAND_LABELS.get(st, "LLM OFF")

    def _size_to_content(self) -> None:
        fm_main = QFontMetrics(self._font)
        fm_band = QFontMetrics(self._band_font)
        w_main = fm_main.horizontalAdvance(self._label_main())
        w_band = fm_band.horizontalAdvance(self._label_band())
        w = PADDING_X + w_main + BAND_GAP + BAND_PADDING_X + w_band + BAND_PADDING_X + PADDING_X
        h = max(fm_main.height(), fm_band.height()) + PADDING_Y * 2
        self.resize(w, h)

    def _position_bottom_right(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            screen.right() - self.width() - CORNER_MARGIN,
            screen.bottom() - self.height() - CORNER_MARGIN,
        )

    def _force_topmost(self) -> None:
        _win32.set_topmost(int(self.winId()))

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Background pill -- green normally, briefly brighter on flash.
        if self._state.flash_rescore:
            bg = RESCORE_FLASH_BG
        elif self._state.flash_activate:
            bg = FLASH_BG
        else:
            bg = ON_BG
        p.setBrush(bg)
        p.setPen(QPen(ON_BORDER, 1))
        radius = self.height() / 2
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), radius, radius)

        # Main label "あ  IME ON"
        p.setFont(self._font)
        p.setPen(ON_FG)
        fm_main = QFontMetrics(self._font)
        fm_band = QFontMetrics(self._band_font)
        main_label = self._label_main()
        band_label = self._label_band()

        main_x = PADDING_X
        main_w = fm_main.horizontalAdvance(main_label)
        main_rect = self.rect()
        main_rect.setLeft(main_x)
        main_rect.setWidth(main_w)
        p.drawText(main_rect, Qt.AlignmentFlag.AlignVCenter, main_label)

        # LLM band chip, right side.
        effective_status = self._effective_band_status()
        band_color = LLM_BAND_COLORS.get(effective_status, LLM_BAND_COLORS["disabled"])
        band_w = fm_band.horizontalAdvance(band_label) + BAND_PADDING_X * 2
        # Reserve extra width for the rotating dots while thinking.
        if self._state.thinking:
            band_w += 22
        band_h = self.height() - PADDING_Y - 2
        band_x = main_x + main_w + BAND_GAP
        band_y = (self.height() - band_h) // 2
        band_rect = self.rect()
        band_rect.setLeft(band_x)
        band_rect.setTop(band_y)
        band_rect.setWidth(band_w)
        band_rect.setHeight(band_h)
        band_radius = band_h / 2
        p.setBrush(band_color)
        p.setPen(QPen(band_color.darker(130), 1))
        p.drawRoundedRect(band_rect, band_radius, band_radius)

        p.setFont(self._band_font)
        p.setPen(ON_FG)
        if self._state.thinking:
            # Draw the label flush-left, leaving room for the dots on
            # the right. Draws three dots, the one matching the
            # current phase rendered full-opacity; the others dimmed.
            text_rect = band_rect.adjusted(0, 0, -20, 0)
            p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, band_label)
            dot_area_x = band_rect.right() - 22
            dot_cy = band_rect.center().y()
            for k in range(3):
                alpha = 240 if k == self._state.thinking_phase else 110
                dot_color = QColor(255, 255, 255, alpha)
                p.setBrush(dot_color)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(dot_area_x + k * 6, dot_cy - 2, 4, 4)
        else:
            p.drawText(band_rect, Qt.AlignmentFlag.AlignCenter, band_label)
