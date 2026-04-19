"""Small always-visible IME status indicator.

Sits in a fixed corner of the primary screen and clearly shows whether the
IME is currently ON. Necessary because the tray icon isn't a reliable
signal — Windows 11 hides tray icons by default, so users who haven't
pinned the app have no idea it's running.

UX informed by 桜井式:
  * やりすぎぐらいでちょうどいい — the pill uses a brighter FLASH color
    for 600ms after activation so the user's eye catches the state change.
  * メリハリ — the pill only appears while IME is ON; there is no
    persistent clutter in OFF state.
  * 遅さは罪 — animation is timer-driven, 600ms start-flash only.

When OFF: hidden.
When ON: pill in the bottom-right corner reading "あ IME ON".
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QApplication, QWidget

from ..input import _win32


# Visible, confident green. Strong enough to catch the eye on any app
# background, but rounded + tightly-packed so it reads as "status chip"
# not "alert".
ON_BG = QColor(40, 170, 90, 240)
ON_FG = QColor(255, 255, 255)
ON_BORDER = QColor(22, 120, 66)

FLASH_BG = QColor(80, 215, 130, 250)  # briefly brighter on toggle-on

PADDING_X = 14
PADDING_Y = 6
CORNER_MARGIN = 16


class StatusIndicator(QWidget):
    """Tiny floating chip in the screen corner showing IME ON."""

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
        self._flash = False

        # Flash timer: briefly use a brighter colour right after toggle-on
        # so the user's eye catches the state change.
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)

        # Periodically reassert topmost while visible. Fullscreen or
        # game-mode apps push us down otherwise and the user can lose
        # track of IME state. 750ms is fast enough to feel persistent
        # without being a CPU hog.
        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(750)
        self._topmost_timer.timeout.connect(self._force_topmost)

        self._size_to_content()
        self.hide()

    # --- public API ----------------------------------------------------

    def set_active(self, active: bool) -> None:
        if active:
            self._flash = True
            self._flash_timer.start(600)
            self._size_to_content()
            self._position_bottom_right()
            self.show()
            self._force_topmost()
            if not self._topmost_timer.isActive():
                self._topmost_timer.start()
            self.update()
        else:
            self._flash = False
            self._topmost_timer.stop()
            self.hide()

    # --- painting / layout --------------------------------------------

    def _end_flash(self) -> None:
        self._flash = False
        self.update()

    def _size_to_content(self) -> None:
        fm = QFontMetrics(self._font)
        label = "あ  IME ON"
        w = fm.horizontalAdvance(label) + PADDING_X * 2
        h = fm.height() + PADDING_Y * 2
        self.resize(w, h)

    def _position_bottom_right(self) -> None:
        """Bottom-right — stays near the Windows taskbar so it reads as
        'this app is running' without getting in the way of the work area."""
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

        bg = FLASH_BG if self._flash else ON_BG
        p.setBrush(bg)
        p.setPen(QPen(ON_BORDER, 1))
        radius = self.height() / 2
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), radius, radius)

        p.setFont(self._font)
        p.setPen(ON_FG)
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "あ  IME ON")
