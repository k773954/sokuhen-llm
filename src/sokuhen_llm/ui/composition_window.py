"""Floating composition + candidate window.

UX informed by 桜井式アセット制作原則 (see .agents/skills/
sakurai2_asset_creation):

  * やりすぎぐらいでちょうどいい  — active segment is markedly bigger
    than inactive ones, underline is 3px not 1, color contrast is high.
  * ヒットストップ (hit-stop)     — on commit we flash the window green
    for ~120ms before hiding so the user sees "that just committed".
  * 情報階層 (大事・小事)          — composition text is the biggest,
    candidate surfaces medium, key-hint footer smallest.
  * 右脳的なUI                     — frozen text is grey, live is near-
    black, active segment is blue, pending romaji is warm orange.
    Users can read the state at a glance without parsing labels.
  * 遅さは罪                       — all animations are timer-driven
    and ≤200ms; no opacity fades or stagger delays.
"""
from __future__ import annotations

import sys

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QPen,
)
from PyQt6.QtWidgets import QApplication, QWidget

from ..engine.composer import CANDIDATE_PAGE_SIZE, ComposerState
from ..input import _win32


# --- palette -----------------------------------------------------------

# Background: off-white with a faint warm tint so it reads as "our window"
# rather than a system panel. High alpha so text stays readable over
# patterned app backgrounds.
BG_COLOR = QColor(252, 252, 254, 248)
BG_COLOR_FLASH = QColor(210, 245, 220, 250)  # commit hit-stop tint
BORDER_COLOR = QColor(60, 130, 220)
BORDER_COLOR_FLASH = QColor(50, 190, 90)

FROZEN_COLOR = QColor(150, 158, 168)       # muted grey — "past"
LIVE_COLOR = QColor(20, 20, 28)            # near-black — "present"
ACTIVE_COLOR = QColor(20, 72, 160)         # strong blue — "here"
PENDING_COLOR = QColor(220, 110, 20)       # warm orange — "not yet kana"

ACTIVE_UNDERLINE_COLOR = QColor(60, 130, 220)
ACTIVE_UNDERLINE_BG = QColor(60, 130, 220, 45)
INACTIVE_UNDERLINE_COLOR = QColor(160, 190, 230)

CAND_BG = QColor(253, 253, 255, 252)
CAND_BORDER = QColor(200, 210, 220)
CAND_TEXT = QColor(30, 30, 40)
CAND_INDEX_BG = QColor(235, 238, 245)
CAND_INDEX_TEXT = QColor(100, 115, 140)
CAND_SELECTED_BG = QColor(60, 130, 220)
CAND_SELECTED_TEXT = QColor(255, 255, 255)
CAND_SELECTED_INDEX_BG = QColor(30, 95, 185)

FOOTER_COLOR = QColor(130, 140, 160)
FOOTER_BG = QColor(244, 246, 250)

# --- dimensions --------------------------------------------------------

PADDING_X = 14
PADDING_Y = 10
ACTIVE_UNDERLINE_H = 3
INACTIVE_UNDERLINE_H = 1
CAND_PADDING = 4
CAND_ROW_HEIGHT = 30
CAND_MAX_VISIBLE = CANDIDATE_PAGE_SIZE
CAND_INDEX_W = 26
MIN_WIDTH = 380
FOOTER_HEIGHT = 22


# --- the widget --------------------------------------------------------


class CompositionWindow(QWidget):
    """Frameless always-on-top window that renders the composer state."""

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

        self._state: ComposerState | None = None
        self._anchor: QPoint | None = None
        self._flash = False  # hit-stop flash currently active?

        # Fonts: text is noticeably larger than previous revs so the active
        # segment dominates the visual hierarchy (沙くらい式「大事・小事」).
        self._text_font = QFont("Yu Gothic UI", 16)
        self._text_font.setWeight(QFont.Weight.Medium)
        self._active_font = QFont("Yu Gothic UI", 17)
        self._active_font.setWeight(QFont.Weight.DemiBold)
        self._cand_font = QFont("Yu Gothic UI", 13)
        self._cand_font.setWeight(QFont.Weight.Medium)
        self._footer_font = QFont("Yu Gothic UI", 9)

        # Timer that reasserts topmost while visible. Fullscreen / Electron
        # windows push us down otherwise. 250ms is frequent enough to stay
        # visible in most "fighting for topmost" scenarios without being
        # CPU-wasteful.
        self._topmost_timer = QTimer(self)
        self._topmost_timer.setInterval(250)
        self._topmost_timer.timeout.connect(self._force_topmost)

        # Hit-stop timer — briefly flash green when a commit fires, then
        # hide. Visible feedback matters even on input that gets typed
        # elsewhere.
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._end_flash)

        self.resize(MIN_WIDTH, 50)
        self.hide()

    # -- public API ------------------------------------------------------

    def set_state(self, state: ComposerState) -> None:
        prev_empty = self._state is None or self._state.is_empty
        prev_text = self._state.display_text if self._state else ""
        self._state = state

        if state.is_empty:
            self._anchor = None
            self._topmost_timer.stop()
            # When commit just happened (non-empty → empty), let the user
            # see it. A 120ms green flash feels like a "pop" acknowledgement.
            if not prev_empty and prev_text:
                self._trigger_commit_flash()
            elif self._flash_timer.isActive():
                # Mid-flash; the timer will hide us when it finishes.
                pass
            else:
                self.hide()
            return

        # We only use the anchor to decide WHICH screen to pin to (in
        # multi-monitor setups), not for actual x/y placement. See
        # _reposition(): placement is fixed bottom-center of that screen.
        if prev_empty:
            self._anchor = QPoint(*_win32.input_anchor_position())

        self._resize_to_content()
        self._reposition()
        if not self.isVisible():
            self.show()
        self._force_topmost()
        if not self._topmost_timer.isActive():
            self._topmost_timer.start()
        self.update()

    # -- flash helpers ---------------------------------------------------

    def _trigger_commit_flash(self) -> None:
        """Brief green hit-stop then hide. Signals 'committed'."""
        self._flash = True
        self.update()
        self._flash_timer.start(120)

    def _end_flash(self) -> None:
        self._flash = False
        self.hide()
        self.update()

    # -- layout ----------------------------------------------------------

    def _text_metrics(self) -> QFontMetrics:
        return QFontMetrics(self._text_font)

    def _active_metrics(self) -> QFontMetrics:
        return QFontMetrics(self._active_font)

    def _cand_metrics(self) -> QFontMetrics:
        return QFontMetrics(self._cand_font)

    @property
    def _candidates_visible(self) -> bool:
        return bool(self._state and self._state.show_candidates)

    def _footer_hint(self) -> str:
        if self._candidates_visible:
            return "1-9選択  ↑↓  PgUp/PgDn  Home/End  Enter  Esc"
        return "Space候補  ←→文節  Shift+←→幅  Ctrl+Backspace削除  Enter確定"

    def _resize_to_content(self) -> None:
        if not self._state:
            return
        # Measure with the larger "active" font because one segment uses it.
        fm = self._active_metrics()
        display_text = self._state.display_text or " "
        text_w = fm.horizontalAdvance(display_text)
        text_h = fm.height()
        footer_w = QFontMetrics(self._footer_font).horizontalAdvance(self._footer_hint())

        w = max(text_w + PADDING_X * 2, footer_w + PADDING_X * 2, MIN_WIDTH)
        h = text_h + PADDING_Y * 2 + ACTIVE_UNDERLINE_H + FOOTER_HEIGHT

        if self._candidates_visible and self._state.result and self._state.result.segments:
            seg = self._state.result.segments[self._state.selected_segment]
            start = self._candidate_window_start()
            rows = min(CAND_MAX_VISIBLE, len(seg.candidates) - start)
            cfm = self._cand_metrics()
            cand_text_w = max(
                cfm.horizontalAdvance(c.surface)
                for c in seg.candidates[start:start + rows]
            ) if rows else 0
            cand_w = CAND_INDEX_W + cand_text_w + CAND_PADDING * 4
            w = max(w, cand_w + CAND_PADDING * 2)
            h += rows * CAND_ROW_HEIGHT + CAND_PADDING * 2 + 2

        # Grow only during a single composition; avoids flicker as the user
        # types more characters.
        self.resize(max(w, self.width()) if self.isVisible() and not self._flash else w, h)

    def _reposition(self) -> None:
        """Pinned position: bottom-center of the screen containing the
        active window, with a small gap above the taskbar.

        User feedback was that caret-tracking keeps landing in the wrong
        place across apps (Claude / Electron / Chromium / different
        DPI). A predictable fixed spot — always visible, never off-screen
        — is more useful than an accurate-when-it-works floater.

        Which screen? Whichever holds the foreground window. That way a
        user typing in a browser on the secondary display sees the
        composition on the same display.
        """
        app = QApplication.instance()
        screen = None
        if app is not None:
            anchor_pt = self._anchor if self._anchor is not None else QPoint(
                *_win32.input_anchor_position()
            )
            screen_obj = app.screenAt(anchor_pt)  # type: ignore[attr-defined]
            if screen_obj is None:
                screen_obj = app.primaryScreen()
            if screen_obj is not None:
                screen = screen_obj.availableGeometry()
        if screen is None:
            self.move(40, 40)
            return

        w, h = self.width(), self.height()
        MARGIN_BOTTOM = 32  # above taskbar / dock area
        x = screen.left() + (screen.width() - w) // 2
        y = screen.bottom() - h - MARGIN_BOTTOM
        # Clamp defensively for tiny displays.
        if y < screen.top():
            y = screen.top()
        self.move(x, y)

    def _force_topmost(self) -> None:
        _win32.set_topmost(int(self.winId()))

    # -- painting --------------------------------------------------------

    def _candidate_window_start(self) -> int:
        """First candidate index to show, keeping the active one visible."""
        assert self._state is not None and self._state.result is not None
        seg = self._state.result.segments[self._state.selected_segment]
        total = len(seg.candidates)
        if total <= CAND_MAX_VISIBLE:
            return 0
        active_idx = self._state.overrides.get(self._state.selected_segment, 0)
        active_idx = max(0, min(active_idx, total - 1))
        return min(max(active_idx - CAND_MAX_VISIBLE + 1, 0), total - CAND_MAX_VISIBLE)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        rect = self.rect()
        bg = BG_COLOR_FLASH if self._flash else BG_COLOR
        border = BORDER_COLOR_FLASH if self._flash else BORDER_COLOR
        painter.setBrush(bg)
        painter.setPen(QPen(border, 2))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)

        if self._state is None:
            return

        self._paint_composition_line(painter)

        if (
            self._candidates_visible
            and self._state.result
            and self._state.result.segments
        ):
            self._paint_candidates(painter)

        self._paint_footer(painter)

    # -- composition line ----------------------------------------------

    def _paint_composition_line(self, p: QPainter) -> None:
        assert self._state is not None
        # We paint text with two fonts: normal for non-active segments and
        # a slightly larger bolder font for the active segment. That
        # visual "+1" makes the cursor location obvious without waiting
        # for the user to parse the underline.
        normal_fm = self._text_metrics()
        active_fm = self._active_metrics()
        baseline = PADDING_Y + active_fm.ascent()
        x = PADDING_X

        # 1) frozen prefix — muted grey
        if self._state.frozen_surface:
            p.setFont(self._text_font)
            p.setPen(FROZEN_COLOR)
            p.drawText(QPoint(x, baseline), self._state.frozen_surface)
            x += normal_fm.horizontalAdvance(self._state.frozen_surface)

        # 2) live segments — highlight active with larger font + underline
        if self._state.result and self._state.result.segments:
            for i, seg in enumerate(self._state.result.segments):
                idx = self._state.overrides.get(i, 0)
                surface = (
                    seg.candidates[idx].surface
                    if 0 <= idx < len(seg.candidates)
                    else seg.surface
                )
                is_active = i == self._state.selected_segment
                fm = active_fm if is_active else normal_fm
                seg_w = fm.horizontalAdvance(surface)
                underline_y = PADDING_Y + active_fm.height() + 2

                if is_active:
                    # Soft blue background + thick underline — the眼球
                    # jumps here immediately.
                    bg_rect = QRect(
                        x - 3,
                        PADDING_Y - 2,
                        seg_w + 6,
                        active_fm.height() + 4,
                    )
                    p.fillRect(bg_rect, ACTIVE_UNDERLINE_BG)
                    p.setFont(self._active_font)
                    p.setPen(ACTIVE_COLOR)
                    p.drawText(QPoint(x, baseline), surface)
                    p.fillRect(
                        QRect(x - 1, underline_y, seg_w + 2, ACTIVE_UNDERLINE_H),
                        ACTIVE_UNDERLINE_COLOR,
                    )
                else:
                    p.setFont(self._text_font)
                    p.setPen(LIVE_COLOR)
                    p.drawText(QPoint(x, baseline), surface)
                    p.fillRect(
                        QRect(x, underline_y + (ACTIVE_UNDERLINE_H - INACTIVE_UNDERLINE_H),
                              seg_w, INACTIVE_UNDERLINE_H),
                        INACTIVE_UNDERLINE_COLOR,
                    )
                x += seg_w
        elif self._state.kana_buffer:
            p.setFont(self._text_font)
            p.setPen(LIVE_COLOR)
            p.drawText(QPoint(x, baseline), self._state.kana_buffer)
            x += normal_fm.horizontalAdvance(self._state.kana_buffer)

        # 3) pending romaji — warm orange, clearly "not yet kana"
        pending = self._state.pending_romaji
        if pending:
            display_pending = "ん" if pending == "n" else pending
            p.setFont(self._text_font)
            p.setPen(PENDING_COLOR)
            p.drawText(QPoint(x, baseline), display_pending)

    # -- candidates ------------------------------------------------------

    def _paint_candidates(self, p: QPainter) -> None:
        assert self._state is not None and self._state.result is not None
        seg = self._state.result.segments[self._state.selected_segment]
        active_idx = self._state.overrides.get(self._state.selected_segment, 0)
        start = self._candidate_window_start()

        rows = min(CAND_MAX_VISIBLE, len(seg.candidates) - start)
        y_start = (
            PADDING_Y
            + self._active_metrics().height()
            + ACTIVE_UNDERLINE_H
            + 6
        )
        panel = QRect(
            CAND_PADDING,
            y_start,
            self.width() - CAND_PADDING * 2,
            rows * CAND_ROW_HEIGHT + CAND_PADDING * 2,
        )
        p.setBrush(CAND_BG)
        p.setPen(QPen(CAND_BORDER, 1))
        p.drawRoundedRect(panel, 6, 6)

        p.setFont(self._cand_font)
        for row in range(rows):
            candidate_index = start + row
            cand = seg.candidates[candidate_index]
            row_rect = QRect(
                panel.left() + CAND_PADDING,
                panel.top() + CAND_PADDING + row * CAND_ROW_HEIGHT,
                panel.width() - CAND_PADDING * 2,
                CAND_ROW_HEIGHT,
            )
            selected = candidate_index == active_idx

            # Row background
            if selected:
                p.fillRect(row_rect, CAND_SELECTED_BG)

            # Number badge — distinct background so the digit is scannable
            index_rect = QRect(
                row_rect.left() + 4,
                row_rect.top() + 4,
                CAND_INDEX_W - 2,
                row_rect.height() - 8,
            )
            p.setBrush(CAND_SELECTED_INDEX_BG if selected else CAND_INDEX_BG)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(index_rect, 4, 4)
            p.setPen(CAND_SELECTED_TEXT if selected else CAND_INDEX_TEXT)
            p.drawText(
                index_rect,
                Qt.AlignmentFlag.AlignCenter,
                str(row + 1),
            )

            # Surface text — the actual candidate
            text_rect = QRect(
                row_rect.left() + CAND_INDEX_W + 8,
                row_rect.top(),
                row_rect.width() - CAND_INDEX_W - 12,
                row_rect.height(),
            )
            p.setPen(CAND_SELECTED_TEXT if selected else CAND_TEXT)
            p.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                cand.surface,
            )

    # -- footer (key hints) ---------------------------------------------

    def _paint_footer(self, p: QPainter) -> None:
        """Small hint strip at the bottom: Space/↑↓/Enter/Esc cheat sheet."""
        if self._flash:
            return
        footer_rect = QRect(
            1,
            self.height() - FOOTER_HEIGHT - 1,
            self.width() - 2,
            FOOTER_HEIGHT,
        )
        # Soft separator line above the hint
        p.setPen(QPen(QColor(220, 225, 235), 1))
        p.drawLine(
            footer_rect.left(),
            footer_rect.top(),
            footer_rect.right(),
            footer_rect.top(),
        )
        p.setFont(self._footer_font)
        p.setPen(FOOTER_COLOR)

        p.drawText(
            footer_rect.adjusted(PADDING_X, 0, -PADDING_X, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._footer_hint(),
        )
