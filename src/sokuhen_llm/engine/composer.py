"""Live composer: maintains composition state and applies sliding freeze.

Core idea (the "いい塩梅" requirement):

- The user's trailing input lives in a window of at most ``WINDOW_KANA``
  hiragana characters. We re-run the Viterbi converter on this window every
  keystroke so the trailing part gets a fresh, context-aware conversion.
- When the window would overflow, we "bake" the earliest segments out of
  the window into a frozen surface string. Frozen text is never rewritten
  as the user keeps typing — it's visually stable.
- The last surface of the frozen region is carried over as LM context so
  the first segment of the live window still sees a realistic prev word.

States:

    frozen_surface          live_kana (≤ WINDOW_KANA chars)   pending_romaji
    ┌─────────────────────┐┌───────────────────────────────┐┌──────────────┐
    │ 今日はいい天気で     ││ しんぶんをよんでい             ││ m            │
    └─────────────────────┘└───────────────────────────────┘└──────────────┘
                            ^ converted live every keystroke   ^ tail not yet
                                                                 a full syllable

On commit (Enter), ``commit()`` returns ``frozen_surface + live_surface + pending_romaji``.
Pending romaji gets passed through verbatim if the user commits mid-syllable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .dictionary import hiragana_to_katakana
from .learning import LearningStore
from .romaji import RomajiConverter
from .viterbi import ConversionResult, Converter


# Default live window size (hiragana chars). Large enough to hold a full
# short sentence without ever baking, so typical input never gets the
# "前のほうの変換が再度行われて変になる" problem the user hit.
WINDOW_KANA = 32

# Bake hysteresis: only start baking when the buffer exceeds window by
# this much. Prevents single-kana trickle-baking at bad boundaries.
BAKE_HYSTERESIS = 8

# Punctuation always safe to bake across. These are unambiguous sentence
# / phrase boundaries — the Viterbi rarely splits them weirdly.
_SAFE_PUNCTUATION = frozenset("、。・！？!?\n")


def _is_safe_boundary(reading: str) -> bool:
    """True if a segment is a natural phrase break.

    Only punctuation is reliably safe. We also require the whole segment
    to be punctuation — if it's a compound like "。き" (punctuation got
    swallowed into the next kana), we wait for a cleaner boundary rather
    than freezing past it. And we don't use particles even when they form
    a standalone segment, because words like とても get segmented as
    "とて" + "も" on long buffers and the stray も would mislead us.
    """
    return bool(reading) and all(c in _SAFE_PUNCTUATION for c in reading)


@dataclass
class ComposerState:
    """All composition state. Used by the UI to render and by the app to
    know what to inject on commit."""

    active: bool = False  # IME on/off
    frozen_surface: str = ""  # stable prefix (past the live window)
    frozen_last_surface: str = ""  # last baked surface, for LM continuity
    kana_buffer: str = ""  # hiragana in the live window
    pending_romaji: str = ""  # trailing romaji waiting for its vowel
    # Raw ASCII keystrokes as typed, in order. Kept alongside the kana
    # view so F9/F10 can recover "what the user actually typed" as
    # full-/half-width ASCII — we can't derive it back from kana_buffer
    # because the romaji→kana mapping is lossy.
    raw_input: str = ""
    result: Optional[ConversionResult] = None
    overrides: dict[int, int] = field(default_factory=dict)  # seg idx -> cand idx
    selected_segment: int = 0
    # True when the user pressed Space / arrow keys to start picking an
    # alternative. The UI reads this to decide whether to render the
    # candidate list panel below the composition line.
    show_candidates: bool = False
    # True once the user has manually picked a candidate (Space) or
    # resized a segment (Shift+Arrow). Two effects:
    #   1. LLM rescoring is suspended while this flag is set, so the
    #      LLM can never silently overwrite the user's deliberate
    #      choice.
    #   2. The first Enter absorbs-but-doesn't-commit as an explicit
    #      confirmation step; a second Enter actually commits. This
    #      prevents "I pressed Space 3 times to pick the right one,
    #      then Enter to confirm the PICK, but it committed with the
    #      wrong one instead" style mishaps.
    # Reset by any new romaji input, by cancel(), or by the
    # confirmation Enter itself.
    manually_edited: bool = False

    # --- computed views ---------------------------------------------------

    @property
    def live_surface(self) -> str:
        if self.result is None:
            return self.kana_buffer
        return self.result.surface_at(self.overrides)

    @property
    def display_text(self) -> str:
        """What the floating IME window should show. Pending romaji is
        rendered raw so the user sees their keystrokes immediately —
        except a lone 'n' which previews as ん (it always resolves to
        ん on commit or the next consonant)."""
        pending = self.pending_romaji
        display_pending = "ん" if pending == "n" else pending
        return self.frozen_surface + self.live_surface + display_pending

    @property
    def is_empty(self) -> bool:
        return (
            not self.frozen_surface
            and not self.kana_buffer
            and not self.pending_romaji
        )


class LiveComposer:
    """Orchestrates romaji input, live conversion, freezing, and commit."""

    def __init__(
        self,
        converter: Converter,
        learning: LearningStore,
        *,
        romaji: Optional[RomajiConverter] = None,
        window_kana: int = WINDOW_KANA,
        bake_hysteresis: Optional[int] = None,
        rescorer: Optional[object] = None,
    ) -> None:
        self.converter = converter
        self.learning = learning
        self.romaji = romaji or RomajiConverter()
        self.window = window_kana
        # Default: ~25% of window, so smaller windows still exercise baking.
        self.bake_hysteresis = (
            bake_hysteresis if bake_hysteresis is not None else max(2, window_kana // 4)
        )
        self.state = ComposerState()
        # Optional LLM rescorer. The rescorer is ducktyped (we only call
        # ``rescore(frozen_prefix, result, overrides)`` on it) to avoid
        # hard-importing the transformers-heavy LLM module in contexts
        # that don't use it -- tests, selftest, users without torch.
        self.rescorer = rescorer

    # --- IME state -------------------------------------------------------

    def set_active(self, active: bool) -> None:
        if self.state.active == active:
            return
        if not active and not self.state.is_empty:
            # Switching off with pending composition: cancel it.
            self.cancel()
        self.state.active = active

    def cancel(self) -> None:
        self.state = ComposerState(active=self.state.active)

    def commit(self) -> str:
        """Confirm the current composition. Returns the text to inject into
        the foreground app. Updates the learning model with observed surfaces.

        Resolves any trailing ``pending_romaji`` first — a lone "n" becomes
        ん, otherwise the raw tail is kept. This avoids the common case where
        the user presses Enter mid-syllable and sees a stray "n" appear,
        then needs to press Enter again to commit it.

        We intentionally do NOT run LLM rescoring here. The live
        rescoring layer wired into ``_reconvert`` already keeps
        ``self.state.overrides`` up-to-date with the LLM's preferred
        surface as the user types, so commit is fast (< 1 ms). If the
        user hits Enter before the background rescore finishes for the
        very latest keystroke, they'll get the Viterbi result for that
        edge -- a micro-second visual loss, but the hook stays snappy
        and under Windows' LowLevelHooksTimeout no matter what.
        """
        # Resolve any trailing romaji before reading display_text.
        self._flush_pending()

        text = self.state.display_text
        self._observe_commit()
        self.state = ComposerState(active=self.state.active)
        return text

    def _observe_commit(self) -> None:
        """Update LM with the full committed sequence of surfaces.

        When a frozen prefix exists, we prepend its last surface so the
        bigram bridging frozen_last → first_live_segment is learned too.
        The full frozen history isn't kept (each bake calls observe
        immediately), so we only reconstruct the junction.
        """
        surfaces: list[str] = []
        if self.state.result is not None:
            for i, seg in enumerate(self.state.result.segments):
                idx = self.state.overrides.get(i, 0)
                if 0 <= idx < len(seg.candidates):
                    surfaces.append(seg.candidates[idx].surface)
                else:
                    surfaces.append(seg.surface)
                self.learning.record_pick(seg.reading, surfaces[-1])
        if not surfaces:
            return
        if self.state.frozen_last_surface:
            # Include the bridging bigram (frozen_last, first_live).
            self.learning.lm.observe([self.state.frozen_last_surface, *surfaces])
        else:
            self.learning.lm.observe(surfaces)

    # --- key handling ----------------------------------------------------

    def input_char(self, ch: str) -> None:
        """Append a printable character to the buffer. Assumes IME is active.

        Only runs the Viterbi conversion when a full kana was produced — if
        the keystroke is mid-syllable (e.g., 'k' waiting for its vowel), we
        just update ``pending_romaji`` so the UI shows the live key without
        re-running the whole lattice search on the unchanged ``kana_buffer``.
        """
        if not self.state.active:
            return
        self.state.raw_input += ch

        combined = self.state.pending_romaji + ch
        r = self.romaji.convert(combined)

        if r.kana:
            self.state.kana_buffer += r.kana
            self._maybe_bake()
            self.state.pending_romaji = r.pending
            self._reconvert()
        else:
            # Only the pending tail changed — no conversion work needed.
            self.state.pending_romaji = r.pending

    def backspace(self) -> bool:
        """Delete one unit from the tail. Returns True if we consumed the
        backspace (caller should suppress it); False to pass through (e.g.
        composition is empty and the OS-level backspace should fire)."""
        if not self.state.active:
            return False
        if self.state.pending_romaji:
            self.state.pending_romaji = self.state.pending_romaji[:-1]
            self.state.raw_input = self.state.raw_input[:-1]
            self._reconvert()
            return True
        if self.state.kana_buffer:
            self.state.kana_buffer = self.state.kana_buffer[:-1]
            # Pop the raw char(s) that produced the removed kana. Since
            # the mapping is not strictly 1:1 (ka → か = 2 romaji → 1
            # kana), we just pop one raw char per call; this stays in
            # sync well enough for F9/F10 use and is visually sensible.
            if self.state.raw_input:
                self.state.raw_input = self.state.raw_input[:-1]
            self._reconvert()
            return True
        # Don't eat backspace into frozen — frozen is user-committed context.
        # Letting it fall through could look surprising; safer to clear frozen
        # entirely so the user can resume.
        if self.state.frozen_surface:
            self.state.frozen_surface = ""
            self.state.frozen_last_surface = ""
            return True
        return False

    def next_candidate(self, delta: int = 1) -> None:
        """Cycle the currently selected segment's candidate by ``delta``.

        Also marks the state as "candidate list visible" so the UI panel
        opens on the first Space / arrow key. Subsequent presses keep
        cycling within the list.

        Sets ``manually_edited`` so the background LLM rescorer stops
        overriding, and the next Enter needs an explicit confirm.
        """
        if self.state.result is None or not self.state.result.segments:
            return
        idx = self.state.selected_segment
        seg = self.state.result.segments[idx]
        cur = self.state.overrides.get(idx, 0)
        new = (cur + delta) % max(1, len(seg.candidates))
        self.state.overrides[idx] = new
        self.state.show_candidates = True
        self.state.manually_edited = True

    def hide_candidates(self) -> None:
        self.state.show_candidates = False

    def select_segment(self, delta: int) -> None:
        """Move the segment cursor left (-1) or right (+1)."""
        if self.state.result is None or not self.state.result.segments:
            return
        n = len(self.state.result.segments)
        self.state.selected_segment = max(0, min(n - 1, self.state.selected_segment + delta))

    def resize_segment(self, delta: int) -> None:
        """Extend (+1) or shrink (-1) the END of the currently selected
        segment, MS-IME style. Everything before the active segment stays
        locked; everything after is re-converted with Viterbi using the
        resized segment's chosen surface as the LM context.

        Called on Shift+Right / Shift+Left.
        """
        if self.state.result is None or not self.state.result.segments:
            return
        idx = self.state.selected_segment
        segs = self.state.result.segments
        seg = segs[idx]

        # Bounds: the segment must stay at least 1 char wide; the new end
        # must not exceed the buffer.
        new_end = seg.end + delta
        min_end = seg.start + 1
        max_end = len(self.state.kana_buffer)
        new_end = max(min_end, min(max_end, new_end))
        if new_end == seg.end:
            return

        locked = list(segs[:idx])
        new_reading = self.state.kana_buffer[seg.start:new_end]

        # Pick the lowest-cost surface covering the new reading. If nothing
        # in the dict matches, fall back to hiragana.
        dict_entries = sorted(self.converter.dict.lookup(new_reading), key=lambda e: e.cost)
        if dict_entries:
            chosen = dict_entries[0]
        else:
            # Import locally to avoid a top-level cycle (dictionary ↔ viterbi
            # already import each other).
            from .dictionary import Dictionary as _D

            chosen = _D.generate_fallback(new_reading)[0]

        alts = self.converter.alternatives(new_reading, chosen)
        # Build the resized segment.
        from .viterbi import ConversionSegment

        new_seg = ConversionSegment(
            start=seg.start,
            end=new_end,
            reading=new_reading,
            surface=chosen.surface,
            candidates=alts,
        )

        # Re-run Viterbi on everything after the new boundary, seeded with
        # the resized segment's surface so the very next segment sees a
        # realistic left context.
        remainder = self.state.kana_buffer[new_end:]
        remainder_segs: list[ConversionSegment] = []
        if remainder:
            rr = self.converter.convert(remainder, bos=chosen.surface)
            for s in rr.segments:
                remainder_segs.append(
                    ConversionSegment(
                        start=s.start + new_end,
                        end=s.end + new_end,
                        reading=s.reading,
                        surface=s.surface,
                        candidates=s.candidates,
                    )
                )

        from .viterbi import ConversionResult

        self.state.result = ConversionResult(
            reading=self.state.kana_buffer,
            segments=locked + [new_seg] + remainder_segs,
        )
        # Drop overrides for segments past the modified one — their indices
        # shifted, and their candidates changed. Keep earlier ones.
        self.state.overrides = {
            i: v for i, v in self.state.overrides.items() if i < idx
        }
        # Keep selection on the resized segment, clamped.
        self.state.selected_segment = min(idx, len(self.state.result.segments) - 1)
        self.state.show_candidates = False
        self.state.manually_edited = True

    def _flush_pending(self) -> None:
        """Resolve pending romaji into the kana buffer. 'n'/'nn' become ん,
        partial syllables pass through. Called before any operation that
        wants to see the full composition (F7/F8 force conversion, commit).
        """
        if not self.state.pending_romaji:
            return
        tail = self.romaji.finalize(self.state.pending_romaji)
        if tail and all(ord(c) >= 0x3000 for c in tail):
            self.state.kana_buffer += tail
            self.state.pending_romaji = ""
            self._reconvert()

    def force_conversion(self, kind: str) -> None:
        """Apply a whole-buffer transform (F7/F8/F9/F10 style).

        - F7 katakana     : hira → kata on ``kana_buffer``
        - F8 hankaku_kana : hira → half-width kata on ``kana_buffer``
        - F9 zenkaku_ascii: replace live with full-width ASCII of
                            the raw keystroke log (``raw_input``)
        - F10 hankaku_ascii: replace live with the raw keystroke log
                             as-is (half-width ASCII)

        F9/F10 work by replaying ``raw_input`` — the exact characters the
        user typed. That way ``hello`` committed through F10 goes to the
        app as ``hello``, not ``へっろ`` or the partial kana we had
        accumulated. This matches MS IME's behavior where F9/F10 answer
        "what if I'd typed the same keys with IME off?".
        """
        if not (self.state.kana_buffer or self.state.pending_romaji):
            return
        if kind in ("katakana", "hankaku_kana"):
            # F7/F8 operate on the full kana composition. Flush the
            # pending tail (lone 'n', 'nn') into the buffer first so the
            # user's intended ん shows up in the converted surface.
            self._flush_pending()
        if kind == "katakana":
            self._replace_live(
                hiragana_to_katakana(self.state.kana_buffer)
            )
            self.state.pending_romaji = ""
        elif kind == "hankaku_kana":
            self._replace_live(
                _to_halfwidth_katakana(self.state.kana_buffer)
            )
            self.state.pending_romaji = ""
        elif kind == "zenkaku_ascii":
            self._replace_live(_to_fullwidth_ascii(self.state.raw_input))
            self.state.kana_buffer = self.state.raw_input
            self.state.pending_romaji = ""
        elif kind == "hankaku_ascii":
            self._replace_live(self.state.raw_input)
            self.state.kana_buffer = self.state.raw_input
            self.state.pending_romaji = ""

    def _replace_live(self, surface: str) -> None:
        """Override the live conversion with a single surface string."""
        from .viterbi import ConversionSegment
        from .dictionary import DictEntry

        seg = ConversionSegment(
            start=0,
            end=len(self.state.kana_buffer),
            reading=self.state.kana_buffer,
            surface=surface,
            candidates=[DictEntry(reading=self.state.kana_buffer, surface=surface, cost=0, source="forced")],
        )
        self.state.result = ConversionResult(
            reading=self.state.kana_buffer, segments=[seg]
        )
        self.state.overrides = {}
        self.state.selected_segment = 0

    # --- internals -------------------------------------------------------

    def _reconvert(self) -> None:
        """Re-run conversion on the live window after a state change.

        After setting ``self.state.result``, this also fires
        ``self.on_reconverted`` if it's set. sokuhen-llm's ImeCore
        registers a callback there to schedule background LLM
        rescoring on the new reading (debounced, async, so the hot
        keystroke path stays fast).

        Any fresh typing invalidates the ``manually_edited`` state --
        the user moved on and LLM rescoring should resume.
        """
        self.state.manually_edited = False
        if not self.state.kana_buffer:
            self.state.result = None
            self.state.overrides = {}
            self.state.selected_segment = 0
            cb = getattr(self, "on_reconverted", None)
            if cb is not None:
                cb()
            return
        # Use the last frozen surface as BOS so the first segment sees a
        # realistic left context in the LM's bigram lookup.
        bos = self.state.frozen_last_surface or None
        self.state.result = self.converter.convert(self.state.kana_buffer, bos=bos)

        self.state.overrides = {}
        # Note: we deliberately do NOT hard-override the segment choice
        # from `learning.preferred_surface` here. A single accidental
        # commit would otherwise pin the wrong surface forever (the
        # "今す" poisoning case the user hit). Instead, learning influences
        # selection through the bigram language model (see language_model.py)
        # so a single bad pick is outvoted once it's seen in context a
        # handful of times.
        # Reset cursor to the last segment on reconversion — matches how macOS
        # tends to focus the trailing 文節 as you type.
        if self.state.result.segments:
            self.state.selected_segment = len(self.state.result.segments) - 1
        cb = getattr(self, "on_reconverted", None)
        if cb is not None:
            cb()

    def _maybe_bake(self) -> None:
        """Bake early segments into frozen_surface when the buffer outgrows
        the live window.

        Rules (tuned from the failure mode where the user saw earlier
        conversion re-arrange as they typed):

        1. Wait until buffer exceeds ``window + BAKE_HYSTERESIS``. This
           avoids baking a single kana every keystroke past the threshold,
           which was freezing bad boundaries.

        2. Only bake segments whose END lands on a "safe" boundary — the
           last character must be a particle or punctuation. Compound words
           can't be mid-word-frozen.

        3. If no safe boundary exists in the bakeable prefix, bail and
           wait for more input. The worst case is a very long buffer, but
           Viterbi on ~200 kana is still sub-millisecond so we don't
           actually need to bake for performance reasons.
        """
        if len(self.state.kana_buffer) <= self.window + self.bake_hysteresis:
            return
        max_bake = len(self.state.kana_buffer) - self.window

        tmp = self.converter.convert(self.state.kana_buffer)

        # Walk segments in order. For each segment fully inside the bake
        # zone, remember its surface and — if this segment IS a safe
        # boundary (standalone particle or punctuation) — mark it as the
        # latest acceptable bake endpoint. Checking the whole segment (not
        # just its last char) avoids treating words like とても (ends in も)
        # as particle phrases.
        running: list[str] = []
        accepted_end = 0
        accepted_surfaces: list[str] = []
        for i, seg in enumerate(tmp.segments):
            if seg.end > max_bake:
                break
            idx = self.state.overrides.get(i, 0)
            surface = (
                seg.candidates[idx].surface
                if 0 <= idx < len(seg.candidates)
                else seg.surface
            )
            running.append(surface)
            if _is_safe_boundary(seg.reading):
                accepted_end = seg.end
                accepted_surfaces = list(running)

        if accepted_end == 0:
            # No safe boundary in the bakeable prefix — keep waiting.
            return

        self.state.frozen_surface += "".join(accepted_surfaces)
        self.state.frozen_last_surface = accepted_surfaces[-1]
        self.state.kana_buffer = self.state.kana_buffer[accepted_end:]
        self.state.overrides = {}
        self.learning.lm.observe(accepted_surfaces)


# --- kana width helpers (for F8/F9 compatibility) -----------------------


_HANKAKU_MAP = {
    "ア": "ｱ", "イ": "ｲ", "ウ": "ｳ", "エ": "ｴ", "オ": "ｵ",
    "カ": "ｶ", "キ": "ｷ", "ク": "ｸ", "ケ": "ｹ", "コ": "ｺ",
    "サ": "ｻ", "シ": "ｼ", "ス": "ｽ", "セ": "ｾ", "ソ": "ｿ",
    "タ": "ﾀ", "チ": "ﾁ", "ツ": "ﾂ", "テ": "ﾃ", "ト": "ﾄ",
    "ナ": "ﾅ", "ニ": "ﾆ", "ヌ": "ﾇ", "ネ": "ﾈ", "ノ": "ﾉ",
    "ハ": "ﾊ", "ヒ": "ﾋ", "フ": "ﾌ", "ヘ": "ﾍ", "ホ": "ﾎ",
    "マ": "ﾏ", "ミ": "ﾐ", "ム": "ﾑ", "メ": "ﾒ", "モ": "ﾓ",
    "ヤ": "ﾔ", "ユ": "ﾕ", "ヨ": "ﾖ",
    "ラ": "ﾗ", "リ": "ﾘ", "ル": "ﾙ", "レ": "ﾚ", "ロ": "ﾛ",
    "ワ": "ﾜ", "ヲ": "ｦ", "ン": "ﾝ",
    "ァ": "ｧ", "ィ": "ｨ", "ゥ": "ｩ", "ェ": "ｪ", "ォ": "ｫ",
    "ッ": "ｯ", "ャ": "ｬ", "ュ": "ｭ", "ョ": "ｮ",
    "ー": "ｰ", "、": "､", "。": "｡", "「": "｢", "」": "｣", "・": "･",
}


def _to_halfwidth_katakana(kana: str) -> str:
    kata = hiragana_to_katakana(kana)
    out = []
    for ch in kata:
        if ch in _HANKAKU_MAP:
            out.append(_HANKAKU_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


def _to_fullwidth_ascii(ascii_str: str) -> str:
    out = []
    for ch in ascii_str:
        cp = ord(ch)
        if 0x21 <= cp <= 0x7E:
            out.append(chr(cp - 0x21 + 0xFF01))
        elif ch == " ":
            out.append("\u3000")
        else:
            out.append(ch)
    return "".join(out)
