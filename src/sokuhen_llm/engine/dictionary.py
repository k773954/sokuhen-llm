"""Dictionary: hiragana reading -> list of surface candidates.

Loads SKK-JISYO format files (EUC-JP encoded text, one entry per line:
``reading /surface1/surface2;annotation/.../``). Indexes by reading and
by reading prefix (for live segmentation where we look up the longest
prefix of the remaining input that exists as a word).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .boosts import (
    EXTRA_WORDS,
    FOREIGN_WORDS,
    PARTICLES,
    particle_cost,
    should_register_extra_word,
    should_register_function_word,
)


# SKK annotations like "word;meaning" — we drop the annotation for display
# but keep the surface form itself.
_ANNOT_RE = re.compile(r";.*$")

# Core case particles whose 3-4 char compound reading is almost certainly
# better split. Narrow on purpose — adding `さ`/`と`/`も` causes collateral
# damage to legitimate compounds.
# 3-4 char reading ends in one of these → compound is penalized so it
# loses to [noun][particle] segmentation. Sentence-final 'よ' is included
# because 3-char compounds ending in よ (いくよ → 幾世) tend to be rare
# names. 'で' included for the same reason — bogus okuri-ari forms like
# しあいで → 仕上いで otherwise beat [試合][で]. 'か' is NOT included —
# it appears in everyday nouns like おなか/さかな.
_COMPOUND_PARTICLE_TAIL = {"は", "が", "を", "に", "へ", "と", "よ", "で"}

# Particle-initial compound readings — narrow set (excludes に, で which
# prefix many legitimate compounds like にほん, にわ, でんわ, でんしゃ).
_COMPOUND_PARTICLE_HEAD = {"が", "を", "へ"}

# SKK escape: characters like '/' are encoded as "\057" (octal). Rare in
# Japanese entries but present in a few cases; decode minimally.
_OCTAL_RE = re.compile(r"\\0([0-7]{2})")


def _decode_skk_value(val: str) -> str:
    val = _ANNOT_RE.sub("", val)
    val = _OCTAL_RE.sub(lambda m: chr(int(m.group(1), 8)), val)
    return val


@dataclass(frozen=True)
class DictEntry:
    """A single dictionary entry (reading + surface).

    ``cost`` is lower-is-better (mozc-style). Default base cost favors entries
    that actually came from the dictionary over auto-generated fallback
    (single-character kana) entries.
    """

    reading: str
    surface: str
    cost: int = 6000
    source: str = ""  # "dict", "fallback", "user", ...

    @property
    def length(self) -> int:
        return len(self.reading)


@dataclass
class Dictionary:
    """In-memory dictionary with reading -> entries index.

    The reading index uses a simple dict. For prefix lookups we expose
    ``has_any_prefix`` (fast) and callers iterate over ``prefixes_of`` to
    enumerate matching lengths at a given position — see Converter.
    """

    _by_reading: dict[str, list[DictEntry]] = field(default_factory=dict)
    _readings: set[str] = field(default_factory=set)
    _prefix_lens: dict[str, set[int]] = field(default_factory=dict)

    # --- loading ---------------------------------------------------------

    @classmethod
    def load_skk(cls, path: Path, *, encoding: str = "euc_jp") -> "Dictionary":
        d = cls()
        d.merge_skk(path, encoding=encoding)
        return d

    def merge_katakana_from_edict(
        self, path: Path, *, encoding: str = "utf-8", cost: int = 3500
    ) -> int:
        """Extract katakana-surface entries from an SKK-style dictionary file.

        edict2's readings are English / abbreviations, not hiragana, so a
        normal ``merge_skk`` lookup misses them. But the surfaces often
        contain excellent katakana loanword coverage (プラスマイナス,
        モニター, スピーカー, etc.). We walk every entry, pick
        pure-katakana surfaces, and add them under the hiragana version
        of that surface as the lookup reading.

        Only surfaces whose hiragana reading contains a "foreign signal"
        (the long-vowel mark ー or a foreign digraph like ふぁ, ゔぃ,
        ちぇ, とぅ) are added. This filters out trivial katakana
        transliterations of native Japanese words — SKK also ships
        entries like ``baseball /野球/ベースボール/ヤキュウ/`` where
        ヤキュウ is just kana-form of the reading and would wrongly
        beat 野球 if added as a normal katakana entry.

        Cost defaults to 3500 — a little above typical kanji cost so
        genuine kanji wins by default; foreign words reach the
        surface when the kanji path is unavailable or via Space
        cycling.
        """
        count = 0
        try:
            f = open(path, "r", encoding=encoding, errors="replace")
        except FileNotFoundError:
            return 0
        with f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line or line.startswith(";"):
                    continue
                try:
                    _reading, body = line.split(" ", 1)
                except ValueError:
                    continue
                body = body.strip()
                if not body.startswith("/") or not body.endswith("/"):
                    continue
                for raw_surface in body[1:-1].split("/"):
                    surface = _decode_skk_value(raw_surface)
                    if not surface or len(surface) < 2:
                        continue
                    # Pure katakana only (plus ー, ・, and space).
                    if not all(
                        0x30A0 <= ord(c) <= 0x30FF or c in "ー・ "
                        for c in surface
                    ):
                        continue
                    # Split compound surfaces on ・ and space.
                    if any(sep in surface for sep in "・ "):
                        parts = [surface.replace("・", "").replace(" ", "")]
                        parts.extend(
                            p for p in surface.replace("・", " ").split()
                            if len(p) >= 2
                        )
                    else:
                        parts = [surface]
                    seen_surfaces: set[str] = set()
                    for part in parts:
                        if part in seen_surfaces or len(part) < 2:
                            continue
                        seen_surfaces.add(part)
                        hira_reading = katakana_to_hiragana(part)
                        if hira_reading == part:
                            continue
                        # Skip unless the reading has a foreign signal.
                        # This filters out pure-kana transliterations of
                        # native words (ヤキュウ, ニホン, カンコク) that
                        # would wrongly beat the kanji forms.
                        if not _has_foreign_signal(hira_reading):
                            continue
                        self.add(
                            DictEntry(
                                reading=hira_reading,
                                surface=part,
                                cost=cost,
                                source="kata-dict",
                            )
                        )
                        count += 1
        return count

    def merge_skk(
        self, path: Path, *, encoding: str = "euc_jp", cost_offset: int = 0
    ) -> int:
        """Load an SKK-JISYO file. Returns entry count (including generated
        okuri-ari conjugation forms).

        ``cost_offset`` is added to every entry's base cost. Use a positive
        offset for name/place dictionaries (jinmei, geo) so their readings
        are available as candidates but don't beat ordinary words — e.g.
        ``しきの /敷野/`` at cost 3000 would otherwise out-rank a split of
        [式][の] when typing 折りたたみ式の...
        """
        count = 0
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line or line.startswith(";"):
                    continue
                try:
                    reading, body = line.split(" ", 1)
                except ValueError:
                    continue
                body = body.strip()
                if not body.startswith("/") or not body.endswith("/"):
                    continue
                surfaces = [s for s in body[1:-1].split("/") if s]

                # Detect okuri-ari: SKK encodes these with a single trailing
                # ascii letter (the stem consonant, e.g. "たべr", "かk").
                okuri_stem = ""
                kana_reading = reading
                if reading and reading[-1].isascii() and reading[:-1] and not reading[:-1][-1].isascii():
                    okuri_stem = reading[-1]
                    kana_reading = reading[:-1]

                for i, raw_surface in enumerate(surfaces):
                    surface = _decode_skk_value(raw_surface)
                    if not surface:
                        continue
                    cost = 3000 + i * 50 + cost_offset
                    if okuri_stem:
                        # R-stem on a bare-kanji base can be one of:
                        #   (a) ichidan-非-RERU: でr→出, みr→見, きr→着,
                        #       おきr→起, etc. The raw r-table form is
                        #       correct: で+た→出た, み+た→見た.
                        #   (b) ichidan-RERU: つかr→疲, みだr→乱. The raw
                        #       form is wrong (疲た); the れ-inserted form
                        #       is right (疲れた).
                        #   (c) godan-r: でr→照, のr→乗. Raw past is wrong
                        #       (乗た, should be 乗った); re-inserted is also
                        #       wrong (乗れた is potential form, not past).
                        # We can't classify without POS info. Emit BOTH the
                        # raw form (covers case a) and the れ-inserted form
                        # (covers case b) at normal cost, and let context +
                        # the LM bigram table disambiguate. For godan-r
                        # past/て forms, neither is correct — those have to
                        # fall through to a separate explicit entry or
                        # remain unconverted (acceptable tradeoff since
                        # most godan-r past forms either live as proper
                        # nouns in SKK or are infrequent enough that the
                        # user will pick from alternatives).
                        bare_kanji_r = (
                            okuri_stem == "r" and not _has_hiragana(surface)
                        )
                        for kana_suffix, surf_suffix, extra_cost in _conjugations(okuri_stem):
                            full_reading = kana_reading + kana_suffix
                            full_surface = surface + surf_suffix
                            entry_cost = cost + extra_cost
                            if extra_cost > 0 and _has_hiragana(full_surface):
                                # Real inflected forms with a hiragana okuri
                                # (書いた, 話して, 食べた, 出た, 見た) get a
                                # -80 discount so they can beat same-reading
                                # proper-noun okuri-nasi entries like 吹田/
                                # 海田 at 3000.
                                entry_cost -= 80
                            self.add(
                                DictEntry(
                                    reading=full_reading,
                                    surface=full_surface,
                                    cost=entry_cost,
                                    source="dict",
                                )
                            )
                            count += 1

                            if bare_kanji_r:
                                # Also generate the れ-inserted form for
                                # possible ichidan-RERU verbs. Pricing is
                                # slightly higher than the raw form so
                                # ichidan-非-RERU verbs (the more common
                                # case) win ties.
                                re_reading = kana_reading + "れ" + kana_suffix
                                re_surface = surface + "れ" + surf_suffix
                                self.add(
                                    DictEntry(
                                        reading=re_reading,
                                        surface=re_surface,
                                        cost=cost + extra_cost,
                                        source="dict",
                                    )
                                )
                                count += 1
                    else:
                        self.add(
                            DictEntry(
                                reading=kana_reading,
                                surface=surface,
                                cost=cost,
                                source="dict",
                            )
                        )
                        count += 1
        return count

    def add(self, entry: DictEntry) -> None:
        # Apply particle boost: if the entry IS a known particle (hiragana
        # surface equals known particle), override cost to the particle table.
        if entry.surface == entry.reading:
            boost = particle_cost(entry.surface)
            if boost is not None and entry.cost > boost:
                entry = DictEntry(
                    reading=entry.reading,
                    surface=entry.surface,
                    cost=boost,
                    source=entry.source,
                )

        # Symbol/emoji penalty: SKK ships a few novelty entries like
        # ``るん /♪/`` that have a "typical" cost of 3000 and therefore
        # hijack common verb conjugations. Penalize any surface whose
        # characters aren't kana/kanji/ASCII.
        if entry.source == "dict" and _is_symbolish(entry.surface):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 5000,
                source=entry.source,
            )

        # Digit-reading penalty: SKK ships entries like "15 /十五/" which
        # assume the user typed the reading in hiragana, but in our IME
        # the user who types "15" almost always wants digit passthrough,
        # not kanji conversion. Penalize any dict entry whose reading is
        # purely ASCII digits and whose surface isn't the same digits.
        if (
            entry.source == "dict"
            and entry.reading
            and all(c.isdigit() and c.isascii() for c in entry.reading)
            and entry.surface != entry.reading
        ):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 10000,
                source=entry.source,
            )

        # Penalize short rare kanji: if reading is 1-2 kana and the surface
        # is a single kanji from a long candidate list, bump cost.
        if (
            len(entry.reading) <= 2
            and entry.source == "dict"
            and entry.surface != entry.reading
            and entry.cost >= 3100
        ):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 400,
                source=entry.source,
            )

        # Penalize compound words whose reading ends in a core case particle
        # (は/が/を/に/へ). Most 3-4 char readings like きょうは (教派) are
        # better split as [noun][particle]; the rare compound can still be
        # selected from candidates.
        if (
            3 <= len(entry.reading) <= 4
            and entry.source == "dict"
            and entry.reading[-1] in _COMPOUND_PARTICLE_TAIL
            and entry.surface != entry.reading
        ):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 3500,
                source=entry.source,
            )

        # Same penalty (smaller) for 5-6 char compounds ending in a case
        # particle. Catches things like おんがくは → 音楽派 (which escapes
        # the 3-4 char rule) while leaving set phrases mostly intact —
        # こんにちは (今日は) loses here, but the split [今日][は] produces
        # the same visual result so there's no regression. +800 is tuned
        # to tip the balance: a 5-char compound at SKK base cost 3000 ends
        # up at 3800, vs the 4-char compound + particle path at ~3100 +
        # transition ≈ 3600.
        if (
            5 <= len(entry.reading) <= 6
            and entry.source == "dict"
            and entry.reading[-1] in _COMPOUND_PARTICLE_TAIL
            and entry.surface != entry.reading
        ):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 800,
                source=entry.source,
            )

        # Mirror penalty: 3-char compound whose reading STARTS with a particle
        # (が, を, へ). E.g., がさい → 画才 is almost always better split as
        # [particle][noun]. Scoped to 3-char and a narrow head set because
        # 'に'/'で' prefixes many legitimate compounds (にほん, にわ, でんわ).
        if (
            len(entry.reading) == 3
            and entry.source == "dict"
            and entry.reading[0] in _COMPOUND_PARTICLE_HEAD
            and entry.surface != entry.reading
        ):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 2000,
                source=entry.source,
            )

        # Mid-particle penalty: 3-4 char compound whose reading has a case
        # particle character in an interior position (e.g., やまがみ with が
        # at pos 2). The interior position suggests a natural word+particle
        # boundary was crossed. Skip inflected-form surfaces (containing
        # hiragana) because those particles are usually part of the verb
        # stem (e.g. ながした = 流 stem なが + した, where が is just a
        # coincidence — 流 = なが, not な).
        if (
            len(entry.reading) in (3, 4)
            and entry.source == "dict"
            and entry.surface != entry.reading
            and not _has_hiragana(entry.surface)
            and any(
                entry.reading[i] in _COMPOUND_PARTICLE_TAIL
                for i in range(1, len(entry.reading) - 1)
            )
        ):
            entry = DictEntry(
                reading=entry.reading,
                surface=entry.surface,
                cost=entry.cost + 1800,
                source=entry.source,
            )

        bucket = self._by_reading.setdefault(entry.reading, [])
        for existing in bucket:
            if existing.surface == entry.surface:
                if entry.cost < existing.cost:
                    bucket.remove(existing)
                    break
                return
        bucket.append(entry)
        self._readings.add(entry.reading)
        first = entry.reading[0]
        self._prefix_lens.setdefault(first, set()).add(len(entry.reading))

    def ensure_particles(self) -> None:
        """Insert explicit hiragana entries for all known particles and
        katakana entries for common foreign words. Called after loading so
        that items absent from SKK-JISYO still appear as first-class,
        low-cost lattice candidates.

        Foreign words are also registered under their double-vowel reading
        (けえき alongside けーき) since wāpuro romaji users naturally type
        "keeki" which becomes けえき, not けーき.
        """
        for surface, cost in PARTICLES.items():
            if should_register_function_word(surface):
                self.add(
                    DictEntry(
                        reading=surface,
                        surface=surface,
                        cost=cost,
                        source="particle",
                    )
                )
        for reading in FOREIGN_WORDS:
            kata = hiragana_to_katakana(reading)
            if kata == reading:
                continue
            self.add(DictEntry(reading=reading, surface=kata, cost=500, source="foreign"))
            # Also register the double-vowel variant of the reading so
            # "deeta"/"keeki"/"koodo" (how wāpuro typists actually enter
            # foreign words) reach データ/ケーキ/コード.
            alt = _expand_long_vowels(reading)
            if alt and alt != reading:
                self.add(
                    DictEntry(reading=alt, surface=kata, cost=500, source="foreign")
                )
        for reading, surface, cost in EXTRA_WORDS:
            if should_register_extra_word(reading, surface):
                self.add(
                    DictEntry(reading=reading, surface=surface, cost=cost, source="extra")
                )

    # --- lookup ----------------------------------------------------------

    def lookup(self, reading: str) -> list[DictEntry]:
        return list(self._by_reading.get(reading, ()))

    def contains(self, reading: str) -> bool:
        return reading in self._readings

    def candidate_lengths_at(self, text: str, pos: int) -> list[int]:
        """Return sorted list of word lengths that start at ``text[pos]``
        and match some dictionary entry. Empty list if none.
        """
        if pos >= len(text):
            return []
        first = text[pos]
        lens = self._prefix_lens.get(first)
        if not lens:
            return []
        remaining = len(text) - pos
        # Only lengths that fit and whose substring is actually in the dict.
        out = []
        for ln in lens:
            if ln <= remaining and text[pos : pos + ln] in self._readings:
                out.append(ln)
        out.sort()
        return out

    @property
    def size(self) -> int:
        return sum(len(v) for v in self._by_reading.values())

    def readings(self) -> Iterable[str]:
        return self._readings

    # --- static helpers --------------------------------------------------

    @staticmethod
    def generate_fallback(reading: str) -> list[DictEntry]:
        """Produce fallback entries for a reading not in the dictionary.

        Used when Viterbi can't find a covering path with dictionary words
        alone. Returns entries that keep the reading as-is (hiragana) and
        a katakana version. Cost is high so dictionary matches always win.
        """
        out = [DictEntry(reading=reading, surface=reading, cost=8000, source="fallback")]
        kata = hiragana_to_katakana(reading)
        if kata != reading:
            out.append(
                DictEntry(reading=reading, surface=kata, cost=9000, source="fallback")
            )
        return out


# --- kana utilities -----------------------------------------------------

_HIRA_START, _HIRA_END = ord("ぁ"), ord("ゖ")
_KATA_OFFSET = ord("ァ") - ord("ぁ")


def _has_hiragana(s: str) -> bool:
    return any(_HIRA_START <= ord(c) <= _HIRA_END for c in s)


# "Foreign signal" patterns — same as in viterbi._foreign_run. A hiragana
# reading containing one of these is almost certainly a katakana loanword
# rather than a native Japanese word.
_FOREIGN_DIGRAPHS_FOR_DICT = frozenset([
    "うぃ", "うぇ", "うぉ",
    "ふぁ", "ふぃ", "ふぇ", "ふぉ",
    "ゔぁ", "ゔぃ", "ゔぇ", "ゔぉ",
    "てぃ", "とぅ", "でぃ", "どぅ",
    "ちぇ", "しぇ", "じぇ",
    "つぁ", "つぇ", "つぉ",
    "くぉ",
])


def _has_foreign_signal(reading: str) -> bool:
    """Return True if the hiragana reading contains a marker that indicates
    a katakana loanword: the long-vowel mark ー, or one of the foreign
    digraphs that don't occur in native Japanese readings."""
    if "ー" in reading:
        return True
    for i in range(len(reading) - 1):
        if reading[i : i + 2] in _FOREIGN_DIGRAPHS_FOR_DICT:
            return True
    return False


# Kana -> following vowel that would double via long-vowel mark (ー).
# Used to expand readings like けーき → けえき so wāpuro double-vowel input
# still reaches foreign words even though they're stored with the long mark.
_LONG_VOWEL_MAP = {
    # a-row
    "あ": "あ", "か": "あ", "さ": "あ", "た": "あ", "な": "あ",
    "は": "あ", "ま": "あ", "や": "あ", "ら": "あ", "わ": "あ",
    "が": "あ", "ざ": "あ", "だ": "あ", "ば": "あ", "ぱ": "あ",
    "ゃ": "あ",
    # i-row
    "い": "い", "き": "い", "し": "い", "ち": "い", "に": "い",
    "ひ": "い", "み": "い", "り": "い",
    "ぎ": "い", "じ": "い", "ぢ": "い", "び": "い", "ぴ": "い",
    # u-row
    "う": "う", "く": "う", "す": "う", "つ": "う", "ぬ": "う",
    "ふ": "う", "む": "う", "ゆ": "う", "る": "う",
    "ぐ": "う", "ず": "う", "づ": "う", "ぶ": "う", "ぷ": "う",
    "ゅ": "う",
    # e-row (long mark typically expands to え or to い depending on
    # convention; modern wāpuro usage accepts either so we emit え).
    "え": "え", "け": "え", "せ": "え", "て": "え", "ね": "え",
    "へ": "え", "め": "え", "れ": "え",
    "げ": "え", "ぜ": "え", "で": "え", "べ": "え", "ぺ": "え",
    # o-row — same story; emit お.
    "お": "お", "こ": "お", "そ": "お", "と": "お", "の": "お",
    "ほ": "お", "も": "お", "よ": "お", "ろ": "お", "を": "お",
    "ご": "お", "ぞ": "お", "ど": "お", "ぼ": "お", "ぽ": "お",
    "ょ": "お",
}


def _expand_long_vowels(reading: str) -> str:
    """Replace ー with the appropriate double-vowel kana.

    「けーき」 → 「けえき」 so that typing 'keeki' (→ けえき via wāpuro)
    hits the same FOREIGN_WORDS entry as the long-mark form.
    """
    if "ー" not in reading:
        return reading
    out: list[str] = []
    for i, c in enumerate(reading):
        if c == "ー" and i > 0:
            prev = reading[i - 1]
            dup = _LONG_VOWEL_MAP.get(prev)
            if dup is not None:
                out.append(dup)
                continue
        out.append(c)
    return "".join(out)


def _is_symbolish(s: str) -> bool:
    """Heuristic: the surface contains a character that isn't ordinary
    Japanese text (kana, kanji, ASCII, full-width ASCII, or the long-vowel
    mark ー). A surface like ``♪`` or ``☆`` counts; so do CJK punctuation
    like ＜＞. Used to push novelty SKK entries down the ranking."""
    for c in s:
        cp = ord(c)
        # Hiragana / Katakana / Half-width katakana / CJK unified
        if 0x3040 <= cp <= 0x30FF:
            continue
        if 0xFF65 <= cp <= 0xFF9F:
            continue
        if 0x3400 <= cp <= 0x9FFF:
            continue
        if 0x4E00 <= cp <= 0x9FFF:
            continue
        # ASCII printable
        if 0x20 <= cp <= 0x7E:
            continue
        # Full-width ASCII
        if 0xFF00 <= cp <= 0xFF5F:
            continue
        # The long-vowel mark (already in katakana block 30FC but double-check)
        if cp == 0x30FC:
            continue
        return True
    return False


def hiragana_to_katakana(s: str) -> str:
    return "".join(
        chr(ord(c) + _KATA_OFFSET) if _HIRA_START <= ord(c) <= _HIRA_END else c
        for c in s
    )


_KATA_START, _KATA_END = ord("ァ"), ord("ヶ")


def katakana_to_hiragana(s: str) -> str:
    return "".join(
        chr(ord(c) - _KATA_OFFSET) if _KATA_START <= ord(c) <= _KATA_END else c
        for c in s
    )


# --- conjugation tables --------------------------------------------------

# For each SKK okuri stem letter, list the (reading-suffix, surface-suffix,
# extra-cost) tuples to generate. Extra cost favors the citation form.
# Stems follow the godan/ichidan convention.
_CONJ_TABLE: dict[str, list[tuple[str, str, int]]] = {
    "r": [  # ichidan る verbs (食べる) and some godan-r (送る)
        ("る", "る", 0), ("た", "た", 20), ("て", "て", 20),
        ("ない", "ない", 30), ("ます", "ます", 30), ("よう", "よう", 50),
        ("れば", "れば", 60), ("られ", "られ", 70), ("らせ", "らせ", 70),
        ("たい", "たい", 40), ("たかった", "たかった", 60),
        ("ている", "ている", 40), ("ていた", "ていた", 50),
    ],
    "e": [  # potential form of ichidan (みえる from みr)
        ("える", "える", 0), ("えた", "えた", 20), ("えて", "えて", 20),
        ("えない", "えない", 30), ("えます", "えます", 30),
        ("えれば", "えれば", 50), ("えよう", "えよう", 50),
        ("えている", "えている", 40),
    ],
    "w": [  # certain う-verbs written as stem+う (買う)
        ("う", "う", 0), ("った", "った", 20), ("って", "って", 20),
        ("わない", "わない", 30), ("います", "います", 30), ("おう", "おう", 50),
        ("えば", "えば", 60), ("われ", "われ", 70), ("わせ", "わせ", 70),
        ("いたい", "いたい", 40),
    ],
    "k": [  # godan く (書く)
        ("く", "く", 0), ("いた", "いた", 20), ("いて", "いて", 20),
        ("かない", "かない", 30), ("きます", "きます", 30), ("こう", "こう", 50),
        ("けば", "けば", 60), ("かれ", "かれ", 70), ("かせ", "かせ", 70),
        ("きたい", "きたい", 40),
        # Potential-family forms. SKK also encodes several lexicalized
        # ichidan-looking compounds as k-stems, notably みつk/見付.
        # Generating the inflected け-* forms keeps 見付けて/見付けた
        # reachable instead of letting the Viterbi split みつけて as
        # [実][つけて].
        ("ける", "ける", 40), ("けた", "けた", 40), ("けて", "けて", 40),
        ("けない", "けない", 50), ("けます", "けます", 50),
        ("ければ", "ければ", 60), ("けよう", "けよう", 60),
        ("けている", "けている", 60), ("けていた", "けていた", 70),
    ],
    "g": [  # godan ぐ (泳ぐ)
        ("ぐ", "ぐ", 0), ("いだ", "いだ", 20), ("いで", "いで", 20),
        ("がない", "がない", 30), ("ぎます", "ぎます", 30), ("ごう", "ごう", 50),
        ("げば", "げば", 60), ("がれ", "がれ", 70), ("がせ", "がせ", 70),
        ("ぎたい", "ぎたい", 40), ("げる", "げる", 40),
    ],
    "s": [  # godan す (話す)
        ("す", "す", 0), ("した", "した", 20), ("して", "して", 20),
        ("さない", "さない", 30), ("します", "します", 30), ("そう", "そう", 50),
        ("せば", "せば", 60), ("され", "され", 70), ("させ", "させ", 70),
        ("したい", "したい", 40), ("せる", "せる", 40),
    ],
    "t": [  # godan つ (立つ)
        ("つ", "つ", 0), ("った", "った", 20), ("って", "って", 20),
        ("たない", "たない", 30), ("ちます", "ちます", 30), ("とう", "とう", 50),
        ("てば", "てば", 60), ("たれ", "たれ", 70), ("たせ", "たせ", 70),
        ("ちたい", "ちたい", 40), ("てる", "てる", 40),
    ],
    "n": [  # godan ぬ (死ぬ)
        ("ぬ", "ぬ", 0), ("んだ", "んだ", 20), ("んで", "んで", 20),
        ("なない", "なない", 30), ("にます", "にます", 30), ("のう", "のう", 50),
    ],
    "m": [  # godan む (読む)
        ("む", "む", 0), ("んだ", "んだ", 20), ("んで", "んで", 20),
        ("まない", "まない", 30), ("みます", "みます", 30), ("もう", "もう", 50),
        ("めば", "めば", 60), ("まれ", "まれ", 70), ("ませ", "ませ", 70),
        ("みたい", "みたい", 40), ("める", "める", 40),
    ],
    "b": [  # godan ぶ (遊ぶ)
        ("ぶ", "ぶ", 0), ("んだ", "んだ", 20), ("んで", "んで", 20),
        ("ばない", "ばない", 30), ("びます", "びます", 30), ("ぼう", "ぼう", 50),
        ("べば", "べば", 60), ("ばれ", "ばれ", 70), ("ばせ", "ばせ", 70),
        ("びたい", "びたい", 40), ("べる", "べる", 40),
    ],
    "i": [  # adjective い (高い)
        ("い", "い", 0), ("く", "く", 20), ("かった", "かった", 20),
        ("くない", "くない", 30), ("ければ", "ければ", 50), ("さ", "さ", 60),
        ("くて", "くて", 30), ("くなかった", "くなかった", 40),
        ("くなる", "くなる", 40), ("くなって", "くなって", 50),
        ("くなった", "くなった", 50),
    ],
}


def _conjugations(stem: str) -> list[tuple[str, str, int]]:
    """Return (reading-suffix, surface-suffix, cost-bump) for the stem letter."""
    return _CONJ_TABLE.get(stem, [("", "", 100)])
