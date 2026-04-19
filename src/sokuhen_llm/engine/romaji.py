"""Romaji -> hiragana converter.

Longest-match table-driven converter that handles sokuon (っ), hatsuon (ん),
common Hepburn/kunrei/wapuro variants, and partial input (for live conversion,
we need to preserve a trailing romaji fragment like "ko" so the UI can show
progress without committing).
"""
from __future__ import annotations

from dataclasses import dataclass


# Base syllable table. Keys are romaji, values are hiragana.
# Ordered so that longer keys are tried first in ``convert``.
_TABLE: dict[str, str] = {
    # vowels
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    # k-row
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ", "kye": "きぇ",
    # g-row
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ",
    # s-row
    "sa": "さ", "si": "し", "shi": "し", "su": "す", "se": "せ", "so": "そ",
    "sya": "しゃ", "syu": "しゅ", "syo": "しょ",
    "sha": "しゃ", "shu": "しゅ", "she": "しぇ", "sho": "しょ",
    # z-row
    "za": "ざ", "zi": "じ", "ji": "じ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "zya": "じゃ", "zyu": "じゅ", "zyo": "じょ",
    "ja": "じゃ", "ju": "じゅ", "je": "じぇ", "jo": "じょ",
    "jya": "じゃ", "jyu": "じゅ", "jyo": "じょ",
    # t-row
    "ta": "た", "ti": "ち", "chi": "ち", "tu": "つ", "tsu": "つ", "te": "て", "to": "と",
    "tya": "ちゃ", "tyu": "ちゅ", "tyo": "ちょ",
    "cha": "ちゃ", "chu": "ちゅ", "che": "ちぇ", "cho": "ちょ",
    "tsa": "つぁ", "tse": "つぇ", "tso": "つぉ",
    # d-row
    "da": "だ", "di": "ぢ", "du": "づ", "de": "で", "do": "ど",
    "dya": "ぢゃ", "dyu": "ぢゅ", "dyo": "ぢょ",
    # n-row
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "nya": "にゃ", "nyu": "にゅ", "nyo": "にょ",
    # h-row
    "ha": "は", "hi": "ひ", "hu": "ふ", "fu": "ふ", "he": "へ", "ho": "ほ",
    "hya": "ひゃ", "hyu": "ひゅ", "hyo": "ひょ",
    "fa": "ふぁ", "fi": "ふぃ", "fe": "ふぇ", "fo": "ふぉ",
    # b-row
    "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "bya": "びゃ", "byu": "びゅ", "byo": "びょ",
    # p-row
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
    "pya": "ぴゃ", "pyu": "ぴゅ", "pyo": "ぴょ",
    # m-row
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "mya": "みゃ", "myu": "みゅ", "myo": "みょ",
    # y-row
    "ya": "や", "yu": "ゆ", "yo": "よ", "ye": "いぇ",
    # r-row
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "rya": "りゃ", "ryu": "りゅ", "ryo": "りょ",
    # w-row
    "wa": "わ", "wi": "うぃ", "we": "うぇ", "wo": "を",
    # v / extended
    "va": "ゔぁ", "vi": "ゔぃ", "vu": "ゔ", "ve": "ゔぇ", "vo": "ゔぉ",
    # small kana (x-prefix / l-prefix)
    "xa": "ぁ", "xi": "ぃ", "xu": "ぅ", "xe": "ぇ", "xo": "ぉ",
    "la": "ぁ", "li": "ぃ", "lu": "ぅ", "le": "ぇ", "lo": "ぉ",
    "xya": "ゃ", "xyu": "ゅ", "xyo": "ょ",
    "lya": "ゃ", "lyu": "ゅ", "lyo": "ょ",
    "xtu": "っ", "xtsu": "っ", "ltu": "っ", "ltsu": "っ",
    "xn": "ん",
    # punctuation
    "-": "ー", ",": "、", ".": "。", "?": "？", "!": "！",
    "[": "「", "]": "」", "/": "・", " ": "　",
}

# Keys that, when a matching key isn't complete yet, could still grow into
# a match. Precomputed for the partial-input check.
_ALL_PREFIXES: frozenset[str] = frozenset(
    {k[:i] for k in _TABLE for i in range(1, len(k) + 1)}
)

# Sort keys by descending length for longest-match scanning.
_KEYS_BY_LEN: list[str] = sorted(_TABLE.keys(), key=len, reverse=True)


@dataclass(frozen=True)
class RomajiResult:
    """Output of ``RomajiConverter.convert``.

    ``kana`` is the converted hiragana portion. ``pending`` is the trailing
    romaji that could not yet be consumed — e.g. a lone "k" or "ky" waiting
    for its vowel. Live conversion shows ``kana + pending`` so the user sees
    their partial input.
    """

    kana: str
    pending: str


class RomajiConverter:
    """Stateless romaji -> hiragana conversion.

    Stateless because the composer keeps the raw romaji buffer; we reconvert
    the whole buffer on each keystroke. This keeps sokuon (double consonants)
    and ん handling consistent when the user backspaces.
    """

    __slots__ = ()

    def convert(self, romaji: str) -> RomajiResult:
        out: list[str] = []
        i = 0
        n = len(romaji)

        while i < n:
            ch = romaji[i]

            # Sokuon: double consonant (kk, tt, ss, ...) except n.
            if (
                i + 1 < n
                and ch == romaji[i + 1]
                and ch.isalpha()
                and ch not in "aiueon"
            ):
                out.append("っ")
                i += 1
                continue

            # 'n' special handling. 'n' alone may become ん depending on the
            # next character, but must not accidentally eat ni/nya/nn.
            if ch == "n":
                if i + 1 < n:
                    nxt = romaji[i + 1]
                    # "n'" -> explicit ん terminator, consume the apostrophe.
                    if nxt == "'":
                        out.append("ん")
                        i += 2
                        continue
                    # "n" followed by "n": decide based on what comes after
                    # the second 'n'. If a vowel or y follows (nna, nnya…),
                    # the user almost certainly meant ん + n[vowel] — i.e.,
                    # the second 'n' starts a new kana. Otherwise (nnc, nn
                    # at end, nnB) treat "nn" as an explicit ん terminator
                    # consuming both characters. This makes 'zannen' produce
                    # ざんねん (the natural spelling), while 'zannn' still
                    # produces just ざん.
                    if nxt == "n":
                        # "nn" is the user's explicit ん terminator — both
                        # chars are consumed. To type ん followed by ni/na/
                        # nya the convention is to use three n's in a row
                        # (sikennnokekka → しけんのけっか) or "n'" (si'kenno).
                        out.append("ん")
                        i += 2
                        continue
                    # "n" followed by a vowel or y: let longest-match fall
                    # through to handle 'ni', 'nya', etc.
                    if nxt in "aiueoy":
                        pass  # fall through to longest-match
                    else:
                        # Any other consonant/punctuation terminates as ん.
                        out.append("ん")
                        i += 1
                        continue
                else:
                    # 'n' at end: keep as pending so typing 'na' next works.
                    return RomajiResult(kana="".join(out), pending="n")

            matched = False
            for key in _KEYS_BY_LEN:
                kl = len(key)
                if kl > n - i:
                    continue
                if romaji[i : i + kl] == key:
                    out.append(_TABLE[key])
                    i += kl
                    matched = True
                    break

            if matched:
                continue

            # Could the remaining tail still become a valid key as the user types more?
            tail = romaji[i:]
            if tail in _ALL_PREFIXES:
                return RomajiResult(kana="".join(out), pending=tail)

            # Unknown character: pass through. Keeps punctuation and digits.
            out.append(ch)
            i += 1

        return RomajiResult(kana="".join(out), pending="")

    def finalize(self, romaji: str) -> str:
        """Convert and resolve any trailing pending tail.

        A lone "n" at the end becomes ん. Other partial tails (half-typed
        syllables like "k", "sh") pass through as raw romaji so the user
        can correct.
        """
        result = self.convert(romaji)
        if not result.pending:
            return result.kana
        if result.pending == "n":
            return result.kana + "ん"
        return result.kana + result.pending
