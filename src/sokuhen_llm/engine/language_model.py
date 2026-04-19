"""Lightweight bigram language model over dictionary surfaces.

We don't have a full mozc-style cost model, so we approximate:
- unigram cost derives from the dictionary's per-entry cost
- bigram cost is a small learned bonus/penalty between surface pairs that
  co-occur frequently in committed user text (see ``learning.py``)
- a fixed grammatical-rules table handles homophone disambiguation that
  the unigram cost alone can't (e.g. "を → 使用" vs "を → しよう")

The model is deliberately tiny and swappable. It's reasonable enough to
prefer common collocations ("今日 は", "する こと") once the user has
committed a few sentences, while starting from a sensible default on day one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# "Transition" cost when no bigram data exists. A bit higher than a confident
# match so that learned pairs can pull ahead; low enough that a fresh LM
# doesn't bias Viterbi significantly.
DEFAULT_BIGRAM_COST = 500

# A strong co-occurrence bonus (subtracted from cost). Capped so a single pair
# never completely dominates unigram costs.
MAX_BIGRAM_BONUS = 600


# --- grammatical rules table ------------------------------------------
#
# Hard-coded bigram adjustments for known-good / known-bad transitions the
# unigram cost can't capture. Applied on top of the learned bigram bonus;
# large negative values act as strong grammatical priors.
#
# Keep this list SMALL and well-motivated. Every entry should answer:
# "which homophone pair would the user always disambiguate this way?"
#
# Format: (prev_surface, curr_surface) -> delta-cost
#   Negative delta = bonus (more likely); positive = penalty (less likely).

_GRAMMATICAL_RULES: dict[tuple[str, str], int] = {
    # Volitional しよう almost never takes する/します/して directly after it.
    # If the user typed "...しよう+する", they almost certainly meant 使用+する.
    ("しよう", "する"): +1800,
    ("しよう", "します"): +1800,
    ("しよう", "した"): +1800,
    ("しよう", "して"): +1800,
    ("しよう", "しない"): +1800,
    ("しよう", "しました"): +1800,
    # Case particle + noun reading "使用" is a very common noun phrase.
    # Bonus pulls 使用 ahead of volitional しよう in this context.
    ("を", "使用"): -1200,
    ("に", "使用"): -1200,
    ("で", "使用"): -1200,
    ("が", "使用"): -1200,
    ("の", "使用"): -1200,
    # Same pattern for other Sino-Japanese homophones that overlap with
    # inflection kana.
    ("を", "購入"): -800,
    ("を", "利用"): -800,
    ("を", "作成"): -800,
    ("を", "受信"): -800,
    ("を", "送信"): -800,
    # Particle-governed noun disambiguation for readings shared by verbs:
    # "の 服" wins over "の 吹く" (same reading ふく but ~に is clearly a
    # noun phrase).
    ("の", "服"): -1000, ("の", "寿司"): -600,
    ("と", "寿司"): -800, ("と", "パン"): -600,
    # Common "…を飲みたい"
    ("を", "飲みたい"): -500, ("を", "食べたい"): -500,
    ("を", "見たい"): -500, ("を", "買いたい"): -500,
    # 〜て + みたい (auxiliary "like to try") — prefer kana over 見たい kanji
    ("買って", "みたい"): -800, ("食べて", "みたい"): -800,
    ("行って", "みたい"): -800, ("使って", "みたい"): -800,
    ("聞いて", "みたい"): -800, ("読んで", "みたい"): -800,
    # Opposite: after を, 見たい is the right verb-form (want to see X)
    ("を", "みたい"): +500,
    # 事故 vs 自己 — in "で/が/は 事故" context, 事故 wins
    ("で", "事故"): -1000, ("が", "事故"): -800, ("は", "事故"): -800,
    # News verbs after が (auxiliary subject)
    ("が", "発生"): -800, ("が", "増加"): -800, ("が", "減少"): -800,
    ("が", "続く"): -400, ("が", "始まった"): -600, ("が", "終わった"): -400,
    # Verb-ending た (past) — avoid 他 interpretation
    ("行われ", "た"): -600, ("開か", "れた"): -600,

    # --- Wikipedia-style disambiguations -------------------------------
    # 以降 in nominal contexts where Wikipedia overwhelmingly means
    # "after/since", not the volitional 行こう. Scope narrowly to
    # explicit trigger words so we don't over-bias.
    ("提案", "以降"): -1500, ("革命", "以降"): -1200,
    ("戦争", "以降"): -1200, ("戦後", "以降"): -1200,
    ("エイジ", "以降"): -1200, ("革新", "以降"): -1000,
    ("発足", "以降"): -1000, ("成立", "以降"): -1000,
    # 移行 after へ/への (「への移行」) — narrower than blanket の+以降.
    ("への", "移行"): -1200, ("社会への", "移行"): -1500,
    # 〜派 as a standalone noun rarely makes sense after a content noun
    # in running prose. Penalize that bigram so は particle wins.
    ("音楽", "派"): +2000, ("文学", "派"): +2000,
    ("物理", "派"): +2000, ("伝統", "派"): +2000,
    # と共に → とともに: in Wikipedia prose, とともに (hiragana) is natural.
    ("と", "共に"): +600,
    # 「使って」 vs 「遣って」 — prefer 使って
    ("を", "使って"): -500, ("を", "使った"): -500,
    # 「〜という」: after kanji noun, prefer という (hiragana)
    ("と", "いう"): -200,
    # 人間社会に みられる
    ("人間社会に", "みられる"): -400,
    ("人間社会に", "見られる"): -400,
    # 廃し (literary verb) after 体制を
    ("体制を", "廃し"): -800,
    # 「とされる」 pattern
    ("と", "される"): -300,
    # ～とも呼ばれる
    ("とも", "呼ばれる"): -400, ("とも", "呼ばれた"): -400,
    # 「〜の一部門」 / 「〜の一分野」
    ("の", "一部門"): -300, ("の", "一分野"): -300,
    # など → particle follow-ons
    ("など", "の"): -300, ("など", "と"): -300,
    ("など", "が"): -300, ("など", "を"): -300,
    ("など", "は"): -300, ("など", "に"): -300,
    # 以後 vs 以降 — 以降 wins in most Wikipedia contexts.
    ("提案", "以後"): +200,
    # 〜学者 bigram: after common discipline-学, prefer 者 over もの.
    # These fire when the reading is (...がくしゃ) but occasionally also
    # when pykakasi outputs (...がくもの). For the (もの) case we also
    # added compound entries above.
    ("物理学", "者"): -1500, ("英文学", "者"): -1500,
    ("情報工学", "者"): -1500, ("理論物理学", "者"): -1500,
    ("哲学", "者"): -1500, ("文学", "者"): -1200,
    ("科学", "者"): -1500, ("医学", "者"): -1500,
    # Also penalize もの after these
    ("物理学", "もの"): +1200, ("英文学", "もの"): +1200,
    ("情報工学", "もの"): +1200, ("理論物理学", "もの"): +1200,
    ("哲学", "もの"): +1200, ("文学", "もの"): +800,
    ("科学", "もの"): +1200, ("医学", "もの"): +1200,
    # 派 after 語 is almost never right (語派 is archaic linguistics term).
    # Penalize so 語 + は particle wins.
    ("語", "派"): +2500, ("語", "は"): -400,
    # 系 vs 計: in sports-statistics prose, 計 is more common.
    ("、", "計"): -300,
    # 一人 / 独り — the 一人 kanji form with reading ひとり is common.
    ("の", "一人"): -400, ("は", "一人"): -300,
    ("作曲家の", "一人"): -600,
    # と呼ばれる generic boost
    ("」", "とも"): -200,
    # Auxiliary 〜て + みる (try to): after て-form, prefer hiragana みた/みる
    # over kanji 見た/見る (already covered for 買って/食べて; extend).
    ("着て", "みた"): -800, ("着て", "みる"): -800, ("着て", "みて"): -800,
    ("寝て", "みた"): -600, ("起きて", "みた"): -600,
    ("来て", "みた"): -500,
    # 〜を + content noun vs verb disambiguation
    ("を", "服"): -500,   # 「服を」の逆の方向も含めて服優先
    ("を", "着"): -400, ("を", "着た"): -500, ("を", "着て"): -500,
    # 昨日 after common sentence starters
    ("は", "昨日"): -300, ("、", "昨日"): -300,
    # 機能 after 技術/システム-like terms
    ("システム", "機能"): -800, ("新しい", "機能"): -500,
    # 〜みたい vs 見た: 〜のみたい (like 〜) vs 〜の見た (past of 見る)
    ("の", "見た"): -300,  # 〜の見た… more common as past than auxiliary
    # 行く/来る/着る disambiguation by particle context:
    #   "〜に 行った"  →  prefer 行った (not 言った, not いった)
    #   "〜と 言った"  →  prefer 言った (the quotative particle と)
    #   "〜って 言った"  →  prefer 言った
    ("に", "行った"): -1500, ("に", "行って"): -1500, ("に", "行く"): -1500,
    ("へ", "行った"): -1500, ("へ", "行って"): -1500, ("へ", "行く"): -1500,
    ("を", "行った"): -600, ("を", "行って"): -600,  # "活動を行った"
    ("に", "言った"): +800, ("と", "言った"): -1200, ("って", "言った"): -1200,
    ("に", "いった"): +1200, ("と", "いった"): +400,
    ("を", "言った"): +600,  # "XXを言った" is odd; prefer 行った
    # 着る after を (を+着る = to wear)
    ("を", "着る"): -800, ("を", "着た"): -800,
    ("を", "着て"): -800, ("を", "着てみた"): -1000,
    ("服を", "着て"): -1200, ("服を", "着た"): -1200,
    # 来る in 場所-based context, 着る in 服-context
    ("家に", "来た"): -800, ("家に", "来て"): -800,
    ("日本に", "来た"): -800,
    # 変な
    ("、", "変な"): -300, ("だか", "変な"): -500,
    # 今日 vs こんにちは: when followed by content noun or particle-free
    # continuation, prefer 今日は (kanji)
    ("今日は", "映画"): -500,  # generic trigger: 今日は + content noun
    # 撮る vs 取る: in camera / photo context, prefer 撮る
    ("写真を", "撮る"): -600, ("写真を", "撮った"): -600,
    ("カメラで", "撮る"): -600, ("カメラで", "撮った"): -600,
    ("カメラで", "撮って"): -600,
    # 壊れる after が particle
    ("が", "壊れた"): -300, ("が", "壊れる"): -300,
    # バグ (tech loanword) after を (を直す, を修正する)
    ("を", "バグ"): -600,
}


def grammatical_delta(prev: str, curr: str) -> int:
    return _GRAMMATICAL_RULES.get((prev, curr), 0)


@dataclass
class LanguageModel:
    """Surface-to-surface bigram scores.

    ``bigram_counts[(prev, curr)]`` holds how many times the sequence has been
    confirmed by the user. ``unigram_counts[surface]`` holds raw frequency
    used to normalize.
    """

    bigram_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    unigram_counts: dict[str, int] = field(default_factory=dict)

    # BOS token used as the previous surface at the start of a sentence.
    BOS: str = "<s>"

    def observe(self, surfaces: list[str]) -> None:
        """Record a committed sequence. Updates both unigram and bigram counts."""
        prev = self.BOS
        for surf in surfaces:
            self.unigram_counts[surf] = self.unigram_counts.get(surf, 0) + 1
            key = (prev, surf)
            self.bigram_counts[key] = self.bigram_counts.get(key, 0) + 1
            prev = surf

    def transition_cost(self, prev: str, curr: str) -> int:
        """Cost of transitioning from ``prev`` to ``curr`` surface.

        Lower is better. Combines (a) a learned bigram bonus derived from
        the user's past commits with (b) a small grammatical-rules table
        for pre-trained disambiguation.
        """
        count = self.bigram_counts.get((prev, curr), 0)
        base = DEFAULT_BIGRAM_COST
        if count > 0:
            # prev_total must always be >= count (prev has appeared at least
            # as many times as the bigram). A bad save file could desync the
            # two tables; clamp defensively so a missing unigram doesn't
            # inflate the bonus to p≈1.
            prev_total = max(self.unigram_counts.get(prev, 0), count)
            prob = (count + 1) / (prev_total + 10)
            bonus = int(
                MAX_BIGRAM_BONUS * min(1.0, -math.log(max(1e-6, 1 - prob)) / 4.0)
            )
            base -= bonus
        base += grammatical_delta(prev, curr)
        return base

    def to_json(self) -> dict:
        return {
            "version": 1,
            "bigrams": [[p, c, n] for (p, c), n in self.bigram_counts.items()],
            "unigrams": list(self.unigram_counts.items()),
        }

    @classmethod
    def from_json(cls, data: dict) -> "LanguageModel":
        m = cls()
        for p, c, n in data.get("bigrams", []):
            m.bigram_counts[(p, c)] = int(n)
        for s, n in data.get("unigrams", []):
            m.unigram_counts[s] = int(n)
        return m
