"""
dialogue/answer_normalizer.py
-----------------------------
Parse and normalise a free-form human answer to a clarification question.

Replaces the V1 `observation_pipeline/answer_converter.py`, which had three
issues:
  1. Only converted the FIRST word of the answer (everything after the
     leading verb was passed through untouched), so adverbs and modifiers
     broke recipe-step style entirely.
  2. Verb inflection (-ing → 3sg-present) only handled a small set of
     overrides and mis-conjugated common cases like "mashing" → "mashs",
     "watching" → "watchs", "finishing" → "finishs".
  3. Negation handling was keyword-only and didn't surface what was
     negated, so the orchestrator couldn't route around the embedder's
     known weakness on negative sentences.

This module returns an `AnswerOutcome` with a typed classification of the
answer (AFFIRMATIVE / NEGATIVE / DOESNT_APPLY / DONT_KNOW / OFF_TOPIC).
The caller branches on the outcome's type and decides whether to update
the belief, skip silently, or apply a negation-aware update path.

The verb inflection uses a comprehensive cooking-verb override table plus
a fall-through that covers the standard English present-3sg suffix rules
(consonant + y → ies, -ss/-sh/-ch/-x/-z → +es, silent-e drop).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from probability.belief_updater_v3 import BeliefUpdaterV3


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

class AnswerType(str, Enum):
    """Categorical classification of a typed human answer."""

    AFFIRMATIVE = "affirmative"      # normal scene-style answer
    NEGATIVE = "negative"            # "no, I'm not using butter"
    DOESNT_APPLY = "doesnt_apply"    # "I'm not at that step yet"
    DONT_KNOW = "dont_know"          # "I don't know" / "no idea"
    OFF_TOPIC = "off_topic"          # unrelated to any active recipe


@dataclass
class AnswerOutcome:
    """
    What the normaliser produces from a raw human answer.

    Attributes
    ----------
    type
        The classified answer type. The orchestrator routes on this.
    raw_text
        The original input, stripped of whitespace and surrounding quotes.
    recipe_sentence
        The recipe-step form ("A cook ...") if conversion succeeded and
        the answer is AFFIRMATIVE. `None` otherwise.
    metadata
        Free-form details for the orchestrator. Currently used to surface
        the entity negated in a NEGATIVE answer.
    """

    type: AnswerType
    raw_text: str
    recipe_sentence: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def skip_update(self) -> bool:
        """
        True iff the orchestrator should NOT call bu.incorporate_answer
        for this outcome (and should not record IG_Q).

        DOESNT_APPLY, DONT_KNOW and OFF_TOPIC all leave the belief alone.
        NEGATIVE is a special case: handled separately by the orchestrator
        because V3's embedder doesn't represent negation reliably (see the
        orchestrator's negation branch).
        """
        return self.type in (
            AnswerType.DOESNT_APPLY,
            AnswerType.DONT_KNOW,
            AnswerType.OFF_TOPIC,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Verb inflection — -ing → 3sg-present
# ═══════════════════════════════════════════════════════════════════════════

# Cooking-specific overrides. Easier to extend than to debug edge cases in
# the regular-rule path. Covers ~95% of cooking-action verbs encountered in
# the recipe dataset.
_ING_TO_S = {
    "adding": "adds", "baking": "bakes", "beating": "beats",
    "blending": "blends", "boiling": "boils", "braising": "braises",
    "breaking": "breaks", "broiling": "broils", "browning": "browns",
    "brushing": "brushes", "buttering": "butters", "carrying": "carries",
    "catching": "catches", "charring": "chars", "checking": "checks",
    "chilling": "chills", "chopping": "chops", "coating": "coats",
    "coming": "comes", "combining": "combines", "cooking": "cooks",
    "cooling": "cools", "coring": "cores", "covering": "covers",
    "cracking": "cracks", "creaming": "creams", "crushing": "crushes",
    "cubing": "cubes", "cutting": "cuts", "deglazing": "deglazes",
    "dicing": "dices", "dipping": "dips", "dividing": "divides",
    "doing": "does", "draining": "drains", "drizzling": "drizzles",
    "dropping": "drops", "drying": "dries", "emulsifying": "emulsifies",
    "filling": "fills", "finishing": "finishes", "flipping": "flips",
    "folding": "folds", "frying": "fries", "garnishing": "garnishes",
    "glazing": "glazes", "going": "goes", "grating": "grates",
    "greasing": "greases", "grilling": "grills", "grinding": "grinds",
    "halving": "halves", "having": "has", "heating": "heats",
    "kneading": "kneads", "layering": "layers", "making": "makes",
    "marinating": "marinates", "mashing": "mashes", "measuring": "measures",
    "melting": "melts", "mincing": "minces", "mixing": "mixes",
    "peeling": "peels", "placing": "places", "plating": "plates",
    "poaching": "poaches", "pouring": "pours", "preheating": "preheats",
    "preparing": "prepares", "pressing": "presses", "putting": "puts",
    "reducing": "reduces", "removing": "removes", "rendering": "renders",
    "resting": "rests", "rinsing": "rinses", "roasting": "roasts",
    "rolling": "rolls", "sauteing": "sautees", "sautéing": "sautées",
    "scoring": "scores", "scraping": "scrapes", "searing": "sears",
    "seasoning": "seasons", "separating": "separates", "serving": "serves",
    "shaking": "shakes", "shaving": "shaves", "simmering": "simmers",
    "slicing": "slices", "smoking": "smokes", "soaking": "soaks",
    "spreading": "spreads", "sprinkling": "sprinkles", "squeezing": "squeezes",
    "steaming": "steams", "stewing": "stews", "stirring": "stirs",
    "straining": "strains", "stuffing": "stuffs", "taking": "takes",
    "tasting": "tastes", "tearing": "tears", "thickening": "thickens",
    "tipping": "tips", "topping": "tops", "tossing": "tosses",
    "transferring": "transfers", "trimming": "trims", "turning": "turns",
    "twisting": "twists", "using": "uses", "washing": "washes",
    "watching": "watches", "whipping": "whips", "whisking": "whisks",
    "wrapping": "wraps", "zesting": "zests",
    # Auxiliary / state verbs that might show up in conversational answers
    "being": "is", "putting in": "puts in",
}

_VOWELS = set("aeiou")


def _inflect_ing_to_3sg(word: str) -> str:
    """
    Convert an English -ing form to its 3rd-person-singular present.

    Honours the cooking-verb override table first. The fallback rule
    covers:
        - consonant + y      → ies   (carrying → carries)
        - -ss / -sh / -ch / -x / -z → +es  (watching → watches)
        - doubled consonant  → strip then +s  (putting → puts)
        - default            → +s             (cooking → cooks)

    Silent-e cases (slicing → slices, making → makes) are caught by the
    override table — the rule can't recover them without a lexicon.
    """
    lower = word.lower()
    if lower in _ING_TO_S:
        return _ING_TO_S[lower]
    if not lower.endswith("ing") or len(lower) <= 4:
        return lower  # not an -ing form; pass through

    stem = lower[:-3]

    # consonant + y → ies (frying → fries)
    if len(stem) >= 2 and stem[-1] == "y" and stem[-2] not in _VOWELS:
        return stem[:-1] + "ies"

    # doubled consonant before -ing → strip the doubled letter
    # (putting → put, swimming → swim, running → run)
    if (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and stem[-1] not in _VOWELS
        and stem[-1] not in ("l", "s")  # exclude common false positives
    ):
        stem = stem[:-1]

    # -ss / -sh / -ch / -x / -z → +es (watching → watches)
    if stem.endswith(("ss", "sh", "ch", "x", "z")):
        return stem + "es"

    return stem + "s"


# ═══════════════════════════════════════════════════════════════════════════
# Sentence-level conversion
# ═══════════════════════════════════════════════════════════════════════════

_FIRST_PERSON_PREFIXES = [
    # ORDER MATTERS: longest / most-specific patterns FIRST so the
    # multi-word future "I'm going to" isn't shortcut by the present
    # continuous "I'm" before it.
    #
    # Future / intent: "I'm going to grate", "I'm gonna grate",
    # "I will grate", "I'll grate", "we are going to grate".
    # Strips back to a bare-stem verb ("grate"), picked up by _BARE_TO_3SG.
    re.compile(r"^(?:i'm going to|i am going to|i'm gonna|i am gonna|we're going to|we are going to)\s+", re.IGNORECASE),
    re.compile(r"^(?:i will|i'll|we will|we'll)\s+", re.IGNORECASE),
    # Present continuous: "I am cracking", "I'm cracking", "we're cracking".
    re.compile(r"^(?:i am|i'm|im|we are|we're)\s+", re.IGNORECASE),
    re.compile(r"^(?:the cook will|the cook is|the cook's)\s+", re.IGNORECASE),
    re.compile(r"^(?:right now|currently|just|now|next)\s+", re.IGNORECASE),
]

# Filler / hedging words that can appear before the verb after stripping
# "I am". We strip these so the verb sits at the head of the sentence.
_LEADING_HEDGES = {
    "just", "still", "now", "currently", "carefully", "slowly", "quickly",
    "finally", "actually", "really", "basically", "kind", "kinda", "sort",
    "right", "about",
}

# Words that END in -ing but are NOT verb-progressives. The naive regex
# would otherwise treat them as cooking verbs and produce nonsense ("a cook
# noths", "a cook things", etc).
_NON_VERB_ING = {
    "nothing", "something", "anything", "everything", "thing",
    "morning", "evening",                       # nouns
    "during", "spring", "ceiling", "string",    # not verb-stems
    "king", "ring", "wing", "ling",
}

_ING_RE = re.compile(r"\b([A-Za-zé]+ing)\b")


def _find_action_ing(text: str) -> re.Match | None:
    """
    Find the first -ing token in `text` that is actually a verb (skip
    nouns like 'nothing', 'something', etc).
    """
    for m in _ING_RE.finditer(text):
        if m.group(1).lower() not in _NON_VERB_ING:
            return m
    return None

# Bare-stem verbs (what comes after "I will" / "I'll"). Derived from the
# gerund table by stripping the -ing and looking up the 3sg form. Used as
# a fallback when no -ing verb is found in the sentence — handles future-
# tense and command-style answers like "I will grate pecorino" or
# "grate the pecorino".
_BARE_TO_3SG: dict[str, str] = {
    # Build from _ING_TO_S by stripping -ing from the keys.
    word[:-3] if word.endswith("ing") else word: tsg
    for word, tsg in _ING_TO_S.items()
    if word.endswith("ing") and len(word) > 4
}
# Add a handful of common silent-e bare stems that get lost in the
# strip-ing transformation above (e.g. "dicing" → "dic" → should be "dice").
_BARE_TO_3SG.update({
    "add": "adds", "bake": "bakes", "beat": "beats", "blend": "blends",
    "boil": "boils", "break": "breaks", "brown": "browns", "brush": "brushes",
    "char": "chars", "check": "checks", "chill": "chills", "chop": "chops",
    "coat": "coats", "combine": "combines", "cook": "cooks", "cool": "cools",
    "cover": "covers", "crack": "cracks", "cream": "creams", "crush": "crushes",
    "cube": "cubes", "cut": "cuts", "deglaze": "deglazes", "dice": "dices",
    "dip": "dips", "divide": "divides", "drain": "drains", "drizzle": "drizzles",
    "drop": "drops", "dry": "dries", "fill": "fills", "finish": "finishes",
    "flip": "flips", "fold": "folds", "fry": "fries", "garnish": "garnishes",
    "glaze": "glazes", "grate": "grates", "grease": "greases", "grill": "grills",
    "grind": "grinds", "halve": "halves", "heat": "heats", "knead": "kneads",
    "layer": "layers", "make": "makes", "marinate": "marinates", "mash": "mashes",
    "measure": "measures", "melt": "melts", "mince": "minces", "mix": "mixes",
    "peel": "peels", "place": "places", "plate": "plates", "poach": "poaches",
    "pour": "pours", "preheat": "preheats", "prepare": "prepares",
    "press": "presses", "put": "puts", "reduce": "reduces", "remove": "removes",
    "render": "renders", "rest": "rests", "rinse": "rinses", "roast": "roasts",
    "roll": "rolls", "saute": "sautees", "sauté": "sautées",
    "score": "scores", "scrape": "scrapes", "sear": "sears", "season": "seasons",
    "separate": "separates", "serve": "serves", "shake": "shakes",
    "shave": "shaves", "simmer": "simmers", "slice": "slices", "smoke": "smokes",
    "soak": "soaks", "spread": "spreads", "sprinkle": "sprinkles",
    "squeeze": "squeezes", "steam": "steams", "stew": "stews", "stir": "stirs",
    "strain": "strains", "stuff": "stuffs", "take": "takes", "taste": "tastes",
    "tear": "tears", "thicken": "thickens", "tip": "tips", "top": "tops",
    "toss": "tosses", "transfer": "transfers", "trim": "trims", "turn": "turns",
    "twist": "twists", "use": "uses", "wash": "washes", "watch": "watches",
    "whip": "whips", "whisk": "whisks", "wrap": "wraps", "zest": "zests",
})


def to_recipe_sentence(answer: str) -> str | None:
    """
    Convert a free-form first-person / present-continuous answer into the
    recipe-step "A cook ..." form used by the rest of the pipeline.

    Returns None if the input is empty or no progressive verb can be
    located in it (in which case the caller treats it as off-topic or
    not-applicable).

    Strategy
    --------
    1. Strip wrapping quotes / whitespace.
    2. Pass-through if already in recipe form ("A cook ...").
    3. Strip first-person prefixes ("I am", "I'm", "we are", ...).
    4. Strip leading filler words ("just", "currently", ...).
    5. Find the first -ing verb anywhere in what's left and inflect it.
       The verb does NOT have to be the first word — adverbs in front
       are tolerated.
    6. Rebuild as "A cook <verb> <rest of sentence>".
    """
    if not answer:
        return None

    text = answer.strip().strip('"').strip("'").strip()
    if not text:
        return None

    # Already in recipe form
    if re.match(r"^a cook\s", text, flags=re.IGNORECASE):
        return _ensure_period(text[0].upper() + text[1:])

    # Strip first-person / framing prefixes
    for pattern in _FIRST_PERSON_PREFIXES:
        text = pattern.sub("", text, count=1)
    text = text.strip()
    if not text:
        return None

    # Strip leading hedges/fillers ("just rolling..." → "rolling...")
    while True:
        head = text.split(None, 1)
        if len(head) < 2 or head[0].lower() not in _LEADING_HEDGES:
            break
        text = head[1]

    # Locate first -ing VERB anywhere in remaining text (skipping nouns
    # like 'nothing' / 'something' that share the suffix).
    match = _find_action_ing(text)
    verb_3sg: str | None = None
    before = after = ""

    if match:
        verb_ing = match.group(1)
        before = text[: match.start()].rstrip()
        after = text[match.end() :].lstrip()
        verb_3sg = _inflect_ing_to_3sg(verb_ing)
    else:
        # No -ing form anywhere. Fall back to a bare-stem verb at the head
        # of the sentence — handles future-tense and command-style answers
        # like "I will grate pecorino" or "grate the pecorino" (the
        # "I will" prefix has already been stripped above).
        head = text.split(None, 1)
        if head:
            candidate = head[0].lower().rstrip(".,!?")
            if candidate in _BARE_TO_3SG:
                verb_3sg = _BARE_TO_3SG[candidate]
                after = head[1] if len(head) > 1 else ""

    if verb_3sg is None:
        return None  # caller decides — likely OFF_TOPIC or DOESNT_APPLY

    # Discard pre-verb noise — it's adverbs/fillers we couldn't strip.
    # The conversion focuses on action+object, which is what the embedder
    # uses anyway.
    rebuilt = f"A cook {verb_3sg}"
    if after:
        rebuilt += f" {after}"
    elif before:
        # Verb was at the tail — keep "A cook <verb>" and append residue
        # in case it's a meaningful object that came before the verb form.
        rebuilt += f" {before}"

    return _ensure_period(rebuilt[0].upper() + rebuilt[1:])


def _ensure_period(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text.endswith((".", "!", "?")) else text + "."


# ═══════════════════════════════════════════════════════════════════════════
# Answer-type classification
# ═══════════════════════════════════════════════════════════════════════════

# Patterns checked in priority order. First match wins.
_DOESNT_APPLY_PATTERNS = [
    re.compile(r"doesn'?t apply", re.IGNORECASE),
    re.compile(r"\b(not at|haven'?t (?:gotten|reached|got)|not (?:yet|there))\b", re.IGNORECASE),
    re.compile(r"\bnot (?:that|this) step\b", re.IGNORECASE),
    re.compile(r"\bskip(?:ping)? this (?:one|question)\b", re.IGNORECASE),
    # "nothing" / "none" / "nothing yet" — natural reply when the question's
    # topic isn't happening right now (e.g. cook is just pouring water and
    # the question asks "What is being cooked in the skillet?").
    re.compile(r"^\s*(?:nothing|none)\s*(?:yet|so far)?\s*[.!?]?\s*$", re.IGNORECASE),
]

_DONT_KNOW_PATTERNS = [
    re.compile(r"\b(?:i )?don'?t know\b", re.IGNORECASE),
    re.compile(r"\bno idea\b", re.IGNORECASE),
    re.compile(r"\bnot sure\b", re.IGNORECASE),
    re.compile(r"\bunclear\b", re.IGNORECASE),
    re.compile(r"^\s*skip\s*$", re.IGNORECASE),
    re.compile(r"^\?+$"),
]

_NEGATIVE_PATTERNS = [
    re.compile(r"^\s*no\b[,.!\s]", re.IGNORECASE),
    re.compile(r"^\s*nope\b", re.IGNORECASE),
    re.compile(r"\b(?:i'?m|i am|we'?re|we are)\s+not\b", re.IGNORECASE),
    re.compile(r"\b(?:don'?t|doesn'?t|do not|does not|didn'?t|did not)\s+(?:use|add|include|have|need|put)\b", re.IGNORECASE),
    re.compile(r"\b(?:never|without|no\s+(?:butter|cream|herbs|garlic|cheese|oil|sauce|pancetta|bacon))\b", re.IGNORECASE),
    re.compile(r"\b(?:skip(?:ping)? the|replac(?:e|ing) [a-z]+ with)\b", re.IGNORECASE),
]


def _classify_answer_type(text: str) -> AnswerType | None:
    """
    Classify by text patterns. Returns None for AFFIRMATIVE / unknown so
    the relevance gate can have the final say.
    """
    for pat in _DOESNT_APPLY_PATTERNS:
        if pat.search(text):
            return AnswerType.DOESNT_APPLY
    for pat in _DONT_KNOW_PATTERNS:
        if pat.search(text):
            return AnswerType.DONT_KNOW
    for pat in _NEGATIVE_PATTERNS:
        if pat.search(text):
            return AnswerType.NEGATIVE
    return None


# Common negation entities to surface for the orchestrator's negation
# handler. Extending this list is cheap; it just changes how the
# orchestrator reports the negation.
_NEGATABLE_ENTITIES = (
    "butter", "cream", "milk", "yogurt",
    "garlic", "onion", "shallot",
    "bacon", "pancetta", "guanciale", "sausage", "chicken", "shrimp",
    "parmesan", "pecorino", "mozzarella", "ricotta", "cheese",
    "tomato", "tomatoes", "basil", "parsley", "oregano", "thyme",
    "olive oil", "butter oil", "sunflower oil",
    "wine", "vinegar", "lemon",
    "herbs", "chili", "pepper", "salt",
)


def _extract_negated_entity(text: str) -> str | None:
    lowered = text.lower()
    for entity in _NEGATABLE_ENTITIES:
        # Look for the entity inside a negation context (5-token window
        # after a negation marker).
        for marker in ("no ", "not ", "without ", "skip ", "don't ", "doesn't "):
            idx = lowered.find(marker)
            if idx < 0:
                continue
            window = lowered[idx : idx + 80]
            if entity in window:
                return entity
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Polar (yes/no) classification
# ═══════════════════════════════════════════════════════════════════════════

# Patterns checked first — these take priority over the affirmative/negative
# split below because "doesn't apply" answers can otherwise be misread as
# "no" by the negation patterns.
_POLAR_YES = re.compile(
    r"^\s*(?:y|yes|yep|yeah|yup|sure|correct|right|true|aff|affirmative)\b[!.\s,]*",
    re.IGNORECASE,
)
_POLAR_NO = re.compile(
    r"^\s*(?:n|no|nope|nah|not\s+really|not\s+at\s+all|negative|wrong|incorrect|false|never)\b[!.\s,]*",
    re.IGNORECASE,
)


def classify_polar_answer(raw_answer: str) -> AnswerOutcome:
    """
    Classify a typed yes/no answer for the polar condition.

    Returns an `AnswerOutcome` with one of:
      - AFFIRMATIVE  — recipe_sentence holds the literal "yes" token.
      - NEGATIVE     — recipe_sentence holds the literal "no" token.
      - DOESNT_APPLY — the cook hasn't reached this step.
      - DONT_KNOW    — empty / "skip" / "don't know".
      - OFF_TOPIC    — input couldn't be parsed as any of the above.

    The orchestrator branches on `outcome.type`:
      - AFFIRMATIVE → `bu.incorporate_answer(proposition)`
      - NEGATIVE    → `bu.incorporate_negative_answer(proposition)`
      - everything else → skip silently (no belief update, no IG_Q logged).

    Symmetric with `normalize_answer` for the wh condition so both
    orchestrators have a single shape for human-answer routing.
    """
    raw = (raw_answer or "").strip().strip('"').strip("'").strip()

    if not raw:
        return AnswerOutcome(type=AnswerType.DONT_KNOW, raw_text="")

    # Pre-screen for the "I'm not at that step" and "I don't know" cases.
    # These must come before the yes/no split because "I'm not at that step"
    # starts with a word the _POLAR_NO pattern would otherwise catch.
    for pat in _DOESNT_APPLY_PATTERNS:
        if pat.search(raw):
            return AnswerOutcome(type=AnswerType.DOESNT_APPLY, raw_text=raw)
    for pat in _DONT_KNOW_PATTERNS:
        if pat.search(raw):
            return AnswerOutcome(type=AnswerType.DONT_KNOW, raw_text=raw)

    if _POLAR_YES.match(raw):
        return AnswerOutcome(
            type=AnswerType.AFFIRMATIVE,
            raw_text=raw,
            recipe_sentence="yes",
        )
    if _POLAR_NO.match(raw):
        return AnswerOutcome(
            type=AnswerType.NEGATIVE,
            raw_text=raw,
            recipe_sentence="no",
        )

    # The cook typed a full sentence instead of yes/no. Best-effort: scan
    # for any leading negation marker so e.g. "without cream" is read as
    # NEGATIVE rather than off-topic.
    leading_negation = _classify_answer_type(raw)
    if leading_negation == AnswerType.NEGATIVE:
        return AnswerOutcome(
            type=AnswerType.NEGATIVE,
            raw_text=raw,
            recipe_sentence="no",
        )

    return AnswerOutcome(type=AnswerType.OFF_TOPIC, raw_text=raw)


# ═══════════════════════════════════════════════════════════════════════════
# Relevance gate (off-topic detection)
# ═══════════════════════════════════════════════════════════════════════════

# Minimum cosine between the answer and any recipe's unseen near-future
# scenes for the answer to count as on-topic. Below this the answer is
# treated as OFF_TOPIC and skipped. The gate's job is to catch truly
# unrelated input ("my cat jumped on the counter"), not to enforce
# alignment with what we currently believe.
RELEVANCE_THRESHOLD = 0.25

# How far ahead to look in each recipe's remaining scenes when checking
# topical relevance.
RELEVANCE_HORIZON = 5


def _max_relevance_cosine(
    candidate: str,
    belief_updater: "BeliefUpdaterV3",
) -> float:
    """
    Max cosine between `candidate` and the near-future scenes of ANY
    recipe in the database — not just the currently-active subset.

    Why all recipes, not just the active ones: the relevance gate is
    supposed to ask "is this a sensible cooking sentence?", not "does it
    match the recipes we currently think are likely?". If we filter by
    current belief, we throw away exactly the answers that could RESCUE
    a recipe from low probability — which is the whole point of letting
    the cook clarify. A perfect answer for pesto pasta got dismissed as
    off-topic in a session where pesto pasta's belief had dropped below
    the active threshold; scanning all recipes prevents that failure
    mode. The cost is ~29 small embedding lookups instead of ~5; the
    embedder caches everything so it's effectively free.
    """
    cand_vec = belief_updater._embedder.embed(candidate)
    cand_norm = float(np.linalg.norm(cand_vec))
    if cand_norm == 0.0:
        return 0.0

    best = 0.0
    for name in belief_updater.belief.keys():
        scenes = belief_updater.unseen_recipe_scenes(name)[:RELEVANCE_HORIZON]
        if not scenes:
            continue
        scene_vecs = belief_updater._embedder.embed_batch(scenes)
        for vec in scene_vecs:
            n = float(np.linalg.norm(vec))
            if n == 0.0:
                continue
            cos = float(np.dot(cand_vec, vec) / (cand_norm * n))
            if cos > best:
                best = cos
    return best


# ═══════════════════════════════════════════════════════════════════════════
# Top-level entry point
# ═══════════════════════════════════════════════════════════════════════════

def normalize_answer(
    raw_answer: str,
    belief_updater: "BeliefUpdaterV3 | None" = None,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> AnswerOutcome:
    """
    Classify and normalise a free-form human answer.

    Parameters
    ----------
    raw_answer
        The text typed by the human (or returned by the stub).
    belief_updater
        Optional. When provided, the relevance gate uses it to check
        whether the answer is topical with respect to active recipes.
        Pass `None` to disable the gate (useful for unit tests).
    relevance_threshold
        Cosine threshold for the relevance gate. Below this the answer
        is classified OFF_TOPIC.

    Returns
    -------
    AnswerOutcome
        See class docstring. The caller branches on `outcome.type` and
        consults `outcome.skip_update` to decide whether to update the
        belief.
    """
    raw = (raw_answer or "").strip().strip('"').strip("'").strip()

    if not raw:
        return AnswerOutcome(
            type=AnswerType.DONT_KNOW,
            raw_text="",
        )

    # 1) Try pattern-based classification first.
    pre_type = _classify_answer_type(raw)
    if pre_type == AnswerType.DOESNT_APPLY:
        return AnswerOutcome(type=AnswerType.DOESNT_APPLY, raw_text=raw)
    if pre_type == AnswerType.DONT_KNOW:
        return AnswerOutcome(type=AnswerType.DONT_KNOW, raw_text=raw)
    if pre_type == AnswerType.NEGATIVE:
        negated = _extract_negated_entity(raw)
        return AnswerOutcome(
            type=AnswerType.NEGATIVE,
            raw_text=raw,
            recipe_sentence=None,
            metadata={"negated_entity": negated} if negated else {},
        )

    # 2) Try to convert to recipe-step style.
    recipe = to_recipe_sentence(raw)
    if recipe is None:
        # No -ing verb anywhere we could lock onto. Best guess: the cook
        # is saying something we don't know how to lift into recipe form.
        # Treat as off-topic; caller will skip the belief update.
        return AnswerOutcome(type=AnswerType.OFF_TOPIC, raw_text=raw)

    # 3) Optional relevance gate against the active belief.
    if belief_updater is not None:
        rel = _max_relevance_cosine(recipe, belief_updater)
        if rel < relevance_threshold:
            return AnswerOutcome(
                type=AnswerType.OFF_TOPIC,
                raw_text=raw,
                recipe_sentence=recipe,  # keep for debugging / logging
                metadata={"max_relevance_cosine": round(rel, 4)},
            )

    return AnswerOutcome(
        type=AnswerType.AFFIRMATIVE,
        raw_text=raw,
        recipe_sentence=recipe,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":  # pragma: no cover
    samples = [
        "I'm cracking eggs into a bowl",
        "I am carefully draining the pasta over the sink",
        "Just mashing the garlic into a paste",
        "A cook grates pecorino into the bowl.",
        "No, I'm not using butter — just olive oil",
        "Without cream, just pancetta and eggs",
        "I don't know yet",
        "That doesn't apply yet — I'm not at that step",
        "My cat just jumped on the counter",          # off-topic
        "I'm whisking the eggs",
        "I am slicing the onion thinly",
        "currently watching the pasta boil",
        "",                                            # empty
    ]
    for s in samples:
        out = normalize_answer(s, belief_updater=None)
        print(f"{out.type.value:<14} | recipe={out.recipe_sentence!r:<55} | raw={s!r}")
