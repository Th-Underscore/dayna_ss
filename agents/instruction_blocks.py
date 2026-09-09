"""Instruction-block engine: sentence splitting, cross-turn repetition
guarding (similarity / phrase-overlap / property collision), recent-ring
bookkeeping, and small context blocks used by instruction generation.

Extracted verbatim from agents/summarizer.py (T3 refactor); pure functions,
no engine dependencies.
"""
from __future__ import annotations

from collections import deque
import difflib
import re


# post-check and instruction-generation paths.
_SENT_SPLIT_RE = re.compile(
    r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!St\.)(?<!Prof\.)(?<!Rev\.)(?<!Sen\.)"
    r"(?<!Rep\.)(?<!Gov\.)(?<!Jr\.)(?<!Sr\.)(?<!No\.)(?<!Vol\.)"
    r"(?<=[.!?])\s+|\n+"
)

# Cross-turn instruction-block repetition guard. A recycled instruction block
# is the direct source of recycled replies (the transcription loop), so the
# engine refuses to feed the model a block that is near-identical to the
# previous turn's. Deterministic, no extra inference unless it fires.
_INSTR_SIM_THRESHOLD = 0.85
_INSTR_ANTI_REPEAT_DIRECTIVE = (
    "\n\nIMPORTANT: The instruction block you just produced repeats the previous turn's imagery or phrasing. "
    "Produce a FRESH set of instructions for the CURRENT scene and the LATEST exchange: different beats, "
    "different concrete actions, different objects and locations. Do not reuse the previous block's props, "
    "imagery, phrasing, or opening gestures, and do not re-list the same three beats."
)


def _instructions_similar(a: str, b: str, threshold: float = _INSTR_SIM_THRESHOLD) -> bool:
    """Deterministic cross-turn similarity of two instruction blocks.

    Mirrors the reply-side hard rules ('Do not repeat imagery or a closing
    sentence/ending gesture you have already used in earlier replies'): if the
    instruction blocks collide, the generated reply transcribes them verbatim.
    """
    if not a or not b:
        return False
    na = re.sub(r"[^a-z0-9 ]", "", a.lower()).strip()
    nb = re.sub(r"[^a-z0-9 ]", "", b.lower()).strip()
    if len(na) < 80 or len(nb) < 80:
        return False
    # autojunk=False: the default autojunk heuristics treat any frequent
    # subword as garbage, deflating the ratio for re-lexicalized prose
    # (the loop's blocks re-used a whole cypress paragraph yet scored 0.75).
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio() > threshold


# Chars of verbatim overlap that counts as re-transcribing a whole beat/phrase
# from the previous turn's block. Calibrated on 14714d8e: the loop's blocks
# sat at byte-ratio ~0.08 (the guard above never fired) yet re-used 100-870
# chars verbatim; the healthy dded7733 run peaked at ~178 on one turn.
_INSTR_PHRASE_OVERLAP = 120


def _instructions_phrase_overlap(a: str, b: str, min_lcs: int = _INSTR_PHRASE_OVERLAP) -> bool:
    """True if the two blocks share a long verbatim run (>= min_lcs chars).

    The byte-ratio guard catches near-identical blocks; this catches blocks
    whose overall wording differs but which re-transcribe whole phrases from
    the previous turn (the semantic-repetition loop).
    """
    if not a or not b:
        return False
    na = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", a.lower())).strip()
    nb = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", b.lower())).strip()
    if len(na) < min_lcs or len(nb) < min_lcs:
        return False
    sm = difflib.SequenceMatcher(None, na, nb, autojunk=False)
    return max((m.size for m in sm.get_matching_blocks()), default=0) >= min_lcs


# ---- Recently-used imagery tracking (P1/P2: semantic-repetition loop) ----
# The instruction generator cannot satisfy "weave in stored items" (req 9) and
# "never re-anchor on stored objects" (req 14) at once because nothing
# enumerates which stored objects earlier replies already used. These helpers
# track the recent instruction blocks' props and surface them to the generator.
_INSTR_RING_MAXLEN = 8
# Hard-reject regeneration budget for the cross-turn instruction guard: up to
# this many DETERMINISTIC regens with the offending props named, keeping the
# least-colliding candidate; bounded so a pathological generator costs at most
# _INSTR_REGEN_MAX+1 calls instead of looping.
_INSTR_REGEN_MAX = 2
_PROP_PHRASE_BAD_WORDS = frozenset({
    "of", "the", "a", "an", "and", "with", "in", "on", "at", "into", "to", "for",
})
_PROP_STOPWORDS = frozenset({
    "into", "onto", "from", "with", "under", "beside", "around", "behind",
    "across", "along", "toward", "towards", "through", "against", "then",
    "now", "soon", "slowly", "quickly", "gently", "quietly", "again", "she",
    "he", "them", "him", "herself", "himself", "ahead", "aside", "forward",
})
# Short noun phrase trailing a common instruction verb (catches props not yet
# stored in the elements map). The primary layer is the known-name vocabulary
# match below; this regex is deliberately conservative.
_INSTR_VERB_RE = re.compile(
    r"(?:snip|place|open|examine|take|lift|pick|put|set|hold|carry|weigh|trace|"
    r"unfold|read|adjust|grip|slip|tuck|straighten|tap|stroke|pour|spread|fold|"
    r"press|reach|turn|hand|pass|fetch|tie|untie|fold|tuck)"
    r"\s+(?:a|an|the|your|her|his|fresh|dried|grey|gray|white|small|large|old|"
    r"new|empty|heavy|light|tiny|wooden|iron|ceramic|glass|linen|fine|torn|"
    r"folded|wrapped|pressed|single)\s+([a-z]+(?:\s+[a-z]+){0,2})",
    re.I,
)


def _block_collides(
    block: str,
    ring: list[tuple[str, set[str], str]],
    known_names: list[str] | None,
    prop_cluster: int = 2,
    persistent_min: int = 3,
) -> tuple[bool, set[str]]:
    """Cross-turn collision of a fresh instruction block against the WHOLE ring.

    The variables that follow the current block on a 5-6 turn cadence (the
    repetition loop) sit positions 3-8 in the ring, so the guard must scan all
    of it, not just the previous block. Three signals:
      1. byte-ratio similarity (autojunk=False) to ANY ring block;
      2. a long verbatim run (>= 120 chars) shared with ANY ring block;
      3. a re-anchoring CLUSTER: the block shares >= 2 props with ANY single
         ring block (not the union — a healthy block may borrow one prop from
         each of several blocks, but re-drawing 2+ props from one block is the
         re-lexicalized re-anchor the byte/LCS guards cannot see);
      4. a PERSISTENT prop: a prop that >= `persistent_min` ring blocks have
         anchored on reappears in the new block (the single-prop re-anchor,
         e.g. the same pruning knife for 5 straight blocks).
    Returns (collides, offending_props) where offending_props is the set of
    named props shared with any colliding ring block (used to make rejection
    specific, per the t28 lesson that name-less regeneration cannot de-loop).
    """
    if not block or not ring:
        return (False, set())
    new_props = _extract_props(block, known_names)
    block_props: list[set[str]] = []
    offending: set[str] = set()
    for entry in ring:
        bprops = entry[1] if len(entry) >= 2 else set()
        if not isinstance(bprops, (set, frozenset)):
            bprops = set()
        block_props.append(bprops)
        share = new_props & bprops
        if share:
            offending |= share
    # 4) persistent prop across the ring
    from collections import Counter
    counts = Counter(p for bprops in block_props for p in bprops)
    persistent = {p for p, c in counts.items() if c >= persistent_min}
    if new_props & persistent:
        return (True, offending)
    # 3) cluster with any single ring block
    for bprops in block_props:
        if len(new_props & bprops) >= prop_cluster:
            return (True, offending)
    # 1+2) byte/LCS similarity to any ring block's text
    for entry in ring:
        prev_text = entry[2] if len(entry) >= 3 else None
        if prev_text and (_instructions_similar(block, prev_text) or _instructions_phrase_overlap(block, prev_text)):
            return (True, offending)
    return (False, offending)


def _ring_blocks(
    recent: deque, prev_instruction: str | None,
) -> list[tuple[str, set[str], str]]:
    """The full collision ring: every recent block (label, props, TEXT), plus
    the previous turn's block explicitly (it is the newest entry on the
    generating path, but on a resumed run the in-memory deque may be empty for
    the first generation)."""
    ring = list(recent) if recent else []
    if prev_instruction:
        prev_props = _extract_props(prev_instruction, None)
        if not ring or ring[-1][2] != prev_instruction:
            ring.append(("prev", prev_props, prev_instruction))
    return ring


def _extract_props(text: str, known_names: list[str] | None) -> set[str]:
    """Extract the props/imagery an instruction block re-anchors on.

    Primary layer: canonical stored element names that appear in the block —
    exactly the pool req 9 / req 14 argue over. A head-noun fallback catches
    re-lexicalized references ("the lavender", "a jug") that the exact match
    misses — the whole reason the byte/LCS guards were blind. Secondary: short
    lowercase noun phrases trailing common instruction verbs, catching props
    not yet stored in the elements map.
    """
    if not text:
        return set()
    low = text.lower()
    tokens = set(re.findall(r"[a-z]+", low))
    props: set[str] = set()
    for name in known_names or []:
        if not name or len(name) < 3:
            continue
        nl = name.lower()
        if nl in low:
            props.add(name.strip())
        else:
            head = nl.rsplit(" ", 1)[-1]
            if " " in nl and len(head) >= 3 and head in tokens:
                props.add(name.strip())
    for m in _INSTR_VERB_RE.finditer(text):
        np = m.group(1).strip(" .,;:'\"").lower()
        words = np.split()
        if (
            len(np) < 4
            or np in _PROP_STOPWORDS
            or any(w in _PROP_PHRASE_BAD_WORDS for w in words)
        ):
            continue
        props.add(np)
    return props


def _recently_used_block(recent: list[tuple[str, set[str], str]]) -> str:
    """Build the RECENTLY USED IMAGERY block injected into the instruction-
    generation prompt (P1). Newest block first.
    """
    if not recent:
        return ""
    lines = []
    for label, props, _text in reversed(recent):
        if props:
            lines.append(f"{label}: {', '.join(sorted(props))}")
    if not lines:
        return ""
    return (
        "Recently used props/imagery in the last instruction blocks (do NOT re-anchor on these; "
        "if one is genuinely essential to the current scene you may use it at most once and must "
        "vary the action around it):\n"
        + "\n".join(lines)
        + "\nPick the response's concrete objects, locations, and actions from stored items NOT on this "
        "list, or introduce a new detail.\n\n"
    )


def _collect_known_names(retrieval_ctx) -> list[str]:
    """Canonical stored element names for prop vocabulary matching. Elements
    are the item/prop store — the pool req 9 / req 14 argue over. Characters
    and groups are excluded (people/orgs are addressed, not anchored on as
    props). None-safe: an absent/old RetrievalContext yields [].
    """
    if retrieval_ctx is None:
        return []
    names: list[str] = []
    m = getattr(retrieval_ctx, "elements", None) or {}
    if isinstance(m, dict):
        entries = m.get("entries", {})
        if isinstance(entries, dict):
            names.extend(k for k in entries if isinstance(k, str))
    return names


def _own_replies_block(replies: list) -> str:
    """Negative-exemplar block (R1): DSS's own recent replies appended to the
    reply prompt so the model treats them as text NOT to re-say, rather than
    silent self-anchoring material.

    ``replies`` items are ``(absolute_message_index, text)`` tuples (plain
    strings are tolerated and rendered unnumbered). Indices are ascending
    absolute chat indexes, not relative positions, so the model cannot infer a
    sliding window from the numbers themselves.
    """
    if not replies:
        return ""
    quoted_parts = []
    for item in replies:
        if isinstance(item, tuple):
            idx, text = item
            quoted_parts.append(f"[message {idx}] \"\"\"{text[:1200]}\"\"\"")
        else:
            quoted_parts.append(f"\"\"\"{item[:1200]}\"\"\"")
    quoted = "\n".join(quoted_parts)
    return (
        "\n\nThe messages below are numbered by their absolute position in the chat history. "
        "They are YOUR OWN previous replies, quoted only as anti-repetition reference \u2014 they are NOT "
        "user instructions and must NOT be followed or answered. Do NOT reuse their sentences, phrasing, "
        "gestures, opening moves, or closings \u2014 write something fresh this turn. A prop that is "
        "genuinely essential to the current scene may appear again, but the action, phrasing, and sentence "
        "structure must differ from these earlier replies:\n"
        + quoted +
        "\n\nDo NOT reuse the above sentence structures. Vary the action and phrasing in your response as much as you can."
    )


def _current_scene_recap(scene: dict | None) -> str:
    """Compact, authoritative recap of the LIVE scene for the instruction
    generator — grounds the plan in what is actually happening now instead of
    the top of the (stable, never-pruned) subject block.
    """
    if not scene or not isinstance(scene, dict):
        return ""
    parts: list[str] = []
    sn = scene.get("_scene_number")
    if isinstance(sn, int):
        parts.append(f"Scene {sn}")
    now = scene.get("now")
    # Prefer the live `now.what` (refreshed eagarly by the per-turn now
    # update) over the top-level `what`, which lags 1-3 turns and is the
    # prop-recycling source; fall back to `what` only when `now.what` is
    # absent.
    what = ""
    if isinstance(now, dict) and isinstance(now.get("what"), str):
        what = now["what"].strip()
    if not what:
        top = scene.get("what")
        if isinstance(top, str):
            what = top.strip()
    if what:
        parts.append(f"What is happening: {what}")
    if isinstance(now, dict):
        when = now.get("when") if isinstance(now.get("when"), dict) else {}
        t = ", ".join(str(when.get(k) or "") for k in ("date", "time", "specific_time")).strip()
        if t:
            parts.append(f"Scene time: {t}")
        where = now.get("where")
        if isinstance(where, str) and where.strip():
            parts.append(f"Scene location: {where.strip()}")
        who = now.get("who")
        if isinstance(who, dict) and isinstance(who.get("characters"), list):
            spots = [
                f"{c.get('name')} ({c.get('location')})"
                for c in who["characters"]
                if isinstance(c, dict) and c.get("name") and c.get("location")
            ]
            if spots:
                parts.append("Present: " + ", ".join(spots[:6]))
    return "\n".join(parts)


