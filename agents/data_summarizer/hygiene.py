"""Deterministic memory-hygiene passes for summarized subject stores.

Root causes observed live in cyberpunk_thriller__4dbf1435 (84 turns):

1. Verbatim sentence duplication INSIDE entities: branch updates replace a
   field value wholesale and the model copies the previous text into the
   replacement, so long sentences restate themselves (Vell carried one
   sentence once in ``description`` and three times in ``biography``).
2. Cross-entity propagation: a single turn injected a participant's profile
   sentence verbatim into OTHER entities' biographies (Handler/Sparrow
   inherited a Vell sentence at turn 71). First-owner semantics are not
   recoverable from a snapshot, so ownership goes to the entity carrying the
   MOST occurrences; ties break lexicographically. Exact normalized matches
   only — near-variants across entities are never touched.
3. Relationship-row storms: update storms mint concept-partner rows
   ("The_Record", "The_Cadence") that exist as NO subject anywhere, under
   dual spellings ("The Handler"/"Handler", "Client"/"The Client"), some
   structurally empty. Sanitation is SHAPE-driven — a row collection is any
   mapping whose values are dicts carrying relation/status/importance keys —
   so no schema-type dispatch or per-field key map is involved (project rule:
   no hardcoded per-data-type dispatchers).

All functions are pure tree mutators (no I/O) and fully deterministic;
callers log the returned stats.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Sentences shorter than this are exempt from every dedup pass.
DEFAULT_MIN_SENTENCE_CHARS = 60

# Near-duplicate collapse ratio within one entity (normalized casefold).
# Deliberately conservative: biography/description stores layer PARAPHRASED
# content by design, so aggressive ratios (e.g. 0.94) destroy legitimate
# summary layers. 0.995 means "identical modulo whitespace/punct cosmetics".
DEFAULT_NEAR_DUP_RATIO = 0.995

# Candidate ceiling for pairwise near-dup comparison per entity; above this
# only exact matching runs (keeps cost bounded for giant stores).
_NEAR_DUP_MAX_CANDIDATES = 400

# Universal per-entity cap on relationship rows (any type). Overflow evicts
# unknown-partner rows first, then lowest importance, then alphabetical.
_DEFAULT_MAX_REL_ROWS = 16

# A row whose relation label is a single word of fewer than this many
# letters AND carries neither status nor importance payload is structural
# noise ("path": {"relation": "that"}), not prose.
_REL_NOISE_MAX_LETTERS = 6

_ROW_SHAPE_KEYS = ("relation", "status", "importance")
_CANON_STRIP_RE = re.compile(r"[^a-z0-9]")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _cfg(summarizer, key: str, default):
    try:
        return (summarizer.config or {}).get(key, default)
    except Exception:
        return default


# --------------------------------------------------------------------------
# Relationship sanitation
# --------------------------------------------------------------------------

def _is_junky_row(payload) -> bool:
    """True for structurally-empty relation rows (no real information)."""
    if not isinstance(payload, dict):
        return False
    rel = str(payload.get("relation") or "")
    letters = re.sub(r"[^A-Za-z]", "", rel)
    words = [w for w in re.split(r"[^A-Za-z]+", rel) if w]
    status = payload.get("status")
    imp = payload.get("importance")
    has_score = isinstance(imp, dict) and imp.get("score") not in (None, "")
    informative = bool(str(status or "").strip()) or has_score
    if informative:
        return False
    return len(words) <= 1 and len(letters) < _REL_NOISE_MAX_LETTERS


def _row_richness(payload: dict) -> int:
    """Information weight of a relation row; richer rows win merges."""
    w = 0
    for v in payload.values():
        if isinstance(v, str):
            w += 1 if v.strip() else 0
        elif isinstance(v, (int, float)):
            w += 1
        elif isinstance(v, dict):
            w += sum(1 for x in v.values() if x not in (None, "", []))
        elif isinstance(v, list):
            w += len(v)
    return w


def canon_name(name: str) -> str:
    """Case/punct/underscore/article-insensitive identity for a partner key."""
    s = _CANON_STRIP_RE.sub("", str(name).lower())
    if s.startswith("the"):
        s = s[3:]
    return s


def _display_canonical(candidates: list[str], universe_titles: set[str]) -> str:
    """Pick the representative spelling for a collapsed alias group.

    An exact existing-subject title wins outright; otherwise prefer the
    underscore-free spelling (stores conventionally use spaces), then the
    shortest variant.
    """
    exact = sorted(c for c in candidates if c in universe_titles)
    if exact:
        return exact[0]
    winner = sorted(candidates, key=lambda c: ("_" in c, len(c), c))[0]
    return winner.replace("_", " ")


def _iter_row_collections(node):
    """Yield every mapping whose VALUES all look like relationship rows."""
    if isinstance(node, dict):
        vals = [v for v in node.values() if isinstance(v, dict)]
        if vals and all(any(k in v for k in _ROW_SHAPE_KEYS) for v in vals):
            yield node
            return
        for v in node.values():
            yield from _iter_row_collections(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_row_collections(v)


def sanitize_relationships(subject_data, universe_titles: set[str], cap: int,
                           stats: dict, prune_unknown: bool = False) -> None:
    """Canonicalize/dedupe/junk-drop/cap every relationship collection found
    by shape under one entity's subtree."""
    for coll in _iter_row_collections(subject_data):
        # --- canonicalize partner spellings ---
        groups: dict[str, list[str]] = defaultdict(list)
        for key in list(coll.keys()):
            groups[canon_name(key)].append(key)
        merged_rows = {}
        for _cid, spellings in groups.items():
            winner_key = _display_canonical(spellings, universe_titles)
            rows = [coll[k] for k in spellings if isinstance(coll[k], dict)]
            if len(rows) > 1:
                stats["rel_rows_merged"] += len(rows) - 1
            richest = max(rows, key=_row_richness) if rows else {}
            # Union payloads: start from the richest, fill gaps from others.
            merged = dict(richest)
            for r in rows:
                if not isinstance(r, dict):
                    continue
                for k, v in r.items():
                    if k == "importance":
                        continue
                    if (merged.get(k) in (None, "", [])) and v not in (None, "", []):
                        merged[k] = v
                ri, mi = r.get("importance"), merged.get("importance")
                rs = ri.get("score") if isinstance(ri, dict) else None
                ms = mi.get("score") if isinstance(mi, dict) else None
                if isinstance(rs, (int, float)) and (
                        ms is None or rs > ms):
                    merged["importance"] = ri
            merged_rows[winner_key] = merged
        coll.clear()
        coll.update(merged_rows)
        # --- structural junk ---
        for k in [k for k, v in coll.items() if _is_junky_row(v)]:
            del coll[k]
            stats["rel_rows_pruned"] += 1
        # --- cap enforcement ---
        overflow = len(coll) - cap
        if prune_unknown and overflow < 0:
            # Aggressive mode: unknown partners are dropped outright, not
            # merely demoted under cap pressure.
            for k in [k for k in coll if k not in universe_titles]:
                del coll[k]
                stats["rel_rows_pruned"] += 1
            overflow = len(coll) - cap
        if overflow <= 0:
            continue

        def sort_key(item):
            name, payload = item
            known = name in universe_titles
            imp = payload.get("importance")
            score = imp.get("score") if isinstance(imp, dict) else None
            sc = -(score if isinstance(score, (int, float)) else -1e9)
            # Under pressure, unknown partners always evict before known ones.
            return (0 if known else 1, sc, name)

        keep_names = {n for n, _ in sorted(coll.items(), key=sort_key)[:cap]}
        for k in [k for k in coll if k not in keep_names]:
            del coll[k]
            stats["rel_rows_capped"] += 1


# --------------------------------------------------------------------------
# Sentence-level duplication
# --------------------------------------------------------------------------

def _norm_sent(s: str) -> str:
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2014", "-").replace("\u2013", "-"))
    s = _WS_RE.sub(" ", s.strip().lower())
    return s.rstrip(".")


def _collect_norms(node, min_chars, out: list) -> None:
    """Ordered long-sentence norms throughout a subtree."""
    if isinstance(node, dict):
        for v in node.values():
            _collect_norms(v, min_chars, out)
    elif isinstance(node, list):
        for v in node:
            _collect_norms(v, min_chars, out)
    elif isinstance(node, str) and len(node) >= min_chars:
        for part in _SENT_SPLIT_RE.split(node):
            if len(part.strip()) >= min_chars:
                out.append(_norm_sent(part))


class _StripCtx:
    """Per-entity rewrite state for one hygiene sweep."""
    __slots__ = ("owners", "hid", "min_chars", "near_ratio", "kept",
                 "kept_exact")

    def __init__(self, owners: dict[str, str], hid: str,
                 min_chars: int, near_ratio: float):
        self.owners = owners
        self.hid = hid
        self.min_chars = min_chars
        self.near_ratio = near_ratio
        # Norms already kept at least once in THIS entity (keep-first).
        self.kept: list[tuple[str, int]] = []
        self.kept_exact: set[str] = set()


def _rewrite_text(text: str, ctx: _StripCtx) -> tuple[str, int]:
    """Keep-first rewrite of one carrier string.

    A long sentence is dropped iff (a) another entity owns it outright
    (cross-entity copy), or (b) an identical/near-identical sentence was
    already kept earlier in THIS entity's walk order (intra duplication).
    The FIRST occurrence in the owner always survives.
    """
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    kept_parts, dropped = [], 0
    for part in parts:
        if not part:
            continue
        if len(part) < ctx.min_chars:
            kept_parts.append(part)
            continue
        norm = _norm_sent(part)
        owner = ctx.owners.get(norm)
        if owner is not None and owner != ctx.hid:
            dropped += 1
            continue
        if norm in ctx.kept_exact:
            dropped += 1
            continue
        dup = False
        nlen = len(part)
        for kn, klen in ctx.kept:
            if kn == norm or (
                    abs(nlen - klen) / max(nlen, klen, 1) <= 0.25
                    and SequenceMatcher(None, kn, norm,
                                        autojunk=False).ratio() >= ctx.near_ratio):
                dup = True
                break
        if dup:
            dropped += 1
            continue
        kept_parts.append(part)
        ctx.kept.append((norm, len(part)))
        ctx.kept_exact.add(norm)
    if not dropped:
        return text, 0
    if not kept_parts:
        return "", dropped
    return (_WS_RE.sub(" ", " ".join(kept_parts)).strip()), dropped


def _visit_strings(node, ctx: _StripCtx) -> int:
    """Apply _rewrite_text to every long string in the tree (walk order =
    dict insertion / list index order — deterministic from load order)."""
    dropped = 0
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if isinstance(v, str) and len(v) >= ctx.min_chars:
                new, d = _rewrite_text(v, ctx)
                if d:
                    node[k] = new
                    dropped += d
            else:
                dropped += _visit_strings(v, ctx)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str) and len(v) >= ctx.min_chars:
                new, d = _rewrite_text(v, ctx)
                if d:
                    node[i] = new
                    dropped += d
            else:
                dropped += _visit_strings(v, ctx)
    return dropped


def _entity_hygiene_pass(entities, min_chars: int, near_ratio: float) -> int:
    """Single orchestrated sentence pass over every entity.

    Occurrence counting runs on the RAW trees (multiplicity intact): the
    entity carrying the MOST occurrences of a globally-shared sentence is
    elected its owner (ties: lexicographically smallest holder id). Then
    every entity rewrites its strings KEEP-FIRST: the owner retains exactly
    one instance (its first), other entities lose all copies, and each
    entity additionally collapses its own later repeats. Deterministic.
    """
    holders: dict[str, Counter] = defaultdict(Counter)
    for hid, ref in entities:
        norms: list[str] = []
        _collect_norms(ref, min_chars, norms)
        for n, c in Counter(norms).items():
            holders[n][hid] += c

    owners: dict[str, str] = {}
    for norm, hc in holders.items():
        if sum(hc.values()) < 2:
            continue
        best_id, best_cnt = None, -1
        for holder, cnt in hc.items():
            if cnt > best_cnt or (cnt == best_cnt and holder < best_id):
                best_id, best_cnt = holder, cnt
        owners[norm] = best_id

    dropped = 0
    for hid, ref in entities:
        ctx = _StripCtx(owners, hid, min_chars, near_ratio)
        dropped += _visit_strings(ref, ctx)
    return dropped


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _entry_map(v) -> bool:
    """True when v is a mapping whose own values are all dicts (rows)."""
    return (isinstance(v, dict) and bool(v)
            and all(isinstance(x, dict) for x in v.values()))


def _walk_entity_groups(node, path="", out=None):
    """Yield (holder_path, dict_ref) for every title->row mapping reached
    within two levels of the subject root. Covers 'entries'-wrapped stores
    and multi-section stores (events.past/scenes/events/chapters) alike."""
    if out is None:
        out = []
    if not isinstance(node, dict):
        return out
    for k, v in node.items():
        p = f"{path}/{k}"
        if _entry_map(v):
            for t, r in v.items():
                if isinstance(r, dict):
                    out.append((f"{p}\x00{t}", r))
        elif isinstance(v, dict):
            inner_maps = [x for x in v.values() if isinstance(x, dict)]
            if inner_maps and not _entry_map(v):
                _walk_entity_groups(v, p, out)
    return out


def _subject_stores(all_subjects_data):
    if not isinstance(all_subjects_data, dict):
        return []
    return [(n, v) for n, v in all_subjects_data.items() if isinstance(v, dict)]


def _entities(all_subjects_data):
    out = []
    for store_name, store in _subject_stores(all_subjects_data):
        for hid, ref in _walk_entity_groups(store):
            out.append((f"{store_name}\x00{hid}", ref))
    return out


def _collect_titles(all_subjects_data) -> set[str]:
    titles: set[str] = set()
    for store_name, store in _subject_stores(all_subjects_data):
        titles.add(store_name)
        groups = _walk_entity_groups(store)
        for hid, _ref in groups:
            titles.add(hid.split("\x00")[-1])
    return titles


def clean_store(all_subjects_data, summarizer=None, peer_scope=True,
                log=lambda *_: None) -> dict:
    """Full hygiene entry point invoked from DataSummarizer.generate.

    Order matters: intra-entity dedup FIRST (the cross-entity ledger then
    sees final occurrence counts), then cross-entity prune, then
    relationship sanitation (canonicalize -> junk-drop -> cap).
    """
    min_chars = int(_cfg(summarizer, "dedup_min_sentence_chars",
                         DEFAULT_MIN_SENTENCE_CHARS))
    near_ratio = float(_cfg(summarizer, "near_dup_ratio", DEFAULT_NEAR_DUP_RATIO))
    cap = int(_cfg(summarizer, "max_relationships_per_entity",
                   _DEFAULT_MAX_REL_ROWS))
    prune_unknown = bool(_cfg(summarizer, "prune_unknown_relationship_targets",
                              False))

    stats = {
        "dropped_sentences": 0,
        "rel_rows_merged": 0,
        "rel_rows_pruned": 0,
        "rel_rows_capped": 0,
    }

    stores = _subject_stores(all_subjects_data)
    universe = _collect_titles(all_subjects_data)

    if peer_scope:
        stats["dropped_sentences"] = _entity_hygiene_pass(
            _entities(all_subjects_data), min_chars, near_ratio)
    else:
        # No-peer mode: intra-entity keep-first only (each entity processed
        # alone, so nothing is ever owned elsewhere).
        stats["dropped_sentences"] = _entity_hygiene_pass(
            [(hid, ref) for hid, ref in _entities(all_subjects_data)],
            min_chars, near_ratio)

    for _name, store in stores:
        for _hid, entity_data in _walk_entity_groups(store):
            sanitize_relationships(entity_data, universe, cap, stats,
                                   prune_unknown=prune_unknown)

    total = (stats["dropped_sentences"]
             + stats["rel_rows_merged"] + stats["rel_rows_pruned"]
             + stats["rel_rows_capped"])
    if total:
        log(f"Memory hygiene: {stats['dropped_sentences']} duplicate "
            f"sentences removed, {stats['rel_rows_merged']} rel rows merged, "
            f"{stats['rel_rows_pruned']} junk-pruned, "
            f"{stats['rel_rows_capped']} capped")
    return stats
