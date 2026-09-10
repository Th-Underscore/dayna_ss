"""Pure parsing/resolution helpers for the data summarizer.

Free functions extracted verbatim from data_summarizer (T4 refactor): entry
name normalization/aliases, path-key resolution, tolerant JSON loading,
negative-verdict detection, new-entry name extraction/filtering.
"""
from __future__ import annotations

import json
import re
from typing import Any

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

import jsonc

from ...utils.helpers import _GRAY, _RESET, strip_response

ADD_NEW_MAX_PER_CALL = 4
ADD_NEW_MIN_CONTAINED_NAME_LEN = 4


def _entries_as_list(entries) -> list:
    """Normalize a schema branch to its canonical list shape.

    The schema declares dict-of-entry branches as ``list[T]``, but legacy
    archive code (check_and_archive_chapter/arc) once wrote dicts keyed by
    title. Read the branch as a list regardless of which shape a run has."""
    if not entries:
        return []
    if isinstance(entries, dict):
        return list(entries.values())
    return list(entries)


def _safe_int(value, default: int = 0) -> int:
    """Coerce a value to an int, tolerating None / strings / other junk.

    ``.get(key, 0)`` only guards a MISSING key — a present-but-null value
    (e.g. a freshly-added arc whose ``ending_chapter`` has not been set)
    still flows through as None and crashes arithmetic downstream."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

# Add-new guards (deterministic, generic across all dict-of-entries branches).
# These stop the LLM from fabricating dozens of near-duplicate micro-entries every
# turn (e.g. the ElementMap explosion that drove the soak's call count): reject
# near-duplicate names against existing state and cap batch + total growth.
# Size gate for the whole-subject context re-statement in per-entry calls
# (_context_restatement, mode "auto"): maps under this many chars keep the full
# boundary re-statement as cheap accentuation insurance; larger maps drop it for
# a compact sibling roster instead (see docs/plans/long_horizon_soak_plan.md section 23).

def _normalize_entry_name(name: Any) -> str:
    """Lowercase + strip to alphanumerics only, for deterministic dedupe."""
    if name is None:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _collect_entry_aliases(entry: Any, out: list[str] | None = None, depth: int = 0) -> list[str]:
    """Collect alias strings from a nested entry structure (schema-agnostic).

    Walks dict/list values looking for ``aliases`` arrays (per-relationship,
    per-group, per-element, etc.) so a key can be resolved by nickname/title
    even though the entry itself has no top-level ``aliases`` field. Bounded
    recursion + list cap for safety.
    """
    if out is None:
        out = []
    if depth > 5 or not isinstance(entry, (dict, list)):
        return out
    if isinstance(entry, dict):
        for k, v in entry.items():
            if k == "aliases" and isinstance(v, list):
                out.extend(a for a in v if isinstance(a, str))
            else:
                _collect_entry_aliases(v, out, depth + 1)
    else:
        for v in entry[:200]:
            _collect_entry_aliases(v, out, depth + 1)
    return out


def _resolve_dict_key(data_dict: dict, key: str) -> str | None:
    """Resolve a single key against a dict: exact, normalized, then alias match.

    Returns the canonical key or None if unresolvable.
    """
    if not isinstance(data_dict, dict) or key is None:
        return None
    if key in data_dict:
        return key
    kn = _normalize_entry_name(key)
    if not kn:
        return None
    for k in data_dict:
        if _normalize_entry_name(k) == kn:
            return k
    for k, v in data_dict.items():
        for alias in _collect_entry_aliases(v):
            if _normalize_entry_name(alias) == kn:
                return k
    return None


def _retarget_dict_key(d: dict, key: Any, data: dict) -> None:
    """Retarget a single dangling dict key onto its canonical form, in place.

    A *dangle* is a key the model spelled as a lookalike (``The_Second_Runner``)
    that would otherwise mint a stray sibling / dangling edge, while the same
    referent resolves (via ``_resolve_dict_key``) to a distinct, already-present
    key (``Second Runner``). The write is re-addressed onto the canonical node.
    The move is a guaranteed no-op when no distinct collision exists: a key that
    is already canonical (resolves to itself) or to nothing is left untouched —
    so a legitimate shared/undercover identity (a single canonical key) is never
    rewritten. A same-level dangle-vs-canonical clash merges dict payloads rather
    than clobbering, so existing data at the canonical key survives."""
    if not isinstance(key, str):
        return
    canonical = _resolve_dict_key(data, key)
    if canonical is None or canonical == key or key not in d:
        return
    moved = d.pop(key)
    existing = d.get(canonical)
    if isinstance(existing, dict) and isinstance(moved, dict):
        existing.update(moved)
    elif canonical not in d:
        d[canonical] = moved


def _retarget_value(value: Any, data: dict, depth: int = 0) -> Any:
    """Retarget dangling referent keys/row-keys inside an update *value* (H3).

    The branch-update path already canonicalizes the update's *path* via
    ``_resolve_path_keys``; this is the symmetric pass over the *value*. It walks
    the value structure and retargets only keys (dict keys and top-level keys of
    dict rows) — the structural partner/row references that dangles enter through —
    leaving string items, scalars, and prose untouched. A referent is retargeted
    only when it is a distinct collision (resolves to a DIFFERENT existing key),
    so it is idempotent and a guaranteed no-op where no collision exists. The
    structure is rewritten in place and returned."""
    if depth > 12:
        return value
    if isinstance(value, dict):
        for key in list(value.keys()):
            _retarget_dict_key(value, key, data)
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                _retarget_value(v, data, depth + 1)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in list(item.keys()):
                    _retarget_dict_key(item, key, data)
                for k, v in item.items():
                    if isinstance(v, (dict, list)):
                        _retarget_value(v, data, depth + 1)
    return value


def _resolve_path_keys(data: Any, keys: list) -> list:
    """Canonicalize a key path against the live structure.

    At each dict level a key may be: the exact key, a dot-split fragment of a
    key that itself contains dots (``Mr`` + ``Peters`` for ``Mr. Peters``
    written without [brackets]), a case/punctuation variant, or a known alias
    (nickname/title). Returns a canonical key list ready for recursive_get/set.
    """
    result: list = []
    current = data
    i, n = 0, len(keys)
    while i < n:
        key = keys[i]
        if isinstance(current, dict):
            if key in current:
                result.append(key)
                current = current[key]
                i += 1
                continue
            joined = key
            resolved = None
            j = i
            while j < n - 1:
                j += 1
                joined = joined + "." + keys[j]
                # Match the joined fragment against the real keys (normalized:
                # "Mr.Peters" joins back to "Mr. Peters").
                joined_key = _resolve_dict_key(current, joined)
                if joined_key is not None:
                    resolved = (joined_key, j)
                    break
            if resolved:
                result.append(resolved[0])
                current = current[resolved[0]]
                i = resolved[1] + 1
                continue
            canonical = _resolve_dict_key(current, key)
            if canonical is not None:
                result.append(canonical)
                current = current[canonical]
                i += 1
                continue
            result.append(key)
            current = current.get(key, {})
            i += 1
        elif isinstance(current, list):
            try:
                idx = int(key)
            except (ValueError, TypeError):
                idx = None
            if idx is not None and -len(current) <= idx < len(current):
                # Keep the index as a string: recursive_get/recursive_set
                # expect numeric list indices in string form and convert them
                # themselves; an int index raises in recursive_set.
                result.append(key if isinstance(key, str) else str(key))
                current = current[idx]
            else:
                result.append(key if isinstance(key, str) else str(key))
                current = {}
            i += 1
        else:
            result.append(key)
            i += 1
    return result


_NEGATIVE_VERDICTS = {
    "NO",
    "NO_UPDATES_REQUIRED",
    "UNCHANGED",
    "N/A",
}


def _is_negative_verdict(text: str) -> bool:
    """True for the no-update markers the prompts demand as valid negatives."""
    t = strip_response(text).strip().upper().rstrip(".:")
    if not t:
        return False
    for marker in _NEGATIVE_VERDICTS:
        if len(marker) >= 5:
            if t.startswith(marker):
                return True
        elif t == marker:
            return True
    return False


def _tolerant_json_loads(text: str):
    """Parse LLM JSON with progressive tolerance.

    Tries strict JSONC first, then json_repair (handles the token-level slips small
    models make: colon-for-comma before a nested value, single quotes, unquoted
    keys, trailing commas). Raises json.JSONDecodeError only if both fail.
    """
    try:
        return jsonc.loads(text)
    except json.JSONDecodeError:
        if _HAS_JSON_REPAIR and text.strip():
            try:
                return json_repair.loads(text)
            except Exception:
                pass
        raise


def _is_schema_echo(data: dict) -> bool:
    """True when the model reproduced the prompt's embedded JSONSchema instead of
    generating entry data.

    The new-entry prompt embeds ``get_relevant_json_schema_definitions`` output,
    which has the shape ``{"main_schema": {"$ref": ...}, "definitions": {...}}``.
    A small-active model can latch onto that block and echo it back — the response
    is *valid* JSON, so it would otherwise be stored as a garbage entry. Legitimate
    entry objects never carry these top-level keys.
    """
    keys = set(data.keys())
    if keys & {"main_schema", "definitions"}:
        return True
    if "schema" in keys:
        return True
    return any(isinstance(v, dict) and "$ref" in v for v in data.values())


def _filter_new_entry_names(new_entry_names: list, data: dict) -> list:
    """Deterministically filter proposed new-entry names against existing state.

    Rejects: exact/near-duplicate of an existing key, exact/near-duplicate of
    another proposed name, and any proposal beyond ADD_NEW_MAX_PER_CALL.
    Keeps mid-scene discovery intact — only obvious duplication is removed.
    """
    existing = [_normalize_entry_name(k) for k in data.keys() if _normalize_entry_name(k)]
    existing_aliases = {
        _normalize_entry_name(a)
        for k, v in data.items()
        for a in _collect_entry_aliases(v)
        if _normalize_entry_name(a)
    }
    kept = []
    seen = set()
    for name in new_entry_names:
        norm = _normalize_entry_name(name)
        if not norm:
            continue
        if norm in seen:
            continue
        # Near-duplicate of an existing entry: exact match, containment where
        # the shorter name is meaningful (>= MIN_CONTAINED_NAME_LEN chars), or
        # an existing entry's known alias (nickname/title of a live entity).
        duplicate = False
        for ex in existing:
            if norm == ex:
                duplicate = True
                break
            if len(ex) >= ADD_NEW_MIN_CONTAINED_NAME_LEN and norm in ex:
                duplicate = True
                break
            if len(norm) >= ADD_NEW_MIN_CONTAINED_NAME_LEN and ex in norm:
                duplicate = True
                break
        if not duplicate and norm in existing_aliases:
            print(f"{_GRAY}add_new: skipping '{name}' (known alias of an existing entry).{_RESET}")
            duplicate = True
        if duplicate:
            print(f"{_GRAY}add_new: skipping '{name}' (near-duplicate of existing entry).{_RESET}")
            continue
        seen.add(norm)
        kept.append(name)
        if len(kept) >= ADD_NEW_MAX_PER_CALL:
            break
    return kept


# Prose prefixes that mean "nothing new" rather than a name, for the bare-text
# fallback in _extract_entry_names. Small models occasionally answer the name
# query with a sentence instead of an array; these must not become entries.
_NO_NEW_ENTRY_PREFIXES = (
    "no new", "no more", "no changes", "no entries", "no additional",
    "none", "nothing", "not applicable", "n/a", "no (", "there are no",
    "there is no", "i don't see", "i do not see", "i didn't see", "i did not see",
    "cannot", "can't find", "no need",
)


def _extract_entry_names(response: str) -> list[str]:
    """Leniently extract a list of new-entry names from an add_new query response.

    The discovery templates demand a JSON array of strings, but small models
    without reasoning frequently answer with a bare name, one name per line,
    or bulleted names instead. Tries strict JSON -> json_repair -> quoted
    strings -> bare lines (bullets/brackets/quotes/trailing commas stripped).
    Returns [] when nothing plausible is found — the caller simply defers
    discovery to a later turn (add_new re-runs).
    """
    stripped = strip_response(response).strip()
    if not stripped:
        return []

    try:
        parsed = _tolerant_json_loads(stripped)
        if isinstance(parsed, list) and parsed:
            names = [n.strip() for n in parsed if isinstance(n, str) and n.strip()]
            if names:
                return names
    except json.JSONDecodeError:
        pass

    if _is_negative_verdict(stripped) or stripped.lower().startswith(_NO_NEW_ENTRY_PREFIXES):
        return []

    quoted = re.findall(r'"([^"]+)"', stripped)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]

    names = []
    for line in stripped.splitlines():
        cleaned = line.strip().strip("\"'[](){},.").lstrip("-•*·\t ").strip()
        if not cleaned or len(cleaned) > 120 or not any(c.isalnum() for c in cleaned):
            continue
        low = cleaned.lower()
        if _is_negative_verdict(cleaned) or low.startswith(_NO_NEW_ENTRY_PREFIXES):
            continue
        names.append(cleaned)
    return names[:ADD_NEW_MAX_PER_CALL]

defaults_to_inherit = [
    "gate_check_prompt_template",
    "branch_query_prompt_template",
    "branch_update_prompt_template",
    "update_prompt_template",
]
