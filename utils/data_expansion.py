from typing import Any

import jsonc


def _build_comments_structure(data):
    """Recursively build the comment tree for jsonc.dumps."""
    if isinstance(data, list):
        comments = {}
        for i, item in enumerate(data):
            if isinstance(item, (list, dict)):
                inner = _build_comments_structure(item)
                if inner:
                    comments[i] = (str(i), inner)
                else:
                    comments[i] = str(i)
            else:
                comments[i] = str(i)
        return comments
    elif isinstance(data, dict):
        comments = {}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                inner = _build_comments_structure(value)
                if inner:
                    comments[key] = (None, inner)
        return comments
    return None


def to_jsonc(data, indent: int = 2) -> str:
    """Serialize data to JSONC with a trailing index comment on every array element.

    Replaces the old dict-expansion approach: data keeps its native shape, and
    the LLM sees `// 0`, `// 1`, … after each array entry.

    Scalars (str, int, float, bool) pass through as their string form;
    None renders as empty string.
    """
    if data is None:
        return "null"
    if not isinstance(data, (dict, list)):
        return str(data)
    tree = _build_comments_structure(data)
    if isinstance(data, list):
        root_comments = (None, tree) if tree else None
    else:
        root_comments = tree
    raw = jsonc.dumps(data, indent=indent, comments=root_comments)  # type: ignore[arg-type]
    return _make_comments_inline(raw)


def _make_comments_inline(jsonc_str: str) -> str:
    """Shift stacked `// N` comment lines to trailing inline position."""
    lines = jsonc_str.splitlines()
    out = []
    pending = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//"):
            pending = stripped
            continue
        if pending and stripped:
            out.append(f"{line} {pending}")
            pending = None
        else:
            out.append(line)
    return "\n".join(out)


def _is_dict_expandable_to_list(data: dict) -> bool:
    """Check if a dictionary's keys are stringified sequential integers (e.g., "0", "1", "2", ...)."""
    if not data:
        return True
    int_keys = []
    for k in data.keys():
        if not isinstance(k, str) or not k.isdigit():
            return False
        int_keys.append(int(k))
    if not int_keys:
        return True
    int_keys.sort()
    return all(int_keys[i] == i for i in range(len(int_keys)))


_COLLECTION_KEY_CANDIDATES = (
    "name",
    "entry_name",
    "relationship_name",
    "relation_name",
    "group_name",
    "title",
    "id",
)


def _infer_collection_key(elements: list) -> str | None:
    """Pick a name-ish field present on every dict element, if unambiguous.

    Used by `coerce_container_types` to rebuild a dict-keyed collection from a
    model-written list whose elements carry their own key. Returns None when no
    single field is common to all elements (e.g. CharacterRelationship objects,
    whose dict key lives *outside* the object — those lists are left untouched).
    """
    if not elements or not all(isinstance(e, dict) for e in elements):
        return None
    for cand in _COLLECTION_KEY_CANDIDATES:
        vals = [e.get(cand) for e in elements]
        if all(isinstance(v, str) and v for v in vals):
            return cand
    return None


def coerce_container_types(data: Any, schema_type, parser) -> Any:
    """Schema-driven repair of LLM-written container shapes (in place).

    Small models frequently emit the wrong container for a field — most commonly
    an empty list ``[]`` where the schema declares ``dict[str, ...]`` (e.g.
    ``Character.relationships`` / ``group_status``). The stored shape then
    mismatches the schema and later name-keyed update paths
    (``relationships.Glove.owner...``) crash in ``recursive_set`` with
    "Cannot use non-integer key 'Glove' on a list".

    This walks the live structure against the parsed schema and coerces:
      * ``dict[...]``-declared fields written as lists — empty list -> ``{}``,
        populated list -> keyed by a name-ish element field when unambiguous;
      * ``list[X]``-declared fields written as int-keyed dicts -> back to lists
        (the ``_is_dict_expandable_to_list`` convention).

    Mirrors the old unexpand traversal; mutates dict/list
    values in place (the root object is preserved, so callers that alias the
    subject dict keep seeing the same object) and only returns a NEW container for
    the coerced cases themselves.
    """
    from .schema_parser import ParsedSchemaClass, TYPE_MAP

    if schema_type in TYPE_MAP.values() or data is None:
        return data

    origin = getattr(schema_type, "__origin__", None)
    if origin is dict:
        args = getattr(schema_type, "__args__", ()) or ()
        value_type = args[-1] if len(args) > 1 else None
        if isinstance(data, list):
            if not data:
                return {}
            key = _infer_collection_key(data)
            if key is not None:
                out = {}
                for elem in data:
                    nm = elem[key]
                    out.setdefault(nm, elem)
                return out
            return [coerce_container_types(item, value_type, parser) for item in data]
        if isinstance(data, dict):
            for k in data:
                data[k] = coerce_container_types(data[k], value_type, parser)
            return data
        return data

    if origin is list:
        args = getattr(schema_type, "__args__", ()) or ()
        item_type = args[0] if args else None
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = coerce_container_types(data[i], item_type, parser)
            return data
        if isinstance(data, dict) and _is_dict_expandable_to_list(data):
            sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
            return [
                coerce_container_types(item, item_type, parser)
                for _, item in sorted_items
            ]
        return data

    if isinstance(schema_type, ParsedSchemaClass):
        cls = schema_type
        if cls.definition_type == "dataclass":
            if isinstance(data, dict):
                for k, v in data.items():
                    field_schema = cls.get_field(k)
                    field_type = field_schema.type if field_schema else None
                    data[k] = coerce_container_types(v, field_type, parser)
            return data
        if cls.definition_type == "alias":
            field_type = cls.get_field().type
            return coerce_container_types(data, field_type, parser)
        return data

    if isinstance(data, dict):
        for k in data:
            data[k] = coerce_container_types(data[k], None, parser)
        return data
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = coerce_container_types(data[i], None, parser)
        return data
    return data



def get_values(data: dict | list):
    """Get the values of a potentially expanded list."""
    if isinstance(data, dict):
        return data.values()
    return data


def enumerate_list(data: dict | list):
    """Enumerate a potentially expanded list."""
    if isinstance(data, dict):
        return enumerate(data.values())  # data.items() for expanded lists
    return enumerate(data)
