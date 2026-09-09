from typing import Any

from .console import _BOLD, _DEBUG, _RESET


def expand_lists_in_data_for_llm(data: dict | list, schema_type, parser) -> Any:
    """Recursively traverse data and expand lists into dictionaries
    with stringified integer keys if the corresponding schema indicates to do so.

    This is for creating a JSON-like structure that is clearer for an LLM.
    This does not mutate the original data.

    Args:
        data (dict | list): The data to process (dict, list, or primitive).
        schema_type (ParsedSchemaClass | None): The type for the current data segment, if available.
        parser (SchemaParser): The SchemaParser instance for resolving type definitions.

    Returns:
        out (Any): A new data structure with specified lists expanded.
    """
    from .schema_parser import ParsedSchemaClass, TYPE_MAP

    if schema_type in TYPE_MAP.values() or data is None:
        return data

    if isinstance(schema_type, ParsedSchemaClass):
        parsed_class_obj = schema_type

        if parsed_class_obj.definition_type == "dataclass" and isinstance(data, dict):
            data_copy = {}
            for k, v in data.items():
                field_schema = parsed_class_obj.get_field(k)
                if field_schema:
                    item_schema_type = field_schema.type if field_schema.type else None
                    data_copy[k] = expand_lists_in_data_for_llm(v, item_schema_type, parser)
            return data_copy
        elif parsed_class_obj.definition_type == "field":
            field_type = parsed_class_obj.get_field().type
            item_schema_type = None

            if isinstance(data, dict):
                if (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is dict
                    and hasattr(field_type, "__args__")
                    and field_type.__args__
                ):
                    item_schema_type = field_type.__args__[1]
                return {k: expand_lists_in_data_for_llm(v, item_schema_type, parser) for k, v in data.items()}

            if parsed_class_obj.do_expand_into_dict and isinstance(data, list):
                if (
                    hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and hasattr(field_type, "__args__")
                    and field_type.__args__
                ):
                    item_schema_type = field_type.__args__[0]
                return {str(i): expand_lists_in_data_for_llm(item, item_schema_type, parser) for i, item in enumerate(data)}

            print(f"Returning '{schema_type.name}' as is: {_BOLD}{data} {_DEBUG} {field_type}{_RESET}")
            return [expand_lists_in_data_for_llm(item, field_type, parser) for item in data] if isinstance(data, list) else data

        elif parsed_class_obj.definition_type == "alias" and isinstance(data, list):
            # Alias of a list (e.g. Chapters = list[Chapter]) rendered directly:
            # keep the LIST shape — list-iterating templates must see entries,
            # not the stringified keys a dict-expansion would produce.
            aliased_type = parsed_class_obj.get_field().type if parsed_class_obj.get_field() else None
            if hasattr(aliased_type, "__origin__") and aliased_type.__origin__ is list:
                item_schema_type = aliased_type.__args__[0] if aliased_type.__args__ else None
                return [expand_lists_in_data_for_llm(item, item_schema_type, parser) for item in data]

    if isinstance(data, dict):
        effective_type = schema_type.__args__[1] if hasattr(schema_type, "__args__") else schema_type
        return {k: expand_lists_in_data_for_llm(v, effective_type, parser) for k, v in data.items()}  # data_copy

    if isinstance(data, list):
        effective_type = schema_type.__args__[0] if hasattr(schema_type, "__args__") else schema_type
        return {str(i): expand_lists_in_data_for_llm(item, effective_type, parser) for i, item in enumerate(data)}  # data_copy

    return data


def expand_list(data: list) -> dict:
    return {str(i): item for i, item in enumerate(data)}


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

    Mirrors the traversal of `unexpand_lists_in_data_from_llm`; mutates dict/list
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


def unexpand_lists_in_data_from_llm(data: Any, schema_type, parser) -> Any:
    """
    Recursively traverse data and convert dictionaries with stringified integer keys
    back into lists if the corresponding schema indicates to do so, or if no schema is provided
    and the dictionary keys suggest it was an expanded list.

    This is the inverse of `expand_lists_in_data_for_llm`.
    This does not mutate the original data.

    Args:
        data (Any): The data to process (dict, list, or primitive).
        schema_type (ParsedSchemaClass | None): The type for the current data segment, if available.
        parser (SchemaParser): The SchemaParser instance for resolving type definitions.

    Returns:
        out (Any): A new data structure with specified dictionaries converted to lists.
    """
    from .schema_parser import ParsedSchemaClass, TYPE_MAP

    if schema_type in TYPE_MAP.values() or data is None:
        return data

    if isinstance(schema_type, ParsedSchemaClass):
        parsed_class_obj = schema_type

        if parsed_class_obj.definition_type == "dataclass" and isinstance(data, dict):
            data_copy = {}
            for k, v in data.items():
                field_schema = parsed_class_obj.get_field(k)
                current_item_schema_type = field_schema.type if field_schema and field_schema.type else None
                data_copy[k] = unexpand_lists_in_data_from_llm(v, current_item_schema_type, parser)
            return data_copy
        elif parsed_class_obj.definition_type == "alias":
            field_type = parsed_class_obj.get_field().type
            if parsed_class_obj.do_expand_into_dict and isinstance(data, dict) and _is_dict_expandable_to_list(data):
                list_item_schema_type = None
                if field_type and hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                    if hasattr(field_type, "__args__") and field_type.__args__:
                        list_item_schema_type = field_type.__args__[0]

                    # Ensure keys are sorted numerically for correct list order
                    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
                    return [
                        unexpand_lists_in_data_from_llm(item_data, list_item_schema_type, parser)
                        for _, item_data in sorted_items
                    ]  # data_copy

            if isinstance(data, dict):
                dict_value_schema_type = None
                if (
                    field_type
                    and hasattr(field_type, "__origin__")
                    and field_type.__origin__ is dict
                    and hasattr(field_type, "__args__")
                    and len(field_type.__args__) > 1
                ):
                    dict_value_schema_type = field_type.__args__[1]
                return {
                    k: unexpand_lists_in_data_from_llm(v, dict_value_schema_type, parser) for k, v in data.items()
                }  # data_copy

            if isinstance(data, list):
                list_item_schema_type = None
                if (
                    field_type
                    and hasattr(field_type, "__origin__")
                    and field_type.__origin__ is list
                    and hasattr(field_type, "__args__")
                    and field_type.__args__
                ):
                    list_item_schema_type = field_type.__args__[0]
                elif field_type:  # If it's a non-generic type for list items
                    list_item_schema_type = field_type

                return [unexpand_lists_in_data_from_llm(item, list_item_schema_type, parser) for item in data]  # data_copy

            return data

    if isinstance(data, dict):
        if _is_dict_expandable_to_list(data):
            sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
            return [unexpand_lists_in_data_from_llm(item_data, None, parser) for _, item_data in sorted_items]
        return {k: unexpand_lists_in_data_from_llm(v, None, parser) for k, v in data.items()}

    if isinstance(data, list):
        return [unexpand_lists_in_data_from_llm(item, None, parser) for item in data]

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
