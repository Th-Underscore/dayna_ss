from types import FunctionType
from typing import Any

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from .console import _RESET, _WARNING


def format_str(string: str, **kwargs) -> str:
    """Format a string with keyword arguments, calling functions as needed.

    Args:
        string (str): The string to format. It should contain placeholders like {key}.

    Returns:
        str: The formatted string.

    Examples:
        >>> format_str("Hello, {name}!", name="John Doe")
        'Hello, John Doe!'
        >>> format_str("The result is {result}.", result=lambda: print(42))
        42  # Only printed if {result} is present.
        # In a real scenario, the lambda would call a function that returns a value.
        'The result is None.'
    """
    for key, value in kwargs.items():
        if f"{{{key}}}" in string:
            if isinstance(value, FunctionType):
                value = value()
            string = string.replace(f"{{{key}}}", str(value))
    return string


_jinja_env = None


def _last_item(d: dict):
    """Return the last value from a dict by insertion order (Python 3.7+).

    Args:
        d: A dictionary

    Returns:
        The last value in the dict, or None if empty
    """
    if not d:
        return None
    return list(d.values())[-1]


def _scene_messages(metadata: list, scene_number: int | None = None):
    """Filter messages metadata to get only messages from a specific scene.

    Args:
        metadata: List of message metadata dicts
        scene_number: Optional scene number to filter by. If None, returns all.

    Returns:
        List of metadata dicts for the specified scene (or all if no scene_number)
    """
    if not metadata:
        return []
    if scene_number is None:
        return metadata
    return [m for m in metadata if m.get("scene_number") == scene_number]


def _get_param(metadata, param, include_system=False):
    """Extract a specific parameter from messages metadata.

    Args:
        metadata: List of message metadata dicts
        param: Parameter name to extract. Common options:
            - "speakers": List of unique speakers
            - "characters_present": Characters present in each message
            - "summary" / "text": Summary text from is_summary=True messages
            - "subjects_referenced": Dict with characters/groups/events
            - "scene_number": Scene numbers
        include_system: For "speakers" param only - whether to include "System"

    Returns:
        For most params: List of unique values
        For "summary"/"text": List of summary texts from is_summary=True messages
        For "subjects_referenced": Dict with aggregated characters/groups/events
    """
    if not metadata:
        return [] if param != "subjects_referenced" else {}

    if param in ("summary", "text"):
        return [m.get("text", "") for m in metadata if m.get("is_summary", False)]

    if param == "subjects_referenced":
        chars = set()
        groups = set()
        events = set()
        for m in metadata:
            refs = m.get("subjects_referenced", {})
            chars.update(refs.get("characters", []))
            groups.update(refs.get("groups", []))
            events.update(refs.get("events", []))
        return {"characters": sorted(chars), "groups": sorted(groups), "events": sorted(events)}

    unique_vals = set()
    for m in metadata:
        vals = m.get(param, [])
        if isinstance(vals, list):
            if param == "speakers" and not include_system:
                unique_vals.update(v for v in vals if v != "System")
            else:
                unique_vals.update(vals)
        elif vals:
            unique_vals.add(vals)

    if param == "speakers":
        return sorted(unique_vals)
    return sorted(unique_vals, key=lambda x: str(x))


def _get_jinja_env():
    """Get or create a cached Jinja environment."""
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.Undefined,
        )
        _jinja_env.filters['last_item'] = _last_item
        _jinja_env.filters['scene_messages'] = _scene_messages
        _jinja_env.filters['get_param'] = _get_param
        _jinja_env.filters['bracket'] = _bracket_dots
    return _jinja_env


def _bracket_dots(value: Any) -> str:
    """Wrap a path key in [brackets] when it contains a dot, for unambiguous paths.

    ``Mr. Peters`` renders as ``[Mr. Peters]`` (a single path component), while
    ``Evelyn`` renders unchanged. Used on format-template path markers so the
    LLM copies bracket-safe paths for dotted entity names.
    """
    s = str(value) if value is not None else ""
    return f"[{s}]" if "." in s else s


def render_jinja_template(template_str: str, **kwargs) -> str:
    """Render a Jinja2 template string with the given context variables.

    Args:
        template_str: The Jinja2 template string (e.g., "Scene {{ scene_number }}")
        **kwargs: Context variables for the template

    Returns:
        str: The rendered string

    Examples:
        >>> render_jinja_template("Scene {{ scene_number }}", scene_number=5)
        'Scene 5'
        >>> render_jinja_template("{{ min }}-{{ max }}", min=4, max=8)
        '4-8'
    """
    if not template_str:
        return ""

    # Call any lazy functions in kwargs
    resolved_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, FunctionType):
            resolved_kwargs[key] = value()
        else:
            resolved_kwargs[key] = value

    try:
        env = _get_jinja_env()
        template = env.from_string(template_str)
        return template.render(**resolved_kwargs).strip()
    except Exception as e:
        print(f"{_WARNING}Jinja template rendering failed: {e}{_RESET}")
        return template_str


def format_str_or_jinja(template_str: str, **kwargs) -> str:
    """Render a template string using either Jinja syntax or legacy Python format syntax.

    Detects whether the template uses Jinja syntax ({{ }}) or Python format syntax ({ }),
    and renders accordingly. This provides backward compatibility.

    Args:
        template_str: The template string
        **kwargs: Context variables

    Returns:
        str: The rendered string
    """
    if not template_str:
        return ""
    if "{{ " in template_str or "{%" in template_str:
        return render_jinja_template(template_str, **kwargs)
    return format_str(template_str, **kwargs)
