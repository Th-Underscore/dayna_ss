import json
from os import PathLike
from pathlib import Path

import jsonc

from .console import _ERROR, _RESET


def load_json(file_path: PathLike, verbose: bool = False) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return jsonc.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        if verbose:
            print(f"Warning: Could not load {file_path}, returning empty dict")
        return {}


def save_json(data: dict, file_path: PathLike) -> bool:
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"{_ERROR}Error saving {file_path}: {str(e)}{_RESET}")
        return False


def validate_path(path: PathLike) -> Path | None:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return None
    return path
