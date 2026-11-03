"""Generate the full merged JSON schema (all subject definitions) and write to disk.

Usage: python scripts/dump_full_schema.py
Output: scripts/all_defs.schema.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.schema_parser import SchemaParser

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "user_data/example/schemas/subjects_schema_sceneagg.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "all_defs.schema.json"


def main():
    parser = SchemaParser(str(SCHEMA_PATH))

    all_defs = {}
    for name, cls in parser.subjects.items():
        schema = parser.get_relevant_json_schema_definitions(cls)
        if "definitions" in schema:
            all_defs.update(schema["definitions"])

    full = {"definitions": all_defs}
    output = json.dumps(full, indent=2, ensure_ascii=False)

    OUTPUT_PATH.write_text(output + "\n")
    print(f"{len(all_defs)} definitions, {len(output)} chars, ~{len(output) // 4} tokens")
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()