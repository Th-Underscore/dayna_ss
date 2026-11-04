#!/usr/bin/env python3
"""Generate subjects_schema_noschema.json from subjects_schema_sceneagg.json.

Replaces all `{{ schema_snippet }}` template blocks with `#!/definitions/{{ schema_class }}`
pointers that resolve against the full schema embedded in the system context.
"""

import json
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "user_data" / "example" / "schemas"
SOURCE = SCHEMA_DIR / "subjects_schema_sceneagg.json"
OUTPUT = SCHEMA_DIR / "subjects_schema_noschema.json"

OLD = "```json\\n{{ schema_snippet }}\\n```"
NEW = "#!/definitions/{{ schema_class }}"


def main():
    raw = SOURCE.read_text(encoding="utf-8")
    count = raw.count(OLD)
    result = raw.replace(OLD, NEW)

    # Verify the replacement didn't break JSON validity
    json.loads(result)

    OUTPUT.write_text(result, encoding="utf-8")
    print(f"Replaced {count} schema_snippet blocks → {OUTPUT.name}")


if __name__ == "__main__":
    main()