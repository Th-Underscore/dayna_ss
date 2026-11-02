"""Build a full tree of dayna_ss subjects and their descendants in execution order.

Walks subjects_schema_sceneagg.json using the real SchemaParser, in the same order
DataSummarizer traverses it (_update_recursive -> _traverse_structure ->
_process_field). Annotates each node with node kind, LLM-calling triggers, growth
behavior (add_new), iteration shape, and a prefix-cache dynamicity classification.
"""
import sys

sys.path.insert(0, "/mnt/c/Users/there/Downloads/Projects/Programming/Python/textgen/extensions/dayna_ss")

from utils.schema_parser import SchemaParser, ParsedSchemaClass, Action

SCHEMA = "/mnt/c/Users/there/Downloads/Projects/Programming/Python/textgen/extensions/dayna_ss/user_data/example/schemas/subjects_schema_sceneagg.json"

ACTION_NAMES = {int(a): a.name for a in Action}
TRIGGER_NAMES = {0: "always", 1: "on_new_scene", 2: "on_existing_scene"}


def trig_summary(sc):
    """[(trigger_name, [action_desc])] for a schema class trigger map.

    action_desc includes the config hints that dictate the LLM call: which prompt
    template is used, `skip_query`, `when=gate_check_fail`.
    """
    out = []
    tm = getattr(sc, "trigger_map", None) or {}
    for tr, actions in tm.items():
        descs = []
        for a, cfg in actions:
            d = ACTION_NAMES.get(int(a), str(a))
            hints = []
            if isinstance(cfg, dict):
                if cfg.get("skip_query"):
                    hints.append("skip_query")
                if cfg.get("when"):
                    hints.append(f"when={cfg.get('when')}")
                pt = cfg.get("prompt_template")
                if pt:
                    hints.append(f"tpl={pt}")
            if hints:
                d += "(" + ",".join(hints) + ")"
            descs.append(d)
        if descs:
            out.append((TRIGGER_NAMES.get(int(tr), str(tr)), descs))
    return out


def grows(sc):
    tm = getattr(sc, "trigger_map", None) or {}
    return any(a == Action.ADD_NEW for acts in tm.values() for a, _ in acts)


def type_label(t):
    return t.__name__ if isinstance(t, type) else str(t)


def container_of(t):
    origin = getattr(t, "__origin__", None)
    args = getattr(t, "__args__", tuple())
    if origin is list and args:
        return "list", args[0]
    if origin is dict and len(args) > 1:
        return "dict", args[1]
    return None


class Node:
    def __init__(self, path, name, kind, field=None, triggers=None):
        self.path = path
        self.name = name
        self.kind = kind
        self.field = field
        self.triggers = triggers or []
        self.grows = False
        self.container = None
        self.children = []

    def add(self, c):
        self.children.append(c)
        return c


def walk(node, sc, seen=None, path=None):
    """Descend one schema class, mirroring _update_recursive/_traverse_structure.

    `path` is the data path string for the node's children ('' at root).
    """
    if path is None:
        path = node.path
    seen = (seen or []) + [sc.name]
    if sc.name in seen[:-1]:
        node.kind = "cycle"
        return
    node.triggers = trig_summary(sc)
    node.grows = grows(sc)

    if sc.definition_type == "dataclass":
        for f in sc.get_fields() or []:
            if f.name.startswith("_") or f.no_update:
                fpath = (path + "." if path else "") + f.name
                node.add(Node(fpath, f"{f.name} [skipped]", "leaf-internal"))
                continue
            t = f.type
            fpath = (path + "." if path else "") + f.name
            child = Node(fpath, f.name, "", field=f)
            node.add(child)
            if isinstance(t, ParsedSchemaClass):
                child.kind = f"dataclass -> {t.name}"
                walk(child, t, seen, path=fpath)
            else:
                c = container_of(t)
                if c:
                    kind, val = c
                    if isinstance(val, ParsedSchemaClass):
                        child.kind = f"{kind}[*] of {val.name}"
                        child.container = (kind, "key" if kind == "dict" else "idx")
                        j = Node(fpath + ("{<key>}" if kind == "dict" else "[i]"), val.name, "container-entry")
                        child.add(j)
                        walk(j, val, seen, path=fpath + ("{<key>}" if kind == "dict" else "[i]"))
                    else:
                        child.kind = f"{kind}[{type_label(val)}] leaf"
                else:
                    child.kind = f"leaf {type_label(t)}"
    else:
        # alias -> unwrap
        wrapped = sc._field.type
        if isinstance(wrapped, ParsedSchemaClass):
            node.kind = f"alias -> {wrapped.name}"
            walk(node, wrapped, seen, path=path)
        elif hasattr(wrapped, "__origin__"):
            c = container_of(wrapped)
            if c:
                kind, val = c
                if isinstance(val, ParsedSchemaClass):
                    node.kind = f"alias {kind}[*] of {val.name}"
                    node.container = (kind, "key" if kind == "dict" else "idx")
                    j = Node(path + ("{<key>}" if kind == "dict" else "[i]"), val.name, "container-entry")
                    node.add(j)
                    walk(j, val, seen, path=path + ("{<key>}" if kind == "dict" else "[i]"))
                else:
                    node.kind = f"alias {kind}[{type_label(val)}] leaf"
            else:
                node.kind = f"alias -> {type_label(wrapped)}"
        else:
            node.kind = f"alias -> {type_label(wrapped)}"


def print_node(node, depth, inherit_path=""):
    mark = []
    if node.triggers:
        tstr = ", ".join(f"{tr}: {','.join(acts)}" for tr, acts in node.triggers)
        mark.append("TRIGGERS {" + tstr + "}")
    if node.grows:
        mark.append("GROWS")
    if node.container:
        mark.append("one LLM cycle per entry")
    md = "  <" + " | ".join(mark) + ">" if mark else ""
    print(f"{'    ' * depth}- [{node.path}] {node.name} ({node.kind}){md}")
    for c in node.children:
        print_node(c, depth + 1)


def main():
    parser = SchemaParser(SCHEMA)
    print("SUBJECTS SCHEMA TREE — execution order — subjects_schema_sceneagg.json\n")
    for i, (subj_name, subj_cls) in enumerate(parser.subjects.items(), 1):
        print("#" * 72)
        print(f"SUBJECT {i}/{len(parser.subjects)}: '{subj_name}'  -> {subj_cls.name}")
        route = (parser.subject_routing or {}).get(subj_name, "")
        if route:
            print(f"  routing: {route}")
        root = Node(subj_name, subj_name, subj_cls.definition_type)
        walk(root, subj_cls, path=subj_name)
        print_node(root, depth=0)
        print()


if __name__ == "__main__":
    main()