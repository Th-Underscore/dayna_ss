"""Draft prefix-cache-optimized prompt templates for subjects_schema_sceneagg.json.

Reorders each template's blocks by the rule: static-common-first, branch-specific
middle, branch-stable variables (branch_name/branch_list/entry_name) next,
turn-dynamic variables (scene_recap/scene_events/current_message_node) last.
IDENTITY BINDING moved near the BOTTOM. Text is never retyped: every block is
EXTRACTED verbatim from the original template and reassembled, so the file's
mixed real-em-dash / literal-\\u2014 escapes are preserved exactly.

Explicit behavioral changes (all flagged for review):
  1. The "This is a MEMORY-MANAGEMENT TASK..." footer moves to the top as a
     lead-in for query-type templates, and its tail line becomes
     "Re-read the instruction below and complete ONLY the requested task."
  2. Discovery queries of the events family (StoryEvents/PastEvents/CrucialEvents)
     GAIN that lead (they lacked it) to unify the family prefix.
  3. "{{ branch_name }}" is lifted out of the aperture sentence into a trailing
     "List to update: '{{ branch_name }}'" line.
  4. Apertures are canonicalized to be fully GENERIC ("...any new entries...
     should be added to the list?" / "generate the full data for the requested
     new entry..."); the specific subject is clarified only near the bottom via
     a "Subject being populated: <label>." line plus each branch's own guidance.
  5. "If new characters/groups/elements/events are identified" is canonicalized to
     "If new entries are identified" so the shared slab is contiguous.
  6. GroupMap gate-check "(all groups)" stays verbatim but moves to a trailing
     "Section scope:" line next to "Section to review: ...".
All guidance paragraphs remain verbatim, only repositioned.
"""
import json
import re

SRC = "user_data/example/schemas/subjects_schema_sceneagg.json"
DST = "user_data/example/schemas/subjects_schema_sceneagg.prefixcache.json"

schema = json.load(open(SRC, encoding="utf-8"))
NEW = {}


def d(defname):
    return (schema["definitions"].get(defname) or {}).get("defaults") or {}


def t(defname, key):
    return d(defname)[key]


def extract(orig, start, end):
    """Extract exact substring from start-exclusive-anchor through end-inclusive."""
    si = orig.find(start)
    if si < 0:
        raise ValueError(f"start not found: {start[:60]!r}")
    ei = orig.find(end, si) + len(end)
    if ei < si:
        raise ValueError(f"end not found after start: {end[:60]!r}")
    return orig[si:ei]


def has(orig, sub):
    return sub in orig


def paragraphs(orig):
    return orig.split("\n\n")


def para(bs, frag):
    """Return the paragraph (verbatim) containing `frag`."""
    for i, p in enumerate(bs):
        if frag in p:
            return i, p
    raise ValueError(f"paragraph not found: {frag[:60]!r}")


def ok(s, *frags):
    for f in frags:
        if f not in s:
            raise ValueError(f"expected fragment missing: {f[:60]!r}")
    return s


def set_new(defname, key, text):
    NEW[f"{defname}.{key}"] = text


# --- shared verbatim fragments pulled from originals (escaping-safe) ---
_ch_ne_q = t("CharacterMap", "new_entry_query_prompt_template")
_footer_orig = extract(_ch_ne_q, "This is a MEMORY-MANAGEMENT TASK",
                       "complete ONLY that instruction.")
LEAD = _footer_orig.replace(
    "Re-read the instruction at the top of this prompt and complete ONLY that instruction.",
    "Re-read the instruction below and complete ONLY the requested task.")
SCENE = extract(_ch_ne_q, "SCENE CONTEXT (what has actually happened so far",
                "{{ scene_events }}")
RESP_ARRAY = extract(_ch_ne_q, "Please provide your response in the specified JSON array format.",
                     "An empty array is acceptable but not ideal.")
IF_NEW_CANON = "If new entries are identified, provide their names as a JSON list of strings."
# Fully generic apertures shared verbatim by every member of the family; the
# subject is clarified ONLY in the per-branch bottom section.
APERTURE_GENERIC = ("In the latest exchange(s) (the most recent user and assistant messages, "
                    "whether or not a scene boundary has been crossed), have there been any new "
                    "entries mentioned or implied that should be added to the list?")
ENTRY_GENERIC = ("Based on all of the information in the last scene, generate the full data for "
                 "the requested new entry described here.")
RESP_ONLY_ARRAY = extract(_ch_ne_q, "Respond ONLY with the requested JSON array of entry name(s),",
                          "or exactly NO.")
BRANCH_LIST = extract(_ch_ne_q, "Here is the current list of existing entries:",
                      "{{ branch_list }}")
NAME_EXACTLY = extract(_ch_ne_q, "NAME EXACTLY: Use the exact name the story uses",
                       "'quince paste' or 'jar').")

_id_binding_orig = t("CharacterMap", "new_entry_prompt_template")
IDENTITY_BINDING = extract(_id_binding_orig, "IDENTITY BINDING", "dialogue or biography.")


# ===========================================================================
# TIER 1a — new_entry_query_prompt_template
# ===========================================================================
def discovery(defname, subject_label, guidance_extract, example_anchor,
              remember_anchor=None, extra_guidance_anchor=None, has_recap=True, name_exactly=True):
    q = t(defname, "new_entry_query_prompt_template")
    bs = paragraphs(q)
    guidance = None
    for p in bs:
        if guidance_extract in p:
            guidance = p
            break
    if guidance is None:
        raise ValueError(f"[{defname}] guidance paragraph not found: {guidance_extract[:40]!r}")
    example = None
    for p in bs:
        if example_anchor in p:
            example = p
            break
    if example is None:
        raise ValueError(f"[{defname}] example paragraph not found: {example_anchor[:40]!r}")
    subject_line = f"Subject being populated: {subject_label}."
    parts = [LEAD, APERTURE_GENERIC, RESP_ARRAY, IF_NEW_CANON, subject_line, guidance]
    if extra_guidance_anchor:
        for p in bs:
            if extra_guidance_anchor in p:
                parts.append(p)
                break
    parts.append(example)
    if remember_anchor:
        for p in bs:
            if remember_anchor in p:
                parts.append(p)
                break
    if name_exactly:
        parts.append(NAME_EXACTLY)
    parts += [f"List to update: '{{{{ branch_name }}}}'", BRANCH_LIST, RESP_ONLY_ARRAY]
    if has_recap:
        parts.append(SCENE)
    return "\n\n".join(parts)


# CharacterMap
set_new("CharacterMap", "new_entry_query_prompt_template", discovery(
    "CharacterMap", "characters",
    "Any new characters should be added to the list.",
    '["David Smith", "Mysterious Stranger", "The Captain"]',
    remember_anchor="REMEMBER: _Any_ relevant character"))

# GroupMap
set_new("GroupMap", "new_entry_query_prompt_template", discovery(
    "GroupMap", "groups",
    "Any and all new groups, small groups, crowds, etc.",
    '["The Survivors", "Marauder Gang X"]'))

# ElementMap
set_new("ElementMap", "new_entry_query_prompt_template", discovery(
    "ElementMap", "elements",
    "Elements include creatures (monsters, animals, beings)",
    '["The Old Bunker", "Subterranean Patrol", "Emergency Radio"]',
    extra_guidance_anchor="WHAT NOT TO INCLUDE in elements:"))

# StoryEvents / PastEvents / CrucialEvents  (no recap originally — keep lean, gain LEAD)
def discovery_lean(defname, subject_label, guidance_anchor, example_anchor):
    q = t(defname, "new_entry_query_prompt_template")
    bs = paragraphs(q)
    guidance = next(p for p in bs if guidance_anchor in p)
    example = next(p for p in bs if example_anchor in p)
    subject_line = f"Subject being populated: {subject_label}."
    parts = [LEAD, APERTURE_GENERIC, RESP_ARRAY, IF_NEW_CANON, subject_line, guidance,
             example, NAME_EXACTLY,
             f"List to update: '{{{{ branch_name }}}}'", BRANCH_LIST, RESP_ONLY_ARRAY]
    return "\n\n".join(parts)


set_new("StoryEvents", "new_entry_query_prompt_template", discovery_lean(
    "StoryEvents", "significant story events",
    "Any significant events should be added to the list.",
    '["The Ambush at Dawn", "Discovery of the Old Bunker"]'))

set_new("PastEvents", "new_entry_query_prompt_template", discovery_lean(
    "PastEvents", "backstory events",
    "Significant past events should be added to the list",
    '["Melissa\'s Anguish", "The Founding of the New Order"]'))

set_new("CrucialEvents", "new_entry_query_prompt_template", discovery_lean(
    "CrucialEvents", "major story events",
    "Any significant events should be added to the list.",
    '["The Ambush at Dawn", "Discovery of the Old Bunker"]'))

# Scenes — verbatim (already near-optimal; unique single-archive semantics)
set_new("Scenes", "new_entry_query_prompt_template", t("Scenes", "new_entry_query_prompt_template"))

# Arcs / Chapters — move dynamic {{ formatted_data }} block after instructions
for defname, title in [("Arcs", "new narrative arc"), ("Chapters", "new chapter")]:
    q = t(defname, "new_entry_query_prompt_template")
    blocks = q.split("\n\n")
    active_i = next(i for i, b in enumerate(blocks) if "{{ formatted_data }}" in b)
    active = blocks.pop(active_i)
    print(f"[{defname}] reordered active-block")
    set_new(defname, "new_entry_query_prompt_template", "\n\n".join([blocks[0]] +
                                                                    [b for b in blocks[1:-1]] +
                                                                    [active] + [blocks[-1]]))

# ===========================================================================
# TIER 1b — new_entry_prompt_template
# ===========================================================================
def gen_entry(defname, subject_label, body_extracts, tail_extracts, tail_literals=()):
    """body/tail: (start,end) extracts from the original template."""
    q = t(defname, "new_entry_prompt_template")
    schema_block = "Relevant Schema:\n```json\n{{ schema_snippet }}\n```"
    example_block = "Example JSON structure:\n```json\n{{ example_json }}\n```"
    subject_line = f"Subject being populated: {subject_label}."
    parts = [ENTRY_GENERIC, schema_block, example_block,
             "Please provide your response as a complete JSON object for the requested entry.",
             subject_line]
    for e in body_extracts:
        parts.append(extract(q, e[0], e[1]))
    for e in tail_extracts:
        parts.append(extract(q, e[0], e[1]))
    for lit in tail_literals:
        parts.append(lit)
    return "\n\n".join(parts)


IMP_CHAR = ("IMPORTANCE: Assign this character an importance score", "so full detail is unnecessary.")
IMP_GROUP = ("IMPORTANCE: Assign this group an importance score", "so full detail is unnecessary.")
IMP_EVENT = ("IMPORTANCE: Assign this event an importance score", "render as one-line roster entries.")
IMP_ELEM = ("IMPORTANCE: Assign this element an importance score", "so full detail is unnecessary.")

set_new("CharacterMap", "new_entry_prompt_template", gen_entry(
    "CharacterMap", "characters",
    [IMP_CHAR],
    [("The entry to populate: '{{ entry_name }}'", "'{{ entry_name }}'")],
    [IDENTITY_BINDING]))

set_new("GroupMap", "new_entry_prompt_template", gen_entry(
    "GroupMap", "groups",
    [IMP_GROUP],
    [("The entry to populate: '{{ entry_name }}'", "{{ entry_name }}")]))

set_new("ElementMap", "new_entry_prompt_template", gen_entry(
    "ElementMap", "elements",
    [("Determine the 'kind' based on what this element is:", "- 'resource': A consumable or material (water, food, ammo, fuel)"),
     IMP_ELEM],
    [("The entry to populate: '{{ entry_name }}'", "{{ entry_name }}")]))

for defname in ("StoryEvents", "CrucialEvents"):
    set_new(defname, "new_entry_prompt_template", gen_entry(
        defname, "story events",
        [IMP_EVENT],
        [("The entry to populate: '{{ entry_name }}'", "{{ entry_name }}")],
        ["Current message node: {{ current_message_node }}"]))

set_new("PastEvents", "new_entry_prompt_template", gen_entry(
    "PastEvents", "backstory events",
    [IMP_EVENT],
    [("The entry to populate: '{{ entry_name }}'", "{{ entry_name }}")],
    ["Current message node: {{ current_message_node }}"]))

set_new("Scenes", "new_entry_prompt_template", gen_entry(
    "Scenes", "complete scenes",
    [("IMPORTANT: This scene was just completed", "preserve narrative history.")],
    [("The entry to populate: '{{ entry_name }}'", "{{ entry_name }}")],
    ["Current message node: {{ current_message_node }}"]))

# Arcs/Chapters — dynamic characters list last
for defname in ("Arcs", "Chapters"):
    q = t(defname, "new_entry_prompt_template")
    blocks = q.split("\n\n")
    cap_i = next(i for i, b in enumerate(blocks) if "{{ subjects.characters.entries }}" in b)
    first = blocks[0]
    rest = [b for b in blocks[1:] if "{{ subjects.characters.entries }}" not in b]
    set_new(defname, "new_entry_prompt_template",
            "\n\n".join([first] + rest + ["Characters involved:\n{{ subjects.characters.entries }}"]))

# ===========================================================================
# TIER 1c — branch_update_prompt_template
def branch_update(defname):
    q = t(defname, "branch_update_prompt_template")
    bs = paragraphs(q)
    # Paragraph 0 is the intro; for some members the "Relevant Schema:" block is
    # glued to it (no blank line) — split it off cleanly. Characters carry the
    # IDENTITY BINDING block up front; pull that out separately.
    ib = extract(q, "IDENTITY BINDING", "dialogue or biography.") if "IDENTITY BINDING" in q else None
    p0 = bs[0]
    if ib and p0.startswith("IDENTITY BINDING"):
        p0 = bs[1]
    if "Relevant Schema:" in p0:
        idx = p0.find("Relevant Schema:")
        intro = p0[:idx].rstrip()
        schema_para = p0[idx:]
    else:
        intro = p0
        for p in bs:
            if "Relevant Schema:" in p:
                schema_para = p
                break
    example_para = next(p for p in bs if "Example JSON structure:" in p)
    i_ex = bs.index(example_para)
    scene_i = next((i for i, p in enumerate(bs) if p.startswith("SCENE CONTEXT") and i > i_ex),
                   len(bs))
    entry_i = next((i for i, p in enumerate(bs) if p.startswith("Entry to review") and i > i_ex),
                   len(bs))
    stop = min(scene_i, entry_i)
    body = bs[i_ex + 1:stop]
    before_entry = "\n\n".join(body)

    recap = SCENE if "SCENE CONTEXT" in q else None
    footer = extract(q, "This is a MEMORY-MANAGEMENT TASK", "complete ONLY that instruction.") \
        if "This is a MEMORY-MANAGEMENT TASK" in q else None

    parts = [intro, schema_para, example_para, before_entry,
             "Entry to review: '{{ branch_name }}'"]
    if ib:
        parts.append(ib)
    if recap:
        parts.append(recap)
    if footer:
        parts.append(footer.replace("Re-read the instruction at the top of this prompt",
                                    "Re-read the instruction below"))
    return "\n\n".join(p for p in parts if p)


for defname in ("Character", "Group", "Element", "GeneralInfo", "SceneState", "StoryEvents"):
    try:
        set_new(defname, "branch_update_prompt_template", branch_update(defname))
    except ValueError as e:
        print(f"[WARN] branch_update({defname}) failed: {e}")

# ===========================================================================
# TIER 2 — gate_check_prompt_template
# ===========================================================================
def gate_map(defname):
    q = t(defname, "gate_check_prompt_template")
    base = extract(q, "Based on the latest exchange, does the entire section/category",
                   "Otherwise, respond with 'YES'.")
    base = base.replace("section/category (all groups) require a detailed review",
                        "section/category require a detailed review")
    scope = extract(q, "(all groups)", "(all groups)") if "(all groups)" in q else None
    footer = extract(q, "This is a MEMORY-MANAGEMENT TASK", "Respond ONLY with: YES or NO")
    recap = SCENE if "SCENE CONTEXT" in q else None
    parts = [base, footer, f"Section to review: '{{{{ branch_name }}}}'"]
    if scope:
        parts.append(f"Section scope: {scope}")
    if recap:
        parts.append(recap)
    return "\n\n".join(p for p in parts if p)


for defname in ("CharacterMap", "GroupMap", "ElementMap"):
    set_new(defname, "gate_check_prompt_template", gate_map(defname))

# Events / GeneralInfo — move recap last
def simple_gate(defname):
    q = t(defname, "gate_check_prompt_template")
    bs = paragraphs(q)
    first = bs[0]
    footer = extract(q, "This is a MEMORY-MANAGEMENT TASK", "Respond ONLY with: YES or NO") if "This is a MEMORY-MANAGEMENT TASK" in q else None
    section = extract(q, "Section to review:", "{{ branch_name }}") if "Section to review:" in q else None
    recap = SCENE if "SCENE CONTEXT" in q else None
    parts = [first]
    if footer:
        parts.append(footer)
    if section:
        parts.append(f"Section to review: '{{{{ branch_name }}}}'")
    if recap:
        parts.append(recap)
    return "\n\n".join(p for p in parts if p)


set_new("Events", "gate_check_prompt_template", simple_gate("Events"))
set_new("GeneralInfo", "gate_check_prompt_template", simple_gate("GeneralInfo"))

# Arcs / Chapters — counters to tail
def gate_arcchap(defname):
    q = t(defname, "gate_check_prompt_template")
    bs = paragraphs(q)
    counter = None
    for b in bs:
        if "{{ chapter_in_arc }}" in b or "{{ scene_in_chapter }}" in b:
            counter = b
            break
    counter_i = bs.index(counter) if counter else 0
    scene_i = next(i for i, b in enumerate(bs) if b.startswith("SCENE CONTEXT"))
    footer_i = next(i for i, b in enumerate(bs) if b.startswith("This is a MEMORY"))
    static = [b for i, b in enumerate(bs) if i not in {0, counter_i, scene_i, footer_i}]
    out = [bs[0]] + static
    if counter and counter_i != 0:
        out.append(counter)
    out.append(bs[scene_i])
    out.append(bs[footer_i])
    return "\n\n".join(b for b in out if b)


set_new("Arcs", "gate_check_prompt_template", gate_arcchap("Arcs"))
set_new("Chapters", "gate_check_prompt_template", gate_arcchap("Chapters"))

# ===========================================================================
# TIER 3 — select_entries_to_update + branch_query
# ===========================================================================
def select_entries(defname):
    q = t(defname, "select_entries_to_update_prompt_template")
    bs = paragraphs(q)
    scene_i = next(i for i, p in enumerate(bs) if p.startswith("SCENE CONTEXT"))
    front = bs[:scene_i]  # question + guidance + respond-array + empty-array
    footer = extract(q, "This is a MEMORY-MANAGEMENT TASK",
                     "Respond ONLY with the requested JSON array of entry names, or exactly NO.")
    parts = front + [BRANCH_LIST, f"Section to review: '{{{{ branch_name }}}}'", footer, SCENE]
    return "\n\n".join(parts)


for defname in ("CharacterMap", "GroupMap", "ElementMap"):
    set_new(defname, "select_entries_to_update_prompt_template", select_entries(defname))


def branch_query(defname):
    q = t(defname, "branch_query_prompt_template")
    bs = paragraphs(q)
    ib = extract(q, "IDENTITY BINDING", "dialogue or biography.") if "IDENTITY BINDING" in q else None
    first = bs[1] if (ib and bs[0].startswith("IDENTITY BINDING")) else bs[0]
    item = extract(q, "Entry to review:", "{{ branch_name }}") if "Entry to review:" in q else None
    footer = extract(q, "This is a MEMORY-MANAGEMENT TASK", "complete ONLY that instruction.") if "This is a MEMORY-MANAGEMENT TASK" in q else None
    recap = SCENE if "SCENE CONTEXT" in q else None
    parts = [first]
    if item:
        parts.append(f"Entry to review: '{{{{ branch_name }}}}'")
    if footer:
        parts.append(footer.replace("Re-read the instruction at the top of this prompt",
                                    "Re-read the instruction below"))
    if ib:
        parts.append(ib)
    if recap:
        parts.append(recap)
    return "\n\n".join(p for p in parts if p)


for defname in ("Character", "Group", "Element", "SceneState", "StoryEvents", "GeneralInfo"):
    set_new(defname, "branch_query_prompt_template", branch_query(defname))

# ---------------------------------------------------------------------------
# Verify: variable-token parity old vs new
# ---------------------------------------------------------------------------
VAR = re.compile(r"\{\{([^{}]+)\}\}")


def tokens(s):
    return sorted(VAR.findall(s))


bad = 0
for key, newtext in NEW.items():
    defname, k = key.rsplit(".", 1)
    oldtext = t(defname, k)
    ot, nt = tokens(oldtext), tokens(newtext)
    if ot != nt:
        bad += 1
        print(f"[TOKEN MISMATCH] {key}\n  old={ot}\n  new={nt}")

print(f"\nToken parity: {len(NEW) - bad}/{len(NEW)} OK" if bad else f"\nToken parity: {len(NEW)}/{len(NEW)} OK")

# persist draft schema
draft = json.loads(json.dumps(schema))
count = 0
for key, newtext in NEW.items():
    defname, k = key.rsplit(".", 1)
    draft["definitions"][defname]["defaults"][k] = newtext
    count += 1
with open(DST, "w", encoding="utf-8") as f:
    json.dump(draft, f, indent=2, ensure_ascii=False)
print(f"Wrote {DST} ({count} templates modified)")