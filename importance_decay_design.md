# Importance Decay & Convergence Design

---

## 1. Terminology (Locked)

| Term | Definition | Scope | Example |
|------|------------|-------|---------|
| **Importance** | Numerical magnitude (0–100) of a single relationship *edge* | Edge-level, static, stored in JSON | John→Paul importance=95 |
| **Relevance** | Computed significance of an *entity* in the current scene/context | Entity-level, dynamic, computed at query-time | Sarah relevance=78 for this scene |
| **Faction** | Direction of a relationship: `"positive"`, `"negative"`, or `"neutral"` | Edge-level, static | "positive" for ally, "negative" for enemy |
| **Effective importance** | Importance after path-min decay (intermediate value) | Path-level, computed per-path | 95 → 89 after depth-2 decay |
| **Path** | Sequence of edges traversed from a scene-character to a discovered entity | Traversal-level | John→Paul→Sarah = path length 2 |
| **Vouch** | Degree to which a source entity validates a target entity (0–1) | Source→target, computed | source_relevance × edge_importance / 10000 |
| **Pull** | Collective nudge from all vouching sources toward their relevance level | Entity-level, computed | 3 sources pull target from 17 to 53 |
| **Depth** | Hop count from the initial scene-entity set | Traversal-level | depth 0 = scene characters |

---

## 2. 10-Level Importance System

Importance = magnitude (0–100). Faction = direction (positive/negative/neutral).
A level-8 positive and level-8 negative share the same magnitude (84) but opposite factions.

### Level Table

| Lvl | Score | Label | Positive meaning | Negative meaning |
|-----|-------|-------|------------------|------------------|
| 10 | **100** | **Obsessive** | Would sacrifice everything — morals, life, existence. The bond IS their being. | All-consuming vendetta. Cannot coexist. Revenge defines purpose. |
| 9 | **92** | **Paramount** | Would die for them. "My person." Peak of normal human bonds. | Mortal enemy. The face of all they oppose. Priority one. |
| 8 | **84** | **Profound** | Ride-or-die. Battle-forged trust. Entrusted with life & secrets. | Intense hatred. Invested in their ruin. Active malice. |
| 7 | **75** | **Strong** | Close friend / beloved family. Takes real risk. Actively maintained. | Bitter grudge. Real conflict history. Celebrates their failures. |
| 6 | **65** | **Significant** | Trusted comrade. Reliable ally. Mutual respect. | Open antagonist. Consistent opposition. |
| 5 | **54** | **Notable** | Regular positive associate. Working relationship with weight. | Rival. Competitive friction. |
| 4 | **42** | **Moderate** | Friendly acquaintance. Get along when paths cross. | Mild dislike. Minor real friction. |
| 3 | **30** | **Mild** | Cordial. "Seems decent." Favorable but distant. | "Something off." Vague wariness. |
| 2 | **17** | **Slight** | Barely registers. Positive periphery. | Occasional minor irritation. Near-neutral. |
| 1 | **6** | **Negligible** | Default untracked state. | One off-putting interaction. |
| 0 | 0 | **None** | No relationship exists. | — |

### Score Distribution Rationale

- Gaps shrink at the high end (92→100: 8pts) because narrative distance between
  "paramount" and "obsessive" is small — both are extreme, just different flavors.
- Gaps widen at the low end (17→30: 13pts, 30→42: 12pts) because sub-L4
  relationships are practically irrelevant for narrative context.
- L7 (75) is the entry point for the current `min_importance=75` default threshold.
- L4 (42) is the "broad discovery" threshold for convergence pass 1.

### Mapping Existing Example Data

| Example value | Maps to level | Notes |
|--------------|---------------|-------|
| 95 (John→Paul, brother) | L9 | Paramount bond, correct |
| 90 (John→Amy, friend) | L8↗ | Near-profound friendship |
| 70 (John→Paul/Amy, comrade) | L6↗ | Slightly high for generic comrade, should be L6 |
| 60 (John→Marcus, comrade/strategist) | L5↗ | Notable-to-significant range |
| 55 (John→Elena, comrade/operative) | L5 | Notable associate |
| 50 (John→Angelina, comrade) | L5↘ | Notable-low |
| 10 (John→Angelina, victim's sister) | L2 | Slight negative, correct |

---

## 3. Path-Min Gated Decay (P4 — Selected as Base)

The effective importance of a discovered entity via a single path is:

```
path_min = min(edge.importance for edge in path)
α(imp) = (imp / 100) ^ 0.4
effective_imp = path_min × (α(path_min) ^ max(0, depth − 1))
```

### Alpha Mapping

| Importance | α = (imp/100)^0.4 |
|-----------|-------------------|
| 100 (L10) | 1.000 |
| 92 (L9) | 0.967 |
| 84 (L8) | 0.932 |
| 75 (L7) | 0.888 |
| 65 (L6) | 0.838 |
| 54 (L5) | 0.777 |
| 42 (L4) | 0.710 |
| 30 (L3) | 0.617 |
| 17 (L2) | 0.492 |
| 6 (L1) | 0.324 |

### Examples

| Path | path_min | α | depth | effective | ≥75? |
|------|----------|---|-------|-----------|------|
| [95, 95, 95] | 95 | 0.97 | 2 | 89 | ✓ |
| [84, 84] | 84 | 0.93 | 1 | 78 | ✓ |
| [92, 75] | 75 | 0.89 | 1 | 67 | ✗ |
| [84, 75] | 75 | 0.89 | 1 | 67 | ✗ |
| [75, 75, 75] | 75 | 0.89 | 2 | 59 | ✗ |
| [65, 92] | 65 | 0.84 | 1 | 55 | ✗ |
| [84, 84, 84] | 84 | 0.93 | 2 | 73 | ✗ |

Key behavior: L7 (75) chains survive at depth 0, fail at depth 1. L8 (84) chains
survive at depth 1, barely fail at depth 2 (73 < 75). L9+ (92+) chains survive
to depth 3+.

---

## 4. Convergence Nudge Model

### Problem

Current `traverse_graph` uses sets — entities are binary present/absent.
An entity reached via 3 independent paths is treated identically to an entity
reached via 1 path. Network prominence is invisible.

### Principle

Convergence should *nudge* an entity's relevance toward the relevance level of
its vouching sources, not apply a flat multiplier. The nudge strength depends
on both the number of paths and the credibility of each vouch.

### Formulas

#### Vouch Strength

Each source vouches for a target entity with strength proportional to the
source's own relevance and the importance of the connecting edge:

```
vouch_i = (source_relevance_i / 100) × (edge_importance_i / 100)
```

Range: 0–1. A high-relevance source (84) with a strong edge (75) → 0.63.
A low-relevance source (30) with a weak edge (17) → 0.05.

#### Pull Target

The weighted average of source relevances, weighted by vouch strength:

```
pull_target = Σ(source_relevance_i × vouch_i) / Σ(vouch_i)
```

Pull target is always between the min and max source relevance, weighted toward
the high-relevance cluster.

#### Pull Strength

The degree of nudge, with diminishing returns:

```
pull_strength = min(1.0, Σ(vouch_i) × pull_rate)
```

`pull_rate` is a configurable constant (default: ~0.4).

#### Final Relevance

```
relevance = base_relevance + (pull_target − base_relevance) × pull_strength
```

An entity is nudged from its best single-path `base_relevance` toward the
vouching cluster's center. The nudge is fractional — no path-independent jumps.

### Worked Examples (pull_rate = 0.4)

#### Ex1: 3 MED sources, MED edges → LOW target (17)

| Source | source_rel | edge_imp | vouch |
|--------|-----------|---------|-------|
| A | 54 (L5) | 54 (L5) | 0.29 |
| B | 54 (L5) | 54 (L5) | 0.29 |
| C | 54 (L5) | 54 (L5) | 0.29 |

```
Σvouch = 0.87
pull_target = 54
pull_strength = min(1.0, 0.87 × 0.4) = 0.35
relevance = 17 + (54 − 17) × 0.35 = 30  → L3 (Mild) ✓
```

#### Ex2: 3 HIGH sources (84), MED edges (54) → same LOW target (17)

| Source | source_rel | edge_imp | vouch |
|--------|-----------|---------|-------|
| A | 84 (L8) | 54 (L5) | 0.45 |
| B | 84 (L8) | 54 (L5) | 0.45 |
| C | 84 (L8) | 54 (L5) | 0.45 |

```
Σvouch = 1.36
pull_target = 84
pull_strength = min(1.0, 1.36 × 0.4) = 0.54
relevance = 17 + (84 − 17) × 0.54 = 53  → L5 (Notable) ✓
```

Same n_paths, same edges — only source relevance differs. HIGH sources pull
harder than MED sources.

#### Ex3: 3 HIGH sources (84), LOW edges (17) → same target (17)

| Source | source_rel | edge_imp | vouch |
|--------|-----------|---------|-------|
| A | 84 (L8) | 17 (L2) | 0.14 |
| B | 84 (L8) | 17 (L2) | 0.14 |
| C | 84 (L8) | 17 (L2) | 0.14 |

```
Σvouch = 0.42
pull_target = 84
pull_strength = min(1.0, 0.42 × 0.4) = 0.168
relevance = 17 + (84 − 17) × 0.168 = 28  → L3 (weaker than Ex2) ✓
```

Low edge importance dilutes vouch credibility even when sources are highly
relevant. Correct — if the sources barely care about the target, convergence
should be weak.

#### Ex4: 1 HIGH source (84), HIGH edge (84) → LOW target (17)

n_paths = 1 → no convergence (pull_strength = 0).

```
relevance = 17 + (anything) × 0 = 17  → L2 ✓
```

Single-path entities rely solely on path-min decay. Convergence only applies
to multi-path entities.

### Negative Faction Handling

Magnitude is magnitude. If 3 high-relevance characters all have high-importance
negative edges to "Lord Black," convergence applies identically:
- vouch values are computed from magnitude only
- pull_target is based on source relevances
- The resulting relevance is the same as for positive convergence
- Faction metadata is preserved for presentation/context formatting

### Edge Cases

- **Single-path entity**: n_paths=1 → Σvouch=0 → pull_strength=0 → relevance = base_relevance
- **Self-vouch / cycle**: Guarded by visited set; entity cannot vouch for itself
- **Source promotion loop**: Pass-2 promoted entities cannot re-vouch Pass-1 sources (see Multi-Pass Pipeline)
- **Divergent source relevances**: pull_target weighted toward high-relevance cluster (higher vouches)
- **Negative convergence**: Same math as positive, magnitude-only

---

## 5. Multi-Pass Pipeline

### Overview

```
┌──────────────────────────────────────────────────────────────────┐
│ Pass 1: Broad Discovery (low threshold ≈ 42 / L4)                │
│   ∀ scene character → traverse_graph(threshold=42, max_depth=N)  │
│   → For each discovered entity, compute base_relevance via P4    │
│   → Track per entity: all paths (path_min, depth, source)        │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Pass 2: Convergence Scoring                                      │
│   ∀ entity with n_paths > 1:                                     │
│     Compute vouches from Pass-1 source relevances                │
│     relevance = base_relevance + (pull_target − base) × strength  │
│   ∀ entity with n_paths == 1:                                    │
│     relevance = base_relevance (from P4 decay)                   │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Pass 3 (optional): Re-traversal with Promoted Relevances         │
│   Entities with promoted relevance ≥ L6 (65) become new sources  │
│   Re-traverse from these entities at depth+1 using P4            │
│   Newly discovered entities go through Pass 2 convergence        │
│   Cap: promoted entities cannot re-vouch their own discoverers   │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Pass 4: Pruning (at final threshold, default 75 / L7)            │
│   Filter: entity_relevance ≥ min_relevance_threshold             │
│   Return: dict[entity_type] → dict[entity_name → relevance]      │
└──────────────────────────────────────────────────────────────────┘
```

### Pass 1: Broad Discovery

```
threshold = decay_config.get("broad_threshold", 42)  # L4
result = entity_graph.traverse_graph(
    initial_entities,
    field_map=field_map,
    min_importance=threshold,
    max_depth=max_depth,
)
```

Returns the current set-of-names result. But we also need **path metadata**.
Modify `traverse_graph` (or add `traverse_graph_detailed`) to return:

```python
# Per discovered entity
path_records: dict[str, PathRecord]

class PathRecord:
    name: str
    entity_type: str
    # All independent paths
    paths: list[Path]
    # Highest effective importance across all paths (decayed)
    best_effective: float
    # Distinct source entities that connect to this entity
    source_ids: set[str]

class Path:
    edges: list[EdgeRef]       # each edge with its importance, relation, faction
    path_min: int              # min importance in this path
    depth: int
    source_entity_id: str      # which initial-character this path originates from
    effective_imp: float       # after P4 decay
```

Two paths are "independent" if they originate from different initial characters
or pass through different first-hop connectors. Simple heuristic: count
*distinct direct connectors* to the target. If Paul AND Amy AND Marcus all have
edges to Sarah, that's 3 independent paths regardless of deeper route details.

### Pass 2: Convergence Scoring

```python
for entity_id, record in path_records.items():
    n_paths = len(record.source_ids)
    if n_paths <= 1:
        # No convergence — use best single-path relevance
        entity_relevance[entity_id] = record.best_effective
        continue

    # Compute vouches
    total_vouch = 0.0
    weighted_sum = 0.0
    for source_id in record.source_ids:
        edge = get_edge(source_id, entity_id)  # or use stored path info
        if edge is None:
            continue
        vouch = (source_relevance[source_id] / 100.0) * (edge.importance / 100.0)
        total_vouch += vouch
        weighted_sum += source_relevance[source_id] * vouch

    if total_vouch == 0:
        entity_relevance[entity_id] = record.best_effective
        continue

    pull_target = weighted_sum / total_vouch
    pull_strength = min(1.0, total_vouch * config.pull_rate)
    base = record.best_effective

    relevance = base + (pull_target - base) * pull_strength
    entity_relevance[entity_id] = relevance
```

### Pass 3: Re-traversal (Optional)

```python
if config.re_traverse:
    new_seeds = {
        eid: rel
        for eid, rel in entity_relevance.items()
        if rel >= config.re_traverse_threshold  # default 65 (L6)
        and eid not in initial_scene_entities   # don't re-traverse from initial
    }
    for seed_id, seed_rel in new_seeds.items():
        # Get neighbors not yet discovered, using P4 with seed's relevance
        # as the "first hop" importance
        neighbors = entity_graph.get_neighbors_decayed(
            seed_id,
            source_relevance=seed_rel,
            min_importance=config.re_traverse_min_importance  # lower, e.g. 42
        )
        # Apply convergence on newly discovered entities
```

Pass 3 creates a "rich get richer" effect in a controlled way — only entities
promoted above L6 can amplify further. This mimics narrative gravity: a
moderately important character vouched for by multiple sources becomes a
credible connector themselves.

### Pass 4: Pruning

```python
final = {
    etype: {
        name: relevance
        for name, relevance in entities.items()
        if relevance >= config.final_threshold  # default 75 (L7)
    }
    for etype, entities in entity_relevance.items()
}
```

Return a dict of entity_type → {name: relevance} so downstream can use
relevance for levels-of-detail decisions.

### Output Change

Currently returns: `dict[str, set[str]]` — entity type → set of names.
New return: `dict[str, dict[str, float]]` — entity type → {name: relevance}.

The `RetrievalContext.relevant_entities` type changes from
`dict[str, set[str]]` to `dict[str, dict[str, float]]`.

---

## 6. Configuration

### DecayConfig

```python
@dataclass
class DecayConfig:
    # Decay method
    method: str = "path_min_gated"  # "path_min_gated", "threshold", "exponential"

    # Path-min decay parameters
    decay_exponent: float = 0.4    # α = (imp/100) ^ decay_exponent
    base_threshold: int = 75       # L7 — final filter threshold
    broad_threshold: int = 42      # L4 — Pass 1 discovery threshold

    # Convergence parameters
    convergence_enabled: bool = True
    pull_rate: float = 0.4         # How fast pull strength accumulates
    convergence_pass_enabled: bool = True

    # Re-traversal (Pass 3)
    re_traverse_enabled: bool = False   # Optional, off by default
    re_traverse_threshold: int = 65     # L6 — minimum relevance to become a seed
    re_traverse_min_importance: int = 42

    # Max depth
    max_depth: int = 10

    # Level of detail (deferred)
    # detail_field_map: dict[int, list[str]] = None
```

---

## 7. Entity Graph Changes

### New Method: `traverse_graph_detailed`

Extends `traverse_graph` to return path metadata needed for convergence.

```python
def traverse_graph_detailed(
    self,
    initial_entities: dict[str, set[str]],
    initial_relevance: dict[str, float] | None = None,
    field_map: dict | None = None,
    decay_config: DecayConfig | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, PathRecord]]:
```

### New Method: `get_edge`

```python
def get_edge(
    self,
    source_id: str,
    target_id: str,
    field_name: str | None = None,
) -> Relationship | None:
    """Get a single relationship between two entities."""
```

### Modified: `get_neighbors` → `get_neighbors_decayed`

```python
def get_neighbors_decayed(
    self,
    entity_name: str,
    entity_type: str,
    source_relevance: float = 100.0,
    target_type: str | None = None,
    field_name: str | None = None,
    decay_config: DecayConfig | None = None,
    direction: str = "outgoing",
) -> list[dict]:
    """Get neighbors with effective importance computed via decay.

    Returns list of dicts:
      {name, entity_type, importance, effective_importance,
       relation, faction, path_min, edges}
    """
```

### Static Centrality Metrics (Stored in Graph JSON)

Per node in `entity_graph.json`:

```python
{
  "id": "character:Sarah",
  "type": "character",
  "name": "Sarah",
  "data": {...},
  "centrality": {
    "total_incoming_connections": 5,
    "total_outgoing_connections": 3,
    "avg_incoming_importance": 68,
    "bidirectional_count": 2,
    "source_entity_count": 3  # distinct characters that have edges to her
  }
}
```

Populated during `_build_from_source_files()` or after adjacency build.
Used as a prior for convergence (optional — can weight pull_target by
static centrality × dynamic relevance).

---

## 8. Importance Scale Presentation

### Decision: Template-constant in `format_templates.json`

The 10-level importance scale is defined as a static template entry, **not** in
general_info (LLM-extracted, risks drift) and **not** hardcoded in Python.

- Added as `"importance_scale"` template in `format_templates.json`
- Listed first in `_context_order` with `"no_prompt": true`
- Rendered with empty data dict (template is entirely self-contained)
- Injected into `custom_state["context"]` once per generation
- LLM cannot modify it; user can customize the template text
- `no_prompt` flag skips the LLM history entry (no Q&A pair generated)

### Context order loop change

```python
for item in context_order:
    no_prompt = item.get("no_prompt", False)
    if no_prompt:
        formatted = FormattedData({}, data_type, parser=None, ...).st
        if to_context and formatted:
            custom_state["context"] += f"\n\n{formatted}"
        continue
    # ... normal flow
```

---

## 9. Levels of Detail (Deferred)

Not implementing yet, but design sketch:

### Concept

Entity relevance determines which fields are included in the formatted context.

| Detail level | Relevance range | Included fields |
|-------------|-----------------|-----------------|
| **Low** | 65–74 (L6) | name, description |
| **Medium** | 75–83 (L7) | name, description, traits, biography, voice, quirks |
| **High** | 84+ (L8+) | all fields (including relationships, group_status, milestones, fears, preferences, attributes) |
| **None** | < 65 | excluded entirely |

### Integration with Pruning (Pass 4)

The pruning threshold (default 75) is the **minimum relevance for inclusion**.
Below that, entities are excluded entirely. The L6 band (65–74) is for
convergence-promoted entities that don't quite reach the final threshold.

When `levels_of_detail` is enabled:
- Pass 4 uses `broad_threshold` (42) as the minimum for inclusion (instead of 75)
- Entities above `final_threshold` (75) get High detail
- Entities between 65–74 get Medium/Low detail
- Template selection: `format_retrieval_data` receives a relevance value and
  selects the appropriate template variant

### Template Implications

The character template in `format_templates.json` would need variants:

```json
{
  "characters_low": "Character -- {{name}}\nDescription -- {{description}}",
  "characters_medium": "Character -- {{name}}\nDescription -- {{description}}\nTraits -- {{traits}}\nBiography -- {{biography}}\nVoice -- {{voice}}",
  "characters_high": "...full template..."
}
```

Or a single template with conditional blocks using relevance:

```jinja2
{% if relevance >= 84 %}
  ...all fields...
{% elif relevance >= 75 %}
  ...medium fields...
{% else %}
  ...low fields...
{% endif %}
```

---

## 10. Summary of Changes

| Component | Change | Priority |
|-----------|--------|----------|
| `entity_graph.py` — `traverse_graph` | Add `traverse_graph_detailed` with path metadata | High |
| `entity_graph.py` — `get_neighbors` | Add `get_neighbors_decayed` with P4 decay | High |
| `entity_graph.py` — `Relationship` | No changes needed (faction, importance already there) | — |
| `entity_graph.py` — `EntityNode` | Add `centrality` dict, populated on build | Medium |
| `context_retriever.py` — `_unified_entity_aggregation` | Refactor to use `traverse_graph_detailed` + convergence passes | High |
| `context_retriever.py` — `RetrievalContext.relevant_entities` | Change type from `set[str]` to `dict[str, float]` | High |
| `context_retriever.py` — `DecayConfig` | New dataclass | Medium |
| `context_retriever.py` — `retrieve_context` | Wire up decay_config parameter | High |
| `summarizer.py` — context order loop | Handle `no_prompt` flag for template-only entries | Done |
| `format_templates.json` — `_context_order` | Added `importance_scale` entry (first, `no_prompt: true`) | Done |
| `format_templates.json` — templates | Added `importance_scale` template | Done |
| `summarizer.py` — `retrieve_and_format_context` | Pass decay_config through | Low |
| `format_templates.json` | Template variants for levels of detail | Deferred |
| All entity JSON data (characters, etc.) | Recalibrate importance scores to new 10-level system | Low (gradual) |

---

## 11. Open Questions (Answered)

The following questions were posed and resolved before implementation began.

1. **Pull rate default**: 0.4 global, uniform across all entity types.
2. **Re-traversal (Pass 3)**: Enabled by default. Performance impact accepted.
3. **Negative convergence aggregation**: Magnitude-only for now. Faction preserved
   as edge metadata but not aggregated in convergence relevance.
4. **Level of detail integration**: Template selection deferred to
   `format_retrieval_data` in `summarizer.py` when LOD is implemented.
5. **Static centrality as a prior**: Skipped in this round. Noted as a potential
   future user-configurable option (e.g. `centrality_prior_weight: 0.0` in
   `DecayConfig`).
