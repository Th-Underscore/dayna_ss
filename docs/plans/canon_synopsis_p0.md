# Canon Synopsis P0 — Design

*Status: IMPLEMENTED 2026-08-25 · roadmap item: comparison.html §05 P0 "Canon synopsis at chapter archival".*
Implementation: `_regenerate_canon_digest` (agents/data_summarizer/archives.py, fired from both
archive success paths), `compute_stale_entities` + `_stale_entities` injection
(agents/summarizer/context_engine.py), roster-line branches in characters/groups/elements templates.
Hermetic coverage: tests/canon_synopsis_test.py. Open questions below were settled with the
recommended defaults: N=6 via config `demote_after_scenes`; digest also regenerates at arc close;
canon_history.json kept; rolling-overview findings NOT fed to the digest in P0.*

## Problem

Rendered context grows linearly with subject data (elements hit 173KB by t40 on
cyberpunk_thriller__4e29a512). DSS renders everything it stores; nothing collapses
old material. The standing budget is **<40k rendered tokens** (user directive,
2026-08-24). The 100-turn Tier-2 run is gated on this valve.

## Shape (from the roadmap)

> Derived subject appended at chapter close: ~300-token narrative digest. Renderer
> swaps closed-arc full profiles for synopsis references. Caps the linear token
> growth of long runs — the one place DSS's render-everything philosophy needs a
> pressure valve.

Two halves that only work together:

1. **Digest** — a bounded story-so-far summary that carries *why* things mattered.
2. **Demotion** — stale entities collapse to roster lines because the digest now
   carries their narrative weight. Demotion without the digest loses information;
   the digest without demotion doesn't stop growth.

## Decision 1: home = `general_info.synopsis` (no new subject)

`GeneralInfo` already declares a `synopsis` string field (both schemas), and the
`general_info` format template already renders `Synopsis --- {{data.synopsis}}`
(N/A while empty). Writing the field makes it appear in every turn's context with
**zero schema surgery** (the schema files are not even git-tracked) and zero new
render plumbing.

Why general_info and not a derived map subject:

- general_info renders **unconditionally** in system context every turn. A new
  subject would be retrieval/query-driven — exactly the wrong guarantee for a
  pressure valve whose value is predictability.
- Fixed position in the render order → prefix-cache friendly (changes only at
  chapter boundaries; #131 ordering principle unaffected — it sits inside the
  existing static-ish general_info block, not before it).
- Rejected: dedicated `canon` subject (retrieval-gated, unpredictable presence);
  appending per-chapter digests (grows linearly — defeats the purpose); storing
  under arcs (arcs may not exist yet in short runs).

## Decision 2: regenerate-from-inputs, never append-to-digest

At each chapter archival (and each arc archival), ONE LLM call regenerates the
whole digest from compact inputs:

- every archived chapter: title + summary + key_changes (bounded, they're already
  summaries),
- every archived arc: title + summary,
- main_objective / premise line from general_info for anchoring.

Output: ≤300 tokens (~1200 chars) of third-person narrative prose — what has
happened, who drove it, what remains open. Not a list of events; the scenes/events
blocks already do that job.

Why regenerate rather than "previous digest + delta": drift accumulates across
rewrites; inputs are cheap (chapter summaries are small — 40 chapters ≈ 3-4k
tokens) and regeneration is deterministic-in, bounded-out. Cost: one extra call
per chapter boundary — negligible against the boundary's own gate/archive calls.

Failure handling: parse failure or empty response keeps the previous digest
(fail-open), logged like other archive-step failures. The digest is a cache of its
inputs — always reconstructible, never load-bearing state.

## Decision 3: demotion rule = render-time staleness, no storage change

An entity (character/group/element) renders as a **roster line** instead of its
full profile when ALL of:

- it does not appear in `current_scene.now.who` (or the map's scene-cast set), and
- it has no participant credit in any of the last N archived scenes (default
  N=6, config `demote_after_scenes`; scenes' `participants` maps are the signal),
- regardless of its importance score.

Roster line shape: name + one-clause gist (first line of description/summary,
truncated) + importance — the existing roster idiom from the events template.

Properties that make this the right rule:

- **Pure render-time**: computed fresh each turn from existing stored shapes; no
  `_active`/`_resolved` flags to maintain, nothing to forget to reset. An entity
  that re-enters a scene instantly re-renders full — self-healing.
- **Reversibility** is why this beats status-field authoring: the model never has
  to declare an entity "closed" (it reliably doesn't), and closed-ness isn't
  stored wrongly.
- Composes with the existing importance tiers: importance still chooses full vs
  dormant among *fresh* entities; staleness forces the floor tier for stale ones.
- Scale-free machinery (ledger decision #2): verifiable hermetically at any
  cadence.

Implementation locus: a small helper computes the stale set once per turn
(events scenes are already loaded for retrieval); FormattedData injects it as a
template context var (`_stale_entities`); the characters/groups/elements templates
gain one conditional around their full-profile branch. Templates stay dumb; the
decision lives in one place.

## Epistemics composition (memory #212)

The synopsis is **objective-layer, plot-level** prose: "the chip changed hands;
Kell suspects Juno kept it." Directive to the digest prompt: state stances as
facts-of-the-record, never private interiority beyond what scenes established.
When epistemics lands, POV-filtered injection groups secret rows by stance over
the scene cast; the synopsis does NOT carry per-character knowledge state (that's
the secrets rows' job) — but if a row is later marked `exposed`, the next
regeneration naturally reflects it since inputs come from live subjects. No
coupling beyond regeneration timing; nothing to redesign later.

Significance-threshold interplay: secrets discovery guards against stub-entity
sprawl; the synopsis inherits that hygiene because it only summarizes archived
chapters — trivial awareness never reaches chapter summaries in the first place.

## Budget math (why this holds <40k)

Per-turn fixed additions after N chapters: synopsis ≈300 tok (constant), roster
lines ≈15-25 tok each. Growth that used to be linear-in-full-profiles becomes:
recent-window full profiles (bounded by cast size per scene, small) + roster tail
(linear but ~25 tok/entity) + constant digest. At cyberpunk-scale casts
(~10 entities) that's hundreds — not tens of thousands — of tokens by t100.
Elements/events retain their own caps (P-P1) and templates already truncate.

## Validation plan

1. **Hermetic**: golden scenario — scripted digest response; assert
   `general_info.synopsis` written at forced archive, previous digest retained on
   garbage response, rendered Synopsis line appears in the built context.
   Unit-test the stale-set helper (staleness math: cast presence, N-boundary,
   re-entry reversal).
2. **Live short**: rerun the aggregation-fix smoke shape
   (`--force-chapter-turn`, Type 2 Mode B, ~16 turns); after the forced archive
   verify: non-empty synopsis in `turn_NNN/state_snapshot/general_info.json`,
   `Synopsis ---` present in subsequent `context.txt`, demoted roster lines for
   stale entities, measured context size trend.
3. **Tier-2 gate**: the 100-turn compressed run reports per-turn context sizes;
   acceptance = flat under 40k.

## Open questions (settle before code)

1. Demotion window default N=6 scenes — right ballpark for compressed cadence?
   (At campaign cadence the same N covers far more turns; config makes it
   tunable per profile — should profiles carry it?)
2. Also regenerate at arc close (proposed yes — arcs add long-range framing).
3. Keep `canon_history.json` (append-only per-archive digests) beside
   general_info.json for forensics? Cheap; recommended.
4. Should the digest prompt see rolling-overview findings? (No for P0 — those
   diagnose reply-quality, not plot canon; revisit with the feedback-loop item.)

## Live validation (canon smoke 2243dc84, 2026-08-25)

16 turns, Type 2 Mode B, forced chapter at t6; run also produced a NATURAL chapter 2 and an
arc 1 archive, so both digest triggers fired organically:

- canon_history.json: `[chapter_2 (1343 chars), arc_1 (1212 chars)]`; general_info.synopsis
  replaced in place and rendered under `Synopsis ---` every turn after.
- Chapters render with real titles ("Chapter [1] --- The Descent and the Drop") — the
  `Chapter [1] --- 0` mush from the prior run is gone via the alias-hint expansion path.
- Arc 'Ghost in the Steel' persisted with deterministic chapter span [1-3].
- Demotion did not trigger organically: the cast recurs every scene, so nobody left the
  6-scene window. Roster-line behavior is proven hermetically (canon_synopsis_test.py);
  organic demotion needs a wider-cast or longer run — verify again on the 100-turn run.
- REGRESSION FOUND + FIXED same-day: the first expander fix (untyped object lists stay
  lists) broke shape parity between core.generate's two FormattedData copies → TypeError
  storms during subject updates → guarded saves propagated stale current_scene/characters
  memory (judge: "run-fatal" stale chip-location memory). Replaced with the alias-hint
  path; suite green. Watch TypeError counts on the next live run vs the 7/run baseline.

Judge overall score 2.5 — driven by model continuity errors (chip destroyed but narrated as
carried) compounded by the parity-bug staleness above; not synopsis-related.
