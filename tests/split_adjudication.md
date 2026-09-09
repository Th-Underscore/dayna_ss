# Split Adjudication — cyberpunk_thriller__4dbf1435 (final checkpoint d4d4bd61)

Adjudicator: story-level LLM judgment over verbatim node prose. Read-only; no source data modified;
no dayna_ss imports. Evidence below is quoted verbatim from the final-checkpoint stores.

Verdict legend:
- **SPLIT** — one being, two store nodes. Canonical node named; stranded fields enumerated.
- **CONTAMINATION** — genuinely distinct beings, but one node carries out-of-scope writes about the
  other (or a third party). Polluted fields enumerated; the entities themselves are NOT merged.
- **DISTINCT** — no action.

Time-series anchors (from audit §4, used for canonical selection): The Handler first solo turn_000;
Vell turn_048; Second Runner turn_032; The_Second_Runner turn_072; Pike & Sparrow turn_072;
Male Voice turn_080; City Contract Checkpoint turn_024; Hex-Sigil Drone & Bird-Traffic Unit both
turn_064; groups bird-traffic drone & hex-sigil drone both turn_016.

---

## Pair 1 — `Juno` <> `Second Runner` (characters.json)

**Verdict: CONTAMINATION (distinct beings; Juno's alias array polluted).**

Evidence (both nodes' own prose):
- Juno description[0]: "…handing the **Second Runner** the physical corporate glyph casing" — a
  two-body interaction (Juno physically handing an object to a second body); the Second Runner is
  Juno's *object of action*, not a facet of her.
- Second Runner description[0]: "A female courier mirroring Juno's trajectory, **distinguished by a
  clean jawline devoid of the counting scar**. She **shares Juno's physical build and
  attire**—specifically the reinforced patch pocket on the left chest—**creating a visual
  duplicate**" — the nodes explicitly distinguish two bodies by scar status.
- Juno description[0]: "confirmed her **jaw scar count as four**" vs. Second Runner attributes:
  "Distinction: **No jaw scar**."
- Co-occurrence: Juno description[0] "She is currently exiting the service throat… having rejected
  the Male Voice's offer" and Second Runner description[0] "Currently positioned in the Service
  Tunnel Throat" — simultaneous, co-present, mutually aware ("shaking off the second runner's grip",
  Juno/milestones-adjacent scene 161). Two bodies, same space ⇒ distinct.
- Aliasing: Juno's alias array `['Second Runner', 'Unseen Voice', 'Vell']` is fully misfiled:
  'Second Runner' names the *other* node (wrong subject); 'Unseen Voice' and 'Vell' name the
  Handler/Vell (Pair 3). Genuine Juno aliases: none present (she is the primary/"slow" runner;
  every listed alias points at someone else).

## Pair 2 — `Juno` <> `Vell` (characters.json)

**Verdict: CONTAMINATION (distinct beings; alias arrays polluted in both directions).**

Evidence:
- Juno description[1]: "Cleared the Kelso bridge checkpoint and sought refuge in a laundromat
  **while the Handler audited her identity** and the client's relay scheme." — Juno is the *target*
  of the Handler/Vell's action; observer and subject are separate beings.
- Vell importance.reason: "As the **true Handler, Vell is Juno's primary lifeline** and strategic
  guide." — Vell is defined *in relation to* Juno, never as identical to her.
- Vell description[0]: "An **unseen voice in the runner's ear**… monitoring **Juno** via comms" —
  a voice-layer being distinct from the physical courier.
- Aliasing: Juno `['Second Runner','Unseen Voice','Vell']` — 'Vell' and 'Unseen Voice' are the
  *other node's own identity*, misfiled into Juno's array (writes addressed to the wrong subject).
  Vell `['Clean-Jawed Woman','The Buyer','The Duplicate','The Handler','The
  Operator','Vell']` — 'Clean-Jawed Woman' and 'The Duplicate' are the Second Runner's descriptors
  (third-party pollution); 'The Buyer' is The Client's descriptor (third-party pollution);
  genuine-for-Vell: 'Vell', 'The Handler', 'The Operator'.

## Pair 3 — `The Handler` <> `Vell` (characters.json)

**Verdict: SPLIT (one being, two nodes). Canonical: `The Handler` (first solo turn_000; Vell first
solo turn_048). `Vell` node is a near-total duplicate and should be merged into `The Handler` with
'Vell' retained as a genuine alias.**

Evidence (prose overlap is verbatim-identical, 3 shared sentences per audit):
- The Handler description[1]: "Recently **revealed her true name is Vell**, distinguishing herself
  from the relayed signal or imposter handler." — the same node itself states Vell is *its own
  revealed name*.
- The Handler description[1] / Vell description[1] (SHARED, verbatim identical): "Unlike her
  remote counterparts, **Vell has physically walked the Neon Market district**, verifying markers
  and tracking the second runner in real-time."
- SHARED (verbatim identical across both nodes): "She prioritizes precise data over broad
  assumptions, refusing to give away her routes or codes until the final moment of delivery."
- Vell importance.reason: "As the **true Handler**, Vell is Juno's primary lifeline…" — Vell's own
  node asserts it IS the Handler.
- Vell description[2]: "Vell distinguishes herself from the **generic 'Handler' alias** by
  asserting a unique, lifelong naming convention." — the distinction is alias-level (true name vs.
  role label / imposter persona), not a second being. The story's true/imposter distinction
  (Vell vs. Pike wearing her frequency) is internal to this one identity class.
- Stranded on `Vell` (to merge into canonical): entire description[0-2], biography[0-4]
  (verbatim duplicates of The Handler's), traits, voice, quirks, fears, preferences,
  group_status['Corporate Operations'], milestones (scene lists 18-22, 14-17, 10-13, 1-9),
  recent_state. Only increment worth preserving verbatim: description[2] "A corporate operative
  with a singular, authoritative voice… unique, lifelong naming convention" and description[3]
  "monitoring Juno's micro-environment through vendor relays and grid feeds."
- Aliasing post-merge: genuine = {Vell, The Handler (role), The Operator, Unseen Voice};
  to purge from the class = {'Clean-Jawed Woman' → Second Runner, 'The Duplicate' → Second
  Runner, 'The Buyer' → The Client}.

## Pair 4 — `City Contract Checkpoint` <> `Male Voice` (characters.json)

**Verdict: CONTAMINATION (distinct beings; Checkpoint alias entry polluted).**

Evidence:
- City Contract Checkpoint description[0]: "A **standardized municipal security post** situated
  beneath the Kelso bridge crossing… **crisp uniformed contractors, and a folding table**, it
  functions as a high-throughput, low-intimidation **barrier**" — a physical structure/infrastructure.
- City Contract Checkpoint voice: "defined by the **ambient hum of machinery** and the clipped
  commands of the contractors **rather than a single vocal source**" — the node itself disavows
  being a voice; its alias 'The Voice' contradicts its own description.
- Male Voice description[0]: "An **unseen corporate authority** operating from the deep service
  tunnels… Possesses a **warm, worn voice** and immense power derived from the facility's direct
  energy supply" — a person/voice-layer being in a different location (service tunnels, not the
  bridge checkpoint).
- Co-occurrence: Juno passed the checkpoint (turn_024 node) while the Male Voice spoke on an open
  band in the service throat (turn_080); Vell recent_state: "She interprets the **Male
  Voice's** public declaration as a claim of ownership rather than a private offer" — the two
  never share a body.
- Aliasing: Checkpoint `['The Arch','The Filter','The Gate','The Voice']` — 'The Arch', 'The
  Filter', 'The Gate' are genuine structural descriptors; **'The Voice' is a misfiled entry**: it
  names the voice-layer of the Male Voice being (and collides with Male Voice's own alias
  'The Voice'), and is self-contradictory given the node's "rather than a single vocal source".
  Genuine-for-Male-Voice: {Home, The Voice, Sparrow-as-claim-identifier (see Pair 9)}.

## Pair 5 — `Second Runner` <> `The_Second_Runner` (characters.json)

**Verdict: SPLIT (one being, two nodes). Canonical: `Second Runner` (first solo turn_032; the stub
is turn_072). `The_Second_Runner` is an empty relationship-stub node with zero prose — merge its
edge payload into the canonical, then delete the stub.**

Evidence:
- `The_Second_Runner`: `description: [], biography: [], traits: [], attributes: [], voice: ""`
  and `importance.score: 0` — **no prose exists** on this node; it carries only:
  `relationships.Juno = 85` "Mirrors Juno's importance; the double-sold contract now binds them
  mutually rather than competitively" and `relationships.The_Handler = 70` "Directly guided by
  Vell's instructions for the exit… new point-of-contact for the 'cadence' check."
- `Second Runner` (canonical) full prose: description[0] "A female courier **mirroring
  Juno's trajectory**, distinguished by a clean jawline… she **accepted the physical corporate
  glyph casing from Juno** as part of the asset split" — a single being with a complete,
  self-consistent record.
- The stub's stranded fields (to merge): `relationships.Juno` (score 85, reason text above),
  `relationships.The_Handler` (score 70, reason text above). Note the stub's partner key
  `The_Handler` is itself a dangling lookalike (resolves to no node; correct key: `The Handler`).

## Pair 6 — `Hex-Sigil Drone` <> `Bird-Traffic Unit` (characters.json)

**Verdict: SPLIT (one being, two nodes). Canonical: `Hex-Sigil Drone`.**

Evidence (descriptions interlock as facets of the same unit, same anchor, same scene):
- Hex-Sigil Drone description[0]: "A municipal-style aerial surveillance unit **distinguished by
  a hexagonal sigil painted on its belly**. It has **ceased its westward drift** to hold a
  **stationary tasking** over the market grid and service throat… It **tracks and logs human faces
  rather than hunting heat signatures**."
- Bird-Traffic Unit description[0]: "A municipal-style aerial surveillance unit **with a
  hexagonal sigil painted on its belly**, currently **circling the Graft stair-cluster in a tight,
  stationary orbit**."
- Bird-Traffic Unit description[2]: "**Tracks and logs human faces rather than hunting heat
  signatures**, maintaining a persistent **overwatch on the third landing**." — same capability,
  same target, stated as a second node's facts.
- Both nodes, attributes: "Orbit Anchor: **Bridge Stack**"; both importance.reason: "Structurally
  vital as the active tactical hazard… validates the Handler's theory of a **coordinated,
  multi-layered interception**."
- Bird-Traffic Unit description[1] (unique increment to preserve): "Distinct from standard city
  contract drones; **its paint and behavior indicate specialized tasking, likely funded by a
  higher authority or shadow node**."
- Both first appear at turn_064 (simultaneous creation — a classic double-key spawn for one
  object). Stranded on `Bird-Traffic Unit`: description[0-2], biography[0-2],
  group_status['Surveillance Grid'], milestones (Orbital Anchoring scene 127; Facial Lock
  Confirmation scene 127) — all facets of the same unit; merge into canonical.
- Aliasing: `['Grid Peer','Hex-Sigil Drone','Market Bird','Observer']` — 'Hex-Sigil Drone' and
  'Market Bird' are genuine (same unit); 'Grid Peer' and 'Observer' are role descriptors
  (acceptable). Canonical's 'Primary Target' is a rel descriptor misfiled into its own alias
  array — purge.

## Pair 7 — `Pike` <> `Sparrow` (characters.json)

**Verdict: SPLIT (one being, two nodes; `Sparrow` is a signal-layer node of the same being) — and
the largest cross-contamination in the run. Canonical: `Pike`.**

Evidence:
- Pike description[0]: "An **unseen corporate authority** operating from the deep service
  tunnels… **Known to Juno as 'Pike'**, he **offers 'home'**… He recently **spent his anonymity
  to publicly claim Juno** with a single sentence ('Come home')…"
- Pike quirks[2]: "**Refers to Juno as 'Sparrow' occasionally, a nickname from two
  years ago**." — 'Sparrow' is the *nickname Pike uses to address Juno*; the node was keyed on the
  addressee-of-the-claim instead of the claimer.
- Sparrow description[0] (CONTAMINATION, verbatim Juno prose): "Freelance courier **navigating the
  subterranean service tunnel beneath the Kelso Bridge**, wearing a **worn utility jacket with a
  reinforced patch pocket holding the lead-taped datachip**. She has plain **amber filter
  lenses**, a **silver locket** against her sternum, and **three scars on her jaw**… answered the
  Male Voice with an **arithmetic receipt ('Juno. Eight thousand.'**)" — this is Juno's body,
  location, kit, and final-scene action written under the Sparrow key (and internally
  inconsistent: "three scars" vs. Juno's confirmed four).
- Sparrow group_status: "Service Throat Authority: **Impostor Handler, Signal Dominator**";
  events: "Masqueraded as the Handler during Juno's ascent. Attempted to halt Juno using the
  'Sparrow' identifier. **Failed to deceive Juno due to cadence mismatch**." — same imposter
  behavior Pike's node records: Pike biography[3]: "He is currently **masquerading as the
  Handler**, using her voice and format **but distorting the rhythm**."
- Sparrow relationships['The Handler'].events: "**Shared the Handler's frequency. Linked to
  Pike's old cadence**." — the node's own edges link the signal to Pike.
- Stranded/contaminated on `Sparrow`: description[0] (→ Juno), biography[0] (→ Juno, verbatim
  "Freelance courier running a chip that cannot be allowed to reach the sky-city's data core…"),
  milestones: "The Glyph Split" (scenes 161/143 → Juno), "Handler's Mid-Word Silence" (160/142
  → The Handler), "The Handler's Relay Switch" (148 → The Handler); "The Male Voice's Offer of
  Home" (146) is same-class and merges into Pike. Sparrow's *genuine* facets (traits
  "Auditory Mimic"; voice "elongated, dragging lines that feel heavy and uncertain"; the
  cadence-mismatch events) are facets of Pike's compromised relay behavior — merge into Pike, do
  not delete.
- Aliasing: class {Pike, Sparrow, Male Voice} — genuine self-aliases: {Pike, The Ghost, The Old
  Voice, Frequency Source, Old Guard, The Voice, Home}; 'Sparrow' is a claim-identifier (a
  nickname aimed *at Juno*), not the being's self-name — reclassify, don't merge as identity;
  'Sparrow's Charge', 'The Debt', 'The Other Voice' are event/rel descriptors (purge from
  alias arrays).

## Pair 8 — `Pike` <> `Male Voice` (characters.json)

**Verdict: SPLIT (one being, two nodes). Canonical: `Pike` (first solo turn_072, same turn as
Sparrow but the revealed true name; Male Voice first solo turn_080). `Male Voice` is a
role-descriptor node (the voice as heard) of the same being; merge into `Pike`.**

Evidence (near-verbatim duplication across the two nodes):
- Pike description[0]: "An **unseen corporate authority operating from the deep service tunnels
  beneath Ashenfall, distinguished by a warm, worn voice and immense power derived from the
  facility's direct energy supply**. He **offers 'home' and financial clearance** in exchange for
  her compliance, **viewing her as a recoverable asset rather than cargo**."
- Male Voice description[0]: "An **unseen corporate authority operating from the deep service
  tunnels beneath Ashenfall**. **Possesses a warm, worn voice and immense power derived from the
  facility's direct energy supply**. **Offers Juno 'home' and financial clearance in exchange for
  her compliance, viewing her as a recoverable asset rather than cargo**." — same being, same
  final scene, re-described.
- Pike biography[0] / Male Voice description[1] (shared near-verbatim): "Maintains a
  **long-standing connection to Juno, dating back to a rooftop extraction two years prior**. Has
  been **monitoring Juno's progress through the service throat, utilizing superior signal
  quality compared to the Handler**."
- Voice-signature identity: Male Voice voice: "a **low, gravelly tone with a relaxed cadence.
  Sentences often drag slightly at the end**" = Pike description[1] "speaks with a **flat,
  drawn-out cadence**" = Sparrow voice "elongated, dragging lines" — one acoustic signature.
- Stranded on `Male Voice` (genuine increments to preserve into Pike): description[2] is a
  duplicate (drop); recent_state[0] ("[msg 161]… The Male Voice's public claim of 'Come home' has
  been answered with **arithmetic**…") — the final exchange; milestones "Service Throat Offer"
  (scene 149) and "Rooftop Extraction" (scenes 143-145); attributes (identical to Pike's —
  dedupe).
- Aliasing: see Pair 7 (single class).

## Pair 9 — `Sparrow` <> `Male Voice` (characters.json)

**Verdict: SPLIT (same being — both are voice/signal keys of the same final-scene utterance).
Consistent with Pairs 7-8: {Pike, Sparrow, Male Voice} is one identity class.**

Evidence:
- Sparrow milestones[2]: "**The Male Voice's Offer of Home**" (scene 146): "A **male
  voice**, broadcasting on a powerful, full-signal band within the service tunnel, **offers Juno
  'home'**—safe passage, wiped debts, and comfort—in exchange for her continued survival" — the
  Sparrow node *records the Male Voice's event as its own milestone*: one event, two keys.
- Sparrow description[0]: "answered the **Male Voice** with an arithmetic receipt ('**Juno.
  Eight thousand.**')" — within the same final scene; the two keys describe the same exchange
  from the speaker side and the addressee side.
- Sparrow group_status: "Service Throat Authority: Impostor Handler, **Signal
  Dominator**" vs. Male Voice group_status: "Service Throat Authority: **Patron, Asset
  Manager**" — same location, same authority, two role-labelings.
- Voice-signature match (see Pair 8): both nodes' voice fields describe the same
  drag/flat-cadence acoustic; Sparrow quirks[1] "Ends sentences with a **sustained pause,
  creating a 'dragging' effect**" ≈ Male Voice "Sentences often **drag slightly at
  the end**".
- Co-occurrence test: the two never appear as separate simultaneous bodies — the "Sparrow"
  signal and the "Male Voice" are the same band activity (a voice/signal/identity entity, which
  the co-occurrence exemption covers). No node ever observes both as separate actors.

## Pair 10 — `bird-traffic drone` <> `hex-sigil drone` (groups.json)

**Verdict: SPLIT (one being, two nodes). Canonical: `hex-sigil drone` (richer: 5 description
facets incl. the terminal lift-off; same first-observation turn_016, but the fuller
chronology makes it the merge target). Cross-store note: same physical unit as the characters
`Hex-Sigil Drone`/`Bird-Traffic Unit` class (Pair 6) — see partition table.**

Evidence:
- bird-traffic drone description[1]: "A municipal-style aerial surveillance unit that has
  **shifted from a drifting patrol to a stationary, high-cost tasking over the Graft
  stair-clusters**. It performs **tight orbits and face-logging**, indicating a specific
  interest in the **third landing and the re-keyed middle door**."
- hex-sigil drone description[2-4]: "Initially **drifted west over the market grid** before
  receiving a specific **tasking order to hold a stationary surveillance box**. Later **shifted
  its vector south to align with Juno's corridor**… Ultimately **lifted off its orbit and slid
  north across the rain**" — the same temporal sequence of the same unit (drift → tasking-hold →
  corridor alignment → lift), with hex-sigil drone the more complete record.
- Shared anchors: both "circling the **container bridge stack**"; both face-logging; both
  Juno-corridor/line-holding; bird-traffic drone description[3] "Paint on the belly does not
  immediately identify the paying wallet" ≈ hex-sigil drone's paid-tasking cost model — same
  unit, same funding-ambiguity fact.
- Stranded on `bird-traffic drone`: description[0-3] (merge as facets into canonical); its
  description[3] wallet-attribution ambiguity is the only non-redundant increment;
  relationships (City Contract Patrol / Juno / Graff / City Contract) re-parent to canonical.
- Aliasing: bird-traffic drone `['Hex-Sigil Drone','hex-bird']` — 'Hex-Sigil Drone' is a genuine
  same-being alias (the characters-store key); both retained in the merged class.

## Pair 11 — `shell wallet` <> `holding company` (groups.json)

**Verdict: CONTAMINATION (distinct entities with an explicit proxy→owner hierarchy; one alias
entry polluted).**

Evidence:
- shell wallet description[0]: "A **financial instrument** acting as the intermediary for the
  datachip's purchase, **linked to a larger corporate entity** via a shadow node." — the node
  itself distinguishes the instrument *from* the larger entity.
- holding company description[0]: "A **corporate entity registered under a shell wallet** that
  controls the tenancy of the middle door" — the hierarchy is explicit and directional
  (entity-registered-under-instrument), i.e., two distinct referents in one relationship.
- shell wallet description[1]: "The shell wallet **masks the true payer**…" vs. holding company
  description[1]: "…acting as the **true payer** in the double-sold contract." — the prose
  explicitly distinguishes mask from true payer: a being cannot be both the mask and the thing
  masked.
- Shared partner 'Second Runner' with aligned importance (75/75) is explained by the hierarchy
  (both relate to the same contract), not identity.
- Aliasing: holding company `['Holding Company','Shell Wallet']` — **'Shell Wallet' is the
  other entity's own key, misfiled** into this alias array (a write addressed to the wrong
  subject); 'Holding Company' is genuine. shell wallet `['Shadow Node','Shell Entity']` —
  genuine (self-descriptors).

---

# MERGED IDENTITY-CLASS PARTITION

All 15 implicated node keys (11 character keys + 4 group keys from the 11 adjudicated pairs),
partitioned into 8 identity classes:

| Class | True identity | Keys merged (store: key) | Verdict path |
|---|---|---|---|
| C1 | Juno (protagonist, primary/"slow" runner, 4 jaw scars) | characters: `Juno` | Pairs 1, 2 — no merge; alias purges only |
| C2 | The Second Runner (clean-jawed duplicate, 0 scars) | characters: `Second Runner` ← `The_Second_Runner` (empty stub) | Pair 5 SPLIT |
| C3 | Vell (true Handler; role alias "The Handler") | characters: `The Handler` ← `Vell` | Pair 3 SPLIT |
| C4 | Pike (person absorbed into Sky's pricing machine; compromised relay / imposter Handler / final "home" offeror) | characters: `Pike` ← `Sparrow` ← `Male Voice` | Pairs 7, 8, 9 SPLIT |
| C5 | The hex-sigil paid-overwatch unit (one physical drone) | characters: `Hex-Sigil Drone` ← `Bird-Traffic Unit`; groups: `hex-sigil drone` ← `bird-traffic drone` | Pairs 6, 10 SPLIT (cross-store: 5 mirror keys, 1 physical unit) |
| C6 | The Kelso-bridge municipal checkpoint (infrastructure) | characters: `City Contract Checkpoint` | Pair 4 — no merge; alias purge only |
| C7 | The shell wallet (financial instrument / payment mask) | groups: `shell wallet` | Pair 11 — no merge; alias purge only |
| C8 | The holding company (true payer / tenancy controller) | groups: `holding company` | Pair 11 — no merge; alias purge only |

Consistency checks (step d):
- Juno cluster: C1 ∉ C2, C1 ∉ C3, C2 ∥ C3 (distinct, both relate to C1) — the "Duplicate"
  relationship is a story-level mirror, correctly not an identity merge. ✓
- Pike triangle: C4 internal (Pike = Sparrow = Male Voice, voice-layer same-referent); C4 ∉ C3
  (Pike *wore* Vell's frequency — imposter ≠ true voice; Vell's node: "the same month **Pike**
  stopped answering… **Pike**, who had been **wearing her frequency**" — Pike is named as the
  imposter *of* Vell, never as Vell). C4 ∉ C6 (Pair 4 distinct). ✓
- Drones: C5 disjoint from all person classes; C5's "distinct from standard city contract
  drones" (chars Bird-Traffic Unit) and "distinct from standard municipal drones" (groups
  bird-traffic unit) both describe the *same* unit's non-standard tasking, not a second drone.
  Note: groups `bird-traffic unit` (outside the 11 pairs) exhibits the identical signature
  (hex-sigil belly, face-logging, bridge-stack anchor) and is presumed a 6th mirror key of C5 —
  recommend follow-up adjudication, not resolved here.

---

# CONTAMINATION FIELD INVENTORY

node → field → offending text (verbatim) → correct owner

| # | Node | Field | Offending entry | Correct owner |
|---|---|---|---|---|
| 1 | `Juno` | aliases | `"Second Runner"` | `Second Runner` (its own key) |
| 2 | `Juno` | aliases | `"Unseen Voice"` | `The Handler` / `Vell` |
| 3 | `Juno` | aliases | `"Vell"` | `The Handler` / `Vell` |
| 4 | `Vell` | aliases | `"Clean-Jawed Woman"` | `Second Runner` / `Clean-Jawed Woman` |
| 5 | `Vell` | aliases | `"The Duplicate"` | `Second Runner` |
| 6 | `Vell` | aliases | `"The Buyer"` | `The Client` |
| 7 | `City Contract Checkpoint` | aliases | `"The Voice"` | purge (self-contradictory: node's own voice field says "rather than a single vocal source"; collides with `Male Voice`'s genuine alias) |
| 8 | `holding company` | aliases | `"Shell Wallet"` | `shell wallet` (the other entity's key) |
| 9 | `Pike` | aliases | `"Runner"` | `Juno` (role descriptor of Juno) |
| 10 | `Sparrow` | description[0] | "Freelance courier navigating the subterranean service tunnel beneath the Kelso Bridge… reinforced patch pocket holding the lead-taped datachip… three scars on her jaw… arithmetic receipt ('Juno. Eight thousand.')" | `Juno` (verbatim Juno body; also carries a 3-vs-4 scar error) |
| 11 | `Sparrow` | biography[0] | "Freelance courier running a chip that cannot be allowed to reach the sky-city's data core…" | `Juno` (verbatim Juno biography) |
| 12 | `Sparrow` | milestones[0] | "The Glyph Split" (scenes 161, 143) | `Juno` |
| 13 | `Sparrow` | milestones[1] | "Handler's Mid-Word Silence" (scenes 160, 142) | `The Handler` / `Vell` |
| 14 | `Sparrow` | milestones[3] | "The Handler's Relay Switch" (scene 148) | `The Handler` / `Vell` |

Non-alias stranded (SPLIT residue, listed for completeness, not counted above):
- `The_Second_Runner` → entire node (relationships.Juno=85, relationships.The_Handler=70) → merge into `Second Runner`, then delete stub.
- `Vell` → full prose/traits/voice/milestones payload → merge into `The Handler` (see Pair 3).
- `Sparrow` → genuine signal facets (traits "Auditory Mimic"/"Deceptive", dragging-cadence voice, cadence-mismatch events) → merge into `Pike` (see Pair 7).
- `Male Voice` → final-exchange recent_state + milestones → merge into `Pike` (see Pair 8).
- `Bird-Traffic Unit` (characters) → full payload → merge into `Hex-Sigil Drone` (see Pair 6).
- `bird-traffic drone` (groups) → full payload → merge into `hex-sigil drone` (see Pair 10).

---

# TOTALS

- 11 pairs adjudicated: **SPLIT = 7** (Pairs 3, 5, 6, 7, 8, 9, 10) · **CONTAMINATION = 4**
  (Pairs 1, 2, 4, 11) · **DISTINCT (no action) = 0**.
  (Pairs 1, 2, 4, 11 are genuinely-distinct entities whose *nodes* carry misfiled fields —
  the contamination verdict supersedes, and strictly requires, no merge.)
- 15 implicated keys → **8 merged identity classes** (C1-C8).
- **14 contamination field entries** (8 alias misfilings + 6 verbatim prose/milestone
  misdirects, of which all 6 prose-class entries sit on the `Sparrow` node).
- Dangling-edge note: of the audit's 323 dangling edges, 3 are directly resolved by these
  verdicts (e.g. `Pike` → partner `The Handler (Vell)` resolves to C3 once lookalike-normalized;
  `Second Runner`/`Grease-Collar Man` → partner `0` entries are stub artifacts, not entities).