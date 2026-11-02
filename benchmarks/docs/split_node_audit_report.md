# Same-Referent Split Audit — dayna_ss subject store

Run: `cyberpunk_thriller__4dbf1435`  
Primary: final checkpoint (soak_100/d4d4bd6164b1f0a03c2b9d9e)  
Time-series: per-turn `state_snapshot/` over sampled turns.

> Read-only, idempotent, CPU + stdlib. Dedup primitives re-implemented
> inline. Full verbatim evidence below.

## 1. FINAL CHECKPOINT

### characters.json  (node count: 20)

#### Candidate pairs (split candidates)

##### `Juno`  <>  `Second Runner`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, unilateral-cross-ref, shared-partners
- shared sentences: **0**  (A has 9, B has 7 description/bio sentences)
- shared partners (raw): ['Juno', 'The Handler']
- shared partners (normalized): ['Juno', 'The Handler']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=85  ==  B rel `Juno` score=85
- A refs B key: False   |   B refs A key: True
- own importance: A=90  B=75
- aliases(A): ['Second Runner', 'Unseen Voice', 'Vell']
- aliases(B): ['Cleanup', 'The Backup', 'The Copy']

##### `Juno`  <>  `Vell`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, mutual-cross-ref, shared-partners
- shared sentences: **0**  (A has 9, B has 7 description/bio sentences)
- shared partners (raw): ['Juno', 'The Client']
- shared partners (normalized): ['Juno', 'The Client']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=85  ==  B rel `Juno` score=85
- A refs B key: True   |   B refs A key: True
- own importance: A=90  B=85
- aliases(A): ['Second Runner', 'Unseen Voice', 'Vell']
- aliases(B): ['Clean-Jawed Woman', 'The Buyer', 'The Duplicate', 'The Handler', 'The Operator', 'Vell']

##### `The Handler`  <>  `Vell`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): sentence-overlap, unilateral-cross-ref, shared-partners
- shared sentences: **3**  (A has 7, B has 7 description/bio sentences)
  - SHARED: `she prioritizes precise data over broad assumptions, refusing to give away her routes or codes until the final moment of delivery.`
  - SHARED: `unlike her remote counterparts, vell has physically walked the neon market district, verifying markers and tracking the second runner in real-time.`
  - SHARED: `she traced the compromised relay feeding juno the kiosk route to a dead node under graft that went dark two years ago, the same month pike stopped answering. she realized the flat cadence juno hummed belonged to pike, who had been wearing her frequency. when the band died again, she waited for juno to enter the dead air of the service throat, sending text with long, held lines to bait her into stopping. she instructed juno to count the ends of the lines—short meant her true voice, long meant the imposter—and to get to the needle.`
- shared partners (raw): ['Juno', 'The Client']
- shared partners (normalized): ['Juno', 'The Client']
- A refs B key: True   |   B refs A key: False
- own importance: A=85  B=85
- aliases(A): ['Runner']
- aliases(B): ['Clean-Jawed Woman', 'The Buyer', 'The Duplicate', 'The Handler', 'The Operator', 'Vell']

##### `The Handler`  <>  `Pike`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap
- shared sentences: **0**  (A has 7, B has 7 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- A refs B key: False   |   B refs A key: False
- own importance: A=85  B=75
- aliases(A): ['Runner']
- aliases(B): ['Frequency Source', 'Old Guard', 'Pike', 'Runner', 'Sparrow', 'The Ghost', 'The Old Voice']

##### `The Owner`  <>  `The Client`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap
- shared sentences: **0**  (A has 6, B has 6 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- A refs B key: False   |   B refs A key: False
- own importance: A=60  B=65
- aliases(A): ['Customers', 'Keeper of the Base', 'Neighbors', 'Stranger', 'The Noodle Lady', 'The Runner']
- aliases(B): ['The Client', 'The Runner']

##### `The Owner`  <>  `Laundry-Line Woman`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap
- shared sentences: **0**  (A has 6, B has 5 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- A refs B key: False   |   B refs A key: False
- own importance: A=60  B=60
- aliases(A): ['Customers', 'Keeper of the Base', 'Neighbors', 'Stranger', 'The Noodle Lady', 'The Runner']
- aliases(B): ['Laundry Lady', 'The Keeper', 'The Runner', 'The Tarp Dweller']

##### `The Client`  <>  `Laundry-Line Woman`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap
- shared sentences: **0**  (A has 6, B has 5 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- A refs B key: False   |   B refs A key: False
- own importance: A=65  B=60
- aliases(A): ['The Client', 'The Runner']
- aliases(B): ['Laundry Lady', 'The Keeper', 'The Runner', 'The Tarp Dweller']

##### `City Contract Checkpoint`  <>  `Male Voice`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap
- split signal(s): importance-aligned-on-shared-partner, shared-partners
- shared sentences: **0**  (A has 4, B has 7 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=80  ==  B rel `Juno` score=80
- A refs B key: False   |   B refs A key: False
- own importance: A=75  B=75
- aliases(A): ['The Arch', 'The Filter', 'The Gate', 'The Voice']
- aliases(B): ['Home', 'Sparrow', 'The Voice']

##### `Second Runner`  <>  `The_Second_Runner`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): containment
- split signal(s): importance-aligned-on-shared-partner, shared-partners
- shared sentences: **0**  (A has 7, B has 0 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno', 'The Handler']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=85  ==  B rel `Juno` score=85
- A refs B key: False   |   B refs A key: False
- own importance: A=75  B=0
- aliases(A): ['Cleanup', 'The Backup', 'The Copy']
- aliases(B): []

##### `Clean-Jawed Woman`  <>  `Vell`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap, alias==key
- shared sentences: **0**  (A has 8, B has 7 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- A refs B key: True   |   B refs A key: False
- own importance: A=75  B=85
- aliases(A): ['Clean-Jawed Woman']
- aliases(B): ['Clean-Jawed Woman', 'The Buyer', 'The Duplicate', 'The Handler', 'The Operator', 'Vell']

##### `Hex-Sigil Drone`  <>  `Bird-Traffic Unit`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, shared-partners
- shared sentences: **0**  (A has 8, B has 6 description/bio sentences)
- shared partners (raw): ['Juno', 'The Handler']
- shared partners (normalized): ['Juno', 'The Handler']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=75  ==  B rel `Juno` score=75
  - A rel `The Handler` score=60  ==  B rel `The Handler` score=60
- A refs B key: False   |   B refs A key: False
- own importance: A=75  B=75
- aliases(A): ['Primary Target']
- aliases(B): ['Grid Peer', 'Hex-Sigil Drone', 'Market Bird', 'Observer']

##### `Pike`  <>  `Sparrow`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, unilateral-cross-ref, shared-partners
- shared sentences: **0**  (A has 7, B has 3 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=80  ==  B rel `Juno` score=75
- A refs B key: False   |   B refs A key: True
- own importance: A=75  B=72
- aliases(A): ['Frequency Source', 'Old Guard', 'Pike', 'Runner', 'Sparrow', 'The Ghost', 'The Old Voice']
- aliases(B): ['Sparrow', "Sparrow's Charge", 'The Debt', 'The Other Voice']

##### `Pike`  <>  `Male Voice`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap
- split signal(s): importance-aligned-on-shared-partner, shared-partners
- shared sentences: **0**  (A has 7, B has 7 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=80  ==  B rel `Juno` score=80
- A refs B key: False   |   B refs A key: False
- own importance: A=75  B=75
- aliases(A): ['Frequency Source', 'Old Guard', 'Pike', 'Runner', 'Sparrow', 'The Ghost', 'The Old Voice']
- aliases(B): ['Home', 'Sparrow', 'The Voice']

##### `Sparrow`  <>  `Male Voice`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, shared-partners
- shared sentences: **0**  (A has 3, B has 7 description/bio sentences)
- shared partners (raw): ['Juno']
- shared partners (normalized): ['Juno']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=75  ==  B rel `Juno` score=80
- A refs B key: False   |   B refs A key: False
- own importance: A=72  B=75
- aliases(A): ['Sparrow', "Sparrow's Charge", 'The Debt', 'The Other Voice']
- aliases(B): ['Home', 'Sparrow', 'The Voice']

**SPLIT verdicts in characters.json:**
- `Juno` / `Second Runner` — signals: importance-aligned-on-shared-partner, unilateral-cross-ref, shared-partners
- `Juno` / `Vell` — signals: importance-aligned-on-shared-partner, mutual-cross-ref, shared-partners
- `The Handler` / `Vell` — signals: sentence-overlap, unilateral-cross-ref, shared-partners
- `City Contract Checkpoint` / `Male Voice` — signals: importance-aligned-on-shared-partner, shared-partners
- `Second Runner` / `The_Second_Runner` — signals: importance-aligned-on-shared-partner, shared-partners
- `Hex-Sigil Drone` / `Bird-Traffic Unit` — signals: importance-aligned-on-shared-partner, shared-partners
- `Pike` / `Sparrow` — signals: importance-aligned-on-shared-partner, unilateral-cross-ref, shared-partners
- `Pike` / `Male Voice` — signals: importance-aligned-on-shared-partner, shared-partners
- `Sparrow` / `Male Voice` — signals: importance-aligned-on-shared-partner, shared-partners

**Distinct/ambiguous candidates in characters.json (no strong split signal):**
- `The Handler` / `Pike` — triggers alias-overlap; shared sentences 0; shared partners ['Juno']; xref A→B=False B→A=False
- `The Owner` / `The Client` — triggers alias-overlap; shared sentences 0; shared partners ['Juno']; xref A→B=False B→A=False
- `The Owner` / `Laundry-Line Woman` — triggers alias-overlap; shared sentences 0; shared partners ['Juno']; xref A→B=False B→A=False
- `The Client` / `Laundry-Line Woman` — triggers alias-overlap; shared sentences 0; shared partners ['Juno']; xref A→B=False B→A=False
- `Clean-Jawed Woman` / `Vell` — triggers alias-overlap, alias==key; shared sentences 0; shared partners ['Juno']; xref A→B=True B→A=False

**Dangling relationship edges in characters.json (partner resolves to NO node):**
- node `Clean-Jawed Woman` → partner `0`
- node `Grease-Collar Man` → partner `0`
- node `Juno` → partner `Third Landing Tenant`
- node `Male Voice` → partner `0`
- node `Pike` → partner `The Handler (Vell)`
- node `Second Runner` → partner `0`
- node `The Owner` → partner `Local Crews`
- node `The Widow` → partner `Grey Coats`


### groups.json  (node count: 28)

#### Candidate pairs (split candidates)

##### `grey coats`  <>  `city contract`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap, alias==key
- shared sentences: **0**  (A has 8, B has 3 description/bio sentences)
- shared partners (raw): []
- shared partners (normalized): []
- A refs B key: False   |   B refs A key: False
- own importance: A=80  B=75
- aliases(A): []
- aliases(B): ['City Contract', 'Grey Coats', 'Picket Line']

##### `local crews`  <>  `market crews`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap, alias==key
- shared sentences: **0**  (A has 1, B has 3 description/bio sentences)
- shared partners (raw): []
- shared partners (normalized): []
- A refs B key: False   |   B refs A key: False
- own importance: A=45  B=65
- aliases(A): ['Graft locals', 'Stack residents']
- aliases(B): ['Local Crews', 'Market Crews']

##### `bird-traffic drone`  <>  `hex-sigil drone`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, unilateral-cross-ref, shared-partners
- shared sentences: **0**  (A has 3, B has 5 description/bio sentences)
- shared partners (raw): ['City Contract', 'Juno']
- shared partners (normalized): ['City Contract', 'Juno']
- importance-aligned on shared partner(s):
  - A rel `Juno` score=70  ==  B rel `Juno` score=65
- A refs B key: False   |   B refs A key: True
- own importance: A=75  B=75
- aliases(A): ['Hex-Sigil Drone', 'hex-bird']
- aliases(B): ['Face-loggers', 'Hex-bird', 'Municipal Eye']

##### `bird-traffic drone`  <>  `bird-traffic unit`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap
- shared sentences: **0**  (A has 3, B has 5 description/bio sentences)
- shared partners (raw): ['City Contract Patrol']
- shared partners (normalized): ['City Contract Patrol']
- A refs B key: False   |   B refs A key: False
- own importance: A=75  B=75
- aliases(A): ['Hex-Sigil Drone', 'hex-bird']
- aliases(B): ['Bird-traffic', 'Hex-sigil drone', 'Sky-eye']

##### `hex-sigil drone`  <>  `bird-traffic unit`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap, alias==key
- shared sentences: **0**  (A has 5, B has 5 description/bio sentences)
- shared partners (raw): []
- shared partners (normalized): []
- A refs B key: True   |   B refs A key: False
- own importance: A=75  B=75
- aliases(A): ['Face-loggers', 'Hex-bird', 'Municipal Eye']
- aliases(B): ['Bird-traffic', 'Hex-sigil drone', 'Sky-eye']

##### `evening commuters`  <>  `night-market stream`  → **distinct/ambiguous**
- candidate trigger(s): alias-overlap, alias==key
- shared sentences: **0**  (A has 1, B has 2 description/bio sentences)
- shared partners (raw): []
- shared partners (normalized): []
- A refs B key: False   |   B refs A key: False
- own importance: A=35  B=45
- aliases(A): ['Night-walkers', 'Pedestrians']
- aliases(B): ['Evening commuters', 'Night-market stream']

##### `shell wallet`  <>  `holding company`  → **SAME-REFERENT-SPLIT**
- candidate trigger(s): alias-overlap, alias==key
- split signal(s): importance-aligned-on-shared-partner, shared-partners
- shared sentences: **0**  (A has 2, B has 2 description/bio sentences)
- shared partners (raw): ['Second Runner']
- shared partners (normalized): ['Second Runner']
- importance-aligned on shared partner(s):
  - A rel `Second Runner` score=75  ==  B rel `Second Runner` score=75
- A refs B key: False   |   B refs A key: False
- own importance: A=85  B=85
- aliases(A): ['Shadow Node', 'Shell Entity']
- aliases(B): ['Holding Company', 'Shell Wallet']

**SPLIT verdicts in groups.json:**
- `bird-traffic drone` / `hex-sigil drone` — signals: importance-aligned-on-shared-partner, unilateral-cross-ref, shared-partners
- `shell wallet` / `holding company` — signals: importance-aligned-on-shared-partner, shared-partners

**Distinct/ambiguous candidates in groups.json (no strong split signal):**
- `grey coats` / `city contract` — triggers alias-overlap, alias==key; shared sentences 0; shared partners []; xref A→B=False B→A=False
- `local crews` / `market crews` — triggers alias-overlap, alias==key; shared sentences 0; shared partners []; xref A→B=False B→A=False
- `bird-traffic drone` / `bird-traffic unit` — triggers alias-overlap; shared sentences 0; shared partners ['City Contract Patrol']; xref A→B=False B→A=False
- `hex-sigil drone` / `bird-traffic unit` — triggers alias-overlap, alias==key; shared sentences 0; shared partners []; xref A→B=True B→A=False
- `evening commuters` / `night-market stream` — triggers alias-overlap, alias==key; shared sentences 0; shared partners []; xref A→B=False B→A=False

**Dangling relationship edges in groups.json (partner resolves to NO node):**
- node `Eel vendors` → partner `City Contract Checkpoint`
- node `Eel vendors` → partner `Green Market`
- node `Ground people` → partner `Juno`
- node `Laundry Alley` → partner `City Contract Checkpoint`
- node `Laundry Alley` → partner `Pawn Shop`
- node `Neon Market` → partner `City Contract Checkpoint`
- node `Neon Market` → partner `Graft Stacks`
- node `Sky's ledger` → partner `Ground Buyers`
- node `Teenagers` → partner `Juno`
- node `bird-traffic drone` → partner `City Contract Patrol`
- node `bird-traffic drone` → partner `Graff`
- node `bird-traffic drone` → partner `Juno`
- node `bird-traffic unit` → partner `City Contract Patrol`
- node `bird-traffic unit` → partner `Data Core`
- node `bridge sensors` → partner `Juno`
- node `bridge sensors` → partner `The Handler`
- node `city contract` → partner `City Contract Checkpoint`
- node `city contract` → partner `Kelso Bridge Checkpoint`
- node `clean-jawed woman` → partner `Juno`
- node `clean-jawed woman` → partner `The Handler`
- node `clean-jawed woman` → partner `Third Landing`
- node `collection unit` → partner `City Contract Patrons`
- node `evening commuters` → partner `Juno`
- node `hex-sigil drone` → partner `City Authority`
- node `hex-sigil drone` → partner `Juno`
- node `holding company` → partner `Juno`
- node `holding company` → partner `Tenant`
- node `imposter` → partner `Grease-Collar Man`
- node `imposter` → partner `Juno`
- node `imposter` → partner `Vell`
- node `market crews` → partner `Juno`
- node `market crews` → partner `The Handler`
- node `shell wallet` → partner `Data Core`
- node `shell wallet` → partner `Third Landing Tenant`
- node `vendor cam` → partner `Handler`
- node `vendor feeds` → partner `City Contract Checkpoint`
- node `vendor feeds` → partner `Juno`
- node `vendor feeds` → partner `The Handler`


### elements.json  (node count: 76)

_No candidate pairs generated._

## 2. DANGLING RELATIONSHIP EDGES (final checkpoint)

Total dangling edges: **323**  (top-10 partners by frequency)

| subject | partner | count | lookalike? |
|---|---|---|---|
| characters.json | `0` | 4 | abstract/foreign |
| characters.json | `Grey Coats` | 1 | abstract/foreign |
| characters.json | `Local Crews` | 1 | abstract/foreign |
| characters.json | `The Handler (Vell)` | 1 | LOOKALIKE |
| characters.json | `Third Landing Tenant` | 1 | abstract/foreign |
| groups.json | `City Authority` | 1 | abstract/foreign |
| groups.json | `City Contract Checkpoint` | 5 | LOOKALIKE |
| groups.json | `City Contract Patrol` | 2 | LOOKALIKE |
| groups.json | `City Contract Patrons` | 1 | LOOKALIKE |
| groups.json | `Data Core` | 2 | abstract/foreign |
| groups.json | `Graff` | 1 | abstract/foreign |
| groups.json | `Graft Stacks` | 1 | LOOKALIKE |
| groups.json | `Grease-Collar Man` | 1 | abstract/foreign |
| groups.json | `Green Market` | 1 | abstract/foreign |
| groups.json | `Ground Buyers` | 1 | abstract/foreign |
| groups.json | `Handler` | 1 | abstract/foreign |
| groups.json | `Juno` | 11 | abstract/foreign |
| groups.json | `Kelso Bridge Checkpoint` | 1 | abstract/foreign |
| groups.json | `Pawn Shop` | 1 | abstract/foreign |
| groups.json | `Tenant` | 1 | abstract/foreign |
| groups.json | `The Handler` | 4 | abstract/foreign |
| groups.json | `Third Landing` | 1 | abstract/foreign |
| groups.json | `Third Landing Tenant` | 1 | abstract/foreign |
| groups.json | `Vell` | 1 | abstract/foreign |
| elements.json | `Above` | 1 | abstract/foreign |
| elements.json | `Band_Traffic` | 1 | abstract/foreign |
| elements.json | `Below` | 1 | abstract/foreign |
| elements.json | `Cadence` | 1 | abstract/foreign |
| elements.json | `Canal Path` | 1 | LOOKALIKE |
| elements.json | `City Patrol` | 1 | abstract/foreign |
| elements.json | `Clean-Jawed Woman` | 1 | abstract/foreign |
| elements.json | `Client` | 4 | abstract/foreign |
| elements.json | `Copper Wire` | 1 | abstract/foreign |
| elements.json | `Count` | 1 | abstract/foreign |
| elements.json | `Drone` | 2 | LOOKALIKE |
| elements.json | `Eel Vendors` | 1 | LOOKALIKE |
| elements.json | `Evidence` | 1 | abstract/foreign |
| elements.json | `Exchange District` | 1 | abstract/foreign |
| elements.json | `Exhaust_Pipe` | 1 | abstract/foreign |
| elements.json | `File` | 1 | abstract/foreign |
| elements.json | `Final Clearance` | 1 | abstract/foreign |
| elements.json | `Final_Clearance` | 1 | abstract/foreign |
| elements.json | `First` | 1 | abstract/foreign |
| elements.json | `Follower` | 1 | abstract/foreign |
| elements.json | `Frequency` | 1 | abstract/foreign |
| elements.json | `Grease-Collared Man` | 1 | abstract/foreign |
| elements.json | `Grease-Stained Tail` | 3 | abstract/foreign |
| elements.json | `Grease-stained Scarf` | 1 | LOOKALIKE |
| elements.json | `Grease_Stained_Tail` | 1 | abstract/foreign |
| elements.json | `Grey Coats` | 3 | abstract/foreign |
| elements.json | `Grey_Coats` | 1 | abstract/foreign |
| elements.json | `Handler` | 19 | abstract/foreign |
| elements.json | `Handshake` | 1 | abstract/foreign |
| elements.json | `Hex Sigil Drone` | 1 | LOOKALIKE |
| elements.json | `Hex-Sigil Drone` | 14 | LOOKALIKE |
| elements.json | `Hex_Sigil_Drone` | 2 | LOOKALIKE |
| elements.json | `Juno` | 76 | abstract/foreign |
| elements.json | `Kelso Bridge Checkpoint` | 1 | LOOKALIKE |
| elements.json | `Linen-clad Follower` | 1 | abstract/foreign |
| elements.json | `Local Crews` | 2 | abstract/foreign |
| elements.json | `Local Kids` | 1 | abstract/foreign |
| elements.json | `Local Youth` | 1 | abstract/foreign |
| elements.json | `Local_Crews` | 1 | abstract/foreign |
| elements.json | `Male Speaker` | 1 | abstract/foreign |
| elements.json | `Man in Linen` | 1 | LOOKALIKE |
| elements.json | `Market` | 1 | LOOKALIKE |
| elements.json | `Movement` | 1 | abstract/foreign |
| elements.json | `Mrs. Patterson` | 1 | abstract/foreign |
| elements.json | `Night Market` | 1 | LOOKALIKE |
| elements.json | `Ninety_Seconds` | 1 | abstract/foreign |
| elements.json | `Order` | 1 | abstract/foreign |
| elements.json | `Path` | 1 | abstract/foreign |
| elements.json | `Pawn Shop Cut-Through` | 1 | LOOKALIKE |
| elements.json | `Price` | 1 | abstract/foreign |
| elements.json | `Pricing` | 1 | abstract/foreign |
| elements.json | `Record` | 1 | abstract/foreign |
| elements.json | `Rhythm` | 1 | abstract/foreign |
| elements.json | `Room` | 1 | abstract/foreign |
| elements.json | `Safehouse Destination` | 1 | abstract/foreign |
| elements.json | `Second Runner Identification` | 1 | LOOKALIKE |
| elements.json | `Sixth_Landing_Second_Runner` | 1 | LOOKALIKE |
| elements.json | `Spoofed Ledgers` | 1 | abstract/foreign |
| elements.json | `Stair-Cluster` | 1 | abstract/foreign |
| elements.json | `Stationary Threat` | 1 | abstract/foreign |
| elements.json | `Stationary Watcher` | 2 | abstract/foreign |
| elements.json | `Tenant` | 7 | LOOKALIKE |
| elements.json | `The Client` | 1 | abstract/foreign |
| elements.json | `The Eel Vendor` | 1 | LOOKALIKE |
| elements.json | `The Exchange Floor` | 1 | LOOKALIKE |
| elements.json | `The Handler` | 29 | abstract/foreign |
| elements.json | `The Owner` | 1 | LOOKALIKE |
| elements.json | `The Widow` | 2 | LOOKALIKE |
| elements.json | `The_Above` | 1 | abstract/foreign |
| elements.json | `The_Band` | 1 | abstract/foreign |
| elements.json | `The_Band_Traffic` | 1 | abstract/foreign |
| elements.json | `The_Below` | 1 | abstract/foreign |
| elements.json | `The_Cadence` | 1 | abstract/foreign |
| elements.json | `The_Canal` | 1 | LOOKALIKE |
| elements.json | `The_Canal_Cut` | 1 | LOOKALIKE |
| elements.json | `The_Corporate_Glyph` | 1 | LOOKALIKE |
| elements.json | `The_Count` | 1 | abstract/foreign |
| elements.json | `The_Data_Core` | 1 | LOOKALIKE |
| elements.json | `The_Death_Zone` | 1 | abstract/foreign |
| elements.json | `The_Drone` | 1 | abstract/foreign |
| elements.json | `The_Evidence` | 1 | abstract/foreign |
| elements.json | `The_Exhaust_Pipe` | 1 | abstract/foreign |
| elements.json | `The_File` | 1 | abstract/foreign |
| elements.json | `The_First` | 1 | abstract/foreign |
| elements.json | `The_Footbridges` | 1 | LOOKALIKE |
| elements.json | `The_Frequency` | 1 | abstract/foreign |
| elements.json | `The_Grease_Collar_Man` | 1 | LOOKALIKE |
| elements.json | `The_Handler` | 5 | abstract/foreign |
| elements.json | `The_Handshake` | 1 | abstract/foreign |
| elements.json | `The_Hex_Sigil_Drone` | 1 | LOOKALIKE |
| elements.json | `The_Laundry_Line_Unit` | 1 | LOOKALIKE |
| elements.json | `The_Movement` | 1 | abstract/foreign |
| elements.json | `The_Ninety_Seconds` | 1 | abstract/foreign |
| elements.json | `The_Order` | 1 | abstract/foreign |
| elements.json | `The_Path` | 1 | abstract/foreign |
| elements.json | `The_Price` | 1 | abstract/foreign |
| elements.json | `The_Pricing` | 1 | abstract/foreign |
| elements.json | `The_Record` | 1 | abstract/foreign |
| elements.json | `The_Rhythm` | 1 | abstract/foreign |
| elements.json | `The_Room` | 1 | abstract/foreign |
| elements.json | `The_Rust_Red_Grate` | 1 | LOOKALIKE |
| elements.json | `The_Second_Runner` | 2 | LOOKALIKE |
| elements.json | `The_Service_Tunnel` | 1 | LOOKALIKE |
| elements.json | `The_Sluice_Plates` | 1 | LOOKALIKE |
| elements.json | `The_Tenant` | 1 | abstract/foreign |
| elements.json | `The_Two_Things` | 1 | abstract/foreign |
| elements.json | `The_Under` | 1 | abstract/foreign |
| elements.json | `The_Vault` | 1 | abstract/foreign |
| elements.json | `The_Wet_Goods_Rows` | 1 | LOOKALIKE |
| elements.json | `The_Window` | 1 | abstract/foreign |
| elements.json | `The_Word` | 1 | abstract/foreign |
| elements.json | `Third Landing` | 2 | LOOKALIKE |
| elements.json | `Third Landing Tenants` | 1 | LOOKALIKE |
| elements.json | `Three suits` | 1 | abstract/foreign |
| elements.json | `Three_Suits` | 1 | abstract/foreign |
| elements.json | `Two_Things` | 1 | abstract/foreign |
| elements.json | `Under` | 1 | abstract/foreign |
| elements.json | `Window` | 1 | abstract/foreign |
| elements.json | `Word` | 1 | abstract/foreign |

## 3. ENTITY-GRAPH DISPLAY-NAME COLLISIONS

Total graph nodes: **173**  |  distinct display names in collision: **18**

### `Exchange Floor Surveillance`  — 2 graph nodes

| node id | type |
|---|---|
| `event:Exchange Floor Surveillance` | event |
| `scene:Exchange Floor Surveillance` | scene |

### `Handler's Route Correction`  — 2 graph nodes

| node id | type |
|---|---|
| `event:Handler's Route Correction` | event |
| `scene:Handler's Route Correction` | scene |

### `Second Runner Identification`  — 2 graph nodes

| node id | type |
|---|---|
| `event:Second Runner Identification` | event |
| `scene:Second Runner Identification` | scene |

### `The Abort Protocol Activation`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Abort Protocol Activation` | event |
| `scene:The Abort Protocol Activation` | scene |

### `The Band Returns Wrong`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Band Returns Wrong` | event |
| `scene:The Band Returns Wrong` | scene |

### `The Glyph Prints`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Glyph Prints` | event |
| `scene:The Glyph Prints` | scene |

### `The Grease-Collar Entrance`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Grease-Collar Entrance` | event |
| `scene:The Grease-Collar Entrance` | scene |

### `The Grease-Collar Test`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Grease-Collar Test` | event |
| `scene:The Grease-Collar Test` | scene |

### `The Handler's Ledger Analysis`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Handler's Ledger Analysis` | event |
| `scene:The Handler's Ledger Analysis` | scene |

### `The Handler's Signal Loss`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Handler's Signal Loss` | event |
| `scene:The Handler's Signal Loss` | scene |

### `The Hex-Sigil Drone's Tasking`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Hex-Sigil Drone's Tasking` | event |
| `scene:The Hex-Sigil Drone's Tasking` | scene |

### `The Identity Audit`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Identity Audit` | event |
| `scene:The Identity Audit` | scene |

### `The Maintenance Spiral Descent`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Maintenance Spiral Descent` | event |
| `scene:The Maintenance Spiral Descent` | scene |

### `The Post-Delivery Marker`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Post-Delivery Marker` | event |
| `scene:The Post-Delivery Marker` | scene |

### `The Triple Knock Head Count`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Triple Knock Head Count` | event |
| `scene:The Triple Knock Head Count` | scene |

### `The Upload of the Slate Photo`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Upload of the Slate Photo` | event |
| `scene:The Upload of the Slate Photo` | scene |

### `The Vault Declaration`  — 2 graph nodes

| node id | type |
|---|---|
| `event:The Vault Declaration` | event |
| `scene:The Vault Declaration` | scene |

### `events`  — 2 graph nodes

| node id | type |
|---|---|
| `event:events` | event |
| `scene:events` | scene |

## 4. TIME-SERIES GROWTH

Sampled turns: turn_000, turn_008, turn_016, turn_024, turn_032, turn_040, turn_048, turn_056, turn_064, turn_072, turn_080, turn_084

| turn | characters.json | groups.json | elements.json | split-pairs present | dangling edges |

|---|---|---|---|---|---|
| turn_000 | turn_000 | 2 | 2 | 4 | 1 | 10 |
| turn_008 | turn_008 | 5 | 4 | 27 | 1 | 60 |
| turn_016 | turn_016 | 6 | 8 | 38 | 1 | 102 |
| turn_024 | turn_024 | 8 | 10 | 41 | 1 | 118 |
| turn_032 | turn_032 | 9 | 13 | 44 | 0 | 130 |
| turn_040 | turn_040 | 10 | 16 | 44 | 2 | 152 |
| turn_048 | turn_048 | 13 | 19 | 58 | 3 | 193 |
| turn_056 | turn_056 | 13 | 21 | 66 | 4 | 217 |
| turn_064 | turn_064 | 16 | 21 | 68 | 5 | 225 |
| turn_072 | turn_072 | 19 | 26 | 74 | 7 | 291 |
| turn_080 | turn_080 | 20 | 28 | 76 | 11 | 323 |
| turn_084 | turn_084 | 20 | 28 | 80 | 11 | 332 |

### First-observation of final-checkpoint split pairs

- `Juno` / `Second Runner`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_032**
  - first solo appearance: `Juno` → **turn_000**,  `Second Runner` → **turn_032**
- `Juno` / `Vell`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_048**
  - first solo appearance: `Juno` → **turn_000**,  `Vell` → **turn_048**
- `The Handler` / `Vell`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_048**
  - first solo appearance: `The Handler` → **turn_000**,  `Vell` → **turn_048**
- `City Contract Checkpoint` / `Male Voice`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_080**
  - first solo appearance: `City Contract Checkpoint` → **turn_024**,  `Male Voice` → **turn_080**
- `Second Runner` / `The_Second_Runner`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_072**
  - first solo appearance: `Second Runner` → **turn_032**,  `The_Second_Runner` → **turn_072**
- `Hex-Sigil Drone` / `Bird-Traffic Unit`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_064**
  - first solo appearance: `Hex-Sigil Drone` → **turn_064**,  `Bird-Traffic Unit` → **turn_064**
- `Pike` / `Sparrow`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_072**
  - first solo appearance: `Pike` → **turn_072**,  `Sparrow` → **turn_072**
- `Pike` / `Male Voice`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_080**
  - first solo appearance: `Pike` → **turn_072**,  `Male Voice` → **turn_080**
- `Sparrow` / `Male Voice`  (characters.json):
  - earliest sampled turn with BOTH nodes: **turn_080**
  - first solo appearance: `Sparrow` → **turn_072**,  `Male Voice` → **turn_080**
- `bird-traffic drone` / `hex-sigil drone`  (groups.json):
  - earliest sampled turn with BOTH nodes: **turn_016**
  - first solo appearance: `bird-traffic drone` → **turn_016**,  `hex-sigil drone` → **turn_016**
- `shell wallet` / `holding company`  (groups.json):
  - earliest sampled turn with BOTH nodes: **turn_056**
  - first solo appearance: `shell wallet` → **turn_056**,  `holding company` → **turn_056**

