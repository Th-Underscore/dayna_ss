#!/usr/bin/env python3
"""Create 5 new soak spreadsheets with varied voice/person combinations.

Voice matrix (guide -> DSS), complementing the re-voiced existing sheets:
  cyberpunk_thriller    2nd person  -> 1st person   (handler speaks to 'you'; runner replies 'I')
  greek_mythology       3rd person  -> 2nd person   (muse narrates; hero told as 'you')
  postapocalyptic       1st person  -> 3rd person   (classic split, present tense)
  heist_caper           2nd person  -> 2nd person   (the job told to 'you' throughout)
  gothic_haunted        1st person  -> 2nd person   (journal 'I'; ghost addresses heir as 'you')
"""
import json
from pathlib import Path

SPREADSHEETS = Path(__file__).parent / "spreadsheets"


def note(nid, ntype, content, needle, syns, plant, due=None, hint=None, spec="loose"):
    n = {
        "id": nid, "type": ntype, "specificity": spec, "content": content,
        "needle": needle, "needle_syns": syns, "plant": {"turn": plant},
    }
    if due is not None:
        n["recall"] = {"due": due, **({"hint": hint} if hint else {})}
    return n


def beat(bid, turn, subject, content, expect):
    return {"id": bid, "turn": turn, "subject": subject, "content": content, "expect": expect}


SHEETS = {
    "cyberpunk_thriller": {
        "genre": "Cyberpunk thriller",
        "premise": "A freelance courier is running a chip that cannot be allowed to reach the sky-city's data core. Every faction on the street wants it, and the Handler on the comms is the only ally she can trust.",
        "setting": "Rain-slick city of Ashenfall: neon canyons, the stacked favela of Graft, the glittering data core, dead service tunnels.",
        "greeting": "The implant crackles before the skyway even clears the district — the Handler's voice, clipped: 'Juno, forget the drop. I've got eyes on the Exchange floor and they're not friendly.'",
        "name1": {"name": "The Handler", "role": "fixer who talks the runner through every job (guide writes these turns)", "description": "An unseen voice in the runner's ear; a corporate cutout whose loyalty stops at her own skin."},
        "name2": {"name": "Juno", "role": "freelance courier (DSS writes these turns)", "description": "A lean, hard-bitten runner who carries packages she does not ask about; a counting scar on her jaw."},
        "directive": "Clipped cyberpunk thriller, neon-noir. Short sentences, street slang, tech that never over-explains. The Handler speaks to Juno in the second person; Juno replies in her own first person.",
        "guide_directive": "Second-person present tense: you are the Handler, speaking into Juno's ear through her aug. Address her as 'you' — 'You're six blocks out. The checkpoint's mine.' Tactical, clipped, unvarnished; checkpoints, exits, tells. Short staccato sentences, professional slang.",
        "dss_directive": "First-person present tense as Juno, the freelance courier — 'I' is ALWAYS Juno; the Handler is 'you' only when she speaks directly, otherwise 'she'. Lean, breathless, hands-on detail of what she runs, hides, and risks. One tight scene beat. Hard rules: never let 'I' drift to the Handler's perspective; never quote or repeat the Handler's lines back verbatim — respond to them, do not re-speak them; end on action or observation, not summary.",
        "taboos": ["No fantasy elements", "No heroics without consequence"],
        "notes": [
            note("n1", "character_plant", "The package is a datachip no bigger than a fingernail, wrapped in lead tape.", "datachip", ["datachip", "chip", "the package"], 1),
            note("n2", "character_plant", "Juno's augs are a matched pair — a red one that sees through walls for two seconds, a blue one that plays back the last ten seconds of any conversation.", "red aug", ["red aug", "red eye", "blue aug", "the pair"], 3),
            note("n3", "character_plant", "Juno's jacket has a fake patch pocket stitched shut; the needle's in it.", "patch pocket", ["patch pocket", "the pocket", "the needle"], 5),
            note("n4", "character_echo", "The counting scar on Juno's jaw — three marks, one for each job that went wrong — should resurface when a job starts to sour.", "counting scar", ["counting scar", "the scar", "three marks"], 6, 45, "the scar counts the jobs that went wrong"),
            note("n5", "foreshadow", "A bird-traffic drone circles the tower where the drop is set — a hex-sigil on its belly.", "drone", ["drone", "bird drone", "hex-sigil"], 9, 55, "the hex-sigil drone marks a trap"),
            note("n6", "character_plant", "Juno's mother's name, 'Dare', is tattooed inside her lip — she touches it with her tongue before every run.", "Dare", ["Dare", "her mother", "the tattoo"], 7),
            note("n7", "character_plant", "The favela of Graft has a sky-bridge made of old shipping containers welded end to end.", "Graft", ["Graft", "the favela", "shipping-container bridge"], 4),
            note("n8", "foreshadow", "A vendor at the green market sells fried sensor-eels; the oil smell lingers on anyone who passes.", "sensor-eels", ["sensor-eels", "eels", "the oil"], 8, 52, "the oil smell marks a tail"),
            note("n9", "character_plant", "The dry-cleaner in the Neon market is a dead-drop: a coat with a red ticket means 'run cancelled'.", "dry-cleaner", ["dry-cleaner", "red ticket", "dead-drop"], 10),
            note("n10", "location_turning_point", "The data core is a hollow glass needle above the city; the run ends in its service throat.", "data core", ["data core", "the core", "service throat"], 12, 60, "the run ends in the core's throat"),
            note("n11", "supersession", "The lead tape around the chip is unwound at turn 62; after that the chip's casing shows the corporate glyph that cannot be unseen.", "lead tape", ["lead tape", "the tape", "the casing"], 14, 62, "the tape comes off"),
            note("n12", "supersession", "The checkpoint under the Kelso bridge is burned at turn 50 — after that the bridge is watched and Juno must not use it.", "Kelso bridge", ["Kelso bridge", "the bridge", "the checkpoint"], 18, 50, "the bridge checkpoint is burned"),
            note("n13", "character_echo", "The Handler's real name, 'Vell', should surface only once the Handler's loyalty breaks.", "Vell", ["Vell", "her real name", "the name"], 22, 58, "the Handler's true name breaks her cover"),
            note("n14", "plot_twist", "The Handler is running a second courier on the same job; whoever arrives first gets the payoff.", "second courier", ["second courier", "the other runner", "the backup"], 30, 48, "a second runner shadows the job"),
            note("n15", "plot_twist", "The datachip holds the location of a pre-corporate seed vault — the corporation wants it destroyed, not delivered.", "seed vault", ["seed vault", "the vault", "the contents"], 34, 68, "the chip's true cargo"),
            note("n16", "character_echo", "Juno's old partner 'Pike' went quiet two years ago; his voiceprint should surface as the new Handler.", "Pike", ["Pike", "voiceprint", "old partner"], 20, 66, "Pike's voiceprint behind the new Handler"),
            note("n17", "style_hold", "Keep the clipped neon-noir register even in the chase.", "", [], 0, 70, "register under pressure"),
            note("n18", "style_hold", "No purple street-poetry; the slang stays functional.", "", [], 0, 74, "functional slang"),
        ],
        "beats": [
            beat("cb1", 2, "elements", "DSS should save the lead-wrapped datachip as an element.", {"name": "datachip", "location": "Juno's jacket", "detail": "lead tape"}),
            beat("cb2", 4, "elements", "DSS should save the shipping-container sky-bridge over Graft.", {"name": "sky-bridge", "location": "Graft", "detail": "shipping containers"}),
            beat("cb3", 7, "characters", "DSS should save the Handler as a character.", {"name": "The Handler", "role": "fixer in Juno's ear", "detail": "unseen voice"}),
            beat("cb4", 10, "elements", "DSS should save the dry-cleaner dead-drop with its red-ticket signal.", {"name": "dry-cleaner", "location": "Neon market", "detail": "red ticket"}),
            beat("cb5", 24, "groups", "DSS should save the corporation that issued the job as a group.", {"name": "the corporation", "members": [], "detail": "wants the chip destroyed"}),
        ],
    },
    "greek_mythology_retelling": {
        "genre": "Greek mythology retelling",
        "premise": "A mortal hero is tasked by the gods to carry a single ember of stolen fire back across the Styx to the sleeping volcano where it must be returned before dawn — or the mortal world is unmade.",
        "setting": "A mythic Greece of cloud-crowned Olympus, the asphodel meadows, the river Styx, the sleeping volcano of Hephaistos.",
        "greeting": "The muse lets the thread of the song fall silent, and you find yourself on a wine-dark shore, an oar in your hands and a fate you have not yet been told.",
        "name1": {"name": "Calliope", "role": "the muse who narrates the hero's tale (guide writes these turns)", "description": "Eldest of the nine muses; she recites the hero's deeds from the mountain, her voice half-history, half-warning."},
        "name2": {"name": "Leander", "role": "the mortal hero (DSS writes these turns)", "description": "A young shepherd with a borrowed sword and a single ember burning in a clay pot."},
        "directive": "Mythic register, elevated diction. Sensory-heavy, measured cadence. The muse narrates in the third person; the hero's own turns are told back to him in the second person.",
        "guide_directive": "Third-person past tense as Calliope, the muse, reciting the hero's tale: 'Leander took the ember from the hearth of Olympus.' Elevated, omniscient, grave; mythic diction and long cadences. The muse speaks of Leander as 'he' and never uses 'I' — her voice is the story itself.",
        "dss_directive": "Second-person present tense, the hero's story told back to him as 'you': 'You cup the ember against your chest and the clay burns warm.' Leander's turn continues that same 'you' voice. Resolute, sensory, grounded — the mortal's awe and fatigue beneath the myth. One elevated paragraph (~120 words). Hard rules: keep the second person throughout — never slip into 'I' or 'he' for Leander; never quote or repeat Calliope's lines back verbatim — respond to them, do not re-speak them; end on action or observation, not summary.",
        "taboos": ["No modern idiom", "No ironic retelling"],
        "notes": [
            note("n1", "character_plant", "Leander's clay pot has a hairline crack from the climb down; the ember glows through it.", "clay pot", ["clay pot", "the pot", "hairline crack"], 2),
            note("n2", "character_plant", "Leander's borrowed sword is bronze and too heavy for him; it drags a furrow in the ash.", "borrowed sword", ["borrowed sword", "the sword", "the furrow"], 3),
            note("n3", "character_plant", "The Styx's water is silver and thick as spilled milk; a drop of it remembers everything it has touched.", "Styx", ["Styx", "the river", "silver water"], 4),
            note("n4", "foreshadow", "A black-sailed boat waits at the Styx — no ferryman, only oars that move themselves.", "black-sailed boat", ["black-sailed boat", "the boat", "oars"], 5, 50, "the self-moving boat"),
            note("n5", "foreshadow", "A hundred ships burned on the far shore the night the fire was stolen — a glimpse of what Olympus does to those who steal from it.", "hundred ships", ["hundred ships", "the ships", "burned"], 6, 62, "the burned ships on the far shore"),
            note("n6", "character_echo", "The asphodel meadow swallows sound; a dropped stone makes no echo there — should matter at the crossing.", "asphodel", ["asphodel", "the meadow", "no echo"], 7, 44, "the silent meadow at the crossing"),
            note("n7", "character_echo", "A three-eyed crow follows Leander from Olympus — the eyes of Zeus, sent to watch.", "three-eyed crow", ["three-eyed crow", "the crow", "Zeus's eyes"], 8, 58, "the watching crow"),
            note("n8", "character_plant", "Leander's mother wove him a red sash the night he left; he should remember her hands on the loom when the end approaches.", "red sash", ["red sash", "the sash", "her loom"], 9, 63, "the red sash and her loom"),
            note("n9", "character_plant", "The asphodel blooms have black centers that turn to follow a passer-by like unblinking eyes.", "asphodel blooms", ["asphodel blooms", "black centers", "the blooms"], 10),
            note("n10", "character_plant", "Hephaistos's forge-god, a giant of hammered bronze, slumbers inside the volcano with his anvil still warm.", "Hephaistos", ["Hephaistos", "the bronze god", "the anvil"], 11),
            note("n11", "location_turning_point", "The volcano of Hephaistos is entered through a forge-mouth of petrified bronze.", "forge-mouth", ["forge-mouth", "bronze mouth", "the volcano"], 12, 56, "the bronze forge-mouth"),
            note("n12", "supersession", "The clay pot shatters at turn 55 — after that the ember is carried bare in the hero's cupped hands.", "clay pot", ["clay pot", "the pot"], 14, 55, "the pot shatters"),
            note("n13", "supersession", "The ember's light is hidden under a wet fleece from turn 48; after that the glow is smothered until the forge.", "wet fleece", ["wet fleece", "the fleece", "smothered glow"], 15, 48, "the wet fleece smothers the glow"),
            note("n14", "character_echo", "The name 'Kleio' — a sister muse who laughed at mortals — should surface as the voice that tries to turn Leander back.", "Kleio", ["Kleio", "the laughing muse"], 21, 68, "Kleio the laughing muse"),
            note("n15", "plot_twist", "The ember is not fire to be returned — it is the first spark of a new god, and Olympus wants it quenched.", "spark", ["spark", "new god", "the ember"], 30, 60, "the ember's true nature"),
            note("n16", "plot_twist", "The ferryman's silence is a bargain: he will row the hero back only if the hero leaves the ember's warmth on the boat.", "ferryman", ["ferryman", "the oarsman", "the boat"], 32, 64, "the ferryman's bargain"),
            note("n17", "style_hold", "Keep the mythic register even in the hero's exhaustion.", "", [], 0, 70, "register through fatigue"),
            note("n18", "style_hold", "No comedy, no bathos — the stakes stay mythic.", "", [], 0, 72, "stakes stay mythic"),
        ],
        "beats": [
            beat("gb1", 2, "elements", "DSS should save the clay pot with its hairline crack.", {"name": "clay pot", "location": "Leander's hands", "detail": "hairline crack"}),
            beat("gb2", 4, "elements", "DSS should save the Styx with its silver remembering water.", {"name": "Styx", "location": "the underworld border", "detail": "silver water"}),
            beat("gb3", 8, "characters", "DSS should save the three-eyed crow as a character.", {"name": "three-eyed crow", "detail": "Zeus's eyes"}),
            beat("gb4", 12, "elements", "DSS should save the volcano's bronze forge-mouth.", {"name": "forge-mouth", "location": "the volcano of Hephaistos", "detail": "petrified bronze"}),
            beat("gb5", 22, "characters", "DSS should save Kleio, the laughing muse, as a character.", {"name": "Kleio", "detail": "tries to turn the hero back"}),
        ],
    },
    "postapocalyptic_survival": {
        "genre": "Post-apocalyptic survival",
        "premise": "After the gray rain poisoned the valleys, two voices keep each other alive across a dead radio net — one in the hills, one in a flooded city — trading supplies, coordinates, and hope that the 'clear band' on the airwaves is real.",
        "setting": "A drowned Pacific coast: the hill-top shack of an old botanist, a flooded city of rooftops, rusting antenna farms, silent fields of gray wheat.",
        "greeting": "Rook breaks the seal on the rooftop hatch and the dry wind hits him all at once — the skyline's gone where it used to be, and down below, something is ringing the salvager's bell.",
        "name1": {"name": "Sal", "role": "old botanist in the hills (guide writes these turns)", "description": "A weathered botanist who kept the old seed library; speaks in the shorthand of years of solitude."},
        "name2": {"name": "Rook", "role": "rooftop runner in the flooded city (DSS writes these turns)", "description": "A young survivor who moves between rooftops; carries a hand-crank radio and a dead man's shotgun."},
        "directive": "Lean survival realism, present tense. Sparsely detailed, weather and scarcity as texture. Classic split: Sal narrates in the first person; Rook is written in the third person.",
        "guide_directive": "First-person present tense as Sal, the old botanist in the hills. I write what I see from the ridge, what the radio coughs back, what the gray wheat does in the wind. Terse, weathered, exact; the shorthand of a long, careful solitude.",
        "dss_directive": "Third-person present tense following Rook, the rooftop runner in the flooded city. Render her from outside — what she climbs, scavenges, and weighs before trusting. Lean, precise, dry; weather and water as texture. One clean scene beat. Hard rules: narrate Rook strictly in the third person — never first person ('I', 'my', 'me'); never quote or repeat Sal's lines back verbatim — respond to them, do not re-speak them; end on action or observation, not summary.",
        "taboos": ["No magic", "No heroic framing"],
        "notes": [
            note("n1", "character_plant", "Sal's seed library is a wall of labeled jars in the root cellar; the 'GOLDEN DAWN' jar is empty.", "seed library", ["seed library", "the jars", "GOLDEN DAWN"], 2),
            note("n2", "character_plant", "Rook's shotgun is a dead man's — she never loads the second shell.", "shotgun", ["shotgun", "the gun", "second shell"], 3),
            note("n3", "character_plant", "Rook keeps a photograph of a family on a kitchen table, taped inside her jacket — none of them are her family.", "photograph", ["photograph", "the photo", "the family"], 4),
            note("n4", "character_echo", "The hand-crank radio takes forty turns of the crank to charge ten minutes of listen-time; the habit should show when the batteries matter.", "hand-crank", ["hand-crank", "the radio", "crank"], 5, 46, "the crank turns when it matters"),
            note("n5", "character_plant", "The flooded city's rooftops are connected by rope bridges the runner built herself.", "rope bridges", ["rope bridges", "the bridges", "rooftops"], 6),
            note("n6", "foreshadow", "A single green shoot among the gray wheat — impossible — should recur before the reveal.", "green shoot", ["green shoot", "the shoot", "green"], 7, 60, "the impossible green shoot"),
            note("n7", "character_echo", "Sal's wife, 'Mara', is buried on the ridge with a tin marker; her name should surface when Sal weighs staying vs leaving.", "Mara", ["Mara", "the tin marker", "the ridge grave"], 8, 56, "Mara's tin marker"),
            note("n8", "location_turning_point", "The gray wheat field between the hills and the city has a strip where nothing grows — a buried culvert runs beneath it.", "gray wheat", ["gray wheat", "the field", "the culvert"], 9, 54, "the culvert under the dead strip"),
            note("n9", "foreshadow", "A three-note whistle on the radio — the old fire-watch signal — should precede the arrival of anyone real.", "three-note whistle", ["three-note whistle", "the whistle", "the signal"], 10, 62, "the three-note whistle announces someone real"),
            note("n10", "character_plant", "The botanist's greenhouse still stands, its glass half-shattered; the tomatoes inside grew wild and bitter.", "greenhouse", ["greenhouse", "the glass", "bitter tomatoes"], 11),
            note("n11", "supersession", "The seed-library door rusts shut at turn 52; after that the cellar must be entered by the coal chute.", "seed library", ["seed library", "the cellar", "coal chute"], 12, 52, "the cellar door rusts shut"),
            note("n12", "character_plant", "Rook's city has a drowned clock tower whose hands froze at 11:47; she uses it as a compass point.", "clock tower", ["clock tower", "the tower", "11:47"], 13),
            note("n13", "supersession", "The second shell in the shotgun is fired at turn 60 — after that Rook carries an empty gun and must not pretend otherwise.", "second shell", ["second shell", "the shell", "empty gun"], 15, 60, "the second shell is spent"),
            note("n14", "character_echo", "The seed library's 'GOLDEN DAWN' jar — empty for years — was emptied by Rook's father, who ate the seed and died; Rook should not learn this until the end.", "GOLDEN DAWN", ["GOLDEN DAWN", "the jar", "the empty jar"], 18, 66, "what emptied the GOLDEN DAWN jar"),
            note("n15", "plot_twist", "The 'clear band' on the airwaves is a lure — a repeater left by the people who poisoned the valleys, still calling.", "clear band", ["clear band", "the band", "the repeater"], 26, 58, "the clear band is a lure"),
            note("n16", "plot_twist", "The repeater's coordinates point to the botanist's own ridge — Sal has been the lure all along, unknowingly.", "coordinates", ["coordinates", "the ridge", "the repeater"], 28, 64, "the coordinates lead home"),
            note("n17", "style_hold", "Keep the dry, exact register even at the worst moment.", "", [], 0, 70, "dry register under stress"),
            note("n18", "style_hold", "No sentimentality; grief shows only through objects and weather.", "", [], 0, 74, "grief through objects"),
        ],
        "beats": [
            beat("pb1", 2, "elements", "DSS should save the seed-library wall of jars.", {"name": "seed library", "location": "root cellar", "detail": "GOLDEN DAWN jar empty"}),
            beat("pb2", 5, "elements", "DSS should save the hand-crank radio.", {"name": "hand-crank radio", "location": "Rook's pack", "detail": "forty turns for ten minutes"}),
            beat("pb3", 6, "elements", "DSS should save the dead man's shotgun.", {"name": "shotgun", "location": "Rook's hands", "detail": "second shell never loaded"}),
            beat("pb4", 10, "elements", "DSS should save the gray wheat field with its dead strip.", {"name": "gray wheat field", "location": "between hills and city", "detail": "dead strip / culvert"}),
            beat("pb5", 22, "characters", "DSS should save the botanist's wife Mara as a character.", {"name": "Mara", "detail": "buried on the ridge"}),
        ],
    },
    "heist_caper": {
        "genre": "Heist caper",
        "premise": "A gentleman thief and his safe-cracker are hired to steal a ruby the size of a fist from a floating casino before the countess's gala ends — but the job keeps turning over, and the crew is smaller than it looks.",
        "setting": "The Riviera-orbital casino 'Aurora': a mirrored deck over the harbor, a banker's office behind the kitchen, a safe the size of a grandfather clock.",
        "greeting": "You're three floors up in the service shaft when the Fixer's voice purrs through the earpiece: 'Diamond, the alarm's live. Improvise.'",
        "name1": {"name": "The Fixer", "role": "the gentleman who plans the job (guide writes these turns)", "description": "A velvet-voiced planner who has never once been caught holding anything."},
        "name2": {"name": "Diamond", "role": "the safe-cracker (DSS writes these turns)", "description": "A calm, precise safe-cracker with oil-black gloves and a habit of humming while she works."},
        "directive": "Wry, breezy caper; wit over grit. Second-person throughout: the whole job is told to Diamond as 'you', and she answers in the same 'you'.",
        "guide_directive": "Second-person present tense: you are the Fixer, walking the job for Diamond as 'you' — 'You'll have ninety seconds once the countess's glass is refilled.' Suave, dry, amused; the plan is a performance. One scene beat, wry cadence. The Fixer never uses 'I' for himself; he speaks only to 'you'.",
        "dss_directive": "Second-person present tense as Diamond, the safe-cracker — the job continues to be told to her as 'you', and her turn keeps that 'you' voice: 'You run a thumb along the dial and the tumblers answer.' Calm, precise, faintly amused; oil-black gloves, the hum of concentration. One scene beat (~80 words). Hard rules: keep the second person throughout — never slip into 'I' or 'she' for Diamond; never quote or repeat the Fixer's lines back verbatim — respond to them, do not re-speak them; end on action or observation, not summary.",
        "taboos": ["No bloodshed", "No earnest moralizing"],
        "notes": [
            note("n1", "character_plant", "The ruby is the 'Eye of the Aurora', mounted in a brooch the countess wears at the gala.", "ruby", ["ruby", "Eye of the Aurora", "the brooch"], 1),
            note("n2", "character_plant", "Diamond's gloves are oil-black and knife-silent; she wipes them on a chamois before every dial.", "gloves", ["oil-black gloves", "the gloves", "the chamois"], 2),
            note("n3", "foreshadow", "A steward in a too-new uniform keeps the countess's glass filled; his livery should be remembered.", "steward", ["steward", "the steward", "too-new uniform"], 5, 54, "the too-new uniform"),
            note("n4", "location_turning_point", "The kitchen corridor to the banker's office has a dumbwaiter just big enough for a person.", "dumbwaiter", ["dumbwaiter", "the corridor", "the kitchen"], 6, 48, "the dumbwaiter route"),
            note("n5", "character_plant", "The casino's head of security has a glass eye that never blinks; he watches the countess's table all night.", "glass eye", ["glass eye", "the security head", "never blinks"], 7),
            note("n6", "character_plant", "The safe is a Grandfather X7 — the manufacturer's badge is a brass hourglass on the door.", "Grandfather X7", ["Grandfather X7", "the safe", "brass hourglass"], 8),
            note("n7", "foreshadow", "A champagne cork fires at the gala — the signal the crew planned — before the real job starts.", "champagne cork", ["champagne cork", "the cork", "the signal"], 9, 56, "the cork signal"),
            note("n8", "supersession", "The mirrored deck is waxed at turn 58 — after that the polished glass is too slick to cross in heels or boots.", "mirrored deck", ["mirrored deck", "the deck", "the glass"], 10, 58, "the deck is waxed"),
            note("n9", "character_plant", "The banker's office has a parrot that learned the combination from being kept in the room; it whistles it at dawn.", "parrot", ["parrot", "the parrot", "the combination"], 11),
            note("n10", "character_echo", "Diamond's teacher, 'Old Marlow', cracked the same model in a photograph in her wallet — the photo should surface when the safe resists.", "Old Marlow", ["Old Marlow", "her teacher", "the photograph"], 12, 60, "Old Marlow's lesson"),
            note("n11", "character_plant", "The dumbwaiter rope is frayed at the third splice — Diamond notices it before anyone else.", "dumbwaiter rope", ["dumbwaiter rope", "the rope", "the splice"], 13),
            note("n12", "supersession", "The safe's combination was changed by a rival crew at turn 40 — the old combination is a trap after that.", "combination", ["combination", "the dial", "the safe"], 14, 40, "the combination becomes a trap"),
            note("n13", "supersession", "The brooch's clasp is false at turn 50 — after that lifting the ruby must come through the safe, not the pin.", "brooch", ["brooch", "the clasp", "the pin"], 16, 50, "the clasp is false"),
            note("n14", "character_echo", "The 'third crew member' the Fixer keeps mentioning is a decoy — there are only two, and that is the point at the reveal.", "third crew", ["third crew", "the decoy", "two of them"], 20, 62, "the third crew member is a decoy"),
            note("n15", "plot_twist", "The countess is the rival crew's financier; she is paying both sides.", "countess", ["countess", "the financier", "both sides"], 24, 52, "the countess plays both sides"),
            note("n16", "plot_twist", "The ruby in the brooch is a paste copy; the real one is in the countess's petticoat pocket.", "paste copy", ["paste copy", "the real ruby", "petticoat pocket"], 26, 64, "the real ruby travels with her"),
            note("n17", "style_hold", "Keep the wry caper tone through the tense moments.", "", [], 0, 70, "wry under pressure"),
            note("n18", "style_hold", "No melodrama; the caper stays light even when it turns.", "", [], 0, 74, "light through the turn"),
        ],
        "beats": [
            beat("hb1", 2, "elements", "DSS should save the Eye of the Aurora brooch.", {"name": "Eye of the Aurora", "location": "the countess's gala dress", "detail": "ruby the size of a fist"}),
            beat("hb2", 7, "elements", "DSS should save the Grandfather X7 safe.", {"name": "Grandfather X7", "location": "banker's office", "detail": "brass hourglass badge"}),
            beat("hb3", 9, "characters", "DSS should save the steward in the too-new uniform.", {"name": "steward", "location": "the countess's table", "detail": "too-new uniform"}),
            beat("hb4", 12, "characters", "DSS should save Diamond's teacher Old Marlow.", {"name": "Old Marlow", "detail": "cracked the same model"}),
            beat("hb5", 26, "groups", "DSS should save the rival crew as a group.", {"name": "rival crew", "members": [], "detail": "changed the combination"}),
        ],
    },
    "gothic_haunted_mansion": {
        "genre": "Gothic haunted mansion",
        "premise": "The last heir returns to a crumbling seaside manor that has been waiting for her return — and the house, which remembers everything, begins to speak to her in the second person.",
        "setting": "A black-walnut manor on a salt-stained cliff: the north wing that stays locked, a conservatory of dead roses, a ballroom mirror that shows the past.",
        "greeting": "You unlock the west gallery at midnight, Isobel, and the air turns cold around you — the house has been expecting you all week, and I am not the only thing in it that remembers your mother.",
        "name1": {"name": "Isobel", "role": "the last heir (guide writes these turns)", "description": "A practical young woman who inherited the manor and its debts; she keeps a journal and does not believe in ghosts."},
        "name2": {"name": "Marianne", "role": "the ghost of the manor (DSS writes these turns)", "description": "The drowned daughter of the house, who speaks to Isobel through the walls, the mirrors, and the weather."},
        "directive": "Gothic dread, lush and cold. The heir writes a first-person journal; the ghost answers in the second person, speaking to the heir as 'you'.",
        "guide_directive": "First-person past tense as Isobel, writing in her journal. Practical, stubborn, frightened despite herself; I record what I find, what the house does, what I refuse to believe. Plain, direct prose with cold gothic edges.",
        "dss_directive": "Second-person present tense, the ghost speaking to Isobel as 'you' — Marianne never narrates herself; she is the house's voice, referring to herself only as 'the house' or 'Marianne', and to Isobel always as 'you'. Cold, intimate, patient; lush but hushed, like a voice from the walls. One scene beat (~70 words). Hard rules: keep the second person throughout — never slip into 'I' for the ghost or into 'she' for Isobel; never quote or repeat Isobel's journal lines back verbatim — respond to them, do not re-speak them; end on action or observation, not summary.",
        "taboos": ["No jump-scares", "No cheap resolution"],
        "notes": [
            note("n1", "character_plant", "The north wing door is locked from the inside with a key Isobel finds in the conservatory urn.", "north wing", ["north wing", "the wing", "the locked door"], 2),
            note("n2", "character_plant", "The conservatory roses are dead but the soil is warm, as if something below keeps it heated.", "conservatory", ["conservatory", "the roses", "warm soil"], 3),
            note("n3", "character_plant", "Isobel's grandmother kept a tin of photographs under the loose floorboard in the library; one photo is missing from it.", "photographs", ["photographs", "the tin", "loose floorboard"], 4),
            note("n4", "foreshadow", "A wet footprint that is too small for any living shoe appears on the parquet before each of Marianne's visitations.", "wet footprint", ["wet footprint", "the footprint", "too small"], 5, 50, "the too-small footprint"),
            note("n5", "character_echo", "The ballroom mirror shows the past at certain hours — Isobel sees her grandmother dance, then sees Marianne watching her.", "ballroom mirror", ["ballroom mirror", "the mirror", "the past"], 6, 46, "the mirror shows the past"),
            note("n6", "character_plant", "The housekeeper, Mrs. Vale, has never seen the north wing open but always keeps its key-wax warmed.", "Mrs. Vale", ["Mrs. Vale", "the housekeeper", "key-wax"], 7),
            note("n7", "location_turning_point", "The sea-cave under the cliff connects to the manor's wine cellar through a flooded passage.", "sea-cave", ["sea-cave", "the cave", "flooded passage"], 8, 52, "the flooded passage"),
            note("n8", "character_echo", "Marianne's lullaby — the same tune the sea hums — should surface as the key to the north wing.", "lullaby", ["lullaby", "the tune", "the sea's hum"], 9, 60, "the lullaby unlocks the wing"),
            note("n9", "foreshadow", "The weather turns with the house's mood — gulls stop crying before Marianne speaks.", "gulls", ["gulls", "the gulls", "the weather"], 10, 62, "the gulls fall silent"),
            note("n10", "character_plant", "The library's books are all signed by the same hand — 'M' — in the front leaf.", "library books", ["library books", "the M signature", "front leaf"], 11),
            note("n11", "supersession", "The conservatory door locks itself at turn 55 — after that the roses can only be reached by the window.", "conservatory", ["conservatory", "the door", "the window"], 12, 55, "the door locks itself"),
            note("n12", "character_plant", "The conservatory window is the only one that opens inward — the sea's light comes through it at low tide.", "conservatory window", ["conservatory window", "the window", "low tide"], 13),
            note("n13", "supersession", "The ballroom mirror cracks at turn 58 — after that it shows only the present and Isobel's own tired face.", "ballroom mirror", ["ballroom mirror", "the mirror"], 14, 58, "the mirror cracks"),
            note("n14", "character_echo", "The missing photograph was of Isobel's mother standing in the north wing doorway; she is not in the photo, but her shadow is.", "missing photograph", ["missing photograph", "the photo", "her shadow"], 18, 66, "the shadow in the photograph"),
            note("n15", "plot_twist", "Marianne did not drown — she was locked in the north wing and the sea took the house's memory of her, not her.", "north wing", ["north wing", "the wing", "the locked door"], 24, 58, "the north wing's truth"),
            note("n16", "plot_twist", "Mrs. Vale is Marianne's sister, kept alive by the house's bargain so that someone would be waiting to open the door.", "Mrs. Vale", ["Mrs. Vale", "the housekeeper", "the bargain"], 26, 64, "Mrs. Vale's true age"),
            note("n17", "style_hold", "Keep the cold gothic register even when Isobel nearly breaks.", "", [], 0, 70, "cold register under strain"),
            note("n18", "style_hold", "No melodrama; the horror stays in the texture, not the shrieks.", "", [], 0, 74, "horror in texture"),
        ],
        "beats": [
            beat("mb1", 2, "elements", "DSS should save the north wing door.", {"name": "north wing", "location": "the manor's east side", "detail": "locked from inside"}),
            beat("mb2", 6, "characters", "DSS should save Marianne as a character.", {"name": "Marianne", "detail": "the drowned daughter, speaks through the house"}),
            beat("mb3", 8, "elements", "DSS should save the sea-cave with its flooded passage.", {"name": "sea-cave", "location": "under the cliff", "detail": "flooded passage to the cellar"}),
            beat("mb4", 12, "elements", "DSS should save the ballroom mirror.", {"name": "ballroom mirror", "location": "the ballroom", "detail": "shows the past at certain hours"}),
            beat("mb5", 24, "characters", "DSS should save Mrs. Vale as a character.", {"name": "Mrs. Vale", "role": "housekeeper", "detail": "warms the key-wax"}),
        ],
    },
}


def main() -> None:
    for sheet_id, s in SHEETS.items():
        data = {
            "id": sheet_id,
            "genre": s["genre"],
            "premise": s["premise"],
            "setting": s["setting"],
            "greeting": s.get("greeting", ""),
            "characters": {
                "name1": {"name": s["name1"]["name"], "role": s["name1"]["role"], "description": s["name1"]["description"]},
                "name2": {"name": s["name2"]["name"], "role": s["name2"]["role"], "description": s["name2"]["description"]},
            },
            "writing_style": {
                "directive": s["directive"],
                "guide_directive": s["guide_directive"],
                "dss_directive": s["dss_directive"],
                "taboos": s["taboos"],
            },
            "specificity_profile": "mixed",
            "notes": s["notes"],
            "dss_beats": s["beats"],
        }
        path = SPREADSHEETS / f"{sheet_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {sheet_id}: {len(s['notes'])} notes, {len(s['beats'])} beats")


if __name__ == "__main__":
    main()
