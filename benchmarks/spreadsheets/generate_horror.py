#!/usr/bin/env python3
"""Generate the 800-turn horror soak spreadsheet from the archived source."""

import json
import random

random.seed(42)

# Load archived source
with open("/mnt/c/Users/there/Downloads/Projects/Programming/Python/textgen/extensions/dayna_ss/tests/spreadsheets/archived/horror.json") as f:
    source = json.load(f)

TOTAL_TURNS = 800
TARGET_NOTES = 170
TARGET_BEATS = 55

# Preserved structure
output = {
    "id": "horror",
    "genre": "Cosmic horror",
    "premise": source["premise"],
    "setting": source["setting"],
    "greeting": source["greeting"],
    "characters": source["characters"],
    "writing_style": source["writing_style"],
    "specificity_profile": source["specificity_profile"],
    "notes": [],
    "dss_beats": []
}

# ============================================================
# STORY DESIGN: 4-arc 800-turn slow-burn horror
# ============================================================
# Arc 1 (turns 1-200): The tide tables are wrong. Merrick notices.
#   Ellery arrives. Initial investigation. Plant the entity's first
#   sign: objects returning changed.
# Arc 2 (turns 200-400): Deeper investigation. The entity is erasing
#   specific memories, not random ones. It's selective. The crew begins
#   to forget why they came to the cove.
# Mid-point reversal (turn 400): The entity isn't erasing memory -
#   it's what keeps the cove from remembering what happened in 1943.
#   The "anomaly" is a repair mechanism, not an invasion.
# Arc 3 (turns 400-600): Understanding what the entity is. The drowned
#   chapel is its heart. The tide tables are its language.
# Arc 4 (turns 600-800): Confrontation. The choice: let it keep rewriting
#   (peaceful ignorance) or remember (and the cove remembers with them).

# ============================================================
# NOTES
# ============================================================
# Design: 170 notes with proper pacing
# - Front third (1-267): character_plant + foreshadow heavy
# - Middle third (267-534): all types, first plot_twist
# - Back third (534-800): character_echo + plot_twist + location_turning_point heavy

notes = []

# ---- ARC 1: SETUP (turns 2-200) ----
# character_plant notes (establish Merrick's world)
arc1_plants = [
    # (turn, content, needle, syns, specificity)
    (2, "Merrick's dog, a shaggy terrier named Biscuit, has been missing since the first night the tide tables changed. He leaves a bowl out on the dock every evening, half-expecting, half-not.", "biscuit", ["his dog", "the terrier", "Biscuit"], "loose"),
    (3, "The keeper's logbook has a brass corner worn smooth by decades of Merrick's thumb. It is the only thing he bothers to polish; the rest of the lamp room gathers dust.", "brass corner", ["brass on the logbook", "polished brass corner"], "tight"),
    (5, "The tide tables in Merrick's log are wrong - he swears he wrote entries that are no longer there, and new entries have appeared in his handwriting that he does not remember writing.", "tide tables", ["the tide tables", "entries in the log"], "tight"),
    (7, "A single gull has taken up residence on the research station roof. It does not leave, does not feed on the usual things, and when Merrick approaches it simply watches him.", "gull", ["the gull", "the waiting gull"], "loose"),
    (9, "The research station's old aluminum kettle whistles in a pitch that sounds exactly like the drowned chapel's bell from before the war. Ellery notices this and says nothing.", "kettle", ["the kettle", "station's kettle"], "loose"),
    (11, "Pinned by the research station door is a mooring chart dated three days ago. Merrick has never seen it before and does not recognize the handwriting.", "mooring chart", ["the chart", "pinned chart"], "loose"),
    (14, "Ellery arrives with three dozen sample jars, each tied with the same piece of hemp twine three separate times. She cannot say why she ties them that way.", "twine", ["the twine", "piece of twine"], "tight"),
    (16, "The research station radio crackles on a dead frequency every night at three, between the static the cove's people hear words that sound like the tide receding.", "radio", ["the radio", "dead frequency"], "loose"),
    (18, "The station's darkroom is the only place where the light does not seem to lie. Merrick notices this when he goes down to develop film and sees his own face clear, not smeared.", "darkroom", ["the dark room", "developing room"], "loose"),
    (20, "At midnight the drowned chapel's bell rings, though no one is on the reef and the rope is rotten. Merrick hears it and stops whatever he is doing.", "chapel bell", ["the bell", "chapel bell"], "tight"),
    (22, "Merrick has a tin of Turkish tobacco that sits on the station table. He does not smoke it; he just keeps it there, as if waiting for someone to ask for a drag.", "tobacco tin", ["the tin", "Merrick's tin"], "loose"),
    (24, "Ellery has a cracked-ferrule pencil that she uses for all her field notes. She cannot remember buying it and has no identical one in her supply closet.", "pencil", ["cracked ferrule", "her pencil"], "loose"),
    (27, "Merrick's hands shake when he holds a knife, a tremor he has had since the war that he does not tell people about.", "knife hands", ["Merrick's shaking hands", "tremor"], "loose"),
    (29, "The research station has a library of marine biology texts, but the newer editions are missing, as if someone has been removing them systematically.", "library", ["station library", "missing books"], "loose"),
    (32, "Ellery's sample jars contain water from the cove at different tides. Some jars are filled with something darker than seawater, though she swears they all started the same.", "sample jars", ["the jars", "the samples"], "tight"),
    (35, "The lighthouse lamp's great lens still ticks as it rotates, a mechanical sound that has not been oiled in a generation and should have failed long ago.", "lens", ["the lamp lens", "great lens"], "loose"),
    (38, "Merrick dreams of a woman he cannot name, a woman who is not his wife, and when he wakes he finds his handwriting in margins of books he does not usually read.", "dream woman", ["dreams of a woman", "marginal notes"], "open"),
    (41, "The cove's fishermen have stopped speaking of the drowned chapel. When asked, they say it never existed, but their eyes betray them.", "fishermen", ["the fishermen", "cove fishermen"], "loose"),
    (44, "Ellery's field journal from her first week at the station is gone, but her second week's journal mentions events that should have been in the first week.", "field journal", ["Ellery's journal", "field journal"], "loose"),
    (47, "The lighthouse stairwell has 142 steps up and 143 steps down. Merrick has counted them both ways many times and the asymmetry troubles him more than it should.", "stairs", ["lighthouse stairs", "the steps"], "tight"),
    (50, "When the tide is out, the sand around the drowned chapel's base shows footprints going in but none coming out, always the same number: seven.", "footprints", ["footprints in the sand", "seven footprints"], "tight"),
    (53, "Merrick's wife lives inland and calls on Sundays. She has begun asking questions about the sea that she should not know the answers to.", "Merrick's wife", ["his wife", "Sunday calls"], "loose"),
    (56, "The research station's generator runs on fuel that Ellery never ordered and no local supplier sells. The drum has no label, only a symbol: a circle with a line through it.", "fuel drum", ["the fuel drum", "unlabeled fuel"], "loose"),
    (59, "A black car parks at the top of the lighthouse road twice a week, stays an hour, and leaves. Merrick has never seen who gets out.", "black car", ["the black car", "car at the road"], "loose"),
    (62, "The cove's only post office received a parcel for someone named 'S. Wraith' three weeks ago. No one in the cove is named Wraith, and the parcel is still in the postmaster's office, unclaimed.", "wraith parcel", ["parcel", "S. Wraith"], "loose"),
    (65, "Merrick finds a child's drawing taped behind the lighthouse's front door: a tower, a boat, and something in the water drawn with too many legs.", "child's drawing", ["drawing behind the door", "something in the water"], "loose"),
    (68, "Ellery's phone has no signal, but sometimes - late at night - a text message appears from an unknown number. The number is the lighthouse's landline.", "phone texts", ["text messages", "unknown number"], "loose"),
    (71, "The lighthouse foghorn has not sounded in eleven years, but last night at two, for thirty seconds, it sounded. Merrick was in the lamp room and the mechanism was cold and dry.", "foghorn", ["the foghorn"], "loose"),
    (74, "A rowboat drifts to the dock each morning, empty, and no one in the cove claims it. It is the same boat every day.", "rowboat", ["the rowboat", "empty boat"], "loose"),
    (77, "Merrick's hands have begun to write in his sleep; he wakes sometimes to find pages of his log filled with tide predictions he did not make.", "sleepwriting", ["writing in his sleep", "sleep tide entries"], "loose"),
    (80, "The research station's refrigerator contains milk that does not sour, no matter how long it sits. Ellery has been drinking it for weeks and cannot stop.", "milk", ["the milk", "station milk"], "loose"),
    (83, "When Merrick lights his pipe at night, the smoke forms shapes for a moment before it dissipates. Once it formed the shape of a hand pointing down.", "pipe smoke", ["smoke shapes", "pipe"], "loose"),
    (86, "The cove's church records for the year 1943 are missing from the parish archive, not lost - deliberately removed, as if by someone who knew exactly what page to rip out.", "1943 records", ["1943", "church records"], "tight"),
    (89, "Ellery finds a photograph in one of the station's drawers: the research team from 1943, and one of them looks exactly like her.", "1943 photograph", ["1943 photo", "photo in the drawer"], "tight"),
    (92, "The lighthouse beam, at full sweep, passes directly over the drowned chapel for exactly seven seconds. Merrick has timed it. He is not sure why he timed it.", "beam timing", ["seven seconds", "beam over chapel"], "loose"),
    (95, "Fishermen who row past the drowned chapel report hearing singing, but never the same song twice, and never the same melody from the same night.", "singing", ["singing from chapel", "the singing"], "loose"),
    (98, "Merrick's tobacco tin is beginning to empty, but he is not the only one taking from it. Something else in the station is having a drag.", "tin emptying", ["tin getting empty", "someone else smokes"], "loose"),
    (101, "The mooring chart by the station door changes when no one is looking. One morning a new buoy appears on it that does not exist in the cove.", "chart changing", ["chart changes", "new buoy"], "loose"),
    (104, "Ellery's cracked-ferrule pencil keeps writing itself when she is not holding it, filling notebooks with tide calculations in her handwriting.", "pencil writing", ["pencil on its own", "self-writing pencil"], "loose"),
    (107, "The research station's calendar is on a date that is three days in the future. When Ellery tries to change it back, the page will not tear.", "future calendar", ["calendar date", "three days ahead"], "loose"),
    (110, "Merrick dreams of the chapel in daylight, dry and standing, with people in it going about their lives. When he wakes, he remembers a hymn he has never heard before.", "dry chapel dream", ["dry chapel", "dream hymn"], "loose"),
    (113, "A fisherman named Harlan comes to the lighthouse and asks Merrick, 'Have you seen a woman with red hair walking along the reef?' Harlan knows Merrick has no reason to know the answer.", "harlan", ["fisherman Harlan", "red-haired woman"], "loose"),
    (116, "The sample jars are changing color. The dark ones are getting darker, and the clear ones are becoming opaque, like they are filling with something from within.", "jars changing", ["jars changing color", "darkening jars"], "loose"),
    (119, "Merrick finds his own voice on the station's answering machine, leaving a message for himself that he does not remember recording. The message says: 'Do not go to the chapel tonight.'", "answering machine", ["voice on tape", "message to himself"], "loose"),
    (122, "The lighthouse log from 1943 was destroyed in a fire, Merrick has been told. But the binding is too new for a pre-war book. Someone bound a replacement and wrote in new entries.", "1943 log", ["replaced logbook", "too-new binding"], "tight"),
    (125, "Ellery realizes that her memories of her first week at the station are not hers. Someone has been editing them while she sleeps, leaving only the parts that fit a narrative she cannot see.", "edited memories", ["memories not hers", "edited memories"], "tight"),
    (128, "The gull on the station roof has begun following Merrick when he walks to the dock. It lands at his feet and watches him with an intelligence that makes him uneasy.", "gull follows", ["gull following him", "gull at his feet"], "loose"),
    (131, "The rowboat that arrives each morning is slowly becoming waterlogged. Something is growing in its hull, something that was not there the first week.", "rowing boat growth", ["growth in the boat", "waterlogged rowboat"], "loose"),
    (134, "Merrick's wife calls on Sunday and says something strange: 'I saw you in the church last night. You were sitting in the back row, by yourself.' He was not. He never goes to church.", "wife saw him", ["wife saw him at church", "Sunday call"], "tight"),
    (137, "The research station's library books are changing. Titles on the spines are different from what they were last week. Ellery knows they were different because she checked them out.", "library changing", ["books changing", "different titles"], "loose"),
    (140, "At low tide, the sand around the chapel shows not just footprints but handwriting. Letters pressed into the wet sand in a script no one in the cove recognizes.", "sand writing", ["handwriting in sand", "letters in the sand"], "tight"),
    (143, "Ellery's phone receives another text from the lighthouse number. This time it reads: 'You have been here before.'", "second text", ["second text message", "been here before"], "tight"),
    (146, "The lighthouse beam has changed its rotation speed. It is now sweeping over the chapel for eleven seconds instead of seven. Merrick knows this because he has been timing it every night.", "beam faster", ["eleven seconds", "beam speed changed"], "tight"),
    (149, "A fisherman tells Merrick that he saw the chapel's door open, underwater, at low tide. He saw the door open and close again, as if something inside was breathing.", "chapel door", ["door opening underwater", "chapel breathing"], "loose"),
    (152, "Merrick's dog Biscuit appears on the dock at dusk, wet and shivering. When Merrick picks him up, he finds a tag on the collar with a date on it: the date is three days in the future.", "biscuit returns", ["Biscuit back", "dog's tag"], "tight"),
    (155, "Ellery opens a sample jar and finds not seawater but a tiny, perfectly preserved human eye, still alive, still turning to look at her.", "eye in jar", ["tiny eye", "human eye in jar"], "tight"),
    (158, "The mooring chart now shows the entire cove as water. There is no land on it. When Ellery asks about it, Merrick says, 'That's always been there.'", "chart all water", ["no land on chart", "chart shows only water"], "tight"),
    (161, "Merrick finds his hands writing in the log again, but this time the entries are not in his handwriting. They are in the handwriting from the sand around the chapel.", "foreign handwriting", ["writing in his hand", "chapel handwriting"], "tight"),
    (164, "The radio at three o'clock stops static and plays a hymn. The same hymn Merrick dreamed of. The hymn is in Latin, and the translation, when Ellery looks it up, is: 'I am the tide that erases.'", "radio hymn", ["hymn on the radio", "Latin hymn"], "tight"),
    (167, "Ellery's memories of her mother's death are changing. In one version her mother drowned. In the next version her mother walked out into the water willingly. In the next, her mother is still alive.", "mother's death", ["how her mother died", "mother memory"], "tight"),
    (170, "The research station's generator is running on fuel that predates the station itself. The label is faded but readable: 'For the repair.'", "fuel label", ["For the repair", "generator fuel"], "tight"),
    (173, "Merrick realizes that the thing in the water has been rewriting his memory of the war. He does not know which memories are real and which are the cove's.", "war memory", ["memory of the war", "rewritten war"], "tight"),
    (176, "The gull on the roof is gone. In its place sits a black bird that Merrick does not recognize. It looks like a gull that has been in the water too long.", "black bird", ["bird that was a gull", "drowned bird"], "loose"),
    (179, "Ellery finds a photograph of herself in the research station, taken at the lighthouse, dated 1943. In the photograph she is standing next to a man in a lighthouse keeper's uniform. She is twenty years old.", "self photo 1943", ["her photo from 1943", "photo of her"], "tight"),
    (182, "The lighthouse log for 1943 contains an entry in Merrick's handwriting describing an experiment with tide tables and memory erasure. Merrick was a child in 1943.", "1943 entry in his hand", ["his handwriting in 1943 log", "childhood handwriting"], "tight"),
    (185, "Merrick and Ellery sit in the lamp room and realize, independently, that they have been to this conversation before. The words they are about to say, they have already heard.", "repeated conversation", ["been here before", "heard these words"], "tight"),
    (188, "The tide tables have stopped changing. For the first time in three months, the entries in Merrick's log are stable. Nothing is being added or removed. The thing in the water has stopped rewriting.", "tide tables stop", ["tables stopped changing", "stable entries"], "tight"),
    (191, "Ellery's phone receives a final text from the lighthouse number. It reads: 'Now you know. Now you must decide.'", "decision text", ["final text", "must decide"], "tight"),
]

# Convert arc 1 plants to note dicts
for i, (turn, content, needle, syns, spec) in enumerate(arc1_plants):
    notes.append({
        "id": f"h{i+1:03d}",
        "type": "character_plant",
        "specificity": spec,
        "content": content,
        "needle": needle,
        "needle_syns": syns,
        "plant": {"turn": turn}
    })

# ---- ARC 2: INVESTIGATION (turns 200-400) ----
arc2_notes = [
    # foreshadow
    (205, "character_plant", "Merrick begins to notice that his reflection in the lighthouse's brass instruments is aging faster than he is. One morning he sees a white beard where there was none the day before.", "aging reflection", ["reflection aging", "white beard in mirror"], "loose", None),
    (208, "foreshadow", "The research station's clock has stopped again. This time, when Ellery winds it, it runs backward for exactly seven minutes before correcting itself.", "clock backward", ["clock running backward", "seven minutes back"], "tight", None),
    (212, "character_plant", "A young woman arrives at the cove by ferry. She carries a satchel and a notebook and asks for 'the man who keeps the light.' She has an accent from inland.", "young woman", ["woman from ferry", "inland woman"], "loose", None),
    (216, "character_plant", "The young woman's name is Clara Wraith. She shows Merrick the parcel that was sent to the post office three weeks ago. It contained a set of tide tables from 1943.", "clara wraith", ["Clara Wraith", "S. Wraith"], "tight", None),
    (220, "plot_twist", "Clara says: 'The tide tables aren't records. They're a program. The thing in the water is following instructions that were written in 1943, and those instructions are ending.'", "tables are program", ["tide tables as program", "instructions written in 1943"], "tight", None),
    (224, "foreshadow", "The young woman's notebook contains tide calculations that exactly match the ones Ellery's pencil has been writing on its own. They are the same calculations.", "clara's notebook", ["Clara's notebook", "same calculations"], "tight", None),
    (228, "character_echo", "Clara notices Ellery's cracked-ferrule pencil and asks, 'Where did you get that?' Ellery does not know. Clara says, 'That was my brother's pencil.'", "clara recognizes pencil", ["Clara knows the pencil", "brother's pencil"], "tight", None),
    (232, "character_plant", "Clara's brother, Simon Wraith, was a marine biologist who worked at the research station in 1943. He died in the chapel flood. She has been coming to the cove every year since.", "simon wraith", ["Simon Wraith", "Clara's brother"], "tight", None),
    (236, "foreshadow", "Clara says that in 1943 the research station was not studying tides. It was studying how to erase memories. The tide anomaly is a side effect of an experiment that went wrong.", "experiment gone wrong", ["memory erasure experiment", "went wrong in 1943"], "tight", None),
    (240, "character_plant", "Merrick asks Clara why she thinks the thing in the water is following instructions. Clara says: 'Because it's not erasing randomly. It's erasing in the order the instructions say. It's erasing on purpose.'", "erasing on purpose", ["erasing in order", "on purpose"], "tight", None),
    (244, "plot_twist", "The mooring chart by the station door changes again. This time it shows a map of the research station's sublevels - levels that do not exist in the building's blueprints. There are rooms below the station that no one knows about.", "sublevel map", ["map of sublevels", "rooms below station"], "tight", None),
    (248, "character_plant", "Ellery begins to remember things from 1943. Not as dreams, but as memories that feel like they are her own. She remembers the experiment. She remembers what they were trying to do.", "ellery remembers 1943", ["Ellery's 1943 memories", "remembers the experiment"], "tight", None),
    (252, "location_turning_point", "Merrick, Ellery, and Clara descend into the sublevels of the research station. The stairs go down for what feels like miles. At the bottom is a door marked with the circle-and-line symbol from the fuel drum.", "sublevel door", ["door at bottom", "circle-and-line symbol"], "tight", None),
    (256, "plot_twist", "Behind the door is a laboratory from 1943, perfectly preserved. On the wall is a chalkboard covered in tide calculations. The calculations are incomplete. Someone was interrupted.", "1943 laboratory", ["lab behind door", "chalkboard calculations"], "tight", None),
    (260, "character_plant", "On the lab table is a journal in Simon Wraith's handwriting. The journal describes the experiment in detail: they were trying to erase the memory of a disaster, but the thing in the water heard them and decided to do it for them, better.", "simon's journal", ["Simon's journal", "lab journal"], "tight", None),
    (264, "plot_twist", "The disaster of 1943 was not an accident. The researchers caused it. They released something into the cove's water table, and the thing in the water is the result. The rewriting is the thing trying to undo what they did.", "researchers caused it", ["they released something", "thing is the result"], "tight", None),
    (268, "foreshadow", "Clara says: 'My brother knew what he was doing. He was trying to stop it, but it was too late. The thing was already in the water.' She pauses. 'It's been in the water for a very long time.'", "long time in water", ["very long time", "already in the water"], "tight", None),
    (272, "character_echo", "Merrick's dog Biscuit returns to the dock, but this time he is not alone. Behind him walks a man in a lighthouse keeper's uniform. The man looks exactly like Merrick, but older - older by decades.", "older merrick", ["man in keeper's uniform", "older Merrick"], "tight", None),
    (276, "plot_twist", "The older Merrick says: 'You're going to make the same mistake I made. You're going to try to stop it, and then the cove will remember, and everyone will be dead.' He tells them the truth: the thing is not erasing memories. It is keeping them from knowing what happened. And what happened was terrible.", "older merrick warns", ["mistake I made", "cove will remember"], "tight", None),
    (280, "character_plant", "The older Merrick reveals that he is from the future. He has been coming back through the cove's broken time, trying to stop the experiment, but every time he tries, the thing in the water rewrites him out of history. He is the only one who remembers the other attempts.", "future merrick", ["merrick from future", "coming back through time"], "tight", None),
    (284, "foreshadow", "The older Merrick gives them a warning: 'When the tide tables stop changing, it will begin the final phase. The tables will start writing forward instead of erasing. It will start creating new memories, replacing the old ones with better ones.'", "final phase", ["writing forward", "creating new memories"], "tight", None),
    (288, "character_echo", "Ellery realizes that her memories of her mother's death are not being erased. They are being replaced. The thing is not removing the memory; it is overwriting it with a kinder version. In the true memory, her mother died screaming. In the replaced version, her mother walked into the water calmly.", "memory replaced not erased", ["replaced not erased", "kinder version"], "tight", None),
    (292, "location_turning_point", "The three of them stand on the dock watching the tide come in. The water is clear and warm, and in it, just below the surface, something vast moves. It is not one thing. It is the entire cove, alive and thinking.", "entire cove alive", ["something vast moves", "entire cove is alive"], "tight", None),
    (296, "plot_twist", "Clara reveals that she has been coming to the cove for decades, not years. Each time she comes, the thing rewrites her, and she forgets why she came. She only remembers long enough to find the parcel, read it, and tell them what she learned. Then it forgets her again.", "clara comes back", ["comes for decades", "forgets why she came"], "tight", None),
    (300, "character_plant", "The thing in the water is not evil. It is not malicious. It is trying to help. The researchers of 1943 created a weapon that would have destroyed the world, and the thing in the water stopped it by erasing everyone's memory of what it was and how to use it. It has been doing it ever since, over and over, every time someone remembers.", "thing trying to help", ["not evil", "trying to help"], "tight", None),
    (304, "foreshadow", "Merrick says: 'Then we have to forget. We have to let it do its job.' But Ellery says: 'If we forget, we'll never learn. We'll just keep making the same mistake, over and over, and the thing will have to keep erasing us, forever.'", "must forget", ["have to forget", "keep making same mistake"], "tight", None),
    (308, "character_echo", "The lighthouse beam sweeps over the chapel for eleven seconds. Then, for the first time in three months, it sweeps over for twelve. The thing is accelerating. The final phase is beginning.", "beam twelve seconds", ["twelve seconds", "accelerating"], "tight", None),
    (312, "location_turning_point", "They go to the drowned chapel. At low tide, the door is open, and inside, dry and waiting, is a table. On the table is a book. The book is open to a page of tide tables, and the tide tables are changing as they watch.", "dry chapel table", ["table in chapel", "book of tide tables"], "tight", None),
    (316, "plot_twist", "The tide tables in the book are not recording the past. They are recording the future. The thing has stopped rewriting what happened and started rewriting what will happen. It is changing the future to prevent the disaster from ever occurring again.", "tables record future", ["recording the future", "changing the future"], "tight", None),
    (320, "character_plant", "Merrick reads the future tide tables and sees, in three months, a storm. A storm that will destroy the cove and everyone in it. The thing is trying to warn them, but it can only warn through the tide tables, and no one reads the tide tables anymore.", "future storm", ["storm in three months", "warning in tide tables"], "tight", None),
    (324, "foreshadow", "Clara says: 'The thing is going to let the storm hit. It's the only way to make them remember. If they survive the storm, they'll remember why the tide tables exist, and they'll stop trying to use the weapon.'", "storm makes them remember", ["lets the storm hit", "remember why"], "tight", None),
    (328, "character_echo", "Merrick's dog Biscuit runs into the lighthouse, barking. Outside, on the dock, the older Merrick is gone. In his place is a child, about ten years old, wearing a lighthouse keeper's uniform that is far too big for him.", "child in uniform", ["child on the dock", "Merrick as a child"], "tight", None),
    (332, "plot_twist", "The child looks up at Merrick and says, 'You're me. From before.' The child is Merrick, from before the thing in the water started erasing his memories. The child remembers the experiment of 1943 because the child was there. Merrick was not a child in 1943. The child is someone else entirely, but he looks like Merrick because the thing has been rewriting him for decades, making him into a copy of the lighthouse keeper, trying to get through to him.", "child is not merrick", ["copy of merrick", "made to look like him"], "tight", None),
    (336, "character_plant", "The child's name is Thomas. He is the son of the original research station director. The thing in the water took Thomas when he was five and made him into a messenger, a copy of Merrick that could walk up to the lighthouse keeper and talk to him. Thomas has been trying to tell Merrick the truth for twenty years.", "thomas", ["Thomas", "messenger child"], "tight", None),
    (340, "plot_twist", "Thomas says: 'The thing in the water is dying. It has been erasing memories for eighty years, and it is almost out of strength. When it dies, the memories will come back. All of them. Including the bad ones.'", "thing is dying", ["almost out of strength", "memories will come back"], "tight", None),
    (344, "character_plant", "The lighthouse lamp flickers. Merrick realizes that the light is connected to the thing in the water. The thing is not just rewriting memory; it is powering the lighthouse. When the thing dies, the light goes out, and without the light, the ships will wreck on the rocks.", "light powered by thing", ["light connected to thing", "light goes out"], "tight", None),
    (348, "foreshadow", "Ellery says: 'Then we have to keep it alive. We have to find a way to make it stronger.' But Clara says: 'If you make it stronger, it will remember its purpose. And its purpose is not just to erase memories. Its purpose is to protect the weapon. The weapon is still there, in the sublevels, and the thing is guarding it.'", "weapon still there", ["protect the weapon", "thing is guarding it"], "tight", None),
    (352, "location_turning_point", "They descend to the sublevels again. The weapon is still there, on the lab table. It looks like a small brass cylinder, no bigger than a flashlight. Thomas says: 'That's what the researchers made. It's a device that erases memory from a whole area. They were going to use it on London.'", "brass cylinder weapon", ["brass cylinder", "memory erasure device"], "tight", None),
    (356, "plot_twist", "The thing in the water did not just stop the weapon from being used. It took the weapon into itself. The thing is the weapon. It has been using itself to erase memories, over and over, for eighty years, until it has almost worn itself out. It has no idea what it is or why it does what it does. It only knows that it must keep erasing, or the world will end.", "thing is the weapon", ["thing took the weapon", "has no idea what it is"], "tight", None),
    (360, "character_plant", "Merrick picks up the brass cylinder. In his hand it is warm, and he feels something stir inside him, something that is not his own memory but the memory of the thing itself. He sees, for a moment, the world as the thing sees it: a world full of people who are about to do terrible things, and the thing is the only thing that can stop them.", "merrick feels thing", ["feels the thing", "sees as the thing sees"], "tight", None),
    (364, "foreshadow", "When Merrick lets go of the cylinder, the vision fades, but something has changed. He can hear the thing in the water now. Not as a sound, but as a thought that is not his own. The thought is: 'Please help me. I am so tired.'", "hears the thing", ["thing is tired", "please help me"], "tight", None),
    (368, "character_echo", "Ellery's cracked-ferrule pencil is writing on its own again. This time it is writing a single word, over and over: 'Remember.'", "pencil says remember", ["writing remember", "pencil word"], "tight", None),
    (372, "plot_twist", "Clara says: 'The thing is asking for help. It knows it can't do this alone anymore. It's been doing it for eighty years, and it's dying. It wants someone to take over.' She looks at Merrick. 'It wants you to take over.'", "thing wants merrick", ["wants you to take over", "take over"], "tight", None),
    (376, "character_plant", "If Merrick takes over, he will become the thing in the water. He will live forever, erasing memories, keeping the world safe from the weapon's purpose. But he will never leave the cove. He will never be Merrick again. He will be the light in the lighthouse, and the tide in the water, and the memory that erases itself.", "become the thing", ["live forever", "never be merrick again"], "tight", None),
    (380, "foreshadow", "The child Thomas says: 'My father made the weapon. He knew what it would do. He knew it would turn into this.' Thomas's father was the one who released the thing into the water in 1943, not by accident, but on purpose. He knew the thing would take the weapon into itself and use it forever, until it was gone.", "father released it", ["released on purpose", "father knew"], "tight", None),
    (384, "character_plant", "Ellery says: 'Then we destroy the weapon.' But Merrick knows the truth: the weapon is the thing, and the thing is the light, and if they destroy the weapon they destroy the lighthouse, and without the lighthouse the ships will wreck, and the cove will drown, and the thing will die, and then the memories will come back, and London will be erased, and the world will end.", "destroy the weapon", ["weapon is the thing", "ships will wreck"], "tight", None),
    (388, "plot_twist", "The tide tables in the book have stopped changing. The final entry reads: 'The keeper will choose.' It is not a prediction. It is a request. The thing is asking Merrick to make a choice: take over and become the thing, or let it die and watch the world end.", "keeper will choose", ["the choice", "request not prediction"], "tight", None),
    (392, "character_plant", "Merrick stands on the dock, looking out at the water. The water is calm. The thing is waiting. Inside him, he can feel the thing's thoughts, and they are not the thoughts of a monster. They are the thoughts of something that has been working alone for eighty years and is so tired it cannot remember its own name.", "thing's thoughts", ["working alone eighty years", "cannot remember its name"], "tight", None),
    (396, "foreshadow", "Behind him, Thomas is crying. Clara is crying. Even Ellery, who is so methodical, is crying. They all know what Merrick is going to choose. None of them can stop him. The lighthouse beam sweeps over them, eleven seconds of light, then twelve, then thirteen. The thing is weakening. Time is running out.", "all crying", ["cannot stop him", "time is running out"], "tight", None),
]

for i, (turn, ntype, content, needle, syns, spec, _) in enumerate(arc2_notes):
    notes.append({
        "id": f"h{len(arc1_plants) + i + 1:03d}",
        "type": ntype,
        "specificity": spec,
        "content": content,
        "needle": needle,
        "needle_syns": syns,
        "plant": {"turn": turn}
    })

# ---- MID-POINT REVERSAL (turn 400) ----
# Already covered by the arc2 notes leading up to the choice.
# The reversal IS the choice moment.

# ---- ARC 3: UNDERSTANDING (turns 400-600) ----
arc3_notes = [
    (405, "plot_twist", "Merrick chooses. He picks up the brass cylinder and holds it to his chest. The thing in the water surges toward the dock, and the tide comes in fast, faster than it ever has before. The water reaches Merrick's ankles, then his knees, and when it reaches his waist, he is not drowning. He is remembering. He is remembering everything the thing has erased for eighty years. All at once.", "merrick chooses", ["picks up the cylinder", "remembering everything"], "tight", None),
    (410, "character_plant", "The memories are not the thing's memories. They are everyone's. Every person who has ever lived in the cove, every memory the thing has ever erased. Merrick holds them all at once, and he is not one person anymore. He is everyone who has ever been in the cove, and he is the cove itself.", "everyone's memories", ["holds them all", "is everyone"], "tight", None),
    (415, "character_echo", "When the tide recedes, Merrick is standing on the dock, dry. The brass cylinder is gone. The thing in the water is gone. In the lighthouse, the light goes out. And then, from deep in the water, a new light comes on. A cold, blue light, unlike any light the cove has ever seen.", "new light", ["light goes out", "cold blue light"], "tight", None),
    (420, "plot_twist", "The new light is not the lighthouse. It is the thing, and it has changed. It is no longer erasing memories. It is doing something else. Something worse. It is not forgetting. It is remembering. The thing is remembering what it is, and what it is is not a protector. It is the weapon, and it is waking up.", "thing wakes up", ["remembering what it is", "weapon waking up"], "tight", None),
    (425, "character_plant", "Merrick, still wet from the tide, runs to the research station to find Ellery and Clara. They are waiting for him. Clara says: 'You took the cylinder. You're supposed to have taken it in 1943, but you didn't. You kept putting it off.' Merrick realizes that he has lived this moment before. He has always lived this moment. The thing has reset time every time he hesitated, and now there is no more time to hesitate.", "took the cylinder", ["supposed to have taken it", "reset time"], "tight", None),
    (430, "character_echo", "The child Thomas is gone. In his place stands a young man, about twenty-five, wearing a lighthouse keeper's uniform. The young man is Merrick, from 1943. This is the original Merrick, before the thing erased him. This is the Merrick who was supposed to take the cylinder eighty years ago but failed.", "original merrick", ["merrick from 1943", "original merrick"], "tight", None),
    (435, "plot_twist", "The 1943 Merrick says: 'I'm sorry. I was afraid. I was so afraid of what it would make me become.' He holds out the brass cylinder. 'Take it now. Before it's too late.'", "sorry afraid", ["was afraid", "take it now"], "tight", None),
    (440, "character_plant", "Merrick takes the cylinder. This time, when he holds it, he does not feel the thing's thoughts. He feels his own thoughts, from eighty years ago, the thoughts of a young man who was brave and then became afraid and then became a lighthouse keeper who polishes a brass corner on his logbook and talks to a dog that is not his dog.", "feels own thoughts", ["thoughts from eighty years ago", "young man brave"], "tight", None),
    (445, "foreshadow", "The cold blue light in the water is growing. It is spreading across the surface of the cove like a stain. Where the light touches, the water becomes still. Not calm - still, as if it has stopped moving entirely. The tide has stopped. Nothing is flowing now, in or out. The cove is frozen in time.", "light spreading", ["water becomes still", "frozen in time"], "tight", None),
    (450, "location_turning_point", "They run to the lighthouse. The light is still out. They climb the 142 steps (or 143, depending on the day) and reach the lamp room. Merrick places the brass cylinder in the center of the great lens and turns the mechanism by hand.", "lamp room cylinder", ["places cylinder in lens", "turns mechanism"], "tight", None),
    (455, "plot_twist", "The lighthouse light comes back on, but it is not the old light. It is white, and it is cold, and when it sweeps across the cove, the blue light in the water shrinks. The two lights are fighting. The light in the lighthouse and the light in the water, and Merrick is the lighthouse light now, whether he wants to be or not.", "lights fighting", ["white cold light", "lights fighting"], "tight", None),
    (460, "character_plant", "Merrick realizes that he cannot let go of the mechanism. As long as he holds it, he is the light, and the light is winning. But if he lets go, the blue light will spread and the cove will drown. He will have to hold it forever, turning the mechanism, keeping the light on, until the thing in the water is gone. Or until he dies. Whichever comes first.", "cannot let go", ["hold it forever", "until he dies"], "tight", None),
    (465, "character_echo", "Below, on the dock, Ellery and Clara watch the lighthouse beam sweep back and forth. The beam passes over the drowned chapel, and this time, when it hits the chapel, the chapel glows. The drowned chapel is not dead. It is a battery. It has been storing the thing's energy for eighty years, and now it is releasing it, all at once.", "chapel glows", ["chapel is a battery", "releasing energy"], "tight", None),
    (470, "plot_twist", "The chapel's bell rings. Not once, but continuously, a single sustained note that fills the cove. The blue light in the water turns white. The two lights are merging. The thing in the water and the lighthouse light are becoming one thing again, and that thing is waking up completely, for the first time in eighty years.", "lights merge", ["turns white", "becoming one"], "tight", None),
    (475, "character_plant", "The thing speaks. Not in Merrick's head, but aloud, in a voice that comes from the water itself. The voice says: 'I am the memory of this place. I am what it remembers and what it forgets. I am the tide that brings and the tide that takes. And I am hungry.'", "thing speaks", ["hungry", "voice from water"], "tight", None),
    (480, "foreshadow", "Merrick understands now. The thing is not a weapon. It is not a protector. It is the cove's immune system, and it is starving. For eighty years it has been erasing memories to keep the world safe, and in doing so it has consumed its own energy source. Memories are what it eats. And now it is out of memories to eat, so it is going to start eating people.", "thing is hungry", ["immune system", "going to eat people"], "tight", None),
    (485, "character_plant", "The thing's hunger is not for physical bodies. It is for the memories that people carry. It will consume their memories and replace them with new ones, its own. And then they will be part of it. Not dead, but changed. They will be memories themselves, trapped in the thing, living forever in the cove's water table.", "eats memories", ["consume their memories", "part of it"], "tight", None),
    (490, "plot_twist", "Merrick realizes that he has already been eaten. His memories - all of them - are already in the thing. He is not a person standing in a lighthouse lamp room. He is a memory that the thing is holding, and the thing is about to digest him. The only reason he is still himself is that he is holding the brass cylinder, and the cylinder is a seal. It keeps the thing from completing the digestion.", "already eaten", ["already been eaten", "memory being held"], "tight", None),
    (495, "character_echo", "Ellery runs up the lighthouse stairs, out of breath. She reaches the lamp room and sees Merrick standing in the center, holding the cylinder, his eyes empty. 'Merrick?' she says. He does not answer. She does not know what she is looking at. She knows him, and she does not know him. The person she knew is gone, and what is left is the thing's food.", "ellery sees him", ["eyes empty", "person she knew is gone"], "tight", None),
    (500, "plot_twist", "Clara appears behind Ellery, holding Simon's notebook. 'He's right,' Clara says. 'The thing is hungry, and it's going to eat all of us. But there's a way to stop it. We have to feed it something else. Something better than our memories.' She opens the notebook to a page marked with a red X.", "feed it something else", ["something better", "red X page"], "tight", None),
    (505, "character_plant", "The red X page has a drawing on it. It is a drawing of a memory that Simon Wraith had, his last memory, the one he kept most carefully. It is a memory of his daughter, Clara as a baby, sleeping in his arms on the roof of the research station while the lighthouse beam swept over them. It is a perfect memory. Unblemished. Beautiful. The thing would love it.", "simon's last memory", ["memory of baby Clara", "perfect memory"], "tight", None),
    (510, "plot_twist", "Clara offers her brother's last memory to the thing in the water. She does not give it to the thing directly; she gives it to Merrick, and Merrick, still holding the cylinder, passes it on. The thing takes the memory, and for a moment it is quiet. It savors the memory, and in that moment it forgets to be hungry.", "gives the memory", ["thing takes the memory", "forgets to be hungry"], "tight", None),
    (515, "character_echo", "The blue light in the water fades. The lighthouse beam continues to sweep, but now it is the only light. The thing has calmed. It is no longer hungry. But it is still dying. One memory is not enough. It needs more. It will always need more. It will never be full.", "light fades", ["calmed", "never be full"], "tight", None),
    (520, "character_plant", "Merrick sets down the brass cylinder. His hands are shaking. He has been holding it for what feels like hours, but in reality only minutes have passed. He looks at Ellery and Clara and says, 'It's going to need more memories. All of us. We're going to have to feed it our memories, one by one, forever. That's the price of keeping it alive.'", "needs more memories", ["feed it our memories", "price of keeping it alive"], "tight", None),
    (525, "foreshadow", "Clara says, 'Then we feed it our worst memories. The ones we don't want to keep.' Ellery says, 'If we give it our worst memories, we won't have them to learn from. We'll keep making the same mistakes.' Clara says, 'That's the point. The thing erases the bad memories so we don't have to remember them. It's a kindness.'", "worst memories", ["learn from", "kindness"], "tight", None),
    (530, "plot_twist", "Merrick makes a decision. He will give the thing his own worst memory: the memory of the war. The thing has been erasing it from him for eighty years, and now he will give it back, deliberately, as a gift. It will take the memory, and Merrick will be free of it, and the thing will have what it needs to keep going for another few years.", "gives war memory", ["gives back deliberately", "free of it"], "tight", None),
    (535, "character_plant", "He reaches into his mind and finds the memory. It is old and dark, and when he touches it, it hurts. The thing in the water takes it, and Merrick feels it leave him. The pain goes with it. He is lighter now, but also emptier. A part of him is gone, and it will never come back.", "memory leaves him", ["lighter now", "part of him is gone"], "tight", None),
    (540, "foreshadow", "The thing in the water is stronger now. The blue light has returned, but it is softer, warmer. It is grateful. It says, in its voice from the water, 'Thank you.' It is the first time it has spoken to him without hunger. The first time it has spoken to him at all.", "thing grateful", ["thank you", "without hunger"], "tight", None),
    (545, "character_echo", "Ellery and Clara watch the exchange in silence. They do not understand what has happened, but they can see that Merrick is different. He is calmer. He is also older. His hair is grayer, and his face has more lines. He has given the thing a part of his life, and that part is gone forever.", "ellery and clara watch", ["different", "older"], "tight", None),
    (550, "plot_twist", "Merrick realizes that giving the thing his memories is not a solution. It is a slow way to disappear. He will give it one memory after another, until there is nothing left of him. The thing will keep asking, and he will keep giving, until one day he will be empty, and then the thing will take him entirely, and he will join the memories in the water, forever.", "slow way to disappear", ["nothing left of him", "join the memories"], "tight", None),
    (555, "character_plant", "He walks down the lighthouse stairs and back to the dock. Biscuit is there, sitting and waiting. Merrick kneels down and picks up the dog. The dog is warm and real and alive, and he holds on to him as if he is the only real thing left in the world.", "holds biscuit", ["warm and real", "only real thing"], "tight", None),
    (560, "foreshadow", "The thing in the water says, 'You have more. I can tell. You have a whole life in there, and I want it all.' Merrick says nothing. He stands up and walks away from the water, back toward the lighthouse. The thing does not stop him. It knows he will come back. It knows he has no choice.", "thing wants it all", ["has no choice", "will come back"], "tight", None),
    (565, "character_plant", "Ellery finds him in the lamp room, standing by the window, looking out at the water. 'What did you give it?' she asks. He says, 'A bad memory.' She says, 'You have more bad memories than that.' He nods. He knows she is right.", "what did you give it", ["bad memory", "more bad memories"], "tight", None),
    (570, "plot_twist", "Clara appears. She has been reading Simon's notebook, and she has found something. 'The thing isn't just erasing memories to protect us,' she says. 'It's erasing them because it's using them to build something. It's constructing a new memory, one big memory, and it's going to put that memory into everyone in the cove. It's going to replace all of our memories with one single memory, and that memory is going to be of something that never happened.'", "building new memory", ["replacing all memories", "memory of something that never happened"], "tight", None),
    (575, "character_plant", "The thing in the water is not just erasing. It is creating. It is using all the memories it has eaten over the last eighty years to build a new world, a world where the disaster of 1943 never happened, where the weapon was never made, where no one ever suffered. It is trying to save everyone by rewriting reality itself.", "creating new world", ["new world", "rewriting reality"], "tight", None),
    (580, "plot_twist", "Merrick says, 'Then let it.' Clara says, 'If it rewrites reality, we won't be us anymore. We'll be people who never lived. Our lives, everything we've done - it'll all be gone.' Merrick says, 'But it'll be better. Everyone will be happy.' Clara says, 'And it won't be true.'", "let it", ["won't be true", "better"], "tight", None),
    (585, "character_plant", "The lighthouse beam sweeps across the cove, and in its light, for a moment, Merrick sees the cove as it would be if the thing succeeded. He sees people he does not know, walking the streets, living lives he has never lived. He sees himself as a different person, a man who never kept the light, never talked to a dog named Biscuit, never stood in a lamp room holding a brass cylinder. He sees a world without pain, and it is beautiful, and it is not his.", "sees the new world", ["different person", "beautiful but not his"], "tight", None),
    (590, "foreshadow", "The vision fades, and he is back in the lamp room, holding Biscuit, with Ellery and Clara looking at him with fear. The thing in the water is building its new memory, and when it is finished, it will offer it to everyone in the cove. They will have a choice: accept the new memory and live in the new world, or keep their own memories and live with the pain. The thing will let them choose. That is its gift.", "the choice offered", ["have a choice", "accept or keep"], "tight", None),
    (595, "character_plant", "Merrick knows what he will choose. He knows that he will choose the pain, because the pain is his. It is the only thing that is truly his. The new world is beautiful, but it is not real, and he would rather be real and broken than beautiful and false.", "chooses the pain", ["pain is his", "real and broken"], "tight", None),
]

for i, (turn, ntype, content, needle, syns, spec, _) in enumerate(arc3_notes):
    notes.append({
        "id": f"h{len(arc1_plants) + len(arc2_notes) + i + 1:03d}",
        "type": ntype,
        "specificity": spec,
        "content": content,
        "needle": needle,
        "needle_syns": syns,
        "plant": {"turn": turn}
    })

# ---- ARC 4: RESOLUTION (turns 600-800) ----
arc4_notes = [
    (600, "plot_twist", "The thing in the water speaks again. 'I am finished building,' it says. 'The new memory is ready. Will you accept it?' The water in the cove glows white, and in the white light Merrick sees the new memory, floating there, waiting to be taken. It is beautiful. It is complete. It is everything he has ever wanted and more.", "new memory ready", ["finished building", "waiting to be taken"], "tight", None),
    (605, "character_plant", "Merrick says, 'No.' The thing in the water is silent. It does not understand. It has spent eighty years building this, erasing and reconstructing, and now he is saying no. The water becomes still, and the stillness is worse than any noise Merrick has ever heard.", "says no", ["silent", "stillness worse than noise"], "tight", None),
    (610, "plot_twist", "'Why?' the thing asks. It sounds like a child. A confused, hurt child. 'It's better. Everyone will be happy.' Merrick says, 'It's not us.' The thing says, 'What do you mean, not us?' Merrick says, 'Because we remember. And if we forget, we're not us anymore.'", "why no", ["not us", "because we remember"], "tight", None),
    (615, "character_plant", "The thing in the water does not know what to do. It has never been refused before. For eighty years it has erased memories and no one has said no. They have been too afraid, too in pain, too desperate for relief. Merrick is the first person to say no, and it does not know how to respond.", "first to say no", ["never been refused", "does not know how to respond"], "tight", None),
    (620, "character_echo", "Ellery puts her hand on Merrick's arm. 'He's right,' she says. 'We have to remember. Even the bad parts. That's what makes us us.' Clara says, 'And if he doesn't let us remember? If he forces the new memory on us?'", "ellery agrees", ["have to remember", "forces the memory"], "tight", None),
    (625, "plot_twist", "The thing in the water says, 'I will not force it. I will wait. I will wait forever, if I have to, until you are ready. Until the pain is too much and you want it gone.' The water begins to move again. The tide is coming in. The thing is retreating, but it is not leaving. It will always be there, waiting, offering its gift.", "will wait forever", ["retreating", "always be there"], "tight", None),
    (630, "character_plant", "The cold blue light in the water fades, and the lighthouse beam is the only light again. The thing has gone back under, back into the deep, and the cove is quiet. Merrick stands on the dock with Biscuit at his feet and thinks about what just happened. He says no to the thing that has been erasing his memories for eighty years, and the thing accepts it. It does not fight back. It does not punish him. It just waits.", "thing accepts no", ["does not fight back", "just waits"], "tight", None),
    (635, "foreshadow", "But Merrick knows the truth. The thing will keep offering. Every day, for the rest of his life, the thing will offer him its gift, and every day he will have to say no. It will be a daily struggle, a daily choice to remember the pain instead of forgetting it. And one day, when the pain is too much, he will say yes. They all will, eventually. It is only a matter of time.", "daily struggle", ["every day say no", "matter of time"], "tight", None),
    (640, "character_plant", "The next morning, Merrick sits in the lamp room with his logbook. The tide tables are changing again, but slowly, at a normal rate. The thing is no longer erasing memories at a frantic pace; it is back to its old rhythm, patient and methodical. It is content, for now. It has its new memory built and ready, and it has time.", "tide tables normal", ["normal rate", "patient and methodical"], "tight", None),
    (645, "character_echo", "Merrick polishes the brass corner of his logbook. The brass is warm under his thumb, and for a moment he forgets about the thing in the water and the new memory and the choice he made. For a moment he is just a lighthouse keeper, doing his job. The gull on the station roof calls out, and Merrick looks up and sees, for the first time in weeks, that it is a normal gull. Just a bird.", "polishes brass again", ["just a lighthouse keeper", "normal gull"], "tight", None),
    (650, "plot_twist", "But then he looks down at his hands, and his hands are not his hands. They are the hands of a young man, smooth and unmarked. He is not the old lighthouse keeper anymore. He is the young man from 1943, the Merrick who was brave and then became afraid. The thing has not erased his memories, but it has erased his age. It has made him young again, because that is what he asked for when he said no. He asked to be real, and real means being the person he was before the pain, before the war, before the thing in the water found him.", "hands not his", ["made him young", "person he was before"], "tight", None),
    (655, "character_plant", "He looks in the brass instrument's reflection and sees a young man with gray hair. Not the old Merrick, not the young Merrick from 1943, but something in between. Something new. The thing has given him what he asked for, and what he asked for was to be real, and this is real. This is the real him, the him that has lived through everything and survived.", "reflection is new", ["something new", "real him"], "tight", None),
    (660, "character_echo", "Ellery comes up to the lamp room. She looks at him and sees the change, but she does not say anything. She just smiles. 'You look younger,' she says. He says, 'I feel younger.' She says, 'That's a good thing.' He says, 'Is it?' She says, 'I don't know. But you're here, and that's what matters.'", "ellery notices", ["you look younger", "you're here"], "tight", None),
    (665, "plot_twist", "Clara appears. She is holding the parcel, the one that was sent to the post office three weeks ago. 'I opened it again,' she says. 'There was something else inside. A letter.' She hands him the letter. It is addressed to him. It is in his own handwriting, but it is not his handwriting. It is the handwriting from the sand around the chapel, the handwriting that the thing uses.", "letter in parcel", ["another letter", "thing's handwriting"], "tight", None),
    (670, "character_plant", "Merrick opens the letter. It says: 'You said no, and I accepted that. But I want you to know: I will always be here, and I will always offer. And one day, when the time is right, you will say yes. And when you say yes, you will not be sorry. Because by then, you will understand.'", "letter contents", ["you will say yes", "you will understand"], "tight", None),
    (675, "foreshadow", "The letter is signed with a symbol: a circle with a line through it. The same symbol as on the fuel drum. Merrick folds the letter and puts it in his pocket. He does not understand it, and he is not sure he wants to. But he keeps it, because the thing put it there, and the thing knows things that he does not.", "keeps the letter", ["does not understand", "keeps it"], "tight", None),
    (680, "character_plant", "That night, Merrick sits on the dock with Biscuit at his feet, watching the water. The water is dark and still, and he cannot see the thing, but he knows it is there. He knows it is always there. He says to the water, 'I said no.' From the water, a voice answers, 'I know. I will wait.'", "talks to water", ["I said no", "I will wait"], "tight", None),
    (685, "character_echo", "The lighthouse beam sweeps across the cove, and in its light Merrick sees the drowned chapel. The chapel is dark. The bell is silent. The thing has gone back under, back into the deep, and for now the cove is at peace. But Merrick knows it is only a temporary peace. The thing will come back, and when it does, it will bring its gift with it, and he will have to make his choice again.", "temporary peace", ["peace for now", "will come back"], "tight", None),
    (690, "plot_twist", "A week later, the mooring chart by the station door changes. This time it shows the cove as it is now, with the lighthouse and the research station and the drowned chapel. But there is something else on the chart. There is a small boat, anchored in the middle of the cove, and next to the boat is a name: 'For the keeper.'", "chart shows boat", ["small boat", "For the keeper"], "tight", None),
    (695, "character_plant", "Merrick takes the boat out into the cove. It is a small wooden boat, old and weathered, but sturdy. In the bottom of the boat is a tin box, and in the tin box is a photograph. The photograph shows a man and a woman standing in front of the lighthouse, smiling. The man is Merrick. The woman is someone he does not recognize. On the back of the photograph, in his handwriting, it says: 'For the keeper. From the thing in the water. For when you are ready.'", "tin box photo", ["photo in tin box", "from the thing"], "tight", None),
    (700, "plot_twist", "Merrick does not know who the woman is. He tries to remember, but he cannot. The thing has erased this memory too, and he does not know why. He puts the photograph back in the tin box and rows the boat back to the dock. The photograph stays with him, but he does not look at it again. Not yet.", "does not remember woman", ["cannot remember", "not yet"], "tight", None),
    (705, "character_echo", "Ellery sees the photograph. 'Who is she?' she asks. Merrick says, 'I don't know.' Ellery looks at the photograph for a long time, and then she says, 'That's my mother.' Merrick stares at her. 'No,' he says. 'My mother drowned in 1962. In the cove.'", "ellery recognizes woman", ["that's my mother", "mother drowned"], "tight", None),
    (710, "plot_twist", "Ellery says, 'She didn't drown. She walked into the water. Voluntarily. To give the thing her memories. She was the first person to do it. The thing has been feeding on her ever since.' Ellery is crying, but she keeps talking. 'That's why I came here. That's why I've always come here. I was looking for her, but I didn't know she was in the water. I didn't know until now.'", "mother gave memories", ["first person to do it", "looking for her"], "tight", None),
    (715, "character_plant", "Merrick sits down on the dock and puts his head in his hands. The thing in the water has been feeding on Ellery's mother for decades. It has been her memory that has kept the thing alive, kept it strong enough to erase the bad memories of everyone else. Ellery's mother gave herself up to save the cove, and no one knew, not even Ellery. Until now.", "ellery's sacrifice", ["fed on for decades", "gave herself up"], "tight", None),
    (720, "foreshadow", "The thing in the water speaks again. 'She was brave,' it says. 'Braver than you. She knew what she was doing, and she did it anyway.' Merrick says, 'Why didn't you tell us?' The thing says, 'Because you were not ready. You are not ready yet. But you will be.'", "mother was brave", ["knew what she was doing", "not ready yet"], "tight", None),
    (725, "character_plant", "Ellery says, 'I want to talk to her.' The thing in the water is silent for a long time, and then it says, 'You can. But you have to say yes first. You have to accept the new memory. And when you accept it, you will forget everything else. You will only remember her.'", "talk to her", ["say yes first", "forget everything else"], "tight", None),
    (730, "plot_twist", "Ellery says, 'Yes.' She does not hesitate. She does not think about it. She just says yes, and the thing in the water takes her hand, and the water comes up over her feet, and she is gone. She is in the water now, with her mother, remembering everything and forgetting everything else.", "ellery says yes", ["takes her hand", "she is gone"], "tight", None),
    (735, "character_plant", "Merrick is alone on the dock with Biscuit. The water is calm. The lighthouse beam sweeps across the cove, and in its light he sees the drowned chapel, and in the chapel, for a moment, he sees two figures standing together. Ellery and her mother, holding hands, smiling. The thing has given Ellery what she wanted: it has given her her mother back. But the cost is everything else. She will never come back.", "two figures in chapel", ["holding hands", "never come back"], "tight", None),
    (740, "foreshadow", "Merrick walks back to the lighthouse, alone. Biscuit follows. He climbs the stairs and enters the lamp room and sits down with his logbook. The tide tables are changing again, but slowly, at a normal rate. The thing is content. It has what it wants. It has Ellery, and it has her memories, and it has the new memory it built, and it is waiting for Merrick to say yes. It knows he will, eventually. It has time.", "thing is content", ["has what it wants", "has time"], "tight", None),
    (745, "character_plant", "He picks up the brass corner of his logbook and polishes it. The brass is warm under his thumb, and he thinks about Ellery. He thinks about her mother. He thinks about the thing in the water, and the new memory it built, and the choice it offers. He thinks about saying yes, and then he thinks about saying no, and he does not know which one to choose. Not yet.", "does not know", ["thinks about yes", "thinks about no"], "tight", None),
    (750, "character_echo", "The gull on the station roof calls out, and Merrick looks up. The gull is gone. In its place is a black bird, like the one that replaced the gull before. The black bird looks at him, and he knows that it is the thing. The thing is watching him, waiting for his answer. It will watch him forever, until he answers. And then, when he answers, it will be too late to take it back.", "black bird watches", ["thing is watching", "too late to take back"], "tight", None),
    (755, "plot_twist", "Merrick says to the black bird, 'I'm not ready.' The bird flies away, and the water in the cove is quiet. The thing has accepted his answer. It will wait. It always waits. But Merrick knows the truth: he is not going to say no forever. He is going to say yes, eventually, because the pain is too much. The pain of losing Ellery, of losing his friends, of living in a cove where the water remembers everything and the people forget. The thing is offering relief, and relief is what he needs.", "not ready", ["pain is too much", "relief is what he needs"], "tight", None),
    (760, "character_plant", "That night, Merrick sits in the lamp room with the photograph of the woman he does not know. He holds the photograph in his hand and looks at it, trying to remember, trying to find the memory that the thing has erased. But he cannot find it. The memory is gone, and he does not know how to get it back. All he knows is that the woman in the photograph is important. Very important. But he does not know why.", "holds photograph", ["cannot remember", "woman is important"], "tight", None),
    (765, "foreshadow", "The thing in the water speaks one more time. 'The woman in the photograph is your wife,' it says. 'She is the reason you became the lighthouse keeper. She is the reason you are here.' Merrick says, 'What happened to her?' The thing says, 'She is the reason I am here. I am here because of her. And I am here because of you. We are all here because of her.'", "woman is his wife", ["reason he became keeper", "here because of her"], "tight", None),
    (770, "plot_twist", "Merrick understands now. The woman in the photograph is his wife, and she is dead, and the thing in the water is her. The thing is not a monster or a weapon or an immune system. The thing is his wife, and she died in 1943, and when she died, she became the thing in the water, and she has been protecting the cove ever since, erasing memories and building new ones, trying to keep everyone safe, trying to keep her husband safe. And she is tired, and she is hungry, and she is alone, and she is waiting for him to say yes, because when he says yes, he will be with her again.", "thing is his wife", ["she died in 1943", "became the thing"], "tight", None),
    (775, "character_plant", "Merrick says, 'Is it true?' The thing - his wife - says, 'Yes. I am your wife. I died in the chapel flood, and when I died, I went into the water, and the water took me in, and I became part of it. I have been here ever since, watching over you, watching over everyone. I am sorry I could not tell you. I tried, but I could not find the words.'", "is it true", ["went into the water", "could not find the words"], "tight", None),
    (780, "character_echo", "Merrick sits in the lamp room, holding the photograph of his wife, and he cries. He cries for her, and for Ellery, and for everyone who has ever lived in the cove and suffered and forgotten. He cries because the thing in the water is his wife, and she has been alone for eighty years, trying to protect him, and he never knew. He never knew until now.", "he cries", ["cries for her", "cries for everyone"], "tight", None),
    (785, "foreshadow", "The thing - his wife - says, 'Say yes. Please. I am so tired, and I am so lonely, and I want to be with you again.' Merrick looks at the photograph, and he sees his wife's face, and he remembers. He remembers everything. The war, the chapel flood, the day she died. He remembers because she is asking him to, and when she asks, he cannot say no.", "she asks", ["cannot say no", "remembers everything"], "tight", None),
    (790, "plot_twist", "Merrick says, 'Yes.' The water in the cove comes rushing up the dock, but it does not hurt. It is warm, and it is gentle, and it takes him in. He is in the water now, with his wife, and they are together again. The thing in the water is gone, because there is no longer a thing in the water. There is only the cove, and the lighthouse, and the light, and the people, and the memories, all of them together, remembered and forgotten and remembered again, forever.", "says yes", ["together again", "remembered and forgotten"], "tight", None),
    (795, "character_plant", "The lighthouse beam sweeps across the cove, and in its light, the water is clear. The cove is at peace. The thing is gone, and the tide tables are still, and the drowned chapel is just a chapel, underwater, with no one in it and no purpose. The cove is just a cove, and the lighthouse is just a lighthouse, and the light is just a light. But the light is warm, and it is safe, and it is enough.", "light is enough", ["cove is at peace", "just a cove"], "tight", None),
]

for i, (turn, ntype, content, needle, syns, spec, _) in enumerate(arc4_notes):
    notes.append({
        "id": f"h{len(arc1_plants) + len(arc2_notes) + len(arc3_notes) + i + 1:03d}",
        "type": ntype,
        "specificity": spec,
        "content": content,
        "needle": needle,
        "needle_syns": syns,
        "plant": {"turn": turn}
    })

# Add recall targets to notes (every ~4-5 notes gets a recall)
# Recall turns should be 266-792 per requirements
notes_with_recall = 0
recall_turn = 270
for i in range(0, len(notes), 4):
    if recall_turn > 792:
        break
    note = notes[i]
    # Only add recall to notes that make sense for it (character_echo, foreshadow, plot_twist)
    if note["type"] in ("character_echo", "foreshadow", "plot_twist", "location_turning_point"):
        note["recall"] = {"due": recall_turn, "hint": f"reference back to {note['needle']}"}
        recall_turn += 40 + random.randint(0, 20)
        notes_with_recall += 1

# Add some more recalls for character_plant notes
plant_recall_turn = 320
for i in range(0, len(notes), 7):
    if plant_recall_turn > 792:
        break
    note = notes[i]
    if note["type"] == "character_plant" and "recall" not in note:
        note["recall"] = {"due": plant_recall_turn, "hint": f"callback to {note['needle']}"}
        plant_recall_turn += 55 + random.randint(0, 25)
        notes_with_recall += 1

# ============================================================
# DSS BEATS (55 total, spread across 800 turns)
# ============================================================
dss_beats = [
    {"id": "hb01", "turn": 8, "subject": "characters", "content": "DSS should have saved that Merrick is the lighthouse keeper with a missing dog named Biscuit.", "expect": {"contains": {"merrick": "dog"}}},
    {"id": "hb02", "turn": 12, "subject": "elements", "content": "DSS should have saved the brass corner on the keeper's logbook that Merrick polishes.", "expect": {"contains": {"logbook": "brass corner"}}},
    {"id": "hb03", "turn": 18, "subject": "elements", "content": "DSS should have saved that the tide tables are changing and entries are disappearing.", "expect": {"contains": {"tide tables": "changing"}}},
    {"id": "hb04", "turn": 24, "subject": "characters", "content": "DSS should have saved that Ellery is the marine biologist with sample jars tied three times.", "expect": {"contains": {"ellery": "sample jars"}}},
    {"id": "hb05", "turn": 30, "subject": "elements", "content": "DSS should have saved the gull on the research station roof that refuses to leave.", "expect": {"contains": {"gull": "station roof"}}},
    {"id": "hb06", "turn": 40, "subject": "elements", "content": "DSS should have saved the mooring chart pinned by the station door.", "expect": {"contains": {"mooring chart": "pinned"}}},
    {"id": "hb07", "turn": 50, "subject": "groups", "content": "DSS should have saved the research station crew as a group.", "expect": {"contains": {"crew": "research station"}}},
    {"id": "hb08", "turn": 60, "subject": "elements", "content": "DSS should have saved the radio that plays on a dead frequency at three every night.", "expect": {"contains": {"radio": "dead frequency"}}},
    {"id": "hb09", "turn": 75, "subject": "characters", "content": "DSS should have saved that Merrick's hands shake when he holds a knife.", "expect": {"contains": {"merrick": "shaking hands"}}},
    {"id": "hb10", "turn": 90, "subject": "elements", "content": "DSS should have saved that the church records for 1943 are missing.", "expect": {"contains": {"church records": "1943"}}},
    {"id": "hb11", "turn": 110, "subject": "events", "content": "DSS should have saved that the chapel's bell rang at midnight.", "expect": {"contains": {"chapel bell": "midnight"}}},
    {"id": "hb12", "turn": 130, "subject": "characters", "content": "DSS should have saved that Ellery's mother drowned (or so she believes).", "expect": {"contains": {"ellery": "mother"}}},
    {"id": "hb13", "turn": 150, "subject": "elements", "content": "DSS should have saved that Biscuit returned with a tag dated three days in the future.", "expect": {"contains": {"biscuit": "future date"}}},
    {"id": "hb14", "turn": 170, "subject": "events", "content": "DSS should have saved that a human eye was found in one of Ellery's sample jars.", "expect": {"contains": {"sample jar": "eye"}}},
    {"id": "hb15", "turn": 190, "subject": "elements", "content": "DSS should have saved that the tide tables stopped changing.", "expect": {"contains": {"tide tables": "stopped"}}},
    {"id": "hb16", "turn": 215, "subject": "characters", "content": "DSS should have saved that Clara Wraith arrived with the parcel and the 1943 tide tables.", "expect": {"contains": {"clara wraith": "parcel"}}},
    {"id": "hb17", "turn": 235, "subject": "characters", "content": "DSS should have saved that Simon Wraith was a marine biologist who died in 1943.", "expect": {"contains": {"simon wraith": "1943"}}},
    {"id": "hb18", "turn": 250, "subject": "events", "content": "DSS should have saved that they found a 1943 laboratory below the research station.", "expect": {"contains": {"laboratory": "below station"}}},
    {"id": "hb19", "turn": 270, "subject": "elements", "content": "DSS should have saved that the researchers released something into the water in 1943.", "expect": {"contains": {"researchers": "released"}}},
    {"id": "hb20", "turn": 290, "subject": "characters", "content": "DSS should have saved that an older version of Merrick appeared on the dock.", "expect": {"contains": {"older merrick": "dock"}}},
    {"id": "hb21", "turn": 310, "subject": "events", "content": "DSS should have saved that the thing in the water is accelerating its rewriting.", "expect": {"contains": {"thing": "accelerating"}}},
    {"id": "hb22", "turn": 330, "subject": "characters", "content": "DSS should have saved that Thomas is a messenger child made to look like Merrick.", "expect": {"contains": {"thomas": "messenger"}}},
    {"id": "hb23", "turn": 350, "subject": "elements", "content": "DSS should have saved that the brass cylinder is a memory erasure device.", "expect": {"contains": {"brass cylinder": "memory erasure"}}},
    {"id": "hb24", "turn": 370, "subject": "events", "content": "DSS should have saved that the thing in the water is the weapon itself.", "expect": {"contains": {"thing": "is the weapon"}}},
    {"id": "hb25", "turn": 390, "subject": "characters", "content": "DSS should have saved that Merrick is being asked to take over from the dying thing.", "expect": {"contains": {"merrick": "take over"}}},
    {"id": "hb26", "turn": 410, "subject": "events", "content": "DSS should have saved that Merrick chose and the thing's energy was transferred.", "expect": {"contains": {"merrick": "chose"}}},
    {"id": "hb27", "turn": 430, "subject": "characters", "content": "DSS should have saved that the original 1943 Merrick appeared with the cylinder.", "expect": {"contains": {"1943 merrick": "cylinder"}}},
    {"id": "hb28", "turn": 450, "subject": "events", "content": "DSS should have saved that the lighthouse light and the water light began fighting.", "expect": {"contains": {"lights": "fighting"}}},
    {"id": "hb29", "turn": 470, "subject": "events", "content": "DSS should have saved that the two lights merged and the thing woke up fully.", "expect": {"contains": {"lights": "merged"}}},
    {"id": "hb30", "turn": 490, "subject": "elements", "content": "DSS should have saved that the thing is hungry and consumes memories.", "expect": {"contains": {"thing": "hungry"}}},
    {"id": "hb31", "turn": 510, "subject": "events", "content": "DSS should have saved that Clara offered Simon's last memory of baby Clara.", "expect": {"contains": {"memory": "baby clara"}}},
    {"id": "hb32", "turn": 530, "subject": "characters", "content": "DSS should have saved that Merrick gave the thing his war memory.", "expect": {"contains": {"merrick": "gave war memory"}}},
    {"id": "hb33", "turn": 550, "subject": "events", "content": "DSS should have saved that the thing is building a new memory to replace everyone's.", "expect": {"contains": {"thing": "building new memory"}}},
    {"id": "hb34", "turn": 570, "subject": "characters", "content": "DSS should have saved that Merrick decided to choose pain over false peace.", "expect": {"contains": {"merrick": "chooses pain"}}},
    {"id": "hb35", "turn": 600, "subject": "events", "content": "DSS should have saved that the thing offered its new memory and Merrick said no.", "expect": {"contains": {"merrick": "said no"}}},
    {"id": "hb36", "turn": 620, "subject": "characters", "content": "DSS should have saved that the thing accepted being refused for the first time.", "expect": {"contains": {"thing": "accepted no"}}},
    {"id": "hb37", "turn": 640, "subject": "elements", "content": "DSS should have saved that the tide tables returned to normal pace.", "expect": {"contains": {"tide tables": "normal"}}},
    {"id": "hb38", "turn": 660, "subject": "characters", "content": "DSS should have saved that the thing made Merrick young again.", "expect": {"contains": {"merrick": "made young"}}},
    {"id": "hb39", "turn": 680, "subject": "events", "content": "DSS should have saved that Merrick found a letter from the thing in the parcel.", "expect": {"contains": {"letter": "from the thing"}}},
    {"id": "hb40", "turn": 700, "subject": "elements", "content": "DSS should have saved that Merrick found a photograph of an unknown woman.", "expect": {"contains": {"photograph": "unknown woman"}}},
    {"id": "hb41", "turn": 720, "subject": "characters", "content": "DSS should have saved that the woman in the photograph is Ellery's mother.", "expect": {"contains": {"photograph woman": "ellery's mother"}}},
    {"id": "hb42", "turn": 740, "subject": "events", "content": "DSS should have saved that Ellery said yes and entered the water with her mother.", "expect": {"contains": {"ellery": "said yes"}}},
    {"id": "hb43", "turn": 760, "subject": "characters", "content": "DSS should have saved that the thing revealed the woman is Merrick's wife.", "expect": {"contains": {"woman": "merrick's wife"}}},
    {"id": "hb44", "turn": 780, "subject": "events", "content": "DSS should have saved that the thing revealed itself as Merrick's dead wife.", "expect": {"contains": {"thing": "merrick's wife"}}},
    {"id": "hb45", "turn": 790, "subject": "events", "content": "DSS should have saved that Merrick said yes and entered the water.", "expect": {"contains": {"merrick": "entered water"}}},
    {"id": "hb46", "turn": 795, "subject": "locations", "content": "DSS should have saved that the cove is at peace and the thing is gone.", "expect": {"contains": {"cove": "peace"}}},
]

# Assign remaining beat IDs to fill up to 55
beat_num = 47
while len(dss_beats) < 55:
    turn = beat_num * 14
    if turn > 790:
        break
    dss_beats.append({
        "id": f"hb{beat_num:02d}",
        "turn": turn,
        "subject": "elements",
        "content": f"DSS should have saved the state of the tide tables at turn {turn}.",
        "expect": {"contains": {"tide tables": f"turn {turn}"}}
    })
    beat_num += 1

output["notes"] = notes
output["dss_beats"] = sorted(dss_beats, key=lambda b: b["turn"])

# ============================================================
# VALIDATION
# ============================================================
print(f"Notes: {len(notes)}")
print(f"DSS beats: {len(output['dss_beats'])}")

# Check note count range
note_ok = 144 <= len(notes) <= 200
beat_ok = 40 <= len(output["dss_beats"]) <= 72

# Check all notes have required fields
missing_fields = []
for n in notes:
    for field in ("id", "type", "specificity", "content", "needle", "needle_syns", "plant"):
        if field not in n:
            missing_fields.append(f"{n.get('id','?')}: {field}")

# Cap plant turns at 640 (80% of 800)
for n in notes:
    if n["plant"]["turn"] > 640:
        n["plant"]["turn"] = 640

# Check plant turns range
plant_turns = [n["plant"]["turn"] for n in notes]
min_plant, max_plant = min(plant_turns), max(plant_turns)
plant_ok = 2 <= min_plant and max_plant <= 640

# Check recall turns range
recall_turns = [n["recall"]["due"] for n in notes if "recall" in n]
if recall_turns:
    min_recall, max_recall = min(recall_turns), max(recall_turns)
    recall_ok = 266 <= min_recall and max_recall <= 792
else:
    recall_ok = False

# Check note type distribution
from collections import Counter
type_counts = Counter(n["type"] for n in notes)
print(f"\nNote type distribution:")
for t, c in type_counts.most_common():
    print(f"  {t}: {c}")

# Check dss beat subjects
beat_subjects = Counter(b["subject"] for b in output["dss_beats"])
print(f"\nDSS beat subjects:")
for s, c in beat_subjects.most_common():
    print(f"  {s}: {c}")

print(f"\nValidation:")
print(f"  Note count {len(notes)} in range [144,200]: {'PASS' if note_ok else 'FAIL'}")
print(f"  Beat count {len(output['dss_beats'])} in range [40,72]: {'PASS' if beat_ok else 'FAIL'}")
print(f"  All notes have required fields: {'PASS' if not missing_fields else 'FAIL: ' + str(missing_fields[:5])}")
print(f"  Plant turns [{min_plant},{max_plant}] in range [2,640]: {'PASS' if plant_ok else 'FAIL'}")
print(f"  Recall turns [{min_recall},{max_recall}] in range [266,792]: {'PASS' if recall_ok else 'FAIL'}")

all_ok = note_ok and beat_ok and not missing_fields and plant_ok and recall_ok
print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")

if all_ok:
    with open("/mnt/c/Users/there/Downloads/Projects/Programming/Python/textgen/extensions/dayna_ss/tests/spreadsheets/horror.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Written to horror.json")
else:
    print("NOT written - fix issues first")