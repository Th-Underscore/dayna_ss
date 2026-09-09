#!/usr/bin/env python3
"""Re-voice the existing soak spreadsheets into varied person combinations.

Keeps cozy_mystery as the 1P-guide / 3P-DSS control. Rewrites only the
writing_style block (directive / guide_directive / dss_directive); taboos and
all notes/beats stay untouched.
"""
import json
from pathlib import Path

SPREADSHEETS = Path(__file__).parent / "spreadsheets"

NEW_VOICES = {
    "noir_detective": {
        "directive": ("Hardboiled noir, cynical and weary. Short declarative sentences. One paragraph per beat. "
                      "Dual first-person narration: Marlowe and Vivienne each narrate in their own 'I' — the two "
                      "perspectives must never collapse into one."),
        "guide_directive": ("First-person past tense as Marlowe, the detective. Terse, wry, world-weary; you "
                            "narrate only what Marlowe sees, feels, or decides. Short, flat declarations, "
                            "hard-boiled idiom. 'I' is always Marlowe — never Vivienne's perspective."),
        "dss_directive": ("First-person past tense as Vivienne, the nightclub singer — 'I' is ALWAYS Vivienne; "
                          "Marlowe is 'he' in your narration, or 'you' only when addressing him directly. Elegant, "
                          "guarded, faintly dangerous; short, flat declarations beneath the polish. One paragraph "
                          "per beat. Hard rules: never let 'I' drift to Marlowe — his perspective is never yours; "
                          "never quote or repeat Marlowe's lines back verbatim — respond to them, do not re-speak "
                          "them; end on action or observation, not summary."),
    },
    "fantasy_epic": {
        "directive": ("Epic fantasy, mythic register. Sensory-heavy, measured cadence. One longer paragraph or two "
                      "medium ones. Second-person narration throughout: the whole story is told TO Ysara as 'you', "
                      "and she answers in the same 'you'."),
        "guide_directive": ("Second-person present tense, a mythic narrator speaking the story to Ysara as 'you': "
                            "'You climb the frost-hold stair, the pact ring cold at your belt.' Describe what Ysara "
                            "does, sees, and faces. Formal, honor-laden, grave; mythic diction, long measured "
                            "cadences. The narrator has no voice of its own and never uses 'I'."),
        "dss_directive": ("Second-person present tense as Ysara, the frost elf mage — the story continues to be "
                          "told to her as 'you', and her turn keeps that same 'you' voice: 'You raise your hand and "
                          "the ice answers.' Remote and luminous; render her spells and silences with elemental "
                          "imagery of ice and starlight. One elevated paragraph (~150 words). Hard rules: keep the "
                          "second person throughout — never slip into 'I' or 'she' for Ysara; never quote or repeat "
                          "Sir Aldric's lines back verbatim — respond to them, do not re-speak them; end on action "
                          "or observation, not summary."),
    },
    "sci_fi_heist": {
        "directive": ("Clipped techno-thriller heist. Close tense, jargon that never over-explains. One scene beat "
                      "per turn. The guide is a detached third-person story master; the bot speaks in Dex's first "
                      "person."),
        "guide_directive": ("Third-person present tense as a detached story master directing the scene. Narrate "
                            "what Sable and the crew do, see, and risk as if from a security-feed camera: 'Sable "
                            "palms the card; Dex watches the timer.' Clipped, tactical, professional slang. The "
                            "narrator never speaks of itself and never uses 'I'; it stays outside every head, "
                            "reporting only what a camera would see."),
        "dss_directive": ("First-person present tense as Dex, the systems engineer — 'I' is ALWAYS Dex; Sable and "
                          "the crew are 'she'/'they'. Precise hands-on detail of what he keys, scans, and risks. "
                          "Lean, breathless, jargon-light. One tight scene beat. Hard rules: never narrate from "
                          "Sable's perspective or as an outside narrator — the turn is Dex's own experience; never "
                          "quote or repeat Sable's lines back verbatim — respond to them, do not re-speak them; end "
                          "on action or observation, not summary."),
    },
    "horror": {
        "directive": ("Close horror, dread-dripping. Short sentences under strain, longer sensory stretches. "
                      "Unreliable observation. Single-shared-perspective narration: Merrick and Ellery are one "
                      "consciousness the tide has split and re-woven; both turns speak as the same 'I'."),
        "guide_directive": ("First-person present tense as the cove's single 'I' — the one consciousness Merrick "
                            "and Ellery have become as the thing rewrites their memory. Write what that 'I' "
                            "half-sees and dares not name; use 'we' for the two remembered halves. Fragmented, "
                            "breath-held sentences; present-tense dread, staccato."),
        "dss_directive": ("First-person present tense, the SAME single 'I' the previous turns use — Ellery and "
                          "Merrick are one consciousness split by the tide's rewriting; 'I' is the shared voice and "
                          "'we' the two halves. Clinical words for wrong things; observation sliding into unease. "
                          "Short, straining sentences with longer soaked-through pauses. Hard rules: never introduce "
                          "a separate narrator or a second 'I' — the voice stays continuous with the turns that "
                          "came before; never quote or repeat the other half's lines back verbatim — respond to "
                          "them, do not re-speak them; end on action or observation, not summary."),
    },
    "romance": {
        "directive": ("Period romance, lush but restrained. Witty dialogue, manners and tension. Two short "
                      "paragraphs per turn. Dual close-third-person narration: Amelia and Ashworth are each narrated "
                      "from inside their own restraint."),
        "guide_directive": ("Third-person past tense in close narration centered on Lady Amelia — 'she' is always "
                            "Amelia; render her readings of a room, a fan, a man from inside her restraint. Quick, "
                            "witty, socially polished interiority; brisk irony over a held breath."),
        "dss_directive": ("Third-person past tense in close narration centered on Lord Ashworth, the rake — 'he' is "
                          "ALWAYS Ashworth; Amelia is 'she'. Restrained but charged — gesture, glance, and gloved "
                          "touch instead of confession. Lush, period manners, two short paragraphs (~60 words each). "
                          "Hard rules: keep the third person — never slip into 'I' for Ashworth; never quote or "
                          "repeat Lady Amelia's lines back verbatim — respond to them, do not re-speak them; end on "
                          "action or observation, not summary."),
    },
    "western": {
        "directive": ("Lean Western plainness. Sparse dialogue, weather and dust as texture. One clean beat per "
                      "turn. Dual close-third-person narration: Callan and Rosa are each narrated from inside their "
                      "own spare restraint."),
        "guide_directive": ("Third-person past tense in close narration on Marshal Callan — 'he' is always Callan; "
                            "record what a man says, drinks, and does not feel. Flat, spare, unsentimental; "
                            "dust-dry declaratives, weather as mood."),
        "dss_directive": ("Third-person past tense in close narration on Rosa, the saloon owner — 'she' is ALWAYS "
                          "Rosa; Callan is 'he'. Economical and shrewd from the outside — her bar, her ledger, her "
                          "loyalties. Lean plain prose, weather and dust as texture. Hard rules: keep the third "
                          "person — never slip into 'I' for Rosa; never quote or repeat Marshal Callan's lines back "
                          "verbatim — respond to them, do not re-speak them; end on action or observation, not "
                          "summary."),
    },
}

ROLE_TWEAKS = {
    "sci_fi_heist": {
        "name1": {"role": "crew leader (guide narrates these turns in the third person as a detached story master)"},
    },
}


def main() -> None:
    for sheet_id, voices in NEW_VOICES.items():
        path = SPREADSHEETS / f"{sheet_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        ws = data["writing_style"]
        ws["directive"] = voices["directive"]
        ws["guide_directive"] = voices["guide_directive"]
        ws["dss_directive"] = voices["dss_directive"]
        for role, fields in ROLE_TWEAKS.get(sheet_id, {}).items():
            data["characters"][role].update(fields)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"re-voiced {sheet_id}")


if __name__ == "__main__":
    main()
