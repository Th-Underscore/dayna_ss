import jsonc
import re
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from typing import TYPE_CHECKING

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.indices.loading import load_index_from_storage

# HuggingFaceEmbedding will be background imported
from llama_index.core.settings import Settings
from llama_index.core.schema import TextNode


if TYPE_CHECKING:
    import nltk
    import spacy
    from spacy.tokens import Doc
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
else:
    nltk = None
    spacy = None
    Doc = None
    HuggingFaceEmbedding = None

from extensions.dayna_ss.utils.helpers import (
    _ERROR,
    _SUCCESS,
    _INPUT,
    _GRAY,
    _HILITE,
    _BOLD,
    _RESET,
    _DEBUG,
)

from extensions.dayna_ss.utils.background_importer import (
    start_background_import,
    get_imported_attribute,
)

start_background_import("nltk")
start_background_import("spacy")
start_background_import("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")


@dataclass
class RetrievalContext:
    context: str = ""
    current_scene: dict[str, dict] = field(default_factory=dict)
    characters: dict[str, dict] = field(default_factory=dict)
    groups: dict[str, dict] = field(default_factory=dict)
    events: dict[str, dict] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)


class StoryContextRetriever:
    def __init__(self, history_path: str | Path):
        """Initialize the context retriever with a history path."""
        history_path = Path(history_path)
        if not history_path.exists():
            raise ValueError(f"History path does not exist: {history_path}")
        self.history_path = history_path

        # Load static data
        self.characters_path = history_path / "characters.json"
        self.events_path = history_path / "events.json"
        self.groups_path = history_path / "groups.json"
        self.current_scene_path = history_path / "current_scene.json"

        self.characters = self._load_json(self.characters_path)
        self.groups = self._load_json(self.groups_path)
        self.events = self._load_json(self.events_path)
        self.current_scene = self._load_json(self.current_scene_path)

        self.chunker = MessageChunker(history_path, self.characters, self.groups, self.events, self.current_scene)

        # Create character name patterns for recognition
        self.character_patterns = self._create_character_patterns()

    def _create_character_patterns(self) -> dict[str, re.Pattern]:
        """Create regex patterns for character name recognition."""
        patterns = {}
        for char_name in self.characters:
            # Create pattern that matches full name and possible first/last name only
            names = char_name.split()
            pattern = f"({char_name}"
            if len(names) > 1:
                pattern += f"|{names[0]}|{names[-1]}"
            pattern += ")"
            patterns[char_name] = re.compile(pattern, flags=re.IGNORECASE)
        return patterns

    def _extract_character_names(self, text: str) -> list[str]:
        """Extract character names from text using regex patterns."""
        found_names = []
        for char_name, pattern in self.character_patterns.items():
            if pattern.search(text):
                if char_name not in found_names:
                    found_names.append(char_name)
        return found_names

    def _get_relevant_groups(self, characters: list[str], context: str) -> dict[str, dict]:
        """Get groups relevant to the current context and characters."""
        relevant_groups = {}

        for group_name, group_data in self.groups.items():
            # Check if group is mentioned in context
            if re.search(group_name, context, flags=re.IGNORECASE) or any(
                alias for alias in group_data.get("aliases", []) if re.search(alias, context, flags=re.IGNORECASE)
            ):
                relevant_groups[group_name] = group_data
                continue

            # Check if any character is in this group
            for char in characters:
                if char in group_data.get("characters", {}):
                    relevant_groups[group_name] = group_data
                    break

        return relevant_groups

    def _get_relevant_events(self, characters: list[str], groups: dict[str, dict], context: str) -> dict[str, dict]:
        """Get events relevant to the current context and groups."""
        relevant_events = {}

        if "scenes" in self.events:
            for scene in self.events["scenes"]:
                scene_name = scene.get("name", "")
                # Check if event is mentioned in context
                if re.search(scene_name, context, flags=re.IGNORECASE):
                    relevant_events[scene_name] = scene
                    continue

                # Check if event is associated with any relevant group
                for group_data in groups.values():
                    if scene_name in group_data.get("events", []):
                        relevant_events[scene_name] = scene
                        break

        return relevant_events

    def _get_message_chunks(self, scene_name: str = None) -> list[str]:
        """Retrieve relevant message chunks based on scene or context."""
        messages = []

        # Get message indices from current scene
        current_scene = self.get_current_scene()
        if current_scene and "messages" in current_scene:
            for msg_range in current_scene["messages"]:
                start, end = msg_range
                # Adjust indices for 1-based indexing in query
                query = f"message_idx:[{start + 1} TO {end + 1}]"
                results = self.chunker.query_similar(query)
                if results and results["documents"]:
                    messages.extend(results["documents"])

        return messages

    def query_messages(self, query: str, n_results: int = 5) -> list[str]:
        """Query messages using semantic search."""
        results = self.chunker.query_similar(query, n_results=n_results)
        return results["documents"] if results else []

    def _load_json(self, path: Path) -> dict:
        """Load and parse a JSON file."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return jsonc.load(f)
            except Exception as e:
                print(f"{_ERROR}Failed to load JSON file {path}: {e}{_RESET}")
                traceback.print_exc()
        return {}

    def get_current_scene(self) -> dict:
        """Get the current scene data."""
        return self._load_json(self.current_scene_path)

    def _get_character_important_relationships(self, char_name: str, importance_threshold: int = 75) -> dict[str, list[dict]]:
        """Get a character's important relationships."""
        if char_name not in self.characters:
            return {}

        char_data = self.characters[char_name]
        rels = {}

        if "relationships" in char_data:
            print(f"{_GRAY}relationships{_RESET}: {char_data['relationships']}")
            for related_char, rel_list in char_data["relationships"].items():
                important_rels = [rel for rel in rel_list if rel.get("importance", 0) >= importance_threshold]
                if important_rels:
                    rels[related_char] = important_rels

        return rels

    def _get_character_scene_relationships(
        self, char1: str, char2: str, correlation_threshold: int = 0
    ) -> dict[str, list[dict]]:
        """Get relationships between two characters in the same scene, regardless of importance."""
        if char1 not in self.characters or char2 not in self.characters:
            return {}

        char_data = self.characters[char1]
        rels = {}

        if "relationships" in char_data and char2 in char_data["relationships"]:
            rel_list = char_data["relationships"][char2]
            scene_rels = [rel for rel in rel_list if rel.get("importance", 0) >= correlation_threshold]
            if scene_rels:
                rels[char2] = scene_rels

        return rels

    def _get_all_relevant_character_relationships(
        self,
        scene_characters: list[str],
        importance_threshold: int = 75,
        correlation_threshold: int = 0,
    ) -> dict[str, dict]:
        """Get all relevant relationships for characters, including both important and scene-based relationships."""
        result = {}
        processed_chars = []

        # First pass: Get important relationships for all characters
        for char_name in scene_characters:
            if char_name in self.characters:
                char_data = self.characters[char_name].copy()  # Copy to avoid modifying original

                # Get important relationships
                important_rels = self._get_character_important_relationships(char_name, importance_threshold)

                # Add character and their important relationships
                if not char_data.get("relationships"):
                    char_data["relationships"] = {}
                char_data["relationships"].update(important_rels)
                result[char_name] = char_data

                # Add related characters to be processed
                for related_char in important_rels:
                    if related_char in self.characters and related_char not in scene_characters:
                        scene_characters.append(related_char)

        # Second pass: Get scene-based relationships between characters
        for char1 in scene_characters:
            for char2 in scene_characters:
                if char1 != char2 and char1 in result:
                    # Get relationships between scene characters
                    scene_rels = self._get_character_scene_relationships(char1, char2, correlation_threshold)

                    # Add scene relationships if they exist and aren't already included
                    if scene_rels and char2 not in result[char1].get("relationships", {}):
                        if "relationships" not in result[char1]:
                            result[char1]["relationships"] = {}
                        result[char1]["relationships"].update(scene_rels)

        return result

    def retrieve_context(self, current_context: str, last_x_messages: list[str]) -> RetrievalContext:
        """Main method to retrieve all relevant context based on current state."""
        result = RetrievalContext()
        current_scene = self.get_current_scene()

        # Get characters from current scene and context
        scene_characters = []
        if current_scene and "who" in current_scene.get("now", {}):
            for char in current_scene["now"]["who"].get("characters", []):
                print(char["name"], "-", scene_characters)
                if char["name"] not in scene_characters:
                    scene_characters.append(char["name"])

        # Add characters mentioned in context and last messages
        context_to_search = current_context + "\n" + "\n".join(last_x_messages)
        mentioned_characters = self._extract_character_names(context_to_search)
        for char in mentioned_characters:
            if char not in scene_characters:
                scene_characters.append(char)

        try:
            # Get character relationships and related data
            result.characters = self._get_all_relevant_character_relationships(scene_characters)
            result.groups = self._get_relevant_groups(scene_characters, context_to_search)
            result.events = self._get_relevant_events(scene_characters, result.groups, context_to_search)

            # Get messages using both retrieval methods
            # scene_messages = self._get_message_chunks()  # Index-based retrieval
            semantic_messages = self.query_messages(context_to_search, n_results=5)  # Semantic search

            # Combine and deduplicate messages
            all_messages = []

            # # First add scene messages to maintain chronological order
            # for msg in scene_messages:
            #     if msg not in all_messages:
            #         all_messages.append(msg)

            # Then add semantically relevant messages
            for msg in semantic_messages:
                if msg not in all_messages:
                    all_messages.append(msg)

            result.messages = all_messages

        except Exception as e:
            print(f"Error in context retrieval: {str(e)}")
            traceback.print_exc()

        return result


from typing import Any


class MessageChunker:
    def __init__(
        self,
        history_path: str | Path,
        characters_data: dict[str, Any],
        groups_data: dict[str, Any],
        events_data: dict[str, Any],
        current_scene_data: dict[str, Any],
    ):
        print(f"{_BOLD}Initializing MessageChunker...{_RESET}")

        # Get background imported modules
        if not TYPE_CHECKING:
            global nltk, spacy, HuggingFaceEmbedding
            if any((nltk is None, spacy is None, HuggingFaceEmbedding is None)):
                nltk = get_imported_attribute("nltk")
                spacy = get_imported_attribute("spacy")
                HuggingFaceEmbedding = get_imported_attribute("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")
        # Other LlamaIndex components (VectorStoreIndex, StorageContext, SimpleNodeParser, etc.)
        # are imported synchronously and used directly due to errors with circular imports

        nltk.download("punkt", download_dir=Path("user_data/nltk_data"), quiet=True)
        nltk.download("punkt_tab", download_dir=Path("user_data/nltk_data"), quiet=True)

        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2"
            # model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = embed_model

        self.history_path = Path(history_path)
        self.storage_dir = self.history_path.parent / "message_index"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Store provided data
        self.characters_data = characters_data
        self.groups_data = groups_data
        self.events_data = events_data
        self.current_scene_data = current_scene_data

        # Initialize or load existing index
        try:
            self.storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
            )
        except Exception:
            self.index = VectorStoreIndex([])
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))

        self.parser = SimpleNodeParser.from_defaults()

        # Load spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(f"{_BOLD}Downloading spaCy model...{_RESET}")
            # Consider platform compatibility and direct subprocess call for better control
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load("en_core_web_sm")

        # Load character patterns for pronoun resolution
        self.pronoun_character_patterns = self._load_pronoun_character_patterns()

        # Create simpler name/alias patterns for direct entity matching
        self.character_name_patterns = self._create_name_alias_patterns(self.characters_data, main_name_key_is_dict_key=True)
        self.group_name_patterns = self._create_name_alias_patterns(self.groups_data, main_name_key_is_dict_key=True)
        self.event_name_patterns = self._create_event_name_patterns(self.events_data)

    DIALOGUE_VERBS = {
        "say",
        "tell",
        "ask",
        "reply",
        "shout",
        "whisper",
        "exclaim",
        "mutter",
        "state",
        "declare",
        "respond",
        "add",
        "continue",
        "begin",
        "murmur",
        "interject",
        "question",
        "answer",
        "stammer",
        "insist",
        "suggest",
        "warn",
    }

    def _create_name_alias_patterns(
        self,
        entity_data: dict[str, dict[str, Any]],
        main_name_key_is_dict_key: bool = True,
    ) -> dict[str, re.Pattern]:
        """Creates regex patterns for entity names and their aliases."""
        patterns = {}
        if not entity_data:
            return patterns
        for main_name, data in entity_data.items():
            names_to_match = [main_name]
            if isinstance(data, dict) and "aliases" in data:
                aliases = data.get("aliases", [])
                if isinstance(aliases, list):
                    names_to_match.extend(aliases)

            # Filter out empty strings and ensure uniqueness
            unique_names = sorted(list(set(filter(None, names_to_match))), key=len, reverse=True)
            if unique_names:
                # Pattern to match whole words, case-insensitive
                pattern_str = r"\b(" + "|".join(re.escape(name) for name in unique_names) + r")\b"
                patterns[main_name] = re.compile(pattern_str, flags=re.IGNORECASE)
        return patterns

    def _create_event_name_patterns(self, events_data: dict[str, list[dict[str, Any]]]) -> dict[str, re.Pattern]:
        """Creates regex patterns for event names."""
        patterns = {}
        if not events_data:
            return patterns

        event_names = []
        for event_list_key in [
            "past",
            "scenes",
            "events",
        ]:  # Iterate through different event categories
            for event_item in events_data.get(event_list_key, []):
                if isinstance(event_item, dict) and "name" in event_item:
                    event_names.append(event_item["name"])

        unique_event_names = sorted(list(set(filter(None, event_names))), key=len, reverse=True)
        if unique_event_names:
            for name in unique_event_names:  # Create a pattern for each unique event name
                # Pattern to match whole words, case-insensitive
                pattern_str = r"\b(" + re.escape(name) + r")\b"
                patterns[name] = re.compile(pattern_str, flags=re.IGNORECASE)
        return patterns

    def _extract_entities(self, text: str, entity_patterns: dict[str, re.Pattern]) -> list[str]:
        """Extract unique entity names from text using provided patterns."""
        found_entities = set()
        for entity_name, pattern in entity_patterns.items():
            if pattern.search(text):
                found_entities.add(entity_name)
        return list(found_entities)

    def _determine_speakers(self, paragraph_text: str) -> list[str]:
        """Determine speakers from text using regex for 'Name: Dialogue' and spaCy for quoted speech."""
        speakers = set()
        doc = self.nlp(paragraph_text)

        # 1. Check for "Name: Dialogue" format line by line (existing refined logic)
        lines = paragraph_text.split("\n")
        char_patterns_for_speakers = {
            name: pattern for name, pattern in self.character_name_patterns.items() if isinstance(pattern, re.Pattern)
        }
        group_patterns_for_speakers = {
            name: pattern for name, pattern in self.group_name_patterns.items() if isinstance(pattern, re.Pattern)
        }
        # Primarily, characters are speakers. Groups might be if they have a collective voice represented.
        speaker_name_patterns = {
            **char_patterns_for_speakers,
            **group_patterns_for_speakers,
        }

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            for name, pattern_obj in speaker_name_patterns.items():
                match = pattern_obj.match(stripped_line)
                if match and match.start() == 0:  # Pattern matches at the beginning of the line
                    # Check if the character(s) immediately following the match is a colon
                    if stripped_line[match.end() :].strip().startswith(":"):
                        speakers.add(name)
                        break  # Found speaker for this line by "Name:" pattern

        # 2. spaCy-based analysis for quoted speech and other dialogue indicators within sentences
        for sent in doc.sents:
            # Basic check for quotes. More sophisticated quote detection might be needed for complex cases.
            has_quote = (
                '"' in sent.text
                or "'" in sent.text
                or "“" in sent.text
                or "”" in sent.text
                or "‘" in sent.text
                or "’" in sent.text
            )

            for token in sent:
                # Check for dialogue verbs
                if token.lemma_.lower() in self.DIALOGUE_VERBS and token.pos_ == "VERB":
                    # Find subject of the verb (potential speaker)
                    subject_token = None
                    for child in token.children:
                        if child.dep_ == "nsubj":
                            subject_token = child
                            break

                    if subject_token:
                        # Extract text of the subject (could be a single name or a phrase)
                        # We can check the subject token itself or its subtree for more complex subjects.
                        subject_text = subject_token.text
                        potential_speakers_from_subject = self._extract_entities(subject_text, self.character_name_patterns)
                        for speaker_name in potential_speakers_from_subject:
                            if has_quote:
                                speakers.add(speaker_name)

                    # Additionally, check for character names directly preceding/following quotes if not caught by subject-verb
                    # This part can be expanded with more rules.
                    # For example, if token is a quote, check previous/next tokens for names.

            if has_quote and not speakers.intersection(self._extract_entities(sent.text, self.character_name_patterns)):
                chars_in_sentence_with_quote = self._extract_entities(sent.text, self.character_name_patterns)
                for char_name in chars_in_sentence_with_quote:
                    # A more robust check would analyze proximity to quote marks.
                    speakers.add(char_name)  # This might over-generate, needs refinement or context.

        return list(speakers)

    def _load_pronoun_character_patterns(self) -> dict[str, dict]:
        """Load character patterns for pronoun resolution from self.characters_data."""
        if not self.characters_data:
            return {}

        patterns = {}
        for char_name, char_data in self.characters_data.items():
            names = [char_name]
            if isinstance(char_data, dict) and "aliases" in char_data:
                aliases = char_data.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(aliases)

            sex = char_data.get("sex") if isinstance(char_data, dict) else None
            pronouns = []
            if sex == "male":
                pronouns = ["he", "him", "his", "himself"]
            elif sex == "female":
                pronouns = ["she", "her", "hers", "herself"]
            else:
                pronouns = ["they", "them", "their", "theirs", "themself", "themselves"]
            patterns[char_name] = {
                "names": list(set(filter(None, names))),  # Ensure unique and non-empty
                "pronouns": pronouns,
            }
        return patterns

    def _detect_character_references(self, text: str, doc: "Doc" = None) -> list[tuple[str, list[str]]]:
        """Detect character references in text, including pronouns."""
        if doc is None:
            doc = self.nlp(text)

        references = []

        # Track the last mentioned character for pronoun resolution
        last_character = None
        possible_characters = set()

        for token in doc:
            # Direct name matches
            matched_char = None
            for char_name, char_data in self.pronoun_character_patterns.items():
                if any(name.lower() in token.text.lower() for name in char_data["names"]):
                    matched_char = char_name
                    last_character = char_name
                    possible_characters = {char_name}
                    break

            # Pronoun handling
            if token.pos_ == "PRON" or (token.pos_ == "DET" and token.dep_ == "poss"):  # Include possessive determiners
                pron = token.text.lower()
                matching_chars = []

                # If we have a recent character mention and the pronoun matches
                if last_character:
                    char_data = self.pronoun_character_patterns.get(last_character, {})
                    if pron in char_data.get("pronouns", []):
                        matching_chars = [last_character]

                # If no match with recent character, find all possible matches
                if not matching_chars:
                    for char_name, char_data in self.pronoun_character_patterns.items():
                        if pron in char_data["pronouns"]:
                            matching_chars.append(char_name)

                if matching_chars:
                    # For reflexive pronouns (himself/herself/themselves), strongly prefer the last character
                    if pron.endswith("self") and last_character in matching_chars:
                        matching_chars = [last_character]

                    # Update possible characters for this reference
                    if len(matching_chars) == 1:
                        possible_characters = {matching_chars[0]}
                        last_character = matching_chars[0]
                    else:
                        possible_characters.update(matching_chars)

                    references.append((token.text, list(possible_characters)))

            elif matched_char:
                references.append((token.text, [matched_char]))

        return references

    def _tag_character_references(self, text: str) -> str:
        """Tag character references in text with possible character names."""
        doc = self.nlp(text)
        references = self._detect_character_references(text, doc)

        # Sort references by position (longest matches first to avoid nested replacements)
        references.sort(key=lambda x: len(x[0]), reverse=True)

        # Replace references with tagged versions
        tagged_text = text
        for ref_text, possible_chars in references:
            if len(possible_chars) == 1:
                replacement = f"{ref_text} [{possible_chars[0]}]"
            elif len(possible_chars) > 1:
                chars_str = "/".join(possible_chars)
                replacement = f"{ref_text} [{chars_str}]"
            tagged_text = tagged_text.replace(ref_text, replacement)

        return tagged_text

    def chunk_message(self, message: str, message_idx: int, current_timestamp: str) -> list:
        """Split message into chunks at different granularities, enrich with metadata."""
        chunks = []

        # Determine characters present in the current scene once
        scene_active_characters = []
        if (
            self.current_scene_data
            and isinstance(self.current_scene_data.get("now"), dict)
            and isinstance(self.current_scene_data["now"].get("who"), dict)
            and isinstance(self.current_scene_data["now"]["who"].get("characters"), dict)
        ):
            for char_info in self.current_scene_data["now"]["who"]["characters"].values():
                if isinstance(char_info, dict) and "name" in char_info:
                    scene_active_characters.append(char_info["name"])

        # Split into paragraphs
        paragraphs = [p.strip() for p in message.split("\n\n") if p.strip()]

        for para_idx, paragraph in enumerate(paragraphs, start=1):
            # Split paragraph into sentences
            sentences = nltk.sent_tokenize(paragraph)

            for sent_idx, sentence_text in enumerate(sentences, start=1):
                chunk_id = f"{message_idx}_{para_idx}_{sent_idx}"

                # Speaker detection (current logic, to be reviewed later)
                speakers = self._determine_speakers(paragraph)

                # Extract entities directly mentioned in the current sentence
                characters_mentioned_in_sentence = self._extract_entities(sentence_text, self.character_name_patterns)
                groups_referenced_in_sentence = self._extract_entities(sentence_text, self.group_name_patterns)
                events_referenced_in_sentence = self._extract_entities(sentence_text, self.event_name_patterns)

                subjects_referenced = {
                    "characters": characters_mentioned_in_sentence,
                    "groups": groups_referenced_in_sentence,
                    "events": events_referenced_in_sentence,
                }

                chunks.append(
                    {
                        "id": chunk_id,
                        "text": sentence_text,
                        "indices": [message_idx, para_idx, sent_idx],
                        "timestamp": current_timestamp,
                        "speakers": speakers,
                        "characters_present": scene_active_characters,
                        "subjects_referenced": subjects_referenced,
                        "scene_id": None,  # To be filled later
                        "event_id": None,  # To be filled later
                    }
                )

        return chunks

    def query_similar(self, query: str, n_results: int = 5):
        """Query similar chunks using LlamaIndex."""
        retriever = self.index.as_retriever(similarity_top_k=n_results)
        nodes = retriever.retrieve(query)

        ids, documents, metadatas, distances = [], [], [], []
        for node in nodes:
            ids.append(node.node.id_)
            documents.append(node.node.text)
            metadatas.append(node.node.metadata)
            distances.append(node.score)
        results = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }
        return results

    def delete_message_chunks(self, message_idx: int):
        """Delete all chunks for a given message index."""
        # Get all nodes
        all_nodes = self.index.docstore.docs

        # Find nodes to delete
        nodes_to_delete = []
        for node_id, node in all_nodes.items():
            if node.metadata["message_idx"] == message_idx:
                nodes_to_delete.append(node_id)

        # Delete nodes
        for node_id in nodes_to_delete:
            del self.index.docstore.docs[node_id]

        # Persist changes
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        import shutil

        shutil.copytree(self.storage_dir, self.history_path / "message_index", dirs_exist_ok=True)

    def update_node_metadata_by_message_idx(self, message_idx: int, metadata_updates: dict[str, Any]):
        """Update metadata for all nodes associated with a message_idx."""
        nodes_to_update = []
        # node_ids_to_delete_for_update = [] # Not strictly needed if insert_nodes handles updates by ID

        for node_id, node in self.index.docstore.docs.items():
            if node.metadata.get("message_idx") == message_idx:
                new_metadata = node.metadata.copy()
                new_metadata.update(metadata_updates)

                updated_node = TextNode(
                    text=node.text,
                    id_=node.id_,
                    metadata=new_metadata,
                    # relationships=node.relationships # Preserve relationships if any
                )
                nodes_to_update.append(updated_node)
                # node_ids_to_delete_for_update.append(node_id)

        if nodes_to_update:
            self.index.insert_nodes(nodes_to_update)
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            try:
                print(f"{_SUCCESS}Updated metadata for {len(nodes_to_update)} nodes for message_idx {message_idx}{_RESET}")
            except Exception as e:
                print(f"{_ERROR}Error during post-update operations for message_idx {message_idx}: {e}{_RESET}")
        else:
            print(f"{_HILITE}No nodes found for message_idx {message_idx} to update metadata.{_RESET}")

    def process_message(self, message: str, message_idx: int, current_timestamp: str):
        """Process and store a new message. Overwrites existing chunks if message_idx exists."""
        # Delete existing chunks for this message if any
        self.delete_message_chunks(message_idx)

        # Create and store new chunks
        chunks = self.chunk_message(message, message_idx, current_timestamp)
        self.store_chunks(chunks)
        return chunks

    def store_chunks(self, chunks: list):
        """Store chunks using LlamaIndex."""

        nodes = []
        for chunk in chunks:
            metadata = {
                "message_idx": chunk["indices"][0],
                "paragraph_idx": chunk["indices"][1],
                "sentence_idx": chunk["indices"][2],
                "timestamp": chunk.get("timestamp"),
                "speakers": chunk.get("speakers", []),
                "characters_present": chunk.get("characters_present", []),
                "subjects_referenced": chunk.get("subjects_referenced", {}),
                "scene_id": chunk.get("scene_id"),  # Will be None initially
                "event_id": chunk.get("event_id"),  # Will be None initially
                "is_summary": chunk.get("is_summary", False),
            }

            node = TextNode(text=chunk["text"], id_=chunk["id"], metadata=metadata)
            nodes.append(node)

        if nodes:  # Only insert if there are nodes to avoid errors with empty list
            self.index.insert_nodes(nodes)
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            try:
                import shutil

                shutil.copytree(
                    self.storage_dir,
                    self.history_path / "message_index",
                    dirs_exist_ok=True,
                )
            except Exception as e:
                print(f"{_ERROR}Error copying message_index after storing chunks: {e}{_RESET}")
