from __future__ import annotations
from typing import TYPE_CHECKING, Any

import jsonc
import logging
import re
import warnings
from pathlib import Path
import traceback
from os import PathLike

from ...utils.helpers import (
    _ERROR,
    _SUCCESS,
    _HILITE,
    _BOLD,
    _RESET,
    _WARNING,
)

from ...utils.background_importer import (
    start_background_import,
    get_imported_attribute,
)

if TYPE_CHECKING:
    import nltk
    import spacy
    from spacy.tokens import Doc
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.indices.loading import load_index_from_storage
    from llama_index.core.settings import Settings
    from llama_index.core.schema import TextNode
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from ...agents.summarizer import Summarizer
else:
    nltk = None
    spacy = None
    Doc = None
    HuggingFaceEmbedding = None
    VectorStoreIndex = None
    StorageContext = None
    SimpleNodeParser = None
    load_index_from_storage = None
    Settings = None
    TextNode = None

start_background_import("nltk")
start_background_import("spacy")
start_background_import("llama_index.core.settings", "Settings")
start_background_import("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")


from typing import Any


class MessageChunker:
    # Class-level singletons to avoid reloading heavy resources
    _embed_model = None
    _nlp = None
    _nltk_downloaded = False
    _spacy_model_downloaded = False
    _initialized = False
    _warning_suppressed = False

    @classmethod
    def _init_shared_resources(cls):
        """Initialize shared resources (embed model, spaCy, NLTK) only once."""
        if cls._initialized:
            return

        # Suppress MPNet warning
        if not cls._warning_suppressed:
            warnings.filterwarnings("ignore", message=".*position_ids.*")
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            cls._warning_suppressed = True

        # Get background imported modules
        if not TYPE_CHECKING:
            global nltk, spacy, HuggingFaceEmbedding, Settings
            if any((nltk is None, spacy is None, HuggingFaceEmbedding is None, Settings is None)):
                nltk = get_imported_attribute("nltk")
                spacy = get_imported_attribute("spacy")
                HuggingFaceEmbedding = get_imported_attribute("llama_index.embeddings.huggingface", "HuggingFaceEmbedding")
                Settings = get_imported_attribute("llama_index.core.settings", "Settings")

        # Download NLTK data and load model once
        if not cls._nltk_downloaded:
            nltk_data_path = Path("user_data/nltk_data")
            nltk.data.path.append(nltk_data_path.resolve())
            nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
            nltk.download("punkt_tab", download_dir=nltk_data_path, quiet=True)
            cls._nltk_downloaded = True

        if cls._embed_model is None:
            cls._embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-mpnet-base-v2"
                # model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            Settings.embed_model = cls._embed_model

        if cls._nlp is None:
            try:
                cls._nlp = spacy.load("en_core_web_sm")
            except OSError:
                print(f"{_BOLD}Downloading spaCy model...{_RESET}")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                cls._nlp = spacy.load("en_core_web_sm")

        cls._initialized = True

    def __init__(
        self,
        history_path: PathLike,
        characters_data: dict[str, Any],
        groups_data: dict[str, Any],
        elements_data: dict[str, Any],
        events_data: dict[str, Any],
        current_scene_data: dict[str, Any],
        summarizer: 'Summarizer' | None = None,
        use_llm_for_speakers: bool = True,
    ):
        print(f"{_BOLD}Initializing MessageChunker...{_RESET}")

        MessageChunker._init_shared_resources()

        # Use class-level shared resources
        self.nlp = MessageChunker._nlp
        self.summarizer = summarizer
        # TODO: Make configurable via UI toggle
        self.use_llm_for_speakers = use_llm_for_speakers

        self.history_path = Path(history_path)
        self.storage_dir = self.history_path / "message_index"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Store provided data
        self.characters_data = characters_data
        self.groups_data = groups_data
        self.elements_data = elements_data
        self.events_data = events_data
        self.current_scene_data = current_scene_data

        # Initialize or load existing index
        if not TYPE_CHECKING:
            StorageContext = get_imported_attribute("llama_index.core", "StorageContext")
            load_index_from_storage = get_imported_attribute("llama_index.core.indices.loading", "load_index_from_storage")
            VectorStoreIndex = get_imported_attribute("llama_index.core", "VectorStoreIndex")
        try:
            self.storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
            )
        except Exception:
            self.index = VectorStoreIndex([])
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))

        if not TYPE_CHECKING:
            SimpleNodeParser = get_imported_attribute("llama_index.core.node_parser", "SimpleNodeParser")
        self.parser = SimpleNodeParser.from_defaults()

        # Load character patterns for pronoun resolution
        self.pronoun_character_patterns = self._load_pronoun_character_patterns()

        # Create simpler name/alias patterns for direct entity matching
        self.character_name_patterns = self._create_name_alias_patterns(self.characters_data, main_name_key_is_dict_key=True)
        self.group_name_patterns = self._create_name_alias_patterns(self.groups_data, main_name_key_is_dict_key=True)
        self.element_name_patterns = self._create_name_alias_patterns(self.elements_data, main_name_key_is_dict_key=True)
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
        """Determine speakers from text using LLM (primary) with regex "Name:" as quick pre-filter."""
        speakers = set()
        doc = self.nlp(paragraph_text)

        # 1. Check for "Name: Dialogue" format line by line
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

        # Use LLM for speaker extraction if enabled (more accurate than regex/spaCy)
        # TODO: Make configurable via UI toggle
        if self.use_llm_for_speakers and self.summarizer:
            try:
                prompt = f'''Analyze the following text in context and identify the names of the character(s) who are speaking.

Respond with a JSON array of character names:
["Character1", "Character2", ...]

Text:
```
{paragraph_text}
```

Do not include generic terms like "you", "someone", "they". Only include characters that are explicitly or implicitly mentioned as speaking. Do not include characters who are only being addressed but not speaking.'''
                response_text, _ = self.summarizer.generate_with_sse(prompt, self.summarizer.last.custom_state, "determine_speakers", "speakers_llm", None)
                if response_text:
                    try:
                        llm_speakers = jsonc.loads(response_text.strip())
                        if isinstance(llm_speakers, list):
                            speakers.update(llm_speakers)
                    except jsonc.JSONDecodeError:
                        pass
            except Exception:
                pass

        if not speakers:
            # 2. Fall back to spaCy-based analysis for quoted speech and other dialogue indicators within sentences
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

    def chunk_message(self, message: str, message_idx: int, current_timestamp: str, do_determine_speakers: bool = True) -> list:
        """Split message into chunks at different granularities, enrich with metadata."""
        chunks = []

        # Determine characters present in the current scene once
        scene_active_characters = []
        if self.current_scene_data and isinstance(self.current_scene_data.get("now"), dict) and isinstance(self.current_scene_data["now"].get("who"), dict):
            characters = self.current_scene_data["now"]["who"].get("characters")
            if isinstance(characters, dict):
                for char_info in characters.values():
                    if isinstance(char_info, dict) and "name" in char_info:
                        scene_active_characters.append(char_info["name"])
            elif isinstance(characters, list):
                for char_info in characters:
                    if isinstance(char_info, dict) and "name" in char_info:
                        scene_active_characters.append(char_info["name"])

        # Split into paragraphs
        paragraphs = [p.strip() for p in message.split("\n\n") if p.strip()]

        for para_idx, paragraph in enumerate(paragraphs, start=1):
            # Split paragraph into sentences
            sentences = nltk.sent_tokenize(paragraph)

            if do_determine_speakers:
                paragraph_speakers = self._determine_speakers(paragraph)
            else:
                paragraph_speakers = []  # Unknown; caller should treat empty as unknown

            for sent_idx, sentence_text in enumerate(sentences, start=1):
                chunk_id = f"{message_idx}_{para_idx}_{sent_idx}"
                speakers = paragraph_speakers

                # Extract entities directly mentioned in the current sentence
                characters_mentioned_in_sentence = self._extract_entities(sentence_text, self.character_name_patterns)
                groups_referenced_in_sentence = self._extract_entities(sentence_text, self.group_name_patterns)
                elements_referenced_in_sentence = self._extract_entities(sentence_text, self.element_name_patterns)
                events_referenced_in_sentence = self._extract_entities(sentence_text, self.event_name_patterns)

                subjects_referenced = {
                    "characters": characters_mentioned_in_sentence,
                    "groups": groups_referenced_in_sentence,
                    "elements": elements_referenced_in_sentence,
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
                        "scene_number": self.current_scene_data.get("_scene_number"),
                        "chapter_number": self.current_scene_data.get("_chapter_number"),
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

    def update_node_metadata_by_message_idx(
        self, message_idx: int, metadata_updates: dict[str, Any], persist_dir: PathLike | None = None
    ):
        """Update metadata for all nodes associated with a message_idx."""
        if not TYPE_CHECKING:
            TextNode = get_imported_attribute("llama_index.core.schema", "TextNode")
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
            self.index.storage_context.persist(persist_dir=str(persist_dir or self.storage_dir))
            try:
                print(f"{_SUCCESS}Updated metadata for {len(nodes_to_update)} nodes for message_idx {message_idx}{_RESET}")
            except Exception as e:
                print(f"{_ERROR}Error during post-update operations for message_idx {message_idx}: {e}{_RESET}")
        else:
            print(f"{_HILITE}No nodes found for message_idx {message_idx} to update metadata.{_RESET}")

    def process_message(self, message: str, message_idx: int, current_timestamp: str, do_determine_speakers: bool = True) -> list:
        """Process and store a new message. Overwrites existing chunks if message_idx exists."""
        # Delete existing chunks for this message if any
        self.delete_message_chunks(message_idx)

        # Create and store new chunks
        chunks = self.chunk_message(message, message_idx, current_timestamp, do_determine_speakers=do_determine_speakers)
        self.store_chunks(chunks)
        return chunks

    def update_message_speakers(self, message_idx: int) -> bool:
        """Update speakers for existing chunks of a message using current state.

        Uses the current stored message text to re-determine speakers via LLM,
        then updates the metadata for all chunks with that message_idx.

        Args:
            message_idx: The message index to update speakers for.

        Returns:
            bool: True if update succeeded, False otherwise.
        """
        try:
            all_nodes = self.index.docstore.docs
            message_chunks = []
            for node_id, node in all_nodes.items():
                if node.metadata.get("message_idx") == message_idx:
                    message_chunks.append(node)

            if not message_chunks:
                print(f"{_WARNING}No chunks found for message_idx {message_idx} to update speakers.{_RESET}")
                return False

            message_chunks_sorted = sorted(message_chunks, key=lambda n: (
                n.metadata.get("paragraph_idx", 0),
                n.metadata.get("sentence_idx", 0),
            ))

            # Group chunks by paragraph to determine speakers per paragraph
            paragraph_groups = []
            current_para_idx = None
            current_para_chunks = []
            for chunk in message_chunks_sorted:
                para_idx = chunk.metadata.get("paragraph_idx", 0)
                if para_idx != current_para_idx and current_para_chunks:
                    paragraph_groups.append((current_para_idx, current_para_chunks))
                    current_para_chunks = []
                current_para_idx = para_idx
                current_para_chunks.append(chunk)
            if current_para_chunks:
                paragraph_groups.append((current_para_idx, current_para_chunks))

            if not paragraph_groups:
                print(f"{_WARNING}No text found in chunks for message_idx {message_idx}.{_RESET}")
                return False

            # Determine speakers per paragraph and update each paragraph's chunks
            for para_idx, para_chunks in paragraph_groups:
                para_sentences = [chunk.text for chunk in para_chunks if chunk.text]
                para_text = " ".join(para_sentences).strip()

                if para_text:
                    speakers = self._determine_speakers(para_text)

                    # Update metadata for all chunks in this paragraph
                    for chunk in para_chunks:
                        chunk.metadata["speakers"] = speakers

            # Persist metadata changes
            if not TYPE_CHECKING:
                TextNode = get_imported_attribute("llama_index.core.schema", "TextNode")
            nodes_to_update = []
            for chunk in message_chunks_sorted:
                updated_node = TextNode(
                    text=chunk.text,
                    id_=chunk.id_,
                    metadata=chunk.metadata,
                )
                nodes_to_update.append(updated_node)

            if nodes_to_update:
                self.index.insert_nodes(nodes_to_update)
                self.index.storage_context.persist(persist_dir=str(self.storage_dir))

            return True

        except Exception as e:
            print(f"{_ERROR}Error updating speakers for message_idx {message_idx}: {str(e)}{_RESET}")
            traceback.print_exc()
            return False

    def store_chunks(self, chunks: list, persist_dir: PathLike | None = None):
        """Store chunks using LlamaIndex."""

        if not TYPE_CHECKING:
            TextNode = get_imported_attribute("llama_index.core.schema", "TextNode")
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
                "scene_number": chunk.get("scene_number"),  # Will be None initially
                "chapter_number": chunk.get("chapter_number"),  # Will be None initially
                "event_id": chunk.get("event_id"),  # Will be None initially
                "is_summary": chunk.get("is_summary", False),
            }

            node = TextNode(text=chunk["text"], id_=chunk["id"], metadata=metadata)
            nodes.append(node)

        if nodes:  # Only insert if there are nodes to avoid errors with empty list
            self.index.insert_nodes(nodes)
            self.index.storage_context.persist(persist_dir=str(persist_dir or self.storage_dir))
            # try:
            #     import shutil

            #     shutil.copytree(
            #         self.storage_dir,
            #         self.history_path / "message_index",
            #         dirs_exist_ok=True,
            #     )
            # except Exception as e:
            #     print(f"{_ERROR}Error copying message_index after storing chunks: {e}{_RESET}")