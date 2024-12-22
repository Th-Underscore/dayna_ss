import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import traceback
import nltk
from spacy.language import Language
from spacy.tokens import Doc
import spacy

# Color codes
_ERROR = "\033[1;31m"
_SUCCESS = "\033[1;32m"
_INPUT = "\033[0;33m"
_GRAY = "\033[0;30m"
_BOLD = "\033[1;37m"
_RESET = "\033[0m"

@dataclass
class RetrievalContext:
    context: str = ""
    characters: Dict[str, Dict] = field(default_factory=dict)
    groups: Dict[str, Dict] = field(default_factory=dict)
    events: Dict[str, Dict] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)

class StoryContextRetriever:
    def __init__(self, history_path: str | Path):
        """Initialize the context retriever with a history path."""
        history_path = Path(history_path)
        if not history_path.exists():
            raise ValueError(f"History path does not exist: {history_path}")
        self.history_path = history_path
        self.chunker = MessageChunker(history_path)
        
        # Load static data
        self.characters_path = history_path / "characters.json"
        self.events_path = history_path / "events.json"
        self.groups_path = history_path / "groups.json"
        self.current_scene_path = history_path / "current_scene.json"
        
        self.characters = self._load_json(self.characters_path)
        self.events = self._load_json(self.events_path)
        self.groups = self._load_json(self.groups_path)
        
        # Create character name patterns for recognition
        self.character_patterns = self._create_character_patterns()
        
    def _create_character_patterns(self) -> Dict[str, re.Pattern]:
        """Create regex patterns for character name recognition."""
        patterns = {}
        for char_name in self.characters:
            # Create pattern that matches full name and possible first/last name only
            names = char_name.split()
            pattern = f"(?i)({char_name}"
            if len(names) > 1:
                pattern += f"|{names[0]}|{names[-1]}"
            pattern += ")"
            patterns[char_name] = re.compile(pattern)
        return patterns

    def _extract_character_names(self, text: str) -> List[str]:
        """Extract character names from text using regex patterns."""
        found_names = []
        for char_name, pattern in self.character_patterns.items():
            if pattern.search(text):
                if char_name not in found_names:
                    found_names.append(char_name)
        return found_names

    def _get_relevant_groups(self, characters: List[str], context: str) -> Dict[str, Dict]:
        """Get groups relevant to the current context and characters."""
        relevant_groups = {}
        
        for group_name, group_data in self.groups.items():
            # Check if group is mentioned in context
            if re.search(f"(?i){group_name}", context) or any(alias for alias in group_data.get("aliases", []) if re.search(f"(?i){alias}", context)):
                relevant_groups[group_name] = group_data
                continue
                
            # Check if any character is in this group
            for char in characters:
                if char in group_data.get("characters", {}):
                    relevant_groups[group_name] = group_data
                    break
        
        return relevant_groups

    def _get_relevant_events(self, characters: List[str], groups: Dict[str, Dict], context: str) -> Dict[str, Dict]:
        """Get events relevant to the current context and groups."""
        relevant_events = {}
        
        if "scenes" in self.events:
            for scene in self.events["scenes"]:
                scene_name = scene.get("name", "")
                # Check if event is mentioned in context
                if re.search(f"(?i){scene_name}", context):
                    relevant_events[scene_name] = scene
                    continue
                
                # Check if event is associated with any relevant group
                for group_data in groups.values():
                    if scene_name in group_data.get("events", []):
                        relevant_events[scene_name] = scene
                        break
        
        return relevant_events

    def _get_message_chunks(self, scene_name: str = None) -> List[str]:
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

    def query_messages(self, query: str, n_results: int = 5) -> List[str]:
        """Query messages using semantic search."""
        results = self.chunker.query_similar(query, n_results=n_results)
        return results["documents"] if results else []

    def _load_json(self, path: Path) -> Dict:
        """Load and parse a JSON file."""
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"{_ERROR}Failed to load JSON file {path}: {e}{_RESET}")
                traceback.print_exc()
        return {}

    def get_current_scene(self) -> Dict:
        """Get the current scene data."""
        return self._load_json(self.current_scene_path)

    def _get_character_important_relationships(self, char_name: str, importance_threshold: int = 75) -> Dict[str, List[Dict]]:
        """Get important relationships for a character."""
        if char_name not in self.characters:
            return {}
            
        char_data = self.characters[char_name]
        rels = {}
        
        if "relationships" in char_data:
            for related_char, rel_list in char_data["relationships"].items():
                important_rels = [rel for rel in rel_list if rel.get("importance", 0) >= importance_threshold]
                if important_rels:
                    rels[related_char] = important_rels
        
        return rels

    def _get_character_scene_relationships(self, char1: str, char2: str, correlation_threshold: int = 0) -> Dict[str, List[Dict]]:
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

    def _get_all_relevant_character_relationships(self, scene_characters: List[str], importance_threshold: int = 75, correlation_threshold: int = 0) -> Dict[str, Dict]:
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

    def retrieve_context(self, current_context: str, last_x_messages: List[str]) -> RetrievalContext:
        """Main method to retrieve all relevant context based on current state."""
        result = RetrievalContext()
        current_scene = self.get_current_scene()
        
        # Get characters from current scene and context
        scene_characters = []
        if current_scene and "who" in current_scene.get("now", {}):
            for char in current_scene["now"]["who"].get("characters", []):
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
            scene_messages = self._get_message_chunks()  # Index-based retrieval
            semantic_messages = self.query_messages(context_to_search, n_results=5)  # Semantic search
            
            # Combine and deduplicate messages
            all_messages = []
            seen = set()
            
            # First add scene messages to maintain chronological order
            for msg in scene_messages:
                if msg not in seen:
                    all_messages.append(msg)
                    seen.add(msg)
            
            # Then add semantically relevant messages
            for msg in semantic_messages:
                if msg not in seen:
                    all_messages.append(msg)
                    seen.add(msg)
            
            result.messages = all_messages
            
        except Exception as e:
            print(f"Error in context retrieval: {str(e)}")
            traceback.print_exc()
        
        return result

    def update_retrieval_context(self, context: RetrievalContext):
        """Update the retrieval context with formatted information."""
        # TODO: Implement context formatting and updates
        pass

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

class MessageChunker:
    def __init__(self, history_path: str | Path):
        print(f"{_BOLD}Initializing MessageChunker...")
        nltk.download('punkt', download_dir=Path("installer_files/nltk_data"))
        nltk.download('punkt_tab', download_dir=Path("installer_files/nltk_data"))

        self.history_path = Path(history_path)
        self.storage_dir = self.history_path / "message_index"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up embedding model
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        Settings.embed_model = embed_model
        
        # Initialize or load existing index
        try:
            self.storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            self.index = load_index_from_storage(
                storage_context=self.storage_context,
            )
        except:
            self.index = VectorStoreIndex([])
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        
        self.parser = SimpleNodeParser.from_defaults()
        
        # Load spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(f"{_BOLD}Downloading spaCy model...{_RESET}")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # Load character patterns
        self.character_patterns = self._load_character_patterns()
        
    def _load_character_patterns(self) -> Dict[str, Dict]:
        """Load character patterns from characters.json"""
        char_file = self.history_path / "characters.json"
        if not char_file.exists():
            return {}
            
        with open(char_file, 'r', encoding='utf-8') as f:
            characters = json.load(f)
            
        patterns = {}
        for char_name, char_data in characters.items():
            # Create pattern for full name and aliases
            names = [char_name] + char_data.get("aliases", [])
            # Pattern for pronouns
            sex = char_data.get("sex")
            if sex == "male":
                pronouns = ["he", "him", "his", "himself"]
            elif sex == "female":
                pronouns = ["she", "her", "hers", "herself"]
            else:
                pronouns = ["they", "them", "their", "theirs", "theirself"]
            patterns[char_name] = {
                "names": names,
                "pronouns": pronouns,
            }
        return patterns
        
    def _detect_character_references(self, text: str, doc: Doc = None) -> List[Tuple[str, List[str]]]:
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
            for char_name, char_data in self.character_patterns.items():
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
                    char_data = self.character_patterns.get(last_character, {})
                    if pron in char_data.get("pronouns", []):
                        matching_chars = [last_character]
                
                # If no match with recent character, find all possible matches
                if not matching_chars:
                    for char_name, char_data in self.character_patterns.items():
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

    def chunk_message(self, message: str, message_idx: int) -> list:
        """Split message into chunks at different granularities."""
        chunks = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in message.split('\n\n') if p.strip()]
        
        for para_idx, paragraph in enumerate(paragraphs, start=1):
            # Tag character references in the paragraph
            tagged_paragraph = self._tag_character_references(paragraph)
            
            # Split paragraph into sentences
            sentences = nltk.sent_tokenize(tagged_paragraph)
            
            for sent_idx, sentence in enumerate(sentences, start=1):
                chunk_id = f"{message_idx}_{para_idx}_{sent_idx}"
                chunks.append({
                    "id": chunk_id,
                    "text": sentence,
                    "indices": [message_idx, para_idx, sent_idx]
                })
                
        return chunks
    
    def query_similar(self, query: str, n_results: int = 5):
        """Query similar chunks using LlamaIndex."""
        retriever = self.index.as_retriever(similarity_top_k=n_results)
        nodes = retriever.retrieve(query)
        
        results = {
            "ids": [node.node.id_ for node in nodes],
            "documents": [node.node.text for node in nodes],
            "metadatas": [node.node.metadata for node in nodes],
            "distances": [node.score for node in nodes]
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
    
    def process_message(self, message: str, message_idx: int):
        """Process and store a new message. Overwrites existing chunks if message_idx exists."""
        # Delete existing chunks for this message if any
        self.delete_message_chunks(message_idx)
        
        # Create and store new chunks
        chunks = self.chunk_message(message, message_idx)
        self.store_chunks(chunks)
        return chunks

    def store_chunks(self, chunks: list):
        """Store chunks using LlamaIndex."""
        from llama_index.core.schema import TextNode
        
        nodes = []
        for chunk in chunks:
            metadata = {
                "message_idx": chunk["indices"][0],
                "paragraph_idx": chunk["indices"][1],
                "sentence_idx": chunk["indices"][2],
                "is_summary": chunk.get("is_summary", False)
            }
            
            node = TextNode(
                text=chunk["text"],
                id_=chunk["id"],
                metadata=metadata
            )
            nodes.append(node)
            
        self.index.insert_nodes(nodes)
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))
