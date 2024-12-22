import copy
import json
import os
from typing import Any, List, Tuple

# import gradio as gr

# import modules.shared as shared
# from modules.chat import generate_chat_prompt
# from modules.html_generator import fix_newlines

from utils._old.tree_handling import Tree, TreeEditor
from extensions.dayna_story_summarizer.agents.summarizer import Summarizer
#from utils.tree_handling import Tree, TreeEditor
import modules.shared as shared

global current_character
current_character: str = shared.settings['character']
print(f"current character {current_character}")

def load_json_with_fallback(file_path: str) -> dict:
    """Load a JSON file with fallback to default configuration if the file is empty or doesn't exist.
    
    Args:
        file_path (str): Path to the JSON file to load
        
    Returns:
        dict: The loaded configuration, either from the file or the default
    """
    try:
        with open(file_path, "rt") as handle:
            content = json.load(handle) or _load_and_save_default(file_path)
            print(f"Loaded config from {file_path}")
            return content
    except FileNotFoundError:
        print(f"Warning: Could not load config from {file_path}, using defaults")
        return _load_and_save_default(file_path)

def _load_and_save_default(file_path: str) -> dict:
    """Internal function to load default config and save it to the specified path.
    
    Args:
        file_path (str): Path where to save the default configuration
        
    Returns:
        dict: The default configuration
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Load default config
    with open("./extensions/dayna_story_summarizer/default_config.json", "rt") as default_handle:
        default_config = json.load(default_handle)
    # Save the default config
    with open(file_path, "wt") as handle:
        json.dump(default_config, handle, indent=4)
    return default_config

def update_config(state: dict) -> bool:
    """Update the current config state based on the current selected generation character.

    Returns:
        bool: True if updated, false if not.
    """
    global current_character, tree, tree_editor
    print(state['character_menu'])
    print(current_character)
    if state['character_menu'] != current_character:
        current_character = state['character_menu']
        config_file = f"./extensions/dayna_story_summarizer/user_data/config/{current_character}.json"
        print(config_file)
        sbj_tree = load_json_with_fallback(config_file)
    
        tree_editor = TreeEditor(copy.deepcopy(sbj_tree))
        tree_editor.json_data = sbj_tree
        tree = Tree(tree_editor)
        return True
    return False

global tree 
tree: Tree
global tree_editor
tree_editor: TreeEditor

global summarizer
summarizer: Summarizer