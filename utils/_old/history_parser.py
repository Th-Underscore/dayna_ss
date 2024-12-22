import copy
import json
import pathlib
import pprint
from typing import Any, List, Tuple

import gradio as gr

import modules.shared as shared
from modules.chat import generate_chat_prompt
from modules.html_generator import fix_newlines

_JSON_PATH = "extensions/dayna_story_summarizer/utils/examples/post_apocalyptic.json"
with open(_JSON_PATH, "rt") as handle:
	_HISTORY = json.load(handle)

print("============ TEST HISTORY ============")
print(_HISTORY)

def read_history():
	"""Stringify Super Story Summarizer subject list.
	
	Returns:
		str: Each subject history 
	"""
	subjects = {}
	with open(_JSON_PATH, "rt") as handle:
		_HISTORY = json.load(handle)
	# Get current subjects separately from other branches
	recursive_get_subjects(_HISTORY["current"], subjects)

	string = ""
	keys = subjects.keys()
	print(f"keys {keys}")
	for key in keys:
		string += '\n'.join(subjects[key]) + "\n\n"
		print("added \\n\\n")

	return string

def recursive_get_subjects(obj: dict, result: dict):
	"""Retrieve all subject histories in a parsed history dictionary.

	Args:
		obj (dict) -> the parsed JSON history dictionary.\n
		result (dict) -> the result dictionary reference (modified recursively).
	"""
	if isinstance(obj, dict):
		for (key, value) in obj.items():
			if isinstance(value, list):
				result.update(obj)
			else:
				recursive_get_subjects(value, result)