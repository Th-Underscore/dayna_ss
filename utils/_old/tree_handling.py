import json
import copy

# Box drawing symbols
_TREE_VERTICAL = "│ "
_TREE_CONNECTOR = "├ "
_TREE_CORNER = "└ "
_TREE_WALL = "└─"
_TREE_SPLIT = "┴─"
_TREE_FINAL = "┴ "

# Colour codes
_ERROR = "\033[1;31m"
_SUCCESS = "\033[1;32m"
_INPUT = "\033[0;33m"
_GRAY = "\033[0;30m"
_BOLD = "\033[1;37m"
_RESET = "\033[0m"

_INDENT = "\033[0;34m"
_BRACKET = "\033[0;35m"
_PROPERTY = "\033[0;36m"
_VALUE = "\033[0;32m"

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class TreeEditor:
	def __init__(self, data: dict):
		if data is None:
			self.data = {}
		else:
			self.data = data

	def add_branch(self, path: str, content: str):
		if content is None:
			content = {}
		current = self.data
		for part in path[:-1]:
			if part not in current:
				current[part] = {}
			current = current[part]
		current[path[-1]] = content

	def edit_branch(self, path: str, new_content: str):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot edit.")
				return
			current = current[part]
		current.clear()
		current.update(new_content) # type: ignore

	def remove_branch(self, path: str):
		current = self.data
		for part in path[:-1]:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot remove.")
				return
			current = current[part]
		del current[path[-1]]

	def add_property(self, path: str, property_name: str, value: str):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot add property.")
				return
			current = current[part]
		current[property_name] = value

	def edit_property(self, path: str, property_name: str, new_value: str):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot edit property.")
				return
			current = current[part]
		current[property_name] = new_value

	def remove_property(self, path: str, property_name: str):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot remove property.")
				return
			current = current[part]
		del current[property_name]

	def add_value(self, path: str, value: str):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot add value.")
				return
			current = current[part]
		if 'values' not in current:
			current['values'] = []
		current['values'].append(value)

	def edit_value(self, path: str, index: int, new_value: str):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot edit value.")
				return
			current = current[part]
		if 'values' not in current or not (0 <= index < len(current['values'])):
			print(_ERROR + "Value index out of range. Cannot edit.")
			return
		current['values'][index] = new_value

	def remove_value(self, path: str, index: int):
		current = self.data
		for part in path:
			if part not in current:
				print(_ERROR + "Path does not exist. Cannot remove value.")
				return
			current = current[part]
		if 'values' not in current or not (0 <= index < len(current['values'])):
			print(_ERROR + "Value index out of range. Cannot remove.")
			return
		del current['values'][index]

	# ADD VALUE PROPERTY COMMANDS HERE
		
	def display(self):
		self._display_helper(self.data, [1], [])
	
	def _display_helper(self, data: dict, last: list, roots: list): # Could have last and roots as one dictionary but eh
		indent = len(last) - 1
		for key, value in data.items():
			last[indent] -= 1
			if isinstance(value, dict):
				new_last = last.copy(); new_last += [len(value)]
				if roots:
					last_root = roots[-1]
					self._display_text(new_last, indent, f"{_BOLD}{key} {_GRAY + '(' + last_root + ')' + _RESET}", True)
					new_roots = roots.copy(); new_roots += [f"{last_root}.{key}"]
				else:
					self._display_text(new_last, indent, _BOLD + key, True)
					new_roots = roots.copy(); new_roots += [key]
				self._display_helper(value, new_last, new_roots)
			else:
				if key == 'values':
					value_names = [v['name'] for v in value]
					self._display_text(last, indent, f"{key}: {_BRACKET}[{_VALUE} {', '.join(value_names).replace(',', _RESET+','+_VALUE)} {_BRACKET}]")
				else:
					self._display_text(last, indent, f"{key}: {_PROPERTY}{value.replace('{', _INDENT+'{'+_RESET).replace('}', _INDENT+'}'+_PROPERTY)}")
				
	def _display_text(self, last: list, indent: int, text: str, is_dict=False):
		# Count zeroes at end of list
		final = 0
		for i in reversed(last):
			if i != 0:
				break
			final += 1
		if final == 1:
			print(_INDENT + _TREE_VERTICAL*indent + _TREE_CORNER + _RESET + text)
		elif final:
			print(_INDENT + _TREE_VERTICAL*(indent-final-is_dict+1) + _TREE_WALL + _TREE_SPLIT*(final-2-is_dict) + _TREE_FINAL + _RESET + text)
		else:
			print(_INDENT + _TREE_VERTICAL*indent + _TREE_CONNECTOR + _RESET + text)
	
	def count_zeroes_at_end(self, last: list):
		zero_count = 0
		for i in reversed(last):
			if i != 0:
				return zero_count
			zero_count += 1
		return zero_count

	def save(self, filename: str):
		with open(filename, 'w') as file:
			json.dump(self.data, file, indent=4)
		self.json_data = self.data

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Tree:
	def __init__(self, editor: TreeEditor):
		"""Initializes a Tree object with a TreeEditor.

		Args:
			editor: A TreeEditor object that contains the tree data.
		"""
		self.editor = editor
			
	def get_custom_prompts(self):
		"""Retrieve all custom prompts from the tree with placeholders replaced.

		This method traverses the tree structure and collects prompts containing
		placeholders (e.g., "{key}"), replacing them with formatted lists of values.

		Returns:
			list: A list of prompts with placeholders replaced.
		"""
		generation_data = {}
		self._recursive_get_keys(self.editor.data, generation_data)

		return self._parse_prompts(generation_data.values())

	def _parse_prompts(self, data: list):
		"""Replace placeholders (i.e. "[key]") in string with formatted list.

		Args:
			string (str): The prompt string
			data (list): The generation prompts and properties.

		Returns:
			list: The list of prompts to use for generation.
		"""
		prompts = []
		for generation in data: # i.e. main or gen1
			# Total dictionary of prompts for current generation
			generation_prompt = {}
			for sbj in generation:
				sbj_properties = sbj["properties"]
				sbj_prompt = sbj_properties["prompt"]
				for (key, property) in sbj_properties.items(): # i.e. type_of_character
					# Find placeholders in string and replace with property value.
					sbj_prompt = sbj_prompt.replace("{" + key + "}", property)
				# Add subject name to list with prompt as key.
				generation_prompt.setdefault(sbj_prompt, []).append(sbj["name"])
			# Loop all collected prompts.
			generation_prompts = ""
			for (prompt, names) in generation_prompt.items():
				# Format value as English-readable string (i.e. "1, 2, and 3").
				formatted_list = get_readable_list(names)
				prompt = prompt.replace("{name}", formatted_list)
				generation_prompts += f"{prompt}\n"
			prompts.append(generation_prompts)
		return prompts

	def _recursive_get_keys(self, obj: dict, result: dict, properties: dict={}):
		"""Get all keys of each branch in the generation dictionary.

		Args:
			obj (dict) -> the generation dictionary.\n
			result (list) -> the list reference to be modified.\n
			generations (dict) -> the generation order of each subject.\n
			properties (dict, optional) -> recursive variable, leave empty.
		"""
		# Recursively loop keys (i.e. "value=type") in dictionary.
		for (key, value) in obj.items(): # i.e. main with 1 recursion
			if isinstance(value, list):
				for sbj in value: # i.e. main.character[0] with 2 recursions
					# Create copy of current properties for each subject.
					sbj_properties = properties.copy()
					# Get generation key of subject, default to last property.
					generation_key = sbj.get("generation_key", list(sbj_properties.values())[-1])
					for (sbj_key, sbj_value) in sbj.items(): # i.e. main.character[0].prompt with 2 recursions
						if sbj_key == "name":
							name = sbj_value
						else:
							sbj_properties[sbj_key] = sbj_value
					debug(f"generation key {generation_key} | name {name}")
					# Add subject name and properties to result using generation key.
					result.setdefault(generation_key, []).append({"name": name, "properties": sbj_properties})
			elif isinstance(value, str):
				properties[key] = value
			elif isinstance(value, dict):
				self._recursive_get_keys(value, result, properties.copy())

def get_readable_list(seq: list) -> str:
	"""Return a grammatically correct human readable string (with an Oxford comma)
	Ref: https://stackoverflow.com/a/53981846/

	Args:
		seq (list) -> the list to stringify.

	Returns:
		str -> the list as a readable string.
	"""
	
	seq = [str(s) for s in seq]
	if len(seq) < 3:
		return ' and '.join(seq)
	return ', '.join(seq[:-1]) + ', and ' + seq[-1]

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

debug_color = 0
def debug(msg: str):
	#global debug_color
	print(f"\033[0;{30+debug_color}m" + msg)
	#debug_color = (debug_color + 1) % 8