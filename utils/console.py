# Colour codes
_ERROR = "\033[1;31m"
_SUCCESS = "\033[1;32m"
_INPUT = "\033[0;33m"
_GRAY = "\033[0;90m"
_HILITE = "\033[0;36m"
_BILITE = "\033[1;36m"
_BOLD = "\033[1;37m"
_WARNING = "\033[1;33m"
_RESET = "\033[0m"

_DEBUG = "\033[1;31m||\033[0;32m"

# Helper types
History = list[list[str]]
Histories = dict[str, History]


class TypedKey:
    def __init__(self, key: str, type: type):
        self.key = key
        self.type = type
