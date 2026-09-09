import re

from .console import _HILITE, _RESET

# --- LLM response parsing ---
patterns = (
    r'^[ \t]*(?:```(?:\S*)?|""")(.*?)(?:```|""")',  # Multi-line JSON object with code blocks
    r"^(?![\s\t])(?:.*?})({.*})",  # Single-line JSON object
    r"({.*?^})",  # Multi-line JSON object without code blocks
)
# json_regex = re.compile(patterns, flags=(re.MULTILINE + re.DOTALL))
print(f"{_HILITE}JSON regex: {_RESET}{patterns}")


def strip_json_response(output: str) -> str:
    """Remove any markdown formatting or extra text from a JSON response."""
    for pattern in patterns:
        match = re.search(pattern, output, flags=(re.MULTILINE + re.DOTALL))
        if match:
            return match.group(1)
    return output


def strip_thinking(output: str) -> str:
    """
    Exclude the "\\<think> ... \\</think>" tags and their content from a response.
    If \\<think> is present but \\</think> is not, returns an empty string.
    If \\</think> is present but \\<think> is not, returns everything after the first \\</think>.

    Compatible with Seed-OSS <seed:think> tags.
    """
    # <think> = 1, seed: = 2, </think> = 3
    open_tag, tag_prefix, close_tag = "", "", ""
    start = re.search(r"<((?:.+?:)?)think>", output, flags=(re.MULTILINE + re.DOTALL))
    if start:
        open_tag = start.group(0)
        tag_prefix = start.group(1)
    tag_prefix = tag_prefix or r"(?:.+?:)?"
    end = re.search(rf"<\/{tag_prefix}think>", output, flags=(re.MULTILINE + re.DOTALL))
    if end:
        close_tag = end.group(0)
    if not start and not end:
        return output
    if open_tag and not close_tag:
        return ""
    cleaned_output = output[end.end(0) :] if close_tag else output
    return cleaned_output.lstrip()


def strip_response(output: str) -> str:
    """Remove the response section from a response."""
    return strip_json_response(strip_thinking(output))


def extract_meaningful_paragraphs(text: str) -> str:
    """
    Extracts meaningful paragraphs from text, attempting to strip out common
    LLM-generated conversational filler, titles, and formatting.
    """
    if not text:
        return ""

    # 1. Strip leading/trailing quotes that might wrap the whole response
    processed_text = text.strip()
    if processed_text.startswith('"') and processed_text.endswith('"'):
        processed_text = processed_text[1:-1].strip()

    # 2. Remove common conversational intros
    intro_patterns = [  # TODO: Make these patterns configurable in the UI, including placeholders
        r"^(Here's|Here is) my response as \"\w+\":\s*\n*",
        r"^(Of course|Certainly|Here is the response|Here's the story|Here is your story), as requested:\s*\n*",
    ]
    for pattern in intro_patterns:
        processed_text = re.sub(pattern, "", processed_text, flags=re.IGNORECASE)

    # 3. Split into blocks and process each one
    blocks = re.split(r"\n\s*\n+", processed_text.strip())
    meaningful_blocks = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 4. Strip list markers and bolded titles (e.g., "* **Tactical Movement**:")
        # This handles cases where the title is on the same line as the paragraph start.
        block = re.sub(r"^\s*[\*\-]\s*\*\*(.*?)\*\*:\s*", "", block)

        # 5. Identify and discard standalone titles/headings
        is_likely_title = False
        if block.startswith("**") and block.endswith("**"):
            content_inside = block[2:-2].strip()
            if len(content_inside.split()) < 8 or content_inside.endswith(":"):
                is_likely_title = True

        if not is_likely_title and len(block.split()) < 8 and block.endswith(":"):
            is_likely_title = True

        if not is_likely_title:
            meaningful_blocks.append(block)

    return "\n\n".join(meaningful_blocks)
