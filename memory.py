import json
import threading
from datetime import datetime

MEMORY_FILE = "memory.json"

# Serialize all file access so concurrent web requests don't corrupt or lose updates.
_memory_lock = threading.Lock()
# Max total character size of the memory JSON file. Oldest entries are removed until the file fits.
MAX_MEMORY_CHARS = 50_000
MAX_ITEM_CHARS = 1000
MAX_IMAGE_CONTEXT_CHARS = 2000
MAX_SOURCES = 20
MAX_SOURCE_URL_CHARS = 500


def _cap_content(content: str, max_len: int = MAX_ITEM_CHARS) -> str:
    """Truncate content to a reasonable length to avoid overloading the prompt."""
    if not content or len(content) <= max_len:
        return content or ""
    return content[: max_len - 3].rstrip() + "..."


def load_memory(memory_file: str | None = None):
    """
    Load memory file. Returns dict with key 'entries' (list of message entries).
    Handles legacy format (plain list of entries or dict with 'images' and 'entries').
    memory_file: path to JSON file; if None, uses global MEMORY_FILE.
    """
    path = memory_file or MEMORY_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {"entries": []}
    if isinstance(raw, list):
        return {"entries": raw}
    if isinstance(raw, dict) and "entries" in raw:
        return {"entries": raw["entries"]}
    return {"entries": []}


def save_memory(data: dict, memory_file: str | None = None) -> None:
    """Persist memory: trim oldest entries until serialized size <= MAX_MEMORY_CHARS, then write."""
    path = memory_file or MEMORY_FILE
    entries = list(data["entries"])
    while len(entries) > 1:
        payload = {"entries": entries}
        if len(json.dumps(payload, indent=2)) <= MAX_MEMORY_CHARS:
            break
        entries.pop(0)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, indent=2)


def add_to_memory(
    role: str,
    content: str,
    image_context: str | None = None,
    memory_file: str | None = None,
    sources: list[str] | None = None,
) -> str:
    """
    Append a message to short-term memory.
    role: "user" or "assistant"
    content: text of the message (capped to MAX_ITEM_CHARS).
    image_context: optional text summary of user-attached image(s) from the vision model (capped).
    memory_file: path to persona memory JSON; if None, uses global MEMORY_FILE.
    sources: optional list of URL strings (for assistant messages with web search); capped by MAX_SOURCES and MAX_SOURCE_URL_CHARS.
    Returns the timestamp of the created entry (for UI remove-from-memory).
    """
    with _memory_lock:
        data = load_memory(memory_file=memory_file)
        ts = datetime.utcnow().isoformat()
        entry = {
            "timestamp": ts,
            "role": role,
            "content": _cap_content(content),
        }
        if image_context:
            normalized = (image_context or "").replace("\n", " ").replace("\r", " ").strip()
            entry["image_context"] = _cap_content(normalized, MAX_IMAGE_CONTEXT_CHARS)
        if sources and role == "assistant":
            urls = [str(u).strip()[:MAX_SOURCE_URL_CHARS] for u in sources if u][:MAX_SOURCES]
            if urls:
                entry["sources"] = urls
        data["entries"].append(entry)
        save_memory(data, memory_file=memory_file)
    return ts


def delete_entry(timestamp: str, memory_file: str | None = None) -> bool:
    """Remove the entry with the given timestamp. Returns True if one was removed."""
    with _memory_lock:
        data = load_memory(memory_file=memory_file)
        n = len(data["entries"])
        data["entries"] = [e for e in data["entries"] if e.get("timestamp") != timestamp]
        if len(data["entries"]) < n:
            save_memory(data, memory_file=memory_file)
            return True
    return False


def clear_memory(memory_file: str | None = None) -> None:
    """Remove all entries from memory."""
    with _memory_lock:
        path = memory_file or MEMORY_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"entries": []}, f, indent=2)


def get_recent_memory(memory_file: str | None = None):
    """Return all memory entries (already size-limited when saved). Each may have role, content, image_context, sources."""
    with _memory_lock:
        data = load_memory(memory_file=memory_file)
        return list(data["entries"])  # return a copy so callers don't mutate under lock


def format_memory_entry(entry: dict, include_image_context: bool = True) -> str:
    """
    Format a single memory entry for inclusion in the prompt.
    Includes image context if present and include_image_context is True.
    Intentionally omits 'sources' (web URLs): stored for UI display only, not sent to the LLM.
    """
    line = f"{entry['role']}: {entry['content']}"
    ctx = entry.get("image_context") or ""
    if ctx and include_image_context:
        line += f"\n[past image memory: {ctx}]"
    return line
