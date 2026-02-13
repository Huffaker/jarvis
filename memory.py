import json
import os
import threading
from pathlib import Path

from datetime import datetime, timezone

from app_types.memory import MemoryEntries, MemoryEntry
from personas import PERSONAS_DIR

MEMORY_FILE = "memory.json"
_ROOT = Path(__file__).resolve().parent

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


def _delete_generated_image_file(url_path: str | None) -> None:
    """
    If url_path is a known generated-image URL, delete the corresponding file from disk.
    Supports /personas/<id>/images/<filename> and legacy /static/generated/<filename>.
    """
    if not url_path or not url_path.startswith("/"):
        return
    url_path = url_path.strip()
    # Persona images: /personas/<persona_id>/images/<filename>
    if url_path.startswith("/personas/") and "/images/" in url_path:
        prefix = "/personas/"
        suffix = "/images/"
        try:
            after_prefix = url_path[len(prefix) :]
            id_part, filename = after_prefix.split(suffix, 1)
            if ".." in filename or "/" in filename or "\\" in filename or not filename:
                return
            fs_path = os.path.join(PERSONAS_DIR, id_part, "images", filename)
            if os.path.isfile(fs_path):
                os.remove(fs_path)
        except (ValueError, OSError):
            pass
        return


def _load_memory_unlocked(memory_file: str | None = None) -> MemoryEntries:
    """Load memory file without acquiring lock. Caller must hold _memory_lock."""
    path = memory_file or MEMORY_FILE
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return MemoryEntries()
    if isinstance(raw, list):
        return MemoryEntries.from_dict_list(raw)
    if isinstance(raw, dict) and "entries" in raw:
        return MemoryEntries.from_dict_list(raw["entries"])
    return MemoryEntries()


def load_memory(memory_file: str | None = None) -> MemoryEntries:
    """
    Load memory file. Returns a MemoryEntries instance.
    Handles legacy format (plain list of entries or dict with 'entries').
    memory_file: path to JSON file; if None, uses global MEMORY_FILE.
    """
    with _memory_lock:
        return _load_memory_unlocked(memory_file)


def _save_memory_unlocked(entries: MemoryEntries, memory_file: str | None = None) -> None:
    """Write memory to file without acquiring lock. Caller must hold _memory_lock."""
    path = memory_file or MEMORY_FILE
    out = MemoryEntries(list(entries.entries))
    while len(out.entries) > 1:
        payload = out.to_dict_list()
        if len(json.dumps({"entries": payload}, indent=2)) <= MAX_MEMORY_CHARS:
            break
        out.entries.pop(0)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": out.to_dict_list()}, f, indent=2)


def save_memory(entries: MemoryEntries, memory_file: str | None = None) -> None:
    """Persist memory: trim oldest entries until serialized size <= MAX_MEMORY_CHARS, then write."""
    with _memory_lock:
        _save_memory_unlocked(entries, memory_file)


def add_to_memory(entry: MemoryEntry, memory_file: str | None = None) -> str:
    """
    Append a MemoryEntry to short-term memory. Content and image_context are capped before saving.
    If entry.timestamp is empty, a new one is assigned.
    memory_file: path to persona memory JSON; if None, uses global MEMORY_FILE.
    Returns the timestamp of the created entry (for UI remove-from-memory).
    """
    with _memory_lock:
        data = _load_memory_unlocked(memory_file)
        ts = entry.timestamp or datetime.now(timezone.utc).isoformat()
        content = _cap_content(entry.content)
        image_context = entry.image_context
        if image_context:
            normalized = (image_context or "").replace("\n", " ").replace("\r", " ").strip()
            image_context = _cap_content(normalized, MAX_IMAGE_CONTEXT_CHARS)
        sources = entry.sources
        if sources and entry.role == "assistant":
            sources = [str(u).strip()[:MAX_SOURCE_URL_CHARS] for u in sources if u][:MAX_SOURCES]
        else:
            sources = []
        generated_image_prompt = None
        if entry.generated_image_prompt and entry.role == "assistant":
            generated_image_prompt = _cap_content(entry.generated_image_prompt)

        entry.timestamp = ts
        data.entries.append(entry)
        _save_memory_unlocked(data, memory_file)
    return ts


def update_entry(timestamp: str, updates: dict, memory_file: str | None = None) -> bool:
    """
    Update an existing entry by timestamp. updates can include content, summarized_content,
    generated_image_path, generated_image_prompt, etc. Content and generated_image_prompt are capped. Returns True if updated.
    """
    with _memory_lock:
        data = _load_memory_unlocked(memory_file)
        for e in data.entries:
            if e.timestamp == timestamp:
                if "content" in updates:
                    e.content = _cap_content(updates["content"])
                if "summarized_content" in updates:
                    e.summarized_content = _cap_content(updates["summarized_content"])
                if "generated_image_path" in updates:
                    e.generated_image_path = updates["generated_image_path"]
                if "generated_image_prompt" in updates:
                    e.generated_image_prompt = _cap_content(updates["generated_image_prompt"], MAX_ITEM_CHARS)
                _save_memory_unlocked(data, memory_file)
                return True
    return False


def delete_entry(timestamp: str, memory_file: str | None = None) -> bool:
    """Remove the entry with the given timestamp. Deletes its generated image file if any. Returns True if one was removed."""
    with _memory_lock:
        data = _load_memory_unlocked(memory_file)
        removed = None
        for e in data.entries:
            if e.timestamp == timestamp:
                removed = e
                break
        if removed is None:
            return False
        _delete_generated_image_file(removed.generated_image_path)
        data.entries = [e for e in data.entries if e.timestamp != timestamp]
        _save_memory_unlocked(data, memory_file)
        return True


def clear_memory(memory_file: str | None = None) -> None:
    """Remove all entries from memory and delete their generated image files."""
    with _memory_lock:
        data = _load_memory_unlocked(memory_file)
        for e in data.entries:
            _delete_generated_image_file(e.generated_image_path)
        path = memory_file or MEMORY_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"entries": []}, f, indent=2)


def get_recent_memory(memory_file: str | None = None) -> MemoryEntries:
    """Return all memory entries as a MemoryEntries instance (already size-limited when saved)."""
    with _memory_lock:
        return _load_memory_unlocked(memory_file)
