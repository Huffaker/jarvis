"""Typed representation of memory entries (conversation history)."""

from dataclasses import dataclass, field


@dataclass
class MemoryEntry:
    """
    One message in short-term conversation memory.
    Stored in .personas/<id>/memory.json (or root memory.json) as a list of entry dicts.
    """

    timestamp: str
    role: str  # "user" or "assistant"
    content: str
    persona_name: str | None = None  # name of the persona (assistant only)
    image_context: str | None = None  # Description of the image provided by the user (only if one was provided)
    web_context: str | None = None  # in-memory only; not persisted to JSON
    sources: list[str] = field(default_factory=list)  # URLs, assistant only
    generated_image_path: str | None = None  # e.g. /static/generated/xxx.png, assistant only; UI only, not sent to LLM
    generated_image_prompt: str | None = None  # prompt used to generate the image (assistant only); optional in LLM context

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        """Build an entry from a JSON/dict (e.g. loaded from memory.json)."""
        return cls(
            timestamp=d.get("timestamp", ""),
            role=d.get("role", "user"),
            persona_name=d.get("persona_name", ""),
            content=d.get("content", ""),
            image_context=d.get("image_context"),
            web_context=None,  # not persisted
            sources=list(d.get("sources") or []),
            generated_image_path=d.get("generated_image_path"),
            generated_image_prompt=d.get("generated_image_prompt"),
        )

    def to_dict(self) -> dict:
        """Serialize for saving to memory.json. Omits web_context (not persisted)."""
        out: dict = {
            "timestamp": self.timestamp,
            "role": self.role,
            "persona_name": self.persona_name,
            "content": self.content,
        }
        if self.image_context:
            out["image_context"] = self.image_context
        if self.sources:
            out["sources"] = self.sources
        if self.generated_image_path:
            out["generated_image_path"] = self.generated_image_path
        if self.generated_image_prompt:
            out["generated_image_prompt"] = self.generated_image_prompt
        return out

    def build_prompt(
        self,
        include_image_context: bool = True,
        include_generated_image_prompt: bool = True,
    ) -> str:
        """Format this entry for inclusion in the LLM prompt. Omits sources, generated_image_path."""
        line = f"{self.persona_name if self.role == 'assistant' else self.role}: {self.content}"
        if self.image_context and include_image_context:
            line += f"\n[past image memory: {self.image_context}]"
        if self.web_context:
            line += f"\n[web context: {self.web_context}]"
        if self.generated_image_prompt and include_generated_image_prompt:
            line += f"\n[generated image prompt: {self.generated_image_prompt}]"
        return line


class MemoryEntries:
    """Container for a list of MemoryEntry; load/save via memory.load_memory / save_memory."""

    def __init__(self, entries: list[MemoryEntry] | None = None):
        self.entries: list[MemoryEntry] = list(entries or [])

    def build_prompt(
        self,
        include_image_context: bool = True,
        include_generated_image_prompt: bool = True,
    ) -> str:
        return "\n".join(
            entry.build_prompt(include_image_context, include_generated_image_prompt)
            for entry in self.entries
        )

    def to_dict_list(self) -> list[dict]:
        """List of dicts for JSON serialization."""
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_dict_list(cls, data: list[dict]) -> "MemoryEntries":
        """Build from a list of dicts (e.g. from memory.json)."""
        return cls([MemoryEntry.from_dict(d) for d in data])
