class Prompt:
    def __init__(
        self,
        question: str,
        memory_entries: "MemoryEntries | None" = None,
        extra_context: str | None = None,
        system_persona: str | None = None,
    ):
        self.question = question
        self.memory_entries = memory_entries
        self.extra_context = extra_context
        self.system_persona = system_persona
        self.web_context: str | None = None
        self.sources: list = []
        self.include_memory_image_context = True
        self.include_generated_image_prompt = True

    def build(self) -> str:
        default_system = "You are a helpful AI assistant. You remain accurate, concise, and calm."
        parts = []
        parts.append(self.system_persona or default_system)
        if self.memory_entries:
            parts.append(f"Conversation memory:\n{self.memory_entries.build_prompt(self.include_memory_image_context, self.include_generated_image_prompt)}\n")
        if self.extra_context:
            parts.append(f"Additional context:\n{self.extra_context}\n")
        if self.web_context:
            parts.append(
                "Use the following web information to answer the question. "
                "Base your answer on this information unless the prompt includes an image, then also use the image context.\n\n"
                f"Web information:\n{self.web_context}\n"
            )
        parts.append(f"Question:\n{self.question}")
        return "\n".join(parts)
