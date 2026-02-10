# Default values when no persona config exists (used by personas.persona_settings / persona_from_id)
DEFAULT_NAME = "Assistant"
DEFAULT_SYSTEM_PERSONA = "You are a helpful AI assistant. You remain accurate, concise, and calm."
DEFAULT_DECISION_MODEL = "qwen3:0.6b"
DEFAULT_MODEL = "qwen3:4b"
DEFAULT_VL_MODEL = "qwen3-vl:4b"


class Decisions:
    """
    Per-persona toggles for LLM-based decisions (each runs once per turn).
    Built from config dict: { "image_generation": true, "web_search": true, "prior_image_context": true }.
    Missing keys default to True (enabled).
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.image_generation = config.get("image_generation", True)
        self.web_search = config.get("web_search", True)
        self.prior_image_context = config.get("prior_image_context", True)

    def get(self, key: str, default: bool = True) -> bool:
        """Dict-like get for compatibility; unknown keys default to True (enabled)."""
        return getattr(self, key, default)


class Persona:
    def __init__(
        self,
        id: str,
        name: str,
        system_persona: str,
        decision_model: str,
        model: str,
        vl_model: str,
        memory_path: str | None,
        decisions: Decisions,
    ):
        self.id = id
        self.name = name
        self.system_persona = system_persona
        self.decision_model = decision_model
        self.model = model
        self.vl_model = vl_model
        self.memory_path = memory_path
        self.decisions = decisions
        self.can_web_search = decisions.web_search
