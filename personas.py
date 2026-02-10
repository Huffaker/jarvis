"""
Persona discovery and config loading.
Each persona lives in .personas/<persona_id>/ with:
  - <persona_id>.config  (JSON: name, public, system_persona, decision_model, model, vl_model,
      optional decisions: { "image_generation": true, "web_search": true, "prior_image_context": true },
      optional comfyui: models_dir, diffusion_model_name, clip_model_name, vae_model_name)
  - memory.json         (persona-specific conversation memory)

.personas/ is created on first use (gitignored). Add persona configs there; see README for an example.

decisions: per-persona toggles for LLM-based decisions (each runs once per turn). Omitted keys default to true.
  image_generation: whether to detect "generate an image" and route to image gen.
  web_search: whether to detect need for current info and run web search.
  prior_image_context: whether to include prior image context from memory when formatting context.
"""
import json
from pathlib import Path

from app_types.persona import (
    DEFAULT_DECISION_MODEL,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_SYSTEM_PERSONA,
    DEFAULT_VL_MODEL,
    Decisions,
    Persona,
)

# Re-export defaults for callers (e.g. agent_core) that use these names
SYSTEM_PERSONA = DEFAULT_SYSTEM_PERSONA
DECISION_MODEL = DEFAULT_DECISION_MODEL
MODEL = DEFAULT_MODEL
VL_MODEL = DEFAULT_VL_MODEL

# Resolve relative to this module so personas are found regardless of process CWD
_ROOT = Path(__file__).resolve().parent
PERSONAS_DIR = _ROOT / ".personas"
MEMORY_FILENAME = "memory.json"


def _ensure_personas_dir():
    """Create .personas if it does not exist so the app can run. Users add persona configs here (see README)."""
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)


def _persona_dirs():
    """Yield (persona_id, dir_path) for each persona directory that has a config."""
    if not PERSONAS_DIR.is_dir():
        return
    for child in PERSONAS_DIR.iterdir():
        if not child.is_dir():
            continue
        persona_id = child.name
        # Support both <id>.config and config
        config_candidates = [child / f"{persona_id}.config", child / "config"]
        for candidate in config_candidates:
            if candidate.is_file():
                yield (persona_id, child)
                break


def list_personas(public_only: bool = False) -> list[dict]:
    """
    Return list of personas. Each item: {id, name, path, config_path, memory_path}.
    public_only: if True, only include configs with "public": true.
    """
    _ensure_personas_dir()
    result = []
    for persona_id, dir_path in _persona_dirs():
        config_path = dir_path / f"{persona_id}.config"
        if not config_path.is_file():
            config_path = dir_path / "config"
        if not config_path.is_file():
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if public_only and not cfg.get("public", True):
            continue
        result.append({
            "id": persona_id,
            "name": cfg.get("name", persona_id),
            "path": str(dir_path),
            "config_path": str(config_path),
            "memory_path": str(dir_path / MEMORY_FILENAME),
        })
    return result


def get_persona_config(persona_id: str) -> dict | None:
    """
    Load full config for a persona. Returns dict with:
      name, public, system_persona, decision_model, model, vl_model, memory_path
    or None if not found / invalid.
    """
    _ensure_personas_dir()
    dir_path = PERSONAS_DIR / persona_id
    if not dir_path.is_dir():
        return None
    config_path = dir_path / f"{persona_id}.config"
    if not config_path.is_file():
        config_path = dir_path / "config"
    if not config_path.is_file():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    cfg["memory_path"] = str(dir_path / MEMORY_FILENAME)
    return cfg


def get_default_persona_id() -> str:
    """Return the first public persona id, or 'assistant' if none found. Private personas are never default."""
    personas = list_personas(public_only=True)
    if personas:
        return personas[0]["id"]
    return "assistant"


def persona_settings(persona_id: str | None):
    """
    Resolve persona_id to (system_persona, decision_model, model, vl_model, memory_path, decisions_config).
    memory_path is None to use global memory. persona_id None uses default persona.
    decisions_config is a Decisions instance; pass to code that expects .get(key, True).
    """
    pid = persona_id or get_default_persona_id()
    cfg = get_persona_config(pid)
    decisions_config = Decisions(cfg.get("decisions", {}) if cfg else {})
    if not cfg:
        return (DEFAULT_SYSTEM_PERSONA, DEFAULT_DECISION_MODEL, DEFAULT_MODEL, DEFAULT_VL_MODEL, None, decisions_config)
    return (
        cfg.get("system_persona") or DEFAULT_SYSTEM_PERSONA,
        cfg.get("decision_model") or DEFAULT_DECISION_MODEL,
        cfg.get("model") or DEFAULT_MODEL,
        cfg.get("vl_model") or DEFAULT_VL_MODEL,
        cfg.get("memory_path"),
        decisions_config,
    )


def persona_from_id(persona_id: str | None) -> Persona:
    """Build a Persona from persona_id. Uses default persona when persona_id is None."""
    pid = persona_id or get_default_persona_id()
    cfg = get_persona_config(pid)
    decisions = Decisions(cfg.get("decisions", {}) if cfg else {})
    if not cfg:
        return Persona(
            id=pid,
            name=DEFAULT_NAME,
            system_persona=DEFAULT_SYSTEM_PERSONA,
            decision_model=DEFAULT_DECISION_MODEL,
            model=DEFAULT_MODEL,
            vl_model=DEFAULT_VL_MODEL,
            memory_path=None,
            decisions=decisions,
        )
    return Persona(
        id=pid,
        name=cfg.get("name", DEFAULT_NAME),
        system_persona=cfg.get("system_persona") or DEFAULT_SYSTEM_PERSONA,
        decision_model=cfg.get("decision_model") or DEFAULT_DECISION_MODEL,
        model=cfg.get("model") or DEFAULT_MODEL,
        vl_model=cfg.get("vl_model") or DEFAULT_VL_MODEL,
        memory_path=cfg.get("memory_path"),
        decisions=decisions,
    )
