"""
Persona discovery and config loading.
Each persona lives in .personas/<persona_id>/ with:
  - <persona_id>.config  (JSON: name, public, system_persona, decision_model, model, vl_model)
  - memory.json         (persona-specific conversation memory)

personas_default/ is the checked-in template; .personas/ is created from it on first use (gitignored).
"""
import json
import shutil
from pathlib import Path

PERSONAS_DIR = Path(".personas")
PERSONAS_DEFAULT_DIR = Path("personas_default")
MEMORY_FILENAME = "memory.json"


def _ensure_personas_dir():
    """If .personas does not exist, copy from personas_default so the app works out of the box."""
    if not PERSONAS_DIR.exists() and PERSONAS_DEFAULT_DIR.is_dir():
        shutil.copytree(PERSONAS_DEFAULT_DIR, PERSONAS_DIR)


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
    """Return the first available persona id, or 'assistant' if none found."""
    personas = list_personas(public_only=False)
    if personas:
        return personas[0]["id"]
    return "assistant"
