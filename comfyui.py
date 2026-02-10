"""
ComfyUI image generation.
Calls a local ComfyUI server (default http://127.0.0.1:8188) to generate an image from a text prompt.
Uses the default ComfyUI workflow: UNETLoader + CLIPLoader + VAE -> KSampler -> VAEDecode -> SaveImage.

The server must already be running. If it is not reachable, image generation returns an error.
Optional config from .env: COMFYUI_URL, COMFYUI_DEBUG.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Load .env from project root so COMFYUI_URL etc. can be set without exporting in the shell
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path)
    except ImportError:
        pass

# Set COMFYUI_DEBUG=1 to log the workflow and full error responses to stderr
DEBUG = os.environ.get("COMFYUI_DEBUG", "").strip() in ("1", "true", "yes")

# Server URL (set COMFYUI_URL in .env to override). Model names from persona config or env.
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
POLL_INTERVAL = float(os.environ.get("COMFYUI_POLL_INTERVAL", "0.5"))
MAX_WAIT_SECONDS = float(os.environ.get("COMFYUI_MAX_WAIT", "520"))

# Default model names when no persona comfyui config (env or these fallbacks)
_DEFAULT_DIFFUSION = os.environ.get("DIFFUSION_MODEL_NAME", "z-image-turbo-fp8-e4m3fn.safetensors")
_DEFAULT_CLIP = os.environ.get("CLIP_MODEL_NAME", "qwen_3_4b.safetensors")
_DEFAULT_VAE = os.environ.get("VAE_MODEL_NAME", "ae.safetensors")


def _build_workflow(diffusion_name: str, clip_name: str, vae_name: str) -> dict:
    """Build the default workflow dict with the given model names (from persona config or env)."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": diffusion_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_name, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "width": 512, "height": 512}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "bad hands, blurry, low quality"}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
                "seed": 0, "steps": 8, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ollama-agent", "images": ["8", 0]}},
        "10":  {
            "inputs": {
            "shift": 3,
            "model": [
                "1",
                0
            ]
            },
            "class_type": "ModelSamplingAuraFlow",
        }
    }


def _is_comfyui_running(timeout: float = 2.0) -> bool:
    """Return True if the ComfyUI server responds."""
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as _:
            return True
    except Exception:
        return False


def _require_comfyui_running() -> None:
    """Raise RuntimeError if the ComfyUI server is not reachable."""
    if not _is_comfyui_running():
        raise RuntimeError(
            f"ComfyUI server is not running at {COMFYUI_URL}. "
            "Start ComfyUI manually (e.g. run python main.py from your ComfyUI folder), "
            "or set COMFYUI_URL in .env if your server uses a different port."
        )


def is_comfyui_running() -> bool:
    """Return True if the ComfyUI server is reachable."""
    return _is_comfyui_running()


def _request(method: str, path: str, data: dict | None = None, timeout: int = 60) -> dict | None:
    url = f"{COMFYUI_URL}{path}"
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        msg = f"ComfyUI request failed: HTTP {e.code} {e.reason}"
        if body:
            msg += f"\nResponse: {body[:2000]}"
        if DEBUG and data is not None:
            msg += f"\nRequest body (first 2000 chars): {json.dumps(data)[:2000]}"
        raise RuntimeError(msg) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"ComfyUI request failed: {e}") from e


def _queue_prompt(prompt: dict) -> str:
    """Submit the workflow to ComfyUI. Returns prompt_id."""
    out = _request("POST", "/prompt", data={"prompt": prompt}, timeout=30)
    if out is None:
        raise RuntimeError("ComfyUI returned no response")
    if "error" in out:
        raise RuntimeError(f"ComfyUI error: {out['error']}")
    pid = out.get("prompt_id")
    if not pid:
        raise RuntimeError("ComfyUI did not return a prompt_id")
    return pid


def _get_history(prompt_id: str) -> dict | None:
    """Get history for a prompt_id. Returns the history entry or None if not yet completed."""
    out = _request("GET", f"/history/{prompt_id}", timeout=10)
    if not out or prompt_id not in out:
        return None
    return out[prompt_id]


def _get_output_image(prompt_id: str, history: dict) -> dict | None:
    """Extract first output image from SaveImage node (9). Returns image dict (filename, type, subfolder) or None."""
    outputs = history.get("outputs") or {}
    node9 = outputs.get("9") or {}
    images = node9.get("images") or []
    if not images:
        return None
    return images[0]


def _fetch_image_bytes(filename: str, type_dir: str = "output", subfolder: str = "") -> bytes:
    """GET /view to retrieve image bytes."""
    params = {"filename": filename, "type": type_dir}
    if subfolder:
        params["subfolder"] = subfolder
    q = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    url = f"{COMFYUI_URL}/view?{q}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def generate_image(prompt: str, comfyui_config: dict | None = None) -> dict:
    """
    Generate an image from a text prompt using a local ComfyUI server.

    Model names and optional models_dir come from persona config (comfyui_config) or from env
    DIFFUSION_MODEL_NAME, CLIP_MODEL_NAME, VAE_MODEL_NAME. models_dir is for reference; ComfyUI
    server uses its own models path.

    Args:
        prompt: Image generation prompt (positive prompt for the default workflow).
        comfyui_config: Optional dict from persona config with keys models_dir, diffusion_model_name,
            clip_model_name, vae_model_name. If None, env vars or defaults are used.

    Returns:
        dict with:
          - status: "ok" | "placeholder" | "error"
          - prompt: the prompt that was used
          - image_base64: optional base64-encoded PNG (when status == "ok")
          - error: optional error message (when status == "error")
    """
    if not prompt or not prompt.strip():
        return {"status": "error", "prompt": prompt, "error": "Empty prompt"}

    cfg = comfyui_config or {}
    diffusion_name = cfg.get("diffusion_model_name") or _DEFAULT_DIFFUSION
    clip_name = cfg.get("clip_model_name") or _DEFAULT_CLIP
    vae_name = cfg.get("vae_model_name") or _DEFAULT_VAE

    try:
        _require_comfyui_running()
        workflow = _build_workflow(diffusion_name, clip_name, vae_name)
        workflow["5"]["inputs"]["text"] = prompt.strip()  # positive prompt = node 5
        workflow["7"]["inputs"]["seed"] = int(time.time() * 1000) % (2**32)

        if DEBUG:
            print("COMFYUI_DEBUG: workflow being sent:", json.dumps(workflow, indent=2), file=sys.stderr)

        prompt_id = _queue_prompt(workflow)
        deadline = time.monotonic() + MAX_WAIT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(POLL_INTERVAL)
            history = _get_history(prompt_id)
            if history is None:
                continue
            if "status" in history and history.get("status") == "error":
                return {
                    "status": "error",
                    "prompt": prompt,
                    "error": history.get("status_messages", ["Unknown error"]) or "ComfyUI reported an error",
                }
            img_info = _get_output_image(prompt_id, history)
            if img_info:
                filename = img_info.get("filename", "")
                type_dir = img_info.get("type", "output")
                subfolder_str = img_info.get("subfolder", "")
                b = _fetch_image_bytes(filename, type_dir=type_dir, subfolder=subfolder_str)
                b64 = base64.b64encode(b).decode("ascii")
                return {
                    "status": "ok",
                    "prompt": prompt,
                    "image_base64": b64,
                }
        return {
            "status": "error",
            "prompt": prompt,
            "error": "ComfyUI did not finish within the timeout",
        }
    except Exception as e:
        return {
            "status": "error",
            "prompt": prompt,
            "error": str(e),
        }
