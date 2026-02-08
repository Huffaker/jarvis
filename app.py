import json
import socket
import sys
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from agent_core import answer, answer_stream, prepare_images_for_stream
from memory import get_recent_memory, delete_entry, clear_memory
from personas import list_personas, get_persona_config, get_default_persona_id

app = Flask(__name__)
CORS(app)


def _request_stay_awake():
    """Ask Windows to avoid sleeping while this process runs. Cleared when the process exits."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass


def _local_network_urls(port: int) -> list[str]:
    """Return URLs that other devices can use (consistent name + IP)."""
    urls = []
    try:
        hostname = socket.gethostname()
        # Many home networks support mDNS: use hostname.local for a stable name
        urls.append(f"http://{hostname}.local:{port}")
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        urls.append(f"http://{ip}:{port}")
    except Exception:
        pass
    return urls


def _memory_file_for_persona(persona_id):
    """Return memory file path for persona, or None for global memory."""
    if not persona_id:
        return None
    cfg = get_persona_config(persona_id)
    return cfg.get("memory_path") if cfg else None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/personas", methods=["GET"])
def personas_list():
    """Return list of available personas (id, name, path, config_path, memory_path)."""
    public_only = request.args.get("public", "true").lower() in ("1", "true", "yes")
    personas = list_personas(public_only=public_only)
    return jsonify({"personas": personas, "default": get_default_persona_id()})


@app.route("/memory/recent", methods=["GET"])
def memory_recent():
    """Return recent memory entries (for loading chat history with timestamps). Query: persona_id (optional)."""
    persona_id = request.args.get("persona_id") or get_default_persona_id()
    memory_file = _memory_file_for_persona(persona_id)
    entries = get_recent_memory(memory_file=memory_file)
    return jsonify({"entries": entries})


@app.route("/memory/delete", methods=["POST"])
def memory_delete():
    """Remove a single entry by timestamp. Body: {"timestamp": "...", "persona_id": "..."} (persona_id optional)."""
    data = request.json or {}
    ts = data.get("timestamp")
    persona_id = data.get("persona_id") or get_default_persona_id()
    if not ts:
        return jsonify({"error": "timestamp required"}), 400
    memory_file = _memory_file_for_persona(persona_id)
    if delete_entry(ts, memory_file=memory_file):
        return jsonify({"ok": True})
    return jsonify({"error": "entry not found"}), 404


@app.route("/memory/clear", methods=["POST"])
def memory_clear():
    """Remove all entries from memory. Body: {"persona_id": "..."} (persona_id optional)."""
    data = request.json or {}
    persona_id = data.get("persona_id") or get_default_persona_id()
    memory_file = _memory_file_for_persona(persona_id)
    clear_memory(memory_file=memory_file)
    return jsonify({"ok": True})


def _normalize_images(images):
    """Accept list of base64 strings; strip data URL prefix if present."""
    if not images:
        return []
    out = []
    for img in images:
        if not isinstance(img, str):
            continue
        if img.startswith("data:"):
            # data:image/png;base64,<payload>
            idx = img.find(",")
            if idx != -1:
                img = img[idx + 1 :]
        out.append(img)
    return out


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    images = _normalize_images(data.get("images") or [])
    persona_id = data.get("persona_id") or get_default_persona_id()
    if not message and not images:
        return jsonify({"error": "message or images required"}), 400
    response, sources = answer(
        message or "What do you see in the image(s)?",
        images=images or None,
        persona_id=persona_id,
    )
    return jsonify({
        "response": response,
        "sources": sources
    })


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json or {}
    message = data.get("message", "").strip()
    images = _normalize_images(data.get("images") or [])
    persona_id = data.get("persona_id") or get_default_persona_id()
    if not message and not images:
        return jsonify({"error": "message or images required"}), 400

    # Pre-compute resize + image context before starting stream so exceptions happen before headers are sent
    try:
        resized_images, image_context = prepare_images_for_stream(images)
    except Exception as e:
        return jsonify({"error": f"Failed to process images: {e!s}"}), 500

    def generate():
        for event in answer_stream(
            message or "What do you see in the image(s)?",
            images=resized_images if resized_images else None,
            image_context=image_context,
            persona_id=persona_id,
        ):
            yield "data: " + json.dumps(event) + "\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    _request_stay_awake()
    port = 5000
    # host="0.0.0.0" makes the server reachable from other devices on your network
    urls = _local_network_urls(port)
    if urls:
        print("On other devices, use one of:")
        for url in urls:
            print(f"  {url}")
    app.run(host="0.0.0.0", port=port, debug=True)
