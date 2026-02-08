"""
Image context: resize images for the LLM and encode them as text summaries via a vision model.
"""
import base64
import io

# Max dimension (width or height) for images sent to Ollama; larger images are scaled down.
IMAGE_MAX_SIZE = 512

IMAGE_CONTEXT_PROMPT = """You are a visual memory encoder.
Summarize this image into compact semantic memory suitable for future AI context.
Write in third person.
Focus on persistent traits and ignore temporary details.
If the image contains a recognizable person, include their name and any other relevant information.
Keep it under 150 tokens."""


def resize_image_for_llm(image_base64: str, max_size: int = IMAGE_MAX_SIZE) -> str:
    """
    Scale down an image so the longest side is at most max_size pixels.
    Returns base64 JPEG. On failure (e.g. no Pillow or invalid image), returns original string.
    Reduces payload size and avoids Ollama 500 errors with large images.
    """
    try:
        from PIL import Image
    except ImportError:
        return image_base64
    raw = image_base64.strip()
    if raw.startswith("data:"):
        idx = raw.find(",")
        if idx != -1:
            raw = raw[idx + 1 :]
    raw = raw.replace("\n", "").replace("\r", "")
    try:
        pad = (4 - len(raw) % 4) % 4
        if pad:
            raw += "=" * pad
        decoded = base64.b64decode(raw, validate=False)
        img = Image.open(io.BytesIO(decoded))
        img.load()
        if max(img.size) <= max_size:
            return image_base64
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img.thumbnail((max_size, max_size), resample)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return image_base64


def image_context_for_image(image_base64: str) -> str:
    """
    Get a compact text summary of one image via the vision model.
    Passes the image to Ollama so the model can describe it.
    Returns fallback text on failure so callers don't break.
    """
    from agent_core import ask_ollama
    try:
        return ask_ollama(IMAGE_CONTEXT_PROMPT, images=[image_base64])
    except Exception:
        return "[Image description unavailable]"


def image_context_for_images(images: list) -> str | None:
    """
    For each image provided by the user, get a text summary via the vision model.
    Returns combined text (one summary per image, numbered) or None if no images.
    """
    if not images:
        return None
    parts = []
    for i, img in enumerate(images, 1):
        summary = image_context_for_image(img)
        if summary:
            parts.append(f"Image {i}: {summary.strip()}")
    return "\n".join(parts) if parts else None


def prepare_images_for_stream(images: list) -> tuple[list, str | None]:
    """
    Resize images and compute image_context before starting a stream.
    Call this in the request handler so any exception happens before response headers are sent.
    Returns (resized_images, image_context). On image-context failure, returns (resized, None).
    """
    if not images:
        return [], None
    resized = [resize_image_for_llm(img) for img in images]
    try:
        ctx = image_context_for_images(resized)
        return (resized, ctx)
    except Exception:
        return (resized, None)
