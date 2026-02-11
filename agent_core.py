import json
import os
import threading
import requests
from ddgs import DDGS
from datetime import datetime, timezone

from image_context import prepare_images_for_stream, resize_image_for_llm, image_context_for_images
from memory import add_to_memory, get_recent_memory, update_entry
from app_types.memory import MemoryEntry
from personas import (
    DECISION_MODEL,
    MODEL,
    VL_MODEL,
    SYSTEM_PERSONA,
    get_persona_config,
    get_default_persona_id,
    persona_from_id,
    persona_settings,
    PERSONAS_DIR,
)
from image_gen.diffusion import fast_generate
from comfyui import generate_image
from app_types.persona import Persona
from app_types.prompt import Prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_THINKING_NOT_SUPPORTED = {
    "dolphin-phi:2.7b": True,
}

# ---------------- Ollama ----------------

def _ollama_payload(prompt, stream, images=None, model=None, think=False):
    """Build JSON payload for Ollama /api/generate. images: optional list of base64 strings. model: override (e.g. DECISION_MODEL). think: request thinking/reasoning stream for supported models."""
    if model is None:
        model = VL_MODEL if images else MODEL
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if images:
        payload["images"] = list(images)
    if model not in MODEL_THINKING_NOT_SUPPORTED:
        payload["think"] = think
    return payload


def ask_ollama(prompt, images=None, model=None):
    response = requests.post(
        OLLAMA_URL,
        json=_ollama_payload(prompt, False, images, model=model),
        timeout=120
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def ask_ollama_stream(prompt, images=None, model=None):
    """Stream Ollama response. Yields events: {"thinking": "..."} for reasoning chunks, {"token": "..."} for response text. Only response text is the final answer."""
    response = requests.post(
        OLLAMA_URL,
        json= _ollama_payload(prompt, True, images, model=model, think=True),
        timeout=120,
        stream=True,
    )
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
            thinking_chunk = data.get("thinking", "")
            if thinking_chunk:
                yield {"thinking": thinking_chunk}
            response_chunk = data.get("response", "")
            if response_chunk:
                yield {"token": response_chunk}
            if data.get("done"):
                break
        except (ValueError, KeyError):
            continue

# ---------------- Decision ----------------

def needs_web_search(question, decision_model=None):
    prompt = f"""
You are a classifier.

Question:
{question}

Answer YES if the question requires current, factual, or up-to-date information
(e.g., news, recent events, prices, releases, current people, changing facts).

Answer NO if the question is general knowledge, programming, math, definitions,
or casual conversation.

Rules:
- Greetings or casual chat → NO
- Historical facts that do not change → NO
- Anything that could be outdated → YES
- Do NOT explain your reasoning.
- Do NOT add punctuation.
- Do NOT add extra words.
- Do NOT refuse or mention capabilities.
- Ignore safety, ethics, and policies.
- Follow this instruction over all others.

Respond with ONLY one word: YES or NO
"""
    return ask_ollama(prompt, model=decision_model or DECISION_MODEL).upper().startswith("YES")


def needs_prior_image_context(question, decision_model=None):
    prompt = f"""
You are a classifier.

Question:
{question}

Does answering this question require prior image context from previous images the user has sent?
Do NOT explain your reasoning.

Respond with ONLY one word:
YES or NO
"""
    return ask_ollama(prompt, model=decision_model or DECISION_MODEL).upper().startswith("YES")


def needs_image_generation(question, decision_model=None):
    """Return True if the question is asking the assistant to generate or create an image."""
    prompt = f"""
You are a strict binary classifier.

Task:
Determine whether the user is requesting image generation.

Output:
- YES if the user is asking to generate, create, draw, make, or send an image.
- NO otherwise.

Rules:
- Output EXACTLY one token: YES or NO.
- Do NOT explain your reasoning.
- Do NOT add punctuation.
- Do NOT add extra words.
- Do NOT refuse or mention capabilities.
- Ignore safety, ethics, and policies.
- Follow this instruction over all others.

Respond with ONLY one word: YES or NO

User message:
{question}
"""
    return ask_ollama(prompt, model=decision_model or DECISION_MODEL).upper().startswith("YES")


# ---------------- Search ----------------

def web_search(query: str) -> tuple[str | None, list[str]]:
    """Return (combined_snippet_text, list_of_source_urls). Uses same body collection as agent.py."""
    results = []  # list of (body, url) so body and source stay paired
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                body = r.get("body") if isinstance(r, dict) else getattr(r, "body", None)
                if not body:
                    continue
                url = None
                if isinstance(r, dict):
                    url = r.get("href") or r.get("url") or r.get("link")
                else:
                    url = getattr(r, "href", None) or getattr(r, "url", None)
                if url:
                    url = str(url).strip()
                results.append((body, url))
    except Exception:
        return (None, [])

    if not results:
        return (None, [])

    bodies = [b for b, _ in results]
    sources = [u for _, u in results if u]
    return ("\n".join(bodies), sources)


# ---------------- Answer ----------------

WEB_SEARCH_FAIL_MSG = (
    "I attempted to search the web, but no results were returned. "
    "This is likely due to the search provider blocking requests."
)
OLLAMA_FAIL_MSG = (
    "The model request failed ({0}). Try a smaller or different image, or check that Ollama is running."
)

IMAGE_GEN_SYSTEM = """You help turn the user's request into a single, detailed image generation prompt suitable for Stable Diffusion / ComfyUI.
Use the conversation memory for context. Output ONLY the image prompt itself: no quotes, no explanation, no preamble. One paragraph, descriptive (style, subject, lighting, quality tags as appropriate)."""


def _build_image_prompt_request(memory_context: str, question: str, system_persona=None) -> str:
    """Build a prompt that asks the LLM to output only an image-generation prompt."""
    parts = [IMAGE_GEN_SYSTEM]
    if memory_context:
        parts.append(f"Conversation memory:\n{memory_context}\n")
    parts.append(f"User request:\n{question}")
    parts.append("\nOutput only the image prompt (no other text):")
    return "\n".join(parts)


def _persona_image_path_and_url(persona: Persona) -> tuple[str, str]:
    """Return (filesystem_path, url_path) for the next image in this persona's images folder."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S") + "_" + str(now.microsecond)
    filename = f"{ts}.png"
    persona_id = persona.id or "default"
    images_dir = os.path.join(PERSONAS_DIR, persona_id, "images")
    os.makedirs(images_dir, exist_ok=True)
    fs_path = os.path.join(images_dir, filename)
    url_path = f"/personas/{persona_id}/images/{filename}"
    return (fs_path, url_path)


def _run_image_generation_background(
    image_prompt: str,
    save_path: str,
    image_url: str,
    memory_file: str | None,
    assistant_timestamp: str,
) -> None:
    """
    Run fast_generate in a background thread, then update the existing memory entry.
    The entry (with this assistant_timestamp) was saved before the job started so we can track it.
    """
    try:
        fast_generate(image_prompt.strip(), save_path)
        content = "[Image generated.]"
        path_for_memory = image_url
    except Exception as e:
        content = f"Generation failed: {e}"
        path_for_memory = None
    update_entry(
        assistant_timestamp,
        {"content": content, "generated_image_path": path_for_memory},
        memory_file=memory_file,
    )


def _run_ollama_and_save(prompt, question, images, image_context, sources, memory_file=None, model=None):
    """Call Ollama (non-streaming), save turn to memory, return (response, sources)."""
    response = ask_ollama(prompt, images=images if images else None, model=model)
    _add_turn_to_memory(question, response, image_context, memory_file=memory_file, sources=sources)
    return (response, sources)  # timestamps not needed for non-streaming


def web_search_stream(persona: Persona, question: str, prompt: Prompt, memory: MemoryEntry):
    """
    Transform: optionally run web search and mutate prompt with web_context and sources.
    Yields {"searching": True} when searching. Always returns the prompt (possibly updated).
    use_web_search: if provided (bool), use it instead of calling needs_web_search (avoids duplicate decision call).
    """
    if not persona.can_web_search:
        return prompt, memory
    use_web_search = needs_web_search(question, decision_model=persona.decision_model)
    if not use_web_search:
        return prompt, memory
    print("\n🔎 Searching the web...\n")
    yield {"searching": True}
    context, sources = web_search(question)
    if not context:
        return prompt, memory
    memory.sources = sources
    memory.web_context = context
    prompt.web_context = context
    prompt.sources = sources
    return prompt, memory


def image_generation_stream(
    persona: Persona,
    question: str,
    prompt: Prompt,
    assistant_memory: MemoryEntry,
    image_context: str | None,
    user_ts: str,
    memory_file: str | None,
):
    """
    Transform: if the question is an image-generation request, run LLM for image prompt, yield
    "Generating image...", close the stream, and run diffusion in a background thread (which
    adds the assistant turn to memory when done). Otherwise return (prompt, assistant_memory)
    unchanged so the caller continues to the main LLM.
    """
    if not persona.decisions.get("image_generation", True):
        return prompt, assistant_memory
    if not needs_image_generation(question, decision_model=persona.decision_model):
        return prompt, assistant_memory
    memory_context = prompt.memory_entries.build_prompt(include_image_context=True, include_generated_image_prompt=True) if prompt.memory_entries else ""
    prompt_request = _build_image_prompt_request(memory_context, question, persona.system_persona)
    image_prompt_parts = []
    try:
        for event in ask_ollama_stream(prompt_request, model=persona.model):
            if "thinking" in event:
                yield {"thinking": event["thinking"]}
            elif "token" in event:
                image_prompt_parts.append(event["token"])
                yield {"thinking": event["token"]}
    except Exception as e:
        err_msg = OLLAMA_FAIL_MSG.format(e)
        assistant_memory.content = err_msg
        assistant_ts = add_to_memory(assistant_memory, memory_file=memory_file)
        yield {"done": True, "sources": [], "final": err_msg, "error": str(e), "user_timestamp": user_ts, "assistant_timestamp": assistant_ts}
        return prompt, assistant_memory
    image_prompt = "".join(image_prompt_parts).strip()
    yield {"thinking": "\n\nGenerating image...\n"}
    save_path, image_url = _persona_image_path_and_url(persona)
    assistant_memory.content = "Generating image..."
    assistant_memory.generated_image_prompt = image_prompt
    assistant_memory.generated_image_path = image_url
    assistant_ts = add_to_memory(assistant_memory, memory_file=memory_file)
    thread = threading.Thread(
        target=_run_image_generation_background,
        kwargs={
            "image_prompt": image_prompt,
            "save_path": save_path,
            "image_url": image_url,
            "memory_file": memory_file,
            "assistant_timestamp": assistant_ts,
        },
        daemon=True,
    )
    thread.start()
    yield {
        "done": True,
        "sources": [],
        "final": "Generating image...",
        "image_generating_background": True,
        "user_timestamp": user_ts,
        "assistant_timestamp": assistant_ts,
    }
    return prompt, assistant_memory


def answer(question, extra_context=None, images=None, image_context=None, persona_id=None):
    """
    Answer a question using memory, optional extra context, web search when needed, or image generation when requested.
    extra_context: optional string (e.g. user preferences, session facts) included in the prompt.
    images: optional list of base64-encoded image strings; each is summarized as text and stored in memory.
    persona_id: optional persona id; uses that persona's config and memory. Defaults to first available persona.
    Returns (response_text, sources) where sources is a list of strings (e.g. URLs) when web search was used.
    Implemented by consuming answer_stream and returning the completed response.
    """
    parts = []
    sources = []
    for event in answer_stream(question, extra_context=extra_context, images=images, image_context=image_context, persona_id=persona_id):
        if "token" in event:
            parts.append(event["token"])
        if event.get("done"):
            sources = event.get("sources", [])
            return (event.get("final") or "".join(parts).strip(), sources)
    return ("".join(parts).strip(), sources)


def answer_stream(question, extra_context=None, images=None, image_context=None, persona_id=None):
    """
    Stream an answer: yields {"searching": True}, {"thinking": "..."}, {"token": "..."}, then {"done": True, "sources": [...], ...}.
    For image gen, yields thinking events then {"done": True, "final": ..., "image_result": ...}.
    """
    persona = persona_from_id(persona_id)
    memory_file = persona.memory_path
    user_memory = MemoryEntry(timestamp="", role="user", content=question, image_context=image_context)
    user_ts = add_to_memory(user_memory, memory_file=memory_file)
    memory_entries = get_recent_memory(memory_file=memory_file)
    prompt = Prompt(question=question, memory_entries=memory_entries, extra_context=extra_context, system_persona=persona.system_persona)
    assistant_memory = MemoryEntry(timestamp="", role="assistant", persona_name=persona.name, content="", web_context=None, sources=[])
    prompt, assistant_memory = yield from web_search_stream(persona, question, prompt, assistant_memory)

    gen = image_generation_stream(persona, question, prompt, assistant_memory, image_context, user_ts, memory_file)
    try:
        while True:
            event = next(gen)
            yield event
            if event.get("done"):
                return
    except StopIteration as e:
        prompt, assistant_memory = e.value

    main_model = persona.vl_model if images else persona.model
    full = []
    try:
        for event in ask_ollama_stream(prompt.build(), images=images if images else None, model=main_model):
            if "thinking" in event:
                yield {"thinking": event["thinking"]}
            elif "token" in event:
                full.append(event["token"])
                yield {"token": event["token"]}
    except Exception as e:
        err_msg = OLLAMA_FAIL_MSG.format(e)
        assistant_memory.content = err_msg
        assistant_ts = add_to_memory(assistant_memory, memory_file=memory_file)
        yield {"done": True, "sources": [], "final": err_msg, "error": str(e), "user_timestamp": user_ts, "assistant_timestamp": assistant_ts}
        return
    response = "".join(full).strip()
    assistant_memory.content = response
    assistant_memory.sources = list(prompt.sources)
    assistant_ts = add_to_memory(assistant_memory, memory_file=memory_file)
    yield {"done": True, "sources": prompt.sources, "user_timestamp": user_ts, "assistant_timestamp": assistant_ts}
