import json
import requests
from ddgs import DDGS
import datetime

from image_context import prepare_images_for_stream, resize_image_for_llm, image_context_for_images
from memory import add_to_memory, get_recent_memory, format_memory_entry
from personas import get_persona_config, get_default_persona_id

OLLAMA_URL = "http://localhost:11434/api/generate"
# Defaults when no persona or persona has no overrides
DECISION_MODEL = "qwen3:0.6b"
MODEL = "qwen3:4b"
VL_MODEL = "qwen3-vl:4b"

SYSTEM_PERSONA = """
You are a helpful AI assistant.
You remain accurate, concise, and calm.
"""

def _persona_settings(persona_id):
    """
    Resolve persona_id to (system_persona, decision_model, model, vl_model, memory_path).
    memory_path is None to use global memory. persona_id None uses default persona.
    """
    pid = persona_id or get_default_persona_id()
    cfg = get_persona_config(pid)
    if not cfg:
        return (SYSTEM_PERSONA, DECISION_MODEL, MODEL, VL_MODEL, None)
    return (
        cfg.get("system_persona") or SYSTEM_PERSONA,
        cfg.get("decision_model") or DECISION_MODEL,
        cfg.get("model") or MODEL,
        cfg.get("vl_model") or VL_MODEL,
        cfg.get("memory_path"),
    )


# ---------------- Ollama ----------------

def _ollama_payload(prompt, stream, images=None, model=None, think=False):
    """Build JSON payload for Ollama /api/generate. images: optional list of base64 strings. model: override (e.g. DECISION_MODEL). think: request thinking/reasoning stream for supported models."""
    if model is None:
        model = VL_MODEL if images else MODEL
    payload = {"model": model, "prompt": prompt, "stream": stream}
    if images:
        payload["images"] = list(images)
    if think:
        payload["think"] = True
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
        json=_ollama_payload(prompt, True, images, model=model, think=True),
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

Respond with ONLY one word: YES or NO
"""
    return ask_ollama(prompt, model=decision_model or DECISION_MODEL).upper().startswith("YES")


def needs_prior_image_context(question, decision_model=None):
    prompt = f"""
You are a classifier.

Question:
{question}

Does answering this question require prior image context from previous images the user has sent?

Respond with ONLY one word:
YES or NO
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


def _build_prompt(memory_context, question, web_context=None, extra_context=None, system_persona=None):
    """Build prompt with optional conversation memory, web context, and extra context."""
    parts = []
    parts.append(system_persona or SYSTEM_PERSONA)
    if memory_context:
        parts.append(f"Conversation memory:\n{memory_context}\n")
    if extra_context:
        parts.append(f"Additional context:\n{extra_context}\n")
    if web_context:
        parts.append(
            "Use the following web information to answer the question. "
            "Base your answer on this information unless the prompt includes an image, then also use the image context.\n\n"
            f"Web information:\n{web_context}\n"
        )
    parts.append(f"Question:\n{question}")
    return "\n".join(parts)


def _add_turn_to_memory(question, response_text, image_context=None, memory_file=None, sources=None):
    """Persist user message and assistant response to short-term memory. Returns (user_ts, assistant_ts)."""
    user_ts = add_to_memory("user", question, image_context=image_context, memory_file=memory_file)
    assistant_ts = add_to_memory(
        "assistant", response_text, memory_file=memory_file, sources=sources or []
    )
    return (user_ts, assistant_ts)


def _run_ollama_and_save(prompt, question, images, image_context, sources, memory_file=None, model=None):
    """Call Ollama (non-streaming), save turn to memory, return (response, sources)."""
    response = ask_ollama(prompt, images=images if images else None, model=model)
    _add_turn_to_memory(question, response, image_context, memory_file=memory_file, sources=sources)
    return (response, sources)  # timestamps not needed for non-streaming


def _stream_ollama_and_save(prompt, question, images, image_context, sources, memory_file=None, model=None):
    """Generator: stream Ollama, save turn to memory, yield thinking/token/done events. Yields error event on failure."""
    full = []
    try:
        for event in ask_ollama_stream(prompt, images=images if images else None, model=model):
            if "thinking" in event:
                yield {"thinking": event["thinking"]}
            elif "token" in event:
                full.append(event["token"])
                yield {"token": event["token"]}
    except Exception as e:
        err_msg = OLLAMA_FAIL_MSG.format(e)
        user_ts, assistant_ts = _add_turn_to_memory(question, err_msg, image_context, memory_file=memory_file, sources=[])
        yield {"done": True, "sources": sources, "error": str(e), "final": err_msg, "user_timestamp": user_ts, "assistant_timestamp": assistant_ts}
        return
    response = "".join(full).strip()
    user_ts, assistant_ts = _add_turn_to_memory(question, response, image_context, memory_file=memory_file, sources=sources)
    yield {"done": True, "sources": sources, "user_timestamp": user_ts, "assistant_timestamp": assistant_ts}


def _prepare_answer(question, images=None, image_context=None, include_prior_image_context=None, memory_file=None):
    """Return (images, image_context, memory_context). Pass include_prior_image_context to avoid calling needs_prior_image_context again."""
    images = images or []
    if image_context is None and images:
        images = [resize_image_for_llm(img) for img in images]
        image_context = image_context_for_images(images)
    elif images:
        images = [resize_image_for_llm(img) for img in images]
    if include_prior_image_context is None:
        include_prior_image_context = not images or needs_prior_image_context(question)
    recent = get_recent_memory(memory_file=memory_file)
    memory_context = "\n".join(format_memory_entry(m, include_prior_image_context) for m in recent)
    return (images, image_context, memory_context)


def _get_prompt_and_sources(question, memory_context, extra_context, log_search=False, use_web_search=None, system_persona=None):
    """
    Run web search if needed and build prompt. Return (prompt, sources, fail_msg).
    fail_msg is non-None only when web search was used and returned no context.
    Pass use_web_search (bool) to avoid calling needs_web_search again.
    """
    if use_web_search is None:
        use_web_search = needs_web_search(question)
    if not use_web_search:
        prompt = _build_prompt(memory_context, question, extra_context=extra_context, system_persona=system_persona)
        return (prompt, [], None)
    if log_search:
        print("\n🔎 Searching the web...\n")
    context, sources = web_search(question)
    if not context:
        return (None, [], WEB_SEARCH_FAIL_MSG)
    prompt = _build_prompt(memory_context, question, web_context=context, extra_context=extra_context, system_persona=system_persona)
    return (prompt, sources, None)


def answer(question, extra_context=None, images=None, persona_id=None):
    """
    Answer a question using memory, optional extra context, and web search when needed.
    extra_context: optional string (e.g. user preferences, session facts) included in the prompt.
    images: optional list of base64-encoded image strings; each is summarized as text and stored in memory.
    persona_id: optional persona id; uses that persona's config and memory. Defaults to first available persona.
    Returns (response_text, sources) where sources is a list of strings (e.g. URLs) when web search was used.
    """
    system_persona, decision_model, model, vl_model, memory_file = _persona_settings(persona_id)
    use_web_search = needs_web_search(question, decision_model=decision_model)
    has_images = bool(images)
    include_prior = not has_images or needs_prior_image_context(question, decision_model=decision_model)
    images, image_context, memory_context = _prepare_answer(
        question, images=images, include_prior_image_context=include_prior, memory_file=memory_file
    )
    prompt, sources, fail = _get_prompt_and_sources(
        question, memory_context, extra_context, log_search=True, use_web_search=use_web_search, system_persona=system_persona
    )
    if fail:
        _add_turn_to_memory(question, fail, image_context, memory_file=memory_file, sources=[])
        return (fail, [])
    main_model = vl_model if has_images else model
    return _run_ollama_and_save(prompt, question, images, image_context, sources, memory_file=memory_file, model=main_model)


def answer_stream(question, extra_context=None, images=None, image_context=None, persona_id=None):
    """
    Stream an answer: yields {"searching": True} when doing web search,
    then {"token": "..."} for each token, then {"done": True, "sources": [...]}.
    images: optional list of base64-encoded image strings (should be already resized if from prepare_images_for_stream).
    image_context: optional pre-computed text summary of images; if None and images given, computed here (can raise).
    persona_id: optional persona id; uses that persona's config and memory.
    """
    system_persona, decision_model, model, vl_model, memory_file = _persona_settings(persona_id)
    use_web_search = needs_web_search(question, decision_model=decision_model)
    if use_web_search:
        yield {"searching": True}

    has_images = bool(images)
    include_prior = not has_images or needs_prior_image_context(question, decision_model=decision_model)
    images, image_context, memory_context = _prepare_answer(
        question, images=images, image_context=image_context, include_prior_image_context=include_prior, memory_file=memory_file
    )
    prompt, sources, fail = _get_prompt_and_sources(
        question, memory_context, extra_context, use_web_search=use_web_search, system_persona=system_persona
    )
    if fail:
        user_ts, assistant_ts = _add_turn_to_memory(question, fail, image_context, memory_file=memory_file, sources=[])
        yield {"done": True, "sources": [], "final": fail, "user_timestamp": user_ts, "assistant_timestamp": assistant_ts}
        return
    main_model = vl_model if has_images else model
    yield from _stream_ollama_and_save(prompt, question, images, image_context, sources, memory_file=memory_file, model=main_model)
