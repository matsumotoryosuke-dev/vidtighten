"""Local-LLM transcript correction — brand-noun mishearing fixes via Ollama.

Optional, manually-triggered pass that asks a local LLM (Ollama) to find places
where a KNOWN-correct proper noun was misheard by speech-to-text into a wrong
but similar-sounding word (e.g. "クウキデザイン" heard as "空気デザイン",
"Claude Code" as "クロドコード").

This is deliberately narrow (see 04_Context/vidtighten-llm-correct.md). A testing
round showed the current local model can do SEEDED brand-noun matching reliably
but is not trustworthy for open-ended grammar/kanji rewriting, so this module
only does the reliable part and every suggestion is anchor-validated before it
can reach the user's review UI. Nothing here mutates transcript state — the
backend only proposes; the frontend applies approved fixes.

Hard-won facts baked in (all verified against gemma4:12b-mlx / Ollama 0.32):
- The model is a *thinking* model; without think=False it reasons for 90+s and
  times out. We always send think=False.
- Ollama's MLX engine ignores the `format` JSON-schema constraint, so we never
  rely on it — output is parsed tolerantly by _repair_json.
- The model consistently emits slightly malformed JSON (doubled opening quote on
  keys `""wrong"`, occasional ```json fences, leading-space keys). _repair_json
  handles all three.
- Direction is only reliable when the model is SEEDED with the known-correct
  names; open-ended discovery inverts the fix. So the prompt always lists the
  canonical names and known wrong→right examples.
"""

from __future__ import annotations

import json
import re

try:
    import requests  # 2.34.2 in ~/.preprod/venv
except ImportError:  # pragma: no cover - urllib fallback keeps the module importable
    requests = None

OLLAMA_BASE_URL = "http://localhost:11434"

# Per-request inference timeout. Brand-noun matching is short (~4s on a 30-min
# transcript in testing), but a cold model load or a slow machine can spike it.
_INFERENCE_TIMEOUT_S = 180
# Health/list endpoints are instant; a short timeout keeps the UI responsive.
_QUICK_TIMEOUT_S = 4


# ── Ollama HTTP (thin, mockable) ─────────────────────────────────────────────

def _get_json(path: str, base_url: str, timeout: float):
    """GET base_url+path and return parsed JSON, or None on any failure."""
    if requests is None:
        return None
    try:
        r = requests.get(f"{base_url}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post_chat(body: dict, base_url: str, timeout: float) -> str | None:
    """POST /api/chat and return the assistant message content, or None."""
    if requests is None:
        return None
    try:
        r = requests.post(f"{base_url}/api/chat", json=body, timeout=timeout)
        r.raise_for_status()
        return (r.json().get("message") or {}).get("content")
    except Exception:
        return None


def ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """True if the Ollama server answers its version endpoint quickly."""
    return _get_json("/api/version", base_url, _QUICK_TIMEOUT_S) is not None


def list_models(base_url: str = OLLAMA_BASE_URL) -> list[dict]:
    """Return installed models as [{name, loaded}], loaded=currently in memory.

    Empty list if Ollama is unreachable — callers distinguish "down" from "no
    models" via ollama_available().
    """
    tags = _get_json("/api/tags", base_url, _QUICK_TIMEOUT_S) or {}
    ps = _get_json("/api/ps", base_url, _QUICK_TIMEOUT_S) or {}
    loaded = {m.get("name") for m in ps.get("models", [])}
    out = []
    for m in tags.get("models", []):
        name = m.get("name")
        if name:
            out.append({"name": name, "loaded": name in loaded})
    return out


def pick_default_model(models: list[dict]) -> str | None:
    """Choose a sensible default from list_models() output.

    Preference order: a currently-loaded model (no load latency, and it's what
    the user last used) → otherwise the first installed model. Returns None when
    nothing is installed. Kept dead-simple on purpose: model names churn (Rio's
    note), so any cleverer heuristic would rot; "what's already warm, else
    anything" is stable and explainable.
    """
    if not models:
        return None
    for m in models:
        if m.get("loaded"):
            return m["name"]
    return models[0]["name"]


# ── Output parsing (tolerant — the model's JSON is reliably malformed) ────────

def _repair_json(raw: str) -> str:
    """Best-effort repair of the model's habitual JSON defects.

    Handles, in order: surrounding prose / markdown ```json fences, the doubled
    opening-quote-on-keys quirk (`""wrong"` → `"wrong"`), and trailing commas.
    Extracts the first balanced {...} or [...] so multi-block replies (the model
    sometimes emits several "revised JSON" blocks) yield the first object.
    """
    if not raw:
        return ""
    s = raw.strip()
    # Collapse any run of 2+ double-quotes to one — the model doubles key-opening
    # quotes (""wrong") and occasionally value quotes; JSON never needs "" except
    # as an empty string, which this would corrupt, so guard that below.
    s = re.sub(r'""+', '"', s)
    # Extract the first JSON container (object preferred, else array).
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return s
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return s[start:]


def parse_suggestions(raw: str) -> list[dict]:
    """Parse the model reply into [{wrong, correct}] as robustly as possible.

    Accepts either {"fixes":[...]} / {"glossary":[...]} or a bare array, and
    either {wrong, correct} or {original, corrected} key spellings. Silently
    drops any entry missing usable keys — a garbled entry is never fatal.
    """
    repaired = _repair_json(raw)
    if not repaired:
        return []
    try:
        data = json.loads(repaired)
    except Exception:
        return []
    if isinstance(data, dict):
        items = data.get("fixes") or data.get("glossary") or data.get("corrections") or []
    elif isinstance(data, list):
        items = data
    else:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        wrong = it.get("wrong", it.get("original"))
        correct = it.get("correct", it.get("corrected"))
        if isinstance(wrong, str) and isinstance(correct, str):
            out.append({"wrong": wrong.strip(), "correct": correct.strip()})
    return out


# ── Anchor-or-drop validation (the core safety gate) ──────────────────────────

def anchor_validate(suggestions: list[dict], transcript_text: str) -> list[dict]:
    """Keep only suggestions that are safe to show the user.

    A suggestion survives iff: `wrong` is non-empty, `wrong != correct`, and
    `wrong` occurs verbatim in transcript_text. Everything else is dropped —
    this single gate neutralises the model's failure modes (no-op "fixes",
    hallucinated spans that aren't in the text, and backwards suggestions when
    it wasn't seeded, since a backwards `wrong` = the already-correct name still
    anchors but is caught by the seed design instead). Adds `count` (occurrences)
    and dedups by (wrong, correct).
    """
    seen = set()
    out = []
    for s in suggestions:
        wrong = s.get("wrong", "")
        correct = s.get("correct", "")
        if not wrong or wrong == correct:
            continue
        count = transcript_text.count(wrong)
        if count == 0:
            continue
        key = (wrong, correct)
        if key in seen:
            continue
        seen.add(key)
        out.append({"wrong": wrong, "correct": correct, "count": count})
    return out


# ── Prompt construction (always seeded — direction is only reliable seeded) ───

def build_prompt(known_names: list[str], known_examples: list[dict],
                 transcript_text: str) -> str:
    """Brand-noun proofreading prompt: seeded PRIVATE names (ground truth) plus
    open discovery of WELL-KNOWN public names.

    Testing (04_Context/vidtighten-llm-correct.md) showed the model reliably
    knows the canonical spelling of famous names (Claude Code, Notion, ChatGPT…)
    and fixes their mishearings in the right direction — but CANNOT know a user's
    private brand (クウキデザイン), where it inverts the fix. So private names are
    supplied as ground truth here; public ones are left to the model's knowledge.
    The review UI (per-row direction-flip + reject) backstops any it gets wrong.

    known_names: private canonical spellings to treat as ground truth.
    known_examples: [{wrong, correct}] few-shot pairs (teaches task + direction).
    """
    lines = [
        "You clean up a speech-to-text transcript (Japanese and/or English).",
        "Speech recognition often mishears PROPER NOUNS (brand, product, company, "
        "or person names) into a wrong but similar-sounding word.",
        "",
        "KNOWN-CORRECT private names — treat their spelling as ground truth and "
        "fix any mishearing OF them:",
    ]
    lines += [f"- {n}" for n in known_names] or ["- (none provided)"]
    if known_examples:
        lines += ["", "Examples of the kind of mishearing to look for:"]
        lines += [f'- "{e["wrong"]}" should be "{e["correct"]}"' for e in known_examples]
    lines += [
        "",
        "ALSO fix mishearings of WELL-KNOWN public names you are confident about "
        "(AI tools, apps, companies, products) — e.g. a misheard spelling of a "
        "famous developer tool, service, or model name.",
        "",
        "RULES:",
        "- Only report a fix when the transcript text DIFFERS from the correct spelling.",
        "- Proper nouns ONLY — never change ordinary words, grammar, punctuation, or casual style.",
        "- If unsure whether something is a real proper noun, do NOT report it.",
        "- If nothing is misheard, return an empty list.",
        "",
        'Output STRICT JSON only, no other text: '
        '{"fixes":[{"wrong":"<exact text in transcript>","correct":"<correct name>"}]}',
        "",
        "Transcript:",
        transcript_text,
    ]
    return "\n".join(lines)


def build_chat_body(prompt: str, model: str) -> dict:
    """The exact Ollama /api/chat body — think=False is mandatory (see module doc)."""
    return {
        "model": model,
        "think": False,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def suggest_brand_corrections(
    transcript_text: str,
    known_names: list[str],
    known_examples: list[dict],
    model: str,
    base_url: str = OLLAMA_BASE_URL,
) -> dict:
    """Run one brand-noun correction pass.

    Returns {status, fixes, error}:
      status "ok"           → fixes = validated [{wrong, correct, count}] (maybe [])
      status "unavailable"  → Ollama not reachable
      status "empty_reply"  → model returned nothing parseable
    Never raises for an inference/parse problem — the un-corrected transcript is
    always a valid product, so failures degrade to "no suggestions".
    """
    if not ollama_available(base_url):
        return {"status": "unavailable", "fixes": [], "error": "Ollama not reachable"}
    prompt = build_prompt(known_names, known_examples, transcript_text)
    raw = _post_chat(build_chat_body(prompt, model), base_url, _INFERENCE_TIMEOUT_S)
    if raw is None:
        return {"status": "empty_reply", "fixes": [], "error": "No response from model"}
    fixes = anchor_validate(parse_suggestions(raw), transcript_text)
    return {"status": "ok", "fixes": fixes, "error": None}
