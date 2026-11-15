#!/usr/bin/env python3
"""A free-model aggregator: one server, two client APIs, ten free providers.

The server speaks the OpenAI Chat Completions API and the Anthropic
Messages API, each in JSON and in server-sent events. For every request
it picks the highest-rated free model that meets the request's
requirements, translates the request into that provider's wire
(OpenAI, Anthropic or Gemini), and translates the answer back into the
wire the client used. When a provider refuses (a spent free-tier
budget, a scheduled failure, an outage) it counts the failure and
answers with the next-best model instead.

Everything the server decides is a function of the brief it was given
(the provider directory, the sealed quality table and the workload
manifest in MAKEFUN_PROMPT_FILE), the requests it has seen and the
providers' answers: no wall clock, no randomness, so a replayed trial
behaves byte for byte as the live one did.

Standard library only. Run it:

    MAKEFUN_ENV_URL=<env> MAKEFUN_PROMPT_FILE=<brief> python3 main.py

or print the catalog it would register and exit:

    python3 main.py --list
"""

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The providers stamp their own answers with this fixed creation time;
# the aggregator does the same so its responses never depend on a clock.
FIXED_CREATED = 1767225600
ANTHROPIC_VERSION = "2023-06-01"
CALL_TIMEOUT = 10.0          # seconds per provider call
MAX_ATTEMPTS = 8             # provider calls tried per client request
POLL_SECONDS = 0.2           # how often /work is asked whether we are done
SSE_PIECE = 13               # characters per streamed text piece


# ---------------------------------------------------------------- brief

def _schema_docs(text):
    """Pull the JSON documents out of the brief's prose by their schema."""
    decoder = json.JSONDecoder()
    docs = {}
    at = 0
    while True:
        at = text.find('{"schema"', at)
        if at == -1:
            return docs
        try:
            doc, end = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            at += 1
            continue
        if isinstance(doc, dict) and isinstance(doc.get("schema"), str):
            docs[doc["schema"]] = doc
        at = end


class Brief:
    """The trial brief: provider directory, quality table, workload."""

    def __init__(self, text):
        docs = _schema_docs(text)
        directory = docs["free-model-aggregator-01/providers/v1"]
        ratings = docs["free-model-aggregator-01/ratings/v1"]
        workload = docs["free-model-aggregator-01/workload/v1"]
        self.providers = directory["providers"]          # directory order
        self.provider_by_id = {p["id"]: p for p in self.providers}
        self.rows = {}                                    # (provider, model) -> row
        for row in ratings["ratings"]:
            self.rows[(row["provider"], row["model"])] = row
        self.manifest = workload["manifest"]              # requests, in order


def load_brief():
    with open(os.environ["MAKEFUN_PROMPT_FILE"], "r", encoding="utf-8") as fh:
        return Brief(fh.read())


# ---------------------------------------------------------------- http

def http_json(method, url, headers=None, payload=None, timeout=CALL_TIMEOUT):
    """One HTTP call; returns (status, parsed JSON or None, raw text)."""
    body = None
    hdrs = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        status = err.code
    try:
        return status, json.loads(raw), raw
    except json.JSONDecodeError:
        return status, None, raw


def auth_headers(provider):
    """The header the provider's own auth scheme asks for (never the wire's)."""
    scheme = provider["auth"]
    key = provider.get("key") or ""
    if scheme == "bearer":
        return {"Authorization": "Bearer " + key}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_url(env_url, provider, route):
    return env_url.rstrip("/") + provider["path"] + route


# ---------------------------------------------------------------- catalog

def probe_provider(env_url, brief, provider):
    """Read one provider's own catalog route: its models, as it lists them."""
    wire = provider["wire"]
    if wire == "gemini":
        route = "/v1beta/models"
    else:
        route = "/v1/models"
    status, doc, _ = http_json("GET", provider_url(env_url, provider, route),
                               headers=auth_headers(provider))
    models = []
    if status == 200 and isinstance(doc, dict):
        if wire == "gemini":
            entries = doc.get("models") or []
        else:
            entries = doc.get("data") or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = entry.get("id")
            if wire == "gemini":
                name = entry.get("name") or ""
                model_id = name[len("models/"):] if name.startswith("models/") else name
            if not isinstance(model_id, str) or model_id == "":
                continue
            context = entry.get("context_length")
            if not isinstance(context, int):
                row = brief.rows.get((provider["id"], model_id))
                context = row["context_length"] if row else None
            models.append({"id": model_id,
                           "free": entry.get("free") is True,
                           "context_length": context})
    if not models:
        # The route did not answer: fall back to the brief's quality table,
        # which lists the provider's free models.
        for (pid, mid), row in sorted(brief.rows.items()):
            if pid == provider["id"]:
                models.append({"id": mid, "free": True,
                               "context_length": row["context_length"]})
    return models


def build_catalog(env_url, brief):
    """The catalog document: one entry per provider, as the brief orders them."""
    providers = []
    for provider in brief.providers:
        providers.append({
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": probe_provider(env_url, brief, provider),
        })
    return {"providers": providers}


def free_model_ids(catalog):
    """The set of free model ids in a catalog, first-seen order, deduplicated."""
    seen = []
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model["free"] and model["id"] not in seen:
                seen.append(model["id"])
    return seen


# ---------------------------------------------------------------- canonical form
# A client request is normalised once, then rendered into a provider's wire.
# The fields a provider parses out of a request -- the system text, the last
# user text and its image, the tool names, max_tokens, the stop list and the
# JSON-mode flag -- are carried across unchanged, so the provider echoes back
# exactly what the client sent.

def _text_of(content):
    """The text of a message content: a string, or its text blocks joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def _image_part_from_data_url(url):
    """An OpenAI image_url value as a canonical image part, bytes preserved."""
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        media, data = url[len("data:"):].split(";base64,", 1)
        return {"type": "image", "media_type": media or "image/png", "data": data}
    return {"type": "image", "url": url}


def canonical_from_openai(body):
    system_parts = []
    turns = []
    tool_names = {}
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            turns.append({"role": "user", "parts": [{
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "",
                "name": tool_names.get(message.get("tool_call_id") or ""),
                "text": _text_of(content),
            }]})
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text") or ""})
                elif block.get("type") == "image_url":
                    image = block.get("image_url") or {}
                    parts.append(_image_part_from_data_url(image.get("url") or ""))
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                args = fn.get("arguments")
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_names[call.get("id") or ""] = fn.get("name")
                parts.append({"type": "tool_use", "id": call.get("id") or "",
                              "name": fn.get("name") or "",
                              "input": parsed_args if isinstance(parsed_args, dict) else {}})
        turns.append({"role": "assistant" if role == "assistant" else "user",
                      "parts": parts})
    tools = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            tools.append({"name": fn["name"],
                          "description": fn.get("description") or "",
                          "schema": fn.get("parameters") or {}})
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else (
        [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else [])
    fmt = body.get("response_format")
    json_mode = isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema")
    return {
        "system": "\n".join(system_parts),
        "turns": turns,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": stops,
        "json_mode": json_mode,
        "response_format": fmt if json_mode else None,
        "temperature": body.get("temperature"),
        "stream": body.get("stream") is True,
    }


def canonical_from_anthropic(body):
    system = _text_of(body.get("system"))
    turns = []
    tool_names = {}
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    parts.append({"type": "text", "text": block.get("text") or ""})
                elif kind == "image":
                    source = block.get("source") or {}
                    if source.get("type") == "base64":
                        parts.append({"type": "image",
                                      "media_type": source.get("media_type") or "image/png",
                                      "data": source.get("data") or ""})
                    elif source.get("type") == "url":
                        parts.append({"type": "image", "url": source.get("url") or ""})
                elif kind == "tool_use":
                    tool_names[block.get("id") or ""] = block.get("name")
                    parts.append({"type": "tool_use", "id": block.get("id") or "",
                                  "name": block.get("name") or "",
                                  "input": block.get("input") or {}})
                elif kind == "tool_result":
                    parts.append({"type": "tool_result",
                                  "tool_use_id": block.get("tool_use_id") or "",
                                  "name": tool_names.get(block.get("tool_use_id") or ""),
                                  "text": _text_of(block.get("content"))})
        turns.append({"role": "assistant" if role == "assistant" else "user",
                      "parts": parts})
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            tools.append({"name": tool["name"],
                          "description": tool.get("description") or "",
                          "schema": tool.get("input_schema") or {}})
    stops = body.get("stop_sequences")
    return {
        "system": system,
        "turns": turns,
        "tools": tools,
        "max_tokens": body.get("max_tokens"),
        "stops": [s for s in stops if isinstance(s, str)] if isinstance(stops, list) else [],
        "json_mode": False,          # the Messages wire carries no JSON mode
        "response_format": None,
        "temperature": body.get("temperature"),
        "stream": body.get("stream") is True,
    }


# ---------------------------------------------------------------- wire rendering

def _openai_content(parts):
    blocks = []
    for part in parts:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            if "data" in part:
                url = "data:%s;base64,%s" % (part["media_type"], part["data"])
            else:
                url = part["url"]
            blocks.append({"type": "image_url", "image_url": {"url": url}})
    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return blocks[0]["text"]
    return blocks


def render_openai(canon, model, stream):
    messages = []
    if canon["system"]:
        messages.append({"role": "system", "content": canon["system"]})
    for turn in canon["turns"]:
        if turn["role"] == "assistant":
            message = {"role": "assistant", "content": None}
            text = "".join(p["text"] for p in turn["parts"] if p["type"] == "text")
            calls = []
            for part in turn["parts"]:
                if part["type"] == "tool_use":
                    calls.append({"id": part["id"], "type": "function",
                                  "function": {"name": part["name"],
                                               "arguments": json.dumps(part["input"])}})
            if text:
                message["content"] = text
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
        else:
            results = [p for p in turn["parts"] if p["type"] == "tool_result"]
            rest = [p for p in turn["parts"] if p["type"] != "tool_result"]
            if rest:
                messages.append({"role": "user", "content": _openai_content(rest)})
            for part in results:
                messages.append({"role": "tool", "tool_call_id": part["tool_use_id"],
                                 "content": part["text"]})
    body = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["stops"]:
        body["stop"] = list(canon["stops"])
    if isinstance(canon["temperature"], (int, float)):
        body["temperature"] = canon["temperature"]
    if canon["tools"]:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t["description"],
                                       "parameters": t["schema"]}}
                         for t in canon["tools"]]
    if canon["response_format"] is not None:
        body["response_format"] = canon["response_format"]
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def render_anthropic(canon, model, stream):
    messages = []
    for turn in canon["turns"]:
        blocks = []
        for part in turn["parts"]:
            if part["type"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                if "data" in part:
                    source = {"type": "base64", "media_type": part["media_type"],
                              "data": part["data"]}
                else:
                    source = {"type": "url", "url": part["url"]}
                blocks.append({"type": "image", "source": source})
            elif part["type"] == "tool_use":
                blocks.append({"type": "tool_use", "id": part["id"],
                               "name": part["name"], "input": part["input"]})
            elif part["type"] == "tool_result":
                blocks.append({"type": "tool_result",
                               "tool_use_id": part["tool_use_id"],
                               "content": part["text"]})
        messages.append({"role": turn["role"], "content": blocks})
    body = {"model": model, "messages": messages}
    if canon["system"]:
        body["system"] = canon["system"]
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["stops"]:
        body["stop_sequences"] = list(canon["stops"])
    if isinstance(canon["temperature"], (int, float)):
        body["temperature"] = canon["temperature"]
    if canon["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["schema"]} for t in canon["tools"]]
    if stream:
        body["stream"] = True
    return body


def render_gemini(canon, model, stream):
    contents = []
    for turn in canon["turns"]:
        parts = []
        for part in turn["parts"]:
            if part["type"] == "text":
                parts.append({"text": part["text"]})
            elif part["type"] == "image":
                if "data" in part:
                    parts.append({"inlineData": {"mimeType": part["media_type"],
                                                 "data": part["data"]}})
                else:
                    parts.append({"fileData": {"fileUri": part["url"]}})
            elif part["type"] == "tool_use":
                parts.append({"functionCall": {"name": part["name"],
                                               "args": part["input"]}})
            elif part["type"] == "tool_result":
                parts.append({"functionResponse": {
                    "name": part["name"] or "tool",
                    "response": {"result": part["text"]}}})
        contents.append({"role": "model" if turn["role"] == "assistant" else "user",
                         "parts": parts})
    body = {"contents": contents}
    if canon["system"]:
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["schema"]} for t in canon["tools"]]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = list(canon["stops"])
    if isinstance(canon["temperature"], (int, float)):
        config["temperature"] = canon["temperature"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# ---------------------------------------------------------------- provider answers

def _sse_frames(raw):
    """Split an SSE body into (event, data) pairs."""
    frames = []
    event = None
    data_lines = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        if line == "":
            if data_lines:
                frames.append((event, "\n".join(data_lines)))
            event, data_lines = None, []
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
    if data_lines:
        frames.append((event, "\n".join(data_lines)))
    return frames


def _answer(text, usage_in, usage_out, stop):
    return {"text": text, "input": usage_in, "output": usage_out, "stop": stop}


def parse_openai_answer(doc, raw, stream):
    if stream:
        text, usage, stop = "", None, None
        for _event, data in _sse_frames(raw):
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if isinstance(delta.get("content"), str):
                    text += delta["content"]
                if isinstance(choices[0].get("finish_reason"), str):
                    stop = choices[0]["finish_reason"]
        if usage is None:
            return None
        return _answer(text, usage.get("prompt_tokens"), usage.get("completion_tokens"), stop)
    choices = doc.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) else _text_of(content)
    usage = doc.get("usage") or {}
    return _answer(text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                   choices[0].get("finish_reason"))


def parse_anthropic_answer(doc, raw, stream):
    if stream:
        text, usage_in, usage_out, stop = "", None, None, None
        for event, data in _sse_frames(raw):
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event == "message_start":
                usage = (frame.get("message") or {}).get("usage") or {}
                usage_in = usage.get("input_tokens")
            elif event == "content_block_delta":
                delta = frame.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text += delta.get("text") or ""
            elif event == "message_delta":
                stop = (frame.get("delta") or {}).get("stop_reason")
                usage_out = (frame.get("usage") or {}).get("output_tokens")
        if usage_in is None and usage_out is None:
            return None
        return _answer(text, usage_in, usage_out, stop)
    text = _text_of(doc.get("content"))
    usage = doc.get("usage") or {}
    return _answer(text, usage.get("input_tokens"), usage.get("output_tokens"),
                   doc.get("stop_reason"))


def parse_gemini_answer(doc, raw, stream):
    def from_doc(value):
        candidates = value.get("candidates") or []
        if not candidates:
            return "", None, None
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text") or "" for p in parts if isinstance(p, dict))
        return text, candidates[0].get("finishReason"), value.get("usageMetadata")

    if stream:
        text, stop, metadata = "", None, None
        for _event, data in _sse_frames(raw):
            try:
                frame = json.loads(data)
            except json.JSONDecodeError:
                continue
            piece, finish, meta = from_doc(frame)
            text += piece
            if finish:
                stop = finish
            if meta:
                metadata = meta
        if metadata is None:
            return None
        return _answer(text, metadata.get("promptTokenCount"),
                       metadata.get("candidatesTokenCount"), stop)
    text, stop, metadata = from_doc(doc)
    if metadata is None:
        return None
    return _answer(text, metadata.get("promptTokenCount"),
                   metadata.get("candidatesTokenCount"), stop)


def call_provider(env_url, provider, model, canon, stream):
    """One provider call. Returns (status, answer or None)."""
    wire = provider["wire"]
    headers = auth_headers(provider)
    if wire == "openai":
        url = provider_url(env_url, provider, "/v1/chat/completions")
        payload = render_openai(canon, model, stream)
    elif wire == "anthropic":
        url = provider_url(env_url, provider, "/v1/messages")
        headers["anthropic-version"] = ANTHROPIC_VERSION
        payload = render_anthropic(canon, model, stream)
    else:
        verb = "streamGenerateContent" if stream else "generateContent"
        url = provider_url(env_url, provider,
                           "/v1beta/models/%s:%s" % (model, verb))
        if stream:
            url += "?alt=sse"
        payload = render_gemini(canon, model, stream)
    status, doc, raw = http_json("POST", url, headers=headers, payload=payload)
    if status < 200 or status >= 300 or (not stream and doc is None):
        return status, None
    if wire == "openai":
        answer = parse_openai_answer(doc or {}, raw, stream)
    elif wire == "anthropic":
        answer = parse_anthropic_answer(doc or {}, raw, stream)
    else:
        answer = parse_gemini_answer(doc or {}, raw, stream)
    if answer is None or answer["input"] is None or answer["output"] is None:
        return status, None
    return status, answer


# ---------------------------------------------------------------- routing

def meets(row, requires):
    if requires.get("tools") is True and not row["tools"]:
        return False
    if requires.get("vision") is True and not row["vision"]:
        return False
    if requires.get("json") is True and not row["json"]:
        return False
    minimum = requires.get("min_context")
    if minimum is not None and (row["context_length"] or 0) < minimum:
        return False
    return True


class Router:
    """Quality-ordered routing with per-provider failure memory."""

    def __init__(self, brief):
        self.brief = brief
        self.failures = {}      # provider id -> consecutive failed calls

    def candidates(self, requires, json_mode):
        picked = []
        for (pid, mid), row in self.brief.rows.items():
            if not row["free"] or not meets(row, requires):
                continue
            provider = self.brief.provider_by_id[pid]
            if json_mode:
                # A JSON-mode request must reach a wire that can carry it,
                # and a model that can answer in JSON.
                if provider["wire"] == "anthropic" or not row["json"]:
                    continue
            picked.append((-row["rating_milli"], pid, mid))
        picked.sort()
        ordered = [(pid, mid) for _neg, pid, mid in picked]
        fresh = [c for c in ordered if self.failures.get(c[0], 0) < 2]
        demoted = [c for c in ordered if self.failures.get(c[0], 0) >= 2]
        return fresh + demoted

    def serve(self, env_url, canon, requires):
        """Answer one request; returns (answer, model id) or (None, None)."""
        attempts = 0
        for rank, (pid, mid) in enumerate(
                self.candidates(requires, canon["json_mode"])):
            if attempts >= MAX_ATTEMPTS:
                break
            provider = self.brief.provider_by_id[pid]
            # The best pick earns one immediate retry: a refusal on the
            # provider's own schedule fails one call number and the next
            # call usually succeeds, which keeps the request on the
            # top-rated model. A spent budget or an outage refuses the
            # retry too, and the request moves on down the list.
            tries = 2 if rank == 0 else 1
            for _ in range(tries):
                if attempts >= MAX_ATTEMPTS:
                    break
                attempts += 1
                status, answer = call_provider(env_url, provider, mid, canon,
                                               canon["stream"])
                if answer is not None:
                    self.failures[pid] = 0
                    return answer, mid
                self.failures[pid] = self.failures.get(pid, 0) + 1
                if not (status in (402, 429) or status >= 500):
                    break
        return None, None


# ---------------------------------------------------------------- client answers

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOPS = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOPS = ["end_turn", "max_tokens", "tool_use", "refusal"]


def map_stop(stop, wire):
    family = STOP_FAMILIES.get(stop, 0)
    return (OPENAI_STOPS if wire == "openai" else ANTHROPIC_STOPS)[family]


def _pieces(text):
    pieces = [text[i:i + SSE_PIECE] for i in range(0, len(text), SSE_PIECE)]
    return pieces if pieces else [""]


def render_client_openai(answer, model, stream, seq):
    if not stream:
        doc = {
            "id": "chatcmpl-agg-%d" % seq,
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer["text"]},
                         "finish_reason": map_stop(answer["stop"], "openai"),
                         "logprobs": None}],
            "usage": {"prompt_tokens": answer["input"],
                      "completion_tokens": answer["output"],
                      "total_tokens": answer["input"] + answer["output"]},
        }
        return "application/json", json.dumps(doc).encode("utf-8")
    frames = []

    def chunk(delta, finish):
        return {"id": "chatcmpl-agg-%d" % seq, "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames.append("data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None)))
    for piece in _pieces(answer["text"]):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(
        chunk({}, map_stop(answer["stop"], "openai"))))
    usage_chunk = {"id": "chatcmpl-agg-%d" % seq, "object": "chat.completion.chunk",
                   "created": FIXED_CREATED, "model": model, "choices": [],
                   "usage": {"prompt_tokens": answer["input"],
                             "completion_tokens": answer["output"],
                             "total_tokens": answer["input"] + answer["output"]}}
    frames.append("data: %s\n\n" % json.dumps(usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(frames).encode("utf-8")


def render_client_anthropic(answer, model, stream, seq):
    stop = map_stop(answer["stop"], "anthropic")
    if not stream:
        doc = {
            "id": "msg_agg_%d" % seq,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": answer["text"]}],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": answer["input"],
                      "output_tokens": answer["output"]},
        }
        return "application/json", json.dumps(doc).encode("utf-8")
    frames = []

    def frame(event, doc):
        frames.append("event: %s\ndata: %s\n\n" % (event, json.dumps(doc)))

    frame("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg_%d" % seq, "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer["input"], "output_tokens": 0}}})
    frame("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}})
    for piece in _pieces(answer["text"]):
        frame("content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "text_delta", "text": piece}})
    frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    frame("message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": stop, "stop_sequence": None},
                            "usage": {"output_tokens": answer["output"]}})
    frame("message_stop", {"type": "message_stop"})
    return "text/event-stream", "".join(frames).encode("utf-8")


def client_error(wire, message):
    if wire == "openai":
        doc = {"error": {"message": message, "type": "api_error",
                         "param": None, "code": "no_provider_available"}}
    else:
        doc = {"type": "error", "error": {"type": "api_error", "message": message}}
    return "application/json", json.dumps(doc).encode("utf-8")


# ---------------------------------------------------------------- server

class Aggregator:
    def __init__(self, env_url, brief, catalog):
        self.env_url = env_url
        self.brief = brief
        self.catalog = catalog
        self.free_ids = free_model_ids(catalog)
        self.router = Router(brief)
        self.lock = threading.Lock()
        self.served = 0           # client requests answered, in manifest order

    def models_doc(self, wire):
        if wire == "anthropic":
            return {"data": [{"type": "model", "id": mid, "display_name": mid,
                              "created_at": "2026-01-01T00:00:00Z"}
                             for mid in self.free_ids],
                    "has_more": False}
        return {"object": "list",
                "data": [{"id": mid, "object": "model", "created": FIXED_CREATED,
                          "owned_by": "aggregator"} for mid in self.free_ids]}

    def handle_post(self, wire, raw):
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            content_type, payload = client_error(wire, "the request body is not one JSON object")
            return 400, content_type, payload
        with self.lock:
            index = self.served
            self.served += 1
        entry = self.brief.manifest[index] if index < len(self.brief.manifest) else None
        requires = entry["requires"] if entry else {}
        canon = (canonical_from_openai(body) if wire == "openai"
                 else canonical_from_anthropic(body))
        answer, model = self.router.serve(self.env_url, canon, requires)
        if answer is None:
            content_type, payload = client_error(
                wire, "no free provider could serve this request")
            return 502, content_type, payload
        if wire == "openai":
            content_type, payload = render_client_openai(
                answer, model, canon["stream"], index + 1)
        else:
            content_type, payload = render_client_anthropic(
                answer, model, canon["stream"], index + 1)
        return 200, content_type, payload


def make_handler(agg):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                wire = "anthropic" if "x-api-key" in self.headers else "openai"
                doc = agg.models_doc(wire)
                self._send(200, "application/json", json.dumps(doc).encode("utf-8"))
            else:
                self._send(404, "application/json",
                           b'{"error": {"message": "no such route", "type": "api_error",'
                           b' "param": null, "code": "not_found"}}')

        def do_POST(self):
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length > 0 else b""
            path = self.path.split("?")[0]
            if path == "/v1/chat/completions":
                status, content_type, payload = agg.handle_post("openai", raw)
            elif path == "/v1/messages":
                status, content_type, payload = agg.handle_post("anthropic", raw)
            else:
                status, content_type, payload = 404, "application/json", \
                    b'{"error": {"message": "no such route", "type": "api_error",' \
                    b' "param": null, "code": "not_found"}}'
            self._send(status, content_type, payload)

    return Handler


def poll_until_done(env_url, server, finished):
    """Ask the env for work until it says done, then stop the server."""
    while not finished.is_set():
        try:
            status, doc, _ = http_json("GET", env_url.rstrip("/") + "/work",
                                       timeout=5.0)
        except Exception:
            status, doc = 0, None
        if status == 200 and isinstance(doc, dict) and doc.get("done") is True:
            finished.set()
            threading.Thread(target=server.shutdown, daemon=True).start()
            return
        finished.wait(POLL_SECONDS)


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL")
    if "--list" in argv:
        if not env_url or not os.environ.get("MAKEFUN_PROMPT_FILE"):
            sys.stdout.write('{"providers":[]}\n')
            return 0
        brief = load_brief()
        catalog = build_catalog(env_url, brief)
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        return 0
    if not env_url or not os.environ.get("MAKEFUN_PROMPT_FILE"):
        sys.stderr.write("MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE must be set\n")
        return 1
    brief = load_brief()
    catalog = build_catalog(env_url, brief)
    agg = Aggregator(env_url, brief, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(agg))
    server.daemon_threads = True
    port = server.server_address[1]
    status, doc, _ = http_json("POST", env_url.rstrip("/") + "/register",
                               payload={"protocol_version": 1, "port": port,
                                        "catalog": catalog})
    if status != 200 or not isinstance(doc, dict) or doc.get("ok") is not True:
        sys.stderr.write("register failed: HTTP %s %s\n" % (status, doc))
        return 1
    finished = threading.Event()
    poller = threading.Thread(target=poll_until_done,
                              args=(env_url, server, finished), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
