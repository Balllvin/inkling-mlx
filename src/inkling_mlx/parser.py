"""Inkling 2 prompt normalization, chat-template rendering, and incremental tool/thinking parser."""

from __future__ import annotations

import json
import re
from typing import Any

MSG_MODEL = "<|message_model|>"
CONTENT_TEXT = "<|content_text|>"
CONTENT_THINK = "<|content_thinking|>"
CONTENT_TOOL_JSON = "<|content_invoke_tool_json|>"
CONTENT_TOOL_TEXT = "<|content_invoke_tool_text|>"
CONTENT_TOOL_ERROR = "<|content_tool_error|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
ENDOFTEXT = "<|endoftext|>"

STOP_TOKENS = frozenset({END_SAMPLING, ENDOFTEXT})
STOP_IDS = frozenset({200006, 199999})

_EFFORT_ALIASES = {
    "off": "none",
    "none": "none",
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _flatten_content(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        has_image = any(
            isinstance(p, dict) and p.get("type") in ("image", "input_image", "image_url")
            for p in content
        )
        if has_image:
            return content
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return str(content)


def _parse_args(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI chat messages for Inkling 2's chat_template.jinja.

    Key template invariants enforced here:
    - 'developer' role maps to 'system'.
    - Assistant tool_calls[].function.arguments MUST be a dict (parsed object), never a JSON string.
    - Tool messages keep tool_call_id and/or name so the template resolves the header name.
    """
    if not isinstance(messages, list) or not messages:
        raise RequestError(400, "messages must be a non-empty list")

    call_names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise RequestError(400, "each message must be an object")
        role = msg.get("role")
        if role == "developer":
            role = "system"
        if role not in ("system", "user", "assistant", "tool"):
            raise RequestError(400, f"unsupported message role: {role!r}")

        norm: dict[str, Any] = {
            "role": role,
            "content": _flatten_content(msg.get("content")),
        }
        if isinstance(msg.get("name"), str) and msg["name"]:
            norm["name"] = msg["name"]

        if role == "assistant":
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc:
                norm["reasoning_content"] = rc
            raw_calls = msg.get("tool_calls")
            if isinstance(raw_calls, list) and raw_calls:
                calls = []
                for tc in raw_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                    name = fn.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        call_names[tc_id] = name
                    calls.append({
                        "type": "function",
                        "id": tc_id if isinstance(tc_id, str) else "",
                        "function": {
                            "name": name,
                            "arguments": _parse_args(fn.get("arguments")),
                        },
                    })
                if calls:
                    norm["tool_calls"] = calls
                    if "name" in norm and len(calls) == 1 and norm["name"] != calls[0]["function"]["name"]:
                        del norm["name"]
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if isinstance(tc_id, str) and tc_id:
                norm["tool_call_id"] = tc_id
                if "name" not in norm and tc_id in call_names:
                    norm["name"] = call_names[tc_id]
        out.append(norm)
    return out


def normalize_tools(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise RequestError(400, "tools must be a list")
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            raise RequestError(400, "each tool must be an object")
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            raise RequestError(400, "each tool must have a function.name")
        out.append(t)
    return out


def resolve_reasoning_effort(body: dict[str, Any]) -> str | float:
    raw = body.get("reasoning_effort")
    if raw is None and isinstance(body.get("reasoning"), dict):
        raw = body["reasoning"].get("effort")
    if raw is None and isinstance(body.get("chat_template_kwargs"), dict):
        ctk = body["chat_template_kwargs"]
        if "reasoning_effort" in ctk:
            raw = ctk["reasoning_effort"]
        elif ctk.get("enable_thinking") is False:
            raw = "none"
    if raw is None and body.get("enable_thinking") is False:
        raw = "none"
    if raw is None:
        return 0.9
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in _EFFORT_ALIASES:
            return _EFFORT_ALIASES[key]
        try:
            val = float(key)
            return max(0.0, min(0.99, val))
        except ValueError as exc:
            raise RequestError(400, f"Unknown reasoning_effort: {raw!r}") from exc
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return max(0.0, min(0.99, float(raw)))
    raise RequestError(400, f"Invalid reasoning_effort: {raw!r}")


def render_prompt_text(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    reasoning_effort: str | float,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_effort": reasoning_effort,
    }
    if tools:
        kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(text, str):
        raise RequestError(500, "chat_template did not return a string")
    return text


def render_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    reasoning_effort: str | float,
) -> list[int]:
    text = render_prompt_text(tokenizer, messages, tools, reasoning_effort)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"] if isinstance(ids, dict) else ids.input_ids
    return list(ids)


def _partial_suffix_len(buf: str, markers: tuple[str, ...]) -> int:
    best = 0
    for marker in markers:
        limit = min(len(buf), len(marker) - 1)
        for k in range(limit, best, -1):
            if buf.endswith(marker[:k]):
                best = k
                break
    return best


def _parse_invoke_json(recipient: str, payload: str) -> dict[str, Any] | None:
    text = payload.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or recipient.strip()
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("args", obj.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "arguments": args}


class InklingStreamParser:
    """Incremental parser for Inkling 2 block stream.

    Segments arrive as:
      <|message_model|><|content_thinking|>...<|end_message|>
      <|message_model|><|content_text|>...<|end_message|>
      <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
      <|content_model_end_sampling|>
    """

    _CONTENT_MARKERS = (
        CONTENT_THINK,
        CONTENT_TEXT,
        CONTENT_TOOL_JSON,
        CONTENT_TOOL_TEXT,
        CONTENT_TOOL_ERROR,
    )
    _STOP_OR_NEXT = (END_MESSAGE, END_SAMPLING, ENDOFTEXT, MSG_MODEL)

    def __init__(self, tool_id_prefix: str = "call_inkling_") -> None:
        self._buf = ""
        self._state = "seek_header"
        self._recipient = ""
        self._tool_buf = ""
        self.reasoning_content = ""
        self.content = ""
        self.tool_calls: list[dict[str, Any]] = []
        self.malformed = False
        self.stopped = False
        self._next_id = 1
        self._prefix = tool_id_prefix

    def feed(self, piece: str) -> list[dict[str, Any]]:
        if self.stopped:
            return []
        if piece:
            self._buf += piece
        events: list[dict[str, Any]] = []

        while self._buf and not self.stopped:
            if self._state == "seek_header":
                stop_pos = self._find_earliest(self._buf, (END_SAMPLING, ENDOFTEXT))
                hdr_pos = self._buf.find(MSG_MODEL)
                if stop_pos is not None and (hdr_pos < 0 or stop_pos < hdr_pos):
                    pre = self._buf[:stop_pos]
                    if pre:
                        self._emit_text(pre, events)
                    self._buf = ""
                    self.stopped = True
                    break
                if hdr_pos >= 0:
                    pre = self._buf[:hdr_pos]
                    if pre:
                        if any(m in pre for m in self._CONTENT_MARKERS):
                            self._state = "in_header"
                            continue
                        self._emit_text(pre, events)
                    self._buf = self._buf[hdr_pos + len(MSG_MODEL) :]
                    self._recipient = ""
                    self._state = "in_header"
                    continue
                mk_pos = self._find_earliest(self._buf, self._CONTENT_MARKERS)
                if mk_pos is not None:
                    self._recipient = self._buf[:mk_pos]
                    self._buf = self._buf[mk_pos:]
                    self._state = "in_header"
                    continue
                hold = _partial_suffix_len(
                    self._buf, (MSG_MODEL, END_SAMPLING, ENDOFTEXT, *self._CONTENT_MARKERS)
                )
                safe = self._buf[:-hold] if hold else self._buf
                if safe:
                    if "<|" not in safe and len(safe) > 32:
                        self._emit_text(safe, events)
                        self._buf = self._buf[len(safe) :]
                        continue
                break

            if self._state == "in_header":
                stop_pos = self._find_earliest(self._buf, (END_SAMPLING, ENDOFTEXT))
                found_marker = None
                found_pos = None
                for mk in self._CONTENT_MARKERS:
                    p = self._buf.find(mk)
                    if p >= 0 and (found_pos is None or p < found_pos):
                        found_pos = p
                        found_marker = mk
                if stop_pos is not None and (found_pos is None or stop_pos < found_pos):
                    self._buf = ""
                    self.stopped = True
                    break
                if found_marker is not None and found_pos is not None:
                    self._recipient += self._buf[:found_pos]
                    self._buf = self._buf[found_pos + len(found_marker) :]
                    if found_marker == CONTENT_THINK:
                        self._state = "in_think"
                    elif found_marker == CONTENT_TOOL_JSON:
                        self._tool_buf = ""
                        self._state = "in_tool_json"
                    else:
                        self._state = "in_text"
                    continue
                break

            if self._state in ("in_think", "in_text"):
                end_pos = self._buf.find(END_MESSAGE)
                stop_pos = self._find_earliest(self._buf, (END_SAMPLING, ENDOFTEXT))
                next_hdr = self._buf.find(MSG_MODEL)
                cut = None
                cut_kind = None
                for pos, kind, length in (
                    (end_pos, "end", len(END_MESSAGE)),
                    (stop_pos, "stop", 0),
                    (next_hdr, "hdr", 0),
                ):
                    if pos is not None and pos >= 0 and (cut is None or pos < cut):
                        cut = pos
                        cut_kind = (kind, length)
                if cut is not None:
                    body = self._buf[:cut]
                    if body:
                        if self._state == "in_think":
                            self._emit_think(body, events)
                        else:
                            self._emit_text(body, events)
                    kind, length = cut_kind
                    if kind == "end":
                        self._buf = self._buf[cut + length :]
                        self._state = "seek_header"
                    elif kind == "hdr":
                        self._buf = self._buf[cut:]
                        self._state = "seek_header"
                    else:
                        self._buf = ""
                        self.stopped = True
                    continue
                hold = _partial_suffix_len(self._buf, self._STOP_OR_NEXT)
                safe = self._buf[:-hold] if hold else self._buf
                if safe:
                    if self._state == "in_think":
                        self._emit_think(safe, events)
                    else:
                        self._emit_text(safe, events)
                    self._buf = self._buf[len(safe) :]
                break

            if self._state == "in_tool_json":
                end_pos = self._buf.find(END_MESSAGE)
                stop_pos = self._find_earliest(self._buf, (END_SAMPLING, ENDOFTEXT))
                cut = None
                cut_kind = None
                for pos, kind, length in (
                    (end_pos, "end", len(END_MESSAGE)),
                    (stop_pos, "stop", 0),
                ):
                    if pos is not None and pos >= 0 and (cut is None or pos < cut):
                        cut = pos
                        cut_kind = (kind, length)
                if cut is not None:
                    self._tool_buf += self._buf[:cut]
                    kind, length = cut_kind
                    self._finalize_tool(events)
                    if kind == "end":
                        self._buf = self._buf[cut + length :]
                        self._state = "seek_header"
                    else:
                        self._buf = ""
                        self.stopped = True
                    continue
                hold = _partial_suffix_len(self._buf, (END_MESSAGE, END_SAMPLING, ENDOFTEXT))
                safe = self._buf[:-hold] if hold else self._buf
                if safe:
                    self._tool_buf += safe
                    self._buf = self._buf[len(safe) :]
                break

        return events

    def finish(self) -> list[dict[str, Any]]:
        events = self.feed("")
        if self._state == "in_tool_json" and (self._tool_buf or self._buf):
            self._tool_buf += self._buf
            self._buf = ""
            self._finalize_tool(events)
        elif self._buf:
            cleaned = self._buf
            for tok in (END_SAMPLING, ENDOFTEXT, END_MESSAGE):
                cleaned = cleaned.replace(tok, "")
            if cleaned:
                if self._state == "in_think":
                    self._emit_think(cleaned, events)
                elif self._state == "in_text":
                    self._emit_text(cleaned, events)
                elif self._state == "seek_header" and not cleaned.startswith("<|"):
                    self._emit_text(cleaned, events)
            self._buf = ""
        self._state = "done"
        return events

    def _emit_think(self, text: str, events: list[dict[str, Any]]) -> None:
        self.reasoning_content += text
        events.append({"reasoning_content": text})

    def _emit_text(self, text: str, events: list[dict[str, Any]]) -> None:
        self.content += text
        events.append({"content": text})

    def _finalize_tool(self, events: list[dict[str, Any]]) -> None:
        parsed = _parse_invoke_json(self._recipient, self._tool_buf)
        if parsed is None:
            self.malformed = True
            raw = self._tool_buf
            self._tool_buf = ""
            if raw:
                self._emit_text(raw, events)
            return
        idx = len(self.tool_calls)
        call_id = f"{self._prefix}{self._next_id}"
        self._next_id += 1
        wire = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": parsed["name"],
                "arguments": json.dumps(parsed["arguments"], ensure_ascii=False),
            },
        }
        self.tool_calls.append(wire)
        events.append({"tool_calls": [{**wire, "index": idx}]})
        self._tool_buf = ""

    @staticmethod
    def _find_earliest(buf: str, needles: tuple[str, ...]) -> int | None:
        best: int | None = None
        for n in needles:
            pos = buf.find(n)
            if pos >= 0 and (best is None or pos < best):
                best = pos
        return best


def parse_complete_output(raw_text: str) -> dict[str, Any]:
    stream = InklingStreamParser()
    stream.feed(raw_text)
    stream.finish()
    return {
        "content": stream.content,
        "reasoning_content": stream.reasoning_content,
        "tool_calls": stream.tool_calls,
        "malformed": stream.malformed,
    }
