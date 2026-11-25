"""OpenAI-compatible HTTP server for Inkling 2 Mini on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
import uuid
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx

from .engine import GenRequest, InklingEngine
from .model import load_model
from .mtp import load_mtp_drafter
from .parser import (
    RequestError,
    normalize_messages,
    normalize_tools,
    render_prompt_ids,
    resolve_reasoning_effort,
)


def _allowed_model_names(model_id: str, model_path: Path | None) -> set[str]:
    names = {model_id, "inkling", "inkling-2-mini"}
    if model_path is not None:
        names.update({str(model_path), str(model_path.expanduser().resolve()), model_path.name})
    return names


def _parse_stop_strings(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str) and s]
    return []


def _format_non_stream_response(
    resp_id: str,
    created: int,
    model_name: str,
    res: dict[str, Any],
) -> dict[str, Any]:
    tool_calls = res.get("tool_calls") or []
    content = res.get("content")
    if content is None and not tool_calls:
        content = ""
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if res.get("reasoning_content"):
        msg["reasoning_content"] = res["reasoning_content"]
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": resp_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": res["finish_reason"],
            }
        ],
        "usage": res["usage"],
        "x_metrics": res.get("metrics"),
    }


def make_handler(
    engine: InklingEngine,
    *,
    model_id: str,
    model_path: Path | None = None,
    adapter_id: str | None = None,
    activity_cb: Any = None,
):
    allowed_models = _allowed_model_names(model_id, model_path)
    if adapter_id:
        allowed_models.add(adapter_id)
        allowed_models.add(f"{model_id}+{adapter_id}")
    created_ts = int(time.time())

    class InklingHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise RequestError(400, "invalid Content-Length header") from exc
            if n <= 0:
                raise RequestError(400, "empty request body")
            raw = self.rfile.read(n)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise RequestError(400, f"invalid JSON body: {exc}") from exc
            if not isinstance(parsed, dict):
                raise RequestError(400, "request body must be a JSON object")
            return parsed

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/healthz"):
                pc = engine.prefix_cache
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "model": model_id,
                        "adapter": adapter_id,
                        "ready": True,
                        "mlx_active_gb": round(mx.get_active_memory() / 1e9, 3),
                        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 3),
                        "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 3),
                        "prefix_cache": {
                            "entries": len(pc._entries),
                            "bytes": pc.total_bytes,
                            "hits": pc.hits,
                            "misses": pc.misses,
                        },
                        "mtp": {
                            "enabled": engine.drafter is not None,
                            "spec_draft_k": engine.spec_draft_k,
                            "steps": engine.mtp_total_steps,
                            "accepted": engine.mtp_total_accepted,
                            "mean_acc": (
                                round(engine.mtp_total_accepted / engine.mtp_total_steps, 3)
                                if engine.mtp_total_steps > 0
                                else None
                            ),
                        },
                    },
                )
                return
            if path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": created_ts,
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return
            if path == "/v1/capabilities":
                self._send_json(
                    200,
                    {
                        "streaming": True,
                        "tools": True,
                        "reasoningContent": True,
                        "vision": False,
                        "prefixCache": True,
                        "loops": False,
                        "batching": True,
                        "contextWindow": engine.max_context,
                    },
                )
                return
            self._send_json(404, {"error": {"message": f"unknown endpoint: {path}", "type": "not_found"}})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": f"unknown endpoint: {path}", "type": "not_found"}})
                return

            if activity_cb is not None:
                activity_cb()

            try:
                body = self._read_json()
                req_model = body.get("model")
                if req_model and req_model not in allowed_models:
                    raise RequestError(404, f"unknown model: {req_model!r}")

                messages = normalize_messages(body.get("messages"))
                tools = normalize_tools(body.get("tools"))
                effort = resolve_reasoning_effort(body)
                prompt_ids = render_prompt_ids(engine.tokenizer, messages, tools, effort)
                if len(prompt_ids) >= engine.max_context:
                    raise RequestError(
                        400,
                        f"prompt length {len(prompt_ids)} reaches or exceeds context limit {engine.max_context}",
                    )

                raw_max = body.get("max_completion_tokens", body.get("max_tokens", 32768))
                try:
                    max_tokens = int(raw_max if raw_max is not None else 32768)
                except (TypeError, ValueError) as exc:
                    raise RequestError(400, f"invalid max_tokens: {raw_max!r}") from exc
                max_tokens = max(1, min(max_tokens, engine.max_context - len(prompt_ids)))

                raw_temp = body.get("temperature", 0.0)
                raw_top_p = body.get("top_p", 1.0)
                temperature = float(0.0 if raw_temp is None else raw_temp)
                top_p = float(1.0 if raw_top_p is None else raw_top_p)
                stop_strings = _parse_stop_strings(body.get("stop"))

                no_cache_hdr = (self.headers.get("x-no-prefix-cache") or "").strip().lower()
                use_prefix_cache = not (
                    no_cache_hdr in ("1", "true", "yes") or body.get("prefix_cache") is False
                )
                stream = bool(body.get("stream", False))
                resp_model = str(req_model) if req_model else model_id
            except RequestError as exc:
                self._send_json(exc.status, {"error": {"message": exc.message, "type": "invalid_request_error"}})
                return
            except Exception as exc:
                self._send_json(500, {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}})
                return

            stream_q: queue.Queue | None = queue.Queue() if stream else None
            gen_req = GenRequest(
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_strings=stop_strings,
                use_prefix_cache=use_prefix_cache,
                stream_queue=stream_q,
            )
            engine.submit(gen_req)

            resp_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
            created = int(time.time())

            if not stream:
                gen_req.done_event.wait()
                res = gen_req.result
                if "error" in res:
                    self._send_json(500, {"error": {"message": res["error"], "type": "server_error"}})
                    return
                self._send_json(200, _format_non_stream_response(resp_id, created, resp_model, res))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True

            sent_role = False
            assert stream_q is not None
            try:
                while True:
                    if activity_cb is not None:
                        activity_cb()
                    try:
                        kind, payload = stream_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if kind == "delta":
                        delta = dict(payload)
                        if not sent_role:
                            delta["role"] = "assistant"
                            sent_role = True
                        chunk = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": resp_model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    elif kind == "done":
                        tail = {
                            "id": resp_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": resp_model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": payload["finish_reason"]}],
                            "usage": payload["usage"],
                            "x_metrics": payload.get("metrics"),
                        }
                        self.wfile.write(f"data: {json.dumps(tail, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break
                    elif kind == "error":
                        err = {"error": {"message": str(payload), "type": "server_error"}}
                        self.wfile.write(f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break
            except (BrokenPipeError, ConnectionResetError):
                pass

    return InklingHandler


def warmup_engine(engine: InklingEngine) -> None:
    """Compile core Metal prefill, decode, and MTP verify graphs upfront without polluting the prefix cache."""
    pad_id = getattr(engine.tokenizer, "eos_token_id", None)
    if not isinstance(pad_id, int):
        pad_id = 0
    for n in (4, 32):
        req = GenRequest(
            prompt_ids=[pad_id] * n,
            max_tokens=2,
            temperature=0.0,
            top_p=1.0,
            stop_strings=[],
            use_prefix_cache=False,
        )
        engine.submit(req)
        req.done_event.wait(timeout=120)
    if engine.drafter is not None:
        orig_k = engine.spec_draft_k
        for k_w in (2, 3, 4, 5, 6, 7):
            engine.spec_draft_k = k_w
            req = GenRequest(
                prompt_ids=[pad_id] * 16,
                max_tokens=k_w + 3,
                temperature=0.0,
                top_p=1.0,
                stop_strings=[],
                use_prefix_cache=False,
            )
            engine.submit(req)
            req.done_event.wait(timeout=120)
        engine.spec_draft_k = orig_k
        engine.mtp_total_steps = 0
        engine.mtp_total_accepted = 0
    mx.synchronize()
    engine.prefix_cache.clear()
    engine.prefix_cache.hits = 0
    engine.prefix_cache.misses = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inkling 2 Mini OpenAI-compatible MLX engine")
    p.add_argument("--model", default="thinkingmachines/Inkling-2-Mini")
    p.add_argument("--adapter", default=None, help="Optional PEFT LoRA adapter local directory or HF repo ID (e.g. balllvin/BS-0.5-Mini)")
    p.add_argument("--model-id", default="inkling-2-mini")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--prefill-chunk", type=int, default=8192)
    p.add_argument("--prefix-cache-gb", type=float, default=4.0)
    p.add_argument("--cache-limit-gb", type=float, default=4.0)
    p.add_argument("--max-context", type=int, default=262144)
    p.add_argument("--no-mtp", action="store_true", help="Disable built-in MTP speculative drafter")
    p.add_argument("--spec-draft-k", type=int, default=0, help="Draft depth 1..7, or 0 for adaptive 3..7")
    p.add_argument("--idle-exit", type=int, default=0)
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mx.set_cache_limit(max(0, int(args.cache_limit_gb * (1 << 30))))
    from .cli import resolve_adapter_dir, resolve_model_dir
    model_path = resolve_model_dir(args.model)
    adapter_path = resolve_adapter_dir(args.adapter) if args.adapter else None
    t0 = time.perf_counter()
    print(f"[inkling] loading {model_path} ...", flush=True)
    model, tokenizer, _cfg = load_model(model_path, adapter_dir=adapter_path)
    load_s = time.perf_counter() - t0
    print(f"[inkling] loaded in {load_s:.2f}s (active={mx.get_active_memory()/1e9:.2f} GB)", flush=True)

    drafter = None
    if not args.no_mtp:
        tm = time.perf_counter()
        drafter = load_mtp_drafter(model_path, model)
        if drafter is not None:
            print(
                f"[inkling] MTP drafter loaded+folded in {time.perf_counter() - tm:.2f}s "
                f"(active={mx.get_active_memory()/1e9:.2f} GB, spec_draft_k={args.spec_draft_k})",
                flush=True,
            )

    engine = InklingEngine(
        model,
        tokenizer,
        max_batch=args.max_batch,
        prefill_chunk=args.prefill_chunk,
        prefix_cache_gb=args.prefix_cache_gb,
        max_context=args.max_context,
        drafter=drafter,
        spec_draft_k=args.spec_draft_k,
    )
    if not args.no_warmup:
        tw = time.perf_counter()
        warmup_engine(engine)
        print(f"[inkling] warmup done in {time.perf_counter() - tw:.2f}s", flush=True)

    last_active = [time.monotonic()]
    touch = partial(last_active.__setitem__, 0)

    def _on_activity() -> None:
        touch(time.monotonic())

    handler_cls = make_handler(
        engine,
        model_id=args.model_id,
        model_path=model_path,
        adapter_id=args.adapter,
        activity_cb=_on_activity,
    )
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    server.daemon_threads = True

    stop_event = threading.Event()

    def _shutdown(signum: int, _frame: Any) -> None:
        print(f"[inkling] signal {signum} received, shutting down ...", flush=True)
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.idle_exit > 0:
        def _idle_watchdog() -> None:
            while not stop_event.wait(timeout=5.0):
                if time.monotonic() - last_active[0] >= args.idle_exit:
                    print("[inkling] idle timeout reached, shutting down", flush=True)
                    stop_event.set()
                    server.shutdown()
                    break

        threading.Thread(target=_idle_watchdog, daemon=True).start()

    print(f"[inkling] serving {args.model_id} on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        engine.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
