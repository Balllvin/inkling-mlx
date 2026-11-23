"""Unit and integration tests for the Inkling 2 Mini engine, parser, and HTTP server."""

from __future__ import annotations

import json
import queue
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import mlx.core as mx

from inkling_mlx.config import LoopConfig, TextConfig
from inkling_mlx.engine import (
    GenRequest,
    InklingEngine,
)
from inkling_mlx.model import LanguageModel
from inkling_mlx.parser import (
    InklingStreamParser,
    RequestError,
    normalize_messages,
    parse_complete_output,
    resolve_reasoning_effort,
)
from inkling_mlx.serve import make_handler


class _FakeTokenizer:
    eos_token_id = 200006

    def __init__(self, vocab_map: dict[int, str] | None = None):
        self._vocab = vocab_map or {
            10: "hello ",
            11: "world",
            12: "<|content_thinking|>",
            13: "<|end_message|>",
            14: "<|content_text|>",
            15: "ok",
            200000: "<|message_user|>",
            200001: "<|message_model|>",
            200006: "<|content_model_end_sampling|>",
        }

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="high",
        tools=None,
    ):
        return "|".join(f"{m['role']}:{m.get('content', '')}" for m in messages) + f"|effort={reasoning_effort}"

    def encode(self, text, add_special_tokens=False):
        return [200000 if i == 5 else (10 + (i % 5)) for i in range(max(8, len(text) // 2))]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._vocab.get(int(i), f"t{i}") for i in ids)


def _tiny_model() -> LanguageModel:
    cfg = TextConfig(
        hidden_size=128,
        num_hidden_layers=6,
        vocab_size=64,
        unpadded_vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=32,
        sliding_window_size=16,
        d_rel=8,
        rel_extent=16,
        intermediate_size=64,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_layers_per_attn_res_block=2,
        loop_config=LoopConfig(core_start=2, core_end=4, num_passes=2, gate_rank=16),
    )
    model = LanguageModel(cfg)
    mx.eval(model.parameters())
    return model


class TestInklingEngine(unittest.TestCase):
    def test_parser_thinking_text_and_tool_calls(self):
        raw = (
            "<|message_model|><|content_thinking|>Let me check.<|end_message|>"
            "<|message_model|><|content_text|>Running code.<|end_message|>"
            "<|message_model|>run_code<|content_invoke_tool_json|>"
            '{"name":"run_code","arguments":{"code":"print(42)"}}'
            "<|end_message|><|content_model_end_sampling|>"
        )
        parsed = parse_complete_output(raw)
        self.assertFalse(parsed["malformed"])
        self.assertEqual(parsed["reasoning_content"], "Let me check.")
        self.assertEqual(parsed["content"], "Running code.")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        tc = parsed["tool_calls"][0]
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "run_code")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"code": "print(42)"})

        stream = InklingStreamParser()
        chunks = [raw[i : i + 3] for i in range(0, len(raw), 3)]
        ev_think, ev_text, ev_tools = [], [], []
        for ch in chunks:
            for ev in stream.feed(ch):
                if "reasoning_content" in ev:
                    ev_think.append(ev["reasoning_content"])
                if "content" in ev:
                    ev_text.append(ev["content"])
                if "tool_calls" in ev:
                    ev_tools.extend(ev["tool_calls"])
        for ev in stream.finish():
            if "tool_calls" in ev:
                ev_tools.extend(ev["tool_calls"])
        self.assertEqual("".join(ev_think), "Let me check.")
        self.assertEqual("".join(ev_text), "Running code.")
        self.assertEqual(len(ev_tools), 1)
        self.assertEqual(ev_tools[0]["function"]["name"], "run_code")

    def test_normalize_messages_and_reasoning_effort(self):
        msgs = [
            {"role": "developer", "content": [{"type": "text", "text": "sys"}]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "run_code", "arguments": '{"code":"x=1"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        out = normalize_messages(msgs)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "sys")
        self.assertIsInstance(out[1]["tool_calls"][0]["function"]["arguments"], dict)
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], {"code": "x=1"})

        self.assertEqual(resolve_reasoning_effort({"reasoning_effort": "off"}), "none")
        self.assertEqual(resolve_reasoning_effort({"enable_thinking": False}), "none")
        self.assertEqual(resolve_reasoning_effort({"reasoning": {"effort": "low"}}), "low")

        with self.assertRaises(RequestError):
            normalize_messages([])

    def test_utf8_split_token_buffering(self):
        model = _tiny_model()

        class _SplitUtf8Tok(_FakeTokenizer):
            def decode(self, ids, skip_special_tokens=False):
                if ids == [1]:
                    return "\ufffd"
                if ids == [1, 2]:
                    return "🚀"
                return "!"

        engine = InklingEngine(model, _SplitUtf8Tok(), max_batch=2, prefill_chunk=16, prefix_cache_gb=0.0)
        try:
            import inkling_mlx.engine as eng_mod

            seq_tokens = iter([1, 2, 3, 200006])
            orig_sample = eng_mod.sample_token
            orig_sample_batch = eng_mod.sample_batch_tokens
            eng_mod.sample_token = lambda *a, **kw: next(seq_tokens)
            eng_mod.sample_batch_tokens = lambda logits, temps, top_ps: [next(seq_tokens) for _ in temps]
            try:
                q: queue.Queue = queue.Queue()
                req = GenRequest(
                    [10, 11, 12, 13],
                    max_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                    stop_strings=[],
                    use_prefix_cache=False,
                    stream_queue=q,
                )
                engine.submit(req)
                self.assertTrue(req.done_event.wait(timeout=10))
                self.assertNotIn("\ufffd", req.result["content"] or "")
                self.assertEqual(req.result["content"], "🚀!")
            finally:
                eng_mod.sample_token = orig_sample
                eng_mod.sample_batch_tokens = orig_sample_batch
        finally:
            engine.stop()

    def test_continuous_batching_staggered_admission_and_prefix_cache(self):
        model = _tiny_model()
        tok = _FakeTokenizer()
        engine = InklingEngine(model, tok, max_batch=4, prefill_chunk=8, prefix_cache_gb=0.1)
        try:
            reqs = [
                GenRequest([20, 21, 22, 23, 24], max_tokens=5, temperature=0.0, top_p=1.0, stop_strings=[]),
                GenRequest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], max_tokens=3, temperature=0.0, top_p=1.0, stop_strings=[]),
                GenRequest([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], max_tokens=7, temperature=0.0, top_p=1.0, stop_strings=[]),
                GenRequest([30, 31, 32, 33, 34, 35, 36], max_tokens=4, temperature=0.0, top_p=1.0, stop_strings=[]),
            ]
            for r in reqs:
                engine.submit(r)
            for r in reqs:
                self.assertTrue(r.done_event.wait(timeout=20))
                self.assertNotIn("error", r.result, r.result)
                self.assertLessEqual(r.result["usage"]["completion_tokens"], r.max_tokens)
                self.assertEqual(r.result["usage"]["prompt_tokens"], len(r.prompt_ids))

            sys_prefix = [10 + (i % 10) for i in range(132)]
            r1 = GenRequest(sys_prefix + [200000, 11, 12, 13], max_tokens=2, temperature=0.0, top_p=1.0, stop_strings=[])
            engine.submit(r1)
            self.assertTrue(r1.done_event.wait(timeout=10))
            self.assertEqual(r1.result["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

            r2 = GenRequest(sys_prefix + [200000, 21, 22, 23], max_tokens=2, temperature=0.0, top_p=1.0, stop_strings=[])
            engine.submit(r2)
            self.assertTrue(r2.done_event.wait(timeout=10))
            self.assertEqual(r2.result["usage"]["prompt_tokens_details"]["cached_tokens"], 132)

            r3 = GenRequest(sys_prefix + [200000, 11, 12, 13, 42], max_tokens=2, temperature=0.0, top_p=1.0, stop_strings=[])
            engine.submit(r3)
            self.assertTrue(r3.done_event.wait(timeout=10))
            self.assertEqual(r3.result["usage"]["prompt_tokens_details"]["cached_tokens"], 136)
        finally:
            engine.stop()

    def test_http_server_endpoints_and_sse(self):
        model = _tiny_model()
        tok = _FakeTokenizer()
        engine = InklingEngine(model, tok, max_batch=4, prefill_chunk=16, prefix_cache_gb=0.0)
        handler = make_handler(engine, model_id="inkling-2-mini")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            base = f"http://127.0.0.1:{port}"
            with urllib.request.urlopen(f"{base}/health") as resp:
                self.assertEqual(resp.status, 200)
                self.assertTrue(json.loads(resp.read())["ok"])

            with urllib.request.urlopen(f"{base}/v1/models") as resp:
                self.assertEqual(json.loads(resp.read())["data"][0]["id"], "inkling-2-mini")

            with urllib.request.urlopen(f"{base}/v1/capabilities") as resp:
                cap = json.loads(resp.read())
                self.assertEqual(cap["contextWindow"], 262144)
                self.assertTrue(cap["tools"])

            req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({
                    "model": "inkling-2-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 3,
                    "stream": False,
                }).encode(),
                headers={"Content-Type": "application/json", "x-no-prefix-cache": "1"},
            )
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
                self.assertGreaterEqual(body["usage"]["completion_tokens"], 1)

            bad_req = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({
                    "model": "unknown-model",
                    "messages": [{"role": "user", "content": "hi"}],
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(bad_req)
            self.assertEqual(ctx.exception.code, 404)

            sreq = urllib.request.Request(
                f"{base}/v1/chat/completions",
                data=json.dumps({
                    "model": "inkling-2-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 3,
                    "stream": True,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(sreq) as resp:
                raw_sse = resp.read().decode("utf-8")
                self.assertIn("data: [DONE]", raw_sse)
                data_lines = [
                    ln[6:] for ln in raw_sse.splitlines() if ln.startswith("data: ") and ln != "data: [DONE]"
                ]
                last_obj = json.loads(data_lines[-1])
                self.assertGreaterEqual(last_obj["usage"]["completion_tokens"], 1)
        finally:
            server.shutdown()
            server.server_close()
            engine.stop()

    def _build_synthetic_bf16_moe_model(self) -> LanguageModel:
        cfg = TextConfig(
            hidden_size=256,
            num_hidden_layers=4,
            vocab_size=64,
            unpadded_vocab_size=64,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            swa_num_attention_heads=8,
            swa_num_key_value_heads=2,
            swa_head_dim=32,
            sliding_window_size=12,
            layer_types=["hybrid_sliding", "hybrid_sliding", "full_attention", "hybrid_sliding"],
            d_rel=16,
            rel_extent=16,
            intermediate_size=128,
            n_routed_experts=8,
            num_experts_per_tok=4,
            n_layers_per_attn_res_block=1,
            loop_config=LoopConfig(core_start=1, core_end=3, num_passes=3, gate_rank=16),
        )
        mx.random.seed(123)
        from inkling_mlx.model import QuantizedExpertLinear

        model = LanguageModel(cfg)
        rbf = lambda shp, sc=0.05: (mx.random.normal(shp) * sc).astype(mx.bfloat16)
        rf32 = lambda shp, sc=0.05: (mx.random.normal(shp) * sc).astype(mx.float32)
        tm = model.model
        tm.embed_tokens.weight = rbf(tm.embed_tokens.weight.shape, 0.1)
        tm.embed_norm.weight = (mx.ones(tm.embed_norm.weight.shape) + rf32(tm.embed_norm.weight.shape)).astype(mx.bfloat16)
        tm.message_input.query = rf32(tm.message_input.query.shape, 0.1)
        tm.final_input.query = rf32(tm.final_input.query.shape, 0.1)
        tm.norm.weight = (mx.ones(tm.norm.weight.shape) + rf32(tm.norm.weight.shape)).astype(mx.bfloat16)
        model.lm_head.weight = rbf(model.lm_head.weight.shape, 0.1)
        for idx, layer in enumerate(tm.layers):
            if layer.attention_input is not None:
                layer.attention_input.query = rf32(layer.attention_input.query.shape, 0.1)
            layer.mlp_input.query = rf32(layer.mlp_input.query.shape, 0.1)
            if layer.is_sliding:
                layer.input_layernorm.weight = (mx.ones((256,)) + rf32((256,))).astype(mx.bfloat16)
            else:
                for nm in ("q_input_layernorm", "k_input_layernorm", "v_input_layernorm"):
                    getattr(layer, nm).weight = (mx.ones((256,)) + rf32((256,))).astype(mx.bfloat16)
            layer.post_attention_layernorm.weight = (mx.ones((256,)) + rf32((256,))).astype(mx.bfloat16)
            layer.attn_sconv.conv.weight = rf32(layer.attn_sconv.conv.weight.shape, 0.15)
            layer.mlp_sconv.conv.weight = rf32(layer.mlp_sconv.conv.weight.shape, 0.15)
            if hasattr(layer, "q_message_gate"):
                for gname in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                    g = getattr(layer, gname)
                    g.down_proj.weight = rbf(g.down_proj.weight.shape, 0.05)
                    g.forget_gate_proj.weight = rf32(g.forget_gate_proj.weight.shape, 0.05)
                    g.message_gate_proj.weight = rf32(g.message_gate_proj.weight.shape, 0.05)
            a = layer.self_attn
            for pn in ("q_proj", "k_proj", "v_proj", "r_proj", "o_proj", "gate_proj", "sink_proj"):
                mod = getattr(a, pn)
                mod.weight = rbf(mod.weight.shape, 0.05)
            a.q_scale = (mx.ones(a.q_scale.shape) + rf32(a.q_scale.shape, 0.1)).astype(mx.bfloat16)
            a.rel_scale = (mx.ones(a.rel_scale.shape) + rf32(a.rel_scale.shape, 0.1)).astype(mx.bfloat16)
            a.rel_proj = rbf(a.rel_proj.shape, 0.05)
            if layer.is_sliding:
                a.k_sconv.conv.weight = rf32(a.k_sconv.conv.weight.shape, 0.15)
                a.v_sconv.conv.weight = rf32(a.v_sconv.conv.weight.shape, 0.15)
            else:
                a.k_sconv.static_weight = rbf(a.k_sconv.static_weight.shape, 0.15)
                a.v_sconv.static_weight = rbf(a.v_sconv.static_weight.shape, 0.15)
                a.k_sconv.weight_proj.weight = rbf(a.k_sconv.weight_proj.weight.shape, 0.05)
                a.v_sconv.weight_proj.weight = rbf(a.v_sconv.weight_proj.weight.shape, 0.05)
            moe = layer.mlp
            moe.gate_weight = rf32(moe.gate_weight.shape, 0.1)
            moe.e_score_correction_bias = rf32(moe.e_score_correction_bias.shape, 0.05)
            mode = "mxfp8" if idx == 3 else "nvfp4"
            moe.w13_proj = QuantizedExpertLinear(8, 256, 256, mode)
            moe.down_proj = QuantizedExpertLinear(8, 256, 128, mode)
            w13_q, s13_q = mx.quantize(rbf((8, 256, 256), 0.05), group_size=(16 if mode == "nvfp4" else 32), bits=(4 if mode == "nvfp4" else 8), mode=mode)
            w2_q, s2_q = mx.quantize(rbf((8, 256, 128), 0.05), group_size=(16 if mode == "nvfp4" else 32), bits=(4 if mode == "nvfp4" else 8), mode=mode)
            moe.w13_proj.weight, moe.w13_proj.scales = w13_q, s13_q
            moe.down_proj.weight, moe.down_proj.scales = w2_q, s2_q
            if mode == "nvfp4":
                moe.w13_proj.global_scale = mx.random.uniform(1800.0, 3600.0, shape=(8,), dtype=mx.float32)
                moe.down_proj.global_scale = mx.random.uniform(1800.0, 3600.0, shape=(8,), dtype=mx.float32)
        mx.eval(model.parameters())
        model.eval()
        return model

    def test_batched_native_continuous_batching_parity(self):
        model = self._build_synthetic_bf16_moe_model()
        tok = _FakeTokenizer()

        def run_workload(disable_native: bool) -> list[list[int]]:
            engine = InklingEngine(model, tok, max_batch=12, prefill_chunk=8, prefix_cache_gb=0.0)
            if disable_native and engine.native_decode is not None:
                engine.native_decode.close()
                engine.native_decode = None
            try:
                prompts = [
                    ([10 + (i + j) % 15 for j in range(6 + (i % 7))], 2 + (i % 6))
                    for i in range(10)
                ]
                reqs = [
                    GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[])
                    for p, mt in prompts
                ]
                for r in reqs:
                    engine.submit(r)
                for r in reqs:
                    self.assertTrue(r.done_event.wait(timeout=30))
                    self.assertNotIn("error", r.result, r.result)
                late = [
                    GenRequest([11, 12, 13, 14, 15, 16, 17, 18], max_tokens=5, temperature=0.0, top_p=1.0, stop_strings=[]),
                    GenRequest([21, 22, 23, 24, 25], max_tokens=3, temperature=0.0, top_p=1.0, stop_strings=[]),
                ]
                for r in late:
                    engine.submit(r)
                for r in late:
                    self.assertTrue(r.done_event.wait(timeout=20))
                    self.assertNotIn("error", r.result, r.result)
                return [
                    [int(r.result["usage"]["completion_tokens"])]
                    + [hash(r.result.get("content") or "")]
                    for r in (reqs + late)
                ]
            finally:
                engine.stop()

        if mx.default_device() == mx.gpu:
            nat_res = run_workload(False)
            mlx_res = run_workload(True)
            self.assertEqual(nat_res, mlx_res)
            from inkling_mlx.native_decode import try_create_native_decode_runner

            runner = try_create_native_decode_runner(model, max_m=12)
            self.assertIsNotNone(runner)
            try:
                for bsz in (4, 8):
                    tok_in = [10 + i for i in range(bsz)]
                    c_nat = model.make_batch_cache([0] * bsz)
                    c_mlx = model.make_batch_cache([0] * bsz)
                    prompt = mx.array([[11, 12, 13, 14, 15]] * bsz, dtype=mx.int32)
                    model(prompt, cache=c_nat, skip_logits=True)
                    model(prompt, cache=c_mlx, skip_logits=True)
                    _eval_c = lambda cs: mx.eval(
                        [x for cl in cs for x in (cl.caches[0].keys, cl.caches[0].values, *cl.caches[1].cache) if x is not None]
                    )
                    _eval_c(c_nat)
                    _eval_c(c_mlx)
                    runner.clear_cache_binding()
                    nat_toks = runner.step_batch_greedy(tok_in, c_nat)
                    runner.sync_batch_cache(c_nat)
                    runner.clear_cache_binding()
                    mlx_toks = [
                        int(v)
                        for v in model(mx.array(tok_in, dtype=mx.int32)[:, None], cache=c_mlx, greedy_argmax=True)["argmax"]
                        .reshape(-1)
                        .tolist()
                    ]
                    self.assertEqual(nat_toks, mlx_toks)
            finally:
                runner.close()

    def _build_synthetic_mtp_drafter(self, model: LanguageModel):
        from mlx.utils import tree_map
        from inkling_mlx.mtp import InklingMTPDrafter, MTPConfig

        mcfg = MTPConfig(
            hidden_size=256,
            num_hidden_layers=3,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=32,
            intermediate_size=128,
            sliding_window_size=12,
            block_size=8,
            markov_rank=32,
            target_layer_ids=(1, 3, 5, 7),
            vocab_size=64,
            unpadded_vocab_size=64,
            final_logit_softcapping=30.0,
        )
        mx.random.seed(456)
        drafter = InklingMTPDrafter(mcfg)
        drafter.update(
            tree_map(
                lambda a: (mx.random.normal(a.shape) * 0.05).astype(mx.bfloat16) if a.ndim >= 2 else a.astype(mx.bfloat16),
                drafter.parameters(),
            )
        )
        drafter.fuse_weight = (mx.random.normal(drafter.fuse_weight.shape) * 0.1).astype(mx.float32)
        mx.eval(drafter.parameters())
        drafter.bind(model)
        return drafter

    def test_unified_1cb_mtp_greedy_and_sampled(self):
        if mx.default_device() != mx.gpu:
            return
        model = self._build_synthetic_bf16_moe_model()
        drafter = self._build_synthetic_mtp_drafter(model)
        tok = _FakeTokenizer()
        eng_plain = InklingEngine(model, tok, max_batch=4, prefill_chunk=8, prefix_cache_gb=0.0)
        eng_mtp = InklingEngine(
            model, tok, max_batch=4, prefill_chunk=8, prefix_cache_gb=0.0, drafter=drafter, spec_draft_k=5
        )
        self.assertIsNotNone(eng_mtp.native_decode)
        self.assertTrue(eng_mtp.native_decode.mtp_attached)
        try:
            prompt = [10, 11, 12, 13, 14, 15, 16, 17]
            r_p = GenRequest(prompt, max_tokens=16, temperature=0.0, top_p=1.0, stop_strings=[])
            r_g = GenRequest(prompt, max_tokens=16, temperature=0.0, top_p=1.0, stop_strings=[])
            eng_plain.submit(r_p)
            eng_mtp.submit(r_g)
            self.assertTrue(r_p.done_event.wait(timeout=20))
            self.assertTrue(r_g.done_event.wait(timeout=20))
            self.assertEqual(r_g.result["content"], r_p.result["content"])
            self.assertGreater(r_g.result["metrics"]["mtp_steps"], 0)
            self.assertEqual(r_g.result["usage"]["completion_tokens"], 16)

            r_s = GenRequest(prompt, max_tokens=14, temperature=0.7, top_p=0.95, stop_strings=[])
            eng_mtp.submit(r_s)
            self.assertTrue(r_s.done_event.wait(timeout=20))
            self.assertNotIn("error", r_s.result, r_s.result)
            self.assertEqual(r_s.result["usage"]["completion_tokens"], 14)
            self.assertGreater(r_s.result["metrics"]["mtp_steps"], 0)
        finally:
            eng_plain.stop()
            eng_mtp.stop()

    def test_batched_mtp_concurrent_agents_parity(self):
        if mx.default_device() != mx.gpu:
            return
        import time as _time

        model = self._build_synthetic_bf16_moe_model()
        for layer in model.model.layers:
            layer.mlp.e_score_correction_bias = mx.arange(8, dtype=mx.float32) * 2.0
        mx.eval(model.parameters())
        drafter = self._build_synthetic_mtp_drafter(model)
        tok = _FakeTokenizer()
        eng_plain = InklingEngine(model, tok, max_batch=6, prefill_chunk=64, prefix_cache_gb=0.0)
        eng_plain.user_message_token_id = 63
        eng_mtp = InklingEngine(
            model, tok, max_batch=6, prefill_chunk=64, prefix_cache_gb=0.1, drafter=drafter, spec_draft_k=4
        )
        eng_mtp.user_message_token_id = 63
        self.assertIsNotNone(eng_mtp.native_decode)
        self.assertTrue(eng_mtp.native_decode.mtp_attached)
        try:
            def _assert_same_gen(got: dict, ref: dict) -> None:
                self.assertEqual(
                    (
                        got.get("content"),
                        got.get("reasoning_content"),
                        got.get("finish_reason"),
                        got["usage"]["completion_tokens"],
                    ),
                    (
                        ref.get("content"),
                        ref.get("reasoning_content"),
                        ref.get("finish_reason"),
                        ref["usage"]["completion_tokens"],
                    ),
                )

            def _plain_ref(pids: list[int], max_toks: int, *, use_pc: bool = False) -> dict:
                r = GenRequest(pids, max_tokens=max_toks, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=use_pc)
                eng_plain.submit(r)
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                return r.result

            def _burst_plain_refs(prompt_list: list[tuple[list[int], int]]) -> list[list[int]]:
                from inkling_mlx.engine import ActiveSeq
                from inkling_mlx.parser import InklingStreamParser

                aseqs = [
                    ActiveSeq(
                        req=GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False),
                        cache=None,
                        parser=InklingStreamParser(),
                    )
                    for p, mt in prompt_list
                ]
                ready = eng_mtp._prefill_burst_mtp(aseqs, len(aseqs))
                self.assertEqual(len(ready), len(prompt_list))
                out_ids: list[list[int]] = []
                for idx_s, s in enumerate(ready):
                    ref_s = ActiveSeq(
                        req=GenRequest(prompt_list[idx_s][0], max_tokens=prompt_list[idx_s][1], temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False),
                        cache=None,
                        parser=InklingStreamParser(),
                    )
                    while not eng_plain._step_prefill_chunk(ref_s):
                        pass
                    self.assertEqual(s.next_token, ref_s.next_token)
                    for cl_b, cl_r in zip(s.cache, ref_s.cache):
                        kvb, kvr = cl_b.caches[0], cl_r.caches[0]
                        if hasattr(kvb, "_idx"):
                            kb = kvb._temporal_order(kvb.keys)[..., -min(kvb.offset, kvb.max_size) :, :]
                            kr = kvr._temporal_order(kvr.keys)[..., -min(kvr.offset, kvr.max_size) :, :]
                        else:
                            kb = kvb.keys[..., : kvb.offset, :]
                            kr = kvr.keys[..., : kvr.offset, :]
                        self.assertLess(float(mx.max(mx.abs(kb - kr)).item()), 0.05)

                    for _ in range(s.req.max_tokens - 1):
                        tk = eng_mtp.native_decode.step_m1_greedy(s.next_token, s.cache)
                        s.next_token = tk
                        s.generated_ids.append(tk)
                    out_ids.append(list(s.generated_ids))
                    eng_mtp._release_seq_bindings(s)
                return out_ids

            sys_pre = [10 + (i % 7) for i in range(132)] + [63]
            prompts_b4 = [
                ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], 14),
                ([15, 16, 17, 18, 19, 20, 21, 22, 23], 10),
                (sys_pre + [10, 11, 12, 13, 14], 12),
                (sys_pre + [20, 21, 22, 23, 24, 25, 26], 16),
            ]
            refs_b4_burst_ids = _burst_plain_refs(prompts_b4)
            refs_b2_burst_ids = _burst_plain_refs(prompts_b4[:2])
            refs_b4_single = [_plain_ref(p, mt, use_pc=False) for p, mt in prompts_b4]

            def _capture_result_tuple(ids: list[int]) -> tuple[str | None, str | None, str, int]:
                p = InklingStreamParser()
                for tid in ids:
                    p.feed(tok.decode([tid], skip_special_tokens=False))
                p.finish()
                return (p.content or None, p.reasoning_content or None, "length", len(ids))

            reqs_b2 = [
                GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False)
                for p, mt in prompts_b4[:2]
            ]
            for r in reqs_b2:
                eng_mtp.submit(r)
            for r, ref_ids in zip(reqs_b2, refs_b2_burst_ids):
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                self.assertEqual(
                    (r.result.get("content"), r.result.get("reasoning_content"), r.result.get("finish_reason"), r.result["usage"]["completion_tokens"]),
                    _capture_result_tuple(ref_ids),
                )
                self.assertGreater(r.result["metrics"]["mtp_steps"], 0)

            reqs_b4 = [
                GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False)
                for p, mt in prompts_b4
            ]
            for r in reqs_b4:
                eng_mtp.submit(r)
            for r, ref_ids in zip(reqs_b4, refs_b4_burst_ids):
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                self.assertEqual(
                    (r.result.get("content"), r.result.get("reasoning_content"), r.result.get("finish_reason"), r.result["usage"]["completion_tokens"]),
                    _capture_result_tuple(ref_ids),
                )
                self.assertGreater(r.result["metrics"]["mtp_steps"], 0)

            eng_plain_pc = InklingEngine(model, tok, max_batch=4, prefill_chunk=64, prefix_cache_gb=0.1)
            eng_plain_pc.user_message_token_id = 63
            try:
                pc_refs = []
                for p, mt in prompts_b4[2:]:
                    rp = GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=True)
                    eng_plain_pc.submit(rp)
                    self.assertTrue(rp.done_event.wait(timeout=30))
                    self.assertNotIn("error", rp.result, rp.result)
                    pc_refs.append(rp.result)
            finally:
                eng_plain_pc.stop()

            reqs_pc = [
                GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=True)
                for p, mt in prompts_b4[2:]
            ]
            for r in reqs_pc:
                eng_mtp.submit(r)
            for r, ref in zip(reqs_pc, pc_refs):
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                _assert_same_gen(r.result, ref)
                self.assertGreater(r.result["metrics"]["mtp_steps"], 0)
                self.assertEqual(r.result["usage"]["prompt_tokens_details"]["cached_tokens"], 132)

            stagger_p = [
                ([12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 20),
                ([14, 15, 16, 17, 18, 19, 20, 21], 12),
                ([16, 17, 18, 19, 20, 21, 22], 14),
                ([18, 19, 20, 21, 22, 23, 24, 25], 10),
            ]
            refs_st = [_plain_ref(p, mt) for p, mt in stagger_p]
            reqs_st = [
                GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False)
                for p, mt in stagger_p
            ]
            for idx_st, r in enumerate(reqs_st):
                eng_mtp.submit(r)
                if idx_st < 2:
                    _time.sleep(0.015)
            for r, ref in zip(reqs_st, refs_st):
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                _assert_same_gen(r.result, ref)
                self.assertGreater(r.result["metrics"]["mtp_steps"], 0)

            mixed = [
                GenRequest(stagger_p[0][0], max_tokens=stagger_p[0][1], temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False),
                GenRequest(stagger_p[1][0], max_tokens=stagger_p[1][1], temperature=0.7, top_p=0.95, stop_strings=[], use_prefix_cache=False),
                GenRequest(stagger_p[2][0], max_tokens=stagger_p[2][1], temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False),
                GenRequest(stagger_p[3][0], max_tokens=stagger_p[3][1], temperature=0.7, top_p=0.95, stop_strings=[], use_prefix_cache=False),
            ]
            for r in mixed:
                eng_mtp.submit(r)
            for idx, r in enumerate(mixed):
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                self.assertGreater(r.result["metrics"]["mtp_steps"], 0)
                if idx in (0, 2):
                    _assert_same_gen(r.result, refs_st[idx])

            b5_prompts = prompts_b4 + [([22, 23, 24, 25, 26, 27, 28, 29], 12)]
            plain_b5 = [
                GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False)
                for p, mt in b5_prompts
            ]
            for r in plain_b5:
                eng_plain.submit(r)
            for r in plain_b5:
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)

            reqs_b5 = [
                GenRequest(p, max_tokens=mt, temperature=0.0, top_p=1.0, stop_strings=[], use_prefix_cache=False)
                for p, mt in b5_prompts
            ]
            for r in reqs_b5[:4]:
                eng_mtp.submit(r)
            _time.sleep(0.015)
            eng_mtp.submit(reqs_b5[4])
            for r, ref_r in zip(reqs_b5, plain_b5):
                self.assertTrue(r.done_event.wait(timeout=30))
                self.assertNotIn("error", r.result, r.result)
                self.assertEqual(r.result["usage"]["completion_tokens"], ref_r.result["usage"]["completion_tokens"])
                _assert_same_gen(r.result, ref_r.result)
            del refs_b4_single
        finally:
            eng_plain.stop()
            eng_mtp.stop()


if __name__ == "__main__":
    unittest.main()
