"""CPU contract tests; load the boundary without importing the GPU engine.

Real Starlette responses and Pydantic validation exercise the protocol that
FluentRouter's embedded Python consumes. Model inference is covered by the
Triton smoke client in examples/triton.
"""

import ast
import asyncio
import importlib.util
import json
import unittest
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).resolve().parents[4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boundary = load_module(
    "fluent_router_boundary", ROOT / "python/sglang/srt/entrypoints/fluent_router.py"
)
config_tool = load_module(
    "fluent_router_config", ROOT / "examples/triton/prepare_config.py"
)
smoke_client = load_module(
    "fluent_router_smoke", ROOT / "examples/triton/smoke_client.py"
)


def load_method(path, class_name, method_name, namespace):
    """Execute the real method while isolating GPU-only module imports."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


class Request(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    stream: bool = False


class Reply(BaseModel):
    choices: list


class Handler:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def handle_request(self, request, raw_request):
        self.requests.append((request, raw_request))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class Engine(boundary.FluentRouterEngineMixin):
    def __init__(self, response):
        self.handler = Handler(response)
        self.kind = None

    def _get_fluent_router_handler(self, kind):
        self.kind = kind
        return self.handler

    def _parse_fluent_router_request(self, kind, payload):
        return Request(**payload)


class TestFluentRouter(unittest.IsolatedAsyncioTestCase):
    async def test_text_nonstream_service_unavailable_remains_retryable(self):
        async def body():
            raise HTTPException(503, "busy")
            yield

        result = [c async for c in boundary.fluent_router_text_stream(body())]
        finish = result[0]["meta_info"]["finish_reason"]
        self.assertEqual(finish["err_type"], 1)
        self.assertEqual(finish["message"], "busy")

    def test_smoke_client_rejects_both_native_and_wrapped_errors(self):
        for error in (
            {"object": "error", "message": "bad", "code": 400},
            {"error": {"message": "busy"}},
        ):
            with self.assertRaises(RuntimeError):
                smoke_client.raise_for_error(error)
        smoke_client.raise_for_error({"choices": []})

    async def test_real_engine_method_keeps_native_token_input_and_adapts_router_text(
        self,
    ):
        method = load_method(
            "python/sglang/srt/entrypoints/engine.py",
            "Engine",
            "async_generate",
            {
                "GenerateReqInput": SimpleNamespace,
                "fluent_router_text_stream": boundary.fluent_router_text_stream,
                "aclosing": aclosing,
            },
        )
        closed = []
        received = []

        async def generate(obj, raw_request):
            received.append(obj)
            try:
                yield {
                    "text": "hello",
                    "meta_info": {
                        "finish_reason": {
                            "type": "abort",
                            "status_code": 400,
                            "err_type": "BadRequestError",
                        }
                    },
                }
            finally:
                closed.append(True)

        engine = SimpleNamespace(
            server_args=SimpleNamespace(enable_fluent_router=False),
            tokenizer_manager=SimpleNamespace(generate_request=generate),
            _resolve_routed_dp_rank=lambda rank, legacy: rank,
        )
        native = await method(engine, input_ids=[1, 2])
        self.assertEqual(
            native["meta_info"]["finish_reason"]["err_type"], "BadRequestError"
        )
        self.assertEqual(received[0].input_ids, [1, 2])
        engine.server_args.enable_fluent_router = True
        router = await method(engine, prompt="hello")
        self.assertEqual(router["meta_info"]["finish_reason"]["err_type"], 4)
        stream = await method(engine, prompt="hello", stream=True)
        self.assertEqual((await anext(stream))["text"], "hello")
        await stream.aclose()
        with self.assertRaisesRegex(ValueError, "n > 1"):
            await method(engine, prompt="hello", sampling_params={"n": 2})
        with self.assertRaisesRegex(ValueError, "one text prompt"):
            await method(engine, input_ids=[1, 2])
        self.assertGreaterEqual(len(closed), 2)

    def test_real_scheduler_reports_actual_capacity(self):
        method = load_method(
            "python/sglang/srt/managers/scheduler.py", "Scheduler", "get_init_info", {}
        )
        scheduler = SimpleNamespace(
            max_total_num_tokens=12345,
            max_req_input_len=8191,
            max_running_requests=7,
            chunked_prefill_size=2048,
            model_config=SimpleNamespace(context_len=8192),
            startup_time={},
        )
        info = method(scheduler)
        self.assertEqual(info["context_length"], 8192)
        self.assertEqual(info["max_running_requests"], 7)
        self.assertEqual(info["chunked_prefill_size"], 2048)

    async def test_chat_keeps_v41_fields_and_pd_metadata_without_mutating_input(self):
        reply = Reply(
            choices=[
                {
                    "message": {
                        "content": "你好",
                        "reasoning_content": "想",
                        "tool_calls": [],
                    }
                }
            ]
        )
        engine = Engine(reply)
        request = {
            "model": "v41",
            "messages": [{"role": "user", "content": "你好"}],
            "tools": [],
            "reasoning_effort": "high",
            "chat_template_kwargs": {"thinking": True},
            "bootstrap_room": 1,
        }
        result = await engine.openai_v1_chat_completions(request, "host", 8998, 2)
        self.assertEqual(json.loads(result), reply.model_dump())
        seen, raw = engine.handler.requests[0]
        self.assertIsNone(raw)
        self.assertEqual(seen.bootstrap_room, 2)
        self.assertEqual(seen.bootstrap_host, "host")
        self.assertEqual(seen.bootstrap_port, 8998)
        self.assertEqual(seen.reasoning_effort, "high")
        self.assertEqual(seen.chat_template_kwargs, {"thinking": True})
        self.assertEqual(request["bootstrap_room"], 1)

    async def test_completion_dispatch_and_json_response(self):
        engine = Engine(JSONResponse({"choices": [{"text": "hello"}]}))
        result = await engine.openai_v1_completions(
            {"model": "v41", "prompt": "Hi", "method": "complete"}
        )
        self.assertEqual(engine.kind, "completion")
        self.assertNotIn("method", engine.handler.requests[0][0].model_dump())
        self.assertEqual(json.loads(result)["choices"][0]["text"], "hello")

    async def test_sse_preserved_exactly_one_done_and_background_runs(self):
        events = []
        chunks = [
            'data: {"choices":[{"delta":{"reasoning_content":"想"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            "data: [DONE]\n\n",
        ]

        async def body():
            try:
                for chunk in chunks:
                    yield chunk
                self.fail("Must not read beyond DONE")
            finally:
                events.append("closed")

        async def cleanup():
            events.append("cleanup")

        engine = Engine(StreamingResponse(body(), background=BackgroundTask(cleanup)))
        stream = await engine.openai_v1_chat_completions(
            {"model": "v41", "stream": True}
        )
        result = [chunk async for chunk in stream]
        self.assertEqual(
            result, [c.decode() if isinstance(c, bytes) else c for c in chunks]
        )
        self.assertEqual(events, ["closed", "cleanup"])

    async def test_empty_stream_terminates(self):
        async def body():
            if False:
                yield

        engine = Engine(StreamingResponse(body()))
        stream = await engine.openai_v1_chat_completions(
            {"model": "v41", "stream": True}
        )
        self.assertEqual([c async for c in stream], ["data: [DONE]\n\n"])

    async def test_validation_and_handler_errors_are_terminal_strings(self):
        for stream in (False, True):
            for response, payload in [
                (None, {}),
                (
                    JSONResponse({"error": {"message": "bad"}}, status_code=400),
                    {"model": "v41"},
                ),
                (HTTPException(503, "busy"), {"model": "v41"}),
            ]:
                engine = Engine(response)
                result = await engine.openai_v1_chat_completions(
                    {**payload, "stream": stream}
                )
                chunks = [c async for c in result] if stream else [result]
                self.assertEqual(len(chunks), 1)
                self.assertFalse(chunks[0].startswith("data:"))
                self.assertIn("error", json.loads(chunks[0]))

    async def test_midstream_failure_and_early_close_cleanup(self):
        for fail in (False, True):
            events = []

            async def body():
                try:
                    yield "data: first\n\n"
                    raise HTTPException(503, "worker failed")
                finally:
                    events.append("closed")

            async def cleanup():
                events.append("cleanup")

            engine = Engine(
                StreamingResponse(body(), background=BackgroundTask(cleanup))
            )
            stream = await engine.openai_v1_chat_completions(
                {"model": "v41", "stream": True}
            )
            self.assertEqual(await anext(stream), "data: first\n\n")
            if fail:
                rest = [c async for c in stream]
                self.assertEqual(json.loads(rest[0])["error"]["code"], 503)
                self.assertEqual(len(rest), 1)
            else:
                await stream.aclose()
            self.assertEqual(events, ["closed", "cleanup"])

    async def test_cancellation_propagates_and_closes_body(self):
        entered = asyncio.Event()
        closed = []

        async def body():
            try:
                entered.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.append(True)

        engine = Engine(StreamingResponse(body()))
        stream = await engine.openai_v1_chat_completions(
            {"model": "v41", "stream": True}
        )
        task = asyncio.create_task(anext(stream))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(closed, [True])

    async def test_text_abort_mapping_preserves_native_objects_and_cumulative_text(
        self,
    ):
        outputs = [
            {"text": "你", "meta_info": {"finish_reason": None}},
            {
                "text": "你好",
                "meta_info": {
                    "finish_reason": {
                        "type": "abort",
                        "status_code": 503,
                        "err_type": "ServiceUnavailable",
                        "message": "busy",
                    }
                },
            },
        ]
        closed = []

        async def body():
            try:
                for output in outputs:
                    yield output
            finally:
                closed.append(True)

        result = [c async for c in boundary.fluent_router_text_stream(body())]
        self.assertEqual([c["text"] for c in result], ["你", "你好"])
        self.assertEqual(result[1]["meta_info"]["finish_reason"]["err_type"], 1)
        self.assertEqual(
            outputs[1]["meta_info"]["finish_reason"]["err_type"], "ServiceUnavailable"
        )
        self.assertEqual(closed, [True])

    def test_scheduler_snapshot_and_nonzero_rank(self):
        engine = boundary.FluentRouterEngineMixin()
        engine._scheduler_init_result = SimpleNamespace(
            scheduler_infos=[{"max_running_requests": 8, "max_req_input_len": 16000}]
        )
        info = engine.scheduler_info
        info["max_running_requests"] = 99
        self.assertEqual(engine.scheduler_info["max_running_requests"], 8)
        engine.tokenizer_manager = None
        with self.assertRaisesRegex(ValueError, "node_rank=0"):
            engine._get_fluent_router_handler("chat")

    def test_compatibility_config_rejects_incompatible_modes(self):
        for changed in (
            {"tokenizer_worker_num": 2},
            {"skip_tokenizer_init": True},
            {"incremental_streaming_output": True},
        ):
            engine = boundary.FluentRouterEngineMixin()
            config = {
                "tokenizer_worker_num": 1,
                "skip_tokenizer_init": False,
                "incremental_streaming_output": False,
                **changed,
            }
            engine.server_args = SimpleNamespace(resolved_dict=lambda: config)
            with self.assertRaises(ValueError):
                engine._init_fluent_router()

    def test_unknown_config_and_absolute_model_path_are_rejected(self):
        config = {"enable_fluent_router": True, "model_path": "hf_weight"}
        fields = set(config)
        config_tool.validate_config(config, fields)
        with self.assertRaisesRegex(ValueError, "silently drop"):
            config_tool.validate_config({**config, "cuda_graph_max_bs": 8}, fields)
        with self.assertRaisesRegex(ValueError, "relative"):
            config_tool.validate_config(
                {**config, "model_path": "/weights/v41"}, fields
            )


if __name__ == "__main__":
    unittest.main()
