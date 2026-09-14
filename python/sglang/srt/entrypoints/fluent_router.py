"""Python boundary for the embedded FluentRouter Triton backend.

The C++ backend imports entrypoints.engine.Engine directly. Its JSON protocol
expects an awaitable returning a JSON string, or an awaitable returning an async
iterator of SSE strings. No HTTP listener or Triton Python backend is involved.
"""

from __future__ import annotations

import json
import logging
from contextlib import aclosing

from fastapi import HTTPException
from starlette.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)
_DONE = "data: [DONE]\n\n"


def _error_json(exc):
    if isinstance(exc, HTTPException):
        code, message = exc.status_code, str(exc.detail)
    elif isinstance(exc, ValueError):
        code, message = 400, str(exc)
    else:
        code, message = 500, str(exc)
        logger.exception("FluentRouter request failed")
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": "BadRequestError" if code < 500 else "InternalServerError",
                "code": code,
            }
        },
        ensure_ascii=False,
    )


def _serialize_response(response):
    if isinstance(response, Response):
        return response.body.decode(response.charset)
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json()
    raise TypeError(f"Unsupported OpenAI response type: {type(response).__name__}")


async def _serialized_stream(response):
    if not isinstance(response, StreamingResponse):
        # A rejected streaming request is a plain error JSON. FluentRouter
        # treats a non-SSE string as terminal, including before the first token.
        try:
            yield _serialize_response(response)
        except Exception as exc:
            yield _error_json(exc)
        return
    body = response.body_iterator
    try:
        async with aclosing(body):
            async for chunk in body:
                chunk = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                if not isinstance(chunk, str):
                    raise TypeError("OpenAI stream chunks must be strings or bytes")
                yield chunk
                if chunk == _DONE:
                    return
        # Preserve the backend's terminal marker even for an empty stream.
        yield _DONE
    except Exception as exc:
        # One terminal error, not a dict (C++ expects str on the JSON path).
        yield _error_json(exc)
    finally:
        # We consume the body without ASGI, so Starlette will not run its
        # background cleanup for us. This owns abort/cleanup on early close.
        if response.background is not None:
            try:
                await response.background()
            except Exception:
                # The final frame may already have been sent. Do not emit a
                # second response or replace cancellation with a cleanup error.
                logger.exception("FluentRouter stream cleanup failed")


async def fluent_router_text_stream(generator):
    """Translate aborts for the legacy text protocol without changing native IO."""
    try:
        async with aclosing(generator):
            async for output in generator:
                if isinstance(output, list):
                    raise ValueError(
                        "FluentRouter text_input supports one choice; use json_input for n > 1"
                    )
                finish = output.get("meta_info", {}).get("finish_reason")
                if finish and finish.get("type") == "abort":
                    # RouterErrorCode: ServiceUnavailable=1, UnknownError=4.
                    # Native SGLang err_type is a string, not this C++ enum.
                    output = dict(output)
                    output["meta_info"] = dict(output["meta_info"])
                    output["meta_info"]["finish_reason"] = {
                        **finish,
                        "err_type": 1 if finish.get("status_code") == 503 else 4,
                    }
                yield output
    except HTTPException as exc:
        # Non-streaming TokenizerManager requests raise rather than yield the
        # abort reason. Keep the same retry classification on both paths.
        yield {
            "meta_info": {
                "finish_reason": {
                    "type": "abort",
                    "message": str(exc.detail),
                    "err_type": 1 if exc.status_code == 503 else 4,
                }
            }
        }


class FluentRouterEngineMixin:
    @property
    def scheduler_info(self):
        """Local rank's startup snapshot; never issues an event-loop RPC."""
        infos = self._scheduler_init_result.scheduler_infos
        return dict(infos[0]) if infos else {}

    def _init_fluent_router(self):
        cfg = self.server_args.resolved_dict()
        if cfg["tokenizer_worker_num"] != 1:
            raise ValueError("FluentRouter requires tokenizer_worker_num=1")
        if cfg["skip_tokenizer_init"]:
            raise ValueError("FluentRouter text/JSON inputs require a tokenizer")
        if cfg["incremental_streaming_output"]:
            raise ValueError("FluentRouter requires incremental_streaming_output=false")
        if self.tokenizer_manager is not None:
            self._get_fluent_router_handler("chat")
            self._get_fluent_router_handler("completion")

    def _get_fluent_router_handler(self, kind):
        if self.tokenizer_manager is None:
            raise ValueError("Submit FluentRouter requests to node_rank=0")
        if self.template_manager is None:
            raise ValueError("FluentRouter requires tokenizer_worker_num=1")
        name = f"_fluent_router_{kind}"
        handler = getattr(self, name, None)
        if handler is None:
            if kind == "chat":
                cls = self.tokenizer_manager.serving_chat_class
            else:
                from sglang.srt.entrypoints.openai.serving_completions import (
                    OpenAIServingCompletion,
                )

                cls = OpenAIServingCompletion
            handler = cls(self.tokenizer_manager, self.template_manager)
            setattr(self, name, handler)
        return handler

    def _parse_fluent_router_request(self, kind, payload):
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionRequest,
            CompletionRequest,
        )

        cls = ChatCompletionRequest if kind == "chat" else CompletionRequest
        return cls(**payload)

    async def _fluent_router_openai(
        self, kind, request_dict, bootstrap_host, bootstrap_port, bootstrap_room
    ):
        stream = request_dict.get("stream", False)
        try:
            payload = dict(request_dict)
            payload.pop("method", None)  # FluentRouter dispatch metadata
            if bootstrap_host is not None:
                payload.update(
                    bootstrap_host=bootstrap_host,
                    bootstrap_port=bootstrap_port,
                    bootstrap_room=bootstrap_room,
                )
            request = self._parse_fluent_router_request(kind, payload)
            # C++ selects iteration before awaiting this coroutine. Reject
            # values whose Pydantic coercion would change that choice.
            if not isinstance(stream, bool):
                raise ValueError("stream must be a boolean")
            handler = self._get_fluent_router_handler(kind)
            response = await handler.handle_request(request, None)
            return (
                _serialized_stream(response)
                if stream
                else _serialize_response(response)
            )
        except Exception as exc:
            error = _error_json(exc)
            if stream:

                async def error_stream():
                    yield error

                return error_stream()
            return error

    async def openai_v1_chat_completions(
        self,
        request_dict,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
    ):
        return await self._fluent_router_openai(
            "chat", request_dict, bootstrap_host, bootstrap_port, bootstrap_room
        )

    async def openai_v1_completions(
        self,
        request_dict,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
    ):
        return await self._fluent_router_openai(
            "completion", request_dict, bootstrap_host, bootstrap_port, bootstrap_room
        )
