"""Exercise FluentRouter through Triton gRPC streaming (also for non-stream IO).

Dependencies on the client machine: numpy and tritonclient[grpc]. A --request
JSON file can supply a full OpenAI request, including tools or multimodal input.
"""

import argparse
import json
import queue
import time
from pathlib import Path


def decode_string(result, name):
    values = result.as_numpy(name)
    if values is None or values.size == 0:
        return ""
    value = values.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def raise_for_error(payload):
    if "error" in payload or payload.get("object") == "error":
        raise RuntimeError(payload.get("error", payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="localhost:26384")
    parser.add_argument("--model", default="deepseek_v41")
    parser.add_argument(
        "--protocol", choices=("chat", "completion", "text"), default="chat"
    )
    parser.add_argument("--prompt", default="你好，请用一句话介绍自己。")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    import numpy as np
    import tritonclient.grpc as grpcclient

    def string_input(name, value):
        tensor = grpcclient.InferInput(name, [1], "BYTES")
        tensor.set_data_from_numpy(np.array([value.encode("utf-8")], dtype=object))
        return tensor

    if args.protocol == "text":
        if args.request:
            parser.error("--request is for chat/completion JSON requests")
        stream = grpcclient.InferInput("stream", [1], "BOOL")
        stream.set_data_from_numpy(np.array([args.stream], dtype=bool))
        inputs = [
            string_input("text_input", args.prompt),
            string_input(
                "sampling_parameters",
                json.dumps({"max_new_tokens": 64, "temperature": 0}),
            ),
            stream,
        ]
        output_names = [
            "text_output",
            "finish_reason",
            "input_length",
            "sequence_length",
        ]
    else:
        if args.request:
            request = json.loads(args.request.read_text())
            if not isinstance(request, dict):
                parser.error("--request must contain a JSON object")
            request.setdefault("model", args.model)
            request.setdefault("stream", args.stream)
        else:
            request = {
                "model": args.model,
                "stream": args.stream,
                "temperature": 0,
                "max_tokens": 64,
            }
            if args.protocol == "chat":
                request["messages"] = [{"role": "user", "content": args.prompt}]
                request["chat_template_kwargs"] = {"thinking": False}
            else:
                request["prompt"] = args.prompt
        if args.protocol == "completion":
            request["method"] = "complete"
        inputs = [string_input("json_input", json.dumps(request, ensure_ascii=False))]
        output_names = ["json_output"]

    responses = queue.Queue()

    def callback(result, error):
        responses.put((result, error))

    deadline = time.monotonic() + args.timeout
    with grpcclient.InferenceServerClient(url=args.url) as client:
        client.start_stream(callback=callback, stream_timeout=args.timeout)
        client.async_stream_infer(
            args.model,
            inputs,
            outputs=[grpcclient.InferRequestedOutput(name) for name in output_names],
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No terminal FluentRouter response")
            try:
                result, error = responses.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError("No terminal FluentRouter response") from exc
            if error:
                raise RuntimeError(str(error))
            if args.protocol == "text":
                print(decode_string(result, "text_output"), end="", flush=True)
                reason = decode_string(result, "finish_reason")
                if reason:
                    print(f"\nfinish_reason={reason}")
                    break
            else:
                payload = decode_string(result, "json_output")
                if not payload:
                    continue
                print(
                    payload, end="" if payload.startswith("data:") else "\n", flush=True
                )
                if payload == "data: [DONE]\n\n":
                    break
                if not payload.startswith("data:"):
                    parsed = json.loads(payload)
                    raise_for_error(parsed)
                    break
                # SSE errors are otherwise easy to mistake for a successful
                # stream when the next response contains [DONE].
                for line in payload.splitlines():
                    if line.startswith("data:"):
                        parsed = json.loads(line[len("data:") :].strip())
                        raise_for_error(parsed)
    print("Triton response completed.")


if __name__ == "__main__":
    main()
