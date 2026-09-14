"""Export tested SGLang CLI options to FluentRouter's version/config.json.

Run in the installed SGLang environment. This parses options without loading a
model or resolving device-dependent defaults. --check also validates the two
JSON environment overrides used by the existing FluentRouter startup script.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path


def validate_config(config, field_names):
    if not isinstance(config, dict):
        raise ValueError("Engine configuration must be a JSON object")
    unknown = set(config) - field_names
    if unknown:
        raise ValueError(
            f"Unsupported ServerArgs fields (Router would silently drop them): {sorted(unknown)}"
        )
    if config.get("enable_fluent_router") is not True:
        raise ValueError("enable_fluent_router must be true")
    if not isinstance(config.get("model_path"), str) or not config["model_path"]:
        raise ValueError("model_path must be a nonempty relative path")
    if config.get("tokenizer_worker_num", 1) != 1 or config.get(
        "skip_tokenizer_init", False
    ):
        raise ValueError("Use tokenizer_worker_num=1 and skip_tokenizer_init=false")
    if config.get("incremental_streaming_output", False):
        raise ValueError("Use incremental_streaming_output=false")
    for key in ("model_path", "speculative_draft_model_path", "chat_template"):
        value = config.get(key)
        if value and Path(value).is_absolute():
            raise ValueError(
                f"{key} must be relative to the model version directory; use a symlink for external files"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    from sglang.srt.server_args import ServerArgs, prepare_server_args

    field_names = {f.name for f in dataclasses.fields(ServerArgs) if f.init}
    if args.check:
        if args.engine_args:
            parser.error("--check does not take engine arguments")
        config = json.loads(args.check.read_text())
        for name in ("CUSTOM_CONFIG", "ARGS_CUSTOM_CONFIG"):
            if os.environ.get(name):
                override = json.loads(os.environ[name])
                unknown = set(override) - field_names
                if unknown:
                    raise ValueError(f"{name}: unsupported fields {sorted(unknown)}")
                config.update(override)
        validate_config(config, field_names)
        print(
            "FluentRouter config and environment overrides validated (model not loaded)."
        )
        return

    argv = args.engine_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    server_args = prepare_server_args(argv + ["--enable-fluent-router"])
    # Keep raw declarations: resolution must run exactly once in Engine.
    raw = dataclasses.asdict(server_args)
    defaults = dataclasses.asdict(ServerArgs(model_path=server_args.model_path))
    config = {key: value for key, value in raw.items() if value != defaults[key]}
    config["model_path"] = server_args.model_path
    validate_config(config, field_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(config, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Wrote {args.output}; run --check before starting Triton.")


if __name__ == "__main__":
    main()
