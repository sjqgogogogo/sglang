# 通过 FluentRouter 接入内部 Triton Server

沿用 `mt_triton_server` 的 `staging/mtos-x86` 和现有 FluentRouter C++ backend。
在已有 Triton 基础镜像中按已经验证过的方法安装当前 SGLang。

```text
Triton → FluentRouter ServerAgent → ZMQ IPC → 嵌入式 Python Engine
       → Tokenizer / Scheduler / Detokenizer → 原路返回
```

## 运行时边界

- FluentRouter 继续导入 `sglang.srt.entrypoints.engine.Engine`，无需改 C++ import。
- `enable_fluent_router=true` 开启兼容模式，初始化当前 PR 的 OpenAI handler。
- `scheduler_info` 返回 Scheduler 的实际启动信息，供 Router 判断容量和 warmup。
- `json_input` 接口保留 OpenAI JSON/SSE；`method=complete` 走 completions，其余走 chat。
- `text_input` 接口返回累计文本，由 Router 切为增量；必须保持
  `incremental_streaming_output=false`。该接口只支持一条文本和一个输出选择，
  多 choice、tools、reasoning 或多模态输入请走 `json_input`。
- 使用一个 Triton CPU 实例、一个 tokenizer worker；GPU 上的批处理由 SGLang 负责。
- 不启动 SGLang HTTP server。当前 Triton 分支的普通 HTTP infer 不支持 decoupled 模型，
  验证使用 gRPC streaming；非流式模型请求也通过这条 gRPC 通道发送。

V4.1 请求仍由当前 SGLang 的 chat encoding / reasoning / tool parser 处理。
适配层不改写消息或套用旧 FluentLLM 模板。需要结构化 reasoning/tool 输出时，
沿用已验证的 parser 参数；也可以用 `--reasoning-parser auto --tool-call-parser auto`。

## 基础镜像中的 backend

这里需要的是 FluentRouter 构建出来的 `backends/python/libtriton_python.so`，
不是上游 Triton Python backend。如果基础镜像只有 Triton server，需要一起安装
FluentRouter 的 backend 和依赖库。

FluentRouter 嵌入并链接 Python。当前 SGLang 安装使用 Python 3.12 时，backend 也必须
使用匹配的 Python 3.12 构建，不能仅修改 `PYTHONPATH` 就复用其他 Python ABI 的库。
在已有 FluentRouter 源码和构建环境中，按其 Makefile 执行依赖构建与安装：

```bash
# 在 FluentRouter 仓库中；python3 / python3-config 应对应镜像的 Python 3.12。
make deps
make deps
make install
```

使用平台已有的发布步骤安装生成的 `backends/python`、`lib64`。内部 core 依赖由
FluentRouter 的构建配置指定；当前配置是 `staging/mtos_x86`，不是 server 的同名分支。
需要 OCTO 的镜像沿用原来的 `deps-octo` / `install-octo` 构建方式。

## 模型目录和启动配置

在当前 SGLang 仓库根目录执行。下面的 TP 和容量参数只是目录准备示例；
请替换成已经跑通 V4.1 的那组参数，保持 kernel、并行和投机配置一致。

```bash
mkdir -p /models/deepseek_v41/1
cp examples/triton/config.pbtxt /models/deepseek_v41/config.pbtxt
ln -s /实际权重目录 /models/deepseek_v41/1/hf_weight

python examples/triton/prepare_config.py \
  --output /models/deepseek_v41/1/config.json -- \
  --model-path hf_weight \
  --served-model-name deepseek_v41 \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend dsv4 \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 4096 \
  --max-running-requests 8 \
  --cuda-graph-max-bs 8 \
  --reasoning-parser auto \
  --tool-call-parser auto
```

生成工具调用本版本的 CLI parser，自动添加 `enable_fluent_router=true`，
并保存原始配置，不加载模型或解析 GPU 默认值。它会拒绝覆盖已存在的文件。
CLI 别名会转换成真正的 ServerArgs 字段，例如 `--cuda-graph-max-bs` 会写成
`cuda_graph_max_bs_decode`，避免 C++ Router 按构造函数过滤时悄悄丢失配置。

Router 会将 `model_path`、`speculative_draft_model_path`、`chat_template` 拼到版本目录下。
这几个字段使用相对路径；外部权重或模板用符号链接。不要把这里的启动 `config.json`
与 `hf_weight/config.json` 模型配置混淆。

如果沿用平台的 `CUSTOM_CONFIG` / `ARGS_CUSTOM_CONFIG`，启动前在相同环境中检查：

```bash
python examples/triton/prepare_config.py --check /models/deepseek_v41/1/config.json
```

检查工具会按原 Router 顺序合并两份环境覆盖，并拒绝未知字段和不兼容选项。
不要继续使用旧 FluentLLM 的 `attn_tp_size`、`moe_parallel_strategy`、`stream_output` 等配置。
该检查只验证接入配置，不代替模型加载验证。C++ Router 自身的过滤逻辑未改，
所以平台启动脚本应执行这条检查。

## 启动

SGLang 必须在 backend 使用的 Python 环境中安装好。检查实际导入位置和接口：

```bash
python - <<'PY'
import sys, sglang
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.server_args import ServerArgs
print(sys.executable, sys.version, sglang.__file__)
assert hasattr(Engine, 'scheduler_info')
assert hasattr(Engine, 'openai_v1_chat_completions')
assert hasattr(Engine, 'openai_v1_completions')
assert ServerArgs(model_path='hf_weight', enable_fluent_router=True).enable_fluent_router
PY
```

继续使用平台原有的 Triton 启动流程，模型仓库指向 `/models`。本地容器联调可以使用：

```bash
export SGLANG_BLOCK_NONZERO_RANK_CHILDREN=0
# 初次联调跳过 Router 自己的大范围 warmup；SGLang 的初始化/capture 仍然执行。
export SKIP_WARM_UP=1
# Router 的 PD 本机地址识别读取 GRPC_PORT。
export GRPC_PORT=26384

# 目录以镜像中 FluentRouter 的实际安装位置为准。
export LD_LIBRARY_PATH=/opt/fluentrouter/lib64:${LD_LIBRARY_PATH}
/opt/tritonserver/bin/tritonserver_real \
  --model-repository=/models \
  --backend-directory=/opt/fluentrouter/backends \
  --model-control-mode=explicit \
  --load-model=deepseek_v41 \
  --allow-grpc=true \
  --grpc-port=26384 \
  --allow-http=false
```

如果需要通过 `PYTHONPATH` 选择源码，替换原来的 `/home/fluentllm/python` 路径，
确认没有优先导入旧 FluentLLM。GPU 镜像使用已验证的 CUDA 环境；不要沿用 NPU
脚本的 Ascend 环境初始化。保留平台需要的 Whale RPC/OCTO 参数即可。

多机启动保留 `SGLANG_BLOCK_NONZERO_RANK_CHILDREN=0`。非零节点只启动模型 worker，
请求发往 rank 0。Router 的 `PD_MASTER_HOST` 会设置分布式初始化地址，
`ARGS_NODE_RANK` 会覆盖节点 rank。

## 联调

在客户端安装 `numpy` 和 `tritonclient[grpc]`，从仓库根目录执行：

```bash
python examples/triton/smoke_client.py --url localhost:26384
python examples/triton/smoke_client.py --url localhost:26384 --stream
python examples/triton/smoke_client.py --url localhost:26384 --protocol completion --stream
python examples/triton/smoke_client.py --url localhost:26384 --protocol text
python examples/triton/smoke_client.py --url localhost:26384 --protocol text --stream
# OpenAI 请求 JSON 文件可包含 messages、tools、reasoning_effort 等。
python examples/triton/smoke_client.py --url localhost:26384 --request /path/to/request.json
```

请求文件中的 `stream` 优先于命令行开关。脚本等待最终响应，遇到 Triton 错误、
JSON/SSE error 或超时会失败。建议将同一组请求与直接启动的 SGLang 做对比，
覆盖普通文本、thinking、tool calls、非法参数和并发长输出。

本地 CPU 协议测试：

```bash
python -m pytest -q test/registered/unit/entrypoints/test_fluent_router.py
```

## 当前范围

本次接入保留 bootstrap 参数透传，但没有宣称完成 PD/KV 传输验证。
V4.1 的 PD/DSpark 组合必须遵守当前模型的配置约束。
`staging/mtos-x86` 和现有 Router 没有完整的客户端断连取消通路；
这里的生成器清理不等于 Triton 断连已经能够取消 GPU 上的请求。
固定 IPC 文件名和 ServerAgent 单例沿用原实现，一个容器按一个模型实例部署。
