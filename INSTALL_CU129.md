# DeepSeek V4.1 源码安装：Hopper + CUDA 12.9

`dsv4.1_cu129` 基于官方 `dsv4.1` 的提交
`8ba15052e93be305b1696b5ce89629789cada6cb`，针对 Linux x86_64、Python 3.12
和 Hopper（H100/H200）固化 CUDA 12.9 依赖。无需 Docker。

## 安装

先激活自己的 Python 3.12 环境。在仓库根目录执行：

```bash
# 改成服务器实际的 CUDA 12.9 Toolkit 路径。
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install --upgrade pip
python -m pip install -r requirements-cu129.txt
```

`requirements-cu129.txt` 保存 wheel 下载源并执行 `-e ./python`。
运行依赖与构建依赖中的 PyTorch 都固定为 `2.13.0+cu129`，保留 pip 默认的
构建隔离；不需要手工预装 PyTorch、传 `--no-deps` 或修改依赖文件。
首次安装会下载较大的 wheel 并编译 Rust 扩展。

`cuda-tile` 从 NVIDIA 源直接安装二进制 wheel，不运行 PyPI 上的占位源码包。
这样下载由 pip 处理，避免占位包使用独立的 Python `urllib` 证书配置导致
`Preparing metadata` 阶段出现 `CERTIFICATE_VERIFY_FAILED`。
如果 pip 本身仍报告证书错误，需要配置服务器信任的 CA 证书，例如：

```bash
# 改为服务器实际使用的 CA bundle，企业代理环境需包含企业根证书。
export PIP_CERT=/path/to/ca-bundle.pem
export SSL_CERT_FILE="$PIP_CERT"
python -m pip install -r requirements-cu129.txt
```

不要通过关闭 TLS 证书校验来解决此问题。

要求宿主机有兼容 CUDA 12.9 的 NVIDIA 驱动、CUDA 12.9 Toolkit（含 `nvcc`）、
C/C++ 编译工具、Rust/cargo 和 `protoc`。Python 3.12.10、cargo/rustc 1.98.1、
protoc 24.3 已具备这些工具，不必为了安装重复运行 `install_rust_protoc.sh`；
实际编译结果仍需在服务器上检查。

```bash
nvcc --version
nvidia-smi
cargo --version
protoc --version
```

## 验证

```bash
python -m pip check
python - <<'PY'
import torch
import sglang
import flashinfer
import deep_gemm
from sglang.srt.configs.deepseek_v41 import DeepseekV41Config

print("SGLang:", sglang.__file__)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
assert torch.version.cuda == "12.9", torch.version.cuda
print("GPU:", torch.cuda.get_device_name(0))
print(torch.ones(1, device="cuda"))
PY
python -m sglang.launch_server --help
```

验证通过后再启动模型。依赖解析成功不等于已经在 H200 上完成编译或模型推理验证。

## 分支中的依赖调整

- PyTorch / torchvision / torchaudio / torchcodec 固定 cu129 构建。
- FlashInfer、Humming 和 CUTLASS DSL 使用 CUDA 12 依赖。
- `cuda-python` 限制在 CUDA 12 系列。
- DeepEP、DeepGEMM 和 sglang-kernel 使用官方 cu129 wheel。
- FlashAttention 4 固定为 `4.0.0b19`，与该 PR 的 `apache-tvm-ffi==0.1.11`
  依赖配套，避免解析时选择要求更新 TVM FFI 的版本。

后续更新源码后仍使用 `python -m pip install -r requirements-cu129.txt`；
单独执行 `pip install -e python` 不会读取这个文件中的额外下载源。
