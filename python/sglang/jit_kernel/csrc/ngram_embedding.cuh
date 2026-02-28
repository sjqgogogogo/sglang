#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace device::ngram_embedding {

__global__ void ComputeNGramIdsKernel(
    int batch_size,
    int ne_n,
    int ne_k,
    int* ne_weights,                  // [ne_n-1,ne_k,ne_n]
    int* ne_mods,                     // [ne_n-1,ne_k]
    int* exclusive_ne_embeder_size_sums, // [(ne_n-1)*ne_k]
    int* tokens,                      // [token_num]
    int* exclusive_req_len_sums,      // [batch_size+1]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                 // [batch_size]
    int* column_starts,               // [batch_size]
    int* n_gram_ids                   // [ne_n-1,ne_k,token_num]
) {
    // 先搞清当前block要处理的是哪个n、哪个k、第几条req
    /**
    以[req0,req1,req2] n=3,k=2为例
    n       k       req_id      blockIdx.x  config_id(指n和k的组合)
    2       1       0           0           0
    2       1       1           1           0
    2       1       2           2           0
    2       2       0           3           1
    2       2       1           4           1
    2       2       2           5           1
    3       1       0           0           2
    3       1       1           1           2
    3       1       2           2           2
    3       2       0           3           3
    3       2       1           4           3
    3       2       2           5           3
    */
    const int req_id = blockIdx.x % batch_size;
    const int config_id = (blockIdx.x - req_id) / batch_size;
    // 这里n和k不是物理含义上的n和k，而是有个偏移 n = real_n - 2; k = real_k - 1
    // 有这个偏移的原因是，n和k将来要作为索引从ne_weights和ne_mods里面取数，算索引的时候还要偏移回去，保证顺序是对的即可
    const int k = config_id % ne_k;
    const int n = (config_id - config_id % ne_k) / ne_k;
    // weights形状为[ne_n-1,ne_k,ne_n]，最后一个维度是token之间的距离，因此只能先算出来base idx
    const int ne_weight_base_idx = n * ne_k * ne_n + k * ne_n;
    // mod形状为[ne_n-1,ne_k]
    const int ne_mod = ne_mods[n * ne_k + k];
    // stride loop
    for (int i = exclusive_req_len_sums[req_id] + threadIdx.x; i < exclusive_req_len_sums[req_id + 1]; i += blockDim.x) {
        uint64_t n_gram_id = 0;
        // 目前在处理当前请求的第几个token
        int current_token_offset = i - exclusive_req_len_sums[req_id];
        // 先计算当前请求在token table中的起始index，在这个index以前的token就是跨请求的token，不参与计算
        int req_token_table_index = row_indices[req_id] * max_context_len;
        // 再计算当前token在token table中的位置
        int current_token_table_index = req_token_table_index + column_starts[req_id] + current_token_offset;
        for (int j = 0; j < n + 2; j++) {
            if (current_token_table_index-j < req_token_table_index) {
                // 非当前请求或者前面没有信息，不用来计算n_gram_id
                break;
            }
            if (ne_token_table[current_token_table_index-j] < 0) {
                // 写入的时候判断过这是个需要忽略的token
                break;
            }
            const uint64_t term = (uint64_t)ne_token_table[current_token_table_index-j] * (uint64_t)ne_weights[ne_weight_base_idx + j];
            n_gram_id += term % ne_mod;
        }
        n_gram_id %= ne_mod;
        n_gram_id += exclusive_ne_embeder_size_sums[n * ne_k + k];
        // [token_num, ne_n-1, ne_k]
        n_gram_ids[i*(ne_n-1)*ne_k + n*ne_k + k] = (int)(n_gram_id);
    }
}

__global__ void UpdateTokenTableKernel(
    int batch_size,
    int* tokens,                      // [token_num]
    int* ne_token_table,              // [max_running_reqs, max_context_len]
    int max_context_len,              // max_context_len
    long* row_indices,                // [batch_size]
    int* column_starts,               // [batch_size]
    int* req_lens,                     // [batch_size]
    int ignore_token_num,             // 有多少token需要被ignore
    int* ignore_tokens               // [ignore_token_num]
) {
    /**
     * 每个block处理一个req
     */
    const int req_id = blockIdx.x % batch_size;
    int start = 0;
    int end = 0;
    for (int i = 0; i < req_id; i++) {
        start += req_lens[i];
    }
    end = start + req_lens[req_id];
    // stride loop
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        // 目前在处理当前请求的第几个token
        int current_token_offset = i - start;
        // 先计算当前请求在token table中的起始index，在这个index以前的token就是跨请求的token，不参与计算
        int req_token_table_index = row_indices[req_id] * max_context_len;
        // 再计算当前token在token table中的位置
        int current_token_table_index = req_token_table_index + column_starts[req_id] + current_token_offset;
        ne_token_table[current_token_table_index] = tokens[i];
        for (int j = 0; j < ignore_token_num; j++) {
            if (ignore_tokens[j] == tokens[i]) {
                ne_token_table[current_token_table_index] = -tokens[i];
                break;
            }
        }
    }
}

}  // namespace device::ngram_embedding

namespace {

struct NgramEmbeddingKernel {
  static void compute_n_gram_ids(
      const int64_t ne_n,
      const int64_t ne_k,
      const tvm::ffi::TensorView ne_weights,
      const tvm::ffi::TensorView ne_mods,
      const tvm::ffi::TensorView exclusive_ne_embeder_size_sums,
      const tvm::ffi::TensorView tokens,
      const tvm::ffi::TensorView exclusive_req_len_sums,
      const tvm::ffi::TensorView ne_token_table,
      const tvm::ffi::TensorView row_indices,
      const tvm::ffi::TensorView column_starts,
      const tvm::ffi::TensorView n_gram_ids) {
    using namespace host;

    auto device_ = SymbolicDevice{};

    // Verify tensor shapes and types using -1 (kAnySize) for dynamic dimensions
    TensorMatcher({-1, -1, -1})  // [ne_n-1, ne_k, ne_n]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(ne_weights);

    TensorMatcher({-1, -1})  // [ne_n-1, ne_k]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(ne_mods);

    TensorMatcher({-1})  // [(ne_n-1)*ne_k + 1]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(exclusive_ne_embeder_size_sums);

    TensorMatcher({-1})  // [token_num]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(tokens);

    TensorMatcher({-1})  // [batch_size+1]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(exclusive_req_len_sums);

    TensorMatcher({-1, -1})  // [max_running_reqs, max_context_len]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(ne_token_table);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>()
        .verify(row_indices);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(column_starts);

    TensorMatcher({-1, -1})  // [token_num, (ne_n-1)*ne_k]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(n_gram_ids);

    const int batch_size = static_cast<int>(exclusive_req_len_sums.size(0) - 1);
    const int max_context_len = static_cast<int>(ne_token_table.size(1));
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());

    constexpr int BLOCK_THREADS = 256;
    const int num_configs = (static_cast<int>(ne_n) - 1) * static_cast<int>(ne_k);
    const int grid_size = num_configs * batch_size;

    LaunchKernel(grid_size, BLOCK_THREADS, stream)(
        device::ngram_embedding::ComputeNGramIdsKernel,
        batch_size,
        static_cast<int>(ne_n),
        static_cast<int>(ne_k),
        static_cast<int*>(ne_weights.data_ptr()),
        static_cast<int*>(ne_mods.data_ptr()),
        static_cast<int*>(exclusive_ne_embeder_size_sums.data_ptr()),
        static_cast<int*>(tokens.data_ptr()),
        static_cast<int*>(exclusive_req_len_sums.data_ptr()),
        static_cast<int*>(ne_token_table.data_ptr()),
        max_context_len,
        static_cast<long*>(row_indices.data_ptr()),
        static_cast<int*>(column_starts.data_ptr()),
        static_cast<int*>(n_gram_ids.data_ptr())
    );
  }

  static void update_token_table(
      const tvm::ffi::TensorView tokens,
      const tvm::ffi::TensorView ne_token_table,
      const tvm::ffi::TensorView row_indices,
      const tvm::ffi::TensorView column_starts,
      const tvm::ffi::TensorView req_lens,
      const tvm::ffi::TensorView ignore_tokens) {
    using namespace host;

    auto device_ = SymbolicDevice{};

    // Verify tensor shapes and types using -1 (kAnySize) for dynamic dimensions
    TensorMatcher({-1})  // [token_num]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(tokens);

    TensorMatcher({-1, -1})  // [max_running_reqs, max_context_len]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(ne_token_table);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int64_t>()
        .with_device<kDLCUDA>()
        .verify(row_indices);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(column_starts);

    TensorMatcher({-1})  // [batch_size]
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>()
        .verify(req_lens);

    // ignore_tokens can be empty or have values
    void* ignore_tokens_ptr = ignore_tokens.data_ptr();
    const bool has_ignore_tokens = ignore_tokens_ptr != nullptr && ignore_tokens.numel() > 0;
    if (has_ignore_tokens) {
      TensorMatcher({-1})  // [ignore_token_num]
          .with_dtype<int32_t>()
          .with_device<kDLCUDA>()
          .verify(ignore_tokens);
    }

    const int batch_size = static_cast<int>(req_lens.size(0));
    if (batch_size <= 0) {
      return;
    }

    const int max_context_len = static_cast<int>(ne_token_table.size(1));
    const auto stream = LaunchKernel::resolve_device(device_.unwrap());

    constexpr int BLOCK_THREADS = 256;
    const int grid_size = batch_size;

    int ignore_token_num = 0;
    int* ignore_tokens_typed_ptr = nullptr;
    if (has_ignore_tokens) {
      ignore_token_num = static_cast<int>(ignore_tokens.numel());
      ignore_tokens_typed_ptr = static_cast<int*>(ignore_tokens_ptr);
    }

    LaunchKernel(grid_size, BLOCK_THREADS, stream)(
        device::ngram_embedding::UpdateTokenTableKernel,
        batch_size,
        static_cast<int*>(tokens.data_ptr()),
        static_cast<int*>(ne_token_table.data_ptr()),
        max_context_len,
        static_cast<long*>(row_indices.data_ptr()),
        static_cast<int*>(column_starts.data_ptr()),
        static_cast<int*>(req_lens.data_ptr()),
        ignore_token_num,
        ignore_tokens_typed_ptr
    );
  }
};

}  // namespace
