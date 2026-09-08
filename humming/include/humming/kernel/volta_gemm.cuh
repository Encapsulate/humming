#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

// Reference SM70 GEMM for the packed Humming layouts used by GSQ.  It keeps
// uint2/uint3 weights packed in global memory, expands only a 16x16 tile in
// shared memory, and uses Volta WMMA for FP16 x FP16 accumulation.
using namespace nvcuda;

template <bool kScaleBf16>
__device__ __forceinline__ float volta_load_scale(const void *scales, int index) {
  if constexpr (kScaleBf16) {
    return __bfloat162float(static_cast<const __nv_bfloat16 *>(scales)[index]);
  } else {
    return __half2float(static_cast<const half *>(scales)[index]);
  }
}

template <int kWeightBits, bool kScaleBf16>
__global__ void volta_humming_gemm(
    const half *inputs,
    const int *weights,
    const void *scales,
    half *outputs,
    int shape_m,
    int shape_n,
    int shape_k) {
  constexpr int kTile = 16;
  constexpr int kGroupSize = 128;
  constexpr int kMask = (1 << kWeightBits) - 1;

  const int tid = threadIdx.x;
  const int m0 = blockIdx.y * kTile;
  const int n0 = blockIdx.x * kTile;
  const int packed_k = shape_k * kWeightBits / 32;
  const int scale_k = shape_k / kGroupSize;

  __shared__ half shared_a[kTile * kTile];
  __shared__ half shared_b[kTile * kTile];
  __shared__ float shared_c[kTile * kTile];

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accum;
  wmma::fill_fragment(accum, 0.0f);

  for (int k0 = 0; k0 < shape_k; k0 += kTile) {
    for (int i = tid; i < kTile * kTile; i += blockDim.x) {
      const int row = i / kTile;
      const int col = i % kTile;
      const int m = m0 + row;
      const int k = k0 + col;
      shared_a[i] = (m < shape_m && k < shape_k) ? inputs[m * shape_k + k] : __float2half(0.0f);

      const int n = n0 + col;
      const int wk = k0 + row;
      float value = 0.0f;
      if (n < shape_n && wk < shape_k) {
        // Humming packs every 32 values into `kWeightBits` int32 words.  For
        // uint3 a code may straddle two words, so do not assume an integral
        // number of values per word.
        const int bit_index = (wk % 32) * kWeightBits;
        const int word_index = n * packed_k + (wk / 32) * kWeightBits + bit_index / 32;
        const int bit_offset = bit_index % 32;
        unsigned int code = (static_cast<unsigned int>(weights[word_index]) >> bit_offset) & kMask;
        if (bit_offset + kWeightBits > 32) {
          const int first_bits = 32 - bit_offset;
          const int second_bits = kWeightBits - first_bits;
          code |= (static_cast<unsigned int>(weights[word_index + 1]) & ((1u << second_bits) - 1u))
                  << first_bits;
        }
        const float scale = volta_load_scale<kScaleBf16>(scales, n * scale_k + wk / kGroupSize);
        value = static_cast<float>(static_cast<int>(code) - (1 << (kWeightBits - 1))) * scale;
      }
      shared_b[i] = __float2half(value);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half, wmma::row_major> b;
    wmma::load_matrix_sync(a, shared_a, kTile);
    wmma::load_matrix_sync(b, shared_b, kTile);
    wmma::mma_sync(accum, a, b, accum);
    __syncthreads();
  }

  wmma::store_matrix_sync(shared_c, accum, kTile, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < kTile * kTile; i += blockDim.x) {
    const int row = i / kTile;
    const int col = i % kTile;
    const int m = m0 + row;
    const int n = n0 + col;
    if (m < shape_m && n < shape_n) outputs[m * shape_n + n] = __float2half(shared_c[i]);
  }
}
