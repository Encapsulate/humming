#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

// SM70 GEMM for the packed Humming layouts used by GSQ. It keeps uint2/uint3
// weights packed in global memory, expands only 16x16 tiles in shared memory,
// and uses Volta WMMA for FP16 x FP16 accumulation. A block owns four output
// N tiles, so the activation tile is loaded once and reused by four warps.
using namespace nvcuda;

template <bool kScaleBf16>
__device__ __forceinline__ float volta_load_scale(const void *scales, int index) {
  if constexpr (kScaleBf16) {
    return __bfloat162float(static_cast<const __nv_bfloat16 *>(scales)[index]);
  } else {
    return __half2float(static_cast<const half *>(scales)[index]);
  }
}

template <int kWeightBits, int kGroupSize, bool kScaleBf16>
__global__ void volta_humming_gemm(
    const half *inputs,
    const int *weights,
    const void *scales,
    half *outputs,
    int shape_m,
    int shape_n,
    int shape_k) {
  constexpr int kTile = 16;
  constexpr int kNTilesPerBlock = 4;
  constexpr int kMask = (1 << kWeightBits) - 1;

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int m0 = blockIdx.y * kTile;
  const int n0 = blockIdx.x * (kTile * kNTilesPerBlock) + warp * kTile;
  const int packed_k = shape_k * kWeightBits / 32;
  const int scale_k = shape_k / kGroupSize;

  __shared__ half shared_a[kTile * kTile];
  __shared__ half shared_b[kNTilesPerBlock * kTile * kTile];
  __shared__ float shared_c[kNTilesPerBlock * kTile * kTile];

  wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float> accum;
  wmma::fill_fragment(accum, 0.0f);

  for (int k0 = 0; k0 < shape_k; k0 += kTile) {
    // The four warps in this block consume the same A tile. Loading it once
    // removes the dominant redundant activation traffic for decode (M=1).
    if (warp == 0) {
      for (int i = lane; i < kTile * kTile; i += 32) {
        const int row = i / kTile;
        const int col = i % kTile;
        const int m = m0 + row;
        const int k = k0 + col;
        shared_a[i] = (m < shape_m && k < shape_k)
                          ? inputs[m * shape_k + k]
                          : __float2half(0.0f);
      }
    }
    for (int i = lane; i < kTile * kTile; i += 32) {
      const int row = i / kTile;
      const int col = i % kTile;
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
      shared_b[warp * kTile * kTile + i] = __float2half(value);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, half, wmma::row_major> b;
    wmma::load_matrix_sync(a, shared_a, kTile);
    wmma::load_matrix_sync(b, shared_b + warp * kTile * kTile, kTile);
    wmma::mma_sync(accum, a, b, accum);
    __syncthreads();
  }

  wmma::store_matrix_sync(shared_c + warp * kTile * kTile, accum, kTile, wmma::mem_row_major);
  __syncthreads();
  for (int i = lane; i < kTile * kTile; i += 32) {
    const int row = i / kTile;
    const int col = i % kTile;
    const int m = m0 + row;
    const int n = n0 + col;
    if (m < shape_m && n < shape_n) {
      outputs[m * shape_n + n] = __float2half(shared_c[warp * kTile * kTile + i]);
    }
  }
}

// Decode is a GEMV (M=1), not a GEMM.  Running a 16x16 WMMA tile for it
// computes fifteen rows which are immediately discarded.  This kernel gives
// each warp four adjacent output channels, reuses its activation values for
// all four dot products, and reduces each dot product in-warp.
template <int kWeightBits, int kGroupSize, bool kScaleBf16>
__global__ void volta_humming_gemv(
    const half *inputs,
    const int *weights,
    const void *scales,
    half *outputs,
    int shape_m,
    int shape_n,
    int shape_k) {
  constexpr int kOutputsPerWarp = 4;
  constexpr int kWarpsPerBlock = 8;
  constexpr int kOutputsPerBlock = kOutputsPerWarp * kWarpsPerBlock;
  constexpr int kMask = (1 << kWeightBits) - 1;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n0 = blockIdx.x * kOutputsPerBlock + warp * kOutputsPerWarp;
  const int packed_k = shape_k * kWeightBits / 32;
  const int scale_k = shape_k / kGroupSize;
  float sums[kOutputsPerWarp] = {0.f, 0.f, 0.f, 0.f};

  // All lanes visit one scale group at a time. Each input value is fetched
  // once and contributes to four output channels held in registers.
  for (int group0 = 0; group0 < shape_k; group0 += kGroupSize) {
    const int group = group0 / kGroupSize;
    float scales4[kOutputsPerWarp];
#pragma unroll
    for (int j = 0; j < kOutputsPerWarp; ++j) {
      const int n = n0 + j;
      scales4[j] = n < shape_n
          ? volta_load_scale<kScaleBf16>(scales, n * scale_k + group)
          : 0.f;
    }
    for (int offset = lane; offset < kGroupSize; offset += 32) {
      const int k = group0 + offset;
      const float input = __half2float(inputs[k]);
      const int bit_index = (k % 32) * kWeightBits;
      const int bit_offset = bit_index % 32;
#pragma unroll
      for (int j = 0; j < kOutputsPerWarp; ++j) {
        const int n = n0 + j;
        if (n >= shape_n) continue;
        const int word_index = n * packed_k + (k / 32) * kWeightBits + bit_index / 32;
        unsigned int code = (static_cast<unsigned int>(weights[word_index]) >> bit_offset) & kMask;
        if constexpr (kWeightBits == 3) {
          if (bit_offset + kWeightBits > 32) {
            const int first_bits = 32 - bit_offset;
            const int second_bits = kWeightBits - first_bits;
            code |= (static_cast<unsigned int>(weights[word_index + 1]) & ((1u << second_bits) - 1u))
                << first_bits;
          }
        }
        sums[j] += input * static_cast<float>(static_cast<int>(code) - (1 << (kWeightBits - 1))) * scales4[j];
      }
    }
  }

#pragma unroll
  for (int j = 0; j < kOutputsPerWarp; ++j) {
    float value = sums[j];
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0 && n0 + j < shape_n) outputs[n0 + j] = __float2half(value);
  }
}
