# Qwen 3.8 GSQ validation on Tesla V100 32 GB

This is a reproducible, sanitized validation record for this fork's SM70 path.
It intentionally excludes usernames, hostnames, local paths, IP addresses,
process IDs, and timestamps.

## Result

`ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ` loaded and served successfully on a single
**Tesla V100-SXM2 32 GB (SM70)** GPU through a localhost-only vLLM 1.3.0
OpenAI-compatible server. A real chat-completion request asking for exactly
`OK` returned exactly `OK`.

The successful configuration used:

```text
--quantization humming
--dtype half
--language-model-only
--no-enable-prefix-caching
--mamba-cache-mode none
--gpu-memory-utilization 0.80
--max-model-len 2048
--max-num-seqs 1
--max-num-batched-tokens 2048
```

`--language-model-only` is deliberate: this validation covers the text model,
not Qwen's vision tower.

## What ran on the V100

- Group-128 uint3 transformer layers used the packed native Volta path. The
  one-token decode route uses a dedicated GEMV kernel rather than padded
  16-row WMMA tiles.
- Group-64 uint4 embedding lookup remained packed; only requested vocabulary
  rows were dequantized on GPU.
- Group-64 uint4 LM-head logits used packed native Volta GEMM.
- Qwen's quantized QKV/Z Gated Delta Net projection used the native packed
  path.
- Its separate, unquantized B/A projection used the SM70 FP16 dense path.
- Qwen's Gated Delta Net and attention selected the V100-specific paths.

The model-load report was **10.44 GiB**. After initialization, the process
held about **25.3 GiB** VRAM, leaving about **6.4 GiB** free on the 32 GB GPU.

## Sanitized evidence excerpts

```text
Resolved architecture: Qwen3_5ForConditionalGeneration
Casting torch.bfloat16 to torch.float16
Using FlashQLA-SM70 GDN prefill kernel
SM70 dense FP16 fast path enabled for small decode projections
Model loading took 10.44 GiB memory
FLASH_ATTN_V100 prefill path active
FLASH_ATTN_V100 decode path active
Application startup complete
POST /v1/chat/completions -> 200 OK
```

## Benchmark and current limitation

The port is functionally verified, but it is **not a high-throughput
production kernel**. A post-warm-up, one-request 32-token decode benchmark
measured about **2.18 generated tokens/second**. An earlier baseline measured
about **1.6--1.7 generated tokens/second**; the first native tiled optimization
reached about **2.0**, then the decode-specialized GEMV and compiled graph path
reached 2.18.

The Qwen-sized uint3/group-128 projection microbenchmark improved from
**0.541 ms** to **0.131 ms** per single-token call (about **4.1x**). It avoids
the previous 16-row padded decode work, uses a graph-visible Torch custom op,
and has parity coverage for uint2/uint3/uint4 with FP16 and BF16 scales. The
remaining end-to-end bottleneck is primarily Qwen's Gated Delta Net/recurrent
execution. Do not substitute the decode measurement for a production
throughput claim.

## Reproduction outline

Use a V100-compatible vLLM/Humming environment, install this checkout as the
editable `humming-kernels` package, and apply the companion vLLM integration
needed by the Qwen checkpoint:

1. accept SM70 in Humming's vLLM capability gate;
2. pass Humming configuration into Qwen's internal embedding construction;
3. honor regex `ignore` rules so Qwen's split QKV/Z and B/A layout is used;
4. retain quantized vocabulary tables in packed form;
5. route the unquantized narrow B/A projection through an SM70-safe FP16
   dense path.
6. keep the Humming custom-op registration enabled so vLLM can compile and
   CUDA-graph the packed decode launch.

Then launch vLLM with the configuration above, pointing `--model` to a local
copy of the checkpoint. All validation was local-only; no public endpoint is
required.
