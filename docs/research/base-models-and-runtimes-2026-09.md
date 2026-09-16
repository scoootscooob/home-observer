<!-- research note produced 2026-09-16 by a web-research agent; fetch budget 30 web calls; items marked (secondary) or (unverified) were not read from a primary source -->

# Open base models and runtimes for a unified streaming home observer (September 2026)

Companion to `decoding-and-training-2026-09.md` (grammar-constrained decoding on Apple silicon is covered there, section 1, and not repeated) and `teacher-distillation.md` (the per-tick contract: one silence token or one short JSON line per tick).

Reading key: **P** = primary source read on the date given (model card, repo, paper full text, pricing page); **S** = secondary (blog, forum, benchmark aggregator, or a number quoted from a snippet); **E** = this note's own estimate. Every number carries a unit.

## 0. Scope, method, and the target workload

- Budget used: 30 web calls on 2026-09-16 (17 page fetches, 13 searches). Primary pages read: Qwen3.5-4B and Qwen3-VL-4B model cards and `config.json`, Qwen3.5-Omni technical report (arXiv 2604.15804), Gemma 4 model card (ai.google.dev), MiniCPM-o 4.5 card, Nemotron 3 Nano Omni card, InternVL3.5-4B card, SmolVLM2-2.2B card, LFM2.5-VL-3B card, mlx-vlm README, llama.cpp `docs/multimodal.md`, StreamingVLM and EVS arXiv abstracts. Two fetches failed to yield data (OpenBMB/llama.cpp-omni 404; vast.ai prices are dynamic).
- Target workload (from the task and `teacher-distillation.md`): one fixed camera, 1 tick per second, change-gated visual tokens averaging 20 to 80 per second, one second of audio per tick, occasional device-state text diffs, and an output of one silence token per tick or a short JSON line (about 45 tokens) on roughly 0.5 percent of quiet ticks and up to 10 percent of busy ticks. Production on an M5 Max 48 GB now, a sub-US$1K box with a 12 to 16 GB GPU later, training on rented H100s.
- Convention for Qwen-family token counts: the encoder emits one token per 32×32-px block per **temporal patch of two frames** (`patch_size 16`, `spatial_merge_size 2`, `temporal_patch_size 2`); a lone still frame is duplicated to fill the temporal patch and costs the same as a frame pair.
- The present 14 to 20 s per 8-frame request (Gemma 4 E4B through transformers) is a re-prefill problem, not a model-size problem: 8 frames × 280 soft tokens plus context re-encoded and re-prefilled per request. Every recommendation below hinges on reusing the encoded and prefilled prefix across ticks.

## 1. Open base models, 2 to 9B, with video or streaming support (deliverable a)

### 1.1 Model table

Status per cell: P primary, S secondary, E estimate. "Tokens per frame" is at a 448×448 input unless stated. 4-bit footprints marked E are weights-only (parameters × 0.5 byte plus embeddings), excluding KV cache and activations.

| Model (open sizes) | Total params | Vision encoder; tokens per frame | Context | Time-aware position encoding | Audio in | Tool calling | License | Smallest 4-bit footprint | Streaming-relevant notes |
|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-4B** (dense; 9B verified on card; 0.8B, 2B S) | 4B (P) | 24-layer, 1024-d ViT ≈0.3B (E from config); 196 tokens per 448² frame-pair = 98 per video frame; 880 per 1280×704 pair (E from P config) | 262,144 native; 1,010,000 with YaRN (P) | interleaved MRoPE `[11,11,10]` (P); fps parameter, default 2 (P) | no (P) | yes (P) | Apache-2.0 (P) | ≈2.5 GB Q4_K_M + ≈0.7 GB mmproj (E); GGUF/MLX builds exist (S) | 24 Gated DeltaNet + 8 attention layers: fixed recurrent state ≈25 MB, KV 32 KB per token (E); thinking on by default, switchable (P) |
| **Qwen3-VL-4B** Instruct/Thinking (2B, 4B, 8B, 32B dense) | 4B (P) | same ViT geometry + DeepStack taps at layers 5, 11, 17 (P); same token counts | 262,144 (P); "expandable to 1M" (P card) | interleaved MRoPE `[24,20,20]` + text-timestamp alignment (P) | no | yes, GUI agent (P) | Apache-2.0 (P) | ≈2.5 GB + 0.7 GB (E); 108 quantized derivatives listed (P) | 36 attention layers, KV 147 KB per token (E); EVS supported in vLLM (S); no recurrence |
| **Gemma 4 E2B / E4B / 12B** | 2.3B eff. (5.1B) / 4.5B eff. (8B) / 11.95B (P) | ~150M encoder (E2B/E4B), 12B "encoder-free" patch projection; 70, 140, 280, 560 or 1120 soft tokens per image, resolution-independent (P); video = 1 fps frames, ≤60 s (P) | 128K / 128K / 256K (P) | none documented; frames are images | yes on all three, ≤30 s per clip, ~300M audio encoder (P) | native function calling (P) | Apache 2.0 (P) | E4B Q4_0 4.59 GB + mmproj Q8_0 0.56 GB (P, ggml-org via companion note); E2B ≈2.9 GB (E) | sliding window 512 (E2B/E4B) or 1024 (12B) on local layers, shared K/V on global layers (P); no recurrence; gating only at whole-frame granularity |
| **MiniCPM-o 4.5** | 9B (P) | SigLIP2; 3D-Resampler stacks up to 5 frames, video to 10 fps, images ≤1.8 Mpx (P); ≈64 tokens per frame group, "96× compression" (S) | not stated on card (Qwen3-8B base, S) | millisecond time-division multiplexing of all streams (P) | yes, Whisper-medium (P); speech out via CosyVoice2 (P) | via Qwen3 base (S) | Apache-2.0 on card (P; check LICENSE file) | int4 = 11.0 GB GPU memory in PyTorch (P); GGUF in 16 sizes (P), ≈5–6 GB Q4 weights (E) | full-duplex, proactive decision at 1 Hz, TTFT 0.6 s, decode 154 tok/s bf16 / 212 tok/s int4 on an unstated GPU (P) |
| **InternVL3.5-4B** (1B to 241B-A28B family, S) | 4.7B = 0.3B ViT + 4.4B LLM (P) | InternViT-300M; 256 tokens per 448² tile, 64 with the Flash/ViR router (P); video 32 frames (P) | 32K in SFT (P) | none documented | no (P) | GUI and embodied agent (P) | Apache-2.0 (P) | ≈2.8 GB (E); 7 quantized derivatives, no first-party GGUF/MLX (P) | DvD split deployment "4.05× speedup" (P); no streaming design |
| **SmolVLM2-2.2B** (256M, 500M) | 2.2B (P) | SigLIP SO400m-patch14-384; 81 tokens per 384² tile (S, memory) | not stated | none | no | no | Apache 2.0 (P) | ≈1.4 GB (E); "5.2GB of GPU RAM for video inference" (P) | Video-MME 52.1 (P); llama.cpp has SmolVLM video GGUFs (P) |
| **LFM2.5-VL-3B** (450M, 1.6B) | 3.1B (P) | SigLIP2 NaFlex 400M; 512² patches + thumbnail (P); tokens per patch not stated (256, S) | 32,768 (P) | none; **no video mode** (P) | no (P) | yes, Pythonic calls (P) | lfm1.0 (P); commercial free below US$10M revenue (S) | GGUF and MLX-8bit exist (P); ≈1.9 GB Q4 (E) | 228 tok/s decode on M5 Max, 11K tok/s and 34 ms TTFT on H100 (P) |
| **Qwen3.5-Omni** Plus / Flash | not stated in report text (P) | SigLIP2, dynamic fps (P); AuT audio at 6.25 Hz (P) | 256K; 10 h audio or 400 s of 720p at 1 fps (P) | timestamp text before each temporal patch (P) | yes (P) | yes (P) | **open-weight status and license unverified** | — | first-packet latency 235–651 ms (P) |
| Qwen3-Omni-30B-A3B | 30B / 3B active (S) | — | — | — | yes (S) | yes (S) | Apache 2.0 (S) | ≈17 GB Q4 (E) | out of the 2–9B band; llama.cpp ships GGUFs (P) |
| Nemotron 3 Nano Omni 30B-A3B | 31B / ~3B active (P) | C-RADIO v4-H (P); video ≤2 min, 1 fps/128 frames or 2 fps/256 (P) | 256K (P) | — | yes, Parakeet TDT 0.6B, ≤1 h (P) | yes (P) | NVIDIA Open Model Agreement (P) | NVFP4 needs an RTX 5090 32 GB (P) | reference EVS deployment; out of band |
| Research streaming models: StreamingVLM, ProVideLLM, VST-7B, LiveStar, ROMA | 7B class (S) | — | — | — | — | — | code repos (P for StreamingVLM); weights S | — | see 2.5 |

Excluded from the candidate list for the target hardware: Nemotron 3 Nano Omni (32 GB minimum), Qwen3-Omni-30B-A3B and Qwen3.5-Omni (size and openness), Gemma 4 26B-A4B and 31B (no audio, too large for 12 GB at any useful window).

### 1.2 Per-model notes and primary-source details

Findings are appended per fetch batch; the table in 1.1 is composed from these notes.

**Qwen3.5-4B** (P, model card https://huggingface.co/Qwen/Qwen3.5-4B read 2026-09-16). "Causal Language Model with Vision Encoder", 4B parameters, hidden 2560, 32 layers laid out as `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`, i.e. 24 linear-attention layers and 8 full-attention layers (3:1). Gated DeltaNet heads: 32 for V, 16 for QK, head dim 128. Gated Attention: 16 Q heads, 4 KV heads, head dim 256, RoPE dim 64. FFN intermediate 9216. Context 262,144 tokens natively, "extensible up to 1,010,000 tokens" with YaRN. Config has `mrope_interleaved: true`, `mrope_section: [11, 11, 10]` (time-aware interleaved MRoPE). Video input with configurable fps (default 2) and dynamic frame sampling. Tool calling ("excels in tool calling", `--enable-auto-tool-choice --tool-call-parser qwen3_coder` for vLLM/SGLang). Thinking on by default (`enable_thinking: False` to disable). No audio input mentioned. License apache-2.0. Citation dated February 2026. Card links the family blog https://qwen.ai/blog?id=qwen3.5 and lists 415 quantized derivatives; no first-party GGUF/MLX links on the card. Vision encoder parameters and tokens-per-frame are not on the card (see config.json fetch below).

**Qwen3-VL-4B-Instruct** (P, https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct read 2026-09-16). 4B dense; "Interleaved-MRoPE: full-frequency allocation over time, width, and height", DeepStack multi-level ViT feature fusion, "Text–Timestamp Alignment" for timestamp-grounded event localization; native context 256K, "expandable to 1M"; "hours-long video with full recall"; visual agent (GUI operation, tool invocation); Instruct and Thinking editions; 2B/4B/8B/32B dense sizes exist (8B shown in charts); Apache-2.0; 108 quantized derivatives (llama.cpp/Ollama/LM Studio). No audio input. Encoder and per-frame token counts not on the card (config.json fetch below).

**Qwen3.5-Omni** (S so far, search 2026-09-16). Released 2026-03-30 (MarkTechPost https://www.marktechpost.com/2026/03/30/...); technical report arXiv 2604.15804 (https://arxiv.org/html/2604.15804v1); 256K context, ">10 hours of continuous audio" and ">400 seconds of 720p audio-visual content"; ASR for 113 languages; one secondary source says Apache 2.0, another says open-weight status was not confirmed at release. Sizes and weights to be checked against the report (next batch).

**Gemma 4 family** (S, search 2026-09-16; primary card fetch in the next batch). Five sizes: E2B, E4B, 12B, 26B-A4B (MoE), 31B (https://ai.google.dev/gemma/docs/core, https://huggingface.co/google/gemma-4-12B). Context 128K (E2B/E4B) and 256K (12B, 26B-A4B, 31B). Audio input on E2B, E4B and 12B only (max 30 s per clip); video "as frames" at 1 frame per second up to 60 s on all sizes. Apache 2.0. E4B facts already verified in `decoding-and-training-2026-09.md` section 2.1: 4.5B effective (8B with embeddings), 42 layers, ~150M vision encoder, per-image budget 70/140/280/560/1120 soft tokens (default 280), vocabulary 262,144.

**MiniCPM-o 4.5** (S, search 2026-09-16; card fetch next batch). 9B total, built end-to-end from SigLIP2 + Whisper-medium + CosyVoice2 + Qwen3-8B; "processes 1.8M pixel images and 10FPS video with 96x video token compression"; full-duplex: video+audio input streams and text+speech output streams "do not block each other". GGUF at https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf; Ollama `openbmb/minicpm-o4.5` (8b, q8_0 tags); demo repo https://github.com/OpenBMB/MiniCPM-o-Demo.

**NVIDIA Nemotron 3 Nano Omni** (S, search 2026-09-16; card fetch next batch). Released 2026-04-28; 30B total / 3B active, hybrid Mamba-2 + MoE with 6 attention layers ("23 Mamba-2 and MoE layers, along with 6 Attention layers"); text, image, video, audio in one model; BF16 and NVFP4 repos: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 , https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 ; "available for commercial use". Note: 30B total parameters is outside the 2 to 9B band and above a 12 to 16 GB GPU at BF16; a 4B text-only Nemotron 3 Nano exists (https://huggingface.co/blog/nvidia/nemotron-3-nano-4b) but is not omni.

**InternVL3.5-4B** (P, https://huggingface.co/OpenGVLab/InternVL3_5-4B read 2026-09-16). 4.7B total = InternViT 0.3B + LLM 4.4B (Qwen3-4B base); 448×448 tiles, 1024 patches pixel-shuffled to 256 tokens per tile; dynamic tiling `min_num=1, max_num=12`; video sampled at 32 frames; Visual Resolution Router: the Flash variant routes tiles to 64 tokens, "50% reduction in visual tokens while maintaining nearly 100% performance"; Decoupled Vision-Language Deployment gives "4.05× inference speedup compared to InternVL3"; 32K context during SFT; thinking mode; GUI/embodied agent support; no audio; Apache-2.0 weights and code; released 2025-08-25; 7 quantized derivatives listed, no first-party GGUF/MLX.

**LFM2.5-VL** (S, search 2026-09-16). Sizes 450M (https://huggingface.co/LiquidAI/LFM2.5-VL-450M), 1.6B (ONNX repo exists), 3B (https://huggingface.co/LiquidAI/LFM2.5-VL-3B, blog https://www.liquid.ai/blog/lfm2-5-vl-3b). 3B pairs a SigLIP2 400M NaFlex encoder with the LFM2.5-2.6B backbone; user-tunable max image tokens and tile count at inference; "about 34 ms on a 5-frame clip on a single H100" (blog claim, S). License `lfm1.0`: free for research, commercial use free below US$10M revenue (S). Video support depth to verify.

**SmolVLM2-2.2B-Instruct** (P, https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct read 2026-09-16). 2.2B, Idefics3 architecture, SigLIP SO400m-patch14-384 encoder; sizes 256M/500M/2.2B; Apache 2.0; "5.2GB of GPU RAM for video inference"; Video-MME 52.1, MLVU 55.2, MVBench 46.27; paper arXiv 2504.05299 (2025-04-07). Card does not state tokens per frame, max frames, or context; no audio; no tool calling. Per the SmolVLM paper (S, from memory, not re-read): the 2.2B uses pixel-shuffle factor 3 on 384-px tiles, 81 tokens per tile.

**Streaming-specific research models found** (S, search 2026-09-16; none verified as open weights in the 2 to 9B band yet): VST-7B "Video Streaming Thinking" (arXiv 2603.12262), StreamingVLM (arXiv 2510.09608), LiveStar (arXiv 2511.05299), ROMA real-time omni assistant (arXiv 2601.10323), "A Simple Baseline for Streaming Video Understanding" (arXiv 2604.02317), WAT (arXiv 2603.13412), StreamingEval protocol (arXiv 2603.21493). VideoLLM-online, Flash-VStream, Dispider and MMDuet are covered in the companion note.

**Batch 2 (primary configs and cards, read 2026-09-16).**

*Qwen3.5-4B config.json* (P, https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json): `model_type qwen3_5`, `Qwen3_5ForConditionalGeneration`. Vision: depth 24, hidden 1024, 16 heads, `patch_size 16`, `spatial_merge_size 2`, `temporal_patch_size 2`, `out_hidden_size 2560`, `num_position_embeddings 2304`. Text: 32 layers, `full_attention_interval 4` (layer_types alternate 3 linear : 1 full), 16 attention heads, 4 KV heads, head_dim 256, linear attention 16 key heads / 32 value heads at dim 128, conv kernel 4, intermediate 9216, `max_position_embeddings 262144`, `vocab_size 248320`, tied embeddings, `mrope_interleaved true`, `mrope_section [11, 11, 10]`. Token IDs: image 248056, video 248057, vision_start 248053. Derived (E): one visual token per 32×32 px block per temporal patch of 2 frames; a 448×448 frame pair = 14×14 = 196 tokens; a single still image is duplicated to fill the temporal patch, so a lone 448×448 frame also costs 196 tokens, while video at 2 frames per temporal patch costs 98 tokens per frame. KV cache per token for the 8 full-attention layers (E): 8 × 4 heads × 256 × 2 (K and V) × 2 bytes = 32 KB per token in bf16 (1.0 GB at 32K tokens); the 24 Gated DeltaNet layers hold a fixed recurrent state (32 value heads × 128 × 128 × 2 bytes ≈ 1 MB per layer, ≈ 25 MB total) independent of sequence length.

*Qwen3-VL-4B-Instruct config.json* (P, https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json): `qwen3_vl`. Vision identical in shape to Qwen3.5 (depth 24, hidden 1024, patch 16, merge 2, temporal patch 2, out 2560, 2304 position embeddings) plus `deepstack_visual_indexes [5, 11, 17]`. Text: 36 layers, 32 heads, 8 KV heads, head_dim 128, intermediate 9728, `max_position_embeddings 262144`, vocab 151,936, tied embeddings, `mrope_section [24, 20, 20]`, `mrope_interleaved true`. Token IDs image 151655, video 151656. KV per token (E): 36 × 8 × 128 × 2 × 2 bytes = 147 KB in bf16 (4.7 GB at 32K tokens), 4.6× Qwen3.5-4B.

*Qwen3.5-Omni technical report* (P, https://arxiv.org/html/2604.15804v1, dated 2026-04-17): Thinker–Talker; "Plus and Flash variants" (parameter counts not stated in the fetched text); Thinker is a "Hybrid Attention Mixture-of-Experts" with a Gated DeltaNet module; audio encoder AuT at 6.25 Hz (160 ms frames); vision encoder SigLIP2 with dynamic frame rate; "up to 256k tokens, 10 hours of audio, or 400 seconds of 720P video at 1 FPS"; each video or audio-video temporal patch is prepended with "an explicit timestamp represented as a formatted text string"; first-packet latency audio Plus 435 ms / Flash 235 ms, video Plus 651 ms / Flash 426 ms; chunked prefill; WebSearch and FunctionCall invocation. The report text fetched does not state open-weight status, license, or throughput. **Open-weight status of Qwen3.5-Omni remains unverified** (see section 7); the predecessor Qwen3-Omni-30B-A3B (Instruct/Thinking/Captioner, Apache 2.0, September 2025) is open but 30B total (S, memory, not re-read).

*Gemma 4 model card* (P, https://ai.google.dev/gemma/docs/core/model_card_4): E2B 2.3B effective (5.1B with embeddings), 35 layers, sliding window 512, 128K, text+image+audio, vision encoder ~150M, audio encoder ~300M, image budget 70/140/280/560/1120 soft tokens, audio max 30 s, native function calling, thinking via `<|think|>`; E4B 4.5B effective (8B), 42 layers, sliding window 512, 128K, same encoders and budgets; 12B 11.95B total, 48 layers, sliding window 1024, 256K, text+image+audio, described as "encoder-free" with direct patch/waveform projection; 26B A4B 25.2B total / 3.8B active, 30 layers, 128 experts (8 active + 1 shared), 256K, text+image only, vision ~550M; 31B 30.7B dense, 60 layers, 256K, text+image only. All: video max 60 s at 1 fps, global layers use unified K/V sharing with p-RoPE, Apache 2.0, training cutoff January 2025. The card page reported "July 2026" for E2B (field ambiguous; treat the family release date as unverified). No timestamp encoding is documented for video (frames are images).

*MiniCPM-o 4.5 card* (P, https://huggingface.co/openbmb/MiniCPM-o-4_5): 9B total (SigLIP2 + Whisper-medium + CosyVoice2 + Qwen3-8B); images up to 1.8 Mpx; video up to 10 fps with a 3D-Resampler that stacks up to 5 frames ("high refresh rate mode"); streams synchronised by time-division multiplexing on a millisecond timeline; proactive-interaction decision at 1 Hz; time to first token 0.6 s (bf16); decode 154.3 tok/s bf16, 212.3 tok/s int4; GPU memory 19.0 GB bf16, 11.0 GB int4 (GPU model not stated on the card as fetched); "at least 28GB GPU memory" for the full PyTorch demo, or "M3/M4/M5 chip with at least 16GB RAM" via llama.cpp-omni; frameworks: transformers 4.51.0, llama.cpp-omni, Ollama, vLLM (FlagOS), SGLang; GGUF in 16 sizes; license Apache-2.0 on the card (earlier MiniCPM-V releases used a separate MiniCPM model license, so re-check the LICENSE file before commercial use). Card gives no per-frame token count; the 96× compression claim (search snippet, S) implies roughly 64 tokens per 6-frame stack at 448 px as in MiniCPM-V 4.5 (S, memory).

*Nemotron 3 Nano Omni card* (P, https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16): 31B total, ~3B active, Mamba2-Transformer hybrid MoE; vision encoder CRADIO v4-H; audio encoder Parakeet TDT 0.6B v2, audio up to 1 hour; video up to 2 minutes, recommended "1 FPS / 128 frames" at 1080p or "2 FPS / 256 frames" at lower resolution; Efficient Video Sampling exposed as vLLM `--video-pruning-rate 0.5` ("removes 50% of redundant video tokens; halves video-prefill VRAM/TTFT"); 256k context; reasoning on by default; tool calling; hardware minimums BF16 1× H100 80 GB, FP8 1× L40S 48 GB, NVFP4 1× RTX 5090 32 GB or Jetson Thor; Ampere/Hopper/Lovelace/Blackwell listed as supported; engines vLLM, TensorRT-LLM, TensorRT Edge-LLM, llama.cpp, Ollama, SGLang; NVIDIA Open Model Agreement; released 2026-04-28. Consequence: it does not fit a 12 to 16 GB GPU at any published precision; relevant here only as the reference implementation of EVS and hybrid-Mamba streaming.

**Batch 4.**

*LFM2.5-VL-3B card* (P, https://huggingface.co/LiquidAI/LFM2.5-VL-3B read 2026-09-16): 3.1B total; "SigLIP2 NaFlex shape-optimized 400M" encoder; LFM2.5-2.6B backbone; context 32,768 tokens; native-resolution processing, "large images are split into non-overlapping 512×512 patches and a resized whole-image thumbnail"; **no video mode** (image-text; frames must be sent as multiple images), no audio; tool calling with "Pythonic function calls" between special tokens; license `lfm1.0`; GGUF, ONNX and MLX-8bit variants; runtimes Transformers, vLLM, SGLang, llama.cpp, MLX; speeds "228 tok/s on an Apple M5 Max", "116 tok/s on an AMD Ryzen AI Max+ 395", H100 "11K tokens per second" throughput and "34 ms" TTFT. Tokens per 512×512 patch are not stated on the card (LFM2-VL used 256 after 2×2 pixel unshuffle: S, memory). The earlier "34 ms on a 5-frame clip" blog claim therefore refers to a 5-image prompt, not a video mode.

*Qwen3-Omni-30B-A3B* (S, memory, not re-read this round): Thinker–Talker MoE, 30B total / 3B active, Instruct/Thinking/Captioner checkpoints, Apache 2.0, released 2025-09; 30B total puts it outside the 2 to 9B band and above a 16 GB GPU at 4-bit (≈17 GB weights, E). llama.cpp lists "Qwen3 Omni (2 variants)" as pre-converted (P, multimodal.md).

*Qwen3.5 in llama.cpp* (S, search 2026-09-16): Qwen3.5 GGUFs exist with MTP drafts (https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF); "MTP speculative decoding for ~1.5–2× faster generation has been merged as of May 16th, 2026"; users run Qwen3.5-122B with `mmproj-BF16.gguf` in llama.cpp on DGX Spark (https://forums.developer.nvidia.com/t/missing-vision-reasoning-with-qwen3-5-122b-q4-on-vllm-works-on-llama-cpp/363196); a tutorial covers Qwen3.5 on vLLM and llama.cpp (https://debuggercafe.com/introduction-to-qwen3-5-overview-vllm-and-llama-cpp/); a merged community fix moved one Qwen3.5 configuration from ~44 to >62 tok/s decode (hardware unstated). Newer families with the same hybrid layout exist by September 2026: Qwen3.6-35B-A3B GGUFs (https://huggingface.co/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF) and Qwen3.8 (section 2.2). Treat "Qwen3.5 text + vision runs in llama.cpp via mmproj" as S until the ggml-org model list is re-read.

*ProVideLLM* (P abstract via search, https://arxiv.org/abs/2504.13915, ICCV 2025; project page https://dibschat.github.io/ProVideLLM/): multimodal cache holding "verbalized text tokens" (compressed long-term summaries) and DETR-QFormer visual tokens (short-term); "reduces token count by 22× over existing methods in representing one hour of long-term observations"; "per-frame streaming inference at 10 FPS and streaming dialogue at 25 FPS, with a minimal 2GB GPU memory footprint"; GPU model for those numbers and code release not confirmed (section 7).

## 2. Runtimes (deliverable b)

### 2.1 Runtime table with speeds by machine

Measured numbers are quoted with their machine; E rows are this note's estimates (section 5 assumptions).

| Runtime (version) | Machine | Model | Prefill | Decode | Cross-request reuse | Status, source |
|---|---|---|---|---|---|---|
| mlx-vlm 0.7.1 server | Apple silicon, chip not stated | Gemma-4-26B | "~48" tok/s uncached → "~550-825" tok/s with vision-feature and prefix caching | — | `APC_ENABLED=1` automatic prefix caching; vision-feature cache ("11x+" multi-turn prompt TPS); `--kv-bits 3.5 --kv-quant-scheme turboquant` (−76 % KV); DFlash/MTP drafts 2–3.9× | P README 2026-09-16 |
| Ollama MLX backend (preview) / mlx-lm | M5 Max | Qwen3.5-35B-A3B NVFP4 | 1,810 tok/s | 112 tok/s | prompt cache | S, antekapetanovic.com |
| MLX | M4 Max ≥36 GB | Qwen3.5-35B-A3B | — | 40–55 tok/s at 32K context | — | S, willitrunai.com |
| MLX | Mac Studio M5 Max 128 GB | 7B Q4 | — | 95–110 tok/s | — | S, llmcheck.net |
| Liquid stack (framework not stated) | M5 Max | LFM2.5-VL-3B | — | 228 tok/s | — | P, LiquidAI card |
| llama.cpp Metal | M5 Max | unnamed 4-bit model (tg suggests ~30B) | 966 tok/s pp512 | 30.7 tok/s tg128 | `cache_prompt` prefix reuse, `--cache-reuse` KV shift | S, promptquorum.com |
| llama.cpp Metal | M4 Max / M1 Pro | 7B Q4_0 | — | 83.1 / 36.4 tok/s | same | P (companion note, discussion #4167, 2026-08-25) |
| llama.cpp Metal | M5 Max | 4B VLM Q4 + mmproj | E 1,200–2,000 tok/s text prefill; ViT ≈50 ms per 448² frame | E 100–150 tok/s | same | E, no measurement found |
| llama.cpp CUDA | RTX 3060 12 GB | 4B VLM Q4 + mmproj | E 600–1,200 tok/s; ViT ≈45 ms per frame | E 50–80 tok/s | same | E, no measurement found |
| vLLM (stable) | 12–16 GB GPU | Qwen3-VL-4B or Qwen3.5-4B, AWQ/GPTQ int4 | — | — | prefix caching (hashes multimodal inputs), `--video-pruning-rate` EVS (bug #30847 with Qwen3-VL timestamps); hybrid-model prefix caching unverified | S docs; memory derived in section 5 |
| SGLang | — | — | — | — | not fetched | gap (section 7) |
| StreamingVLM runtime | 1× H100 | Qwen2.5-VL-7B-based (S) | 8 FPS end-to-end | — | attention sinks + short vision window + long text window, contiguous RoPE | P abstract (ICLR 2026) |
| ProVideLLM | GPU not stated | DETR-QFormer + LLM | 10 FPS per-frame streaming; 25 FPS dialogue; 2 GB GPU memory | — | verbalized long-term cache (22× fewer tokens per hour) | P abstract (ICCV 2025) |
| MiniCPM-o 4.5 (transformers / vLLM / SGLang) | GPU not stated (≥28 GB for the full demo) | 9B | TTFT 0.6 s (bf16) | 154.3 tok/s bf16; 212.3 tok/s int4 | TDM streaming prefill/generate by chunk | P card |
| MiniCPM-o 4.5 llama.cpp-omni | "M3/M4/M5 chip with at least 16GB RAM" | 9B GGUF | — | — | — | P card claim; repo URL 404 (gap) |
| LFM2.5-VL-3B (vLLM) | H100 | 3.1B | 11K tok/s throughput, 34 ms TTFT | — | — | P card |

### 2.2 mlx-vlm and mlx-lm on M-series

**mlx-vlm README** (P, https://github.com/Blaizzy/mlx-vlm read 2026-09-16; version 0.7.1 of 2026-09-14 per the companion note). Supported families named in the README: Qwen2-VL, Qwen2.5-VL, **Qwen3-VL, Qwen3.5, Gemma 4, MiniCPM-o, MiniCPM-V, InternVL, SmolVLM, LFM2(-VL)**, Idefics3, LLaVA, Phi-4, Moondream, Granite Vision and OCR variants. Video: `--video path/to/video.mp4 --fps 1.0` (README lists Qwen2-VL, Qwen2.5-VL, Idefics3, LLaVA, LLaVA-OneVision as video-tested; Qwen3-VL/Qwen3.5 video path not explicitly listed). Audio: `--audio /path/to/audio.wav`, `--output-modality audio` (Gemma 3n example). Server: OpenAI-compatible `/v1/chat/completions`, continuous batching, **Automatic Prefix Caching** enabled with `APC_ENABLED=1`, KV-cache quantization `--kv-bits 3.5 --kv-quant-scheme turboquant` ("76% reduction" in KV memory), speculative decoding `--draft-model`/`--draft-kind` (DFlash "2–3× faster"; Gemma 4 MTP "up to 3.94×"). Vision-feature caching: "11x+" prompt-TPS speedup in multi-turn use; Gemma-4-26B prompt throughput "~48" tok/s uncached to "~550-825" tok/s cached (chip not stated). So incremental requests that share a prefix (system prompt, journal, earlier frames) can skip re-encoding and re-prefilling; the remaining per-tick cost is the new tokens only, provided the client keeps the prefix byte-identical.

**mlx-lm / MLX speeds** (S, search 2026-09-16). Qwen3.5 has both GGUF and MLX (mlx-community) builds; Ollama now runs an MLX backend on Apple silicon in preview (https://ollama.com/blog/mlx). Reported: M5 Max, Qwen3.5-35B-A3B NVFP4 via Ollama-MLX, **prefill 1,810 tok/s, decode 112 tok/s** (https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/); M4 Max (≥36 GB) Qwen3.5-35B-A3B MLX 40–55 tok/s decode at 32K context (https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks); Mac Studio M5 Max 128 GB "approximately 95–110 tps" single-stream MLX decode for 7B Q4 models (https://llmcheck.net/blog/apple-silicon-m5-max-local-ai-guide/). MLX reads only Q4_0/Q4_1/Q8_0 GGUF; other quants are cast to FP16 (S). Newer Qwen3.8 family (Qwen3.8-Flash-Next: 36 of 48 layers Gated DeltaNet; Qwen3.8-27B: 48 GDN + 16 full-attention layers) appears in September 2026 blogs (S, not in scope of this note beyond flagging it in section 7).

### 2.3 llama.cpp with mtmd

**docs/multimodal.md** (P, https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md read 2026-09-16; the page lists pre-converted `ggml-org` repos, not every supported architecture). Vision: Gemma 3 (3 variants), SmolVLM (5 variants "including video"), Pixtral 12B, Qwen2-VL (2), Qwen2.5-VL (4), Mistral Small 3.1 24B, InternVL 2.5 and 3 (6), Llama 4 Scout, Moondream2, **Gemma 4 (4 variants)**. Audio: Ultravox 0.5, Voxtral Mini 3B, Qwen3-ASR (2). Mixed: Qwen2.5-Omni (2), **Qwen3-Omni (2)**, Gemma 4. "Currently, we support image, audio and video input." Tools: llama-cli, llama-server (`/chat/completions`), llama-mtmd-cli. **Not in the fetched list: Qwen3-VL, Qwen3.5, MiniCPM-V/o, LFM2-VL, InternVL3.5** (status checked in batch 4 below; MiniCPM-o 4.5 ships its own GGUF and runs under Ollama per its card). KV reuse: the server's `cache_prompt` prefix reuse and `--cache-reuse` chunk shifting are documented in the server README cited in the companion note; whether image/audio chunks participate in prefix matching is not stated in multimodal.md (see section 7).

**Apple silicon speeds** (S, search 2026-09-16). "On the same 4-bit quant, the M5 Max shows prefill performance of 966 tokens per second at pp512 and generation of 30.7 tokens per second at tg128" (llama.cpp Metal; model size not given in the snippet, the low tg suggests a ~30B dense model) https://www.promptquorum.com/local-llms/m5-pro-max-llm-benchmarks-2026 . Companion note (P): llama.cpp 7B Q4_0 83.1 tok/s decode on M4 Max, discussion #4167. No M4/M5 Max number for a 4B VLM prefill with images was found; section 5 derives one (E).

### 2.4 vLLM and SGLang on a 12 to 16 GB GPU

**vLLM** (S/P mix, search 2026-09-16). `--video-pruning-rate <fraction>` prunes that fraction of video tokens per video; the default method is EVS, which "drops tokens with the lowest temporal dissimilarity to the previous frame"; pruning is applied post-encoder on the embeddings with positional IDs preserved; example `vllm serve Qwen3-VL-8B --video-pruning-rate=0.75` (https://docs.vllm.ai/en/stable/features/multimodal_inputs/). Open bug #30847: with Qwen3-VL + EVS the number of tokens after each `<t s>` timestamp text is no longer aligned with the pruned token count (https://github.com/vllm-project/vllm/issues/30847), i.e. EVS and Qwen3-VL's text-timestamp alignment interact badly today. RFC #45098 extends pruning to image tokens (https://github.com/vllm-project/vllm/issues/45098). Nemotron 3 Nano Omni's card documents the same flag at 0.5 as halving video-prefill VRAM and TTFT (P, section 1.2). Memory for a 4B model at a 32K window is derived in section 5 (E) from the KV sizes in section 1.2. SGLang: not fetched this round (section 7).

### 2.5 Purpose-built streaming engines

**StreamingVLM** (P, https://arxiv.org/abs/2510.09608, submitted 2025-10-10, ICLR 2026, revised 2026-05-31). Inference keeps "a compact KV cache by reusing states of attention sinks, a short window of recent vision tokens, and a long window of recent text tokens"; training is SFT "on short, overlapped video chunks" with full attention, which matches inference without training on infinite contexts; Inf-Streams-Eval averages over two hours per video; **8 FPS on a single NVIDIA H100**; 66.18% win rate vs GPT-4o mini; code https://github.com/mit-han-lab/streaming-vlm (base model Qwen2.5-VL-7B and weight release: S, memory, verify). This is the closest published design to the direction in this project (sliding vision window, longer text window) and is the reference for the SFT recipe.

**MiniCPM-o llama.cpp-omni**: named on the MiniCPM-o 4.5 card as the Mac / low-resource path ("M3/M4/M5 chip with at least 16GB RAM"), but https://github.com/OpenBMB/llama.cpp-omni returned 404 on 2026-09-16; the actual repo path and any speed numbers are unverified (section 7).

**ProVideLLM**: see batch 4 below.

Also found in the streaming search (S): VST-7B "Video Streaming Thinking" (arXiv 2603.12262) amortises reasoning latency over playback; LiveStar (arXiv 2511.05299); ROMA real-time omni-multimodal assistant (arXiv 2601.10323); "A Simple Baseline for Streaming Video Understanding" (arXiv 2604.02317); StreamingEval protocol (arXiv 2603.21493); curated list https://github.com/sotayang/Awesome-Streaming-Video-Understanding . None was verified as shipping a 2 to 9B open checkpoint with a runtime; ProVideLLM (above) and StreamingVLM are the two with explicit real-time numbers.

## 3. Token reduction and gating without retraining

### 3.1 Methods table (measured reductions and accuracy deltas)

| Method | Where it acts | Retraining | Measured reduction and accuracy | Encoders / models shown | Runtime availability | Source (status) |
|---|---|---|---|---|---|---|
| Efficient Video Sampling (EVS) | post-encoder, drops patches whose temporal dissimilarity to the previous frame is lowest; keeps position IDs | none ("requires no architectural changes or retraining"); optional uptraining with stochastic pruning rates makes models robust across rates | "reduces LLM time-to-first-token by up to 4×" with "minimal accuracy loss" (abstract; per-benchmark deltas in the PDF, not read); Nemotron card: rate 0.5 "halves video-prefill VRAM/TTFT" | Qwen3-VL, Qwen2.5-VL, Nemotron 3 Nano Omni via vLLM `--video-pruning-rate` (works on Qwen-style 32×32-px patch tokens) | vLLM (stable docs), default pruning method; bug #30847 with Qwen3-VL timestamp text | https://arxiv.org/abs/2510.14624 (P abstract, 2025-10-16); https://docs.vllm.ai/en/stable/features/multimodal_inputs/ (S snippet); Nemotron card (P) |
| Conv3D tubelets (temporal patching) | in-encoder: `temporal_patch_size 2` merges 2 frames per token (Qwen3-VL, Qwen3.5); MiniCPM-o 4.5 3D-Resampler stacks up to 5 frames | built in; not a plug-in | 2× per frame (Qwen) by construction; MiniCPM "96× video token compression" (S) | Qwen3-VL/3.5 ViT, MiniCPM-o 4.5 | native in every runtime that runs the model | config.json (P), MiniCPM-o card (P), search (S) |
| PruneVid | classifies static vs dynamic spatio-temporal tokens, merges static ones, prunes by query attention | training-free | "prune over 80% of visual tokens while maintaining, or even improving, model performance" on MVBench, Video-MME, EgoSchema, VideoChatGPT-Bench, 3 video LLMs | PLLaVA, ST-LLaVA, LLaVA-OneVision (S, memory) | reference code only (https://github.com/visual-ai/prunevid) | https://arxiv.org/abs/2412.16117 (S snippet, ACL 2025) |
| DyCoke | temporal token merging across frames plus dynamic KV-cache pruning at decode | training-free | (S, memory, not re-read) ~1.5× speed-up and ~1.4× memory reduction on LLaVA-OneVision-7B at near-equal accuracy | LLaVA-OV | reference code | arXiv 2411.15024 (S) |
| VisionZip | selects dominant tokens by encoder [CLS] attention, merges the rest into contextual tokens | training-free (optional projector tune) | (S, memory) keeps 64 of 576 tokens on LLaVA-1.5 at ~95% of accuracy; prefill ~8× faster | CLIP/SigLIP encoders with a CLS token; not applicable to Qwen's NaViT-style encoder without a CLS | reference code | arXiv 2412.04467 (S) |
| FastV | drops the least-attended image tokens after LLM layer K using instruction cross-attention | training-free | (S, memory) K=2, R=50% → ~45% FLOP reduction on LLaVA-1.5-13B with negligible loss | any LLM (acts inside the decoder) | reference code; needs engine hooks | arXiv 2403.06764 (S) |
| FastVID (for scale) | dynamic density pruning | training-free | at 90% pruning "preserves 98.0% of the vanilla model's performance", beating PruneVid by 3.1, FastV by 7.8, VisionZip by 8.4 points | LLaVA-OV / Qwen2-VL (S) | reference code | https://www.arxiv.org/pdf/2503.11187v1 (S snippet) |

Reading: for a fixed camera the only training-free method that (a) is shipped in a serving engine and (b) prunes by temporal redundancy is EVS; everything else needs custom hooks. Because Qwen3.5/Qwen3-VL tokens are aligned 32×32-px blocks, an external change gate can produce exactly the EVS effect before the encoder (skip unchanged blocks) with the same positional-ID preservation; the open vLLM bug shows the one constraint: the count of tokens that follows each timestamp string must be rewritten to match what survives.

### 3.2 Compressed-domain gating (H.264/H.265 motion vectors)

All entries are S (search snippets, 2026-09-16); none was read in full and no accuracy numbers were captured.

- **CodecSight** (arXiv 2604.06036v3, April 2026): "leveraging video codec signals for efficient streaming VLM inference"; H.264 decoded with NVIDIA NVDEC, uses motion vectors and frame-type metadata for GOP-aligned processing; the primitives are shared by H.265/HEVC, VP9 and AV1. Closest to a drop-in gate for a streaming VLM. https://arxiv.org/html/2604.06036v3
- **HY-Himmel** (arXiv 2605.08158, May 2026): sparse anchor I-frames go to the frozen host ViT; dense inter-frame intervals are encoded by a "compressed tri-stream adapter" reading quarter-pixel, variable-block motion vectors directly from the H.264 bitstream and injected as aligned motion tokens. Requires training the adapter. https://arxiv.org/html/2605.08158
- **CodecTokenizer** (May 2026): learnable native-bitstream tokenizer over motion vectors, residuals, block syntax, QP maps and GOP metadata with block-, frame- and GOP-level tokens under an adaptive budget (ResearchGate listing only).
- **Efficient Motion-Aware Video MLLM** (arXiv 2503.13016, March 2025): encodes a GOP's RGB plus motion "with the same number of tokens as a single frame". https://arxiv.org/pdf/2503.13016
- **CoPE-VideoLM**: Δ-Encoder over motion vectors and residuals with modality-alignment pre-training (secondary summary only).

Practical takeaway for this project: motion-vector magnitude per macroblock from the camera's own H.264/H.265 stream (ffmpeg `-flags2 +export_mvs`, or NVDEC on the box) is a free change gate at the 16×16 macroblock level, which maps onto Qwen's 32×32-px token grid at 2×2 macroblocks per token; the published systems above confirm the signal is usable, but they train adapters, so the gate here should stay a pre-filter (skip static tokens) rather than a new input modality until data exists.

## 4. Rental prices

| Item | Price | Status | Source |
|---|---|---|---|
| Runpod H100 SXM 80 GB, on-demand | US$2.69/h Community, US$3.49/h Secure | P, read September 2026 by the companion note | https://www.runpod.io/pricing (via `decoding-and-training-2026-09.md` section 5.3) |
| Runpod H100 PCIe 80 GB | US$1.99/h Community, US$2.89/h Secure | P (same) | same |
| Runpod A100 80 GB | US$1.19–1.59/h | P (same) | same |
| Lambda H100 SXM / PCIe | US$4.29/h / US$3.29/h | P (same) | https://lambda.ai/pricing (via companion note) |
| Vast.ai RTX 3060 12 GB / RTX 5060 Ti 16 GB / RTX 3090 24 GB | **not captured**: https://vast.ai/pricing loads prices dynamically and returned no figures on 2026-09-16 (only "per-second billing" and "50%+ cheaper" for interruptible) | unverified; E from memory of 2025–2026 marketplace levels: 3060 ≈ US$0.05–0.10/h, 5060 Ti ≈ US$0.08–0.15/h, 3090 ≈ US$0.12–0.25/h on-demand | https://vast.ai/pricing (P page, no numbers) |
| Owning the box instead (E) | a used RTX 3060 12 GB at ≈US$150–200 or a new RTX 5060 Ti 16 GB at ≈US$430–480 plus a ≈US$400 host stays under US$1K; at 24/7 the rental-equivalent of a 3060 (≈US$0.07/h) is ≈US$50/month, so purchase pays back in 3 to 10 months | E, unverified street prices | — |

## 5. Per-tick budget worked example (deliverable c)

All figures in this section are **E** unless marked; they are derived from the primary geometry in section 1.2 and the secondary speed points in section 2. Re-measure the four starred (*) rates on the real hardware before committing.

### 5.1 Tokens per tick

| Input | Quiet tick | Busy tick | Basis |
|---|---|---|---|
| Tick header (timestamp text, stream markers) | 8 | 8 | Qwen3-VL/3.5 timestamp text (P); one short line |
| Visual tokens after change gating | 20 | 80 | task premise; 20 = 10 % and 80 = 41 % of a 196-token 448² Qwen frame-pair; Gemma 4 cannot go below one 70-token image per frame it sees |
| Audio | 1 (VAD-gated ASR text; 150 words/min ≈ 3.5 tokens/s only while someone speaks) | 4 | Qwen3.5 has no audio input (P); Gemma 4 native audio would cost ≈6–7 tokens/s if it follows the 160 ms framing of Qwen3.5-Omni's AuT (P for Omni, E for Gemma) |
| Device-state diff | 2 | 5 | occasional lines, ≤20 tokens |
| **Input total** | **≈31 → use 35** | **≈97 → use 100** | |
| Output | 1 silence token; a 45-token JSON line on 0.5 % of ticks | 1 silence token; 45-token line on 10 % of ticks | teacher pass: silence ≈90 % of busy ticks, nearly all quiet ticks (`teacher-distillation.md` §4) |
| Expected decode steps per tick | 1.2 | 5.5 | 1 + p(burst) × 45 |

Context growth: 35 tokens/s fills a 32K window in ≈15 min; 100 tokens/s in ≈5.5 min. Journal lines add ≈0.8K tokens/h quiet and up to ≈16K tokens/h busy, so both the vision window (minutes) and the text window (hours) need eviction or compaction.

### 5.2 Unit costs

| Cost | Mac M5 Max, 4B at 4-bit, mlx-vlm or llama.cpp | RTX 3060 12 GB, 4B Q4, llama.cpp CUDA | Basis |
|---|---|---|---|
| Vision encoder, one 448² frame-pair (784 patch tokens through a 0.3B ViT ≈ 0.47 TFLOP) | ≈50 ms* | ≈45 ms* | ≈10 TFLOPS effective on either (E) |
| Prefill of new tokens onto a cached prefix | ≈1,200 tok/s* + 20 ms call overhead | ≈900 tok/s* + 15 ms | M5 Max 1,810 tok/s for a 3B-active MoE (S); 3060 from 360 GB/s and ≈50 FP16 TFLOPS peak at low utilisation |
| Decode | ≈8 ms/token (≈125 tok/s) | ≈15 ms/token (≈65 tok/s) | M5 Max 95–110 tok/s for 7B Q4 (S), 228 tok/s for 3.1B (P); 3060 bandwidth-bound |
| VAD-gated ASR (whisper.cpp small or Qwen3-ASR GGUF), averaged | ≈10 ms quiet, ≈40 ms busy | ≈8 ms, ≈30 ms | real-time factor ≈0.05–0.1 while speech is present |
| Gate hit rate (frame actually encoded) | 30 % of quiet ticks, 100 % of busy ticks | same | motion-vector or pixel-diff gate at 16×16 blocks |

### 5.3 Per-tick time and utilisation, one camera

| Case | Mac M5 Max | RTX 3060 12 GB |
|---|---|---|
| Quiet tick: 0.3 × ViT + prefill 35 + 1.2 decode steps + ASR | 15 + 49 + 10 + 10 ≈ **85 ms → 8 %** | 14 + 54 + 18 + 8 ≈ **95 ms → 10 %** |
| Busy tick: ViT + prefill 100 + 5.5 decode steps + ASR | 50 + 103 + 44 + 40 ≈ **240 ms → 24 %** | 45 + 126 + 83 + 30 ≈ **285 ms → 29 %** |
| Worst single tick (busy + 45-token burst) | 50 + 103 + 368 + 40 ≈ **0.56 s** | 45 + 126 + 690 + 30 ≈ **0.89 s** (a 60-token line ≈ 1.1 s, one tick of lag) |
| Same busy tick **without** prefix reuse (re-prefill an 8K window every tick) | 8,000 / 1,500 ≈ **5.3 s → not real-time** | 8,000 / 900 ≈ **8.9 s → not real-time** |
| Three cameras, busy | ≈0.72 s → 72 % (feasible, no headroom) | ≈0.86 s → 86 % (bursts queue; needs a 5060 Ti/3090 or lower token rates) |
| Periodic compaction: rebuild a 4K-token context every 10 min | 4,000 / 1,500 ≈ 2.7 s → +0.5 % | 4,000 / 900 ≈ 4.4 s → +0.7 % |

24/7 verdict: feasible on both machines for one camera at 8–30 % utilisation, with 3–10× headroom for a second look at higher resolution on `unsure` ticks (a 896² frame ≈ 4× the ViT and ≈780 tokens ≈ 0.9 s on the Mac, 1.2 s on the 3060, which is why second looks must be rare). The single deciding requirement is prefix/KV reuse across ticks; without it the system is where it is today (5–20 s per request). Power (E): the 3060 at ≈25 % duty averages ≈60–80 W, ≈1.4–1.9 kWh/day; the Mac ≈15–30 W at this duty.

### 5.4 Memory at a 32K window

| Model, 4-bit | Weights + mmproj | KV or state at 32K | Buffers, CUDA/Metal context, ASR | Total | Fits |
|---|---|---|---|---|---|
| Qwen3.5-4B | ≈2.6 + 0.7 GB | 1.0 GB KV (8 attention layers, f16) + 25 MB GDN state | ≈2.0 GB | **≈6.3 GB** | 12 GB card with room for a second camera's context; Mac trivially |
| Qwen3-VL-4B | ≈2.6 + 0.7 GB | 4.7 GB f16 (2.4 GB with q8_0 KV) | ≈2.0 GB | ≈10.0 GB (7.7 GB with q8 KV) | 12 GB card only with q8 KV or a 16K window |
| Gemma 4 E4B | 4.59 + 0.56 GB (P) + ≈0.6 GB audio encoder | ≤2 GB (local layers bounded by the 512-token window; global layers share K/V) | ≈2.0 GB | ≈9.5 GB | 12 GB card, little headroom |
| MiniCPM-o 4.5 | ≈6 GB Q4 (E); int4 PyTorch path 11.0 GB (P) | ≈4.7 GB (Qwen3-8B geometry, S) | ≈2.0 GB | ≈12.7 GB | 16 GB card only |
| Qwen3.5-9B | ≈5.5 GB (E) | ≈2 GB (E) | ≈2.0 GB | ≈9.5 GB | 12 GB card marginal, 16 GB fine; Mac fine |

### 5.5 Eviction policy that keeps the cache valid

Prefix caches (mlx-vlm APC, vLLM, llama.cpp `cache_prompt`) match exact prefixes, so evicting old vision tokens from the middle of the sequence invalidates everything after the cut. Two workable policies: (a) **checkpoint and re-prefill**: keep appending for N minutes, then rebuild a compacted context (system prompt, journal, summarised earlier observations, last K seconds of vision) and prefill it once (cost in 5.3, under 1 % utilisation at N = 10 min); (b) llama.cpp `--cache-reuse` KV shifting, which moves cached chunks and re-applies RoPE, but its behaviour under Qwen's 3-axis interleaved MRoPE is unverified (section 7). For Qwen3.5 the Gated DeltaNet state cannot be edited, only recomputed, so (a) is the natural fit and is also what StreamingVLM trains for ("full attention on short, overlapped video chunks", P). Training the student on the same chunking rule closes the train/serve gap.

## 6. Ranked base model plus runtime (deliverable d)

### 6.1 Mac (M5 Max 48 GB), production now

1. **Qwen3.5-4B (Instruct behaviour with `enable_thinking: False`) on mlx-vlm 0.7.x with `APC_ENABLED=1`, llama.cpp Metal as the fallback.** Deciding reasons: the only 2–9B open model whose backbone already has the bounded memory the design asks for (24 Gated DeltaNet layers with a fixed ≈25 MB state, 8 attention layers at 32 KB per token), so a 32K window costs 1 GB instead of 4.7 GB; time-aware interleaved MRoPE with an fps parameter; tokens are 32×32-px blocks per 2-frame patch, so a motion-vector gate can drop tokens at block granularity (Gemma's per-image soft-token budget cannot); 262K native context; Apache-2.0; tool calling; mlx-vlm lists it as supported (P). Risks: no audio input (run VAD + whisper.cpp/Qwen3-ASR and inject text; ≈10–40 ms per tick); mlx-vlm's video/fps path is documented for Qwen2-VL/2.5-VL, not explicitly for Qwen3.5 (P README); prefix caching of hybrid (recurrent) state in mlx-vlm is unverified; the timestamp-count mismatch seen with EVS in vLLM (#30847) will recur with any external gate unless the `<t s>` text is rewritten to the surviving token count; GDN training needs the flash-linear-attention kernels on H100 (S); Qwen3.6/3.8 may supersede it before the student is trained.
2. **Gemma 4 E4B on llama.cpp (ggml-org GGUF + mmproj) or mlx-vlm.** Reasons: native audio (30 s clips, ≈300M encoder) so no ASR side channel; native function calling; 70-token image budget is the cheapest whole-frame representation of any candidate without pruning; already running in this project; grammar tooling verified in the companion note. Risks: no timestamp encoding and a 1 fps/60 s video assumption, so the tick clock must be carried in text; gating only at whole-frame granularity (send or skip); 42-layer 8B-with-embeddings model decodes slower than a 4B dense at the same quantisation; vision-fix and >1.2 Mpx regressions in llama.cpp (companion note); 262K-entry vocabulary doubles grammar-mask cost.
3. **Qwen3-VL-4B-Instruct on mlx-vlm or llama.cpp.** Reasons: most mature runtime support (llama.cpp, mlx-vlm, vLLM with EVS), DeepStack, text-timestamp alignment, no custom training kernels. Risks: 147 KB per token KV (4.7 GB at 32K) and no recurrence, so bounded memory must be built entirely from eviction; no audio.
4. **MiniCPM-o 4.5 with its own TDM streaming stack.** Reasons: the only candidate with a shipped full-duplex streaming loop (1 Hz proactive decision, 10 fps video, native audio). Risks: 9B on a Qwen3-8B decoder (slower bursts), Mac path (llama.cpp-omni) unverified, streaming logic lives in a custom demo stack rather than a general server, licence file to confirm.

### 6.2 Sub-US$1K box (12–16 GB GPU), later

1. **Qwen3.5-4B Q4_K_M GGUF + mmproj on llama.cpp CUDA** (≈6.3 GB total at a 32K window, section 5.4), or **vLLM with an int4 AWQ/GPTQ checkpoint, prefix caching and `--video-pruning-rate`** once hybrid-model prefix caching and #30847 are confirmed. Same reasons as 6.1; on a 12 GB card the small KV is what leaves room for a second camera.
2. **Gemma 4 E4B Q4_0 + mmproj on llama.cpp CUDA** (≈9.5 GB): audio native, fits, less headroom.
3. **Qwen3-VL-4B on vLLM (EVS) or llama.cpp** with q8_0 KV; prefer a 16 GB card.
MiniCPM-o 4.5 needs the 16 GB card and leaves no headroom; Nemotron 3 Nano Omni is out (32 GB minimum at NVFP4, P).

### 6.3 Training target on rented H100s

Train the same checkpoint you serve: Qwen3.5-4B (transformers ≥4.57 per its config; GDN kernels via flash-linear-attention, S) with StreamingVLM-style SFT on short overlapped chunks that reproduce the serve-time checkpoint-and-re-prefill rule, silence-token-dominant streams from the teacher, and EVS-style stochastic token dropping so the student is robust to the gate ("uptraining phase using stochastic pruning rates", P). Fallback with zero custom kernels: Qwen3-VL-4B. Costs follow the companion note's section 5.3 (H100 SXM US$2.69–3.49/h): a LoRA over 20K–100K ticks-worth of chunks is tens to a few hundred dollars (E).

## 7. Not verified / gaps (deliverable e)

Fetch budget was exhausted before these could be closed; each is marked where it appears.

1. **Vast.ai prices** for RTX 3060 12 GB, RTX 5060 Ti 16 GB and RTX 3090: the pricing page is dynamic and returned no numbers; section 4 ranges are memory estimates.
2. **Qwen3.5-Omni**: open-weight status, licence, and the Plus/Flash parameter counts are not in the fetched report text; two secondary sources disagree.
3. **Qwen3.5 in llama.cpp**: GGUF text and MTP support are evidenced only by third-party repos and forum posts; the ggml-org multimodal list fetched does not name Qwen3-VL, Qwen3.5, MiniCPM-V/o, InternVL3.5 or LFM2-VL, so their mtmd status (vision, video, and the mmproj converter) needs a direct check of `tools/mtmd` and the model list.
4. **KV/prefix reuse with multimodal chunks in llama.cpp** (whether image/audio chunks are hashed into `cache_prompt` prefix matching) and **`--cache-reuse` semantics under interleaved MRoPE**.
5. **Prefix caching of hybrid (Gated DeltaNet / Mamba) state** in mlx-vlm, mlx-lm, vLLM and llama.cpp; the budget in section 5 assumes it works or is replaced by checkpoint-and-re-prefill.
6. **mlx-vlm video/fps path for Qwen3.5, Qwen3-VL and Gemma 4** (README lists video for Qwen2-VL/2.5-VL, Idefics3, LLaVA); and the mlx-vlm 0.7.1 status of issue #1294 (JSON schema + thinking) from the companion note.
7. **No measured prefill or decode numbers for a 4B VLM on M4 Max, M5 Max, RTX 3060, RTX 5060 Ti or RTX 3090**; all such rates in sections 2.1 and 5 are E and starred for measurement. The M5 Max llama.cpp figures found (966 tok/s pp512, 30.7 tok/s tg128) are for an unnamed model.
8. **Gemma 4**: audio tokens per second, the local:global layer ratio for E2B/E4B (needed for an exact KV figure), and the family release date (card page showed "July 2026" in an ambiguous field).
9. **MiniCPM-o 4.5**: exact tokens per frame of the 3D-Resampler, the GPU behind the 154/212 tok/s numbers, the llama.cpp-omni repository location (404 at OpenBMB/llama.cpp-omni), and the licence file versus the card's Apache-2.0 tag.
10. **Tokens per frame from memory (S)**: SmolVLM2-2.2B (81 per 384² tile), LFM2.5-VL (256 per 512² patch), Qwen3-Omni-30B-A3B facts, StreamingVLM's base model and weight release, ProVideLLM code release and GPU.
11. **Token-pruning numbers**: EVS per-benchmark accuracy deltas (abstract only; PDF not read); DyCoke, VisionZip and FastV headline numbers are from memory; only PruneVid's ">80 %" and FastVID's comparison came from fetched text.
12. **Compressed-domain gating**: CodecSight, HY-Himmel, CodecTokenizer, CoPE-VideoLM and the motion-aware MLLM were seen only as snippets; no accuracy or speed numbers captured.
13. **SGLang** on 12–16 GB GPUs (multimodal prefix caching, memory) was not fetched.
14. **Newer Qwen families** (Qwen3.6, Qwen3.8-Flash-Next with 36 of 48 GDN layers) appeared in September 2026 blogs and were not assessed; if a 2–9B Qwen3.8 dense exists with the same tokenizer/vision stack it may supersede Qwen3.5-4B.
15. Qwen3.5 family small sizes other than 4B and 9B (0.8B, 2B) and the vision-encoder parameter count (≈0.3B derived from depth 24 × hidden 1024) are E/S.

## 8. Source list

Primary (read 2026-09-16 unless noted):
- https://huggingface.co/Qwen/Qwen3.5-4B and https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct and https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json
- https://arxiv.org/html/2604.15804v1 (Qwen3.5-Omni technical report, 2026-04-17)
- https://ai.google.dev/gemma/docs/core/model_card_4 (Gemma 4 model card)
- https://huggingface.co/openbmb/MiniCPM-o-4_5
- https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (released 2026-04-28)
- https://huggingface.co/OpenGVLab/InternVL3_5-4B (released 2025-08-25)
- https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- https://huggingface.co/LiquidAI/LFM2.5-VL-3B
- https://github.com/Blaizzy/mlx-vlm (README)
- https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- https://arxiv.org/abs/2510.09608 (StreamingVLM, ICLR 2026) ; https://arxiv.org/abs/2510.14624 (EVS, 2025-10-16) ; https://arxiv.org/abs/2504.13915 (ProVideLLM, ICCV 2025; abstract via search)
- https://vast.ai/pricing (fetched; no figures rendered) ; https://github.com/OpenBMB/llama.cpp-omni (404)
- Reused from `decoding-and-training-2026-09.md` (read by the earlier agent in September 2026): https://www.runpod.io/pricing , https://lambda.ai/pricing , https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF , https://github.com/ggml-org/llama.cpp/discussions/4167 , https://huggingface.co/google/gemma-4-E4B-it

Secondary (search snippets, 2026-09-16):
- https://www.marktechpost.com/2026/03/30/alibaba-qwen-team-releases-qwen3-5-omni-a-native-multimodal-model-for-text-audio-video-and-realtime-interaction/ ; https://www.spheron.network/blog/deploy-qwen3-5-omni-gpu-cloud/ ; https://wavespeed.ai/blog/posts/what-is-qwen3-5-omni/
- https://ai.google.dev/gemma/docs/core ; https://huggingface.co/google/gemma-4-12B ; https://huggingface.co/google/gemma-4-E2B-it ; https://lmstudio.ai/models/gemma-4
- https://github.com/openbmb/MiniCPM-V ; https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf ; https://ollama.com/openbmb/minicpm-o4.5 ; https://github.com/OpenBMB/MiniCPM-o-Demo/
- https://huggingface.co/blog/nvidia/nemotron-3-nano-4b ; https://www.buildfastwithai.com/blogs/nvidia-nemotron-3-nano-omni-2026
- https://huggingface.co/LiquidAI/LFM2.5-VL-450M ; https://www.liquid.ai/blog/lfm2-5-vl-3b ; https://huggingface.co/blog/LiquidAI/lfm2-5-vl-3b
- https://arxiv.org/abs/2603.12262 (VST) ; https://arxiv.org/pdf/2511.05299 (LiveStar) ; https://arxiv.org/pdf/2601.10323 (ROMA) ; https://arxiv.org/pdf/2604.02317 ; https://arxiv.org/pdf/2603.21493 (StreamingEval) ; https://github.com/sotayang/Awesome-Streaming-Video-Understanding
- https://antekapetanovic.com/blog/qwen3.5-apple-silicon-benchmark/ ; https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks ; https://ollama.com/blog/mlx ; https://dev.to/thefalkonguy/installing-qwen-35-on-apple-silicon-using-mlx-for-2x-performance-37ma ; https://www.orcarouter.ai/blog/qwen3-8-flash-next-uncensored
- https://www.promptquorum.com/local-llms/m5-pro-max-llm-benchmarks-2026 ; https://llmcheck.net/blog/apple-silicon-m5-max-local-ai-guide/ ; https://llmcheck.net/benchmarks
- https://docs.vllm.ai/en/stable/features/multimodal_inputs/ ; https://github.com/vllm-project/vllm/issues/30847 ; https://github.com/vllm-project/vllm/issues/45098
- https://arxiv.org/abs/2412.16117 and https://github.com/visual-ai/prunevid (PruneVid) ; https://www.arxiv.org/pdf/2503.11187v1 (FastVID) ; arXiv 2411.15024 (DyCoke), 2412.04467 (VisionZip), 2403.06764 (FastV) from memory
- https://arxiv.org/html/2604.06036v3 (CodecSight) ; https://arxiv.org/html/2605.08158 (HY-Himmel) ; https://arxiv.org/pdf/2503.13016 (motion-aware MLLM)
- https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF ; https://forums.developer.nvidia.com/t/missing-vision-reasoning-with-qwen3-5-122b-q4-on-vllm-works-on-llama-cpp/363196 ; https://debuggercafe.com/introduction-to-qwen3-5-overview-vllm-and-llama-cpp/ ; https://huggingface.co/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF
- https://dibschat.github.io/ProVideLLM/
