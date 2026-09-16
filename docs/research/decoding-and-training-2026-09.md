<!-- Research note produced 2026-09-16 by a web-research agent for the model roadmap; costs marked estimate are the note's own derivations; verify before relying on any single number. -->

I have everything needed; nothing else is pending on another fetch. Here is the report.

# Local structured decoding, small-VLM temporal fine-tuning, and detector alternatives for the home-observer event pipeline

All numbers below were read from the linked primary pages (arXiv full text/abstracts, GitHub sources, model cards, dataset/challenge pages, pricing pages). Items that could not be verified are listed at the end. Cost figures marked *estimate* are derived from stated assumptions.

## 0. Executive summary

- **Validity and content are separate problems.** Grammar constraints will take Gemma 4 E4B from 0% to ~95–100% schema-valid JSON on Apple silicon at <1.5% per-token overhead, but they cannot supply the missing perception: forcing the unsupervised fields makes the model fill them, and grammar-constrained greedy decoding provably distorts the model's own distribution toward whatever is grammatical (Park et al., NeurIPS 2024). Expect valid JSON with the same 0–3% recall and no abstention unless the schema is redesigned (`event_present` first, bounded arrays, evidence IDs as a per-request enum) and the decision is read from token log-probs rather than a verbalized confidence.
- **Language-only LoRA on ~200 windows cannot learn temporal or contact concepts.** Every published temporal-grounding tune uses 10^5–10^6 instances with the projector and usually the vision tower trainable; the one published replica of your recipe (LLM-only tuning of Qwen2.5-VL-7B on EPIC-KITCHENS-style data) left multi-step localization flat at 26% while Gemini Pro scored 88%. Small-data successes are all RL (GRPO with a tIoU reward) on a base model that already localizes; 2.5–5K samples raise Charades-STA mIoU from ~29 to ~59–61.
- **Untrimmed kitchen event detection is still won by frozen video features plus a 1-D temporal detector,** not by generative VLMs: EPIC-KITCHENS-100 action detection tops out at ~32 avg mAP (36 at tIoU 0.1) with InternVideo2-1B features and CausalTAD, trained on ~90K segments. Hand-contact detectors give a structural abstention path (no contact track, no event).
- **Calibration:** verbalized confidence from ≤4B VLMs is near-random (AUROC ~0.5); renormalized log-prob of a binary decision token or K-sample agreement is usable, then Platt/temperature on event-matched labels. Set the operating point with Learn-then-Test on quiet footage; with 10 quiet hours, guaranteeing ≤2 false events/hour at 90% confidence requires ≤13 emitted false events in the calibration set (≤1.3/h empirically).
- **Budget:** a frozen-feature detector head plus VideoMAE-S/B fine-tune costs single-digit to tens of dollars on an H100 ($2.69–4.29/h); a 4B-VLM LoRA over 20K–100K 8-frame windows costs roughly $40–350 (*estimate*). The scarce resource is labeled hours, not GPU time: EPIC-KITCHENS is ~900 actions/hour, and point-level labels retain ~90%+ of full-supervision mAP at 6× lower annotation cost.

---

## 1. Constrained/structured decoding that runs locally on Apple silicon for Gemma 4

### 1.1 llama.cpp (GGUF + mtmd)

| Item | Status (Sept 2026) | Source |
|---|---|---|
| Gemma 4 vision in llama.cpp | `ggml-org/gemma-4-E2B-it-GGUF`, `-E4B-it-GGUF`, `-26B-A4B-it-GGUF`, `-31B-it-GGUF` listed under "Gemma 4 (Vision & Audio)"; run with `llama-server -m model.gguf --mmproj file.gguf`; works in `llama-cli`, `llama-server` (OpenAI `/chat/completions`), `llama-mtmd-cli` | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md |
| E4B GGUF files | Q4_0 4.59 GB, Q8_0 8.03 GB, BF16 15.1 GB; `mmproj-gemma-4-E4B-it-Q8_0.gguf` (560 MB); `llama server -hf ggml-org/gemma-4-E4B-it-GGUF:BF16` | https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF , https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf |
| Vision fix | PR #28335 (merged 2026-09-04): enforce causal attention for E2B/E4B, causal global layers, vision token budget raised from (40, 280) to (70, 1120) | https://github.com/ggml-org/llama.cpp/pull/28335 |
| Open regression | Issue #28954 (opened 2026-09-15): images above ~1.2 Mpx hit `ggml_assert "non-causal attention requires n_ubatch >= n_tokens"` on 12B/26B/31B QAT; images ≤1.15 Mpx work; caused by the 09-04 commit; unresolved | https://github.com/ggml-org/llama.cpp/issues/28954 |
| Grammar/JSON schema per request | `grammar` (GBNF), `json_schema`, `response_format` (`json_object` with `schema`, or `json_schema`) on `/completion` and `/v1/chat/completions`; schema is only used to constrain sampling, "not injected into the prompt" | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md , https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md |
| Schema features | Types, `minLength/maxLength`, `minItems/maxItems`, integer `minimum/maximum`, `required`, `properties`, `$ref`, `anyOf/oneOf`, anchored `pattern`; `additionalProperties` defaults to false; unsupported: `uniqueItems`, `contains`, `not`, if/then/else, `patternProperties`, number-typed min/max; `prefixItems` broken. GBNF adds token terminals `<[token-id]>` and warns that `x? x? x?…` repetition "may result in extremely slow sampling" (use `x{0,N}`) | https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md |
| Lazy grammars | PR #9639 (merged 2025-01-30): output "completely unconstrained before triggers, and completely constrained after"; trigger words/tokens wired into the sampler — lets free-form evidence text precede the JSON | https://github.com/ggml-org/llama.cpp/pull/9639 |
| llguidance backend | `cmake -B build -DLLAMA_LLGUIDANCE=ON` (needs Rust); grammars starting with `%llguidance` and all JSON-schema requests are routed to llguidance; Lark-style CFGs + JSON schema; ~50 µs mean / 0.5 ms p99 / 20 ms p100 per token mask on a 128k tokenizer | https://github.com/ggml-org/llama.cpp/blob/master/docs/llguidance.md , https://github.com/guidance-ai/llguidance |
| LoRA adapters | `--lora`, `--lora-scaled FNAME:SCALE`, `--lora-init-without-apply`; per-request `lora` list; `GET/POST /lora-adapters`; `convert_lora_to_gguf.py` reuses the HF converter's model classes (so any supported text arch incl. gemma4), but exits if the adapter contains embeddings or `lm_head`, and has no handling for vision-tower/projector tensors | https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md , https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py |
| Log-probs | `n_probs`, `post_sampling_probs`, `top_logprobs` in responses (needed for §4) | server README above |
| Video/multi-image | `image_url` (URL or base64), `input_audio`, `input_video` message parts; local files need `--media-path` | server README above |

Practical implication: if you later unfreeze the vision tower/projector, merge the adapter into the HF weights and reconvert both the text GGUF and the mmproj (`convert_hf_to_gguf.py --mmproj`); the LoRA converter only carries text-side adapters. https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md

### 1.2 MLX ecosystem

| Stack | Gemma 4 | Structured output | Engine | Notes | Source |
|---|---|---|---|---|---|
| **mlx-vlm** (v0.7.1, 2026-09-14) | Yes (`mlx-community/gemma-4-E4B-it-bf16`, plus E2B/E4B `-assistant` drafters for speculative decoding) | `json_schema` on `/v1/chat/completions` and `/v1/responses` | **llguidance ≥1.7.0**: `LLMatcher` + `llguidance.numpy.fill_next_token_bitmask`, mask applied by a custom Metal kernel (`-inf` on disallowed tokens); grammar compiled per request | `--adapter-path` for LoRA; `--video --fps` documented for Qwen2-VL; **open regression #1294** (2026-06-05): on 0.6.0/0.6.1, `json_schema` + thinking → HTTP 500 `ParserTooComplex` with a small `thinking_budget`, or runaway invalid JSON to `max_tokens` otherwise; fix status not confirmed | https://github.com/Blaizzy/mlx-vlm , https://raw.githubusercontent.com/Blaizzy/mlx-vlm/main/requirements.txt , https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/structured.py , https://github.com/Blaizzy/mlx-vlm/issues/1294 , https://github.com/Blaizzy/mlx-vlm/releases |
| **mlx-lm** | text only | none documented; server has `logprobs` and per-request `adapters` | `generate`/`stream_generate` accept `logits_processors` "which take the token history and current logits" | Hook for XGrammar/llguidance/Outlines | https://github.com/ml-explore/mlx-lm , https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md |
| **XGrammar** | via mlx-lm hook | JSON schema, regex, EBNF, structural tags | Ships `python/xgrammar/kernels/apply_token_bitmask_mlx.py` (`@mx.compile def apply_token_bitmask_mlx(bitmask, logits, vocab_size)`), install `pip install 'xgrammar[metal]'`; "keep one persistent compiler per model" so the compile cache is shared; compilation "non-negligible", do it asynchronously | Official integrations list vLLM/SGLang/TRT-LLM/MLC/MAX/OpenVINO; MLX kernel exists but no packaged mlx-vlm integration | https://github.com/mlc-ai/xgrammar , https://github.com/mlc-ai/xgrammar/tree/main/python/xgrammar/kernels , https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html |
| **Outlines** | via mlx-lm (`outlines.from_mlxlm(*mlx_lm.load(...))`) | JSON schema, regex, CFG, `Literal` choices | regex→FSM index (outlines-core) | "constrained generation is not supported with batching"; Apple-silicon-only backend; no VLM path | https://dottxt-ai.github.io/outlines/latest/features/models/mlxlm/ , https://github.com/dottxt-ai/outlines-core |
| **vllm-mlx** | Gemma 3/4, Qwen3-VL, Pixtral, Llama vision; video | `response_format` JSON schema | **lm-format-enforcer** | continuous batching, paged KV, prefix cache; LoRA not mentioned | https://github.com/waybarrios/vllm-mlx |
| **LM Studio** | GGUF and MLX | `response_format: json_schema` | GGUF → llama.cpp grammars; MLX → Outlines | Warns "Not all models are capable of structured output, particularly LLMs below 7B parameters" | https://lmstudio.ai/docs/app/api/structured-output |
| **Ollama** | `gemma4` e2b/e4b (Text, Image, Audio; 128K ctx; `-mlx` variants) | `format` = JSON schema (since 2024-12-06), shown working with a vision model | not documented | | https://ollama.com/library/gemma4 , https://ollama.com/blog/structured-outputs |
| Guidance | — | select()/regex/JSON/CFG | llguidance | Backends: Transformers, llama.cpp, OpenAI; **no MLX backend** | https://github.com/guidance-ai/guidance |
| lm-format-enforcer | — | JSON schema, regex | character-level tokenizer trie | Integrations: transformers, LangChain, LlamaIndex, llama.cpp, vLLM, Haystack, TRT-LLM, ExLlamaV2; **no MLX** (vllm-mlx wires it itself) | https://github.com/noamgat/lm-format-enforcer |

### 1.3 Overhead

| Engine | Per-token mask cost (JSON schema, 128k vocab) | CFG | Schema compile | Source |
|---|---|---|---|---|
| XGrammar | <40 µs | <40 µs | cached per compiler; "non-negligible" | https://arxiv.org/html/2411.15100 (Fig. 9) |
| llama.cpp GBNF | ~100 µs | ~2,500 µs | 0.05 s median | XGrammar Fig. 9; https://arxiv.org/html/2501.10868 (Table 2) |
| Outlines | ~120 µs | ~4,000 µs | **3.71 s median** (TTFT 3.97 s) | same |
| lm-format-enforcer | ~150 µs | — | lazy | XGrammar Fig. 9 |
| llguidance (Guidance) | ~50 µs mean, p99 0.5 ms | same engine | ~0 ("essentially no startup cost") | https://github.com/guidance-ai/llguidance ; JSONSchemaBench Table 2: compile 0.00 s, TPOT 7.44 ms vs 15.83 ms unconstrained (token fast-forwarding) |

Reference latency on Apple silicon (llama.cpp, 7B Q4_0, updated 2026-08-25): 36.4 t/s M1 Pro → 83.1 t/s M4 Max (F16: 12.8–41 t/s), i.e., 12–80 ms/token, so even the slow CFG path in llama.cpp (~2.5 ms) is ≤20% and JSON-schema masks are ≤1.5%. https://github.com/ggml-org/llama.cpp/discussions/4167 . Gemma 4's vocabulary is 262,144 entries (`mlx-community/gemma-4-E4B-it-bf16` config), twice Llama-3's, so expect roughly 2× the vocab-proportional costs above (*inference*). https://huggingface.co/mlx-community/gemma-4-E4B-it-bf16/blob/main/config.json . Coverage/compliance on JSONSchemaBench (GlaiveAI): Guidance 0.96/0.98, llama.cpp 0.95/0.97, Outlines 0.95/0.96, XGrammar 0.93/0.93; on GitHub-Hard schemas Guidance 0.41 vs Outlines 0.03. https://arxiv.org/html/2501.10868

### 1.4 Binding enums to a per-request vocabulary (allowed evidence IDs)

Yes, in every local engine, with different per-request costs:
- **llama.cpp**: the grammar/schema is a per-request field; put the allowed IDs in a JSON-schema `enum` (or a GBNF alternation of literals, or `<[token-id]>` terminals); compile ≈0.05 s. Cross-field constraints (e.g., `pre` before `post`) are not expressible in JSON schema but can be enumerated in GBNF/Lark as allowed pairs.
- **llguidance (mlx-vlm server, llama.cpp with `LLAMA_LLGUIDANCE`)**: masks computed lazily, no precomputation, so a fresh enum per request costs ~nothing; mlx-vlm builds the `LLMatcher` per request from the submitted schema. https://github.com/guidance-ai/llguidance , https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/structured.py
- **XGrammar**: each distinct schema is a new compile (cache keyed on the schema), so a per-request enum defeats the cache; compile asynchronously and keep the enum small. https://xgrammar.mlc.ai/docs/latest/using_xgrammar/engine_integration.html
- **Outlines**: a new schema per request means a new FSM index (3.7 s median on JSONSchemaBench) — unsuitable for dynamic enums unless the ID alphabet is fixed (e.g., always `F0..F7`, `T0..T15`). https://arxiv.org/html/2501.10868
- **Guidance `select()`** takes a runtime list, but has no MLX backend. https://github.com/guidance-ai/guidance

### 1.5 What constraints will and will not fix

- Grammar-constrained decoding "distort[s] the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM"; ASAp fixes this only via repeated sampling (Park et al., NeurIPS 2024). https://arxiv.org/abs/2405.21047 . DOMINO shows token misalignment between constraints and sub-word vocabularies "significantly impair[s] task accuracy". https://arxiv.org/abs/2403.06988
- Format restrictions hurt reasoning when the answer field precedes the reasoning field: GPT-3.5 GSM8K 76.6% text → 49.3% JSON-mode; Claude-3-Haiku 86.5 → 23.4; LLaMA-3-8B 74.7 → 48.9; "100% of GPT 3.5 Turbo JSON-mode responses placed the 'answer' key before the 'reason' key". https://arxiv.org/html/2408.02442 . With the schema in the prompt and reasoning before the answer, constraints help slightly (GSM8K 0.77 → 0.78, Last Letter 0.73 → 0.77, Shuffled Objects 0.41 → 0.44; JSONSchemaBench GSM8K: unconstrained 80.1 → Guidance 83.8 / llama.cpp 82.4 / Outlines 81.6). https://blog.dottxt.ai/say-what-you-mean.html , https://arxiv.org/html/2501.10868
- Your observations map directly: the adapter learned only the supervised fields, so a grammar that demands the rest forces fabrication; the base model's "verbose object lists and truncation" are what `maxItems` and `max_tokens` bounds exist for. Recommended schema shape: `{"reasoning_evidence": str (bounded), "event_present": bool, "events": [...maxItems 3...], "evidence_ids": enum-of-request-IDs}`, with a lazy grammar so the model can write free evidence text before the JSON, and event scores read from the renormalized log-prob of the `event_present` token (§4.1).

---

## 2. Why language-only LoRA on a frozen tower fails, and the published recipes

### 2.1 Mechanisms

- **LoRA learns less.** "In the standard low-rank settings, LoRA substantially underperforms full finetuning"; full fine-tuning perturbations have rank "10–100× larger than typical LoRA ranks"; code IFT HumanEval r=16 0.358 vs full 0.497 (r=256 closes it: 0.498); targeting MLP/all modules beats attention-only; LoRA LR should be ~10× higher (5e-5–5e-4). https://arxiv.org/html/2405.09673
- **Frozen features bound what the LLM can extract.** Fusing DINOv2+SigLIP features gives "5–10% on localization and challenge tasks" in Prismatic — localization lives in the visual features. https://arxiv.org/html/2402.07865 . Cambrian-1 Finding 4: "Unfreezing the vision encoder is widely beneficial. Language-supervised models always benefit; SSL models particularly benefit on vision-centric benchmarks"; DINOv2 gains sharply from 737K → 5M instruction samples with unfreezing. https://arxiv.org/html/2406.16860 . Idefics2: LoRA-unfreezing both backbones raised the fully-autoregressive score 60.3 → 69.5 (frozen backbones diverge without LoRA). https://arxiv.org/html/2405.02246 . Apollo Finding 9: "Finetuning video encoders on only video data further improves overall performance, especially on reasoning and domain-specific tasks"; Finding 1: design decisions transfer from 2–4B models trained on ~500K samples. https://arxiv.org/html/2412.10360 . Qwen2.5-VL trains its ViT in every stage; VideoLLaMA 3 trains "only the vision encoder and projector" in stage 1 and everything afterwards. https://arxiv.org/html/2502.13923 , https://arxiv.org/html/2501.13106
- **Counter-evidence to respect:** Prismatic found "finetuning the visual backbone significantly degrades performance (p=0.00381), especially on tasks requiring fine-grained spatial reasoning" under a LLaVA-1.5-scale (665K) single-stage recipe — unfreezing needs LoRA/regularization and data, not 200 windows. https://arxiv.org/html/2402.07865
- **Gemma 4 E4B specifics.** 4.5B effective (8B with embeddings), 42 layers, ~150M vision encoder, per-image budget 70/140/280/560/1120 tokens (config default `vision_soft_tokens_per_image: 280`), video "as frames" at 1 fps up to 60 s, 128K context; no timestamp encoding is documented. So 8 frames per 3-s window is already above the 1–2 fps these models use; the deficit is frame-independent pooled features, no time signal, and an output format never trained. https://huggingface.co/google/gemma-4-E4B-it , https://huggingface.co/mlx-community/gemma-4-E4B-it-bf16/blob/main/config.json

### 2.2 Sample counts and recipes of published temporal-grounding fine-tunes

Charades-STA = R@1 IoU 0.5 / 0.7 unless marked mIoU; ZS = not trained on Charades train.

| Model | Base LLM | Training data | Trained components | Frames / fps | Charades-STA | URL |
|---|---|---|---|---|---|---|
| Vid2Seq (2023, non-LLM) | T5-Base 314M | YT-Temporal-1B; YouCook2 few-shot 1%/10% train → CIDEr 10.1/18.4 vs 47.1 full | full | 100 time tokens | — | https://arxiv.org/abs/2302.14115 |
| TimeChat (CVPR'24) | LLaMA-2-7B | TimeIT 125K (6 tasks, 12 datasets) + Valley 65–73K; 3 epochs on 8×V100 | ViT + LLM frozen; Q-Formers + linear tuned; LoRA r=32 | 96 frames | ZS 32.2 / 13.4 | https://arxiv.org/html/2312.02051 |
| VTimeLLM (CVPR'24) | Vicuna-1.5 7B/13B | S1 LCS-558K; S2 134K multi-event InternVid videos; S3 ~36K dialogues | LoRA r=64 (α=128), linear projector; CLIP frozen | 100 frames | 27.5/11.4 (7B), 34.3/14.7 (13B) | https://arxiv.org/html/2311.18445 |
| Momentor (ICML'24) | LLaMA-7B | Moment-10M: 64.9K videos, 1.46M segments, 7,260 h; 8×A100 ≈ 60 h | projector + Temporal Perception Module only | 300 frames; 300 time tokens | 26.6 / 11.6 | https://arxiv.org/html/2402.11435 |
| HawkEye (2024) | Vicuna-7B (VideoChat2) | InternVid-G: 83,614 videos, 715,489 queries; negative spans avg 203 s vs 4.4 s positives | Q-Former + LoRA; encoder frozen | **12 frames**; 4-way {beginning, middle, end, throughout} + recursive refinement | FT 58.3 / 28.8 (mIoU 49.3) | https://arxiv.org/html/2403.10228 |
| LITA (ECCV'24) | Vicuna 7B/13B | ~500K mix incl. ActivityNet-RTL 33,557 QA | — | 100 frames → 100 fast + 4 slow (356 tokens); 100 time tokens | ANet-RTL mIoU 28.6, P@0.5 25.9 (13B) | https://arxiv.org/html/2403.19046 |
| E.T. Chat (2024) | Phi-3-Mini 3.8B | E.T. Instruct 164K | LoRA + matching projectors | 1 fps | E.T. Bench TVG F1 38.6 | https://arxiv.org/abs/2409.18111 |
| Grounded-VideoLLM (2024) | Phi-3.5-V 3.8B | S1 1.28M; S3 ~1.1M incl. 17K grounded VQA | S1 projectors; S2 +embeddings/head; S3 LoRA | 96 frames / 12 segments; 300 temporal tokens | 36.4 / 19.7 (mIoU 36.8) | https://arxiv.org/html/2410.03290 |
| TRACE (ICLR'25) | Mistral-7B | S1 1.9M; S2 0.9M | S1 compressor + time/score heads; S2 LLM full; encoder frozen | 128 frames × (8 visual + 6 time tokens); 8→128 frames: 28.8→41.2 R@0.5 | ZS 40.3 / 19.4 (mIoU 38.7) | https://arxiv.org/html/2410.05643 |
| TimeSuite (2024) | Mistral-7B | TimePro 349K | LoRA + TAPE | 128 frames | ZS 48.7 / 24.0 | https://arxiv.org/abs/2410.19702 |
| TimeMarker (2024) | LLaMA3-8B | ~5M video-text + images | all params incl. encoder; "Second{i}" text before each frame | 2 fps short videos | ZS 51.9 / 26.9 | https://arxiv.org/abs/2411.18211 |
| VideoLLaMA 3 (2025) | Qwen2.5 2B/7B | 0.21M temporal-grounding (Charades, Moment-10M, ANet…) within 3.03M video SFT | encoder+projector (S1), all (S2–4) | 1 fps, ≤180 frames | mIoU 55.5 (2B) / 60.7 (7B), in-domain | https://arxiv.org/html/2501.13106 |
| InternVideo2.5 (2025) | InternLM2.5-7B | TG 116.5K (S2) + 7.5K QVH (S3) | ViT trained in S3 | 64–512 frames, ~16 tok/frame | R@0.5 43.3, mIoU 41.7 | https://arxiv.org/html/2501.12386 |
| Qwen2.5-VL (2025) | 3B/7B/72B | undisclosed; timestamps in s and hmsf | ViT + LLM all stages; absolute-time MRoPE; dynamic fps | ≤768 frames | mIoU 38.8 / 43.6 / 50.9 | https://arxiv.org/html/2502.13923 |
| Time-R1 (2025, RL) | Qwen2.5-VL-7B | **2.5K** samples (from 339K); GRPO; reward = timestamp-aware tIoU + format | full | 2 fps | ZS 60.8 / 35.3 (mIoU 58.1) | https://arxiv.org/html/2503.13377 |
| VideoChat-R1 (2025, RL) | Qwen2.5-VL-7B | **5,338** Charades samples (18,031 total) | GRPO | — | base 24.2/11.1 (mIoU 29.0) → SFT 45.0/25.3 (46.3) → GRPO 71.7/50.2 (60.8) | https://arxiv.org/html/2504.06958 |
| TempSamp-R1 (2025) | Qwen2.5-VL-7B | Charades train | GRPO + off-policy GT | 2 fps | R1@0.7 52.9; ANet R1@0.5 56.0 | https://arxiv.org/abs/2509.18056 |
| VideoLLM-online (CVPR'24, egocentric streaming) | Llama-2-7B / Llama-3-8B | Ego4D narration streams + COIN | LoRA r=128 + MLP; streaming-EOS loss w=1.0 | 2 fps, 1 token/frame | Ego4D narration TimeDiff 2.32 s, Fluency 42.6% | https://arxiv.org/html/2406.11816 |

### 2.3 Frames, fps and token budgets
Apollo: fps sampling beats uniform; "8–32 tokens per frame being optimal" with little dependence on fps beyond that; 10–14% text in the mix. https://arxiv.org/html/2412.10360 . TRACE's 8→64→128-frame ablation (28.8→37.0→41.2 R@0.5) and TimeChat's 32→96-frame gains say the model needs context well beyond one 3-s window to place boundaries; HawkEye shows 12 frames suffice only if the target is coarse bins refined recursively. https://arxiv.org/html/2410.05643 , https://arxiv.org/html/2403.10228

### 2.4 Abstention and negative-heavy supervision
- VideoLLM-online supervises an EOS token on every frame where no response is due (streaming loss, default weight 1.0); ablating it drops Fluency 42.6% → 37.7% and worsens TimeDiff 2.32 → 2.52 s; at inference EOS is accepted only above a threshold θ≈0.5–0.8. https://arxiv.org/html/2406.11816
- HawkEye's InternVid-G is negative-span-heavy (203 s negative vs 4.4 s positive per query); TimeChat-Online adds 20K unanswerable samples of 139K; LRV-Instruction (400K, three negative types) finds models hallucinate heavily on negatives and (per the paper) a 1:1 pos:neg ratio is optimal. https://arxiv.org/html/2403.10228 , https://arxiv.org/abs/2504.17343 , https://arxiv.org/abs/2306.14565
- Video LLMs' yes-bias is large: EventHallusion binary accuracy Video-LLaVA 45.97%, VideoChat2 32.76%, LLaVA-NeXT-Video 64.79% vs GPT-4o 91.93%; description matching 0–20% for open models. https://arxiv.org/html/2409.16597

### 2.5 Token-weighted losses
Rho-1 trains only on high-excess-loss tokens (up to +30 points few-shot math; DeepSeekMath-level with 3% of tokens). https://arxiv.org/abs/2404.07965 . Instruction Modelling (loss on instruction tokens too) helps most with brief outputs and few examples (AlpacaEval +100%). https://arxiv.org/abs/2405.14394 . No paper studies down-weighting JSON scaffolding specifically; the extrapolation is to mask fixed keys/brackets and up-weight timestamps, class, and the `event_present`/`no_event` tokens.

### 2.6 DPO/RL for hallucination and abstention
RLAIF-V: −80.7% object hallucination (7B). https://arxiv.org/abs/2405.17220 . RLHF-V: 1.4K annotated samples, −34.8% hallucination. https://arxiv.org/abs/2312.00849 . For temporal grounding, GRPO with a tIoU reward is the data-efficient route (Time-R1 2.5K; VideoChat-R1 5.3K; SFT on the same data reaches only mIoU 46.3 vs 60.8). Caveat: these start from Qwen2.5-VL, which already localizes at mIoU ~29–44; RL cannot bootstrap a capability the base scores ~0 on, so measure Gemma 4's zero-shot window-level event-presence accuracy first.

### 2.7 Egocentric / EPIC-KITCHENS specifics
- LLaVAction (ICLR 2026): 7B video-LLMs trained on 530K EK-100-derived pairs with vision encoder, projector and LLM all trained (12 h on 32 GH200) reach EK100-MQA 74.1% vs GPT-4o 52.2% — clip classification, not untrimmed detection. https://arxiv.org/abs/2503.18712
- The closest replica of your recipe (Qwen2.5-VL-7B, "only LLM components were tuned, freezing the vision tower and MLP projector", EPIC/Ego4D/VISOR data): overall 33.5% → 41.6%, but multi-step localization stayed at 26% vs Gemini Pro 88%. https://arxiv.org/abs/2601.10228
- No 2024–2026 paper fine-tunes a <8B VLM on EK-100/Ego4D and reports untrimmed-video recall/precision; untrimmed detection on EK-100 is done with VideoMAE/InternVideo features + temporal detectors (§3).

---

## 3. Alternatives to generative JSON

### 3.1 Frozen video features + temporal detector head (EPIC-KITCHENS-100 Action Detection, mAP@[0.1:0.5])

| Method | Features | Verb | Noun | Action | URL |
|---|---|---|---|---|---|
| ActionFormer (val) | SlowFast-R50 (official) | 23.51 | 21.88 | — | https://github.com/happyharrycn/actionformer_release |
| TriDet (val) | SlowFast | 25.51 | 23.76 | — | https://github.com/dingfengshi/TriDet |
| DyFADet (val) | SlowFast | 25.0 | 23.4 | — | https://arxiv.org/html/2407.03197 |
| CausalTAD (val / **test, 2024 1st**) | **InternVideo2-1B** (EPIC-finetuned) | 27.67 / 30.02 | 34.44 / 35.22 | 29.63 / **31.97** | https://arxiv.org/abs/2407.17792 |
| EgoAction (2026 3rd) | VideoMAE-L EPIC-finetuned; separate causal verb/noun detectors; top-K composition; class-wise Soft-NMS | 28.66 | 28.61 | 25.94 (29.56 at tIoU 0.1) | https://arxiv.org/html/2605.24496v1 |

Swapping features (SlowFast → InternVideo2-1B) moved noun mAP ~24 → 34; three years of detector-head changes moved verb mAP ~23.5 → 26.4. Features dominate. Training the head on EK100 needs ~4.5 GB GPU memory (ActionFormer README) and runs on one GPU (TriDet: single A100-40G).

### 3.2 Dense per-frame heads (TSU / Charades, per-frame mAP)
MS-TCT (I3D RGB, 6.6 GFLOPs): Charades 25.4, TSU-CS 33.7, MultiTHUMOS 43.1 (PDAN 23.7 Charades; I3D alone 15.6). https://ar5iv.labs.arxiv.org/html/2112.03902 . With frozen CLIP features: AAN 41.3 TSU / 32.0 Charades; MS-Temba (10M params, 0.6 GFLOPs) 42.0 / 32.3. https://arxiv.org/abs/2309.00696 , https://arxiv.org/html/2501.06138v1 . Event-level mAP on TSU is far lower than per-frame mAP (AGNet 22.7/15.3/6.0 at IoU 0.3/0.5/0.7 vs 33.2 per-frame). https://arxiv.org/abs/2010.14982

### 3.3 Online (causal) detection
LSTR 69.5 mAP THUMOS14 / 89.1 mcAP TVSeries; MiniROAD 71.8 / 89.6; E2E-LOAD 72.4 / 90.3 at 17.3 FPS end-to-end. https://arxiv.org/abs/2107.03377 , https://github.com/jbistanbul/MiniROAD , https://ar5iv.labs.arxiv.org/html/2306.07703 . Online instance formation (MATR) drops to 49.5 avg mAP vs ~70 offline; OAD-derived instances ~20. https://arxiv.org/html/2408.02957 . None report FP/hour.

### 3.4 Hand-object contact detectors, tracks, and per-track verification

| System | Output | Numbers | URL |
|---|---|---|---|
| 100DOH (CVPR'20) | hand box, side, 5-way contact state {none, self, other person, portable, non-portable}, object box | AP: hand 89.6, +side 78.9, +state 64.0, +object 46.9, all 38.5; training on 15K instead of 90K frames drops "all" to 19.7; weak on egocentric hand side | https://ar5iv.labs.arxiv.org/html/2006.06669 |
| VISOR HOS (NeurIPS'22) | hands, contact state, in-contact object masks; 272K manual masks, 67K hand-object relations, 36 h | Mask AP (val/test): hands 90.9/95.4, contact state 73.5/78.7, in-contact object 30.5/33.7, active object 24.1/25.7 | https://ar5iv.labs.arxiv.org/html/2209.13064 |
| EgoHOS (ECCV'22) | 11,243 images; hands, 1st/2nd-order objects, contact boundary | mIoU LH 87.7, RH 88.8, LH-obj 62.2 | https://arxiv.org/abs/2208.03826 |
| InterTracker (2023) | discover + track interacting objects | object AP@0.5 50.27 vs 100DOH 47.83 | https://ar5iv.labs.arxiv.org/html/2308.03061 |
| Frame-level interaction/no-interaction (2022) | hand pose + masks | 89% frame accuracy at >30 FPS | https://ar5iv.labs.arxiv.org/html/2211.09067 |
| TouchMoment (CVPR Findings 2026) | frame-precise contact moments; 4,021 videos, 8,456 moments | AP with a 2-frame tolerance, +16.91 AP over event-spotting baselines | https://arxiv.org/abs/2604.12343 |
| EgoLoc (2025) | zero-shot VLM contact/separation timestamps with hand-dynamics-guided sampling | qualitative in abstract | https://arxiv.org/abs/2508.12349 |
| Ego4D STA | next-active object + verb + time-to-contact | Overall Top-5 mAP 5.12 (StillFast v2 test) → 7.21 (EgoVideo 2024) | https://ar5iv.labs.arxiv.org/html/2304.03959 , https://arxiv.org/html/2406.18070v1 |
| Ego4D MQ (110 classes) | temporal moments | test avg mAP 23.59 (2022) → 34.99 (CausalTAD 2024); Soft-NMS σ 0.9→2.0 +1.3–1.8 mAP; FN rate 54% on the shortest moments | https://arxiv.org/abs/2211.09529 , https://arxiv.org/abs/2407.17792 , https://arxiv.org/abs/2307.02025 |

Components for a hybrid pipeline (all verified): Grounding DINO 52.5 zero-shot COCO AP https://arxiv.org/abs/2303.05499 ; OWLv2 LVIS-rare 44.6 AP https://arxiv.org/abs/2306.09683 ; SAM 2 streaming-memory video segmentation, "3× fewer interactions" https://arxiv.org/abs/2408.00714 ; Cutie +8.7 J&F over XMem on MOSE https://arxiv.org/abs/2310.12982 . Actor-centric detection (AVA, 80 atomic actions, 1.58M labels; VideoMAE ViT-L 39.3 mAP) has no egocentric-hand equivalent. https://arxiv.org/abs/1705.08421 , https://ar5iv.labs.arxiv.org/html/2203.12602

### 3.5 Grounded decoding: select from candidates instead of generating coordinates
- Set-of-Mark: GPT-4V asked to output coordinates scores 25.7 on RefCOCOg REC; with numbered marks 86.4 (RES 75.6 mIoU). https://arxiv.org/html/2310.11441
- Groma: Deformable-DETR proposes 300 regions → ≤100 region tokens; the LLM selects tokens; RefCOCO/+/g avg 86.52 vs Shikra 82.93, Ferret 83.91, MiniGPT-v2 84.29; LVIS-Ground AR 28.8 vs Ferret 16.8. https://arxiv.org/html/2404.13013
- ViP-LLaVA: visual overlays 70.86 vs textual coordinates 61.4 on PointQA-LookTwice. https://arxiv.org/html/2312.00784
- Florence-2 (0.23B/0.77B) generates 1,000-bin location tokens: fine-tuned RefCOCO 95.3 / COCO 43.4 mAP — a small specialist, not a chat model. https://arxiv.org/html/2311.06242
- Temporal analogue: Grounded-VideoLLM's 300 discrete time tokens give +1.5 mIoU over plain-text timestamps; E.T. Chat replaces timestamp generation with matching. https://arxiv.org/html/2410.03290 , https://arxiv.org/abs/2409.18111
- Video: tracker/decoder-coupled outputs (Sa2VA 75.2 J&F Ref-DAVIS17; VideoGLaMM GCG mIoU 62.34) beat LLM-emitted per-frame boxes (PG-Video-LLaVA 35.1 mIoU VidSTG; Elysium 56.1 AUC LaSOT). https://arxiv.org/html/2501.04001 , https://arxiv.org/html/2411.04923 , https://ar5iv.labs.arxiv.org/html/2311.13435 , https://arxiv.org/html/2403.16558

### 3.6 Systems that report false-positive rates on continuous video
- UCF-Crime (1,900 videos, 128 h): "false alarm rate on normal testing videos" at threshold 0.5 — MIL 1.9%, Lu et al. 3.1%, Hasan et al. 27.2% (AUC 75.41). https://ar5iv.labs.arxiv.org/html/1801.04264 . Later VAD work reports AUC only: VadCLIP 88.02 AUC / XD-Violence 84.51 AP; LAVAD (training-free, BLIP-2 + Llama-2-13b, 10-s windows) 80.28 AUC; VERA (frozen InternVL2-8B) 86.55 AUC; Holmes-VAD 89.51. https://arxiv.org/abs/2308.11681 , https://arxiv.org/html/2404.01014 , https://arxiv.org/html/2412.01095 , https://arxiv.org/html/2406.12235
- Hallucination rates on absent events: VideoHallucer FP ratio 0.15–0.91 across models; ARGUS ~40–45% sentence-level hallucination cost even for Gemini-2.0-Flash/GPT-4o; Vript-HAL GPT-4V caption precision 77.3 / recall 36.2. https://arxiv.org/html/2406.16338 , https://arxiv.org/html/2506.07371v2 , https://arxiv.org/html/2406.06040
- Streaming event-start detection with FP-limited recall (SDQES, Ego4D): best SR@1 29.1% on 1-min clips vs human 72.4%. https://arxiv.org/abs/2412.03567

---

## 4. Calibration, abstention, and duplicate suppression

### 4.1 Which confidence signal to use
- Verbalized confidence: overconfident, clustered at 80–100% (Xiong et al., ICLR 2024; consistency across samples raised GSM8K AUROC 54.8 → 92.7 with 5 samples); Tian et al.'s "verbalized beats token-prob" result holds for RLHF frontier models, not small ones. https://arxiv.org/abs/2306.13063 , https://arxiv.org/abs/2305.14975 . For Qwen2-VL-2B/SmolVLM-class models, verbalized confidence AUROC ≈0.39–0.75 (typically ~0.5) vs mean token probability 0.92–0.99. https://arxiv.org/abs/2607.22034
- Usable signals: (i) renormalized probability of the binary decision token under the grammar, s = p(true)/(p(true)+p(false)) (llama-server `n_probs`/`post_sampling_probs`; mlx-lm `logprobs`), bias-corrected with a content-free input (Zhao et al.) https://arxiv.org/abs/2102.09690 ; (ii) P(True) second pass with few-shot examples (Kadavath: zero-shot P(True) "lies close to 50%", improves with few-shot) https://arxiv.org/abs/2207.05221 ; (iii) K-sample agreement / semantic entropy (~5 samples) https://arxiv.org/abs/2302.09664 ; (iv) a separate calibrated head (APRICOT) https://arxiv.org/abs/2403.05973 .
- Post-hoc: temperature/Platt scaling on a held-out set (Guo et al.) https://arxiv.org/abs/1706.04599 ; for detections, calibrate conditionally on box position/scale (Küppers et al., D-ECE; netcal) https://arxiv.org/abs/2004.13546 , https://github.com/EFS-OpenSource/calibration-framework . The correctness label must be "event matched to ground truth within collar/tIoU", not "JSON parsed".

### 4.2 Guarantees: conformal prediction, risk control, selective prediction
- Split conformal: q̂ = ⌈(n+1)(1−α)⌉/n quantile of calibration scores; coverage in [1−α, 1−α+1/(n+1)]; n≈1000 gives coverage ~0.88–0.92 at target 0.90. https://arxiv.org/abs/2107.07511
- RCPS / Learn-then-Test: choose λ by testing R(λ) > α with a binomial (or Hoeffding-Bentkus) p-value; guarantee P(R(λ̂) ≤ α) ≥ 1−δ; ~1,000 calibration points for α=0.1, ~10,000 for α=0.001; supports FDR/FNR/IoU risks. https://arxiv.org/abs/2101.02703 , https://arxiv.org/abs/2110.01052 . Conformal Risk Control: λ̂ = inf{λ : (n/(n+1))R̂_n(λ) + B/(n+1) ≤ α} gives E[loss] ≤ α (slack B/(n+1)). https://arxiv.org/abs/2208.02814
- Detection applications: Andéol et al. (railway signals) control box false-negative rate via CRC with 1,000–1,914 calibration images; no conformal paper for temporal action localization was found. https://arxiv.org/abs/2304.06052 . LLM applications: conformal factuality (drop sub-claims until 80–90% correctness) https://arxiv.org/abs/2402.10978 ; conformal abstention (self-consistency score + conformal threshold bounds hallucination rate) https://arxiv.org/abs/2405.01563 .
- Selective prediction: softmax-response thresholding with a risk guarantee (SGR): "2% error in top-5 ImageNet … with probability 99.9%, and almost 60% test coverage"; SelectiveNet trains coverage end-to-end. https://arxiv.org/abs/1705.08500 , https://arxiv.org/abs/1901.09192

**Worked operating point.** N = 10 quiet hours, target ≤2 false events/h at 90% confidence, with a ≥60-s refractory so each minute is a 0/1 unit: n = 600, α = 2/60 = 0.0333. LTT with the exact binomial tail: p(k) = P(Bin(600, 0.0333) ≤ k); p(13) ≤ 0.1 < p(14) = 0.101, so any threshold whose quiet-set count is **k ≤ 13 (≤1.3 FP/h empirically)** is certified; take the loosest such threshold. A Hoeffding bound on the same n wastes 2.6 FP/h of slack, and CRC on 10 hour-units needs ≤0.2 empirical FP/h because the B/(n+1) term eats 91% of the budget — use binomial/LTT with minute units, and Adaptive Conformal Inference for drift (minutes within an hour are only approximately exchangeable). https://arxiv.org/abs/2110.01052 , https://arxiv.org/abs/2106.00170

### 4.3 Duplicate suppression across overlapping windows

| Stage | Practice | Values | Source |
|---|---|---|---|
| Score smoothing | median filter over window scores | DESED baseline `median_window: 7` frames; class-dependent 9–27 frames | https://raw.githubusercontent.com/DCASE-REPO/DESED_task/master/recipes/dcase2023_task4_baseline/confs/default.yaml , https://arxiv.org/abs/1906.06909 |
| Hysteresis | onset threshold > offset threshold; merge gaps below tolerance | +28.6 F1 with class-wise thresholds (Cances et al.) | https://arxiv.org/abs/1906.06909 |
| Count-up trigger + refractory | Mycroft Precise `TriggerDetector`: `trigger_level=3`, `sensitivity=0.5`, negative cooldown after activation | prevents "multiple close activations" | https://raw.githubusercontent.com/MycroftAI/mycroft-precise/dev/runner/precise_runner/runner.py |
| Temporal (Soft-)NMS | ActionFormer/TriDet defaults `nms_method 'soft'`, `nms_sigma 0.5` (EK100 config 0.4), `iou_threshold 0.1`, `duration_thresh 0.05`, `multiclass_nms True`, `pre_nms_topk 5000`, `max_seg_num 2000` | Ego4D MQ: σ 2.0 when GT has near-replicates | https://github.com/happyharrycn/actionformer_release/blob/main/configs/epic_slowfast_verb.yaml , https://raw.githubusercontent.com/happyharrycn/actionformer_release/main/libs/core/config.py , https://arxiv.org/abs/2307.02025 , https://arxiv.org/abs/1704.04503 |
| Streaming LLMs | VideoLLM-online: EOS supervised on silent frames, accept EOS only if P(EOS) ≥ θ (0.5–0.8); MMDuet: relevance/informative heads with a threshold and "remove previous response" de-dup; Dispider: binary "respond now" head + `<SILENT>` | StreamingBench proactive output counted only within 2 s of GT | https://arxiv.org/html/2406.11816 , https://arxiv.org/abs/2411.17991 , https://arxiv.org/abs/2501.03218 , https://arxiv.org/abs/2411.03628 |
| Metrics that penalize duplicates | ActivityNet/EK100 evaluator locks each GT once; extra overlaps are FPs; sed_eval event F1 with 200 ms onset collar / 50% offset; PSDS with e_max = 100 FP/h; ODAS point-AP disallows duplicates | | https://raw.githubusercontent.com/activitynet/ActivityNet/master/Evaluation/eval_detection.py , https://tut-arg.github.io/sed_eval/sound_event.html , https://arxiv.org/abs/1910.08440 , https://arxiv.org/abs/1802.06822 |

Events-per-hour conventions from other fields: wake words report FRR at 0.5 FA/h (Hey Snips: 0.12% clean, 1.60% at 5 dB) https://arxiv.org/abs/1811.07684 ; seizure wearables 1.43 FA/24 h at 89.9% sensitivity https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1833080/full ; real-world fall detection ~5 false alarms/day at SE 83%/SP 97%, with >2 FA/h judged unacceptable https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0037062 . Your 1,800/h is three orders of magnitude off every deployed alerting system.

---

## 5. Training infrastructure at this scale

### 5.1 Data volumes in the reference datasets

| Dataset | Hours | Instances | Classes | Splits | URL |
|---|---|---|---|---|---|
| EPIC-KITCHENS-100 | 100 (20M frames, 700 videos, 45 kitchens) | 89,977 segments, avg 3.1±5.4 s, 28.1% overlap | 97 verbs, 300 nouns, 4,053 actions | train 67,217 (74.7 h) / val 9,668 (13.2 h) / test 13,092 (12.1 h) | https://ar5iv.labs.arxiv.org/html/2006.13256 , https://epic-kitchens.github.io/2020-100 |
| Charades | ~82 (9,848 × 30 s) | 66,500 intervals | 157 | — | https://arxiv.org/abs/1604.01753 |
| TSU | ~188 (536 × 21 min) | dense per-frame labels | 51 | CS / CV | https://project.inria.fr/toyotasmarthome/ , https://arxiv.org/abs/2010.14982 |
| Ego4D | 3,670 total; episodic memory ~74K queries over 800 h; MQ 110 classes | — | — | — | https://arxiv.org/abs/2110.07058 , https://ego4d-data.org/docs/benchmarks/episodic-memory/ |
| Ego4D STA v2 | — | 98,276 train instances | 128 nouns, 81 verbs | — | https://ar5iv.labs.arxiv.org/html/2304.03959 |

So kitchens yield ~900 labeled actions per hour; your 200 windows are ~13 minutes of EPIC-scale supervision.

### 5.2 Label-efficiency evidence
- Single-frame (point) supervision: SF-Net THUMOS14 mAP@0.5 29.3 vs fully-supervised SSN 29.1; LACP 45.3 at IoU 0.5, "6× cheaper than frame-level ones (50 s vs. 300 s per 1-min video)". https://ar5iv.labs.arxiv.org/html/2003.06845 , https://ar5iv.labs.arxiv.org/html/2108.05029
- Semi-supervised: SSTAP with 60% labels matches BMN at 100% (THUMOS AR@100 48.02 vs 47.72; ANet AUC 67.23 vs 67.10); 10% labels → 40.92. https://ar5iv.labs.arxiv.org/html/2104.03214
- Small-data video classifiers: VideoMAE fine-tunes to 91.3% UCF101 (9.5K clips) / 62.6% HMDB51 (3.5K clips) with no extra data. https://arxiv.org/abs/2203.12602 . Few-shot CLIP adaptation works on appearance classes (HMDB51 66.8% at 16-shot) and fails on motion classes (SSv2 12.4%). https://github.com/muzairkhattak/ViFi-CLIP
- Hand-contact detectors degrade sharply below ~90K frames (100DOH "all" AP 38.5 → 19.7 at 15K). https://ar5iv.labs.arxiv.org/html/2006.06669

### 5.3 Cost on H100

Prices (on-demand, single GPU): Runpod H100 SXM $2.69 (Community) / $3.49 (Secure), H100 PCIe $1.99/$2.89, A100 80GB $1.19–1.59; Lambda H100 SXM $4.29, PCIe $3.29. https://www.runpod.io/pricing , https://lambda.ai/pricing

Anchors: LLaVAction full fine-tune of a 7B video-LLM (encoder+projector+LLM) on 530K 8–16-frame clips took 12 h × 32 GH200 ≈ 384 GPU-h (~1.4K samples/GPU-h) https://arxiv.org/abs/2503.18712 ; Momentor 8×A100 × 60 h https://arxiv.org/html/2402.11435 ; RLHF-V 1.4K DPO samples in <1 h on 8×A100 https://arxiv.org/abs/2312.00849 ; Unsloth reports E4B LoRA at 17 GB VRAM https://unsloth.ai/docs/models/gemma-4/train ; VideoMAE ViT-S/B = 57/180 GFLOPs per 16×224² clip, fine-tune 50–100 epochs on small datasets, 800-epoch SSv2 pretraining 19.5 h on 64 V100 https://ar5iv.labs.arxiv.org/html/2203.12602 .

| Option | Assumptions (*estimate*) | GPU-hours | Cost at $3.49/h |
|---|---|---|---|
| (A) LoRA on 4B VLM, 8 frames × 280 tokens (~2.6K tokens/sample) | 2–5K samples per H100-h (LoRA on 4.5B effective; LLaVAction anchor scaled) | 5K windows × 3 ep: 3–8 h; 20K × 3: 12–30 h; 100K × 2: 40–100 h | $10–28 / $42–105 / $140–350 |
| (A′) GRPO on the same VLM | 8 rollouts/sample, ~4× SFT cost; 5K samples | 12–40 h | $40–140 |
| (B) VideoMAE-S/B full fine-tune from K400/SSv2 checkpoints | pre-decoded frames; 200–400 clips/s (S), 70–150 clips/s (B) | 20K clips × 50 ep ≈ 1M passes: ~1 h (S) / ~3 h (B); 100K × 30 ep: 2–4 h / 8–12 h | $4–10 / $7–42 |
| (C) Frozen-feature TAD head | feature extraction 100 h at 16-frame clips stride 4 ≈ 2.7M clips: ~0.5 h (VideoMAE-S) to ~5 h (VideoMAE-L, 597 GFLOPs); ActionFormer/TriDet head: minutes to a few hours on one GPU (4.5 GB) | 1–8 h | $4–28 |

Compute is not the constraint at any tier; labeled hours are.

### 5.4 Evaluation protocols for untrimmed video
- EPIC-KITCHENS-100 Action Detection: "Mean Average Precision (mAP) @ IOU 0.1 to 0.5", per-class AP averaged over tIoU {0.1,…,0.5}, verb/noun/action; ActivityNet-style evaluator (each GT matched once; extra overlapping predictions are FPs). https://epic-kitchens.github.io/2025 , https://github.com/epic-kitchens/C2-Action-Detection , https://raw.githubusercontent.com/activitynet/ActivityNet/master/Evaluation/eval_detection.py
- Charades localization: per-frame mAP at 25 equidistant timepoints per video (ignores duplicates and boundaries). https://arxiv.org/abs/1612.06371 . TSU: per-frame mAP (CS/CV) plus event-based mAP at IoU 0.3/0.5/0.7. https://arxiv.org/abs/2010.14982
- Ego4D MQ: average mAP over tIoU 0.1–0.5 and Recall@1x at tIoU 0.5; STA: Top-5 mAP with noun/verb/TTC constraints. https://ego4d-data.org/docs/benchmarks/episodic-memory/ , https://arxiv.org/abs/2307.02025 , https://ar5iv.labs.arxiv.org/html/2304.03959
- FP-per-hour protocols: PSDS (eTPR vs eFPR/h up to e_max=100, DTC/GTC/CTTC) https://arxiv.org/abs/1910.08440 ; UCF-Crime false-alarm rate on normal videos at threshold 0.5 https://ar5iv.labs.arxiv.org/html/1801.04264 ; wake-word FRR at fixed FA/h https://arxiv.org/abs/1811.07684 ; streaming timing (TimeDiff; proactive output within 2 s) https://arxiv.org/html/2406.11816 , https://arxiv.org/abs/2411.03628 .
- **Recommended report for this project:** (1) event-level precision/recall/F1 with a 1-s onset collar (sed_eval convention scaled to 3-s windows) and mAP@tIoU {0.1…0.5} with GT locking; (2) false events per hour on ≥10 h of held-out quiet footage, plus Pmiss at 1, 5 and 20 FP/h (PSDS-style curve); (3) per-class verb/noun breakdown as in EK100; (4) bootstrap CIs over videos; (5) for streaming, TimeDiff and SR@k with FP limits (SDQES). https://tut-arg.github.io/sed_eval/sound_event.html , https://arxiv.org/abs/2412.03567

### 5.5 Data plan tiers (*estimates* from §5.1–5.3)

| Tier | Labeled footage | Windows/events | Method | GPU-h | Cost |
|---|---|---|---|---|---|
| Minimum viable | EK100 (free, 100 h, 90K segments) + 10 h own home footage with **point labels** for 6–10 classes (≈5–9K events at kitchen density; likely fewer in living areas) + ≥10 h quiet | ~10–20K positive windows, ≥ equal quiet | (C) frozen InternVideo2/VideoMAE features + ActionFormer/CausalTAD head, hand-contact gate (100DOH/VISOR) | 5–15 | <$60 |
| Solid | 30–50 h own footage, point or segment labels (20–45K events) + EK100 pretraining | 50–100K windows | (B)+(C): fine-tune VideoMAE-B on own clips, detector head on its features; VLM only for description/evidence over detector proposals (§3.5) | 20–60 | $70–210 |
| Ambitious | 100+ h own + EK100 + Ego4D MQ/STA subsets | 200K+ windows | (A′) GRPO on the VLM (5–20K windows, tIoU + abstention reward) on top of the detector pipeline; unfreeze video encoder | 100–400 | $350–1,400 |

---

## 6. Ranked recommendations (expected impact × effort)

1. **Move event detection off the generative path: hand-contact tracks + frozen-video-feature temporal head, with the VLM as a describer/verifier over candidates.** Highest impact, medium effort. Evidence: EK100 detection 32 avg mAP with InternVideo2-1B + CausalTAD vs 0–3% recall now https://arxiv.org/abs/2407.17792 ; contact-state AP 64–79 and 89% frame-level interaction accuracy at >30 FPS https://ar5iv.labs.arxiv.org/html/2006.06669 , https://ar5iv.labs.arxiv.org/html/2209.13064 , https://ar5iv.labs.arxiv.org/html/2211.09067 ; abstention becomes structural (no contact track → no event). Cost <$60 (§5.3).
2. **Redesign the JSON contract and read decisions from log-probs, then gate with a calibrated threshold.** High impact, low effort. `event_present` first, bounded `events` (`maxItems`), evidence IDs as a per-request `enum` (llama.cpp/llguidance compile ≈0), lazy grammar so evidence text precedes the JSON https://github.com/ggml-org/llama.cpp/pull/9639 ; score = renormalized p(true) from `n_probs`/`logprobs`, Platt-scaled on event-matched labels https://arxiv.org/abs/2102.09690 , https://arxiv.org/abs/1706.04599 ; verbalized confidence discarded https://arxiv.org/abs/2607.22034 .
3. **Post-process like a deployed alerting system: median filter → hysteresis → min-duration + refractory → Soft-NMS (σ≈0.5, IoU 0.1) → LTT-certified threshold on ≥10 quiet hours.** High impact, low effort; alone it caps output at the refractory rate and gives a guarantee (worked example §4.2). https://arxiv.org/abs/1906.06909 , https://github.com/happyharrycn/actionformer_release/blob/main/configs/epic_slowfast_verb.yaml , https://arxiv.org/abs/2110.01052
4. **Evaluate with FP/hour and collar-based event F1 on held-out quiet footage before any further training.** High impact, low effort; this is the metric the literature uses for streams (PSDS e_max=100 FP/h; SDQES SR@k). https://arxiv.org/abs/1910.08440 , https://arxiv.org/abs/2412.03567
5. **If the Gemma path continues, replace SFT with GRPO using a tIoU + abstention reward on 2.5–5K windows, but first measure zero-shot event-presence accuracy.** Medium-high impact, medium effort; SFT 46.3 vs GRPO 60.8 mIoU on identical data https://arxiv.org/html/2504.06958 , https://arxiv.org/html/2503.13377 ; RL cannot bootstrap from ~0.
6. **Unfreeze projector + LoRA on the Gemma vision tower, raise the per-frame token budget, add per-frame "Second{i}" text, and extend context beyond 3 s.** Medium impact, medium effort. https://arxiv.org/html/2406.16860 , https://arxiv.org/html/2412.10360 , https://arxiv.org/abs/2411.18211 , https://arxiv.org/html/2410.05643 . Note the LoRA→GGUF converter carries text adapters only; merge and reconvert the mmproj. https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py
7. **Grounded decoding for object references: Set-of-Mark IDs over detector/tracker boxes instead of generated coordinates.** Medium impact, low effort on top of (1): 25.7 → 86.4 REC for GPT-4V; Groma +12 AR on multi-object grounding. https://arxiv.org/html/2310.11441 , https://arxiv.org/html/2404.13013
8. **Bootstrap 10^4–10^5 weakly labeled windows (EK100 narrations, point labels on own footage, negatives at ≥1:1) with a trimmed→untrimmed curriculum; down-weight JSON scaffolding tokens.** Medium impact, higher effort (annotation). https://ar5iv.labs.arxiv.org/html/2108.05029 , https://arxiv.org/html/2311.18445 , https://arxiv.org/abs/2306.14565 , https://arxiv.org/abs/2404.07965
9. **Runtime choice on the Mac:** llama.cpp (`ggml-org/gemma-4-E4B-it-GGUF` + mmproj, keep frames ≤1.15 Mpx while #28954 is open, `-DLLAMA_LLGUIDANCE=ON` for cheap per-request enums) or mlx-vlm 0.7.x (llguidance + `--adapter-path`, disable thinking with `json_schema` until #1294 is confirmed fixed). Avoid Outlines-backed paths for per-request schemas (3.7 s compile). Low effort, necessary but not sufficient.

## 7. Not verified / gaps
- Whether the 2026 EK100 leader (31.98 action mAP) is CausalTAD (inferred from identical numbers); 2023/2025 EK100 leaderboards (Codabench is JS-rendered).
- mlx-vlm #1294 fix status; mlx-vlm `--video` path for Gemma 4 specifically (documented for Qwen2-VL).
- Effect of Gemma's 262K vocabulary on grammar overhead (scaled from 128K measurements).
- Whether E2B/E4B are affected by llama.cpp #28954 (reported on 12B–31B QAT).
- Any conformal-prediction paper for temporal action localization or streaming video events (none found).
- Throughput of LoRA training on Gemma 4 E4B (no published number; costs in §5.3 are estimates from the stated anchors).
- Ego4D MQ instance counts and splits (only aggregate episodic-memory figures were retrievable); PDAN's own paper (cited through MS-TCT's table); LITA's trained components.
