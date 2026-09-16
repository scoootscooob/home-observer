<!-- research note produced 2026-09-16 by a web-research agent -->

# Streaming video-LLM architectures and training recipes, 2024–2026: how they decide when to speak, and a streaming-SFT recipe for a 4B student

Status: research note, 2026-09-16. Web budget: 30 fetches/searches. Builds on
`docs/research/decoding-and-training-2026-09.md` §2 and §4 (VideoLLM-online streaming EOS
loss, MMDuet heads, HawkEye negatives, Time-R1/VideoChat-R1 GRPO, calibration, duplicate
suppression) and `docs/research/perception-stack-2026-09.md` §6 (Apple-silicon throughput,
Qwen3-VL textual timestamps), and on the per-tick contract in `docs/teacher-distillation.md`
(`silent` / `context` / `act`, teacher-labelled at 2 Hz with a 4 Hz second look). Those facts
are not repeated here; this note adds what the streaming literature says about target
encoding, negative ratio, data volume, compute, latency and false-alarm measurement, and turns
it into a recipe for one unified streaming VLM (no detector cascade).

Conventions: **[P]** = primary source read during this pass (paper HTML, model card, repo);
**[S]** = secondary claim (a paper's description of another system, a leaderboard, a blog) or
recalled from training data and not re-verified this pass. Numbers carry units. Dollar
figures are estimates from quoted rental prices, marked as such.

## 0. Bottom line

1. **Every open streaming VLM that decides for itself when to speak uses one of three
   encodings**, and all three are in production-quality code: (a) a per-frame EOS/placeholder
   token in the text stream (VideoLLM-online's EOS, LiveCC's and StreamingVLM's "…", StreamPro's
   `</Silence>` / `</Response>`, MMDuet2's literal "NO REPLY"); (b) a binary head on a special
   token (MMDuet's two heads, Dispider's `<TODO>` head, Proact-VL's `<|FLAG|>` head,
   StreamBridge's separate 0.5B `<ACT>` scorer); (c) a heuristic gate outside the model
   (TimeChat-Online's token-drop ratio, Em-Garde's embedding-similarity jump). The owner's
   silence-token-or-burst contract is (a); it is the encoding used by the two 3–4B systems
   that work (StreamPro on Qwen3-VL-4B, MMDuet2 on Qwen2.5-VL-3B).
2. **The silence imbalance is solved by loss weighting, not by throwing away quiet data.**
   Published values: ProAssist keeps 10 % of silent frames in the loss (rho = 0.1; narration
   F1 30.1 → 58.7); StreamPro class-balances with beta = 0.9999 and doubles response-text
   weight; Proact-VL weights state transitions 5× and regularises the global speaking rate;
   StreamBridge marks only the last 0–50 % of an event's frames positive. Nobody reweights
   VideoLLM-online's EOS (w = 1.0) and it scores 3.92 % on StreamingBench proactive output.
3. **Data and compute are small.** Streaming behaviour was learned from 1,087 self-labelled
   trajectories (EvoStreaming, LoRA r = 8), 668 queries (Em-Garde), 3 K RL contexts
   (StreamPro), 1,900 RL videos (MMDuet2) and 30 K teacher-written dialogues over 479 h
   (ProAssist). SFT costs run 128 GPU-h (MMDuet2, 3B) to 200 H100-h (Proact-VL, 7B full);
   RL adds 160–192 GPU-h at 3–4B. A 4B student on 300 labelled hours is a 200–350 H100-hour,
   roughly $400–1,050 cycle at 2026 rental prices (§4.7, derived).
4. **RL is what fixes duplicates and timing.** MMDuet2's replication penalty drops the
   duplicate-reply ratio from 81–99 % to 1–15 %; Em-Garde's reward carries an explicit
   false-positive term; StreamPro's F1 reward has a time tolerance. All start from an SFT
   model that already knows silence, which is the owner's order of operations.
5. **No benchmark measures false alarms per quiet hour.** StreamingBench PO and OVO-Bench
   FAR are recall-style; the 2026 proactive papers report trigger F1, TimeDiff, PAUC and
   duplicate ratio, but nobody reports events per hour on quiet footage, and nobody trains or
   evaluates on fixed home cameras. The owner's quiet-hours protocol has no published
   counterpart and has to be built (decoding note §4.2).
6. **Same-model second look exists once**: Eyes Wide Open adds an `ask_high` action
   (request high-resolution frames) to silence/response on an 8B egocentric streamer. It is
   the training precedent for a `look` token that re-encodes the last 2 s at 4 Hz (§5).
7. **Base model**: Qwen3-VL-4B-Instruct for both machines today (measured MLX throughput,
   textual timestamps, StreamPro precedent at 4B, vLLM/AWQ for the box); Qwen3.5-4B is the
   likely successor because its 3:1 Gated-DeltaNet layout keeps the KV cache small for
   hours-long streams, but its Apple toolchain and video defaults are unverified. Gemma 4
   E4B only if native audio in one model outweighs the missing timestamp support and the
   8B-parameter memory footprint on the box (§6).

## 1. Comparison table (deliverable a)

Abbreviations: SB = StreamingBench (overall / PO = proactive output, RTVU = real-time visual
understanding); OVO = OVO-Bench (RT = real-time, FAR = forward active responding); n/s = not
stated in the source read; [S] = secondary. "Open" lists what the source itself claims.

| System (date) | Base (size) | Open | Silence / respond encoding | Negatives and loss weighting | Data | Trained parts; fps; tokens per frame | Compute | SB / OVO (egocentric or fixed-camera) | Latency | False alarms measured? |
|---|---|---|---|---|---|---|---|---|---|---|
| VideoLLM-online (CVPR 2024) | Llama-2-7B / Llama-3-8B + CLIP-L frozen | code, weights, data | EOS target on every non-response frame token; text at response frames; accept EOS iff P ≥ 0.5–0.8 | all silent frames supervised, w = 1.0, no reweighting | Ego4D narration 5-min clips, COIN, GoalStep (counts n/s) | LoRA r128 α256 all linear + MLP; 2 fps; 1 (demo 10) tok/frame; 4 K ctx | 2 epochs; GPUs n/s | SB 32.48 / PO 3.92; OVO BT 17.73, RT 20.79; OVBench 9.6; Ego4D TimeDiff 2.32 s | > 10 fps (13.5) on A100, < 20 GB | no (TimeDiff, Fluency) |
| MMDuet (Nov 2024) | LLaVA-OV-7B | code, weights, data [S] | two per-frame binary heads (informative, relevance); fire on cumulative score ≥ 2 or sum ≥ 0.3–0.6; drop own previous reply from KV | negatives = frames outside last 50 % of each event; no weights | MMDuetIT 109 K | LoRA r16 α32 + projector + heads; 0.5–2 fps; 49 tok/frame; 120 / 400 frames | 8×V100 × 1 day, 1 epoch, LR 2e-5 | SB PO 29.44 [S]; ESTP-F1 17.8 [S] (Ego4D) | 1.9–2.9× baseline (V100) | no |
| MMDuet2 (Dec 2025) | Qwen2.5-VL-3B | code page (unconfirmed) | literal "NO REPLY" or reply text per 2 s turn (1 s at inference) | SFT then GRPO: 3×PAUC + 2×replication + 0.5×in-span + 2×prefix rewards | SFT ~50 K videos + 50 K offline; RL 1,500 web + 400 ego videos | n/s; 1–2 frames/turn; 128 tok/frame | 16 H800 × 8 h + 8 H800 × 20 h | SB PO 34.69; PAUC WEB 53.3, EGO 33.6 | +17 % wall time vs MMDuet | yes: duplicate ratio 1–15 %, reply turns 3.3 vs 5.7 |
| Dispider (CVPR 2025) | Qwen2-1.5B decider + Qwen2-7B responder + CLIP-L | code, weights | BCE head on `<TODO>` per scene clip; responder can still emit `<SILENT>` | positive + negative clips, ratio n/s | S1 GroundVQA + ET-Instruct + 50 K; S2 122 K | n/s; 1 fps | n/s | SB 53.12 / PO 25.34; ET-Bench DVC F1 33.8; EgoSchema 55.6 | n/s | no (ablation only) |
| Flash-VStream (ICCV 2025) | 7B (Qwen-based) | code, weights | none: query-driven STAR memory | — | n/s | n/s | n/s | SB 24.04 / PO 1.96 (2024 ver.); OVO 33.61; OVBench 31.2 | ~4 s per reply at 64 frames [S] | no |
| VideoChat-Online (Jan 2025) | InternViT-300M + Phi-3 (4B) | models, data | none: query-driven; 3-tier memory 8/2/1 fps at 256/64/16 tok, 832 tokens | — | 96 K online IT + offline mix | projector + LLM; 1 fps train, ≤ 64 frames | LR 1e-4, batch 1024, 1 + 1 epochs | OVBench 54.9 | n/s | no |
| TimeChat-Online (MM 2025) | Qwen2.5-VL-7B, ViT frozen | code, weights, data [S] | external gate: reply when token-drop ratio < 60 % (scene change); "unanswerable" text | 20 K unanswerable of 139 K | 139 K QA, 11,043 videos, ~2,050 h + 203 K mix | LLM + proj; 1 fps; ≤ 64 frames; 448² | 8×A800, 1 epoch, batch 128, LR 1e-5 | SB RTVU 75.36; OVO 47.6 | 1.8–3.2 s per reply | no |
| LiveCC (CVPR 2025) | Qwen2-VL-7B-Base | code, weights, data | "…" per-frame EOS; silent frames predict "…" only | none stated (natural ASR density) | Live-CC-5M (5.7 M videos) + Live-WhisperX-526K | full; 2 fps; ≤ 480 frames; 24 K ctx | 128 GPUs, batch 512 | OVO RT 59.8; LiveSports-3K 41.5 % win vs GPT-4o | 0.17 s per frame | no |
| StreamingVLM (Oct 2025) | Qwen2.5-VL-7B-Instruct | code, data (weights [S]) | "…" placeholder per silent second; loss on per-second text | none stated | 525 K + 526 K + 14.8 K anneal; > 6,000 h sports | full; 24 s chunks / 12 s overlap; sinks 512 + text 512 + 16 s vision window | ~128 H100-days | OVO RT 61.96; Inf-Streams-Eval 66.18 % win vs GPT-4o mini | 8 fps on H100; > 2 h stable | no |
| ProVideLLM (Apr 2025) | Llama-3.2-1B / 3.1-8B + DINOv2 | page only | per-frame step class incl. background; verbalized long-term memory | plain CE | EK-100 pretrain; GoalStep, EgoExo4D, Assembly101, COIN | LoRA r128 α256 + connector; 4 fps; 5 / 11 tok/frame | n/s | Ego4D GoalStep online mAP 13.0 (egocentric) | 10 fps in 2 GB; 24.6 fps dialogue (A6000) | no |
| StreamBridge (May 2025, v2 Sep 2025) | Qwen2-VL / LLaVA-OV / Oryx-1.5 7B + LLaVA-OV-0.5B activator | n/s | separate 0.5B per-frame score on `<ACT>`, threshold 0.35, parallel thread | positives = last P % of event frames, P ~ U(0, 50 %); 180 K activator samples | Stream-IT ~265 K + 600 K offline (GPT-4o QA) | proj + LLM full; 1 fps; ≤ 256 frames; activator 16 tok/frame | 1 epoch LR 2e-5; activator 5 epochs LoRA LR 2e-4 | SB RTVU 77.04; OVO RT 71.30; ET-Bench DVC Sim 25.1 | near-constant with 16,384-token cap | partial (threshold sweep) |
| ProAssist (EMNLP 2025) | LLaMA-3.1-8B + SigLIP-SO400M | "upon publication" | EOS at last visual token per frame | negative-frame sub-sampling rho = 0.1 (F1 30.1 → 58.7) | 478.7 h, 30,135 dialogues; LLaMA-3.1-70B teacher; 2:4:4 user types | VideoLLM-online recipe; 1 / 5 / 10 tok/frame | n/s | dialogue F1 36.07 (Ego4D, EK-100, HoloAssist, Assembly101) | n/s | no (F1 only) |
| Eyes Wide Open (Oct 2025) | LLaMA-3 + SigLIP (8B) | page only | 3 actions: silence / response / `ask_high` | weighted loss (values n/s) | 60 K + 20 K questions on Ego4D | LoRA; 3 SFT stages | n/s | ESTP-F1 34.7 (Ego4D); OVO 24.16 | n/s | yes (FP in ESTP-F1) |
| Proact-VL (Mar 2026) | LiveCC-7B-Base | data "released"; code n/s | response head on `<|FLAG|>` per 1 s chunk, threshold 0.3 | head loss α = 0.2; transition weight γ = 5; smoothness + speaking-rate regulariser | 561 h gaming + LiveCC + Ego4D; 128 K samples | full; 2 fps; ≤ ~540 tok/frame | ~200 H100-h; 2,000 steps × batch 64 | in-domain trigger F1 63–77, TimeDiff 0.7–1.2 s | 0.36 s per 1 s chunk | yes (F1, TimeDiff, PAUC, sweep) |
| Em-Garde (Mar 2026) | Qwen2.5-VL-7B parser + Ops-MM-V1-2B matcher | code | similarity jump > 0.04 on 2 s segments | GRPO reward with FP penalty | Parse2Prop-1K: 668 queries (GPT-5 / human) | parser SFT 1 h + GRPO 5 h | 8×A100 × 6 h | SB PO 38.0; OVO FAR online F1 30.99 | 10–15 fps on A100 | yes (precision / recall at 2 s) |
| StreamPro (May 2026) | Qwen3-VL-4B / Qwen2.5-VL-3B | n/s | `</Silence>` or `</Response>` + text per step | CB loss β = 0.9999, λ_text = 2; GRPO F1 + rubric | SFT 63 K + 224 K; RL 3 K | proj + LLM; 1 fps | 64 H100 × 24 h + 8 H100 × 24 h | SB RTVU 78.9; OVO FAR 20.6, RT 57.6 | n/s (200-turn window) | yes (F1 with tolerance) |
| EvoStreaming (May 2026) | 8B offline VLMs (Qwen2/2.5/3-VL, InternVL-3.5, MiniCPM-V 4.5) | code | self-labelled silent / respond via sliding roll-out | multiplicative verbosity penalty | 1,087 self trajectories | LoRA r8 α32; 0.5 fps; 128 tok/frame | 8×H200, 1 epoch per iteration | RealStreamEval 54.6; OVO +0.6 | n/s | partial (tokens per turn) |
| MiniCPM-o 4.5 (2026) | Qwen3-8B + SigLIP2 + Whisper-medium (9B) | weights, code (Apache-2.0) | TDM timeline; 1 Hz proactive decision; duplex speech | undisclosed | undisclosed | ≤ 10 fps | undisclosed | Video-MME 70.4; LiveSports duplex 54.4 % | TTFT 0.6 s; int4 11 GB | no |
| Qwen3-VL-4B / Qwen3.5-4B | dense | weights (Apache-2.0) | none native; textual timestamps (3-VL) | — | — | — | — | Charades-STA mIoU 55.5 (3-VL-4B) | 143 tok/s MLX on M4 Max [perception note] | — |
| Gemma 4 E4B | 4.5B effective / 8B total | weights (Apache-2.0) | none; frames-as-images ≤ 60 s at 1 fps; audio ≤ 30 s | — | — | — | — | none published | — | — |

## 2. Per-system notes: primary-source facts, then secondary claims

### 2.1 VideoLLM-online (CVPR 2024)

[P] https://arxiv.org/html/2406.11816 (read 2026-09-16). Facts not already in the decoding note:
- **Data.** Ego4D narration streams cut into 5-minute clips at 2 fps (about 600 frames per
  clip); COIN and Ego4D GoalStep converted to streaming dialogue by inserting templated
  questions (150 templates: 50 past, 50 present, 50 future tense) and letting an LLM answer
  from the timestamped annotations; at most 3 inserted queries per training sample. No hour
  count or sample count is stated in the HTML.
- **Target encoding.** For every frame between a response's trigger time t1 and the next
  response time t2 the target on the frame token is EOS (`max P(EOS | ctx, frame)`); at
  response frames the target is the text. Loss weight w = 1.0. Inference accepts EOS only
  when P(EOS) >= theta; theta in 0.5–0.8 "yields much better results". The EOS-to-text token
  ratio is never stated; with 1 token per frame at 2 fps and Ego4D narrations every ~3–5 s,
  most frame positions carry an EOS target [S, inferred].
- **Model.** Llama-2-7B-Chat or Llama-3-8B-Instruct; LoRA r = 128, alpha = 256 on every
  linear layer of the LLM; CLIP ViT-L frozen; 2-layer MLP projector; 1 CLS token per frame
  (10 tokens per frame in the demo variant); 4096-token context covers a half-hour stream.
- **Compute.** 2 epochs for the streaming task (5–6 epochs for offline benchmarks); GPU
  count, hours, batch size and LR not given in the HTML.
- **Latency.** Over 10 fps on one A100 for a 5-minute stream with under 20 GB memory;
  13.5 fps with the streaming formulation vs 7.5 fps for per-frame dialogue.
- **Metrics.** LM-PPL, TimeDiff (s), Fluency, LG-Match. No false-alarm or over-trigger rate.
- **Open.** Code, weights, data and demo (showlab.github.io/videollm-online).

### 2.2 MMDuet (2024)

[P] https://arxiv.org/html/2411.17991 (read 2026-09-16).
- **Model.** LLaVA-OneVision-7B; trained parts are LoRA r = 16 (alpha = 32) on the LLM, the
  projector, and two binary heads (informative, relevance); everything else frozen. 49 tokens
  per frame (7×7 after 4× pooling); up to 120 frames in training, 400 at inference; 2 fps on
  Shot2Story, 0.5 fps on COIN.
- **Target encoding.** The informative head is labelled TRUE for frames "between 50 % of
  this segment and the insertion point of the response", FALSE elsewhere (so negatives are
  every other frame, no re-weighting stated); the relevance head is a per-frame query
  relevance classifier; the LM loss is only on response turns. Trigger: cumulative
  informative score >= s (s = 2 for dense captioning) or informative + relevance >= t
  (t in 0.3–0.6 for MAGQA). Duplicate suppression: previous responses are dropped from the KV
  cache ("rm. prev. resp.") so the model does not copy its own last line.
- **Data.** MMDuetIT, 109 K examples: Shot2Story 43 K dense-caption + 36.8 K MAGQA, COIN
  dense captioning, DiDeMo/HiREST/QueryD for grounding. Hours not stated.
- **Compute.** One epoch, about one day on 8×V100 (32 GB), LR 2e-5, batch 1 × grad-acc 8.
- **Results.** YouCook2 dense captioning CIDEr improves with response removal; Charades-STA
  R@0.5 42.4 / R@0.7 18.0 under its streaming protocol; StreamingBench proactive-output
  score for MMDuet is reported by later papers at 29.44 % [S]. Inference on one V100 is
  1.9–2.9× the baseline cost depending on threshold. No false-trigger rate is reported.
- **Open.** Code, weights and MMDuetIT are on GitHub/HF (yellow-binary-tree/MMDuet) [S, not
  re-fetched this pass].

### 2.3 Dispider (CVPR 2025)

[P] https://arxiv.org/html/2501.03218 (read 2026-09-16).
- **Architecture.** Three disentangled parts: scene-based perception (CLIP-L/14 frame
  features, SigLIP cosine-similarity scene boundaries), a real-time decision module on a
  Qwen2-1.5B LLM with `<TODO>` / `<ANS>` tokens, and an asynchronous reaction module on
  Qwen2-7B that can still emit `<SILENT>`. 1 fps.
- **Target encoding.** A binary classification head on the final-layer embedding of `<TODO>`
  predicts "respond now" at each clip boundary with BCE; grounding uses a KL loss against a
  time distribution; training mixes "positive (response-required) and negative
  (no-response-required) samples" (ratio not stated). `<SILENT>` in the 7B model is a
  second filter: "if the preceding streaming processor incorrectly identifies a timestamp as
  requiring a response, the `<SILENT>` token enables the LLM to rethink".
- **Data.** Stage 1: GroundVQA + ET-Instruct with time labels + 50 K implicit QA pairs;
  stage 2: 122 K streaming QA pairs (ET-Instruct timestamps, VideoChatGPT, LLaVA-Next-Video).
- **Compute.** Not stated in the HTML.
- **Results.** StreamingBench overall 53.12 (real-time visual 67.63, omni-source 35.66,
  contextual 33.61, proactive output 25.34); ET-Bench streaming TVG F1 36.1, DVC F1 33.8,
  DVC Sim 18.9; EgoSchema 55.6, MLVU 61.7, VideoMME 57.2. Latency not quantified. No false
  positive rate reported; the `<SILENT>` ablation is the only over-trigger evidence.
- **Open.** Code and model at github.com/Mark12Ding/Dispider.

### 2.4 Flash-VStream (2024 / ICCV 2025)

[P] https://github.com/IVGSZ/Flash-VStream (read 2026-09-16): the repo now carries the ICCV
2025 version (arXiv 2506.23825) with a Qwen-based 7B variant on Hugging Face
(Flash-VStream-Qwen) and the earlier LLaVA-based variant; evaluated on EgoSchema, MLVU,
LVBench, MVBench, Video-MME. The landing page gives no memory sizes, data counts, compute or
latency. [S, from the earlier 2024 paper as summarised in the perception note] STAR memory
(spatial, temporal, abstract, retrieved) with an asynchronous frame-encoding process; it is
query-driven: no silence token, no proactive trigger, no false-alarm metric. Its OVO-Bench
score is 33.2 overall per TimeChat-Online's comparison (47.6 vs "14.4 points over
Flash-VStream") [S]. Not re-fetched; §7.

### 2.5 VideoChat-Online (2025)

[P] https://arxiv.org/html/2501.00584 (OVBench + VideoChat-Online, read 2026-09-16).
- **Model.** 4B: InternViT-300M + Phi-3; MLP projector and LLM trained; training input at
  1 fps, max 64 frames. Hierarchical memory bank with three tiers sampled at 8 / 2 / 1 fps
  and 256 / 64 / 16 tokens per frame, capacities 12 / 2 / 2 frames, 832 tokens total at
  inference; eviction merges the most similar adjacent pair and average-pools the older one.
- **Data and compute.** 96 K online instruction samples from 5 tasks across 12 datasets,
  after an offline stage (VideoChat2-IT, STAR, PerceptionTest, ShareGPT4V/4o,
  LLaVA-OneVision); LR 1e-4, 1 epoch offline + 1 epoch joint, batch 1024; GPUs and hours
  not stated.
- **When to speak.** Query-driven only; no silence token or proactive trigger is described.
- **OVBench.** 6 task types, 16 subtasks, 7,000 annotations, evaluated in a 32 s / 2 fps
  sliding window and in full streaming. Results: VideoChat-Online-4B 54.9 %, Qwen2-VL-7B
  49.7 %, Flash-VStream-7B 31.2 %, VideoLLM-online-7B 9.6 %. No StreamingBench/OVO-Bench
  numbers; no latency figure beyond "designed for mobile deployment".
- **Open.** "All the models and data are publicly available" (videochat-online.github.io).
- Relevance: the only 4B streaming model with released online-IT data; its 16-tokens-per-frame
  spatial tier is a measured operating point for cheap long-horizon memory.

### 2.6 TimeChat-Online (ACM MM 2025)

[P] https://arxiv.org/html/2504.17343 (read 2026-09-16).
- **Trigger = change gate.** Differential Token Drop compares consecutive patches at pixel
  level (L1 threshold) and feature level (cosine threshold tau_feat); tau_feat = 0.25 drops
  about 85 % of video tokens, tau_feat = 0.5 about 45–55 %. Frames with a low drop ratio are
  scene transitions; the proactive mode generates a response at every frame whose drop ratio
  falls below a hard 60 % threshold, using the most recent content, and the model is trained
  to answer "unanswerable" (wait for the next transition) when the question cannot be
  answered yet. This is the closest published analogue of the change-gated token stream in
  the owner's direction, but the gate is a heuristic outside the model, not a learned silence.
- **Data.** TimeChat-Online-139K: 139 K QA pairs on 11,043 videos averaging 11.1 min
  (about 2,050 h), four task types, 20 K "unanswerable" negatives for forward responding;
  12 sources (COIN, ActivityNet, YouCook2, QVHighlights, HD-VILA, YouMakeup, VideoIC,
  Movie101, TVSum, ViTT, QuerYD, HiREST). Mixed with LLaVA-Video-178K (100 K), Tarsier2
  (100 K), VideoChat-Flash (3 K).
- **Model and compute.** Qwen2.5-VL-7B, vision encoder frozen; 1 fps, max 64 frames per
  streaming sample, 448×448; batch 128, LR 1e-5, 1 epoch, 8×A800-80G; DTD applied to 50 % of
  training batches.
- **Results.** StreamingBench real-time 75.36 (73.64 at 82.6 % drop); OVO-Bench 47.6;
  VideoMME 63.3; MLVU 65.4; LongVideoBench 57.7. Latency 3,220 ms → 1,820 ms at 81.1 % drop
  ("maximum latency of 2 seconds for responding" with 6 K kept tokens). No false-trigger rate.
- **Open.** Project page timechat-online.github.io; code, weights and dataset on GitHub/HF
  [S, not re-fetched].

### 2.7 LiveCC (CVPR 2025)

[P] https://arxiv.org/html/2504.16030 (read 2026-09-16).
- **Target encoding.** ASR words are assigned to the frame interval in which they were
  spoken and interleaved after each frame's visual tokens; an ellipsis token "…" is appended
  to every frame's text as the per-frame EOS, and silent frames predict only "…". This is a
  learned per-frame silence token trained at scale, the direct precedent for a silence token
  in the owner's contract.
- **Data.** Live-CC-5M pre-training: 5.7 M YouTube videos with closed captions, 2 fps;
  Live-WhisperX-526K SFT: 526 K clips with WhisperX word timestamps, up to 240 s each.
- **Model and compute.** Qwen2-VL-7B-Base, full training, batch 512 on 128 GPUs, LR 2e-5
  (pre-train) and 1e-5 (SFT), 24 K visual-context tokens, up to 480 frames.
- **Results and latency.** LiveSports-3K-CC win rate vs GPT-4o 41.5 % (Instruct) / 43.2 %
  (Base); VideoMME 64.1 (70.3 with subtitles); OVO-Bench RTVP 59.8. Response latency 0.17 s
  per frame at 2 fps, "only a few words per frame". Over-commentary is not measured.
- **Open.** Data (Live-CC-5M, Live-WhisperX-526K, LiveSports-3K), weights (7B-Base,
  7B-Instruct) and code at showlab.github.io/livecc.

### 2.8 StreamingVLM (MIT Han Lab, Oct 2025)

[P] https://arxiv.org/html/2510.09608 (read 2026-09-16).
- **Inference.** KV cache = 512 attention-sink text tokens + a 512-token recent-text window
  + a short vision window of about 16 s of frames (the fetch rendered the frame count
  ambiguously; treat "16 s" as the reliable figure); RoPE indices are re-numbered so kept
  tokens stay contiguous after eviction. Input 360p–720p video; up to 8 fps sustained on one
  H100; stable for more than 2 h of continuous commentary (evaluation games average 2.12 h).
- **Training = overlapped-chunk SFT.** Chunks of W = 24 s with O = 12 s overlap, full
  attention within a chunk, loss only on the text positions aligned to per-second narration;
  a second with no narration gets a placeholder "…" token, which is how the model learns
  "when to speak and when to remain silent" (same convention as LiveCC).
- **Data and compute.** Inf-Streams-Train 525 K streaming samples plus LiveCC's
  Live-WhisperX-526K, then a 14,786-sample annealing set; raw source over 6,000 h of sports
  broadcasts (712 basketball, 544 soccer, 402 ice hockey, 399 baseball, 392 American
  football games). Base Qwen2.5-VL-7B-Instruct; about 128 H100-days.
- **Results.** Inf-Streams-Eval 66.18 % win rate vs GPT-4o mini; LongVideoBench 54.70 →
  59.00; OVO-Bench real-time 56.00 → 61.96; LiveSports-3K-CC 87.81 % win vs LiveCC;
  VideoMME 65.10; MVBench 69.16. No over-trigger or false-alarm rate.
- **Open.** Code (github.com/mit-han-lab/streaming-vlm), Inf-Streams-Train and
  Inf-Streams-Eval released.

### 2.9 ProVideLLM (2025)

[P] https://arxiv.org/abs/2504.13915 (abstract only, read 2026-09-16): "Memory-efficient
Streaming VideoLLMs for Real-time Procedural Video Understanding" (Chatterjee et al.,
April 2025). Multimodal interleaved cache of verbalized text tokens for past observations
plus DETR-QFormer visual tokens for recent frames; 22× token reduction for one hour of
observations; per-frame streaming inference at 10 fps and streaming dialogue at 25 fps in a
2 GB GPU footprint; state of the art on six procedural tasks over four datasets. Full-text
details (base LLM, data, decision rule, per-dataset numbers) are filled in below if the HTML
fetch succeeds; otherwise see §7.

[P] https://arxiv.org/html/2504.13915 (full text read 2026-09-16).
- **Model.** Llama-3.2-1B-Instruct (ProVideLLM-1B/5: DINOv2 ViT-S, 5 tokens per frame) and
  Llama-3.1-8B-Instruct (ProVideLLM-8B/11: DINOv2 ViT-L, 11 tokens per frame) behind a
  DETR-QFormer connector. Long-term memory is verbalized: past steps are stored as text,
  about 630 text tokens for one hour of video; short-term cache Ns = 64 frames, long-term
  cache Nl = 5 observed steps. 4 fps for online step detection.
- **When to speak.** Per-frame online step detection is a classification over step classes
  including background, trained with plain cross-entropy; no explicit silence token, no
  stated background ratio or loss weighting.
- **Training.** Stage 1 pre-trains the DETR-QFormer on EPIC-KITCHENS; stage 2 LoRA r = 128,
  alpha = 256 on the decoder plus the connector; vision encoder frozen in both stages.
  Datasets: Ego4D Goal-Step, EgoExo4D, Assembly101, COIN. Epochs, LR, GPUs not stated.
- **Results.** Ego4D GoalStep online step detection per-frame mAP 13.0 (val) / 12.9 (test);
  COIN step forecasting top-1 53.6 %, step recognition 67.3 %; EgoExo4D step recognition
  44.36 % / 50.74 %; Assembly101 anticipation top-5 recall 13.8 %.
- **Latency.** 10 fps per-frame streaming in 2 GB GPU memory; dialogue 24.6 fps on an
  A6000; vision encoder 74.8 fps, LLM 10.4 fps (the bottleneck).
- **Open.** Project page dibschat.github.io/ProVideLLM; code/weights not confirmed (§7).
- Relevance: the per-frame mAP of 13 on Ego4D GoalStep is the honest ceiling for per-frame
  step detection from a 1B–8B streaming LLM on egocentric footage; it argues for coarse,
  state-change targets rather than per-frame action classes.

### 2.10 StreamBridge (2025)

[P] https://arxiv.org/html/2505.05467 (v2, Sept 2025; read 2026-09-16).
- **Decoupled proactive activation.** A separate LLaVA-OV-0.5B with 16 tokens per frame
  and a score head replacing the LM head, fed a learnable `<ACT>` token per frame, predicts
  "respond now" per frame. Targets: only the last P % of frames of each event segment are
  positive, with P sampled uniformly in 0–50 % per sample (so positives are a minority and
  sit at the end of the evidence, exactly the "emit once, when established" rule of the
  owner's contract); about 180 K samples from five tasks (dense captioning, sequential
  steps, temporal action detection, grounded QA, temporal grounding); 5 epochs of LoRA at LR
  2e-4. Inference threshold alpha = 0.35 and the activation model runs in a parallel thread.
- **Main model.** LLaVA-OV-7B, Oryx-1.5-7B, Qwen2-VL-7B; image encoder frozen, projector and
  LLM fully trained; 1 epoch, LR 2e-5, AdamW; 1 fps, videos over 256 s sampled to 256
  frames; memory buffer capped at 16,384 embeddings with round-decayed average pooling of the
  oldest turns (near-constant latency beyond the cap).
- **Stream-IT.** About 265 K samples: 54 K dense captioning + 22 K sequential steps + 69 K
  grounded QA + about 120 K StreamingQA-120K (GPT-4o-generated over 1.28 M clips filtered
  from WebVid-10M / Panda-70M / InternVid-10M); 80 % multi-turn dialogue, 20 % proactive
  format; augmentation: random QA drop p = 0.55, QA interval shift p = 0.1; mixed with about
  600 K offline samples (LLaVA-178K, VCG-Plus, ShareGPT4Video).
- **Results.** OVO-Bench real-time average 71.30 (Qwen2-VL + Stream-IT) vs GPT-4o 64.46;
  StreamingBench real-time average 77.04 vs Gemini 1.5 Pro 75.69; ET-Bench streaming DVC Sim
  25.1, SLC Sim 17.1 (vs Dispider 18.9 / 12.4). The only over-trigger evidence is the alpha
  sweep: F1 falls at both low and high alpha. No standalone false-positive rate.
- **Open.** Not stated in the HTML; treat code/weights/data as unverified (§7).

### 2.11 Proactive / "when to respond" work, 2025–2026

[S] Search results 2026-09-16 identify the 2026 line of work: Em-Garde (arXiv 2603.19054,
propose-match framework for proactive streaming), Proact-VL (2603.03447, proactive VideoLLM
for real-time companions), Response-G1 (2605.07575, scene-graph alignment between evidence
and the query's response conditions), StreamPro (2605.16381, reactive perception to proactive
decision-making), "Proactive Assistant Dialogue Generation from Streaming Egocentric Videos"
(2506.05904), and a curated list at github.com/sotayang/Awesome-Streaming-Video-Understanding.
Two target encodings are described in those papers' related work: a **state-token unified
trigger** with explicit Silence / Standby / Response state tokens in one autoregressive
sequence and a **focal-weighted loss** for the extreme state imbalance; and **MMDuet2**, which
makes the per-turn decision a text output ("NO REPLY" or a response) trained with multi-turn
RL under a PAUC-inspired reward that favours early correct responses without reply-time
labels. Primary-source details follow in the fetch notes below.

**Proact-VL** [P] https://arxiv.org/html/2603.03447 (read 2026-09-16).
- **Decision.** A response head on the hidden state of a `<|FLAG|>` token appended to each
  1-second chunk gives p_t; speak iff p_t >= tau (tau = 0.3 main, 0.5 in ablations). 2 fps,
  per-frame pixel budget min(540×28×28, 36×540×28×28 / T), i.e. at most about 540 visual
  tokens per frame and about 19 K video tokens per sample.
- **Loss.** Causal LM loss on assistant tokens only, plus the response-head loss with weight
  alpha = 0.2, with **transition-aware weighting gamma = 5** (a chunk where the state flips
  silence→speak or speak→silence weighs 5× a persistence chunk; transitions are about 1:5 of
  chunks) and a regulariser for temporal smoothness plus a global speaking-rate term that
  matches the predicted rate to the labelled rate. This is the first published fix for the
  problem in roadmap F2 (silence dominates tokens; positives dominate loss).
- **Data.** Live Gaming Dataset, 561 h from 12 game titles plus LiveCC and Ego4D clips;
  128 K training samples; labels from ASR alignment turned into second-level captions;
  evaluation clips stratified by response rate (60 clips at 0–30 %, 120 at 30–70 %, 60 at
  70–100 % of seconds spoken).
- **Model and compute.** Full fine-tune of LiveCC-7B-Base (also tried on Qwen2-VL,
  Qwen2.5-VL, Qwen3-VL); LR 1e-5 cosine, batch 64, 2,000 steps, about **200 H100 GPU-hours**.
- **Results and over-trigger metrics.** In-domain solo commentary CC 53.62 / trigger F1
  63.25 / TimeDiff 1.20 s; co-commentary F1 77.44 / TimeDiff 0.71 s; guidance F1 53.91 /
  TimeDiff 3.21 s; out-of-domain (Black Myth: Wukong) F1 60.06 / TimeDiff 0.90 s. F1 is a
  precision/recall of trigger events, so false alarms lower it; a threshold sweep shows
  "severe over-triggering" at tau = 0.1. PAUC is also reported. No StreamingBench/OVO-Bench.
- **Latency.** About 0.36 s per 1-second chunk (cache + forward + generation); "10–15 fps"
  expected. Release: homepage proact-vl.github.io, dataset "released", code not confirmed.

**Em-Garde** [P] https://arxiv.org/html/2603.19054v2 (read 2026-09-16).
- **Method.** Not a token stream: an instruction-guided proposal parser (Qwen2.5-VL-7B,
  invoked once per query with 5 s of history at 1 fps) writes k declarative visual cues
  (avg 3.99 per query, 13.5 words each); a frozen 2B embedding model (Ops-MM-V1-2B) scores
  2-second segments at 2 fps against the cues, and a trigger fires when any cue's similarity
  jumps by more than theta = 0.04 between consecutive steps.
- **Training.** SFT (about 1 h) then GRPO (about 5 h) on 8×A100 with Parse2Prop-1K: 668
  queries on 92 videos (COIN, Ego4D, BEHAVIOR), half with human- or **GPT-5-authored**
  proposal annotations; ground-truth response times with 4 s tolerance. Reward
  r = (1 − lambda·r_fp) · n_c / n with r_fp = 1 − 2^(−n_fp/n), lambda = 1, i.e. an explicit
  false-positive penalty.
- **Results.** StreamingBench PO 38.0 at 2 fps; OVO-Bench FAR F1 30.99 (CRR 26.40, SSR
  23.40, REC 43.16) under a new *online* protocol where a response counts only inside a 2 s
  tolerance window of a ground-truth time and precision/recall are reported (recall about
  50 %, precision lower). 10–15 fps on A100 with constant latency. Code at
  github.com/air-embodied-brain/Em-Garde; data and weights not confirmed.

**StreamPro** [P] https://arxiv.org/html/2605.16381 (read 2026-09-16).
- **Target encoding.** One decision token per step from {`</Silence>`, `</Response>`};
  `</Response>` is followed by the text. End-to-end in the LM, no gating module. 1 fps.
- **Loss.** "CB-Stream" class-balanced loss: each token class is reweighted by the effective
  number E_k = (1 − beta^{n_k}) / (1 − beta) with **beta = 0.9999**, plus **lambda_text = 2**
  on response text tokens. This is the class-balanced (Cui et al. 2019) answer to the
  silence/response imbalance.
- **Data and compute.** SFT on StreamPro-SFT-63K plus TimeChat-Online-139K,
  VideoChat-Flash-3K and Streamo-Instruct-465K (287 K after filtering); projector + LLM for 1
  epoch on **64 H100 for 24 h** (about 1,536 H100-hours), LR 1e-5, batch 512. RL:
  StreamPro-RL-3K with GRPO, G = 8 samples at temperature 1.0, **8 H100 for 24 h**, LR 1e-6,
  1 epoch. Reward = 0.1 format + 0.45 turn-level F1 (S = S_time + S_acc,
  S_time = max(0, 1 − |t_pred − t_gt| / tau), tau = 8 s in RL) + 0.45 trajectory rubric
  judged by an LLM against checkpoints. **Backbones: Qwen2.5-VL-3B and Qwen3-VL-4B**, the
  only 3–4B end-to-end proactive streamers found.
- **Results.** StreamPro-Bench (577 videos, 1,285 QA) trajectory F1 41.5 for
  StreamPro-GRPO-4B vs 10.4 prior best (PU 66.0, TR 61.6, proactive agency 45.0);
  StreamingBench RTVU 78.9; OVO-Bench FAR 20.6 (prior best Streamo 5.4); OVO-Bench
  real-time (RTVP+BT) 57.6. Precision and recall in the F1 penalise excessive and missed
  responses; tolerance windows are task-specific (e.g. anomaly alert [0, +5 s], object
  ±3 s). Inference uses a sliding window of 200 dialogue turns; no per-frame latency.
  Release not stated.

**Eyes Wide Open (VideoLLM-EyeWO)** [P] https://arxiv.org/html/2510.14560v1 (read
2026-09-16). Ego streaming proactive task on Ego4D; ESTP-Bench 2,264 questions, 14 task
types, 890 videos, average 3.96 valid answer intervals per question; ESTP-F1 =
2·ΣS / (2·ΣS + FP + FN) with S combining answer correctness and timeliness, so false
positives are counted directly. Model: LLaMA-3 + SigLIP (VideoLLM-online lineage), LoRA;
**three actions per step: silence, response, and `ask_high` (request high-resolution
frames)**; proactive dynamic compression to about one tenth of tokens. Data engine: 60 K
single-turn and 20 K multi-turn questions generated by LVLM captioning + RAG
(one-to-one, one-to-many, many-to-many), teacher LLM not named. Three SFT stages (passive
interval responsiveness with weighted loss; proactive just-in-time with `ask_high`;
multi-turn); no RL. ESTP-F1: EyeWO 34.7 vs VideoLLM-online 15.5 (threshold 0.9), MMDuet
17.8, best polled offline model (MiniCPM-V) 22.9; OVO-Bench 24.16 vs 8.05 baseline. No GPU
hours, no latency; release via project page only.

**MMDuet2** [P] https://arxiv.org/html/2512.06810 (read 2026-09-16).
- **Decision as text.** Every user turn (one turn per 2 s in SFT, per 1 s at inference, 1–2
  frames per turn, 128 tokens per frame) the model writes either "NO REPLY" or a reply.
  Base Qwen2.5-VL-3B. No head, no special token: the decision is ordinary text, which is
  exactly the owner's "silence token or burst" contract expressed in words.
- **Recipe.** SFT on about 50 K videos + 25 K offline QA + 25 K captioning samples, 16 H800
  for about 8 h (about 128 GPU-h); then multi-turn GRPO with 4 rollouts on 1,500 web + 400
  egocentric videos, 8 H800 for about 20 h (about 160 GPU-h). Reward = 3 × PAUC-style
  timing/correctness score (max 4) + 2 × replication penalty (duplicate information) +
  0.5 × in-span reward (off-topic replies) + 2 × prefix reward (verbose repetition).
- **Over-trigger evidence.** ProactiveVideoQA PAUC / duplicate ratio: WEB 53.3 / 4.2 % vs
  MMDuet 38.9 / 81.3 %; EGO 33.6 / 8.1 % vs 46.0 / 99.4 %; TV 43.4 / 1.0 % vs 21.1 / 92.8 %;
  VAD 28.9 / 15.2 % vs 27.4 / 99.2 %. Average reply turns on WEB 3.3 vs 5.7. StreamingBench
  PO 34.69 vs 29.44. Wall time on 64 WEB samples on one H100: 2 min 52 s vs 2 min 27 s.
  Code homepage github.com/yellow-binary-tree/mmduet2 (availability not confirmed).
- Relevance: the duplicate ratio collapsing from 81–99 % to 1–15 % under a replication
  penalty is the strongest published evidence that "a repeated report is a false report"
  can be trained by RL with a cheap reward, on a 3B model, in about 160 GPU-hours.

**EvoStreaming** [P] https://arxiv.org/html/2605.10343v1 (read 2026-09-16). Turns an
offline 8B VLM (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, InternVL-3.5, MiniCPM-V 4.5) into a
streaming assistant with no architectural change: the model itself classifies task type,
writes questions, labels segment relevance (binary), rolls out a causal sliding window that
converts relevance into silent/respond decisions with a multiplicative verbosity penalty,
and is LoRA-tuned (r = 8, alpha = 32, all linear layers, LR 1e-4, batch 8, 1 epoch per
iteration, 1–3 iterations) on 1,087 self-generated trajectories from the TimeChat-Online
video pool, on 8×H200. 0.5 fps, 128 tokens per frame in training (128 / 768 at inference).
RealStreamEval overall 54.6 % (Qwen2 variant; FAR 30.6 % vs 12.6 % baseline, RTVP 68.6 % vs
56.4 %); OVO-Bench original protocol +0.6 (50.94 %); offline scores barely move (Qwen2.5
56.5 → 55.8). Tokens per turn fall from 51 → about 10 (Qwen2-VL) and 283 → about 10
(Qwen3-VL). No false-alarm rate; code at github.com/BoxueYang/EvoStreaming. Relevance: a
proof that the silent/respond behaviour is learnable from about 1 K trajectories with a
rank-8 LoRA when the labels are self-consistent, which bounds the minimum for the owner's
teacher-labelled set.

**Secondary (search snippets only, not fetched):** LiveStar's "streaming verification
decoding" gates emission with a perplexity threshold [S]; Response-G1 (2605.07575) aligns
accumulated evidence to the query's response conditions via scene graphs [S]; "Don't Pause!
Every prediction matters in a streaming video" (2604.24317), Pro²Assist (2605.04227),
EgoSAT (2606.24422) and EgoPro-Bench (2605.07299) are 2026 egocentric proactive benchmarks
or systems not read this pass. The "Silence / Standby / Response state tokens with a
focal-weighted loss" design surfaced in a search summary without a paper identifier; it is
possibly Streamo (the source of Streamo-Instruct-465K used by StreamPro) [S, unverified].

### 2.12 RL on streams (Ego-R1 and similar)

[S] Search results 2026-09-16: Ego-R1 (arXiv 2506.13654, github.com/egolife-ai/Ego-R1) is
chain-of-tool-thought RL (GRPO variant) for ultra-long *offline* egocentric video QA with
tool calls (RAG search, video-LLM, VLM), not a streaming when-to-speak policy. The
streaming-proactive RL line is "Eyes Wide Open: Ego Proactive Video-LLM for Streaming
Video" (arXiv 2510.14560, ESTP-Bench and the ESTP-F1 metric), EgoPro-Bench (2605.07299,
personalised proactive interaction in egocentric streams), and MMDuet2's multi-turn RL
(above). DeepVideo-R1 (2506.07464) and LongVideo-R1 (2602.20913) are offline video RL.
Primary-source details below.

Summary of the RL-on-streams recipes read this pass (details in §2.11):

| System | Base | RL data | Algorithm | Reward terms | Compute | Effect |
|---|---|---|---|---|---|---|
| MMDuet2 | Qwen2.5-VL-3B | 1,900 videos | multi-turn GRPO, 4 rollouts | 3×PAUC + 2×replication penalty + 0.5×in-span + 2×prefix | 8 H800 × 20 h | duplicates 81–99 % → 1–15 %; PO 29.4 → 34.7 |
| StreamPro | Qwen3-VL-4B / Qwen2.5-VL-3B | 3 K contexts | GRPO, G = 8, T = 1.0, LR 1e-6 | 0.1 format + 0.45 turn F1 (time × acc, tau = 8 s) + 0.45 LLM rubric | 8 H100 × 24 h | own-bench F1 10.4 → 41.5; OVO FAR 5.4 → 20.6 |
| Em-Garde | Qwen2.5-VL-7B (parser only) | 668 queries | GRPO after 1 h SFT | (1 − r_fp) · n_c / n, explicit FP penalty | 8 A100 × 5 h | PO 38.0; OVO FAR online F1 30.99 |

All three start from an SFT model that already speaks and stays silent; RL then fixes the
timing and the duplicates. None trains from a base with no silence behaviour, so the owner's
order (distil first, RL second) matches the literature.

### 2.13 MiniCPM-o 4.5 duplex listen/speak tokens

[P] https://huggingface.co/openbmb/MiniCPM-o-4_5 (read 2026-09-16).
- 9B total: Qwen3-8B LLM, SigLIP2 vision, Whisper-medium audio encoder, CosyVoice2 speech
  decoder; Apache-2.0 weights and code; 16 quantized variants.
- **Duplex mechanism.** All input and output streams are synchronised on a millisecond
  timeline and time-division multiplexed: parallel omni-modal streams are chopped into
  sequential "info groups" per small periodic time slice; the speech decoder interleaves
  text and speech tokens for full-duplex output; proactive interaction decisions are made at
  1 Hz. The card does not name the listen/speak tokens or describe how the decision is
  trained (training recipe undisclosed).
- **Streaming video.** Up to 10 fps, any aspect ratio, up to 1.8 Mpixel; tokens per frame
  not stated. Video-MME 70.4; OpenCompass 77.6; "LiveSports vision duplex" 54.4 win rate vs
  GPT-4o. No StreamingBench or OVO-Bench figure on the card.
- **Latency and deployment.** TTFT 0.6 s and 154.3 tok/s in bf16 on an NVIDIA GPU (28 GB+
  recommended); int4 212.3 tok/s in 11.0 GB. llama.cpp-omni: half-duplex on Apple M3/M4/M5
  with 16 GB, full duplex on an M4 Max 24 GB; vLLM and SGLang supported.
- Relevance: the only open model with a shipped 1 Hz "decide whether to speak" loop over a
  synchronised A/V timeline, and it fits the Mac; but 9B and undisclosed training make it a
  reference behaviour, not a base to fine-tune under the owner's contract.

### 2.14 Qwen3-VL and Qwen3.5: streaming and timestamp support

[S] Search results 2026-09-16 (github.com/QwenLM/Qwen3-VL; arxiv 2604.15804 "Qwen3.5-Omni
Technical Report"; huggingface.co/Qwen/Qwen3.5-35B-A3B; NVIDIA build card for
Qwen3.5-397B-A17B): Qwen3-VL uses "text-timestamp alignment" (explicit textual timestamps
interleaved with frame groups, replacing T-RoPE) and interleaved-MRoPE. Qwen3.5 exists as a
natively multimodal family (35B-A3B MoE and 397B-A17B on the cards found) and Qwen3.5-Omni
has a technical report (April 2026). Whether Qwen3.5 has small dense vision models (2B–8B),
and whether Qwen3.5-Omni exposes a streaming/duplex "when to speak" mechanism, is checked in
the next batch; see the fetch note below.

[P] https://huggingface.co/Qwen/Qwen3.5-35B-A3B (read 2026-09-16): Qwen3.5 (February
2026) is natively vision-language (image, video, text; no audio in the base line); sizes on
that card: 122B-A10B, 35B-A3B, 27B dense; context 262,144 native, up to 1,010,000 with
YaRN; video preprocessor default fps = 2 with a recommended budget up to "224k video
tokens"; Apache-2.0; served by vLLM, SGLang, KTransformers (no llama.cpp/MLX guidance on the
card); video benchmarks listed: Video-MME, VideoMMMU, MLVU, MVBench, LVBench, MMVU; no
temporal-grounding, StreamingBench or OVO-Bench figure; no timestamp-encoding statement
(Qwen3-VL's text-timestamp alignment is presumably inherited [S]). Whether small dense
Qwen3.5 models (<= 9B) exist with vision is checked in the last batch below.

[S] Search 2026-09-16 (huggingface.co/Qwen/Qwen3.5-4B, /Qwen3.5-9B, /Qwen3.5-2B,
huggingface.co/docs/transformers/en/model_doc/qwen3_5, unsloth GGUF builds, NVIDIA
Megatron-Bridge "qwen35 vl" page): Qwen3.5 Small is 0.8B / 2B / 4B / 9B and Medium is
35B-A3B / 27B / 122B-A10B / 397B-A17B; the family is described as early-fusion
vision-language (text, image, video) that "outperforms Qwen3-VL" at parity; every dense
model (2B, 4B, 9B, 27B) uses a hybrid 3:1 Gated DeltaNet : gated attention layout, so only
one layer in four keeps a growing KV cache, which is the property that matters for
hours-long streams on 12–48 GB machines. GGUF builds exist (llama.cpp support implied);
MLX support, the video preprocessor defaults for the small models, timestamp handling and
any streaming mode were not verified this pass (§7). Qwen3-VL (2B/4B/8B, Apache-2.0,
textual timestamps, measured MLX throughput in the perception note) stays the verified
option; Qwen3.5-4B is the likely successor once its Apple toolchain is confirmed.

### 2.15 Gemma 4 interleaved video support

[P] https://huggingface.co/google/gemma-4-E4B-it (read 2026-09-16). Video is "a maximum of
60 seconds assuming the images are processed at one frame per second", handled as a sequence
of images through the normal image path; audio up to 30 s; "freely mix text and images in
any order within a single prompt"; 128 K context; 4.5 B effective / 8 B total parameters;
about 150 M-parameter vision encoder; 70 / 140 / 280 / 560 / 1120 tokens per image; native
structured tool use; Apache 2.0. No video benchmark on the card and no per-frame timestamp
token, streaming mode, or KV-eviction guidance. So "interleaved video" for Gemma 4 means
frames-as-images interleaved with text (which is what the owner's token stream needs), not a
native video stream with time encoding; timestamps would have to be text prefixes as in
Qwen3-VL, and the 60 s / 1 fps limit is a processor convention, not a hard architectural one
[S, inference]. The audio limit of 30 s per clip is a real constraint for a continuous audio
channel: audio would have to be chunked.

### 2.16 Distillation from a frontier teacher for video event streams

Nobody has published exactly the owner's design (a frontier model watching tick by tick
under the student's contract and its decisions becoming the targets), but four groups have
built proactive streaming datasets by letting a large model decide *when* to speak from
timestamped annotations, and one has self-distilled. What they did and what it cost:

**ProAssist** [P] https://arxiv.org/html/2506.05904 (EMNLP 2025; read 2026-09-16). The
closest precedent, and egocentric.
- **Data.** 478.7 h of video from Ego4D-GoalStep, EPIC-KITCHENS, HoloAssist, Assembly101,
  EgoExoLearn and WTaG; 30,135 dialogues; the teacher is **LLaMA-3.1-70B-Instruct** writing
  proactive assistant dialogues from timestamped user-action descriptions (text, not
  pixels); 10 dialogues per video across three user types in a 2 : 4 : 4 ratio (no_talk,
  talk_some, talk_more), which is an explicit way to teach the rate of speech.
- **Student.** VideoLLM-online lineage: LLaMA-3.1-8B-Instruct + SigLIP-SO400M, 1 / 5 / 10
  tokens per frame tested; decision at the last visual token of each frame: EOS = silent,
  otherwise start a response; single-stage training. Iterative Progress Summarization
  writes a text summary when the context fills (verbalized memory, as in ProVideLLM).
- **Imbalance.** "Far more negative samples (predicting EOS) than positive"; fixed by
  **Negative Frame Sub-sampling with proportion rho = 0.1** (only one in ten silent frames
  contributes to the loss), which lifts Ego4D action-narration F1 from 30.1 to 58.7. Dialogue
  F1 36.07 with task knowledge, 32.77 without (I = 10). LLM-judge Likert ratings on four
  dimensions. No events-per-hour or latency; release "upon publication" (unconfirmed).

**StreamBridge / Stream-IT** (§2.10): GPT-4o writes QA over 1.28 M clips; proactive
timing is inherited from the source annotations, not judged by the teacher.
**Eyes Wide Open** (§2.11): an unnamed LVLM captions, RAG composes 60 K + 20 K questions
with valid-answer intervals. **Em-Garde** (§2.11): GPT-5 authors half of 668 proposal
annotations; 4 s tolerance. **TimeChat-Online-139K** (§2.6): 20 K "not yet answerable"
negatives built from the annotation timeline. **EvoStreaming** (§2.11): the student labels
its own relevance and rolls out silent/respond decisions; 1,087 trajectories suffice for a
rank-8 LoRA to change behaviour.

What is new in the owner's design relative to all of these: (i) the teacher looks at the
pixels, not at narrations, so the labels carry the teacher's visual judgement (and its
errors: the take-pan case in the distillation doc); (ii) the journal-in-context makes
"already reported" a visible condition, which none of the above encodes except MMDuet2's
replication penalty at RL time; (iii) whole quiet hours are labelled, which no public set
does (ProAssist's no_talk user type is the nearest analogue). The costs above suggest the
teacher-labelled set can be small: 1 K–30 K chunks is the range in which published students
learned silence behaviour, provided the loss is rebalanced (§4).

## 3. Benchmarks: StreamingBench, OVO-Bench, and how proactivity, latency and false alarms are measured

**StreamingBench** [P] https://arxiv.org/html/2411.03628 (read 2026-09-16). 900 videos,
4,500 QA, 18 tasks, 3 s to 24 min; real-time visual 500 videos / 2,500 questions,
omni-source 200 / 1,000, contextual 200 / 800. **Proactive Output (PO)**: 250 questions, 50
evaluated; models are polled every second in a ±4 s window around the ground-truth time and
an output is correct only if |t_out − t_gt| < 2 s. Early or spurious outputs outside the
window are simply not counted, so PO measures recall-at-2 s, not false alarms. Streaming is
simulated by clipping Video[0:t_question]; 1 fps for most models. Results (overall / PO):
human 91.66 / 100; Gemini 1.5 Pro 67.07 / 45.10; GPT-4o 60.15 / 56.86; Claude 3.5 Sonnet
57.68 / 64.71; LLaVA-OneVision 56.36 / 29.55; Qwen2-VL 54.14 / 22.73; VideoLLM-online
32.48 / 3.92; Flash-VStream 24.04 / 1.96. Later: Dispider 53.12 / 25.34 (§2.3), Em-Garde PO
38.0, StreamPro-4B RTVU 78.9, StreamBridge Qwen2-VL+Stream-IT RTVU 77.04, TimeChat-Online
RTVU 75.36. No latency measurement.

**OVO-Bench** [P] https://arxiv.org/html/2501.05510 (read 2026-09-16). 644 videos, 7
domains, about 2,814 QA with timestamps, minutes to half an hour. Modes: Backward Tracing
(EPM, ASI, HLD), Real-Time Visual Perception (STU, OJR, ATR, ACR, OCR, FPD), **Forward
Active Responding** (REC repetition count, SSR sequential steps, CRR clues reveal). In FAR
the model is queried densely; for REC/SSR once at the start, for CRR before every
ground-truth event; the model must say whether it can answer yet or must keep watching;
score = Σ F(R_m', A_m) · 2^(−(m'−m)·p) with p = 0.2/0.05 (REC) and 0.5 (SSR/CRR), so late
answers decay exponentially but premature or repeated answers are not explicitly penalised.
Results (BT / RTVP / FAR / overall): human 92.33 / 93.20 / 92.90 / 92.81; Gemini 1.5 Pro
62.54 / 69.32 / 57.15 / 63.00; GPT-4o 60.75 / 64.46 / 53.40 / 59.54; Qwen2-VL-72B 56.95 /
61.92 / 49.30 / 56.27; LLaVA-Video-7B 40.4 / 63.52 / 54.82 / 52.91; LLaVA-OneVision-7B 43.71
/ 64.02 / 50.50 / 52.74; InternVL2-8B 43.44 / 60.39 / 46.60 / 50.15; Flash-VStream-7B 27.38 /
28.37 / 45.09 / 33.61; VideoLLM-online-8B 17.73 / 20.79 / —. Latency: with 64 frames
"most efficient Video-LLMs, like Qwen2-VL-7B and Flash-VStream, still need around 4 seconds
on average" per response, growing with frame count. Later FAR numbers under the original
protocol: StreamPro-GRPO-4B 20.6, Streamo 5.4, EyeWO 24.16 overall; under Em-Garde's online
2 s-window protocol: Em-Garde F1 30.99.

**What the benchmarks do not measure.** None of StreamingBench, OVO-Bench or OVBench
reports a false-alarm rate per hour, a precision of unsolicited outputs, or behaviour on
quiet footage; PO and FAR are recall-style. The metrics that do count false positives are
all from 2025–2026 proactive papers: Proact-VL's trigger F1 + TimeDiff + PAUC, Em-Garde's
online precision/recall with a 2 s window and its RL false-positive penalty, StreamPro's
trajectory F1 with task-specific tolerance windows, ESTP-F1 with FP in the denominator, and
MMDuet2's PAUC (below). None reports events per hour on quiet video, which is the number
the roadmap's §5 acceptance targets need; that measurement has to come from the owner's
own quiet-hours protocol (decoding note §4.2).

## 4. A streaming-SFT recipe for a 4B VLM under the silent / context / act contract (deliverable b)


Every number below names its precedent; where there is none it says "owner's choice" and
gives the reason. The teacher measurements it leans on: 2 Hz labels, 86 % silent ticks in
dense cooking footage (138 ticks / 19 pushes / 6 second looks), 15 of 19 pushes right,
teacher-teacher agreement 0.52, v2 second look on every candidate pickup or placement
(about 1 tick in 7) (`docs/teacher-distillation.md` §4.1).

### 4.1 Stream rendering (what the student reads and writes)

| Element | Choice | Precedent / reason |
|---|---|---|
| Tick | 0.5 s (2 Hz), the teacher's rate | Proact-VL 1 s chunks at 2 fps; LiveCC/VideoLLM-online 2 fps; the lid lift needs 2 Hz (teacher §4) |
| Tick marker | text timestamp `<t 123.5>` (4–6 tokens) | Qwen3-VL text-timestamp alignment; TimeMarker "Second{i}" text (decoding note §2.2); keeps any base model usable |
| Frame admission | change-gated: a frame enters only if the gate fires; runs of frameless ticks longer than 2 s collapse to one span marker `<t 120.0-180.0 still>` | TimeChat-Online DTD drops ~85 % of tokens at 98 % of accuracy; "80 % of visual tokens are naturally redundant" (perception note §6); span collapse is owner's choice to keep quiet hours cheap |
| Tokens per admitted frame | 144 (about 384×384 at Qwen3-VL's 32 px per token); ablate 64 | Apollo 8–32 tok/frame optimal offline; VideoChat-Online 16-token tier; MMDuet 49; MMDuet2/EvoStreaming 128; Proact-VL ≤ 540. Gated frames carry motion by construction, so spatial detail can be modest; the second look (§5) supplies resolution on demand |
| Audio | text tags at the tick they occur (ASR words, sound-event labels) | LiveCC interleaves ASR words per frame interval; Gemma 4 audio is 30 s clips only |
| Device state | one text line at the tick it changes | owner's design; no precedent needed |
| Journal | the student's own emitted lines stay in the text stream; older frames are evicted, lines are not | ProVideLLM verbalized memory (630 tokens per hour); ProAssist Iterative Progress Summarization; StreamingVLM long text window + short vision window |
| Silence target | one token per tick (`.`), placed after the tick's frame or marker | LiveCC/StreamingVLM "…"; StreamPro `</Silence>`; VideoLLM-online EOS |
| Burst target | one compact line, ≤ 40 tokens: `{"k":"object_placed","s":"pan","l":"hob","d":"pan set on the hob","c":0.8}` then newline; `act` lines as `act rule_id args` | StreamPro `</Response>` + text; MMDuet2 literal text decisions. Short keys and masked scaffolding answer roadmap F1 (74 % of former target tokens were JSON scaffolding) |
| Second look | a `look` token that the runtime answers with the last 2 s re-sampled at 4 Hz and 2× resolution, then a final decision | Eyes Wide Open `ask_high`; teacher v2 second look; §5 |

### 4.2 Chunking, overlap and attention

- **Training sample** = a fixed prefix (system prompt, goal, action policy, journal-so-far as
  text) + a 30 s lead-in of ticks with loss masked + 120 s of ticks with targets, i.e. six
  consecutive teacher chunks of 20 s. Precedent: StreamingVLM trains on 24 s chunks with 12 s
  overlap, full attention inside a chunk, loss only on non-overlap text; the owner's chunk is
  longer because bursts are sparse (about 1 per 8 s in busy footage, near 0 in quiet hours) and
  the "already reported" condition needs the journal exercised within the sample.
- **Attention**: plain causal attention within the sample; no custom mask. At inference the
  KV cache keeps the prefix + journal as sinks and the last 60–90 s of gated frames as the
  vision window, evicting older frames (StreamingVLM: sinks 512 + text window 512 + 16 s vision
  window, RoPE re-indexed contiguous). On the Mac the simpler route is prefix caching of the
  sink + re-prefill of the window (measured 24–28× speed-up on repeated context, perception
  note §6). Training context per target tick is 30–150 s, inference 60–90 s; StreamingVLM's
  result is that this overlap-trained/window-served mismatch is benign.
- **Sequence length**: busy 120 s ≈ 120 gated frames × 144 + 3 K text ≈ 20 K tokens; quiet
  120 s ≈ 2 K tokens. Cap 24 K; pack quiet samples to fill. Token budget per optimizer step
  256 K (8–16 samples).

### 4.3 Silence-vs-burst loss weighting and negative ratio

Data level: keep every labelled quiet hour; mix by hours ≥ 50 % quiet or near-quiet (≤ 2
bursts per hour) with the rest busy. Precedent: ProAssist's 2:4:4 no_talk / talk_some /
talk_more user types; TimeChat-Online's 20 K unanswerable; the teacher doc's rule 5.4.

Token level (per-token loss weights; the only lever that the papers agree on):

| Token class | Weight | Precedent |
|---|---|---|
| silence token `.` | 0.1–0.2 (or keep a random 10–20 % of silence positions per step, equivalently) | ProAssist negative-frame sub-sampling rho = 0.1: F1 30.1 → 58.7; StreamPro class-balanced beta = 0.9999 gives the same order for 10^5 silent tokens |
| transition ticks: the first burst token after silence, and the first `.` after a burst | ×5 | Proact-VL gamma = 5 on state transitions |
| burst decision-bearing tokens (kind, subject, location, detail, confidence) | ×2 | StreamPro lambda_text = 2 |
| burst scaffolding (braces, keys, quotes) | 0 or 0.1 | roadmap F1; decoding note §2.5 (mask fixed keys) |
| `act` decision token | ×10, and ≥ 5 negative-act contexts (rule present, must not fire) per positive | owner's choice; no published precedent; F2 says abstention is what fails |
| `look` token | ×2; and `look` → `.` cases must be ≥ 50 % of look targets | owner's choice; Eyes Wide Open gives no weights |
| lead-in ticks | 0 | StreamingVLM overlap masking |

Optional: a batch-level speaking-rate regulariser, weight 0.1, on |mean p(burst) − labelled
burst rate| (Proact-VL's global rate term). Check after the first run that the summed silence
loss is 1–2× the summed burst loss per batch and adjust the silence weight to land there.
Inference threshold: emit a burst only when p(`.`) < theta, with theta fixed on the quiet-hours
calibration set by the LTT rule in the decoding note §4.2 (≤ 13 false bursts in 600 quiet
minutes certifies ≤ 2 per hour at 90 %); VideoLLM-online 0.5–0.8, Proact-VL 0.3 and
StreamBridge 0.35 are the published operating points, all on recall-style benchmarks.

### 4.4 Frame rate and tokens per frame

2 Hz ticks; frames admitted on change (expect 30–60 % of ticks in busy footage, under 5 % in
quiet hours); 144 tokens per frame with a 64-token ablation; the second look at 4 Hz for 2 s
at 2× resolution (8 frames × 256–576 tokens). Roadmap F4: 51 % of narrated actions last under
1 s, so temporal density matters more than spatial; the teacher needed 4 Hz to settle lid vs
pan direction, which is what `look` buys.

### 4.5 What to unfreeze, LoRA ranks, epochs

- **Stage A (always)**: projector fully trained; LoRA r = 128, alpha = 256 on every linear layer
  of the LLM (VideoLLM-online, ProVideLLM); vision tower frozen (every streaming recipe read).
  LR 1e-4 for LoRA (EvoStreaming 1e-4 at r = 8; StreamBridge activator 2e-4), 2e-5 for the
  projector; AdamW, cosine, warm-up 3 %; bf16. Full LLM fine-tuning at LR 1e-5 (Proact-VL,
  StreamPro, TimeChat-Online) only when the labelled set exceeds about 20 K chunks; below that
  LoRA is what the 1 K–3 K-sample successes used.
- **Stage B (only if held-out state-change recall misses target after A)**: LoRA r = 32 on the
  top third of vision-tower blocks at LR 1e-5 with the projector still training (decoding note
  §2.1: Cambrian, Idefics2 evidence; Prismatic's warning that unfreezing on small data hurts).
  Gate it on the same-image and reversed-order controls from roadmap F3 before and after.
- **Epochs**: 1 over the mix (TimeChat-Online, StreamBridge, MMDuet, StreamPro, EvoStreaming);
  2 only if the teacher set is under 10 K chunks, with journal dropout p = 0.3–0.5
  (StreamBridge random-QA-drop 0.55), tick jitter ±0.25 s, and resolution jitter 64–196
  tokens per frame.
- **Label noise**: teacher-teacher agreement is 0.52 and 15 of 19 pushes were right; treat the
  SFT target as "speak roughly when the teacher speaks" and leave exact timing and duplicate
  suppression to RL (MMDuet2's replication penalty, Em-Garde's FP term, StreamPro's tolerance
  windows), 2–3 K contexts, GRPO with G = 8 (StreamPro) or 4 (MMDuet2), LR 1e-6, tolerance
  tau = 1.5 s (the audit collar) rather than StreamPro's 8 s.

### 4.6 Expected H100 hours and dollars (derived, not measured)

Assumptions: 300 labelled hours (150 busy at 45 % frame admission, 150 quiet at 5 %) → about
540 K admitted frames × 144 tokens ≈ 78 M visual tokens + ≈ 15 M text tokens + 25 % lead-in
overhead ≈ 115 M tokens per epoch. LLM training cost 6 × 4.4e9 × 1.15e8 ≈ 3.0e18 FLOP
(LoRA saves about a third of the backward pass → ≈ 2.0e18) plus the frozen vision-tower
forward ≈ 2.5e17. At 25 % MFU on one H100 (≈ 2.5e14 FLOP/s) that is ≈ 2.5 H100-hours per
epoch; with video decoding, long-sequence attention and small batches, budget 3–5× → **8–12
H100-hours per SFT run**. Cross-check against a measured recipe: Proact-VL spent about 200
H100-hours on 128 K samples of ≈ 19 K tokens (≈ 2.4 B tokens) for a 7B full fine-tune, i.e.
≈ 83 H100-h per billion tokens; scaled to 4B LoRA on 0.115 B tokens that is ≈ 5–10 H100-h,
consistent. A cycle: 6-run SFT sweep (silence weight, 64 vs 144 tokens, LoRA vs full)
60–100 H100-h; GRPO on 3 K contexts × G = 8 × 120 s, MMDuet2/StreamPro precedent 160–192
GPU-h at 3–4B → 100–200 H100-h; evaluation, controls and `look` ablations 20–40 H100-h.
**Total ≈ 200–350 H100-hours per cycle.** At 2026 on-demand prices (RunPod secure cloud
$2.89–2.99 per H100-hour, community $1.99–2.69, other providers $1.49–6.98 [S, search
2026-09-16]) that is **≈ $400–1,050 per cycle**; an 8×H100 node for 24–44 h. Teacher
labelling cost is separate (teacher doc §6).

### 4.7 What to measure that the papers do not

False bursts per quiet hour (target ≤ 2 per hour, decoding note §4.2), duplicate ratio
(MMDuet2's metric), TimeDiff to narrated onsets with the 1.5 s collar, the same-image and
reversed-order controls (roadmap F3), per-kind recall, and `look` rate and its precision gain.


## 5. Same-model "second look" verification: has anyone done it? (deliverable c)


Found once as a trained action, several times as a cascade or a decode-time gate:

1. **Eyes Wide Open (VideoLLM-EyeWO, Oct 2025)** [P]: the streaming policy chooses per step
   among `silence`, `response` and `ask_high`, where `ask_high` requests high-resolution
   frames for the current moment before deciding; stage 2 of its three-stage SFT trains this
   "proactive just-in-time responsiveness" with the extra action; ESTP-F1 34.7 vs 15.5 for the
   VideoLLM-online baseline. The paper as fetched gives no `ask_high` rate, cost, or ablation
   isolating its contribution (§7). It is the only published same-model second look in a
   streaming VLM.
2. **Dispider** [P]: a two-model cascade; the 7B responder can answer `<SILENT>` when the 1.5B
   decider fired wrongly ("enables the LLM to rethink"); ablation shows a gain but no rate.
3. **MMDuet / MMDuet2** [P]: temporal integration is a soft second look (cumulative
   informative score must reach 2 before speaking; MMDuet2's replication penalty removes the
   repeats that a first look produced).
4. **LiveStar** [S, search snippet]: "streaming verification decoding" gates emission with a
   perplexity threshold at decode time; not read.
5. **StreamBridge** [P]: a separate 0.5B activator runs first; the 7B model only speaks when
   the activator fires (cascade, not same-model).
6. Offline "zoom-in" RL (DeepEyes, Chain-of-Focus, ZoomEye, V*) trains a model to crop and
   re-look at image regions [S, recalled, not verified this pass]; none is streaming.

Proposed student mechanism, matching the teacher's v2 protocol: `look` is a fourth decision
at any tick. The runtime answers it by appending `<look 122.0-124.0 4hz 2x>` plus 8 frames of
the last 2 s at 4 Hz and 2× resolution (256–576 tokens each; about 0.7–1.5 s of prefill on
the M5 Max, estimated from 0.3 s per 1024² frame in the perception note §6), after which the
student emits `.` or a burst. Targets: every teacher second look becomes `look` → the
teacher's post-look decision; teacher v2 takes a second look on every candidate pickup or
placement (≈ 1 tick in 7 in busy footage), so `look` will be about as frequent as bursts;
add hard-negative `look` → `.` cases (hand motion without a state change) until at least half
of all `look` targets end in silence, otherwise the student learns `look` = "about to speak".
Use the pair (first-look decision, post-look decision) as the P(True)-style agreement score of
the decoding note §4.1 for calibration. Cap the `look` rate at 10 % of busy ticks at
inference and report its precision gain separately (§4.7). Risk: no paper reports the false
alarm reduction from a same-model second look, so the gain is unmeasured until the owner's
first run.


## 6. Base-model choice, ranked, for the Mac (M5 Max 48 GB) and the 12–16 GB GPU box (deliverable d)


Requirements from the direction: one model for both targets; interleaved frames, text and
device state with timestamps; hours-long streams with a bounded cache; Apache-2.0 or
equivalent; a training stack on rented H100s; MLX or llama.cpp on the Mac, vLLM or llama.cpp
on the box.

**Mac (M5 Max, 48 GB)**

1. **Qwen3-VL-4B-Instruct** — verified: MLX throughput measured (143 tok/s decode; prefill is
   the cost; 24–28× prefix-cache gain), textual timestamps built in, Charades-STA mIoU 55.5
   at 4B, Apache-2.0, LoRA tooling mature, and StreamPro shipped an end-to-end proactive
   streamer on exactly this backbone (own-bench F1 41.5, OVO FAR 20.6, SB RTVU 78.9). 4-bit
   weights ≈ 3 GB; 48 GB leaves room for a 60–90 s vision window and a co-resident 8B judge.
2. **Qwen3.5-4B** — likely better and cheaper on long streams: early-fusion vision-language,
   claimed to outperform Qwen3-VL at parity, 3:1 Gated-DeltaNet : attention so only a quarter
   of the layers hold a growing KV cache (hours-long streams in a fixed memory budget), GGUF
   builds exist [S]. Unverified this pass: mlx-vlm support, video preprocessor defaults for
   the 4B, timestamp handling, LoRA recipes for the hybrid layers (§7). Promote to first once
   those are confirmed on the Mac; the stream format in §4.1 is base-model agnostic so the
   switch costs one re-run.
3. **Gemma 4 E4B** — only if native audio in the same model is required: Apache-2.0, MLX and
   llama.cpp day-0, 128 K context, but no timestamp encoding, video as ≤ 60 s of 1 fps frames,
   audio in 30 s clips, no published video benchmark, and the owner's own F1–F7 failures were
   on this model (a recipe failure, not a model verdict). Text timestamps and the §4 recipe
   would apply unchanged.
4. **Qwen3.5-9B / Qwen3-VL-8B** — not as the streamer; as the co-resident second-look or
   judge model if the 4B `look` pass is not enough (perception note: Qwen3-VL-8B cold first
   turn 21.7 s, so only with prefix caching).
5. **MiniCPM-o 4.5 (9B)** — reference behaviour for a 1 Hz duplex loop on a Mac (llama.cpp-omni,
   full duplex on an M4 Max 24 GB) but undisclosed training and 9B; not a fine-tuning base.
   InternVL3.5-4B (weak Apple tooling), SmolVLM2-2.2B (no grounding), VideoChat-Online-4B
   (Phi-3 + InternViT, older, query-driven) rank below.

**12–16 GB GPU box**

1. **Qwen3-VL-4B-Instruct** — bf16 weights ≈ 8 GB fit 16 GB with a modest window; on 12 GB
   use AWQ/GPTQ int4 (≈ 3 GB) under vLLM or GGUF under llama.cpp; the same LoRA merges.
2. **Qwen3.5-4B** — vLLM, SGLang and KTransformers are the documented servers for the family
   [P, 35B card]; the small-KV hybrid layout is most valuable exactly on a 12 GB box.
3. **Gemma 4 E4B** — caution: 8 B total parameters with embeddings, so bf16 is ≈ 16 GB and
   does not fit a 12–16 GB box without int4/int8 (≈ 5 GB at 4-bit per the perception note)
   and per-layer-embedding handling; workable, but the tightest of the three.

Pick one backbone for both machines (the adapter is shared); today that is Qwen3-VL-4B.


## 7. Not verified / gaps (deliverable e)


- **Release status not confirmed from the source read**: StreamBridge (code/weights/data),
  Proact-VL (code, weights), StreamPro (everything), ProAssist ("upon publication"), Eyes
  Wide Open (project page only), ProVideLLM (project page only), MMDuet2 (GitHub page named,
  contents unknown). MMDuet and TimeChat-Online releases are recalled, not re-fetched.
- **Flash-VStream ICCV 2025**: memory sizes, data, compute and latency not read (repo landing
  page only); its numbers in the table are from the 2024 version via other papers.
- **StreamingVLM**: the vision-window size was rendered ambiguously by the fetch ("16 s",
  "4 frames at 24 fps"); take 16 s as the figure and check the repo config before copying it.
- **Qwen3.5 small models**: existence of 0.8B/2B/4B/9B with vision comes from search
  snippets and third-party GGUF builds; the 4B card, video preprocessor defaults, timestamp
  scheme, MLX support and any streaming mode were not fetched. Qwen3.5-Omni's streaming
  design (arXiv 2604.15804) was not read.
- **Gemma 4**: nothing beyond the E4B card; whether the 60 s / 1 fps video limit is a
  processor convention or a training-distribution limit is unknown; no per-frame timestamp
  scheme is documented.
- **MiniCPM-o 4.5**: the listen/speak token names and the training of the 1 Hz decision are
  undisclosed on the card; the GitHub README was not fetched.
- **"Silence / Standby / Response state tokens with focal-weighted loss"** appeared in a
  search summary without a paper identifier; possibly Streamo (source of Streamo-Instruct-465K
  in StreamPro). Unidentified. LiveStar's verification decoding, Response-G1, "Don't Pause!",
  Pro²Assist, EgoSAT and EgoPro-Bench were not read.
- **Eyes Wide Open `ask_high`**: no rate, cost or isolated ablation was in the fetched text.
- **No fixed-camera home footage anywhere**: every system above trains on egocentric,
  broadcast, gaming or web video; presence/location behaviour on static cameras is untested.
- **No events-per-hour on quiet footage** in any paper or benchmark; the false-alarm metrics
  that exist (trigger F1, PAUC, duplicate ratio, ESTP-F1, online precision at 2 s) are on
  busy footage with a query.
- **H100-hour and dollar figures in §4.6 are derived** from FLOP counts and one measured
  recipe (Proact-VL), not measured for a 4B student; prices are search snippets dated
  2026-09-16 and move weekly.
- **Loss weights in §4.3 are transplanted** from 7B/8B recipes (ProAssist rho = 0.1,
  Proact-VL gamma = 5, StreamPro beta = 0.9999, lambda_text = 2); none was tuned on a 4B
  model with a 2 Hz silence token, so the first sweep must include the silence weight.
- **Frame gating as the token-stream sampler** has a precedent only as a heuristic outside
  the model (TimeChat-Online DTD, Em-Garde similarity jump); no paper trains a model on a
  change-gated stream and reports the effect on false alarms.


## 8. Sources (URL, date accessed 2026-09-16)


Primary (fetched and read this pass):
- VideoLLM-online: https://arxiv.org/html/2406.11816
- MMDuet: https://arxiv.org/html/2411.17991
- MMDuet2: https://arxiv.org/html/2512.06810
- Dispider: https://arxiv.org/html/2501.03218
- Flash-VStream repo: https://github.com/IVGSZ/Flash-VStream (paper arXiv 2506.23825 not read)
- VideoChat-Online / OVBench: https://arxiv.org/html/2501.00584
- TimeChat-Online: https://arxiv.org/html/2504.17343
- LiveCC: https://arxiv.org/html/2504.16030
- StreamingVLM: https://arxiv.org/html/2510.09608 ; code https://github.com/mit-han-lab/streaming-vlm
- ProVideLLM: https://arxiv.org/abs/2504.13915 and https://arxiv.org/html/2504.13915
- StreamBridge: https://arxiv.org/html/2505.05467
- ProAssist: https://arxiv.org/html/2506.05904 (EMNLP 2025: https://aclanthology.org/2025.emnlp-main.605.pdf)
- Eyes Wide Open: https://arxiv.org/html/2510.14560v1
- Proact-VL: https://arxiv.org/html/2603.03447
- Em-Garde: https://arxiv.org/html/2603.19054v2 ; code https://github.com/air-embodied-brain/Em-Garde
- StreamPro: https://arxiv.org/html/2605.16381
- EvoStreaming: https://arxiv.org/html/2605.10343v1 ; code https://github.com/BoxueYang/EvoStreaming
- StreamingBench: https://arxiv.org/html/2411.03628
- OVO-Bench: https://arxiv.org/html/2501.05510
- MiniCPM-o 4.5 card: https://huggingface.co/openbmb/MiniCPM-o-4_5
- Gemma 4 E4B-it card: https://huggingface.co/google/gemma-4-E4B-it
- Qwen3.5-35B-A3B card: https://huggingface.co/Qwen/Qwen3.5-35B-A3B

Secondary (search snippets, not fetched):
- Qwen3.5 small models: https://huggingface.co/Qwen/Qwen3.5-4B , https://huggingface.co/Qwen/Qwen3.5-9B , https://huggingface.co/Qwen/Qwen3.5-2B , https://huggingface.co/docs/transformers/en/model_doc/qwen3_5 , https://huggingface.co/unsloth/Qwen3.5-9B-GGUF , https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/vlm/qwen35-vl.html
- Qwen3-VL repo and Qwen3.5-Omni report: https://github.com/qwenlm/qwen3-vl , https://arxiv.org/html/2604.15804v1
- Proactive 2026 papers not read: Response-G1 https://arxiv.org/html/2605.07575 ; "Don't Pause!" https://arxiv.org/pdf/2604.24317 ; Pro²Assist https://arxiv.org/pdf/2605.04227 ; EgoSAT https://arxiv.org/pdf/2606.24422 ; EgoPro-Bench https://arxiv.org/pdf/2605.07299 ; Ego-R1 https://arxiv.org/pdf/2506.13654 ; awesome list https://github.com/sotayang/Awesome-Streaming-Video-Understanding
- LiveStar / streaming verification decoding via https://www.emergentmind.com/topics/streaming-video-large-language-model-llm ; ViCoStream https://arxiv.org/pdf/2606.19849 ; V-Rex https://arxiv.org/pdf/2512.12284
- H100 prices: https://www.runpod.io/gpu-models/h100 , https://www.spheron.network/blog/runpod-h100-pricing-2026/ , https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison , https://shattered.io/h100-h200-b200-cloud-gpu-pricing-2026/

Project documents this note builds on: `docs/model-roadmap.md` §1, `docs/research/decoding-and-training-2026-09.md` §2 and §4, `docs/research/perception-stack-2026-09.md` §6, `docs/teacher-distillation.md` §2–§6.

