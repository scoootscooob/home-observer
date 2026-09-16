<!-- Research note produced 2026-09-16 by a web-research agent for the model roadmap; derived costs and plan quantities are the note's own estimates; verify before relying on any single number. -->

# Labeling pipeline and data plan for continuous static-camera home footage — research report (Sept 2026)

Scope: web research only, no repository changes. Numbers below are from the cited primary sources; where I computed something (cost per hour, human-time estimates, plan quantities) it is marked as a derivation or an estimate. Items I could not verify in this pass are flagged at the end.

## Summary

- Frontier VLMs are good enough to **propose** events and second-level boundaries on ordinary footage, but not to produce final labels: on re-annotated Charades, Gemini 2.5 Pro reaches only mIoU 52.8 (R1@0.7 = 34.0), GPT-5 40.5, and open Qwen3-VL-8B 48.3 ([TimeLens](https://arxiv.org/abs/2512.14698)); on 45-minute continuous videos GPT-5's tIoU@0.5 collapses to 12.6 even with 500 frames ([LongEgoRefer](https://arxiv.org/abs/2607.02096)). Gemini timestamps are second-granular by design ([Google forum](https://discuss.ai.google.dev/t/improve-timestamp-accuracy-on-video-understanding/95356)). Closed models are weak at per-object spatial grounding (Molmo2 8B beats Gemini 3 Pro on video pointing F1 38.4 vs 20.0 and tracking J&F 56.2 vs 41.1, [Molmo2](https://arxiv.org/abs/2601.10611)), so boxes/masks/identities should come from SAM-family models, not the VLM.
- Cheapest exhaustive VLM pass is Gemini: roughly **$0.1–0.8 per footage-hour** at 1 fps depending on model/resolution (derived from [video-understanding docs](https://ai.google.dev/gemini-api/docs/video-understanding) + [pricing](https://ai.google.dev/gemini-api/docs/pricing)). Frame-sequence APIs (GPT-5.x, Claude) cost 5–30× more per hour at the same fps.
- Human time dominates: SAM 2's own data engine still needed manual edits on 19% of frames at 4.5 s/frame ([SAM 2](https://arxiv.org/abs/2408.00714)); SAM 3's AI verifiers only doubled throughput ([SAM 3](https://arxiv.org/abs/2511.16719)). Budget 30–90 human-minutes per **active** footage-hour and a few minutes per quiet hour.
- Generalization evidence says homes/viewpoints matter more than hours: within one apartment, cross-view drops I3D from 53.4 to 34.9 mean per-class accuracy ([Toyota Smarthome](https://openaccess.thecvf.com/content_ICCV_2019/papers/Das_Toyota_Smarthome_Real-World_Activities_of_Daily_Living_ICCV_2019_paper.pdf)) and halves per-frame mAP on TSU (26.7 to 13.4, [TSU](https://arxiv.org/abs/2010.14982)). Plan for 10+ training homes and 3–4 frozen held-out homes.
- For deployed alerting, report Pmiss vs false alarms per unit time (ActEV convention, [TRECVID 2019](https://arxiv.org/abs/2009.09984)); real-world fall detectors that looked good in the lab produced 3–85 false alarms/day in the field ([Bagalà et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0037062)). There is no published consensus "acceptable" rate; the field practice is escalation gating (confirm-by-SMS before alerting) rather than a raw threshold ([ambient-sensing feasibility study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12740839/)).

---

## 1. Auto-labeling continuous video with frontier VLMs

### 1.1 Measured temporal-localization accuracy

| Model | Charades (re-annotated) mIoU | ActivityNet mIoU | QVHighlights mIoU | Source |
|---|---|---|---|---|
| Gemini 2.5 Pro | 52.8 (R1@0.3/0.5/0.7 = 74.1/61.1/34.0) | 58.1 | 70.4 | [TimeLens](https://arxiv.org/abs/2512.14698) |
| Gemini 2.5 Flash | 48.6 | 52.5 | 64.3 | same |
| GPT-5 | 40.5 | 42.9 | 56.8 | same |
| GPT-4o | 41.8 | 40.4 | 52.1 | same |
| Qwen3-VL-8B (open, Apache-2.0) | 48.3 | 46.8 | 59.4 | same; license per [HF card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen2.5-VL-7B | 39.3 | 31.4 | 31.6 | same |
| TimeLens-8B (open, RL-tuned) | 55.2 | 53.2 | 65.5 | same |

Key caveats from the same sources:

- **Benchmarks are noisy.** TimeLens found 20.6% of Charades-STA samples violate query uniqueness and 34.9% have annotation accuracy issues; they re-annotated with an automated pipeline plus manual auditing. Expect a similar fraction of your own VLM labels to need human correction.
- **Long, sparse video is the hard case.** LongEgoRefer (avg 2,715 s videos, 1.3% target presence): GPT-5 tIoU@0.5 = 12.55 with 500 frames (API cap) vs 3.60 with 64 frames; reasoning effort minimal→high raised it 8.7→12.61; Gemini 2.5 Pro 8.68 ([arXiv 2607.02096](https://arxiv.org/abs/2607.02096)). Continuous home footage is exactly this regime, so chunk into short windows rather than asking for whole-hour localization.
- **Timestamp precision.** Gemini expects/returns `MM:SS`; a Google staffer confirmed second-level accuracy is the current limit ([forum](https://discuss.ai.google.dev/t/improve-timestamp-accuracy-on-video-understanding/95356)). TimeLens found interleaved textual timestamps outperform position-embedding or visual-overlay encodings; Qwen3-VL prefixes each temporal patch with text like `<3.0 seconds>` (trained in seconds and HMS), and evaluates Charades-STA at 4 fps with up to 2,048 frames / 224K video tokens ([Qwen3-VL report](https://arxiv.org/abs/2511.21631)).
- **Newer Gemini 3.x**: Google's [model page](https://deepmind.google/models/gemini/pro/) lists 1M input context and video input for Gemini 3.1 Pro but no video-grounding benchmarks; I found no independent grounding numbers for 3.x Pro.

### 1.2 API mechanics that determine the recipe

- **Gemini** ([docs](https://ai.google.dev/gemini-api/docs/video-understanding)): native video; default 1 fps, custom `fps`; 66 tokens/frame at default (low) media resolution, 258 at high; ~100 vs ~300 tokens per second of video; audio 32 tokens/s; up to 3 h (low) or 1 h (high) per 1M context; up to 10 videos per request; Files API up to 20 GB; `start_offset`/`end_offset` clipping; `MM:SS` timestamps. New `processing: "agentic"` mode (Gemini 3.8/3.7/3.6 Flash, 3.5 Flash-Lite) navigates the timeline and uses up to 88% fewer tokens, which is useful for search but not for exhaustive labeling. The 2.5 launch post reports low-res costs only 0.5 pt on VideoMME (84.7 vs 85.2) ([blog](https://developers.googleblog.com/en/gemini-2-5-video-understanding/)).
- **OpenAI GPT-5.x** ([vision guide](https://developers.openai.com/api/docs/guides/images-vision)): no video input; frames as images, up to 1,500 images and 512 MB per request; 32×32-pixel patches (GPT-5.5 multiplier 1.2×); `detail: low` fits within 512×512.
- **Claude** ([vision docs](https://platform.claude.com/docs/en/build-with-claude/vision)): no video input; up to 600 images per request (100 on 200k-context models such as Haiku 4.5); 28×28-pixel visual tokens, so a 1920×1080 frame costs 1,560 tokens on standard-tier models and 2,691 on high-resolution-tier (Claude 4.7+); requests with >20 images require every image ≤2000 px per side; docs explicitly say coordinate/localization outputs are approximate.

### 1.3 Cost per hour of footage (my derivation from the cited token rates and price pages)

Input tokens only; output adds ~30–60K tokens per hour of dense JSON labels (e.g., $0.36–0.72/h at Gemini 3.1 Pro's $12/M output).

| Setup | Tokens per footage-hour | Cost per footage-hour |
|---|---|---|
| Gemini 3.1 Pro Preview, 1 fps low-res, chunks ≤200K tokens | 0.36M | $0.72 standard, $0.36 batch ($2/M and $1/M input, [pricing](https://ai.google.dev/gemini-api/docs/pricing)) |
| Gemini 3.1 Pro Preview, 1 fps high-res | 1.08M | $2.16 standard, $1.08 batch (≤200K chunks); $4.32 if sent as one >200K prompt |
| Gemini 2.5 Pro, 1 fps low / high | 0.36M / 1.08M | $0.45 / $1.35 ($1.25/M ≤200K) |
| Gemini 3.8/3.7/3.6 Flash, 1 fps low / high | 0.36M / 1.08M | $0.27 / $0.81 ($0.75/M through Dec 2026; batch half) |
| Gemini 2.5 Flash or 3.5 Flash-Lite, 1 fps low / high | 0.36M / 1.08M | $0.11 / $0.32 ($0.30/M) |
| GPT-5 at 1 fps, 1280×720 (≈920 patches/frame) | ≈3.3M | ≈$4.1 ($1.25/M); gpt-5.4-mini ≈$2.5; gpt-5-nano ≈$0.17; gpt-5.5 ≈$20 ([pricing](https://developers.openai.com/api/docs/pricing)) |
| GPT-5 at 1 fps, `detail: low` | ≈0.9M | ≈$1.2 |
| Claude Sonnet 5 at 1 fps, 1280×720 (46×26 = 1,196 tokens/frame) | ≈4.3M | ≈$8.6 ($2/M); Opus 5 ≈$21.5 ($5/M); Haiku 4.5 ≈$4.3 ($1/M) ([pricing](https://claude.com/pricing)) |
| Claude Sonnet 5 at 1 fps, 640×360 (299 tokens/frame) | ≈1.08M | ≈$2.2; Haiku ≈$1.1 |
| Qwen3-VL-8B/32B self-hosted | GPU time | no primary throughput source; rough estimate $1–4 per footage-hour on a rented H100 at 1–2 fps |

Practical consequence: use Gemini (batch) for the exhaustive first pass; reserve GPT-5.x/Claude for adjudication of flagged windows or for cross-model agreement checks on a sample.

### 1.4 Recipes from published "label with a big model, distill to a small one" pipelines

| Pipeline | Labeler | Chunking / fps | Human role | Distilled student / outcome |
|---|---|---|---|---|
| [LLaVA-Video-178K](https://arxiv.org/abs/2410.02713) | GPT-4o | 1 fps; recurrent 3 levels: 10 s detail → 30 s summary → global | none | 178K videos, 960K QA; trained LLaVA-Video |
| [ShareGPT4Video](https://arxiv.org/abs/2406.04325) | GPT-4V | differential sliding-window captioning (describe change vs previous frames) | none | ShareCaptioner-Video re-captioned 4.8M clips |
| [Vript](https://arxiv.org/abs/2406.06040) | GPT-4V + Whisper | PySceneDetect scene split, 1–5 frames per scene | none | Vriptor captioner |
| [VideoRefer-700K](https://arxiv.org/abs/2501.00599) | multi-agent: analyzer, annotator, segmentor, reviewer, refiner | object-level | reviewer agent only | object-level Video-LLM |
| [Panda-70M](https://arxiv.org/abs/2402.19479) | 8 teacher captioners | clip split | humans pick best caption on a small subset to train a selector | student matches the "All Teachers" ensemble |
| [LaViLa](https://arxiv.org/abs/2212.04501) | LLM narrator on Ego4D | dense pseudo-narrations | none | dual encoder +10.1 pts EGTEA, +5.9 EK-100 MIR; half the narrations beat full human set |
| [LEViL](https://arxiv.org/abs/2606.21358) | InternVL2-8B captions → spaCy verb/noun pseudo-label space (2,391 concepts) | clip-level | none | 3D ResNet-18: UCF101 73.3% with 10% labels vs 87.8% fully supervised |
| [AutoAD III](https://arxiv.org/abs/2404.14412) | LLM converts HowTo100M captions to AD style; AudioVault alignment | — | none | AD generator |
| [LongVALE](https://arxiv.org/abs/2411.19772) | automatic omni-modal boundary detection + captioning | event boundaries | two human groups fully check the test set | 105K events / 8.4K videos |
| [Video Annotator (Netflix)](https://arxiv.org/abs/2402.06560) | VLM embeddings + linear classifier | clip-level | domain experts label ≥100 pos + 100 neg per concept via top-positive / top-negative / uncertainty (|score−0.5|) / random feeds | median +8.3 AP over the best baseline; 153K labels over 56 tasks by 3 editors |
| [Violence detection auto-label](https://arxiv.org/abs/2511.10866) | LLM auto-captions 1–2 s clips | sliding short windows | none | real-time model, 95.25% on RWF-2000 |
| [PixMo/Molmo](https://arxiv.org/abs/2409.17146), [Molmo2](https://arxiv.org/abs/2601.10611) | humans only (speech captions, pointing, tracking); no closed VLMs | — | 100% | open 8B annotator alternative (CC-BY-4.0 data) |

Recipe elements worth copying:

1. **Frame rate**: 1 fps for event proposals; 2–4 fps around candidate boundaries (Qwen3-VL uses 4 fps for Charades-STA). Dense sampling matters more than model size on long video (LongEgoRefer's 64→500 frame effect).
2. **Chunking**: fixed 60–120 s windows with ~10 s overlap, plus a running summary of the previous window as context (LLaVA-Video's recurrence, ShareGPT4Video's differential captioning). For static cameras, gate windows by motion energy: quiet windows get a cheap low-res pass whose only job is to confirm "no interaction".
3. **Timestamp prompting**: request `MM:SS` start/end per event with the event verb, object noun phrase, hand-contact flag and state (lifted/supported), as strict JSON; then snap boundaries with a local signal (hand-contact detector, mask motion) because the VLM is second-granular.
4. **Consistency checks**: (a) sample twice and keep agreements; (b) cross-model agreement (Gemini vs Qwen3-VL) as a disagreement flag; (c) re-query the model on the localized clip only ("zoom-in" verification); (d) exhaustivity verification as in SAM 3's data engine (a separate pass asking "any interaction missed in this window?").
5. **Human adjudication rates observed**: SAM 2 phase 3 still needed manual edits on 19.04% of frames; SAM 3's fine-tuned Llama 3.2 verifiers matched humans and doubled throughput (5× faster on negatives, 36% faster on positives), with humans reserved for failure cases ([SAM 3 §A.4](https://arxiv.org/abs/2511.16719)); TimeLens found roughly a third of benchmark labels wrong. Plan to have humans touch 100% of disagreement/low-confidence windows (expect 20–35%) and a 10% random audit of agreed windows.
6. **Specialist labelers for contact**: the 100DOH hand-object detector predicts hand box, side, contact state and the contacted object's box, trained on a 100K-frame dataset drawn from 131 days of video ([100DOH](https://arxiv.org/abs/2006.06669)); use it to snap boundaries and to supply the hand-object-contact channel the VLM is bad at.

## 2. Semi-automatic tracking labels (boxes, masks, identities)

Models:

- **SAM 2** ([paper](https://arxiv.org/abs/2408.00714), [repo](https://github.com/facebookresearch/sam2), Apache-2.0): points/boxes on any frame, propagation, independent per-object inference; 39.5 FPS (large) to 91.2 FPS (tiny) on A100. Data engine: per-frame annotation time fell from 37.8 s (SAM per frame) to 7.4 s (SAM+SAM 2 mask) to 4.5 s (SAM 2 in the loop, 8.4×); videos annotated at 6 FPS; separate annotators verify each masklet "satisfactory/unsatisfactory"; 19.04% of frames edited in phase 3.
- **SAM 3** ([paper](https://arxiv.org/abs/2511.16719), [repo](https://github.com/facebookresearch/sam3), "SAM License", commercial use permitted with trade-control/military restrictions and a research-acknowledgment clause): text/exemplar prompts return masks and IDs for all instances; video tracking with masklets, detection-score decay and re-prompting; 848M params; ~30 ms/image on H200. Video data engine: a positive video–noun-phrase pair took 2,307 s of human time from scratch vs 3,221 s starting from proposals (which found more masklets, 2.76 vs 2.52 per pair), i.e. roughly 15–20 human-minutes per masklet on SA-V-style clips. In CVAT, SAM 3 is image-only so far (Nov 2025), with video tracking announced as a later phase ([CVAT changelog](https://www.cvat.ai/resources/changelog/sam-3-image-segmentation)).
- **Cutie** ([paper](https://arxiv.org/abs/2310.12982), [repo](https://github.com/hkchengrex/Cutie), MIT): +8.7 J&F over XMem on MOSE at similar speed; GUI with click-based annotation, propagation and XMem++ "permanent memory". **XMem** (MIT) and **XMem++** ([paper](https://arxiv.org/abs/2307.15958)) add attention-based suggestion of the next frame to annotate, keeping annotated frames to a small fraction.

Tool integrations:

| Tool | SAM 2 / SAM 3 video support | Constraints |
|---|---|---|
| CVAT | SAM 2 tracker: Nuclio serverless (Enterprise) or AI-agent on your own GPU (Online + Enterprise); propagate masks/polygons to a target frame, all shapes at once ([docs](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/), [blog](https://www.cvat.ai/resources/blog/sam2-ai-agent-tracking)) | one agent per function; state lost on crash; no skeleton tracking; GPU strongly recommended |
| Label Studio | SAM 2 video ML backend with `VideoRectangle` bbox prompts ([guide](https://labelstud.io/guide/ml_tutorials/segment_anything_2_video)) | single object per video; boxes only, no video masks; GPU only; no Docker |
| Roboflow | Smart Polygon backed by SAM 2, Grounded SAM 2 auto-labeling, Repeat Previous; Inference server hosts SAM/SAM 2/SAM 3 ([launch post](https://blog.roboflow.com/sam-2-roboflow/), [video guide](https://blog.roboflow.com/sam-2-video-segmentation/)) | video propagation via API, not a first-class UI tracker |
| V7 Darwin | Auto-Annotate + Interpolate/Rerun for video ([docs](https://docs.v7labs.com/docs/auto-annotate)) | underlying model unspecified |
| Encord | not verified in this pass (docs URLs 404) | — |

Human time per minute of video (derived): SA-V-quality masklets cost 4.5 s per annotated frame at 6 FPS, i.e. about 27 human-seconds per video-second per object, or ~27 min per footage-minute per object, including verification. For our case (static camera, most objects at rest most of the time) a box-plus-mask scheme with keyframes only when an object moves, SAM 2/Cutie propagation and ~20% correction should land at roughly **2–5 human-minutes per active footage-minute for a 3–6 object scene**, and near zero for quiet minutes if identity carries over (estimate, no primary source).

## 3. Synthetic data for hand-object interactions in homes

| Platform | Humans / hands manipulating objects? | GT exported | License / commercial cleanliness |
|---|---|---|---|
| [BEHAVIOR-1K / OmniGibson](https://behavior.stanford.edu/) | robot-centric (1,000 tasks, 50 scenes, 10K+ objects); no human-avatar rendering documented | RGB, seg, depth via Isaac Sim | Code MIT ([repo](https://github.com/StanfordVL/BEHAVIOR-1K)); requires Isaac Sim EULA and a separate dataset TOS accepted at install (`--accept-dataset-tos`, [install page](https://behavior.stanford.edu/getting_started/installation.html)); asset terms not verified, treat as not clean until read |
| [AI2-THOR / ManipulaTHOR](https://github.com/allenai/ai2thor) | robot agents only (LoCoBot, Kinova-style arm, drone); 200+ scenes | RGB, instance/semantic seg, depth, normals, third-person cameras | Apache-2.0 |
| [Habitat 3.0](https://arxiv.org/abs/2310.13724) | SMPL-X humanoids; AMASS walking clip; VPoser reach poses; pick/place by kinematically attaching the object to the hand (no physical grasp); 188 FPS humanoid, 1,190 FPS humanoid+robot | RGB/depth/seg via Habitat-Sim; HSSD scenes | SMPL-X is non-commercial (see below) |
| [Infinigen Indoors](https://infinigen.org/) | no humans | depth, normals, panoptic seg, flow, occlusion boundaries | BSD-3 |
| [Isaac Sim Replicator Agent](https://docs.isaacsim.omniverse.nvidia.com/latest/action_and_event_data_generation/tutorial_replicator_agent.html) | characters with probabilistic behavior routines; no hand-object manipulation documented | RGB, camera params, boxes + skeleton JSON, per-actor action data, optional seg/normals/motion vectors | Isaac Sim EULA; character-asset terms not stated |
| [Unity Perception](https://github.com/Unity-Technologies/com.unity.perception) | PeopleSansPeople human generator | 2D/3D boxes, seg, keypoints | Apache-2.0 but **discontinued** |
| [BlenderProc](https://github.com/DLR-RM/BlenderProc) + SMPL-X | no built-in human/hand support | RGB, depth, normals, seg, COCO/BOP | GPL-3.0 tooling (outputs unaffected) |
| [BEDLAM](https://bedlam.is.tue.mpg.de/) | SMPL-X bodies, clothing, hair; no object interactions | 3D bodies, masks | research license; commercial via MPI |
| SMPL-X | — | — | [license](https://smpl-x.is.tue.mpg.de/modellicense.html): non-commercial only, no commercial model training; commercial license via Meshcapade |
| [Wan 2.2](https://github.com/Wan-Video/Wan2.2) | generative (T2V/I2V, 720p, 5 s at 24 fps; TI2V-5B runs on an RTX 4090) | none (labels must come from elsewhere) | Apache-2.0 |
| [Cosmos-Transfer1](https://github.com/nvidia-cosmos/cosmos-transfer1) | sim-to-photoreal transfer from seg/depth/edge/blur/keypoint controls; documented robotics augmentation workflow | inherits labels from the sim render | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/): commercial use and outputs allowed, attribution required |
| Veo 3.1 | generative | none | $0.05–0.60 per generated second ([pricing](https://ai.google.dev/gemini-api/docs/pricing)), i.e. $180–2,160 per generated hour |

Reported sim-to-real results for action / hand-object recognition:

- **Sims4Action → Toyota Smarthome** (10 ADL classes): in-domain I3D 89.1% top-1, but gaming→real 22.8% (I3D) / 34.1% (S3D), balanced 23%, chance 10% ([Sims4Action](https://arxiv.org/abs/2107.05617)); the follow-up frames it as a ">60% drop in accuracy" from the synthetic-to-real shift ([Roitberg et al. 2022](https://arxiv.org/abs/2208.01910)).
- **ElderSim / KIST SynADL** (UE4, mocap-driven elderly ADL, robot and surveillance viewpoints): augmenting NTU with synthetic data for cross-dataset testing on ETRI-Activity3D: Glimpse Clouds 39.99→54.79 (+14.8), ST-GCN +2.8, VA-CNN +3.3 ([ElderSim](https://arxiv.org/abs/2010.14742)).
- **RoCoG-v2** (Unity gestures, 107K synthetic vs 482 real videos): synthetic-only vs real-only gap is 2.7 pts from the ground view (80.3 vs 83.0) but 35.8 pts from the air view (34.5 vs 70.3) ([RoCoG-v2](https://arxiv.org/abs/2303.10280)); viewpoint mismatch, not rendering, is the main gap.
- **SURREACT**: synthetic SMPL humans rendered from monocular 3D pose improve unseen-viewpoint recognition on NTU RGB+D and UESTC (abstract only; [paper](https://arxiv.org/abs/1912.04070)).
- **Hand-object specific**: PAM, a pose-appearance-motion engine for sim-to-real HOI video, generated 3,400 videos (207K frames) and let models match the full-real baseline with 50% real data ([arXiv 2603.22193](https://arxiv.org/abs/2603.22193)). Generative-video augmentation gains are so far reported mostly in surgical video (SAW: clipping F1 20.9→43.1, [arXiv 2603.13024](https://arxiv.org/abs/2603.13024); [Mission Balance](https://arxiv.org/abs/2505.09858)).

Takeaway: no commercially clean simulator renders physically plausible hands manipulating household objects out of the box; Habitat's hand-object contact is kinematic attachment, Isaac's agents do not manipulate, and SMPL-X bodies need a Meshcapade license. Use synthetic data for what it is good at (viewpoint, lighting, occlusion, object masks, class balance, negatives) and keep the contact and lifted-vs-supported channels grounded in real footage. If you do use sim, Cosmos-Transfer on sim renders is the license-clean way to close appearance gap while inheriting labels.

## 4. Data collection protocol for a small team

### 4.1 What the generalization evidence says

| Dataset | Diversity | Cross-condition result |
|---|---|---|
| [Toyota Smarthome](https://openaccess.thecvf.com/content_ICCV_2019/papers/Das_Toyota_Smarthome_Real-World_Activities_of_Daily_Living_ICCV_2019_paper.pdf) | 1 apartment, 18 seniors (60–80), 7 Kinect cameras, 16,115 clips, 31 classes; faces blurred; academic-only, GDPR-compliant ([project page](https://project.inria.fr/toyotasmarthome/)) | CS: 11 train / 7 test subjects; CV1: train camera 1, test camera 2 (same dining room); CV2: train cameras 1,3,4,6,7, test camera 2. I3D mean per-class accuracy 53.4 (CS) → 34.9 (CV1) → 45.1 (CV2); Separable STA 54.2 / 35.2 / 50.3 |
| [TSU](https://arxiv.org/abs/2010.14982) | same apartment; 536 untrimmed videos, 21 min avg, 51 classes, >1,000 h recorded; annotation took >6 months with 5 annotators; inter-annotator precision 96.8% on 50 double-annotated videos | CV: train cameras 1,3,4,6,7, test 2,5. Per-frame mAP CS→CV: I3D+TGM 26.7→13.4; I3D+SD-TCN 29.2→18.3; AGNet (RGB+pose) 33.2→23.2 |
| [HOMAGE](https://arxiv.org/abs/2105.05226) | 27 participants, **2 houses**, ego + ≥1 third-person views, 1,752 sequences, 75 activities, 453 atomic actions, 497,534 boxes | no cross-house split reported: a two-home dataset cannot measure home generalization |
| [Charades](https://arxiv.org/abs/1604.01753) | 267 people on three continents filming in their own homes; 9,848 videos, 66,500 temporal intervals, 157 classes | the collection model (many homes, many people, short scripted clips) is the one that generalized |
| [EPIC-KITCHENS-100](https://arxiv.org/abs/2006.13256) | 45 kitchens, 100 h | "test of time": models trained on 2018 footage must adapt to 2020 footage of the same kitchens |
| [ARGO1M](https://arxiv.org/abs/2306.08713) | 1.1M Ego4D clips, 10 scenarios × 13 locations | leave-one-scenario-and-location-out; recognition models "struggle to generalise" |
| [RoCoG-v2](https://arxiv.org/abs/2303.10280) | ground vs air viewpoints | 2.7 vs 35.8-pt gaps: viewpoint is the dominant shift |

Rule of thumb from these: new camera placement in the same room costs 10–20 accuracy points; more homes and more placements buy more than more hours from the same placement; a held-out home is the only honest test.

### 4.2 Recommended quantities (my plan; see section 6 for the full data plan)

- Homes: 10–12 for training/validation, 3–4 frozen for acceptance, none overlapping in people or rooms.
- Placements: 2 per room type (corner-high and shelf-height), kitchen + living room minimum; log height/angle so the acceptance set can stress unseen placements.
- Lighting: daylight, evening lamp, night IR/low light; each ≥20% of labeled hours.
- Negatives: ≥40% of labeled footage must be quiet intervals or non-pick-up reaches/repositioning; script ~10% of sessions to guarantee coverage of the confusable cases (reach without pickup, slide without lift, occlusion).
- Hours: 300–500 h raw across homes; 60–100 h exhaustively labeled; every raw hour gets at least the cheap "quiet or not" VLM pass so false-alarm rates can be measured on the long tail.

### 4.3 Consent and privacy practices from the source datasets

- Ego4D ([paper](https://arxiv.org/abs/2110.07058)): 931 wearers, 74 locations, 9 countries; per-site IRB; informed consent with the right to withdraw and to review and redact one's own video; de-identification of bystander faces, plates and screens (36% of Bristol videos needed it), using commercial tools (brighter.ai, Secure Redact); instructions to avoid sensitive areas and respect others' private spaces.
- Toyota Smarthome / TSU: participants knew they were recorded but not the study purpose; no script; faces blurred; academic-only license with a GDPR compliance statement.
- For this project: written consent per household member and for regular visitors; no bathrooms/bedrooms; participant-controlled deletion window before any clip leaves the device; on-device face/screen blurring before upload; minors excluded from labeled data; retain only derived labels plus consented clips; log every export.

### 4.4 On-device active learning that reduces labeling

- Uncertainty + diversity feeds: Video Annotator's four feeds (top positives, top negatives, borderline |score−0.5|, random) with ≥100 pos + 100 neg per concept gave a median +8.3 AP over the best baseline ([arXiv 2402.06560](https://arxiv.org/abs/2402.06560)).
- Temporal-localization-specific scoring: temporal proposal entropy + temporal context inconsistency in a train-query-annotate-append loop ([arXiv 2208.14856](https://arxiv.org/abs/2208.14856)); boundary-centric clip-budgeted selection that ranks videos by uncertainty then picks boundary frames ([arXiv 2604.15173](https://arxiv.org/abs/2604.15173)).
- Hard-negative mining: SA-Co/HQ is 88.5% negatives, generated by a Llama pipeline proposing phrases adversarial to the current model, and AI verifiers made negative verification 5× cheaper ([SAM 3](https://arxiv.org/abs/2511.16719)). Mirror this: have the deployed detector export its confident-but-wrong quiet-interval firings (after consent) as the negative queue.
- Loop: on-device detector + tracker log (a) low-margin event scores, (b) detector/tracker-state disagreements (e.g., "pick-up" fired but no mask displacement), (c) event bursts in quiet hours; user approves upload of blurred crops; cloud VLM + human adjudicate; retrain; ship. Keep raw media on device throughout.

## 5. Evaluation

- **Freeze the acceptance set**: 3–4 homes never used for training or tuning, whole homes not clips, ≥24 h each including night and lamp lighting, double-annotated with adjudication (TSU reached 96.8% inter-annotator precision with 5 annotators), versioned and never rewritten (your repo already follows the "historical reports never rewritten" rule). Report per home, per placement, per lighting.
- **Temporal thresholds in use**: ActivityNet mAP@tIoU 0.5:0.05:0.95, THUMOS14 0.3–0.7 ([ActivityNet challenge summary](https://arxiv.org/abs/1808.03766)); Charades/TSU per-frame mAP plus event-based mAP (TSU); grounding benchmarks use R1@0.3/0.5/0.7 and mIoU (TimeLens). For second-level VLM labels, report R1@0.5 plus boundary error in seconds; hold R1@0.7 for the snapped labels.
- **Deployment-style metric**: ActEV/TRECVID reports Pmiss vs rate of false alarms per minute of video and the normalized partial AUDC over time-based false alarms; the standard operating point is Pmiss at RFA = 0.15 per minute (9 false alarms per hour), and the best 2019 system scored nAUDC 0.484 ([TRECVID 2019](https://arxiv.org/abs/2009.09984), [ActEV](https://actev.nist.gov/)). That operating point is a research convention far looser than a home user tolerates; adopt the metric, not the threshold.
- **False alarms in real deployments**: 13 published accelerometer fall algorithms had sensitivity 57%±27% and specificity 83%±30% on 29 real falls and produced 3–85 false alarms per day per person in 24-h monitoring, versus near-perfect lab numbers ([Bagalà et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0037062)). A scoping review of real-world fall-detection evaluation recommends sensitivity, precision and F-measure and warns against metrics needing a "non-fall" denominator ([Sensors 2018](https://doi.org/10.3390/s18072060)); a 2025 systematic review of 80 AAL/smart-home studies finds deep learning best and wearables worst but little real-world validation ([Sensors 2025](https://doi.org/10.3390/s25216540)); a video-based validation in assisted living reports 82% sensitivity / 93.2% specificity on 100 clips ([PMC12761293](https://pmc.ncbi.nlm.nih.gov/articles/PMC12761293/)); a 23-home ambient-sensing deployment reduced false alarms by SMS well-being confirmation before escalation, and 24/25 families preferred escalation to a circle of contacts over automatic 911 ([PMC12740839](https://pmc.ncbi.nlm.nih.gov/articles/PMC12740839/)). Reviews since 2013 list user acceptance under real-life false alarms as the unresolved barrier ([Igual et al. 2013](https://doi.org/10.1186/1475-925X-12-66), [JMIR Aging 2021](https://doi.org/10.2196/29744)).
- **Published user-acceptable rates**: none found with a hard number. Recommended targets (mine): report false events per quiet hour and per home-day; ship tiers at ≤1 false notification per home-day for informational events and ≤1 per home-week for anything that escalates to a person, with confirmation gating for the latter.

## 6. Recommended labeling pipeline and data plan

### 6.1 Pipeline (tools, models, human-in-the-loop points, cost)

| Stage | What | Tools/models | Human touch |
|---|---|---|---|
| 0. Capture + preprocess (on device) | 1080p→720p, 1 fps keyframes + motion-triggered 5–10 fps bursts; motion energy; face/screen blur; 60–120 s windows with 10 s overlap | ffmpeg, on-device blur | consent gate; participant review/delete window |
| 1. Event proposals | Gemini 3.1 Pro (batch) at 1 fps high-res on active windows, Gemini 3.x Flash low-res on quiet windows; strict JSON with `MM:SS` start/end, verb, object phrase, contact flag, lifted/supported, confidence; second sample + Qwen3-VL-32B for agreement | Gemini Batch API, vLLM Qwen3-VL | none |
| 2. Boundary snapping | 100DOH hand-contact detector at 5–10 fps around proposed boundaries; snap start/end to first/last contact frame | 100DOH, motion energy | none |
| 3. Objects, masks, identities | SAM 3 text prompts from the object vocabulary → masklets; propagate/repair with SAM 2 or Cutie; SAM 3-style hard-negative phrases | SAM 3, SAM 2, Cutie GUI in CVAT (AI-agent tracker) | correct ~20% of frames on moving objects |
| 4. State channel | lifted vs supported from masklet displacement + contact; VLM confirm on crops only when ambiguous | rules + Gemini on crops | none |
| 5. Adjudication | 100% review of windows with model disagreement, low confidence, or exhaustivity-check hits (expect 20–35%); 10% random audit of agreed windows; 100% double annotation of the acceptance set | CVAT/Label Studio timeline + mask editor | main human cost |
| 6. Distill and loop | train the on-device temporal detector + tracker on adjudicated labels; active-learning feeds (uncertainty, disagreement, hard negatives) select the next batch | your training stack | label only the selected batch |

Cost per footage-hour (derived/estimated): VLM passes $0.5–3 with Gemini batch (or $4–9 with GPT-5/Claude Sonnet at 1 fps); SAM/100DOH GPU time ≈$0.5–1; humans ≈5 min per quiet hour and 30–90 min per active hour (≈$2–40 at $25/h loaded). For home footage that is ~20–30% active, expect **≈$10–20 per footage-hour all-in**, dominated by human adjudication.

### 6.2 Data plan with quantities

| Phase | Homes | Recording | Labeled | Purpose |
|---|---|---|---|---|
| A. Pilot | 2 (team members) | 2 rooms × 2 placements, 12 h/day × 7 days ≈ 170 h raw per home | 10 h per home exhaustively (5 active, 5 quiet), double-annotated to calibrate the pipeline's error rate | validate prompts, snapping, human-time budget |
| B. Breadth | 8–12 consented households | 1–2 weeks each, 2 placements per room; day/lamp/night | 60–100 h selected by active learning + all "quiet" hours get the cheap pass; ≥40% negatives; ~10% scripted confusables | training set with home/placement/lighting diversity |
| C. Acceptance | 3–4 households never used above | ≥24 h each, all lighting, at least one unseen placement | 100% exhaustive, double-annotated, frozen and versioned | Pmiss / false alarms per hour per home; tIoU R1@0.5 and boundary error |
| D. Synthetic (optional) | — | 20–50 h Isaac/Habitat renders for occlusion, viewpoint and rare objects; Cosmos-Transfer appearance augmentation | labels inherited from sim | pretraining/augmentation only, with ablations; avoid SMPL-X unless commercially licensed |
| E. Field loop | all consenting deployments | detector-selected clips (low margin, disagreements, quiet-hour firings) | ~5–10 h per month adjudicated | hard-negative mining and drift tracking against the frozen acceptance set |

Totals: ~1,500–2,500 h raw, 150–250 h with exhaustive labels (including the acceptance set), roughly 200–350 human-hours of annotation plus a few hundred dollars of API/GPU spend per 100 footage-hours.

### Not verified in this pass

Encord's SAM 2 video features (docs URLs unavailable), Mixamo's commercial terms (Adobe pages blocked), the BEHAVIOR-1K asset license text (only the install-time TOS flag was confirmed), and SURREACT's exact per-protocol numbers (PDF text extraction failed). The web-search budget for this session was exhausted midway; the remaining sources were fetched directly by URL.

I can publish this as a shareable page if it is meant for the wider team.
