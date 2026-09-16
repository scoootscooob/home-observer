<!-- research note produced 2026-09-16 by a web-research agent; fetch budget 30 calls, all spent; facts marked (P) primary = fetched from the official page/paper this session, (S) secondary = search snippet, mirror, prior note, or prior knowledge not re-verified, (C) computed here from primary numbers, (I) inferred; verify before relying on any single number. -->

# Public footage for teacher labeling and long-stream evaluation — survey, 2026-09

Scope: footage a frontier teacher can label tick by tick under the `silent / context / act` contract
(`docs/teacher-distillation.md`), plus long-stream benchmarks for evaluating the student. Companion to
`docs/research/datasets-2026-09.md`, which already covers Ego4D, Ego-Exo4D, EPIC-KITCHENS-100, HoloAssist,
RH20T, DROID, OakInk2 and Toyota Smarthome; those are only cross-referenced here, not re-surveyed.

Production geometry is fixed indoor cameras; the training signal is the teacher's per-tick decision, so the
questions asked of every source are: how many hours, from what geometry, how dense and how precise the
existing timestamps are (they audit the teacher, they are not the target), whether quiet time is certified
negative, and what the license allows now and later.

## 0. Method and budget

30 web calls on 2026-09-16 (24 fetches, 6 searches; Appendix A holds the per-batch log and verbatim
quotes). Primary pages reached: EgoLife (arXiv full text + HF card), HOMAGE (full text), BEHAVIOR challenge
index + dataset docs + both HF cards, MultiTHUMOS, StreamingBench, OVO-Bench, SmartHome-Bench (full text),
EgoStream (abstract), EgoMonth (full text), Vinci2/EgoServe (full text), VirtualHome (GitHub), DESED (site +
GitHub), AudioSet download page, DCASE 2018 Task 5 (SINS), CASTLE site, acon96 HF card. Not reached:
virtual-home.org (TLS certificate mismatch), AudioSet strong-label page, SINS EULA, EgoStream project page.
Charades / Charades-Ego numbers are taken from `datasets-2026-09.md` (fetched there, not re-fetched here).
Nothing was downloaded.

## 1. Headline findings

1. **Two public sources have real fixed cameras in a real home running for days: CASTLE 2024 and EgoLife.**
   CASTLE has 5 static UHD 50 fps streams over 4 days (≈200 h of static footage, computed from 600 h / 15
   streams) but no labels at all, so blind teacher labeling is the only route, and it is CC BY-NC-SA 4.0.
   EgoLife has 15 fixed GoPros in the common areas synchronized with 300 h of Aria ego video, near-continuous
   narration (361K phrases averaging 2.65 s, i.e. ≈1,200 per hour) for blind-mode audit, and an MIT license
   on the Hugging Face card — but the speech and annotations are Chinese, the exo hours are not stated, and
   whether the exo streams are in the 512 GB HF bundle is unverified.
2. **HOMAGE is the only set where fixed video, frame-precise atomic actions, scene graphs, and real ambient
   sensor streams coincide** (25.4 h, 2 houses, ≈800 atomic events per hour, 497,534 boxes and 583,481
   relationships on 3–5 frames per action, PIR / light / magnetometer / thermal / gas sensors at 2–10 Hz).
   It is the natural audit set for teacher precision and the seed for synthesized device-state text, but it
   is small and non-commercial.
3. **BEHAVIOR-1K is the only large source of exact object-state timestamps**: 2025 demos 10,000 episodes /
   ≈1,103 h (computed from 119,094,660 frames at 30 fps), 2026 demos 20,000 episodes / 1,950 h / 100 tasks
   / 7 scenes, both MIT on Hugging Face, with head + two wrist RGB-D streams and skill/primitive segments
   that name the object. Replay through `OmniGibson/scripts/learning/replay_obs.py` is documented "to
   replay trajectories and collect additional visual observations", but it needs OmniGibson + Isaac Sim (RTX
   GPU) + the BEHAVIOR asset bundle whose EULA is non-commercial, and an open issue reports that 2026 demos
   replayed on the newer engine diverge (collect on OG 3.7 / Isaac 4.5). Robot embodiment: no humans.
4. **The benchmark that matches the contract best is EgoServe (Vinci2, 2026-07)**: 3,000+ timed
   "should the assistant speak now" instances over ≈128 h of EgoLife + HoloAssist + CaptainCook4D with
   per-category precision/recall/F1 and tolerance windows of 60 / 25 / 10 s. OVO-Bench (forward active
   responding) and StreamingBench (proactive output) test the same "wait, then speak" skill on web video.
   EgoLifeQA, EgoStream and EgoMonth test long-horizon memory over the journal rather than when to speak.
5. **Contamination is structural, not incidental**: EgoServe reuses EgoLife, HoloAssist validation and
   CaptainCook4D validation/test; EgoLifeQA spans all six EgoLife participants and > 24 h certificates; the
   15 exo cameras see everyone, so a participant-disjoint split of EgoLife is impossible. Decide now whether
   EgoLife is a training source or an evaluation source; it cannot cleanly be both.
6. **For a product, nothing here replaces own consented fixed-camera recordings.** CASTLE, HOMAGE, Charades,
   OVO-Bench, EgoMonth and the BEHAVIOR asset bundle are research-only. Commercially compatible pieces:
   EgoLife (MIT per HF card, consent terms unverified), BEHAVIOR demo trajectories as released (MIT card,
   asset-derived renders carry residual risk), VirtualHome code (MIT), StreamingBench (MIT), acon96
   Home-Assistant-Requests (MIT), AudioSet labels (CC BY 4.0), DESED code (MIT, audio per-file), plus
   HoloAssist and CaptainCook4D from the earlier note.

## 2. Table — footage sources and benchmarks

Legend: geometry = fixed exo / ego / robot-sim / handheld; density = timestamped events per hour (stated or
(C) computed); precision = timestamp resolution; quiet = is unlabeled time certified negative; license:
RO = research-only, COM = permits commercial use, COND = bespoke; size = download size.

### 2A. Multi-camera and fixed-camera home footage (real humans)

| Source | Geometry | Hours | Timed annotation density | Precision | Quiet / negatives | License | Download, size |
|---|---|---|---|---|---|---|---|
| **EgoLife** (CVPR 2025) [paper](https://arxiv.org/html/2503.03803) · [HF](https://huggingface.co/datasets/lmms-lab/EgoLife) · [site](https://egolife-ai.github.io/) | 15 fixed GoPros in common areas (P) + Meta Aria ego on 6 people, synchronized (P) | 300 h ego (P); exo hours not stated (gap); 6 people × 1 week, ≈8 h/day (P) | 361K phrases avg 2.65 s (P) ≈ 1,200/h (C), merged into 25K captions ≈ 83/h (C); 50 h of reviewed transcripts (P); speech and annotations "primarily in Chinese" with English translation (P) | phrase-level, ≈2–3 s spans (P); boundary precision not stated | narration is near-continuous (≈266 h covered, C), so silence must be derived | HF card "License: mit" (P); consent/usage terms on project site not fetched (gap) | HF `lmms-lab/EgoLife`, 512 GB, 32,001 rows, "Data cleaning, stay tuned!" (P); exo inclusion unverified |
| **CASTLE 2024** (2025) [site](https://castle-dataset.github.io/) · [arXiv](https://arxiv.org/html/2503.17116) | 5 static + 10 ego streams, UHD 50 fps with audio (P), one vacation home | > 600 h total (P) ≈ 40 h per stream, ≈ 200 h static (C, equal-split assumption) ; 4 days, 10 participants (P); IMU, GPS, biometrics (P) | none; Whisper transcripts only (S, prior note) | n/a | continuous days, so quiet periods exist but are unlabeled | CC BY-NC-SA 4.0 (P) → RO | HF `CASTLE-Dataset/CASTLE2024`, 8.22 TB (P) |
| **HOMAGE** (CVPR 2021) [paper](https://ar5iv.labs.arxiv.org/html/2105.05226) · [site](https://homeactiongenome.org/) | ≥1 fixed third-person view per sequence, "more than 3 views" on average (P) + head ego; sensors "attached to several locations in the room" (P) | 25.4 h (P; site says ~30 h, S); 1.75K sequences, 5,700 videos, 27 people, 2 houses (P) | 453 atomic-action classes, all instances given start/end frames (P); 20,039 train instances (S) ≈ 800/h (C); scene graphs on 3 or 5 frames per action: 497,534 boxes, 583,481 relationships, 86 object + 29 relationship classes (P); 12 modalities incl. PIR 2 Hz, light 2 Hz, color 10 Hz, BME680 gas/humidity/pressure/temp 5 Hz, magnetometer 10 Hz, thermal 10 Hz, mic 48 kHz (P) | frame (30 fps) for actions; 60 % of atomic actions < 2 s, 80 % < 5 s (P) | activity-centric sequences, little idle time; not certified | non-commercial competition terms (S, prior note) → RO | site agreement; sensor modalities "later" (S, prior note) |
| **Toyota Smarthome Untrimmed** — see `datasets-2026-09.md` | 7 fixed | ≈190 h | dense ADL classes | frame | yes (dense) | RO, commercial by email | request form |

### 2B. Third-person home clips (handheld / propped / consumer cameras)

| Source | Geometry | Hours | Timed annotation density | Precision | Quiet / negatives | License | Download, size |
|---|---|---|---|---|---|---|---|
| **Charades** (2016) [page](https://prior.allenai.org/projects/charades) (S, prior note) | propped/handheld phones in 267 homes | ≈82 h, 9,848 clips ≈ 30 s | 66,500 action intervals ≈ 810/h (C); 157 classes; 46 object classes at video level | seconds | scripted acting, little idle | academic / non-profit / government only → RO | open S3 |
| **Charades-Ego** (2018) (S, prior note) | paired third-person + ego | 7,860 videos | 68,536 intervals | seconds | as above | same → RO | open |
| **SmartHome-Bench** (CVPRW 2025) [paper](https://arxiv.org/html/2506.12992) · [code](https://github.com/Xinyi-0724/SmartHome-Bench-LLM) | fixed indoor and outdoor consumer smart-home cameras, YouTube-sourced (P) | 1,203 clips, avg ≈ 20 s, most < 80 s (P) ≈ 6.7 h (C) | per-clip only: category, normal / abnormal / vague-abnormal tag, ≤200-word description, ≤100-word rationale (P); no onset timestamps (P) | none | balanced normal vs abnormal (P), 91 vague (P) | not stated (P); YouTube-sourced → assume RO | GitHub |
| **MultiTHUMOS** (2015) [page](https://ai.stanford.edu/~syyeung/everymoment.html) | sports web video (THUMOS'14), not homes | 30 h, 400 videos (P) | 38,690 annotations, 65 classes, 1.5 labels per frame, 10.5 classes per video (P) ≈ 1,290/h (C) | frame | dense multilabel incl. background | not stated (P) | zip on page (2017-07-19) |

### 2C. Simulators with object-state ground truth

| Source | Geometry | Hours | Timed annotation density | Precision | Quiet / negatives | License | Download, size |
|---|---|---|---|---|---|---|---|
| **BEHAVIOR-1K 2025 challenge demos** [HF](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) | robot (R1 family) head 720×720 + 2 wrist 480×480, RGB + depth + instance-seg, 30 fps (P) | 10,000 episodes, 119,094,660 frames (P) = 1,103 h (C); 50 tasks (P) | skill / subtask annotations (S, challenge text); `observation.task_info`, 256-d state, camera poses (P) | sim step (33 ms) | task-driven, no idle | HF card "mit" (P); assets under non-commercial BEHAVIOR EULA (S, prior note) | HF, 1.95 TB, LeRobot v2.1 parquet (P) |
| **BEHAVIOR-1K 2026 challenge demos** [docs](https://behavior.stanford.edu/challenge/dataset.html) · [HF](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos) | R1Pro robot (S), same three cameras; RGB, depth 0–10 m, proprio, actions, camera poses (P) | 20,000 demos, 1,950 h, 100 tasks, 7 scenes (P); avg trajectory 351.54 s (P) | 270,600 skill segments, 31 unique skills (P) ≈ 139/h (C); `skill_annotation` + `primitive_annotation` rows with `skill_description` and `object_id` (P) | sim step | task-driven, no idle | HF card MIT (P); asset EULA non-commercial (S) | HF `2026-challenge-demos` 3.27 TB + `2026-challenge-rawdata` 1.44 TB HDF5, LeRobot V3 (P) |
| replay / camera control | `OmniGibson/scripts/learning/replay_obs.py` "to replay trajectories and collect additional visual observations" (P); `--demo_id`, `--action-only` (S); adding arbitrary fixed room cameras is possible through OmniGibson's sensor API (S, not verified on the docs page); object states (open, toggled_on, inside, ontop …) are BDDL predicates queryable at replay time (S); per-frame object-state export in the released parquet: not mentioned (gap) | | | | | needs OmniGibson + Isaac Sim, RTX GPU (S); headless mode exists in OmniGibson (S); replay divergence issue #2344: collect on OG 3.7 / Isaac 4.5, eval stack OG 3.9 / Isaac 5.1 diverges (P, issue title) | | |
| **VirtualHome** [GitHub](https://github.com/xavierpuigf/virtualhome) | Unity apartments, animated human characters executing programs, selectable cameras (P) | unlimited generation; 7 apartments, ≈2,800 activity programs (S) | "time-stamped actions, instance/semantic segmentation, and optical flow and depth" + environment graphs (P) | program step | scripted, no idle unless authored | MIT code (P); Unity asset terms not checked (gap) | executable v2.3.0 Linux/macOS/Windows (P) |

### 2D. Audio sets for domestic sound events

| Source | Geometry | Hours | Timed annotation density | Precision | Quiet / negatives | License | Download, size |
|---|---|---|---|---|---|---|---|
| **DESED** [site](https://project.inria.fr/desed/) · [GitHub](https://github.com/turpaultn/DESED) | 10-s AudioSet clips + synthetic soundscapes; 10 classes: alarm/bell, blender, cat, dog, dishes, electric shaver/toothbrush, frying, running water, speech, vacuum cleaner (P) | real: weak 1,578 + unlabeled 14,412 + validation-strong 1,168 + public eval 692 clips (P) = 17,850 clips ≈ 49.6 h (C); strong-labeled ≈ 5.2 h (C); soundbank 2,060 SINS backgrounds + 1,009 Freesound foregrounds (P) | strong labels = onset/offset per event (P) | seconds (decimal precision not stated) | synthetic sets can be generated with silence | code MIT; audio "license file at their root for the attribution of each file" (P) → per-file Freesound / YouTube terms | Zenodo 3702397 (synthetic), 3588172 (public eval) (P) |
| **SINS / DCASE 2018 Task 5** [task page](https://dcase.community/challenge2018/task-monitoring-domestic-activities) | 13 four-mic nodes in a vacation home, one person, one week (S); DCASE subset: living room + kitchen nodes, 16 kHz (P) | dev ≈ 200 h from 4 nodes = 72,984 × 10-s segments (P) ≈ 50 h wall-clock (C); eval 72,972 files from 7 nodes (P) | 9 activity classes per 10-s segment, one activity per segment (P): absence 18,860, cooking 5,124, dishwashing 1,424, eating 2,308, other 2,060, social 4,944, vacuum 972, TV 18,648, working 18,644 (P) | 10 s | **yes**: "Absence" ≈ 52 node-hours (C) | EULA.pdf on Zenodo, variant not read (gap) | Zenodo 1247102 (42.6 GB dl / 87.0 GB), 1964758 (42.2 GB / 87.0 GB) (P) |
| **AudioSet** (domestic subtree) [download](https://research.google.com/audioset/download.html) | YouTube 10-s clips | 2,084,544 segments, 527 classes (P); domestic-node counts not fetched | clip-level labels; temporally-strong subset (May 2021) counts not fetched (gap) | 10 s (weak) / sub-second (strong, S) | no | labels CC BY 4.0, ontology CC BY-SA 4.0 (P); audio stays on YouTube | CSVs + 2.4 GB 1 Hz embeddings (P) |

### 2E. Device-state text generators

| Source | What it gives | Size | Temporal? | License |
|---|---|---|---|---|
| **acon96/Home-Assistant-Requests** [HF](https://huggingface.co/datasets/acon96/Home-Assistant-Requests) | templated `generate_home_assistant_data.py` assembling CSV "piles" of device names, actions and status requests into request/response pairs with service calls (`cover.close_cover`, `climate.set_fan_mode`, `fan.toggle`, light, lock, media_player …) (P) | 35,796 rows (33.4k train / 2.44k test), 93.3 MB parquet (P) | no — single-turn, no state-change sequences (P); English only (P) | MIT (P) |
| **HOMAGE sensor streams** | real PIR, light, color, gas/humidity/pressure/temperature, magnetometer, thermal at 2–10 Hz aligned to video (P) | 25.4 h | yes | RO |
| **BEHAVIOR object states** | BDDL predicates at every sim step during replay (S) | 1,950 h + 1,103 h | yes, exact | asset EULA RO |

### 2F. Long-stream and online-video benchmarks (held-out evaluation candidates)

| Benchmark | Footage | Size | What it measures | Timing tolerance | Negatives | License | Access |
|---|---|---|---|---|---|---|---|
| **EgoLifeQA** (in EgoLife) | EgoLife ego, 6 people × 1 week | 3,000 MCQ, 500 per participant (P); types EntityLog / EventRecall / HabitInsight / RelationMap / TaskMaster (P) | long-horizon memory: 997 questions with < 2 h certificate, 2,003 with > 2 h, up to > 24 h (P) | n/a (QA) | none | MIT (HF card) | HF |
| **EgoServe** (Vinci2, ECCV 2026) [paper](https://arxiv.org/html/2607.11523) | ≈128 h: EgoLife 3 participants × 5 days, HoloAssist 191 validation videos, CaptainCook4D 87 val/test videos (P) | > 3,000 timed service instances, 10 categories in 4 horizons (P) | **when to speak**: per-category P / R / F1 + GPT judge 1–5 (P) | δ = 60 s EgoLife, 25 s CaptainCook4D, 10 s HoloAssist (P) | no explicit false-alarm-per-hour metric (P) | not stated; inherits sources | GitHub "Vinci2" (P) |
| **EgoStream** (2026-06) [arXiv](https://arxiv.org/abs/2605.31557) | egocentric, sources not fetched (gap) | 2,250 questions → 8,528 recall-conditioned evaluations, 7 memory dimensions (P) | streaming episodic memory with Answer Validity Window (P) | AVW | n/a | paper CC BY 4.0 (P); data license gap | project page (gap) |
| **EgoMonth** (2026-08) [arXiv](https://arxiv.org/html/2608.13113) | ego only, 20 participants, 20–120 days, phones + action cams, ≥1K ≥25 fps (P) | 738 clips ≈ 301 h, 1,443 MCQ, 14 tasks (P) | month-level memory | n/a | n/a | "research-only license with tiered access control", "Redistribution and commercial use are prohibited" (P) | raw video by approval on secured servers (P) |
| **OVO-Bench** (CVPR 2025) [GitHub](https://github.com/JoeLeelyf/OVO-Bench) | web video, 644 videos, avg query at 263.42 s (P) | 3,100 queries, 12 subtasks (P) | backward tracing; real-time perception; **forward active responding** (REC, SSR, CRR: wait until evidence suffices) (P) | per query | implicit (must not answer early) | CC BY-NC-SA 4.0 (P) | HF `JoeLeelyf/OVO-Bench`, 44 GB src + 144 GB chunked (P) |
| **StreamingBench** (2024, upd. 2025-05) [GitHub](https://github.com/THUNLP-MT/StreamingBench) | 900 web videos (P) | 4,500 QA, 5 per video at different timestamps (P) | real-time visual, omni-source (audio+video), contextual understanding; **Proactive Output** timing task (P); main setup 60 s of context before the query (P) | per query | n/a | MIT (P) | HF `mjuicem/StreamingBench` (P) |

## 3. What the teacher should label first, and why

Cost model (from `docs/teacher-distillation.md`): ≈200 input tokens per 512 px frame, 2 Hz, 20 s chunks →
≈8K image tokens per chunk, 180 chunks per hour → **≈1.44M image tokens per hour of footage**, plus
≈1–2K tokens of context per chunk (journal, goal, device-state block, schema) → **≈1.7M input tokens per
hour**, ≈50K output tokens per hour, and 10–20 % more for `unsure` second looks at 4 Hz / 896 px. 100 h
≈ 170–200M input tokens. Dollars per hour are to be measured on the first batch, as the protocol says.
(Multi-camera sets multiply this per view; label one view per room, not every camera.)

Ranking for the first teacher batches, production geometry first:

1. **CASTLE 2024, the 5 static streams, one day first (≈50 h).** Only public source with fixed indoor
   cameras running continuously for days over several people in a real home, at UHD 50 fps (downsample to
   512 px / 2 Hz before the teacher sees it). No labels means the teacher's decisions are the only labels,
   and quiet periods are real, which is exactly the silence distribution the student must learn. Costs: the
   8.22 TB bundle (check whether the static streams can be fetched alone; ≈2.7 TB if the split is even, I);
   audit only by the 5 % human review; RO license.
2. **EgoLife exo cameras, if they are in the release (verify first).** 15 fixed GoPros with synchronized
   ego streams and ≈1,200 narrated phrases per hour make blind-mode teacher labeling auditable at scale
   (compare pushes to narrated events with the 1 s onset collar). Presence and location targets come from
   the exo view; object and food events can be cross-checked against the ego view of the same second.
   Caveats: Chinese speech (audio-language mismatch for an English-journal student), exo hours unknown, MIT
   card but consent terms unread. If exo is absent, use the ego streams like EPIC-KITCHENS (object events).
3. **HOMAGE, all 25.4 h, as the precision audit set.** Frame-precise atomic actions at ≈800 per hour and
   scene graphs at 3–5 frames per action give a ground truth for `object_taken`, `object_placed` and
   `activity_started` onsets; the teacher's precision per source is reported before any training run
   (protocol step 3), and this is the set that makes that number trustworthy for fixed cameras. Also the
   seed for device-state text (section 4). RO.
4. **BEHAVIOR-1K 2026 replays, 50–100 h re-rendered from one fixed room camera per scene.** Exact
   timestamps for every container, appliance and object-state change with zero human labeling, cheap to
   scale, and the `skill_annotation` / `primitive_annotation` rows already name the object. Use for the
   object/container/appliance kinds and for calibrating "one event, one line"; useless for
   `person_presence` (robot embodiment). Requires OmniGibson + Isaac Sim on an RTX GPU pinned to the
   collection versions (OG 3.7 / Isaac 4.5 per issue #2344); asset EULA is RO.
5. **Charades (≈82 h) later, for home diversity only.** 267 homes but 30 s scripted clips, so no journal
   chaining and no quiet time; label a few hundred clips to widen appearance, not to learn rate of speech.
6. **Continue the egocentric kitchen footage already on disk** (EPIC-KITCHENS-100; see the first pass in
   `docs/teacher-distillation.md` §4) for object and food events; HoloAssist and CaptainCook4D (both
   commercially clean) are the egocentric sets to add if the product tier needs more.

Defer: EgoMonth (approval-gated, ego only, RO), SmartHome-Bench and MultiTHUMOS (evaluation and method
reference only), audio sets (separate audio head; DESED strong labels audit the audio-event kinds).

## 4. Synthesizing device-state text streams

The context block the teacher and student see includes device-state text. Three ways to make it, none of
which exists ready-made:

- **HOMAGE sensors → real ambient state.** Threshold the PIR (2 Hz) into `presence: yes/no` per room, the
  light sensor (2 Hz) into `light: on/off` with hysteresis, the magnetometer (10 Hz) into door/drawer
  open/close if a magnet was mounted (not stated; check the sensor release), BME680 into slow
  `temp/humidity` lines, and the thermal imager into a coarse hot-appliance flag. Emit one text line per
  transition (`2026-09-16T10:41:12 kitchen.pir presence=true`) and feed it as prior context, not as a
  target. Only 25.4 h and the sensor modalities were "later" on the site, so this is an audit and format
  reference, not a corpus.
- **BEHAVIOR object states → exact appliance / container state.** During replay, poll BDDL predicates
  (`open`, `toggled_on`, `inside`, `ontop` …) each step and write transitions as Home-Assistant-style
  entity lines (`switch.radio_89 = on`); these are exact, dense and free, and they double as the ground
  truth for the `appliance_state` / `container_state` kinds. Per-frame state export is not in the released
  parquet (gap), so it must come from the replay run.
- **acon96 generator → vocabulary and format, plus a tiny temporal simulator.** Reuse its CSV piles (device
  names, domains, service-call syntax) and MIT license, and add what it lacks: a Markov process over
  entity states with realistic dwell times and time-of-day priors, so a stream can be attached to any
  footage as plausible device context; and use its service-call format for `act` payloads. Do not mix its
  request/response pairs into the target stream; they are single-turn and English-only.

Pair synthesized device text with real video only when the two are consistent (e.g., a `light.kitchen =
off → on` line at a visible light change in CASTLE); otherwise keep device streams on the sim tier.

## 5. Usable hours by tier and journal-line targets

| Tier | Sources (this note + prior note) | Hours available | Fixed-camera? | License tier | Teacher cost at 1.7M input tokens/h | Expected journal lines / h (I, from the §4 pass in `docs/teacher-distillation.md`: 3–4 pushes per 29 s of dense cooking) |
|---|---|---|---|---|---|---|
| A. real fixed cameras in homes | CASTLE static ≈200 h (C); EgoLife exo ? h (gap; nominal upper bound 15 cams × ≈8 h × 7 d ≈ 840 camera-hours, C); HOMAGE 25.4 h; TSU ≈190 h (prior note, by email) | 225 h verified now, 400–1,250 h if EgoLife exo and TSU come through | yes | RO (CASTLE, HOMAGE, TSU); EgoLife MIT-card | 100 h ≈ 170M tokens | multi-person common room 100–200; single-person ADL 50–150; overnight / empty 0–5 |
| B. egocentric with dense narration | EgoLife ego 300 h; EPIC-KITCHENS-100 100 h; HoloAssist 169 h; CaptainCook4D 94.5 h; Ego-Exo4D cooking 564 h (prior note) | ≈1,230 h | no | mixed: EgoLife MIT-card, EPIC RO, HoloAssist / CaptainCook4D COM, Ego-Exo4D COND | as above | busy kitchen 300–450 |
| C. simulation with exact object state | BEHAVIOR 2026 1,950 h; BEHAVIOR 2025 ≈1,103 h; VirtualHome unlimited | ≈3,050 h + generated | re-rendered fixed cams | demos MIT-card, assets RO | replay + teacher; or programmatic targets without a teacher | 150–300 (skill rate ≈139/h plus state changes) |
| D. short third-person home clips | Charades ≈82 h; Charades-Ego; SmartHome-Bench ≈6.7 h (eval only) | ≈90 h | propped, not fixed | RO | 82 h ≈ 140M tokens | 700–900 (scripted, no quiet) |
| E. audio | SINS dev ≈200 node-h (≈50 h wall-clock), DESED ≈49.6 h real (5.2 h strong), AudioSet domestic (counts gap) | ≈250 h | n/a | SINS EULA (gap), DESED per-file, AudioSet labels COM | audio head, not the VLM teacher | n/a |

Targets for the first training set (I): 100 h of Tier A + 50 h of Tier B + 50 h of Tier C ≈ 200 h ≈
1.44M ticks at 2 Hz, of which roughly 15–40K are journal lines (1–3 %) and the rest silence — the
distribution the protocol asks for. Budget ≈ 300–350M teacher input tokens for that set.

## 6. Held-out evaluation sets (never trained on)

- **CASTLE day 4 (≈50 h static)**: the fixed-camera long-stream test with real quiet hours; teacher-labeled
  then 100 % human-reviewed for journal lines and a sampled 5 % of silent ticks; reports false pushes per
  hour and onset error. Days 1–3 train. (Same people and rooms, so this measures the contract, not
  generalization across homes.)
- **EgoLife, entirely** (or none of it in training): the only way to use EgoLifeQA (3,000 MCQ, > 24 h
  certificates), EgoServe (its EgoLife part: 3 participants × 5 days, δ = 60 s) and probably EgoStream
  without contamination; the 15 shared exo cameras make a participant-disjoint split impossible, and
  day-disjoint splits straddle the long certificates. Recommendation: hold out EgoLife and take Tier-A
  training hours from CASTLE, HOMAGE and BEHAVIOR replays; revisit only if CASTLE alone under-delivers.
- **EgoServe non-EgoLife parts**: HoloAssist 191 validation videos (δ = 10 s) and CaptainCook4D 87 val/test
  videos (δ = 25 s) — never train on those splits; both are commercially clean, so this evaluation survives
  into the product tier.
- **SmartHome-Bench (1,203 clips)**: hazard / anomaly recall and false alarms on the "normal" half from
  real consumer fixed cameras; per-clip only, so score the journal, not the tick.
- **OVO-Bench forward-active (REC / SSR / CRR) and StreamingBench Proactive Output**: web video, but they
  isolate the "wait, then speak" skill; OVO is CC BY-NC-SA (evaluation only), StreamingBench MIT.
- **Toyota Smarthome Untrimmed** (from the prior note): the dense fixed-camera ADL benchmark, only after
  the written permission it requires.
- **Audio**: DESED public eval 692 clips (strong labels) and DCASE 2018 T5 eval 72,972 segments incl.
  "absence".
- **EgoMonth**: approval-gated and ego-only; skip until fixed-camera evaluation is solid.
Freeze these before the first teacher batch (protocol step 5) and keep the teacher's rows for them in a
directory no training manifest can read.

## 7. License conclusions

### 7.1 For the research phase

Everything above is usable for research. Research-only terms: CASTLE (CC BY-NC-SA 4.0), HOMAGE
(non-commercial competition terms), Charades / Charades-Ego (academic, non-profit, government), OVO-Bench
(CC BY-NC-SA 4.0), EgoMonth (research-only, tiered, approval), BEHAVIOR asset bundle (non-commercial
academic EULA, required for replay), Toyota Smarthome (non-commercial, by email), EPIC-KITCHENS (CC BY-NC).
Unstated: SmartHome-Bench, MultiTHUMOS, SINS EULA variant, EgoServe data (inherits sources), EgoStream
data. ShareAlike (CASTLE, OVO): derived label files should be treated as share-alike; keep teacher rows for
those sources in a separately licensed directory.

### 7.2 For a later product

- Student weights that will ship may be trained only on: EgoLife (MIT on the HF card — confirm the project
  page's consent / usage statement before relying on it; an MIT license on a week of six people's lives is
  unusual), HoloAssist (CDLA-Permissive-2.0), CaptainCook4D (Apache-2.0), Ego4D / Ego-Exo4D (bespoke, models
  allowed, data never visible in the product), DROID / RH20T-C / OakInk2 (prior note), VirtualHome renders
  (MIT code; check Unity asset terms), StreamingBench (MIT, eval), acon96 (MIT), AudioSet labels (CC BY 4.0;
  audio fetched from YouTube under its terms), DESED code (MIT) with per-file audio checks.
- BEHAVIOR demo trajectories: the HF cards say MIT, but every RGB frame is a render of assets whose EULA is
  non-commercial and encrypted for ShapeNet / TurboSquid compliance (prior note); re-rendered replays
  certainly inherit the EULA. Treat released frames as "MIT with residual risk, get counsel" and replays as
  research-only.
- CASTLE, HOMAGE, Charades, TSU, EPIC, OVO-Bench, EgoMonth: research only; models trained on them do not
  ship. Toyota and Bristol offer commercial licenses by email; CASTLE's authors would have to be asked.
- Consequence: the product-tier student has no public fixed-camera home footage except possibly EgoLife's
  exo streams. Own consented recordings remain the only clean production-geometry source, which confirms the
  roadmap's plan; the research-tier teacher labels on CASTLE / HOMAGE are for learning the contract, the
  audit numbers and the pipeline, not for the shipped weights.

## 8. Not verified / gaps

- EgoLife: exo camera hours; whether the 15 GoPro streams are in the 512 GB HF bundle (512 GB looks small
  for 300 h of ego plus 15 exo streams at GoPro bitrates, I); consent / usage terms on egolife-ai.github.io;
  exact language mix; narration timestamp precision.
- CASTLE: language of speech; per-stream hours; whether static streams can be downloaded separately; the
  transcript release; camera models.
- HOMAGE: the 29 relationship classes (Action Genome-style, S); whether door magnets were used with the
  magnetometer; whether the sensor CSVs are released yet; validation / test instance counts.
- BEHAVIOR: whether `replay_obs.py` accepts added external cameras and modalities (OmniGibson's sensor API
  suggests yes, S); per-frame object-state export; headless / OS requirements; the 2026 collection-version
  pin (issue #2344 seen only as a title); whether the MIT card is compatible with the asset EULA.
- VirtualHome: apartment count (7, S), program dataset size (≈2,800, S), Unity asset license.
- AudioSet: temporally-strong subset counts; clip counts under "Domestic sounds, home sounds".
- SINS: EULA variant; whether the full 13-node one-week continuous recording is obtainable outside the
  DCASE subsets.
- DESED: number of generated synthetic soundscapes; onset/offset decimal precision.
- EgoStream: source datasets, hours, download route. EgoServe: which EgoLife participants and days; the
  Vinci2 GitHub URL; data license.
- SmartHome-Bench license; MultiTHUMOS license; StreamingBench and OVO-Bench total hours.
- Charades / Charades-Ego numbers are from the prior note, not re-fetched.
- All journal-line rates in §5 are inferred from one 29 s teacher pass and must be replaced by measured
  rates per source after the first batch.

## Sources (URL, fetched 2026-09-16 unless noted)

- EgoLife: https://arxiv.org/abs/2503.03803 ; https://arxiv.org/html/2503.03803 ; https://huggingface.co/datasets/lmms-lab/EgoLife ; project https://egolife-ai.github.io/ (not fetched)
- CASTLE 2024: https://castle-dataset.github.io/ ; https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024 (not fetched) ; arXiv https://arxiv.org/html/2503.17116 (prior note)
- HOMAGE: https://arxiv.org/abs/2105.05226 ; https://ar5iv.labs.arxiv.org/html/2105.05226 ; https://homeactiongenome.org/ (prior note)
- SmartHome-Bench: https://arxiv.org/html/2506.12992 ; https://github.com/Xinyi-0724/SmartHome-Bench-LLM (not fetched) ; SMH-Bench https://arxiv.org/html/2606.01912 (search only)
- BEHAVIOR-1K: https://behavior.stanford.edu/challenge/index.html ; https://behavior.stanford.edu/challenge/dataset.html ; https://huggingface.co/datasets/behavior-1k/2025-challenge-demos ; https://huggingface.co/datasets/behavior-1k/2026-challenge-demos ; https://github.com/StanfordVL/BEHAVIOR-1K/issues/2344 (search only) ; asset EULA per prior note
- VirtualHome: https://github.com/xavierpuigf/virtualhome ; http://virtual-home.org/ (TLS failure)
- Charades / Charades-Ego: https://prior.allenai.org/projects/charades ; https://prior.allenai.org/projects/charades-ego (prior note)
- MultiTHUMOS: https://ai.stanford.edu/~syyeung/everymoment.html
- DESED: https://project.inria.fr/desed/ ; https://github.com/turpaultn/DESED ; Zenodo 3702397, 3588172
- SINS / DCASE 2018 Task 5: https://dcase.community/challenge2018/task-monitoring-domestic-activities ; https://arxiv.org/pdf/1807.11246 (search only) ; Zenodo 1247102, 1964758
- AudioSet: https://research.google.com/audioset/download.html
- acon96: https://huggingface.co/datasets/acon96/Home-Assistant-Requests ; generator repo acon96/home-llm (S)
- EgoStream: https://arxiv.org/abs/2605.31557
- EgoMonth: https://arxiv.org/html/2608.13113
- EgoServe / Vinci2: https://arxiv.org/html/2607.11523 ; related EgoPro-Bench https://arxiv.org/pdf/2605.07299 , S-EMBER https://arxiv.org/pdf/2607.02689 (search only)
- OVO-Bench: https://github.com/JoeLeelyf/OVO-Bench ; arXiv 2501.05510
- StreamingBench: https://github.com/THUNLP-MT/StreamingBench ; HF mjuicem/StreamingBench

## Appendix A. Raw findings by fetch batch (kept so partial work survives)

### Batch 1 (calls 1-5, 2026-09-16)

- EgoLife, arXiv abstract https://arxiv.org/abs/2503.03803 (PRIMARY, abstract only): "300-hour egocentric, interpersonal, multiview, and multimodal daily life dataset with intensive annotation"; six participants "lived together for one week"; AI glasses for egocentric capture; "synchronized third-person-view video references". Camera count, narration density, language, license, download route NOT on the abstract page — full text needed.
- HOMAGE, arXiv abstract https://arxiv.org/abs/2105.05226 (PRIMARY, abstract only): "multi-view action dataset with multiple modalities and view-points supplemented with hierarchical activity and atomic action labels together with dense scene composition labels". Numbers already in datasets-2026-09.md (27 participants, 2 houses, 1,752 sequences, 5,700 videos, ~30 h, 75 activities, 453 atomic actions, scene graphs on 3-5 frames per action, 12 sensor types, non-commercial competition terms). Sensor list and rates still needed — full text.
- BEHAVIOR challenge page https://behavior.stanford.edu/challenge/index.html (PRIMARY): 2026 challenge; "20,000 human teleoperation demos", "1,950 hours in total", "100 full-length household tasks", "7 scenes, including 4 new scenes"; data = "RGB and depth observations, robot proprioception and actions, and skill/subtask annotations"; collected with JoyLo whole-body teleoperation (Simovation assisted). Download size, HF route, format, license, replay-camera control, headless requirements NOT on this page.
- MultiTHUMOS https://ai.stanford.edu/~syyeung/everymoment.html (PRIMARY): "30 hours across 400 videos in the THUMOS'14 action detection dataset"; "38,690 annotations of 65 action classes"; "an average of 1.5 labels per frame and 10.5 action classes per video"; dense multilabel frame-level over untrimmed video; download resources/multithumos.zip (updated 2017-07-19); no license stated on page. Content is sports (THUMOS'14), not homes.
- StreamingBench https://github.com/THUNLP-MT/StreamingBench (PRIMARY): "900 diverse videos", "4,500 human-annotated QA pairs", "Five questions per video at different timestamps"; evaluated with only preceding context visible (main setup: "60 seconds of context preceding the query time"); three categories: Real-time Visual Understanding, Omni-source Understanding (audio+visual), Contextual Understanding; includes a "Proactive Output" task (timing prediction); MIT license; HF `mjuicem/StreamingBench`; last update noted 2025-05-15; total hours and download size not stated on the README.

### Batch 2 (calls 6-14, 2026-09-16)

- EgoLife full text https://arxiv.org/html/2503.03803 (PRIMARY): 300 h egocentric; 6 participants, one week, "eight hours of egocentric multimodal video daily" on Meta Aria glasses; "15 strategically placed GoPro cameras" fixed in the common areas of the house (Fig. 2 "locations of 15 Exo cameras in the common area"), "synchronized third-person perspective data"; third-person hours not stated. Annotation: "361K brief, subtitle-like phrases, averaging 2.65 seconds" merged into "25K 'merged captions'" over 5-minute clips; "50-hour transcript" reviewed; "All the annotations (transcripts, captions, QAs) are primarily in Chinese" with English translation. EgoLifeQA: 3,000 questions (500 per participant), types EntityLog / EventRecall / HabitInsight / RelationMap / TaskMaster; certificate length: 997 questions < 2 h, 2,003 > 2 h, up to > 24 h. EgoGPT 7B (EgoIT-99K), EgoRAG (top-3 30-s clips). Data license and download size NOT in the paper (paper page only shows the arXiv perpetual license); project page https://egolife-ai.github.io/.
  - Derived: 361K phrases / 300 h ≈ 1,200 phrases per hour ≈ one per 3 s (narration is near-continuous, not event-sparse); timestamp precision sub-second (phrase-level).
- HOMAGE full text https://ar5iv.labs.arxiv.org/html/2105.05226 (PRIMARY): 27 participants, two houses (kitchens, bathrooms, bedrooms, living rooms, laundry rooms); 1.75K sequences; 5,700 videos; 25.4 h; 75 activities, 453 atomic actions; each sequence "one ego-view video as well as at least one or more synchronized third person views", "more than 3 views per action sequence" on average; sensors "attached to several locations in the room". Sensor rig: Video OV5647 30 fps; I2S mic SPH0645LM4H 48 kHz; GridEYE thermal AMG8833 10 Hz; PIR human presence AK9753AE 2 Hz; ambient light TSL2591 2 Hz; ambient color ISL29125 10 Hz; BME680 CO2/humidity/pressure/temp 5 Hz; magnetometer MLX90393 10 Hz; plus acceleration/gyro. Scene graphs on third-person views, "3 or 5 [frames] ... across the range of each atomic action interval (3 for intervals less than 3 seconds and 5 otherwise)"; "497,534 bounding boxes and 583,481 relationships"; "86 object classes (excluding 'person'), and 29 relationship classes". Atomic actions annotated with "the start and end frames and the category"; "about 60% of the atomic actions under 2 seconds and 80% under 5 seconds". License/download not in paper (site terms: non-commercial competition rules, see datasets-2026-09.md).
  - Derived: 20,039 train atomic instances (site) over ~25 h ≈ 800 atomic events per hour; frame-precise (30 fps) boundaries.
- OVO-Bench https://github.com/JoeLeelyf/OVO-Bench (PRIMARY): 644 videos, 3,100 queries, average query timestamp 263.42 s; three modes Backward Tracing (ASI, HLD, EPM), Real-Time Visual Perception (ATR, ACR, OCR, STU, OJR, FPD), Forward Active Responding (REC, SSR, CRR) — the model must "wait for sufficient evidence before responding"; HF `JoeLeelyf/OVO-Bench`; src_videos ~44 GB, chunked_videos ~144 GB; license CC BY-NC-SA 4.0; CVPR 2025; arXiv 2501.05510.
- VirtualHome http://virtual-home.org/ — fetch FAILED (TLS certificate mismatch, host serves ade20k.csail.mit.edu cert). Retry via GitHub.
- DESED https://project.inria.fr/desed/ (PRIMARY, partial): 10 classes "Alarm/bell/ringing, Blender, Cat, Dog, Dishes, Electric shaver/toothbrush, Frying, Running water, Speech, Vacuum cleaner"; 10-s clips; for SED "recognize events with their time boundaries"; download via https://github.com/turpaultn/DESED. Subset sizes/license pending (resolved in Batch 3).
- acon96/Home-Assistant-Requests https://huggingface.co/datasets/acon96/Home-Assistant-Requests (PRIMARY): 35,796 rows (33.4k train, 2.44k test), 93.3 MB parquet; generated by `generate_home_assistant_data.py` from CSV "piles" (device names, templated actions, status requests); device domains seen: cover, climate, fan, light, lock, media_player (and switches per the generator); responses "Currently only `en` is supported"; license MIT; single request/response conversations, no temporal state-change sequences.
- BEHAVIOR HF repos (search, SECONDARY until cards are fetched): `behavior-1k/2026-challenge-demos` — LeRobotDataset v3 format, 20,000 episodes, 100 tasks, "approximately 3.0 TB"; `behavior-1k/2025-challenge-demos` — NeurIPS 2025 challenge, "10,000 teleoperated expert demonstrations (200 per task, over 1,200 hours)" with multi-modal observations and fine-grained skill annotations; also `behavior-1k/2025-challenge-task-instances`. Sources: https://huggingface.co/datasets/behavior-1k/2026-challenge-demos , https://huggingface.co/datasets/behavior-1k/2025-challenge-demos
- SmartHome-Bench (search, SECONDARY): arXiv 2506.12992, CVPR 2025 Workshop (VAND); 1,203 videos recorded by smart home cameras; 7-category anomaly taxonomy (Wildlife, Senior Care, Baby Monitoring, ...); each video annotated with anomaly tags, descriptions, reasoning; code/data at https://github.com/xinyi-0724/smarthome-bench-llm . Also found SMH-Bench (arXiv 2606.01912, LLM agents in smart homes, text-grounded, not video).
- EgoStream (search, SECONDARY): arXiv 2605.31557 (2026-06-01), "diagnostic benchmark for streaming episodic memory", 2,250 curated questions in 7 dimensions (detail, spatial, temporal, event, social, causal, prospective), Answer Validity Window (AVW) expands to 8,528 recall-conditioned evaluations.
- EgoMonth (search, SECONDARY): arXiv 2608.13113 (2026-08-13), "over 300 hours of first-person daily-life recordings from 20 participants spanning 20 to 120 days", 1,443 multiple-choice QA pairs.
- EgoServe (search, SECONDARY, no URL yet): "3K+ manually verified proactive-service instances organized into 10 assistance categories and four temporal horizons from instant alerts to multi-day habit coaching".
- Also surfaced: S-EMBER (arXiv 2607.02689) "Large-Scale Benchmark for Streaming Egocentric Memory Retrieval" — not fetched.

### Batch 3 (calls 15-26, 2026-09-16)

- EgoLife HF card https://huggingface.co/datasets/lmms-lab/EgoLife (PRIMARY): "License: mit"; 512 GB; 32,001 rows; Video-Text-to-Text; "Data cleaning, stay tuned!"; no gating stated; last update 2025-03-07; card defers to https://egolife-ai.github.io/ . Whether the 15 exo GoPro streams are in the HF bundle is NOT stated on the card (gap).
- BEHAVIOR 2026 demos HF card https://huggingface.co/datasets/behavior-1k/2026-challenge-demos (PRIMARY, partial: viewer error): license MIT; tags lerobot-v3, omnigibson; table columns task_name, data_folder, meta_data, skill_annotation, primitive_annotation; skill rows carry skill_idx, skill_id, skill_description (e.g. "move to"), object_id (e.g. "radio_89"). Episode count/hours/size on the card not readable this session (search snippet: 20,000 episodes, 100 tasks, ~3.0 TB — SECONDARY; challenge page: 1,950 h — PRIMARY).
- BEHAVIOR 2025 demos HF card https://huggingface.co/datasets/behavior-1k/2025-challenge-demos (PRIMARY): 10,000 episodes; 119,094,660 frames at 30 fps (= 1,103 h computed; card text "over 1,200 hours" per search snippet); 50 tasks; 1.95 TB; LeRobot v2.1 parquet `data/task-{chunk:04d}/episode_{index:08d}.parquet`; cameras: left wrist 480x480, right wrist 480x480, head 720x720, all 30 fps RGB; depth for all three (`observation.images.depth.*`); instance segmentation (`observation.images.seg_instance_id.*`); `observation.state` 256-d; `observation.cam_rel_poses` 21-d; `observation.task_info`; license "mit"; no replay/re-render instructions on the card.
- SmartHome-Bench https://arxiv.org/html/2506.12992 (PRIMARY): 1,203 videos "recorded by smart home cameras", "captured by both indoor and outdoor smart home cameras", "Videos from public sources, such as YouTube"; average ~20 s, most < 80 s (total ≈ 1,203 x 20 s ≈ 6.7 h, computed); 7 categories "security, baby monitoring, kid monitoring, senior care, pet monitoring, wildlife, and other"; balanced normal/abnormal, 91 "vague abnormal"; per-video labels: category, anomaly tag, description (≤200 words), rationale (≤100 words); NO onset timestamps; license not stated; code/data https://github.com/Xinyi-0724/SmartHome-Bench-LLM .
- EgoStream https://arxiv.org/abs/2605.31557 (PRIMARY, abstract page): 2,250 questions; 8,528 recall-conditioned evaluations; 7 dimensions; AVW "temporal span an answer remains valid as the observed scene evolves"; streaming (no future frames); arXiv listing shows CC BY 4.0 for the paper; authors Forte, Lando, Furnari (Univ. of Catania group, SECONDARY inference); v1 2026-05-29, v2 2026-06-01. Source videos/hours NOT on the abstract page (gap).
- EgoMonth https://arxiv.org/html/2608.13113 (PRIMARY): 738 clips, ~301 h, 20 participants, 20-120 days each; smartphones and action cameras (GoPro, Insta360, DJI named in consent template); ≥1K resolution, ≥25 fps; six activity categories incl. household chores, indoor and outdoor; 1,443 MCQ pairs, 14 tasks in 3 levels (Schema Consolidation 2, Episodic Indexing 8, Cascading Reasoning 4); egocentric only; "research-only license with tiered access control", raw videos "require additional approval and are stored on secured servers", "Redistribution and commercial use are prohibited"; Gaussian blur/masking of bystander faces, plates, addresses; dated 2026-08-13; download route not stated.
- VirtualHome https://github.com/xavierpuigf/virtualhome (PRIMARY): Unity multi-agent household simulator driven by programs; interactions "picking up objects, switching on/off appliances, opening appliances"; outputs "time-stamped actions, instance/semantic segmentation, and optical flow and depth", environment graphs "how the environment evolves"; animated characters in scene; MIT license (repo); executable v2.3.0 for Linux/macOS/Windows at virtual-home.org/release/simulator/v2.0/v2.3.0/ ; apartment count/program-dataset size not on README (paper: 7 apartments, ~2,800 programs — SECONDARY from prior knowledge).
- DESED https://github.com/turpaultn/DESED (PRIMARY): real 10-s clips from AudioSet/YouTube: weak-labeled train 1,578; unlabeled in-domain 14,412; validation strong 1,168; public eval 692 (≈ 49.6 h total real, computed; strong-labeled ≈ 5.2 h); synthetic soundbank: 2,060 background files (SINS) + 1,009 foreground (Freesound) for train; eval 12 Freesound + 5 YouTube backgrounds + 314 foregrounds; strong labels = onset/offset time boundaries (precision not stated on README); code MIT, "The different datasets contain a license file at their root for the attribution of each file" (per-file Freesound/YouTube/MUSAN/SINS terms); Zenodo DESED_synthetic 3702397, DESED_public_eval 3588172; v1.2.5 2020-02-26.
- AudioSet https://research.google.com/audioset/download.html (PRIMARY): 2,084,544 10-s segments (eval 20,383; balanced train 22,176; unbalanced 2,042,985); 527 classes; dataset CC BY 4.0, ontology CC BY-SA 4.0; audio is YouTube content (only CSV of YTID/start/end/labels plus 2.4 GB of 1 Hz embeddings are distributed); temporally-strong labels released May 2021 on a separate page (counts not fetched, gap). "Domestic sounds, home sounds" is an ontology node (SECONDARY, from ontology knowledge).
- CASTLE 2024 https://castle-dataset.github.io/ (PRIMARY): "Over 600 hours of UHD 50fps video with audio"; "15 video streams (10 egocentric, 5 static perspectives)"; four days; 10 participants; 6DoF IMU, GPS, biometrics; 8.22 TB; CC BY-NC-SA 4.0; HF https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024 ; no event annotations listed (transcripts only, per arXiv 2503.17116 in datasets-2026-09.md).
- EgoServe / Vinci2 (search, SECONDARY): arXiv 2607.11523 (2026-07, ECCV 2026); "3K+ manually verified proactive-service instances over EgoLife, HoloAssist, and CaptainCook4D", 10 categories, 4 temporal horizons "from immediate safety alerts to long-term habit coaching"; EgoMemo agent decides at each timestep "whether assistance is warranted". Related: EgoPro-Bench arXiv 2605.07299 (personalized proactive interaction), Ambient@EgoProactive 2026 arXiv 2609.07099, EgoIntent arXiv 2603.12147.
- SINS (search, SECONDARY): continuous recording of one person living in a vacation home for one week, 13 microphone-array nodes (4 linear mics each); DCASE 2018 Task 5 derivative uses 7 arrays in the living room/kitchen, 10-s segments, 9 classes (absence, cooking, dishwashing, eating, other, social activity, vacuum cleaning, watching TV, working). Sources https://arxiv.org/pdf/1807.11246 , https://dcase.community/challenge2018/task-monitoring-domestic-activities .

### Batch 4 (calls 27-29, 2026-09-16)

- BEHAVIOR replay (search, SECONDARY unless confirmed by the dataset docs fetch): 2026-challenge-demos "20,000 human-collected teleoperation demos across 100 tasks and is 3.27 TB in total, following the LeRobot V3 format"; robot R1Pro; replay script `OmniGibson/scripts/learning/replay_obs.py` with `--demo_id` and `--action-only`; open issue https://github.com/StanfordVL/BEHAVIOR-1K/issues/2344 — "2026 challenge demos: action-only replay diverges on the 3.9 eval stack (engine mismatch between collection on OG 3.7 / Isaac 4.5 and evaluation on OG 3.9 / Isaac 5.1)"; cameras: two wrist RealSense (RGB+depth, 480x480) + head camera; RTX-class GPU needed (teams cite RTX 4090). Dataset docs: https://behavior.stanford.edu/challenge/dataset.html
- EgoServe / Vinci2 https://arxiv.org/html/2607.11523 (PRIMARY): "over 3,000 service instances"; "~128h of video from three complementary source datasets": EgoLife (three participants, five days), HoloAssist ("191 validation videos"), CaptainCook4D ("87 videos with explicit step-error annotations"); 10 categories by horizon — Instant: Safety Alerts, Tool Use; Short-term: Error Recovery, Resource Reminder, Next-Step Guidance; Episodic: Task Reminder, Memory Recall; Long-term: Habit Coaching, Routine Optimization, Memory Link; temporal tolerance "δ=60s for EgoLife, δ=25s for CaptainCook4D, and δ=10s for HoloAssist"; metrics: Precision/Recall/F1 per subcategory + GPT judge 1-5; no explicit false-alarm-per-hour metric described; code/benchmark "publicly available at Vinci2" (GitHub); arXiv v1 2026-07-13; data license not stated (inherits source-dataset terms: EgoLife MIT on HF, HoloAssist CDLA-Permissive-2.0, CaptainCook4D Apache-2.0).
- SINS / DCASE 2018 Task 5 https://dcase.community/challenge2018/task-monitoring-domestic-activities (PRIMARY): development set "approximately 200 hours of data from 4 sensor nodes", 72,984 10-s segments, 4 linear mics per node, 16 kHz; Zenodo 1247102 (42.6 GB download, 87.0 GB extracted); evaluation set 72,972 files from all 7 nodes, Zenodo 1964758 (42.2 GB / 87.0 GB); 9 classes with dev counts: Absence 18,860; Cooking 5,124; Dishwashing 1,424; Eating 2,308; Other 2,060; Social activity 4,944; Vacuum cleaning 972; Watching TV 18,648; Working 18,644; "each segment represents one activity" (multi-activity segments dropped); license via "EULA.pdf" on the record (variant not stated on page — gap; the original SINS release is described elsewhere as CC BY-NC, SECONDARY).
  - Derived: Absence = 18,860 x 10 s ≈ 52 h of certified-quiet audio across 4 nodes (≈ 13 h of wall-clock silence).

### Batch 5 (call 30, 2026-09-16) — budget exhausted after this call

- BEHAVIOR 2026 dataset docs https://behavior.stanford.edu/challenge/dataset.html (PRIMARY): "20,000 human-collected teleoperation demos across 100 tasks"; "3.27 TB in total" (demos) and "1.44 TB in total" (raw HDF5 replay data, HF `2026-challenge-rawdata`); "Avg. Trajectory Duration 351.54 seconds / 5.9 minutes"; "270,600 Total Skills", "31 Unique Skills"; "LeRobot V3 format with customizations"; RGB "720 x 720 head camera; 480 x 480 wrist cameras"; depth "Range [0, 10] meters"; low-dimensional "proprioceptions, actions, camera poses"; replay: "OmniGibson/scripts/learning/replay_obs.py to replay trajectories and collect additional visual observations" (requires OmniGibson). NOT on the page: segmentation, per-frame object-state / BDDL ground truth, headless / GPU / OS requirements, dataset license, asset EULA, whether cameras can be changed on replay.
