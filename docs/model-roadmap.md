# Model roadmap: one observer model, distilled from a frontier teacher

Status: proposal v2, 2026-09-16. Supersedes the decomposed perception stack proposed earlier the
same day (git history, commit ab28c05) at the owner's direction: the observer is one unified
vision-language model, not a cascade of detectors. Grounded in the measured failures of
iterations 1 and 2, the first teacher pass (`docs/teacher-distillation.md`), and the research
notes under `docs/research/`. Where a number below comes from a note it says so; anything
load-bearing gets re-measured on the target machine before it is relied on.

## 0. The decision in one paragraph

One model watches the stream. At every tick it emits a silence token, or one structured
observation for the cloud coordinator, or one pre-authorized action. It is trained by
distilling a frontier model's tick-by-tick judgment about *when to speak and what is worth
saying*, not by fitting narration windows into JSON. Its own journal re-enters its context so
it knows what it already said, and it verifies its own candidates by looking again at retained
frames at higher resolution. The privacy boundary, the export scanner, the sandboxed
coordinator, the journals and the device protocol all stay exactly as they are; what changes is
the contract (ticks and silence instead of windows), the data (teacher-labeled streams where
silence dominates), the training (streaming fine-tuning with the vision side trainable, then
reinforcement on timing), the runtime (a streaming engine with cache reuse and change-gated
tokens), and the evaluation (false pushes per hour and onset latency on held-out footage).

## 1. What the measured failures say, and what they do not

Iterations 1 and 2 fine-tuned Gemma 4 E4B with language-only LoRA on about 200 three-second
windows cut from egocentric kitchen footage, with narration-derived JSON targets, and evaluated
it as an offline window judge. The results on fresh footage: strict recall 0 of 242, judged
recall at most 0.034, about 1800 false events per hour, 0 of 31 quiet windows abstained, and a
static-repeat control in which the model claimed events in 20 of 20 windows built from a single
repeated frame (`reports/physical-v2-temporal-control.json`, `PRODUCTION_REPORT.md`).

Those runs tested the opposite of the unified design, so they refute a recipe, not the idea:

- **The contract was offline and per-window.** The model was asked, after the fact, for
  boundaries, boxes, evidence citations and calibrated confidence inside one JSON object, and 74
  percent of every target was JSON scaffolding. A streaming contract asks a different question
  at each tick: is there something to say right now. Onset time is the tick, not a generated
  number.
- **Silence was never a target.** Quiet windows were 25 percent of rows and 3.8 percent of
  tokens, with a ten-token empty target next to eighty-token positives. The model learned scene
  priors because nothing taught it to stay quiet. In the stream contract silence is the default
  token at roughly 90 percent of ticks even in dense cooking footage (teacher pass, section 3)
  and at nearly all ticks in a quiet home; it is the majority class, not an afterthought.
- **The targets were narrations, not judgments.** Narrations record every cut and stir. The
  coordinator needs durable state changes reported once. The teacher pass shows what the
  contract looks like on the same footage: 3 to 4 pushes in 29 seconds of active cooking, one
  event per state change, repeats suppressed.
- **The recipe could not have taught it anyway.** Language-only LoRA on 200 windows; every
  published temporal recipe that works unfreezes the projector and usually the vision tower on
  10^5 to 10^6 instances, or applies reinforcement fine-tuning to a base that already localizes
  (`docs/research/decoding-and-training-2026-09.md` section 2).
- **The runtime was not a stream.** 12.2 s per three-second window through transformers on
  Metal. A streaming engine with cache reuse and change-gated tokens is a prerequisite, not an
  optimization (section 5).

Two findings from the old runs carry over unchanged: capture, buffering, journaling and the
export boundary never lost a frame or leaked media across 4348 fresh-footage frames, and the
frozen protocol and scorers measure everything above honestly.

## 2. Architecture: one model, one time-ordered stream

```
fixed camera(s), microphone, device-state bus (AHP / Home Assistant)
   -> tick clock (1 to 2 Hz), change-gated frame tokens, audio or ASR text, state diffs as text
   -> [camera id][t][visual tokens][audio][state diff]  appended to ONE token stream
   -> observer model (4B class; bounded memory: recent vision window, longer text window,
      recurrent state if the backbone is hybrid)
   -> per tick:  <silent>            (default; decided from the log-probability margin)
              |  {"k": ..., "s": ..., "l": ..., "d": ..., "c": ...}   one compact line
              |  <act> rule_id args
              |  <look>               ask for the last 2 s at 4 Hz and 2x resolution, then decide
   -> the emitted line re-enters the stream (working memory, duplicate suppression)
   -> physical journal -> export scanner -> coordinator (unchanged)
   -> second look: the same model re-reads retained frames at 4 Hz and higher resolution when
      it emits <look> or the coordinator asks; its answer is the verification result
```

- **Silence by default.** The decision at each tick is read from the renormalized
  log-probability of the silence token against the burst-start token, thresholded with
  hysteresis and a refractory period and certified on held-out quiet hours (the calibration
  chain in `docs/research/decoding-and-training-2026-09.md` section 4). No verbalized
  confidence is trusted at this model size.
- **Memory is the journal.** The last N emitted lines are always in context, which is what
  makes "already reported" learnable; the recurrent state of a hybrid backbone keeps the gist
  beyond the window; anything older is a retrieval call by the coordinator over the journal.
- **Change-gated input.** A static home is temporally redundant; frame tokens are admitted in
  proportion to change, so tokens per second follow activity, not the clock. With the Qwen
  vision stack every token is an aligned 32 by 32 pixel block of a two-frame pair, so a gate
  that skips unchanged blocks before the encoder reproduces the effect of Efficient Video
  Sampling, the one training-free pruning method shipped in a serving engine, while keeping
  position IDs intact. The camera's own H.264 or H.265 motion vectors give that gate for free
  at 16 by 16 pixel macroblocks (four per token); frame differencing is the day-one version.
  The student is trained with stochastic token dropping so it is robust to whatever the gate
  admits (`docs/research/base-models-and-runtimes-2026-09.md` section 3).
- **Second look is the verifier, and it is a trained decision.** The teacher's second look on
  every candidate pickup or placement and the student's `look` token are the same mechanism;
  the one published precedent is Eyes Wide Open's `ask_high` action on an 8B egocentric
  streamer, which more than doubled its proactive F1. The runtime answers `look` with the last
  two seconds at 4 Hz and twice the resolution, and the student then emits silence or a burst.
  At least half of the `look` targets must end in silence, otherwise the student learns that
  `look` means "about to speak". `LocalVerifier` keeps its interface; its learned check becomes
  a `look` call whose output and margin produce confirmed / contradicted / unknown. The
  tracker's geometric gate becomes optional evidence.
- **Actions stay gated.** `act` exists in the contract so the student can learn time-critical,
  pre-authorized reactions, but every action still passes the export scanner and the AHP
  command path, and every `act` in training data is human-audited.

## 3. Data: distill the teacher

Protocol and first results: `docs/teacher-distillation.md`. The teacher is a frontier model
(Claude through the research proxy today; a metered key for scale) that sees frames at 2 Hz in
20 s chunks with the journal so far, and returns one decision per tick under the same contract
the student will use. Narrations are hidden from it and used only for audit.

First-pass numbers (2026-09-16):

| Pass | Footage | Ticks | Pushes | Checked against narrations |
|---|---|---|---|---|
| interactive (this session) | four clips at 2 Hz, one 29 s segment at 1 Hz | 109 | 12 | all pushes inside narrated spans after one correction (lid placed on pan read as pan lifted) and one miss at 1 Hz (a plate put down) |
| frontier teacher through the pipeline (`scripts/teacher_label.py`, blind) | all five sources | 138 | 19 | silent on 86 percent of ticks; 18 of 19 pushes inside a narrated manipulation; 10 of 16 durable narrated events get a push of a compatible kind within 1.5 s; 15 of 19 pushes judged right; agreement with the interactive pass 0.52 |

The scorer is `scripts/teacher_agreement.py` (`data/teacher-seed/agreement-2026-09-16.json`).
The four frontier errors and the two misses are analysed in `docs/teacher-distillation.md`
section 4.1 and drive protocol v2: worked examples in the prompt, a second look on every
candidate pickup or placement, goal-conditioned kinds, chunks that always start with a journal,
and kind-aware audits. Teacher-to-teacher agreement at 0.52 is the number to raise before
labeling at scale; the target is above 0.8 on a 1 hour calibration set.

Sources, in the order the teacher should label them (survey and licenses in
`docs/research/streaming-data-2026-09.md` and `docs/research/datasets-2026-09.md`):

1. **CASTLE 2024, the five static streams, one day first (about 50 h).** The only public source
   with fixed indoor cameras running for days over several people in a real home; no labels,
   so the teacher's decisions are the labels and the quiet periods are real. Research-only.
2. **EgoLife exocentric cameras, if the release contains them (verify first).** Fifteen fixed
   GoPros synchronized with the egocentric streams and about 1200 narrated phrases per hour,
   which makes blind teacher labeling auditable at scale. Held out entirely if it is used for
   the EgoLife benchmarks (shared cameras make a participant-disjoint split impossible).
3. **HOMAGE, all 25 h, as the precision audit set.** Frame-precise atomic actions at about 800
   per hour and scene graphs give ground truth for taken, placed and activity-start onsets, and
   its sensors seed the device-state text format. Research-only.
4. **BEHAVIOR-1K 2026 replays, 50 to 100 h re-rendered from one fixed room camera per scene.**
   Exact timestamps for every container, appliance and object-state change, free of human
   labeling; the only cheap source of `act` situations when paired with device rules; useless
   for presence (robot embodiment). Needs OmniGibson and Isaac Sim pinned to the collection
   versions; demos under an MIT card, assets under a research EULA.
5. **The egocentric kitchen footage already on disk**, plus HoloAssist and CaptainCook4D
   (commercially clean) for object and food events; Charades later for home diversity only.
6. **Our own consented recordings**, last, to close the domain gap; no public fixed-camera home
   footage is usable in a product except possibly EgoLife exo, so this tier is not optional.

Device-state text streams: HOMAGE sensors thresholded into transition lines (presence, light,
door), BEHAVIOR predicates polled during replay into Home-Assistant-style entity lines (exact and
free), and the acon96 generator's vocabulary and service-call format with a small Markov
simulator for dwell times. Synthesized device text is paired with real video only where the two
are consistent; otherwise it stays on the simulation tier.

First training set: 100 h of real fixed cameras, 50 h egocentric, 50 h simulation, about 200 h,
roughly 1.44M ticks at 2 Hz of which 1 to 3 percent are journal lines; about 300 to 350M teacher
input tokens at 1.7M per footage hour. Held out and never trained on: CASTLE day 4, EgoLife
entirely, the HoloAssist and CaptainCook4D validation splits used by EgoServe, SmartHome-Bench,
OVO-Bench forward-active and StreamingBench proactive output.

Rules that keep the dataset honest: whole hours are labeled including quiet ones, never
positive-selected clips; device-state text streams are synthesized from sensor logs, simulator
state or a rules engine so the student sees the same modalities as production; splits are
participant- and home-disjoint and frozen before labeling, as in `data/continuous-v2`; a
held-out set is teacher-labeled but never read by training; 5 percent of chunks and every
`act` are human-reviewed; teacher precision against narrations is reported per source before
any training run.

Cost: at 512 px a frame is about 200 input tokens, a 20 s chunk about 8K plus context, an hour
about 180 chunks. The proxy does not report input usage, so the first ten hours run on a
metered key to fix the price per footage hour before scaling to the 200 to 500 hours the
first student needs.

## 4. Training the student

Recipe from `docs/research/streaming-vlm-recipes-2026-09.md`, which found that every open
streaming model that decides for itself when to speak uses one of three encodings, and that
the two working 3 to 4B systems (StreamPro on Qwen3-VL-4B, MMDuet2 on Qwen2.5-VL-3B) use ours:
a silence token in the text stream. The numbers below name their precedent in that note;
"owner's choice" items have none.

- **Stream rendering.** 2 Hz ticks with a text timestamp; frames enter only when the gate
  fires, and runs of frameless ticks longer than 2 s collapse to one span marker so quiet hours
  are cheap; 144 tokens per admitted frame (ablate 64); ASR words and sound-event labels as
  text at their tick; device-state lines at their tick; the student's own lines stay in the
  text stream while older frames are evicted. The silence target is one token per tick; a burst
  is one compact line under 40 tokens with short keys and masked scaffolding, which removes the
  74 percent JSON overhead of the old targets; `act` lines are `act rule_id args`; `look` is a
  fourth decision.
- **Samples and attention.** A sample is the fixed prefix (system prompt, goal, action policy,
  journal so far) plus a 30 s lead-in with loss masked plus 120 s of ticks with targets: six
  consecutive teacher chunks, long enough to exercise "already reported". Plain causal
  attention, no custom mask; at inference the cache keeps the prefix and journal as sinks and
  the last 60 to 90 s of gated frames, or on the Mac the prefix is cached and the window
  re-prefilled, which the StreamingVLM result says is benign. Busy samples are about 20K
  tokens, quiet ones about 2K; cap 24K and pack quiet samples.
- **Silence imbalance is solved by loss weighting, never by dropping quiet data.** Keep every
  labeled quiet hour and mix at least half quiet or near-quiet hours. Per-token weights:
  silence 0.1 to 0.2 (ProAssist's subsampling of silent frames took narration F1 from 30 to 59;
  the one recipe that left silence at weight 1 scores under 4 percent on proactive output),
  transition ticks times 5, decision-bearing burst tokens times 2, scaffolding 0, the `act`
  token times 10 with at least five negative-act contexts per positive, `look` times 2 with at
  least half of its targets ending in silence, lead-in 0. Check after the first run that the
  summed silence loss is one to two times the summed burst loss.
- **What to train.** Stage A, always: the projector fully, LoRA rank 128 on every linear layer
  of the language model, vision tower frozen (every streaming recipe read), one epoch, LoRA
  learning rate 1e-4 and projector 2e-5. Full fine-tuning only past about 20K labeled chunks.
  Stage B, only if held-out state-change recall misses target: LoRA rank 32 on the top third of
  the vision tower, gated on the same-image and reversed-order controls before and after. Below
  10K chunks, two epochs with journal dropout, tick jitter and resolution jitter.
- **Reinforcement fixes timing and duplicates.** Teacher-to-teacher agreement is 0.52, so the
  supervised target means "speak roughly when the teacher speaks"; exact timing and duplicate
  suppression come from GRPO on 2 to 3K contexts with a reward of onset within the 1.5 s audit
  collar, an explicit false-positive term and a replication penalty (MMDuet2's penalty took the
  duplicate ratio from 81 to 99 percent down to 1 to 15 percent). All published successes start
  from a supervised model that already knows silence, which is this order.
- **Decoding and calibration.** Emit a burst only when the silence probability falls below a
  threshold fixed on the quiet-hours calibration set by the certified rule in
  `docs/research/decoding-and-training-2026-09.md` (at most 13 false bursts in 600 quiet minutes
  certifies at most 2 per hour at 90 percent); cap `look` at 10 percent of busy ticks.
- **Compute (derived, not measured).** About 8 to 12 H100 hours per supervised run on 300
  labeled hours; a full cycle of a six-run sweep, GRPO on 3K contexts, controls and `look`
  ablations is about 200 to 350 H100 hours, roughly $400 to $1,050 at 2026 on-demand prices.
  The Runpod runner, staging and hash-verified retrieval already work; teacher labeling cost is
  separate (section 3). The box never trains.
- **Phase-0 gate.** Before any training spend: the candidate base, in the streaming engine,
  under the silent/context contract with the log-probability decision, on our frozen test
  footage and on quiet hours. If zero-shot recall within the collar is near zero or the
  silence margin carries no signal, change the base, not the recipe.

## 5. Runtime and infrastructure

Base model and engine, from `docs/research/base-models-and-runtimes-2026-09.md` (its speed
figures are estimates from published geometry and secondary benchmarks, starred there for
measurement in phase 0):

- **Target: Qwen3.5-4B** (Apache-2.0, thinking off) on mlx-vlm with automatic prefix caching
  on the Mac, llama.cpp Metal as the fallback; on the box, its 4-bit GGUF with the vision
  projector on llama.cpp CUDA, or vLLM with an int4 checkpoint and prefix caching once
  hybrid-state caching is confirmed. It is the only open 2 to 9B model whose backbone already
  has the bounded memory the design asks for: 24 Gated DeltaNet layers with a fixed state of
  about 25 MB and 8 attention layers at 32 KB of cache per token, so a 32K window costs about
  1 GB instead of 4.7 GB for Qwen3-VL-4B; it has time-aware interleaved position encoding with
  an fps parameter, native tool calling, 262K context, and block-aligned tokens that a motion
  gate can drop individually. Its one gap is audio: speech enters as VAD-gated ASR text.
- **Student v1 trains on Qwen3-VL-4B-Instruct unless phase 0 verifies Qwen3.5-4B's toolchain**
  (mlx-vlm video path, hybrid-state prefix caching, LoRA on the hybrid layers). The two notes
  rank these two differently for exactly that reason: Qwen3-VL-4B has measured MLX throughput,
  textual timestamps, mature LoRA tooling and a shipped proactive streamer on the same backbone,
  at the cost of 147 KB per token of cache and no recurrence; the stream format is base-agnostic,
  so switching later costs one re-run.
- **Gemma 4 E4B** only if native audio in one model outweighs the missing timestamp encoding
  and whole-frame-only gating; it is already running here and its grammar tooling is verified,
  but its own failures in iterations 1 and 2 were a recipe failure, not a model verdict.
- **Per-tick budget, one camera, with prefix reuse** (estimated): a quiet tick about 85 ms on
  the Mac and 95 ms on an RTX 3060, a busy tick about 240 and 285 ms, worst single tick with a
  45-token burst about 0.6 and 0.9 s. That is 8 to 30 percent utilization, with headroom for
  rare second looks (a 896 px frame costs about 1 s). Without prefix reuse the same tick costs
  5 to 9 s, which is today's regime; cache reuse across ticks is therefore the single deciding
  requirement. Memory at a 32K window: about 6.3 GB for Qwen3.5-4B, so a 12 GB card keeps room
  for a second camera's context.
- **Eviction that keeps the cache valid**: append for about ten minutes, then rebuild a
  compacted context (system prompt, journal, summarized earlier observations, the last seconds
  of vision) and prefill it once, under 1 percent of the budget. The recurrent state cannot be
  edited, only recomputed, which this policy does.
- **Bursts under a grammar** with a per-request enum for subjects and locations, the silence
  decision from log-probabilities, a second-look call path; the transformers path used so far
  stays research-only.
- **Change gating**: frame differencing and a motion budget first; codec motion vectors read
  from the camera stream when measured to pay for themselves.
- **Virtual house**: datasets replayed as fake cameras (mediamtx and ffmpeg) into the real
  capture path, Home Assistant in a container with emulated devices for the action path, a
  rented 12 GB GPU for the sub-$1K box soak test, Runpod for training.
- **Code mapping**: keep `ProductionExporter`, `ProductionCoordinator`, the journals, the AHP
  manifest, the frozen benchmark data and scorers, `scripts/runpod.py`; add a `StreamObserver`
  (tick loop, cache reuse, silence decision, burst parsing) that replaces `PhysicalLocalModel`'s
  window contract in `run_production.py`; give `LocalVerifier` a second-look learned check; keep
  the tracker as optional diagnostics.

## 6. Evaluation

- **Primary**: false pushes per hour on at least ten held-out quiet hours (no published
  benchmark measures this; it has to be ours); recall and onset latency within a 1.5 s collar
  against teacher labels on held-out sources and kind-aware against narrations where they
  exist; duplicate ratio; per-kind recall; precision of every `act` (all audited); the
  silence rate compared with the teacher's; the `look` rate and its precision gain; the
  same-image and reversed-order controls from the old runs, which must show no events.
- **Usefulness**: the frontier judges the student's journal against its own on the same
  held-out footage (would the coordinator have been misled or left blind).
- **Runtime**: per-tick latency and utilization on the Mac and on the box, a 72 hour soak with
  flat memory.
- **Sanity**: public streaming benchmarks (StreamingBench, OVO-Bench) so a regression in
  general ability is visible.
- **Numbers to beat**: about 1800 false events per hour and recall 0 to 0.034.

## 7. Phased plan with gates

| Phase | Weeks | Work | Gate |
|---|---|---|---|
| 0. Baselines | 1 | Qwen3.5-4B, Gemma 4 E4B and Qwen3-VL-4B in the streaming engines on the Mac: measure the starred rates (encoder time, prefill and decode with prefix reuse), confirm hybrid-state prefix caching and the video path in mlx-vlm and llama.cpp; zero-shot silent/context contract with the log-probability decision on frozen test footage and quiet hours; teacher protocol v2 calibrated to above 0.8 agreement on a one hour set; teacher pass on 10 hours with a metered key | a base with non-zero zero-shot recall and a usable silence margin at under 300 ms per busy tick; measured teacher cost per hour and kind-aware precision against narrations |
| 1. Teacher dataset v1 | 2 to 3 | 100 to 300 hours across tiers 1 to 3, splits frozen first, device-state streams synthesized, audits done | per-source teacher precision reported; held-out set frozen |
| 2. Student v1 | 1 to 2 | stage A supervised run on the section 4 recipe (projector plus LoRA rank 128, silence weight 0.1 to 0.2, transitions times 5, `look` targets), a six-run sweep, the F3 controls; agreement report | on held-out fixed-camera hours: under 20 false pushes per hour, recall above 0.5 within the collar, silence rate within 10 points of the teacher, controls silent |
| 3. Timing and calibration | 1 to 2 | GRPO on 2 to 3K contexts with the collar, false-positive and replication terms; threshold certified on quiet hours; `look` capped and measured | under 5 false pushes per hour at recall above 0.6, duplicate ratio under 15 percent |
| 4. Production wiring | 1 to 2 | `StreamObserver`, second-look verifier, replay of the eight fresh-footage workflows | no false completion; the narrated pickups reach confirmed |
| 5. Box port and soak | 1 to 2 | rented 12 GB GPU, 72 hour soak, power-limited | 24/7 budget holds with flat memory |
| 6. Own footage and consolidation | ongoing | consented recordings, field loop, nightly adaptation as an experiment | field false-push rate tracked monthly |

## 8. Risks and decisions that are the owner's

- Teacher cost and quota: the proxy hit its session limit during this work; scale runs need a
  metered key and a budget per footage hour.
- The teacher's errors are distilled. The first pass shows it is good but not perfect at 2 Hz
  (15 of 19 pushes right, agreement between two teacher passes 0.52); audits, the second-look
  protocol and the reinforcement stage that tolerates label noise are not optional.
- No published streaming model trains or evaluates on fixed home cameras, and none reports
  false alarms per quiet hour; presence and location behaviour on static cameras is untested
  anywhere, so phase 2's gate is the first evidence either way.
- Domain: public footage is mostly egocentric or lab-fixed; presence and location targets need
  fixed-camera sets or simulation, and own recordings eventually.
- `act` has almost no natural examples in public footage; it comes from simulation and scripted
  device rules, so its precision must be measured separately.
- Runtime facts still unverified: prefix caching of the Gated DeltaNet state in every engine,
  llama.cpp's multimodal status for Qwen3.5, the mlx-vlm video path for the candidates, and
  every speed figure in section 5. Phase 0 exists to close them; if hybrid-state caching is
  missing everywhere, Qwen3-VL-4B with q8 cache is the fallback and the box needs 16 GB.
- Newer Qwen families (3.6, 3.8) appeared in September 2026 and were not assessed; a 2 to 9B
  dense release with the same vision stack would supersede the first choice.
- Base model license and dataset agreements (Ego4D, Ego-Exo4D, CASTLE, HOMAGE, BEHAVIOR assets)
  still need signatures; the streaming-data note lists them.

## 9. What we stop doing

- Fitting narration windows into JSON and calling validation loss progress.
- Language-only LoRA on a frozen vision tower.
- Treating a hand-built tracker as the source of truth for what the model may say.
- Judging windows after the fact instead of deciding at the tick.
- Building detector cascades around the model.
