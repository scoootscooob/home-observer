# Model roadmap: one observer model, distilled from a frontier teacher

Status: proposal v2, 2026-09-16. Supersedes the decomposed perception stack proposed earlier the
same day (git history, commit ab28c05) at the owner's direction: the observer is one unified
vision-language model, not a cascade of detectors. Grounded in the measured failures of
iterations 1 and 2, the first teacher pass (`docs/teacher-distillation.md`), and the research
notes under `docs/research/`. Where a number below comes from a note it says so; anything
load-bearing gets re-measured on the target machine before it is relied on. Sections marked
"pending note" are completed when the streaming-recipes, streaming-data and base-models notes
land.

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
              |  {"kind": ..., "subject": ..., "location": ..., "detail": ..., "confidence": ...}
              |  <act> {"command": ..., "args": ...}
   -> the emitted line re-enters the stream (working memory, duplicate suppression)
   -> physical journal -> export scanner -> coordinator (unchanged)
   -> second look: the same model re-reads retained frames at 4 Hz and higher resolution when
      the margin is small or the coordinator asks; its answer is the verification result
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
  proportion to change (frame differencing now, codec motion vectors and encoder-side token
  pruning later, pending note), so tokens per second follow activity, not the clock.
- **Second look is the verifier.** The teacher's `unsure` plus second look and the student's
  small-margin plus second look are the same mechanism. `LocalVerifier` keeps its interface;
  its learned check becomes a second-look call whose output and margin produce
  confirmed / contradicted / unknown. The tracker's geometric gate becomes optional evidence.
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

- **Streaming fine-tuning** (recipe details pending note): overlapped chunks with full attention
  inside a chunk, mimicking sink-plus-window inference; targets are the teacher's ticks
  rendered as tokens, silence included; the loss weight on the silence token is tuned so the
  model neither ignores it nor learns to say nothing; the projector and the vision tower are
  trainable (LoRA on all linear layers at a rank that matters, or full fine-tuning), never
  language-only; 2 Hz input with change gating; sequences long enough to hold a minute of
  stream plus the journal.
- **Reinforcement on timing.** After supervised distillation, GRPO-style fine-tuning with a
  verifiable reward computed from the teacher labels: onset within a 1.5 s collar, one push
  per event, silence elsewhere, penalties for duplicates and for pushes on quiet ticks. This is
  the recipe that moved temporal grounding from about 29 to about 60 mIoU with a few thousand
  samples, and it only works on a base that already localizes in a stream, which is what the
  phase-0 gate checks.
- **Calibration and thresholds.** Temperature or Platt scaling on the silence margin using
  held-out homes; the operating threshold certified on at least ten quiet hours.
- **Compute.** Minutes to hours per run on one H100; the Runpod runner, staging and
  hash-verified retrieval already work. The box never trains; nightly consolidation on own
  footage is a later option.
- **Phase-0 gate.** Before any training spend: the candidate base, in the streaming engine,
  under the silent/context contract with the log-probability decision, on our frozen test
  footage and on quiet hours. If zero-shot recall within the collar is near zero or the
  silence margin carries no signal, change the base, not the recipe.

## 5. Runtime and infrastructure

- **Streaming engine** (base model and engine choice pending note): cache reuse across ticks
  so each tick costs its new tokens only; grammar-constrained bursts with a per-request enum for
  locations and subjects; the silence decision from log-probabilities; a second-look call path.
  The transformers path used so far stays research-only.
- **Change gating**: frame differencing and a motion budget first; codec motion vectors and
  encoder-side token pruning when measured to pay for themselves.
- **Virtual house**: datasets replayed as fake cameras (mediamtx and ffmpeg) into the real
  capture path, Home Assistant in a container with emulated devices for the action path, a
  rented 12 GB GPU for the sub-$1K box soak test, Runpod for training.
- **Code mapping**: keep `ProductionExporter`, `ProductionCoordinator`, the journals, the AHP
  manifest, the frozen benchmark data and scorers, `scripts/runpod.py`; add a `StreamObserver`
  (tick loop, cache reuse, silence decision, burst parsing) that replaces `PhysicalLocalModel`'s
  window contract in `run_production.py`; give `LocalVerifier` a second-look learned check; keep
  the tracker as optional diagnostics.

## 6. Evaluation

- **Primary**: false pushes per hour on at least ten held-out quiet hours; recall and onset
  latency within a 1.5 s collar against teacher labels on held-out sources and against
  narrations where they exist; duplicate rate; precision of every `act` (all audited); the
  silence rate compared with the teacher's.
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
| 0. Baselines | 1 | streaming engine on the Mac with the candidate bases; zero-shot silent/context contract with the log-probability decision on frozen test footage and quiet hours; teacher pass on 10 hours with a metered key | a base with non-zero zero-shot recall and a usable silence margin; measured teacher cost per hour and precision against narrations |
| 1. Teacher dataset v1 | 2 to 3 | 100 to 300 hours across tiers 1 to 3, splits frozen first, device-state streams synthesized, audits done | per-source teacher precision reported; held-out set frozen |
| 2. Student v1 | 1 to 2 | streaming fine-tuning with the vision side trainable; agreement report | on held-out fixed-camera hours: under 20 false pushes per hour, recall above 0.5 within the collar, silence rate within 10 points of the teacher |
| 3. Timing and calibration | 1 to 2 | reinforcement on timing; calibration; certified threshold; duplicate suppression | under 5 false pushes per hour at recall above 0.6 |
| 4. Production wiring | 1 to 2 | `StreamObserver`, second-look verifier, replay of the eight fresh-footage workflows | no false completion; the narrated pickups reach confirmed |
| 5. Box port and soak | 1 to 2 | rented 12 GB GPU, 72 hour soak, power-limited | 24/7 budget holds with flat memory |
| 6. Own footage and consolidation | ongoing | consented recordings, field loop, nightly adaptation as an experiment | field false-push rate tracked monthly |

## 8. Risks and decisions that are the owner's

- Teacher cost and quota: the proxy hit its session limit during this work; scale runs need a
  metered key and a budget per footage hour.
- The teacher's errors are distilled. The first pass shows it is good but not perfect at 2 Hz
  (one confusion resolved only by a second look); audits and the second-look protocol are not
  optional.
- Domain: public footage is mostly egocentric or lab-fixed; presence and location targets need
  fixed-camera sets or simulation, and own recordings eventually.
- `act` has almost no natural examples in public footage; it comes from simulation and scripted
  device rules, so its precision must be measured separately.
- Base model license and dataset agreements (Ego4D, Ego-Exo4D, fixed-camera home sets) still
  need signatures; the streaming-data note lists them.

## 9. What we stop doing

- Fitting narration windows into JSON and calling validation loss progress.
- Language-only LoRA on a frozen vision tower.
- Treating a hand-built tracker as the source of truth for what the model may say.
- Judging windows after the fact instead of deciding at the tick.
- Building detector cascades around the model.
