# Model roadmap: how to make the local perception actually work

Status: proposal, 2026-09-16. Grounded in the measured failures of iterations 1 and 2 and in
three research notes under `docs/research/` (perception stack, datasets, labeling and data
plan), each with primary sources. Numbers from those notes are marked measured or estimated
there; re-measure anything load-bearing on the target machine before committing to it.

## 0. The decision in one paragraph

Stop treating a generative vision-language model as the detector. Two iterations of LoRA
fine-tuning on Gemma 4 E4B produced a model that cannot emit its own output contract, that
claims an event in every window including windows with no motion at all, and that matches
0 to 3 percent of labeled events on fresh footage. Replace it with a small on-device
perception stack (detector, memory tracker, hand contact, per-track interaction classifier,
depth-based lift test) that produces events per tracked object with calibrated probabilities,
and keep a small VLM only as an occasional constrained verifier. Train the small models on
static-camera footage, starting from the commercially usable public sets and our own
consented homes, labeled by a frontier-model-proposes, human-adjudicates pipeline, and
accept nothing that has not been measured on frozen held-out homes with false events per
hour as a first-class metric. The privacy boundary, watch engine, verifier and exports stay
exactly as they are.

## 1. Why the current model fails (measured on our own runs)

**F1. The contract asks a language model to do what it cannot see.** The output schema requires
boxes, causal evidence citations, calibrated confidence, identity links and boundaries in one
JSON object. In the supervised training targets only 26 percent of tokens carry a decision
(verb, noun, times, description); 74 percent are JSON scaffolding. Validation loss falling
from 2.83 to 0.30 mostly measures scaffolding memorization. Native strict validity is 0 of 242
for base, v1 and v2 alike, so more training could not fix format.

**F2. Abstention is unsupervisable at this token budget.** An empty window's target is
`{"events":[],"summary":"no annotated action"}`, about 10 tokens versus 82 for a positive
window. Quiet windows were 25 percent of the supervised rows but 3.8 percent of the supervised
tokens. The result: v1 emits an event in 31 of 31 quiet test windows, v2 in 29 of 31, and both
run at about 1800 false events per covered hour.

**F3. The model learned scene priors, not motion.** On each fresh test video v2's most common
verb is chosen for 18 to 49 percent of all windows regardless of what happens: "check" in 46 of
94 windows of P28_07 (a verb that never occurs in that video's labels), "put-down" in 47 percent
of P22_105. Quiet windows receive "put-down", "put-in", "put-on", "pick-up". The predicted
nouns are the scene's furniture and cookware (pot, pan, oven, drawer). A direct control on 20
fresh test windows (`reports/physical-v2-temporal-control.json`) settles it: with all eight
frames replaced by the same single image, so that no motion exists, v2 still claims an event
in 20 of 20 windows, and in 8 of 20 the verb set is identical to the real sequence; reversing
frame order leaves the verb set unchanged in 12 of 20. Every one of the 12 quiet windows in the
control received an event. The adapter is a scene classifier with a verb vocabulary, not an
event detector.

**F4. Temporal sampling cannot resolve the events.** Eight frames per three-second window is a
0.43 s gap. The median narrated action on the test videos lasts 0.99 s and 51 percent last
under one second, so a typical event spans two sampled frames, often one. At 140 soft tokens
per frame the model sees about 1100 visual tokens for three seconds of video.

**F5. Data volume and domain.** 195 supervised windows from six kitchens; 24 "pick-up" and 17
"take" examples in total. Head-mounted egocentric footage with constant camera motion for a
product that will use a static camera. v1's pretrimmed pilot taught "every window contains a
pickup" (it says pick-up in 120 of 242 windows).

**F6. The tracker breaks exactly at the interaction.** CSRT with a colour-histogram appearance
guard reaches its 0.65 distance threshold as the hand enters and the object starts to move
(pan: appearance distance 0.58 to 0.63 over the last 0.1 s, box area ratio 1.4 to 1.55, then
"implausible center step"; lid: "appearance changed" at 46.3 s, 1.4 s before the narrated
take). Visible-track coverage of the positive clips was 39 and 21 percent, so the verifier's
geometric gate never opened and the learned check never ran. Translation-only camera
compensation also fires `motion_started` on residual camera motion.

**F7. The model path is not real-time.** Median 12.2 s per three-second window on Metal, a
4.1× real-time ratio, before verification calls. The generative model cannot be the per-window
path on this hardware; it can only be an occasional judge.

**F8. What worked, and should be kept.** Capture, buffering and journaling never lost a frame
across 4348 fresh-footage frames; the export boundary, sandboxed planner and verifier never
promoted a false claim into a completion; the frozen protocol and scorers measure all of the
above honestly.

## 2. Architecture: decompose the problem so each part is learnable and checkable

The single generative VLM was asked to detect, track, localize in time, cite evidence, calibrate
and abstain in one JSON. Every one of those is a solved or well-studied problem for a small
specialized model; none of them is something a 4B language model does well from 200 examples.
The proposal keeps the enforced privacy boundary, the watch engine, the verifier and the
export contract unchanged and replaces what sits behind "perception".

```
static camera (or ESP32 LAN capture)
  -> frame ring + activity gate (unchanged)
  -> [A] open-vocabulary detector (household nouns, prompted by the recipe's objects)
  -> [B] segmentation tracker with memory (identity across occlusion, re-entry)
  -> [C] hand-object contact detector (hand box, contact state, contacted object)
  -> [D] per-track interaction classifier over short crop clips at 8-15 fps
         (pick-up / put-down / open / close / carry / reposition / none) with boundaries
  -> [E] geometry: support-surface model + monocular depth -> "lifted", "resting", "moved"
  -> [F] event assembler: software builds IDs, timestamps, evidence citations, identity links,
         calibrated confidence from classifier probabilities, temporal NMS and hysteresis
  -> physical journal -> watches / export boundary / verifier (unchanged)
  -> [G] small VLM as an occasional judge only: structured questions with enumerated answers
         under a grammar, fed crops and tracker facts, never the per-window detector
```

Why this shape:

- **Identity by construction.** Events are emitted per track (B), so the "learned event has no
  subject" problem disappears and the verifier's geometric gate and learned check refer to the
  same object.
- **Abstention becomes a threshold, not a token.** The interaction classifier (D) has a "none"
  class and a probability; a quiet window is the default state and false events per hour is set
  by a threshold tuned on validation, with temporal non-maximum suppression across overlapping
  windows and hysteresis so one action yields one event.
- **Motion is actually observed.** (C) and (D) run on crops at 8-15 fps, so a one-second
  action spans 8-15 frames instead of two, and hand contact is an explicit signal that a scene
  prior cannot fake.
- **"Clear of support" becomes measurable.** With a static camera the support surfaces are
  estimated once; (E) tests whether the tracked mask separated from its surface and rose in
  depth. The verifier's geometric gate finally has a signal that survives handling.
- **The VLM keeps the job it is good at.** Naming an unfamiliar object, answering a
  planner's structured question about a crop sequence, or describing an event for a human,
  under constrained decoding with enumerated fields, dynamic enums for evidence IDs, and a
  frame budget it can afford. It is never in the real-time path.
- **Calibration is real.** Classifier probabilities can be temperature-scaled on held-out
  homes; the export's uncertainty band then means something.

Contract changes this implies (all behind the existing schemas): `confidence_kind` gains
`classifier_calibrated`; events carry `evidence_ids` from the tracker's frames by
construction; the judged-fields path stays only for the VLM judge outputs. Section 2.1 gives
the concrete components and budget; section 3 the data; section 5 the acceptance metrics.

### 2.1 Concrete components and a per-frame budget (from `docs/research/perception-stack-2026-09.md`)

Target: 480p, camera at 30 fps, every other frame processed (15 fps), Apple M-series with the
GPU, Neural Engine and CPU used as three lanes. Figures marked measured come from primary
sources on Apple hardware; the rest must be re-measured on the M5 Max first.

| Stage | Choice (fallback) | Rate | Cost per call | Lane | License |
|---|---|---|---|---|---|
| Motion gate | frame difference / Vision optical flow at low resolution | 15 fps | about 2 ms | CPU | Apple SDK |
| Hand pose | Vision `VNDetectHumanHandPoseRequest`, 21 joints, 2 hands | 15 fps | about 10 ms | ANE | Apple SDK |
| Open-vocabulary detector | YOLOE-26 s with the household vocabulary baked into Core ML (OmDet-Turbo-Tiny if AGPL is unacceptable; RF-DETR small if the vocabulary is fixed) | 5 Hz plus on demand | 10 to 15 ms estimated from YOLO26 s at 57.7 fps on an M4 Pro (measured) | ANE or GPU | AGPL-3.0 (Apache-2.0 fallbacks) |
| Memory tracker | EdgeTAM (SAM 2.1 tiny via MLX as fallback) with DAM4SAM-style memory rules and detector re-prompting every second; at most 4 active objects | 15 fps | 25 to 40 ms estimated from 16 fps at 1024 px on an iPhone 15 Pro Max and 71 ms per frame for SAM 2.1 tiny on an M2 Max (both measured) | GPU | Apache-2.0 |
| Depth | Depth Anything V2 small, Apple Core ML build | 5 Hz, 15 Hz during a lift candidate | 24.6 ms on an M3 Max (measured) | ANE | Apache-2.0 |
| Contact state | fingertip and palm keypoints inside the tracked mask for N frames; Hands23 detector at 2 to 3 Hz only while a hand is near an object (hold versus touch, tool versus container) | event-driven | 100 to 150 ms estimated | GPU background | MIT |
| Interaction classifier | X3D-S (cheap) or VideoMAE-S (accurate) on 16-frame 224 px object-centred crops; classes pick-up, put-down, open, close, carry, reposition, none | event-driven, a few per minute | 5 to 50 ms estimated | GPU | Apache-2.0 |
| Re-identification | MobileCLIP2-S0 crop embedding on new or lost tracks | event-driven | about 2 ms | ANE | Apple research terms, check |
| Verifier | Qwen3-VL-4B Instruct, 4-bit, MLX, constrained decoding through llguidance; 6 to 8 timestamped frames; answers an enumerated question with a forced "cannot tell" option | event-driven | 0.5 to 3 s; 143 tokens per second decode with one image on an M4 Max (measured) | GPU idle slots | Apache-2.0 |

Critical path is the GPU lane at roughly 37 to 45 ms per processed frame, which leaves headroom
at 15 fps; the whole stack needs under a gigabyte of weights plus about 3 GB for the verifier.
Two things in this table are not what one might expect: Grounding DINO 1.5 Edge weights are
not released, so it is out; and Apple's FastVLM is research-licensed, so it is out of a product.

**Events come from a per-track state machine, which is where abstention comes from:**
idle, approached (hand within a distance), contact (keypoints inside the mask for N frames,
Hands23 confirms hold), lifted (mask depth departs the support plane and the centroid rises),
carried (mask moves with the hand), placed (contact ends and the mask is stationary). Open and
close use the articulated-object track plus the classifier. The system emits "unknown"
whenever the tracker's presence score is low, detector and tracker disagree for more than a
second, or the verifier answers "cannot tell". That replaces both failure modes we measured:
the tracker losing the object as the hand arrives, and the VLM never staying silent.

**What the VLM should and should not be trained for.** If the verifier is fine-tuned at all,
use a reinforcement recipe with a verifiable reward (temporal IoU or answer accuracy) on a
few thousand clips, the Time-R1 and TimeLens recipe, not supervised LoRA on JSON: grammar-
constrained greedy decoding provably distorts the distribution and turns unsupervised fields
into confident hallucinations, which is exactly what our judged-fields completion exposed.
Interleaved textual timestamps outperform other time encodings, and Qwen3-VL already uses them.

## 3. Data: what exists, what is usable, and what we must record ourselves

Full survey with licenses: `docs/research/datasets-2026-09.md`. Headline: no public dataset
matches a consumer wall camera in many homes with dense interaction labels and a commercial
license. Every close content match is non-commercial, and every commercially clean set is
egocentric, robot-embodied or a lab tripod capture. So the plan is three-layered.

### 3.1 Public data for pretraining (license conclusions as found; confirm in writing)

| Use | Dataset | Why | License conclusion |
|---|---|---|---|
| Static-camera appearance and per-object identity | **Ego-Exo4D**, cooking takes, exocentric views (about 650 cooking takes, 4 to 5 tripod GoPros each, 189K timestamped atomic descriptions, 1 fps exocentric masks with persistent track IDs) | the only large corpus of real people manipulating real kitchen objects seen from stationary cameras | conditional yes: the agreement permits commercial model development; the data itself may never appear in a product or be redistributed; sign at ego4d.dev and archive the executed text |
| Temporal boundaries and state change, balanced negatives | **Ego4D Hands and Objects** (PRE/PNR/POST frames, about 21K negative clips) plus **HoloAssist** (169 h, fine-grained intervals) | closest existing supervision for outcome verification and abstention | Ego4D conditional yes as above; HoloAssist CDLA-Permissive-2.0 |
| Fixed multi-camera human manipulation | **RH20T-C** (scenes 1 to 5 only), **DROID** (564 real household scenes, two fixed stereo cameras), **OakInk2** (6-DoF object ground truth) | commercially clean analogues of a counter camera watching hands | CC BY-SA 4.0, CC BY 4.0, CC BY-SA 4.0; never touch RH20T scenes 6 to 10 |
| Not usable | Something-Something v2, Toyota Smarthome, HOMAGE, Charades, MPII Cooking, EPIC-KITCHENS and VISOR, HD-EPIC, EPFL-Smart-Kitchen-30 | the right content, the wrong license | non-commercial; HD-EPIC and EPFL-Smart-Kitchen-30 carry conflicting statements, treat as non-commercial until the authors confirm |

Egocentric EPIC-KITCHENS stays as a stress test only. The pilot and continuous benchmark we
built keep their value as diagnostics; they stop being the training domain.

### 3.2 Held-out evaluation

Toyota Smarthome Untrimmed is the best public acceptance set on content (190 h, seven fixed
cameras in a real apartment, dense concurrent labels) but requires a commercial license by
email before it can evaluate a commercial system; EPFL-Smart-Kitchen-30 (nine fixed cameras,
33.8 fine-grained segments per minute) would be the densest static-kitchen benchmark if its
license conflict resolves to CC BY 4.0. Neither replaces our own frozen homes.

### 3.3 Our own footage, and how it gets labeled (from `docs/research/labeling-and-data-plan-2026-09.md`)

Generalization evidence says homes and camera placement matter more than hours: within one
apartment, changing the camera drops a strong model from 53.4 to 34.9 mean per-class accuracy,
and halves per-frame mAP on untrimmed video. Plan for breadth, not depth in one kitchen.

| Phase | Homes | Recording | Labeled | Purpose |
|---|---|---|---|---|
| A. Pilot | 2 (team) | 2 rooms, 2 placements, about 170 h raw per home over a week | 10 h per home exhaustively, double-annotated | calibrate the pipeline's error rate and human time |
| B. Breadth | 8 to 12 consented households | 1 to 2 weeks each, 2 placements per room, day, lamp and night lighting | 60 to 100 h selected by active learning; every quiet hour gets the cheap pass; at least 40 percent negatives; about 10 percent scripted confusables (reach without pickup, slide without lift, occlusion) | training and validation diversity |
| C. Acceptance | 3 to 4 households never used above | at least 24 h each, all lighting, one unseen placement | 100 percent exhaustive, double-annotated, frozen | miss rate and false events per hour per home |
| D. Synthetic (optional) | | 20 to 50 h of renders for occlusion, viewpoint, rare objects | labels inherited from the simulator | pretraining and augmentation only, with ablations |
| E. Field loop | consenting deployments | detector-selected clips: low-margin scores, detector-tracker disagreements, quiet-hour firings | 5 to 10 h per month adjudicated | hard-negative mining and drift tracking |

Labeling pipeline: on-device blur and a participant deletion window before anything leaves;
Gemini in batch mode proposes events with `MM:SS` boundaries, verb, object phrase, contact
flag and lifted/supported state as strict JSON at 1 fps (about $0.4 to $2 per footage-hour;
frontier models reach only about 53 mIoU on re-annotated Charades and collapse on long sparse
video, so proposals, never final labels); a hand-contact detector snaps boundaries; SAM 3
text prompts create masklets, propagated and repaired with SAM 2 or Cutie inside CVAT; humans
adjudicate every disagreement or low-confidence window (expect 20 to 35 percent) plus a
10 percent random audit, and double-annotate the acceptance set. All-in cost is dominated by
people: roughly $10 to $20 per footage-hour, 30 to 90 human-minutes per active hour, a few
minutes per quiet hour. Object boxes, masks and identities come from the SAM family, never
from the VLM, which is measurably poor at spatial grounding.

Synthetic data is worth using only for what it is good at: object masks, articulation state,
viewpoint, lighting, occlusion and class balance, from commercially clean generators
(Infinigen Indoors, ProcTHOR, RoboCasa, Isaac Sim with asset-terms review, ReplicaCAD).
No clean generator renders plausible human hands manipulating objects; SMPL-X bodies and
Unity SyntheticHumans are non-commercial, and sim-to-real gaps for action recognition are
large when viewpoints differ. Keep the contact and lifted-versus-supported channels grounded
in real footage.

## 4. Infrastructure

- **Local runtime.** Apple silicon serves everything: detector and tracker as Core ML or
  MLX, classifier as Core ML, depth as Core ML, the VLM judge through llama.cpp or MLX with a
  grammar. Per-frame budgets are set for 480p at 10 fps with the tracker as the pacing item;
  the activity gate keeps the detector off during quiet periods. An ESP32-S3 camera is a
  capture and cue peripheral over the LAN, speaking AHP to the same node.
- **Training.** Small models train in minutes to hours on one H100: a detector fine-tune,
  a clip classifier, a geometry head. The Runpod runner, staging and hash-verified retrieval
  already work; each run is bounded and the acceptance set is never touched by training.
- **Labeling.** A research-only labeling service (frontier video model plus segmentation
  propagation plus a review UI) produces versioned datasets with the same manifest, hash and
  split discipline as `data/continuous-v2`.
- **Evaluation.** The frozen protocol, stream scorer and matched-comparison tools stay; add
  per-track identity scoring once mask tracks exist, and a continuous "false events per hour"
  dashboard on the acceptance homes after every training run.

## 5. Evaluation and acceptance targets

- Frozen acceptance homes, whole homes never used for training or tuning, versioned and never
  rewritten; report per home, per placement, per lighting.
- Event recall at temporal IoU 0.5 with boundary error in seconds; false events per hour on
  exhaustively covered time; miss rate versus false-alarm rate curves (the ActEV convention),
  identity switches against mask tracks, lifted/supported accuracy on adjudicated outcomes,
  calibration error on held-out homes, and source-frame to verified-response latency.
- Shipping tiers: at most one false informational notification per home-day and one false
  escalation per home-week, with confirmation gating before anything escalates to a person.
  Field deployments of comparable systems produced 3 to 85 false alarms per day per person
  when lab numbers looked fine; the metric matters more than any published threshold.
- The current numbers to beat are not subtle: recall 0 to 0.034, about 1800 false events per
  covered hour, 0 of 31 quiet windows abstained.

## 6. Phased plan with gates

| Phase | Weeks | Work | Gate to pass |
|---|---|---|---|
| 0. Measure the stack on the target Mac | 1 | EdgeTAM Core ML versus SAM 2.1 tiny MLX at 512, 768 and 1024 px with 1 and 4 objects; YOLOE-26 s versus OmDet-Turbo at 480 and 640; Hands23 on MPS; Qwen3-VL-4B with prefix caching; X3D-S versus VideoMAE-S on crops | a 15 fps budget that closes, or a documented fallback at 7.5 fps |
| 1. Tracker and contact replace CSRT | 2 | memory tracker with detector re-prompting, hand pose and contact geometry, state machine emitting motion/contact/placed with "unknown"; rerun the eight fresh-footage workflows | visible-track coverage above 80 percent on the pan and lid clips; the geometric gate opens on the real pickups; no false completion |
| 2. Lift test and verifier | 2 | support-surface model, depth-based lift check, Qwen3-VL-4B verifier under a grammar with a forced abstain option; wire into `LocalVerifier` | positive workflows reach `confirmed` on the narrated pickups; the plate negative stays `contradicted` or `unknown` |
| 3. Pilot data and labeling pipeline | 3 | two team homes, consent and blur tooling, Gemini proposals, snapping, SAM masklets, CVAT adjudication; measure human minutes per hour | pipeline error rate known; 20 labeled hours with negatives |
| 4. Interaction classifier v1 | 2 | X3D-S or VideoMAE-S pretrained on Ego-Exo4D exocentric crops and Ego4D state-change clips, fine-tuned on pilot crops; temperature scaling; temporal NMS and hysteresis | on the pilot's held-out home: recall at IoU 0.5 above 0.5, under 5 false events per hour, calibration error under 0.1 |
| 5. Breadth collection and acceptance freeze | 6 to 8 | 8 to 12 homes, 60 to 100 labeled hours, 3 to 4 frozen acceptance homes | acceptance set frozen and double-annotated before any model sees it |
| 6. Acceptance and field loop | ongoing | ship tiers with confirmation gating; on-device active learning with consent | tier targets met on frozen homes; drift tracked monthly |

Rough spend: Mac-side work is free; H100 fine-tunes of the small models are minutes to a few
hours each (tens of dollars); labeling is the real cost, a few thousand dollars for 100
adjudicated hours at the stated human rates plus a few hundred dollars of API and GPU time.

## 7. Risks and licensing decisions to make now

- Detector license: YOLOE-26 is AGPL-3.0; decide between an Ultralytics enterprise license,
  OmDet-Turbo-Tiny (Apache-2.0), or fine-tuning RF-DETR on our own footage.
- Dataset agreements: sign Ego4D and Ego-Exo4D (separately) and archive the text; email
  Toyota, Bristol (HD-EPIC) and EPFL before using their data for anything commercial.
- MobileCLIP2 and other Apple research weights: check the model terms before shipping.
- Synthetic humans: budget for a commercial character license if realistic hands are needed.
- Consent, blur and deletion windows are not optional; the labeling service handles media only
  in research homes with written consent, and production media never leaves the device.

## 8. What we should stop doing

- Fine-tuning a generative VLM to emit detection, tracking and calibration as JSON.
- Supervising abstention with a ten-token empty target next to eighty-token positives.
- Training on egocentric footage for a static camera and hoping for transfer.
- Training on trimmed action clips: they teach that every window contains an action.
- Reporting validation loss as progress; the only progress metrics are the acceptance ones.
