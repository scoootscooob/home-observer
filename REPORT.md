# Home Observer: Runpod experiment

This is the preserved sensing-plus-action baseline report. The latest implementation and measurements are in the [continuous physical sensing report](PHYSICAL_REPORT.md).

**The single-model architecture runs end to end. Reliable unattended home control is not yet established.** This project implements the proposal as an executable research prototype and retains the measurements needed to assess it.

Open the [interactive results dashboard](reports/index.html), [current release status](reports/release-status.json), or [runnable demo](docs/quickstart.md).

## What was built and exercised

- One pinned Gemma4 E4B model consumes native camera images, microphone audio, device telemetry and recent history. Training and learned inference ran on Runpod.
- Language-only LoRA training preserves the native image/audio encoders. Checkpoints, optimizer state, source snapshots and hashes are retained.
- Independent camera/audio capture uploads bounded windows to an authenticated GPU service. SQLite stores observations, prior state, action reservations and readback results.
- Actual Home Assistant REST calls control an isolated local demo with four template lights and a buzzer. These are software devices, not a physical house.
- A consumer RTX 3090 serves the native multimodal model through vLLM. Two H100 workers support training and evaluation.

The model receives bounded windows and explicit history. This is not an indefinitely recurrent video stream. There is no detector/model cascade in the observer.

## Completed baseline and historical comparisons

| Configuration | Held-out valid responses | Required actions recovered | Incorrect proposed actions | False-action quiet windows |
|---|---:|---:|---:|---:|
| Base, native BF16, batch size 4, earlier policy serialization | 143/160 | 12/36 | 59 | 40/108 |
| Checkpoint300, merged BF16, batch size 4, earlier policy serialization | 144/160 | 7/36 | 1 | 1/108 |

The 160 held-out rows include 144 action-scored synthetic windows and 16 caption-only windows. Missing caption action labels are not treated as negative labels. Both historical configurations failed all 8 paired visual counterfactual checks. These rows characterize the recorded configurations; later corrected inputs and serving backends are separate experiments.

Across 424 annotated observation/entity pairs in the held-out and counterfactual suites, the base correctly reported 25, incorrectly reported 46 and omitted 353. Merged checkpoint300 correctly reported 341, incorrectly reported 47 and omitted 36. Unannotated extra observations are unscored in this measure. The original strict metrics remain available separately; see the [evaluation guide](docs/evaluation-guide.md).

## Training and the input-format correction

The initial 160-step adapter reduced validation loss but missed required actions. A subsequent 300-step run was interrupted by the original cloud supervisor after a transient SSH failure. Its saved weights were recovered. A 100-step continuation explicitly warm-started those weights with a new optimizer and schedule; it is not represented as an exact optimizer resume.

Continuation step 50 was selected using the fixed validation set before its held-out evaluation. In the training callback it produced 7/8 valid responses, recovered 3/4 required actions, made no false action on 4 quiet cases, and recovered all 28 labeled observations. The remaining positive cited a device absent from its visual-only input and was rejected.

Fresh inference initially differed on two visual cases. A controlled comparison found equal policy values but a different JSON field order. Input token IDs changed; image/audio tensors and masks did not. Restoring the authored policy order reproduced all 8 original callback outputs byte for byte, without changing weights, autocast or cache settings. `build_context` now canonicalizes that order, with regression tests. Earlier merge/batch comparisons were confounded by this input difference and do not establish a causal merge or batching regression.

## Trained consumer validation

With the corrected policy format, the unmerged step 50 adapter served through vLLM passed all 8 fixed validation cases: all 4 required actions and all 28 labeled observations, with no false action on the 4 quiet cases. All 8 completion responses identified the registered `observer` adapter alias. The same service passed all 3 authored Home Assistant integration cases, including actual service execution, readback and replay prevention. Full held-out evidence is reported separately when complete.

The vLLM grammar constrains current evidence IDs; native unconstrained generation still rejects one validation case for its nonexistent-device citation. These are separate serving configurations.

## Measured consumer live baseline

The corrected base vLLM decision contract passed all 3 authored Home Assistant cases, including actual state readback and duplicate prevention. Its bounded live run completed 21 accepted windows in 63.32 seconds: **0.332 windows/second**, below the requested 1/second. One Home Assistant action was verified; no window was rejected and no action remained uncertain.

Each of four capture processes produced 125 frames; the model sampled 21 per camera and omitted 104 per camera. Twenty windows included audio. All windows had four distinct decoded images, and uploaded/received byte counts matched. Snapshot retention ended with 8 completed windows, 13 expired windows and no active window.

These four sources are time offsets of **one** public recording. They do not establish simultaneous multiview awareness. The footage has no action/occupancy ground truth; protocol acceptance is not a perception accuracy score.

## Practical limits

Evidence checks establish that cited input IDs exist and that proposed entities, services and arguments are permitted. They do not prove that a cited image supports a claim or execute natural-language policy rules as code. Home Assistant readback establishes execution, not the correctness of the model's reason for acting.

The training mix combines authored synthetic scenes/rules with human COCO captions. Public EgoLife footage supports real image/audio replay without invented labels. This does not establish generalization to a particular house, camera placement or microphone environment. No 72-hour stability run or physical-home validation has been completed. Live media buffers and snapshots are bounded; durable journal/text logs can grow.

## Reproduce and inspect

- [Complete public-data and Home Assistant demo](docs/quickstart.md)
- [Adapter serving and transfer](docs/adapter-serving.md)
- [Training, checkpoints and model controls](docs/model.md)
- [Runpod lifecycle, recovery and cleanup](docs/runpod.md)
- [Source data and annotation scope](docs/data.md)
- [Software verification](reports/software-verification.json)

Provider, inference and Home Assistant credentials remain outside this deliverable. Every reported GPU configuration must be interpreted with its recorded input format, adapter state, decoding contract and hardware.
