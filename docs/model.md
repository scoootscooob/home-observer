# Native model and training

The implementation uses `google/gemma-4-E4B-it`, pinned at revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`. It processes raw images and audio inside
one pretrained multimodal network and outputs text JSON. Speech synthesis is not
implemented. This is a bounded-window observer: each call receives fresh media,
explicit household state, recent journal events and an action policy. It does not
retain or manipulate an infinite KV cache.

E4B has approximately 8B total parameters including large embedding tables, despite
its 4.5B effective parameter label. A successful H100 run does not establish a 12GB
consumer-GPU fit or four-camera throughput. Measure the exported adapter on the
intended hardware before selecting hardware or sampling rates.

## Setup (NVIDIA GPU)

Use Python 3.11 or 3.12 and an Ampere-or-newer GPU with BF16 support. For the H100
CUDA 12.8 environment:

```sh
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
```

`requirements-gpu.txt` pins Transformers 5.17.0 and PEFT 0.21.0. The Gemma4 shared
KV-state implementation was inspected: older releases had incorrect
`use_cache=False` behavior, which could corrupt training. `scripts/model_smoke.py`
explicitly compares cached and uncached logits on the real model before training.
The public model and processor were accessible without gated-model approval during
implementation. `HF_TOKEN` is optional for download rate limits; never put it in
command arguments or artifacts.

## Run the full GPU smoke

```sh
python scripts/model_smoke.py \
  --dataset data/fixtures/train.jsonl \
  --validation data/fixtures/validation.jsonl \
  --dataset-root data/fixtures --policy data/fixtures/policy.json \
  --output artifacts/run --steps 2 \
  --quantization none --inference-quantization 4bit
```

The script performs baseline predictions, two real optimizer steps, saves a LoRA
adapter and processor, releases the model, loads the adapter onto a fresh quantized
base, predicts again and exercises the authenticated `/health` and `/observe` API.
It checks that saved LoRA B tensors actually changed. `smoke_report.json` separates
`pipeline_completed` from `acceptance_pass`: malformed JSON or invented evidence
can fail acceptance even when training executes. Baseline and adapted raw outputs,
latencies, token counts, memory metrics and training provenance are retained.
Synthetic fixtures establish plumbing, not real-world model quality.

## Train on annotated windows

```sh
python -m home_observer.train \
  --train data/train.jsonl --validation data/validation.jsonl \
  --dataset-root data --policy data/policy.json \
  --output artifacts/adapter --quantization none --max-steps 100
```

For a smaller GPU, `--quantization 4bit` uses NF4 with frozen language base weights;
vision and audio encoders and multimodal projection layers remain unquantized.
On H100, BF16 (`none`) avoids quantization overhead during training.

Public caption rows can instead use `target={"summary":"human caption"}` and
`supervision_mask={"summary":true,"actions":false,"observations":false,"noop":false}`.
Only the summary prefix and text receive loss. The partial object closing brace and
end-turn tokens are masked, so early termination and absent action/observation
fields receive no direct supervision. This masking alone does not guarantee valid
full-Decision generations; generation quality is checked separately. The same
full-Decision system prompt remains in place.

The trainer:

- Accepts only explicitly labeled rows; absence of an annotation never becomes a
  negative example or a no-op.
- Uses `window`, a separately supplied policy and optional validated historical
  `runtime_context` for input. Targets, teacher outputs, annotation metadata and
  row IDs do not enter the prompt. Prior state/history default to empty.
- Requires disjoint recording/scenario `group_id` values across train and validation.
- Rejects media outside the dataset root, future frames/audio/events, excess audio
  duration, and excess image/context budgets. It never silently truncates an image
  sequence or labels to make a row fit.
- Adds LoRA only to language attention/MLP projection modules. Modality encoders,
  embeddings and base parameters remain frozen.
- Computes assistant-only loss using a token-exact check of the expanded multimodal
  generation prefix. Prompt, padding and image/audio special-token positions get
  `-100`; JSON completion and end-turn tokens are supervised.
- Uses microbatch size one and gradient accumulation. It does not incorrectly pad
  heterogeneous audio/image batches together. `--max-input-tokens` includes the
  completion during training.
- Reports held-out loss before/after, input hashes, group counts, annotation methods,
  exact library versions and the saved LoRA target names. Loss improvement alone is
  not an action-accuracy evaluation.

## Serve and predict

```sh
export HOME_OBSERVER_API_TOKEN='a-long-random-token'
python -m home_observer.serve --dataset-root data \
  --adapter artifacts/adapter --host 127.0.0.1 --port 8000

python scripts/model_predict.py request.json --dataset-root data \
  --adapter artifacts/adapter --output response.json
```

`POST /observe` accepts `InferenceRequest` from `CONTRACT.md` and returns
`{decision, metrics}`. `GET /health` reports readiness. If a token is configured,
both endpoints require `Authorization: Bearer ...`. Non-loopback binding refuses
to start without a token. Runpod access should use an SSH tunnel or TLS reverse
proxy; this application does not implement TLS. All media paths must resolve inside
`--dataset-root`, including absolute paths and symlinks. Remote media URLs are not
loaded. The API proposes actions; the separate policy/HA adapter decides whether
execution is allowed. Malformed model JSON and unknown evidence IDs cause an error,
not an invented no-op. There is no runtime adapter-reload endpoint.

## Limits and verification

Default limits are eight frames, thirty seconds of combined audio, 140 visual
soft tokens per image and 8192 input tokens. Source timestamps and actual audio
file lengths must agree within 100ms. Multiple frames are provided in their listed
order; the text metadata supplies camera/timestamp/evidence mapping. One model
instance serializes generation to avoid concurrent GPU memory spikes.

Local checks performed during implementation: real Google processor with two images
and one second of audio, including exact assistant-only masking; dependency-light
unit tests for input leakage, future media, group overlap, media root checks, LoRA
isolation and API authentication (10 passed). A tiny randomly initialized native
Gemma4 graph also passed the full Trainer lifecycle (evaluation, one training
step with finite gradients, evaluation, and adapter save), with shared KV layers
and gradient checkpointing. Transformers 5.17 uses `warmup_steps`; this trainer
sets it to `ceil(max_steps * 0.05)`. The first warmup step has zero learning rate,
so use at least two steps to verify changed adapter weights. No local full-weight GPU claim is made. Actual
GPU evidence is written by the smoke script on the pod.

Native audio support is demonstrated primarily for speech recognition/translation
in the model card. Household sound classification, cross-camera identity,
interventions and prolonged replay behavior require separate labeled evaluations.
The initial adapter does not learn raw waveform representations or new speech output.

## Primary references inspected

- [Google model card](https://huggingface.co/google/gemma-4-E4B-it): native image/audio,
  Apache 2.0, effective versus total parameters, modality order and duration limits.
- [Transformers Gemma4](https://huggingface.co/docs/transformers/main/en/model_doc/gemma4):
  processor and conditional-generation APIs.
- [PEFT quantization](https://huggingface.co/docs/peft/main/en/developer_guides/quantization):
  quantized base preparation and language LoRA.
- [Unsloth Gemma4 training](https://unsloth.ai/docs/models/gemma-4/train): encoder freezing
  and known historical cache/FP16 issues. This implementation uses Transformers/PEFT
  directly rather than mixing Unsloth monkey patches into the serving runtime.

## GPU throughput and soak benchmark

Run one model instance across both a static and an all-moving four-camera source:

```sh
python scripts/model_benchmark.py \
  --adapter artifacts/adapter --output artifacts/benchmark-four-camera \
  --duration-s 120 --arrival-interval-s 1 --fixture-cases quiet,moving \
  --fixture-audio --quantization 4bit --max-new-tokens 256 --power
```

This command schedules one window per second **per case**, each containing one
frame from each of four procedural cameras. In `quiet`, all cameras repeat static
images; in `moving`, geometry and pixels change in all four views every source
window. `--fixture-audio` adds a native one-second silence waveform. These are
synthetic stress inputs, not household accuracy evidence. Omit `--adapter` to
measure the baseline in a separate output directory. The model is loaded once per
command and reused for all requests; one warmup per case is logged but excluded
from aggregate timing.

For real public frames, use the existing input windows without camera duplication:

```sh
python scripts/model_benchmark.py \
  --input data/coco-household/test.jsonl --dataset-root data/coco-household \
  --source-kind real_public --output artifacts/benchmark-public \
  --adapter artifacts/adapter --duration-s 16 --max-windows 16 \
  --max-new-tokens 256 --power
```

A one-image COCO window remains a one-camera replay. It does not substantiate a
four-camera deployment. The report explicitly identifies source kind and actual
camera coverage; it does not combine fixture and public accuracy claims.

### What the scheduler measures

The source uses a monotonic arrival clock independent of inference completion.
While the GPU handles one window, later windows continue to become due. The
consumer preserves FIFO order and reports queue wait and backlog. It never shifts
the next arrival to the end of inference or silently drops stale frames. After the
finite source finishes, `--drain-s` allows a bounded catch-up period. If backlog
exceeds `--max-backlog`, processing stops and reports outstanding IDs. Planned,
arrived, attempted, completed and unprocessed windows are separate counters.

`can_keep_pace` cannot be true if a request's service time exceeds the arrival
interval, a request finishes after its next arrival deadline, a request fails, or
observed throughput is below the requested rate. Thus a nominal 1Hz source can
correctly report 0.2 processed windows/s and 0.2 encoded frames/s per camera.
Source duration includes the final frame's interval to avoid inflated rates on
short, fast runs.

The benchmark records:

- Wall service time, preprocessing time and model inference time with p50/p95,
  plus arrival-to-completion delays and backlog. Image/audio encoder GPU work is
  included in model inference; preprocessing covers media loading, processor
  feature preparation and transfer.
- Actual input/output token counts and returned processor tensor counts. The
  benchmark instruments `apply_chat_template` without changing its outputs;
  camera frame counts are recorded only after successful native inference with
  a matching encoded image count. Visual/audio token counts are separate.
- Per-camera encoded-frame totals and actual encoded frames per wall second.
- CUDA allocated/reserved memory before and after each request, peak allocated
  memory, and first/last/min/max trends. A flat short run is not proof against
  leaks over days.
- Optional one-second `nvidia-smi` power, utilization and memory samples. These
  are whole-device readings and may include other GPU processes. Per-case mean,
  p95 and maximum watts are reported when samples are available.
- Raw invalid model outputs, schema/evidence errors, original input evidence
  mapping and source provenance. Invalid outputs are not rewritten to no-ops.

Every request is flushed and fsynced to `requests.jsonl`; source windows are saved
before execution to `scheduled_windows.jsonl`. `cases.jsonl` is written after each
case. Final aggregates are in `benchmark_report.json`. A supervising parent
process enforces `--load-timeout-s` and `--request-timeout-s`, terminates a stalled
GPU child, and leaves `interrupted.json` with recoverable logs. On an interrupted
run, scheduled IDs absent from completed logs remain unprocessed. CUDA model state
is discarded between process runs; the harness does not alter `use_cache` behavior.

Memory, source length and run time are bounded by `--max-windows`, `--max-backlog`,
source/drain deadlines, output-token limits and the parent timeout. To inspect the
procedural source and manifest without loading GPU weights, pass `--prepare-only`;
its output explicitly states `gpu_executed=false`.

## Batched independent windows

`NativeModel.generate_batch(requests)` runs one genuinely batched native generation
and returns a list of `(raw_text, metrics)` pairs. `observe_batch(requests)` validates
all corresponding Decisions against their own input evidence; an invalid item
raises an indexed error. `generate_text` and the single-request API retain their
existing behavior. There is no automatic batching delay in the HTTP API yet.

Each row is an independent bounded session. Images/audio and context remain
isolated by batch masks. The initial interface requires equal image counts and equal
audio-item counts per row; token lengths may differ. The real processor receives a
list of conversations, uses left padding, and retains per-row attention/modality
masks. Completion decoding starts at the **padded** input width. Per-row input-token
counts exclude prompt padding; output-token counts exclude generated trailing pads.
The processor's padding setting is restored after the call. One shared lock
serializes both batched and single generation. Default maximum batch size is four;
`ModelConfig.max_batch_size` allows at most eight.

Measure batch 1/2/4 on the same model instance:

```sh
python scripts/model_batch_benchmark.py \
  --adapter artifacts/adapter --output artifacts/batch-benchmark \
  --batch-sizes 1,2,4 --iterations 3 --warmup-batches 1 \
  --fixture-cases quiet,moving --fixture-audio --quantization 4bit \
  --max-new-tokens 256 --power
```

Omit `--adapter` for a baseline run. Public replay supports `--input`,
`--dataset-root` and explicit `--source-kind`, as in the soak benchmark. Enough
unique source windows must be available for the largest batch and iteration count;
this tool does not duplicate public images to fill a batch. Default procedural
windows each contain four camera frames; warmup batches are excluded from timing.

The comparison reports windows/s, full per-event service latency p50/p95, input
padding, token counts, preprocessing/model time, invalid raw outputs, allocated and
reserved memory and optional power. `batches.jsonl` is flushed after every batch,
`comparisons.jsonl` after each case/size, and `batch_benchmark_report.json` contains
relative throughput compared with batch one. A parent process enforces loading,
batch and total-time limits and preserves logs on failure.

**Every ready event waits for the full batch generation.** Dividing batch time by
batch size gives amortized processing capacity, not individual event latency. The
benchmark assumes all requests are already available. Grouping consecutive windows
from one 1Hz source additionally makes the first event in batch four wait three
seconds for batch formation (1.5 seconds average wait); those waits are reported
separately and are not hidden in the service-latency figure. Independent sources
that arrive together can avoid that formation wait. Increased throughput alone
does not establish a 1Hz response deadline or a consumer GPU's performance.

Local validation of this interface used the real Google processor with two sessions
of different lengths: 1367 and 1421 real input tokens, eight image items, two native
audio items and correctly aligned left padding. Four additional dependency-light
tests cover bounds, padding restoration on exceptions and completion-token counts.

## Release evaluation contexts

`scripts/evaluate_release.py` validates every source row and media path before
loading GPU weights. `--data-root` defines the shared media boundary, and
`--prepare-only` writes the exact selected inputs and source hashes without GPU
execution. The default run retains all held-out and counterfactual rows; optional
sampling selects IDs deterministically and then preserves source order.

The labeled suites describe independent snapshots. Each gets an opaque external
recording ID and an empty prior observation journal, while preserving its target,
window ID and evidence. This matters because the runtime prompt suppresses facts
already present in the journal: repeatedly scoring snapshot labels as new events
would penalize correct suppression of unchanged observations. The evaluation
manifest records this context choice explicitly. It measures snapshot accuracy,
not the accuracy of change detection in a continuous recording.

`--batch-size 2` or `--batch-size 4` runs those snapshots as genuine native batches.
Each request is captured through the production Observer with its own empty prior
journal, then its measured output goes through the same policy and simulated action
checks. Targets are preserved. Every event's reported latency includes the full
batch inference time, not batch time divided by row count. These snapshots are
already available; this is capacity measurement and does not include streaming
batch-formation delay. The public chronological suite always remains sequential.
`--policy` chooses the deployed policy explicitly, and its value/hash are recorded
in the evaluation manifest.

Public video replay is a separate chronological suite with persistent journal
state and no targets. It reports timing, valid output and policy behavior; it
cannot establish perception accuracy. Raw native outputs and parse errors are
fsynced after each request in `native-outputs.jsonl`, including invalid outputs.
Caption-only labels remain caption-only and do not acquire fabricated actions or
no-op labels during evaluation.

## BF16 adapter merging

Inference preserves the dynamic LoRA adapter by default, including BF16.
`ModelConfig(merge_adapter=True)` explicitly opts into PEFT `merge_and_unload` and
`safe_merge=True`. The serving/evaluation CLI exposes `--merge-adapter` for controlled
comparisons. Four-bit inference keeps the dynamic adapter; merging a
quantized model is deliberately unsupported. `NativeModel.merge_adapter_weights()`
can merge an already loaded BF16 adapter while holding the same inference lock.
Metrics record whether the adapter has been merged.

Merging removes the additional LoRA matrix multiplications during decoding. It can
also change BF16 rounding, so identical greedy output is a measured property of a
sample, not a universal guarantee. `scripts/model_merge_benchmark.py` preserves two
before/after raw generations and first-token logit differences, then checks four
unchanged full-policy snapshots through the normal policy and simulated execution
path. It reports actual generation time and output lengths separately; output
length changes must not be presented as an implementation speedup.

## Causal training context

Ordinary rows start with empty prior state and recent events. Authored historical
context uses only the explicit `runtime_context` object: `version: 1`, `state`,
`recent_events`, `source_window_end`, and `source_group_id`. Its source group must
equal the training row's group. Every state fact's `observed_at` and every recent
event timestamp must be at or before `source_window_end`, which must be strictly
before the current window starts. Training rejects malformed or future context.
Arbitrary top-level `state`, teacher outputs, current targets and provenance remain
excluded from the model request. Historical provenance fields are validated but
do not enter the prompt; only state and events do. Context-free public captions
remain valid without this object.

The v2 training experiment uses `--decision-field-order actions-first`. Full targets
serialize as actions, observations, noop, then summary; values and evidence remain
unchanged. Caption-only targets keep their partial summary supervision. This tests
whether asking for the action choice before a repeated summary reduces no-op
collapse; a lower teacher-forced loss alone does not establish that it worked.

`--generation-validation <validation.jsonl> --generation-eval-every 100` runs up to
16 fixed validation rows after a durable checkpoint every 100 steps and at the
final step. Each checkpoint retains raw outputs, targets, strict parsing and
policy results, plus action/observation precision and recall. The callback uses the
same in-memory unmerged adapter, temporarily switches to evaluation mode, and
restores training afterward. Checkpoint selection should compare these measured
decisions; the final adapter is not automatically the best checkpoint. The final
training report records all checkpoint reports, serialization choice, data/policy
and system-prompt hashes and training settings.

## Recovering interrupted training

New training runs save full Trainer checkpoints under
`<output>/checkpoints/checkpoint-N`, every `--checkpoint-every 100` steps and at
the final step. They contain adapter weights, optimizer, scheduler, random state,
Trainer step/data-loader progress, training arguments and processor. A final
`resume_manifest.json` hashes these files and the training inputs, including media
bytes, model settings/revision, data JSONL, policy, system prompt, implementation,
hyperparameters and framework versions. Generation validation runs after this
marker is saved and restores the training random state afterward.

Repeat the original training command with
`--resume-from-checkpoint <output>/checkpoints/checkpoint-100` to restore progress.
Keep the original total `--max-steps` and all other training flags. A different
dataset root is allowed if relative media paths and bytes remain identical. Resume
rejects missing/corrupted state and input/configuration changes before loading the
GPU model. This is single-process, single-GPU recovery; CUDA kernel nondeterminism
can still prevent bit-for-bit numerical reproducibility. A tiny native Gemma4 CPU
test verifies interrupted/resumed training produces identical final LoRA tensors
to an uninterrupted run, including optimizer step continuity.

The earlier v1/v2 `adapter/checkpoint-100`, `-200`, and `-300` directories contain
inference adapters only. Their optimizer and random state were not saved, so they
cannot be exactly resumed. `--warm-start-adapter <adapter-directory>` explicitly
loads trainable LoRA weights into a **new** training run with a new optimizer and
schedule. Parent adapter/configuration hashes and this reset are recorded in the
manifest and final report. Rank and verified language projection targets must
match. To exactly resume a later checkpoint of a warm-started run, repeat its
original warm-start flag as well as `--resume-from-checkpoint`; Trainer then
restores the later full state. Warm starting is a separate experiment, not recovery
of the interrupted optimizer trajectory.

The bounded continuation experiment starts from v2 checkpoint300 for 100 additional
steps at learning rate `2e-5`, with unchanged v2 training rows/policy/prompt and
actions-first serialization. It evaluates the same fixed 8 validation examples at
50 and 100 steps (`--checkpoint-every 50 --generation-eval-every 50`). This choice
was made from validation behavior before inspecting the full heldout test results.

## Batch controls and BF16 export

`scripts/model_batch_parity.py` reconstructs a specified four-row release batch
through the production observer and requires its complete request hash to match.
It runs batch4 and four single requests on one loaded model, retaining raw outputs,
action/observation sets, evidence validity and policy rejections. The recorded base
control matched all four action and observation sets, including the tested false
action. One evidence citation changed and failed validation in the single request;
raw text matched in two of four cases. These finite results do not establish
universal equivalence between single and batched BF16 decoding.

`evaluate_release.py --batch-size 8` is available for a GPU with sufficient memory.
The continuation controller first checks freshly loaded unmerged singles against
saved single-request validation, then checks batch8 and batch4. Each batch size is
eligible only if it fits and action/observation sets and validity match; failed
batch controls fall back to matched singles. Initial continuation50 comparisons
changed two visual cases in batch8, batch4 and freshly loaded singles alike. Those
runs used a different policy field order from the training callback; they do not
isolate a batching or merging effect.
Every event retains the full batch latency. Batch formation wait is excluded from
these preloaded snapshot tests, so their throughput is not a live 1 Hz guarantee.

`scripts/model_select_checkpoint.py` selects continuation50 or100 using fixed
validation only: fewest false proposed actions, highest action F1, highest valid
response rate, highest supervised observation recall, then earlier step on an
exact tie. Selection creates a candidate for heldout evaluation, not deployment
acceptance. Step50 won this comparison: three of four required actions and all
28 observation labels, versus two actions and 26 observation labels at step100.
Both were valid on seven of eight windows and proposed no false actions on four
quiet windows. Step50's remaining failure cited a motion sensor absent from a
visual-only current window; evidence validation rejected it.

`scripts/model_export.py --adapter <checkpoint> --reference
<checkpoint>/generation-validation.jsonl --output <fresh-directory> --batch-smoke`
merges an unquantized adapter, compares merged fixed-validation outputs with the
saved unmerged outputs, and exports the full model and processor in Hugging Face
format. `export-manifest.json` is written last with file sizes and SHA256 hashes.
The export is BF16 model weights, not a running service or an approval to execute
Home Assistant actions. Material action, observation or validity differences are
reported explicitly; raw outputs are retained without repair.

The initial continuation50 export comparison found three required actions and
28 observation labels in the training callback versus two actions and 26 labels
after export. Freshly loaded **unmerged singles also showed the latter result**.
The prompt had the same policy values but a different field order, so the initial
comparison cannot attribute the difference to BF16 merging. The export manifest,
weight hashes, confounded comparisons and partial release remain retained evidence;
the reproducible 14.79 GiB merged weight copies were removed. The selected full release
uses `--no-merge-adapter`; the original checkpoint300 evaluation remains labeled
as the merged variant actually measured. Old checkpoint manifests and their saved
source snapshots remain immutable; the new default is recorded in future source
fingerprints and model configurations.

The pinned [vLLM0.29 Gemma4 implementation](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/gemma4_mm.py#L912)
declares native LoRA support, maps the packed attention/MLP projections, and maps
the Hugging Face language-model parameter prefix. Its
[LoRA manager](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/lora/model_manager.py#L386)
deduplicates aliased module wrappers. This makes runtime adapter loading a supported
path to test; it does not establish matching decisions for this adapter. All 516
saved checkpoint50 adapter tensors are FP32, while the
[vLLM CLI LoRA dtype options](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/config/lora.py#L24)
are automatic, FP16 or BF16, with automatic following the base model's dtype.
Validate the actual enabled adapter on the same fixed inputs before selecting that
backend. Keep the unmerged NativeModel result as the behavioral reference.

## Policy serialization parity

Training originally read the authored policy JSON in the order `entities`,
`allowed_services`, `rules`, then the numeric limits. Reconstructing the same
policy through `Policy.as_dict()` moved `rules` ahead of `allowed_services`.
The dictionaries had equal values and equal token counts, but different input
token IDs. A processor comparison on all eight fixed validation examples found
every image, audio and mask tensor identical; only the text token IDs differed.

On a freshly loaded unmerged checkpoint50, restoring the authored field order
reproduced **all eight original training-callback outputs byte-for-byte**. No
weights, autocast settings, media or targets changed. `build_context` now orders
policy fields explicitly for training and serving. Regression tests cover policy
object reconstruction, preservation of values and stable extension fields. Exact
before/after context and tensor hashes are saved in
`artifacts/policy-order-diagnostic/`; `artifacts/continuation-policy-order/` retains
the GPU control. The system instruction text is unchanged. Older comparison runs
remain recorded with their actual prompt ordering and are not rewritten as matched
controls.

`evaluate_release.py --batch-size 1 --shard-count 2 --shard-index 0` (and index1 on
another GPU) splits independent snapshots by source order without changing their
inputs or targets. The chronological public recording belongs wholly to shard0.
Manifests retain every expected source ID, partition, adapter hash and runtime
configuration. Combining complete disjoint shards retains each event's original
latency and reports worker throughput separately; two GPUs do not establish one-GPU
capacity. This parallelizes independent single-request evaluations without batching
their model inputs.
