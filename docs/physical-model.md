# Physical perception model and registry

This is a separate sensing path. The original home-control model, schema, policy,
journal tables, training runs and baseline evaluations remain available.

## Input and output

`PhysicalInferenceRequest` contains a `PhysicalWindow`, prior physical `tracks`,
`recent_events`, and optional observation `focus`. A window holds chronological
camera clips with explicit frame evidence IDs/timestamps and optional audio.
`physical_request_from_row` selects the window only: labels, narration, annotation
coverage, teacher responses and arbitrary row context never enter model input.

`PhysicalDecision` contains a summary, detected objects, physical observations and
temporal events. It has no action or device-service field. Labels and interaction
`kind` values are free text; new objects do not need a Home Assistant entity.

Each detection has a normalized positive-area box, exact frame/camera/time,
visibility, confidence and explicit uncertainty. Temporal events cite distinct,
chronological pre/post evidence and carry `kind` and optional `object_label`.
An interaction can remain untracked when identity is unknown. Source interval
endpoints may lie between sampled frames; a sampled image is not a continuous
measurement of an event endpoint.

Validation checks schema, finite/ranged values, source-time bounds, evidence IDs,
box/frame consistency and local/known physical subject references. It does not
establish that a visual claim is correct. Audio/image decoding and local-path
containment are checked at the model/transport boundary.

## Durable physical objects

`PhysicalJournal(path)` uses dedicated `physical_*` SQLite tables and can share
the baseline journal's database file. `ingest(window, decision)` atomically returns
a local detection ID → persistent `phys_*` track ID mapping. Repeating identical
window content returns the same mapping; conflicting reuse of an ID fails.

`tracks(as_of)` returns the latest detection available at that source time,
including age/staleness. Missing detections do not imply disappearance or erase a
track. `recent_events(as_of)` returns flattened events with durable `pevt_*` IDs,
original local IDs, window IDs and subjects resolved to physical track IDs.
No registry method executes a Home Assistant action or treats a physical track
as a registered device entity. Association and continuous image tracking are
separate components.

## Fresh LoRA training

The physical trainer starts from the pinned Gemma4 base. It freezes encoders and
base weights, trains only language-projection LoRA weights, and requires a fresh
output directory. It cannot load a home-control adapter as its starting point.

```bash
python -m home_observer.physical_train \
  --train data/temporal-pilot/train.jsonl \
  --validation data/temporal-pilot/validation.jsonl \
  --dataset-root data/temporal-pilot \
  --output artifacts/physical-training-v1/adapter \
  --quantization none --max-steps 100 --rank 16 \
  --learning-rate 1e-4 --gradient-accumulation-steps 2 \
  --checkpoint-every 25 --generation-limit 4
```

Explicit nested supervision masks select known fields. For example, human EPIC
annotations supervise an event's verb (`kind`), noun (`object_label`), description
and source interval. They do not provide boxes, track IDs, confidence, uncertainty
or exhaustive negative examples. Missing fields remain absent from the training
target; partial-container closing tokens and end-of-turn tokens are masked, along
with all prompts, media tokens and padding. An omitted object list is never
converted into a negative object-detection label.

Checkpoints contain optimizer, scheduler, RNG, Trainer state, adapter and processor
files, with a final integrity manifest. Exact resume uses the same original flags
and `--resume-from-checkpoint <checkpoint>`; hashes of data, media, source, prompt,
configuration and runtime versions must match. Low teacher-forced loss does not
demonstrate complete physical output, detection, identity, silence or temporal
localization accuracy. Generation outputs and annotation-scoped metrics are saved
separately, including invalid outputs.

Local verification includes actual pinned-processor encoding of eight-frame EPIC
rows, partial-label masks, a real tiny Gemma4/PEFT/Trainer run with full checkpoint
saving, causal registry queries, and transport tests. GPU results are recorded in
the corresponding run artifacts; these local checks do not assert trained quality.

## HTTP inference

Set `PHYSICAL_API_TOKEN` in the server environment, then run:

```bash
python -m home_observer.physical_serve \
  --adapter artifacts/physical-training-v1/adapter \
  --media-root artifacts/physical-upload-media \
  --host 127.0.0.1 --port 8790 --quantization none
```

Both `/health` and `/observe` require `Authorization: Bearer <token>`. The request
body is `{request: {window, tracks, recent_events, focus}, media: {...}}`, with one
base64 upload and MIME type per frame/audio evidence ID. Client file paths are
replaced by server-generated temporary filenames. Uploaded files, decoded image
pixels/audio duration and aggregate body size are bounded, and files are removed
after inference. The API accepts inline uploads, not server filesystem paths.

`PhysicalRemoteModel(url, api_token, dataset_root=...)` performs this upload and
validates returned physical evidence against the original request. `observe`
retains `{decision, raw_output, metrics, error}`, including failed generations.
Validation failures return HTTP422 with the raw model output; backend failures
return HTTP503. Authentication failures do not invoke the model.

Native Transformers decoding currently uses strict post-validation. The optional
`constrained_physical_schema` helper prepares a JSON schema for a tested structured
decoder: current evidence IDs and known track IDs are constrained, while new
object labels and interaction kinds remain free text. Schema constraints cannot
prove physical truth or cross-field temporal/identity consistency. An actual
grammar-serving backend must still be tested and its output validated; the helper
alone is not a claim that constrained native decoding is enabled.

## Guided vLLM configuration

`PhysicalVLLMModel` sends the same frozen physical system prompt and canonical
context to an existing multimodal vLLM engine. Its separate
`physical_events_first_bounded_v3` decoding grammar emits `events`, `objects`,
`observations`, then `summary`, with at most two items in each list. Labels and
interaction kinds remain free text. Two local detection ID slots can be referenced,
along with supplied known physical tracks; unregistered persistent IDs cannot be
invented. Post-validation still checks that referenced slots were actually emitted.
Each object chooses one sampled-frame branch that binds its camera, timestamp
and evidence ID; its label and box are still model predictions.
Evidence choices come from the current
window. Empty lists are allowed. This reporting budget does not establish that
unreported objects or events are absent. Required confidence and uncertainty
fields are generated judgments, not calibrated measurements.

This configuration is evaluated separately from native unconstrained decoding.
Both base and adapted comparisons must use the same engine, prompt, grammar,
frame/token budgets and request sequence. The client records the served alias,
engine-reported alias, prompt/context/schema hashes, frame IDs, tokens and latency;
it retains raw output on validation failure. GPU memory and the configured model
revision cannot be independently verified through the chat API alone.

For the existing authenticated engine, the upload HTTP frontend can use:

```bash
python -m home_observer.physical_serve \
  --backend vllm --engine-url http://127.0.0.1:8001/v1 \
  --served-model physical-trained \
  --media-root artifacts/physical-upload-media \
  --max-frames 8 --max-input-tokens 8192 --max-new-tokens 1024
```

Set `HOME_OBSERVER_ENGINE_TOKEN` for the private engine and `PHYSICAL_API_TOKEN`
for the upload API. The client does not start or restart an engine. vLLM audio
transport accepts upstream-prepared 16 kHz mono WAV; frames remain JPEG, PNG or
WebP in chronological order. Geometry, evidence causality and track references
are validated after generation, because a grammar alone cannot prove them.

## Reproduce the trained engine with a startup-loaded adapter

The experiment used vLLM 0.29.0 and a private, authenticated runtime adapter load
so base and trained requests shared one engine. For reproduction, load the same
adapter at engine startup; dynamic adapter administration is unnecessary. Install
`requirements-vllm.txt` in its own environment and install this project there.
Run from the project directory containing `artifacts/physical-training-v1/adapter`.

Set `VLLM_API_KEY` to a private engine token in the environment, then launch:

```bash
: "${VLLM_API_KEY:?Set a private engine token before launch}"
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export MAX_JOBS=2
export TORCHINDUCTOR_COMPILE_THREADS=2
unset VLLM_ALLOW_RUNTIME_LORA_UPDATING
vllm serve google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --served-model-name physical-base \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image":8,"audio":1,"video":0}' \
  --mm-processor-kwargs '{"max_soft_tokens":140}' \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-lora --max-lora-rank 16 --max-loras 1 \
  --lora-modules physical-trained=artifacts/physical-training-v1/adapter \
  --host 127.0.0.1 --port 8001
```

The base alias remains `physical-base`; the adapter alias is `physical-trained`.
Check the authenticated `/v1/models` response for both aliases before starting the
frontend. Set `HOME_OBSERVER_ENGINE_TOKEN` to the engine token and set a separate
`PHYSICAL_API_TOKEN` for the upload frontend. The frontend command above selects
`physical-trained` explicitly. Preserve the adapter SHA manifest and engine launch
arguments when comparing a reproduced run. Startup loading is a reproduction
path; the measured runs used the recorded dynamic-load request and response.
