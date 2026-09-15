# Third-party data, models and attribution

The source code, configuration, documentation and synthetic fixtures in this
repository are MIT licensed (see `LICENSE`). The datasets, model weights and
media described below are **not** covered by that license and are **not**
redistributed in this repository. Only manifests, split files, hashes, labels,
provenance records and evaluation outputs derived from them are kept, so that
every measurement can be re-created from the pinned upstream sources.

## EPIC-KITCHENS-100 (CC BY-NC 4.0)

The temporal pilot (`data/temporal-pilot/`), the frozen continuous benchmark v2
(`data/continuous-v2/`), the workflow clips (`data/continuous-v2-workflow/`) and
every frame, contact sheet, sequence and event-evidence image written by the
production, recipe-workflow and recipe-tracking runs under `reports/` were
derived from EPIC-KITCHENS-100.

Credit: Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari,
Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett,
Will Price and Michael Wray, "Rescaling Egocentric Vision: Collection, Pipeline
and Challenges for EPIC-KITCHENS-100", IJCV 2022; and Damen et al., "Scaling
Egocentric Vision: The EPIC-KITCHENS Dataset", ECCV 2018.
https://epic-kitchens.github.io

License: Creative Commons Attribution-NonCommercial 4.0 International,
https://creativecommons.org/licenses/by-nc/4.0/ (full text in
`data/continuous-v2/sources/license.txt` and `data/temporal-pilot/sources/license.txt`).
The material was used for non-commercial research only. No endorsement by the
licensors is implied.

Sources and changes made:

- Untrimmed videos P07_01, P09_04, P13_06, P14_05, P22_105, P23_01, P25_12,
  P26_16, P27_03, P28_07 and P30_01 were fetched from the University of Bristol
  data server (per-video URLs in `data/continuous-v2/sources/attribution.md`),
  re-encoded to 854x480 H.264 with the audio track removed, tiled into 3.0 s
  windows with a 1.5 s stride, and sampled at 8 JPEG frames per window. The
  source SHA-256, re-encoded SHA-256 and exact ffmpeg command of every video are
  recorded in `data/continuous-v2/manifest.json`.
- Trimmed action clips were fetched from the Lightly AI Hugging Face mirror
  `lightly-ai/epic-kitchens-100-clips` (revision
  `e1739df818cbfb7e528fb1ef5005dbb51e96aacd`, dataset card license
  cc-by-nc-4.0; https://huggingface.co/datasets/lightly-ai/epic-kitchens-100-clips),
  renamed to content hashes and sampled at 8 frames per clip. Thanks to Lightly AI
  for hosting the mirror. `data/temporal-pilot/sources/README.md` (the dataset
  card), `data/temporal-pilot/sources/license.txt` and
  `data/temporal-pilot/sources/cut_clips.py` (the mirror's clip-extraction
  script) are unmodified copies from that repository, kept as provenance; they
  are Lightly AI's work under the mirror's terms and are not MIT code of this
  project.
- Official dense narrations from
  https://github.com/epic-kitchens/epic-kitchens-100-annotations at revision
  `ea8b40457a400c3fffa1c7f406ef3dc169cc2522` (SHA-256 per CSV in
  `data/continuous-v2/sources/annotation-files.json`) were converted into window
  labels. The retained `*.jsonl` rows, `data/continuous-v2/sources/annotations/*.json`,
  `data/continuous-v2-workflow/*.json` and the `stream-scores.json` /
  `seed.json` files under `reports/` quote narration text, narration IDs, video
  IDs and timestamps. They are CC BY-NC 4.0 derivatives and carry the same
  attribution and noncommercial condition; they are not relicensed under MIT.

Not in this repository: no EPIC-KITCHENS video, clip, frame or contact sheet is
committed. `.gitignore` excludes `data/continuous-v2/{media,videos}/`,
`data/temporal-pilot/{clips,media}/`, `data/temporal-pilot/recipe-contact-sheet.jpg`,
`data/continuous-v2-workflow/*.{mp4,jpg}` and every image, clip and audio file
under `reports/`. Re-create the media locally with
`scripts/temporal_data_prepare.py` and `scripts/continuous_benchmark_prepare.py`,
which download from the same pinned sources and verify the same hashes. Any use
of that media must respect the noncommercial condition and credit the
EPIC-KITCHENS authors; this project claims no commercial training rights over it.

## MS-COCO household caption subset

`data/coco-household/*.jsonl` keep 96 human captions from MS-COCO (Tsung-Yi Lin,
Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollar and C. Lawrence Zitnick, "Microsoft COCO: Common Objects in Context",
ECCV 2014; https://cocodataset.org). The captions and annotations are CC BY 4.0
and were obtained through the `mlgym/coco-captioning` Hugging Face metadata at
revision `96f5cf8784404ffd62ff541c5aeaea7b7a4d550d`.

The photographs themselves are Flickr images under individual Creative Commons
2.0 licenses, recorded per row as `provenance.image_license_id`
(1 = CC BY-NC-SA 2.0, 2 = CC BY-NC 2.0, 3 = CC BY-NC-ND 2.0, 4 = CC BY 2.0,
5 = CC BY-SA 2.0, 6 = CC BY-ND 2.0). Of the 96 images, 64 are NonCommercial and
31 are NoDerivatives, and the metadata does not carry photographer names, so the
resized copies (`data/coco-household/assets/`) are not redistributed. Re-create
them with `python -m home_observer.data coco-household data/coco-household --raw-dir work/coco-source`;
any reuse of an image must honour that image's own license and credit its
photographer via the recorded `original_coco_url` / `download_url`.

## EgoLife

One 17.6 s clip (`A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4`, SHA-256 in
`data/public-egolife/manifest.json`) from `lmms-lab/EgoLife` (revision
`143fb319be7aa5ae210c936bf4f0f3a86092afb0`; Jingkang Yang et al., "EgoLife:
Towards Egocentric Life Assistant", CVPR 2025; license MIT according to the
dataset card) was used as unlabeled real replay. The clip and the frames and
audio chunks cut from it (`data/public-egolife/{raw,assets}/`,
`reports/live-*/{capture,windows}/`) show identifiable people in a shared
household and are not committed. `data/public-egolife/replay.jsonl` and the
`live_report.json`, `source-provenance.json`, `validation.json` and
`predictions.jsonl` files under `reports/live-*/` keep only relative paths,
hashes, timestamps and model outputs. Re-create the sample with
`python -m home_observer.data public-sample`.

## Models

The base model is `google/gemma-4-E4B-it`, used under the Gemma Terms of Use
(https://ai.google.dev/gemma/terms). The LoRA adapters, optimizer and scheduler
states, and the tokenizer and chat-template copies produced during training are
Gemma derivatives and are not committed (`baselines/*/checkpoints/` and
`artifacts/` are gitignored). `baselines/2026-09-15-home-observer/checkpoint-manifest.json`
records the SHA-256 of every excluded checkpoint file, and the evaluation
reports record the adapter hashes they were measured with. Anyone who
redistributes those adapters must comply with the Gemma Terms of Use and the
Gemma Prohibited Use Policy.

Frontier planning in the production-shaped runs used Anthropic Claude models
through the Anthropic Messages API; no model weights or API material from that
provider are included.

## Home Assistant

The demonstrations used an isolated Home Assistant container
(`ghcr.io/home-assistant/home-assistant`, Apache-2.0), never a real home.

## Synthetic fixtures

`data/fixtures/`, `data/fixtures-expanded/`, `data/counterfactuals/`,
`data/mixed/` and `data/mixed-v2/` are drawings, tones and scenario text
rendered by this project's own code. They contain no third-party content or real
household footage and are MIT licensed together with the code.
