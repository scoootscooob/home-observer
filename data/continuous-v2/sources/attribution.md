# Attribution

This benchmark redistributes derived media from **EPIC-KITCHENS-100** by Dima Damen and the EPIC-KITCHENS authors (University of Bristol and collaborators), licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (https://creativecommons.org/licenses/by-nc/4.0/). The license text is in `license.txt`. Credit: Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price and Michael Wray, *Rescaling Egocentric Vision: Collection, Pipeline and Challenges for EPIC-KITCHENS-100*, IJCV 2022; and Damen et al., *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset*, ECCV 2018.

The material is used here for non-commercial research. No endorsement by the licensor is implied.

## Changes made

- Videos were re-encoded to 854x480 H.264 at the source frame rate with the audio track removed; the originals are not redistributed.
- Each video was tiled into 3.0 s windows with a 1.5 s stride and 8 JPEG frames were sampled per window.
- Official dense narrations (verb, noun, start/stop time, narration text) were converted into window labels. Narration text and labels are kept outside the model input.
- Annotations come from https://github.com/epic-kitchens/epic-kitchens-100-annotations at revision ea8b40457a400c3fffa1c7f406ef3dc169cc2522.

## Sources

- `P28_07` (test): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P28/P28_07.MP4
- `P30_01` (test): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P30/P30_01.MP4
- `P22_105` (test): https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P22/videos/P22_105.MP4
- `P27_03` (test): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P27/P27_03.MP4
- `P26_16` (validation): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P26/P26_16.MP4
- `P09_04` (train): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/P09/P09_04.MP4
- `P25_12` (train): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P25/P25_12.MP4
- `P14_05` (train): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P14/P14_05.MP4
- `P07_01` (train): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P07/P07_01.MP4
- `P23_01` (train): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P23/P23_01.MP4
- `P13_06` (train): https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P13/P13_06.MP4
