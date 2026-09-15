
---
license: cc-by-nc-4.0
tags:
  - video
pretty_name: "EPIC-KITCHENS-100 Extracted Clips"
size_categories:
  - 10K<n<100K
---

# EPIC-KITCHENS-100 Extracted Clips

## About

Dataset of 37455 video clips (24GB) extracted from videos in the EPIC-KITCHENS-100 dataset,
more precisely the extension part not contained in EPIC-KITCHENS-55. For details,
see https://www.lightly.ai/product-updates/epickitchens-100-in-lightlystudio.

The `clips` folder contains one video for every narration from action annotations stored
in `{participant_id}/{narration_id}.mp4`. The videos have been downscaled an compressed for easier
manipulation, the extraction script is `cut_clips.py`. The annotation files are preserved
in `epic-kitchens-100-annotations/` folder.

This dataset was built to showcase [LightlyStudio](https://github.com/lightly-ai/lightly-studio), an open-source
tool to visualise and curate datasets. See also [LightlyTrain](https://github.com/lightly-ai/lightly-train)
for a framework to pretrain, fine-tune and distill vision models.

## Citation

The credit is due to the original authors of the EPIC-KITCHENS-100 dataset: https://epic-kitchens.github.io

```
@ARTICLE{Damen2021PAMI,
   title={The EPIC-KITCHENS Dataset: Collection, Challenges and Baselines},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and Fidler, Sanja and 
           Furnari, Antonino and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
           and Perrett, Toby and Price, Will and Wray, Michael},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
   year={2021},
   volume={43},
   number={11},
   pages={4125-4141},
   doi={10.1109/TPAMI.2020.2991965}
} 
```
