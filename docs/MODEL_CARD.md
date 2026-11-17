# Model card: DeepShield v3

Following the structure of [Mitchell et al., "Model Cards for Model Reporting" (2019)](https://arxiv.org/abs/1810.03993).

## Model details

| | |
|---|---|
| Developer | Moamen Ahmed Fouad, DEPI Machine Learning track, 2026 |
| Type | Binary classifier on face crops (real vs. manipulated), with a separate head for video frames |
| Backbone A | [buildborderless/CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), a ViT-Small trained on the Community Forensics dataset ([Park and Owens, CVPR 2025](https://arxiv.org/abs/2411.04125)); 384-d embedding, one sigmoid logit |
| Backbone B | [prithivMLmods/deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1), a SigLIP image encoder fine-tuned for real vs. fake; 768-d embedding |
| Provenance check | C2PA Content Credentials (via c2pa-python), IPTC digital source type in XMP, AI tool names in EXIF, Stable Diffusion settings in PNG text. An AI label decides the verdict |
| Photo face head (v3) | L2-regularised logistic regression (C = 0.02) on the standardized 1,154-d vector `[emb_A, emb_B, logit_A, logit_B]` of the face crop, Platt-calibrated |
| Whole-image head (v3) | Same features computed on the whole picture (longest side 1,024 px), C = 0.02, Platt-calibrated |
| Stacker (v3) | `logit(p) = 0.455 * logit(p_face) + 0.639 * logit(p_whole) + 0.052`, fitted on held-out training images |
| Video head | Same features and form (C = 0.1), trained on video frames only, with a frame calibration and a whole-video calibration |
| Trained parameters | 1,155 per head (the backbones stay frozen) |
| Explanation | Grad-CAM of the fused score through both backbones |
| License | MIT for this code. The v3 heads are trained partly on OpenFake (CC BY-NC 4.0), so they are for non-commercial use. Each backbone keeps its own license on its Hugging Face page |

## Version 3: real-world images

Version 2 (below) was trained and tested on research datasets. On images people actually
share it called 7.9% of real Reddit photos fake and missed about a third of current AI
images, so version 3 adds newer data and a second view.

**Training data added** (`ml/scripts/collect_wild.py`). Only images whose caption mentions
a person and in which YuNet finds a face of at least 64 px:

| Source | Images (train + validation) |
|---|---|
| [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) train split, AI images from about 70 generators (SD 1.5 to SD 3.5, SDXL fine-tunes, FLUX.1, Flux.2, DALL-E 3, GPT-image-1, Imagen 3 and 4, Ideogram 3, Midjourney 6, Seedream 4, HiDream, Qwen-Image, Grok, video-model frames) | 2,629 |
| OpenFake train split, real LAION and Pexels photos | 2,034 |
| OpenFake Reddit split, half by file hash (real and AI images posted on Reddit) | 642 |
| COCO photos of people (shards 3 to 9; version 2 used 0 to 2) | 400 |

20% of these (by hash) are held out to choose C, calibrate and fit the stacker. The face
head is trained on these faces together with the version 2 training crops.

**Test data** (never used for any choice): the other half of OpenFake's Reddit split (228
real, 632 AI), OpenFake's test split (34 real, 601 AI from GPT-image-1.5 and 2, Nano Banana
Pro, Midjourney 7, Flux.2, Seedream 5, Z-Image, Sora 2, Veo 3 and others; 20 of these
generators never appear in training) and 500 COCO photos from shards 10 to 13.

| 1,995 test images | AUC | Accuracy | Real photos called fake |
|---|---|---|---|
| Version 2 photo head | 0.944 | 79.5% | 7.9% (Reddit) |
| Version 3 face head | 0.973 | 87.0% | |
| Version 3 whole-image head | 0.979 | 86.6% | |
| **Version 3 combined** | **0.987** | **89.5%** | **1.3%** (Reddit) |

With the app's inconclusive band (40 to 60%) and the unfamiliar guard, real Reddit photos
are answered real 90.4%, inconclusive 8.3%, fake 1.3%; AI images on Reddit fake 87.0%,
inconclusive 6.2%, real 6.8%. Weakest generators: Flux.2 klein (26% caught), Recraft v3
(45%), GPT-image-2 (64%), Z-Image Turbo (73%). Full numbers: `reports/wild_metrics.json`.

**Cost of the change.** The v3 face head is slightly weaker on the version 2 research
test sets: all held-out faces AUC 0.965 to 0.959, never-trained sources 0.969 to 0.953.
HeyGen avatar frames drop from 39% to 14% caught and Celeb-DF frames from 68% to 58%.
Videos use the separate video head, which did not change.

**Provenance.** Content Credentials are checked before any pixel model. They are reliable
when present but removed by screenshots, most messaging apps and many social networks,
so a missing label is reported as "proves nothing". OpenFake stores images without
metadata, so the numbers above measure the pixel models only.

The sections below describe version 2, which version 3 extends.

## Intended use

* Educational demo and first-pass screening of photos and short videos of human faces.
* Showing *where* in a face the evidence lies (Grad-CAM), *when* in a video (timeline)
  and *what it resembles* (closest known cases).

**Not intended for:** legal, journalistic or disciplinary decisions on its own; images
without a human face; audio deepfakes (not covered).

## Training data

12,863 face crops from 20 public sources, listed with citations in
[DATASETS.md](DATASETS.md).

| Split | Crops | Role |
|---|---|---|
| Train | 8,075 from 13 sources (plus 16,150 augmented copies) | Fits the heads and makes every modelling choice |
| Test | 2,385 | Held-out people, videos and files from the same 13 sources; evaluated once |
| Unseen | 2,403 from 7 sources | Never trained on: Midjourney, StyleGAN3, Collaborative Diffusion, HeyGen avatars, LFW, YouTube, whichfaceisreal |

Each training crop has two augmented copies: one with harsh degradations (JPEG 30 to
75, down-scaling, blur, noise, colour casts) and one that looks "shared online"
(scaled to 160 to 512 px, JPEG 70 to 95).

## Training procedure

1. **Backbone choice.** Nine public models (five deepfake detectors, two AI-image
   detectors, CLIP and DINOv2) were screened on a 2,550-crop training subset with
   leave-one-source-out validation. The v1 pair scored 0.818; the best alternative
   pair (CommunityForensics + CLIP) scored 0.827 and adding a third model at most 0.834.
   Those gains are within noise and would add download size, so the pair was kept.
2. **Photo head.** Model selection uses the training split only, with
   leave-one-domain-out validation: the 13 sources form 8 domains that each hold real
   and fake faces, and every domain is scored by a head that never saw it. C was tuned
   on the mean per-domain AUC over clean and degraded crops (0.829 at C = 0.0002).
   The final head's logit is Platt-calibrated on those held-out predictions.
3. **Video head** (`ml/scripts/train_video_head.py`). The photo head learned from many
   clean photos and started to read the look of a video frame (compression, motion
   blur) as a sign of a fake: on whole test videos its AUC fell to 0.66. Videos
   therefore use a second head trained only on the 4,320 frames of 360 training videos
   (Celeb-DF v2 and Google DFD). C was chosen by 5-fold cross-validation grouped by
   person (0.737 video-level AUC), then the frames and the aggregated video scores
   were each Platt-calibrated on out-of-fold predictions. The test videos were not
   used for any choice.

Total compute: a few hours on a 2-core CPU, almost all of it feature extraction. No GPU.

## Results on held-out photos (face crops)

`reports/metrics.json`. AUC ranks fakes above real faces (1.0 perfect, 0.5 a coin
flip); accuracy uses the 0.5 threshold.

| Test set | n | DeepShield v2 | v1 head | Average of both backbones | CommunityForensics alone |
|---|---|---|---|---|---|
| Original four test sets | 1,320 | **0.931** (84.2%) | 0.920 (82.7%) | 0.852 | 0.816 |
| Newer sources (FLUX, SDXL, face swaps, everyday photos) | 1,065 | **0.992** (95.1%) | 0.858 (76.0%) | 0.856 | 0.885 |
| All held-out test faces | 2,385 | **0.965** (89.1%) | 0.893 (79.7%) | 0.853 | 0.857 |
| Unseen sources, never trained on | 2,403 | 0.969 (91.1%) | 0.919 (84.4%) | 0.967 (92.6%) | **0.994** (94.0%) |

Brackets show accuracy. On the unseen sources backbone A alone ranks fakes better than
the fused head (0.994 against 0.969). It is much weaker on the test sets the head was
built for (0.857 against 0.965), so the head stays the default. Accuracy per source on
the held-out sets:

| Source | Accuracy | Source | Accuracy |
|---|---|---|---|
| StyleGAN (140k) | 97.4% | Midjourney (unseen) | 100% |
| SDXL faces | 100% | StyleGAN3 (unseen) | 98.7% |
| FLUX faces | 99.0% | Collaborative Diffusion (unseen) | 100% |
| SFHQ-T2I portraits | 99.4% | whichfaceisreal (unseen) | 97.2% |
| DeepFakeFace | 87.7% | YouTube real (unseen) | 93.3% |
| CelebA face swaps | 88.4% | LFW real (unseen) | 82.0% |
| Google DFD frames | 72.5% | CelebA real | 70.4% |
| Celeb-DF v2 frames | 67.5% | HeyGen avatars (unseen) | **39.2%** |

![Accuracy by source](../reports/accuracy_by_source.png)

On the four core test sets v2 gains a lot on Celeb-DF frames and loses a little on
DeepFakeFace and Google DFD frames:

![AUC on the four core test sets](../reports/per_dataset_auc.png)

### Robustness to everyday edits

`ml/scripts/eval_robustness.py` takes 600 held-out faces, applies one change at a time
and checks how often the verdict flips (`reports/robustness.json`):

| Change | AUC v2 | AUC v1 | Verdicts flipped (v2) |
|---|---|---|---|
| None | 0.943 | 0.849 | 0% |
| Shared online (resized, JPEG 70 to 95) | 0.932 | 0.809 | 5.8% |
| Strong JPEG | 0.900 | 0.758 | 14.0% |
| Small face | 0.921 | 0.790 | 12.0% |
| Blur | 0.931 | 0.841 | 15.7% |
| Messaging app (small + strong JPEG) | 0.898 | 0.696 | 15.5% |
| Coloured stage lighting | 0.811 | 0.691 | 26.5% |

## Results on whole held-out videos

90 balanced videos never used in training: 60 from the Celeb-DF v2 official test list
and 30 Google DFD videos of held-out actors (21 to 28). Each goes through the app's
pipeline (24 frames, face detection, video head, mean blended with the most suspicious
quarter, video calibration). `reports/video_metrics.json`, per video in
`reports/video_eval.csv`.

| Videos | n | AUC v2 (video head) | AUC v1 | AUC with the v2 photo head | Accuracy v2 | Inconclusive |
|---|---|---|---|---|---|---|
| Celeb-DF v2 | 60 | **0.817** | 0.713 | 0.549 | 71.7% | 41.7% |
| Google DFD | 30 | 0.907 | **0.951** | 0.867 | 63.3% | 20.0% |
| All | 90 | **0.778** | 0.761 | 0.659 | 68.9% | 34.4% |

When the app does give a verdict (not inconclusive) it is right 79.7% of the time.
Every DFD fake was caught (100% recall), but real DFD videos are often called fake or
inconclusive. With 90 videos the error bars are wide: a bootstrap 95% interval for the
overall AUC is about 0.68 to 0.87. Mean time was 11.5 seconds per video on a shared
2-core CPU. The scores were computed by replaying per-frame features cached from the
production pipeline; the end-to-end run gave identical scores on the first 8 videos.

## Closest known cases and the unfamiliar-image guard

Next to the verdict, the app lists the 5 training faces most similar to the upload,
from an index of the 8,075 training crops (only 64-number fingerprints and labels are
stored, never images). A plain 5-nearest-neighbour vote reaches 0.902 AUC on the test
faces and 0.968 on the unseen sources (`reports/similar_index.json`), so the neighbours
carry real signal, but they stay supporting evidence.

If even the closest known face is a weak match (similarity below 0.62, the 2nd
percentile measured between training faces of different people), the photo is
reported as **inconclusive: unfamiliar image** instead of getting a confident score:

| Held-out faces | Flagged | Accuracy when flagged | Accuracy otherwise |
|---|---|---|---|
| Test | 2.1% | 74.0% | 89.4% |
| Unseen sources | 5.3% | 47.2% | 93.5% |

## Limitations and known failure modes

* **Talking-head avatars.** HeyGen avatar frames are caught only 39% of the time. They
  are a different kind of fake (a whole synthesized speaker) that no training source
  covers.
* **Celeb-DF v2 and other high-quality face swaps** remain the hardest case: 67.5% of
  single frames and 71.7% of whole videos are classified correctly. Celeb-DF
  was built to remove the blending artifacts detectors rely on
  ([Li et al., 2020](https://arxiv.org/abs/1909.12962)).
* **New generators.** Each source the model was trained on is now handled well, but
  generators released later may leave different traces; cross-generator
  generalization is an open problem ([Pei et al., 2024](https://arxiv.org/abs/2407.06174)).
* **Heavy edits.** Strong compression, tiny faces and coloured stage lighting flip
  between 12% and 27% of verdicts (table above).
* **Faces only.** Without a face the app still shows a score but always answers
  "inconclusive". Faces smaller than 40 px are skipped.
* **Demographics were not audited.** Accuracy across skin tones, ages and genders is
  unknown.

## Ethical considerations

A false "fake" can damage a real person's reputation, and a false "real" can lend
credibility to a forgery. The interface shows uncertainty, explains its evidence and
prints a disclaimer on every PDF report. Uploads are processed in memory and
discarded; results live in an in-memory cache of the last 64 analyses and are lost on
restart.
