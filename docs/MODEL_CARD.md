# Model card: DeepShield fusion detector v1.1

Following the structure of [Mitchell et al., "Model Cards for Model Reporting" (2019)](https://arxiv.org/abs/1810.03993).

## Model details

| | |
|---|---|
| Developer | Moamen Ahmed Fouad, DEPI Machine Learning track, 2026 |
| Type | Binary image classifier on face crops (real vs. manipulated), plus video aggregation |
| Backbone A | [buildborderless/CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT), a ViT-Small trained on the Community Forensics dataset ([Park and Owens, CVPR 2025](https://arxiv.org/abs/2411.04125)); 384-d embedding, one sigmoid logit |
| Backbone B | [prithivMLmods/deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1), a SigLIP image encoder fine-tuned for real vs. fake; 768-d embedding |
| Head | L2-regularised logistic regression (C = 0.02) on the standardized 1,154-d vector `[emb_A, emb_B, logit_A, logit_B]` |
| Trained parameters | 1,155 (the backbones stay frozen) |
| Explanation | Grad-CAM on backbone A |
| License | MIT for this code and head; each backbone keeps its own license on its Hugging Face page |

Both backbones were **chosen by the data**, not by hand. Five public detectors were
benchmarked and every pair was scored on a validation split (see "Training" below).

## Intended use

* Educational demo and first-pass screening of photos and short videos of human faces.
* Showing *where* in a face the evidence lies (Grad-CAM) and *when* in a video (timeline).

**Not intended for:** legal, journalistic or disciplinary decisions on its own;
images without a human face; heavily compressed social-media re-uploads (not tested);
audio deepfakes (not covered).

## Training data

4,760 face crops from four public datasets, balanced real/fake per source.
Full details and citations are in [DATASETS.md](DATASETS.md).

| Split | Crops | How it was made |
|---|---|---|
| Train | 2,826 | Used to fit the head |
| Validation | 614 | 20% of training *groups* (identities, actors, videos); used only to pick backbones and C |
| Test | 1,320 | Official or identity/actor-disjoint test splits; evaluated **once** |

Training weights each (source, label) cell equally so the large StyleGAN set does
not dominate.

## Training procedure

1. Run all five candidate detectors on every crop; cache `P(fake)` and the pooled
   embedding that feeds each model's classification layer.
2. On validation, score every single model and every pair with a quick head
   (C = 0.05). Metric: AUC averaged over the four sources.
3. The best pair was CommunityForensics + SigLIP v1 (validation AUC 0.939; best
   single model alone 0.835).
4. Tune C over {0.005, 0.02, 0.05, 0.2, 1.0} on validation (best: 0.02), refit on
   all training crops, evaluate once on test.

Total compute: about 70 minutes on a 2-core CPU, most of it feature extraction. No GPU.
An optional full fine-tune for a GPU is in `ml/notebooks/03_full_finetune_gpu.ipynb`.

## Results on the held-out test split (face crops)

| Model | 140k Faces | DeepFakeFace | Celeb-DF v2 | Google DFD | Overall AUC | Accuracy | F1 |
|---|---|---|---|---|---|---|---|
| **DeepShield fusion head** | **0.999** | **0.968** | 0.586 | **0.848** | **0.920** | **82.7%** | **0.816** |
| Average of the two backbones | 0.992 | 0.781 | 0.655 | 0.574 | 0.852 | 74.7% | 0.678 |
| CommunityForensics ViT alone | 1.000 | 0.929 | 0.488 | 0.700 | 0.816 | 72.5% | 0.644 |
| SigLIP v1 alone | 0.953 | 0.525 | 0.684 | 0.498 | 0.702 | 63.3% | 0.624 |
| Deep-Fake-Detector-v2 alone | 0.530 | 0.488 | 0.547 | 0.495 | 0.515 | 49.6% | 0.553 |

Per-source columns are AUC. At the default threshold of 0.5 the head reaches 87.5%
precision and 76.4% recall; the equal error rate is 17.4%. Every number comes from
`reports/metrics.json`, and `ml/notebooks/02_training_and_evaluation.ipynb` reproduces them.

![Per-dataset AUC](../reports/per_dataset_auc.png)

## Results on whole held-out videos

`ml/scripts/eval_videos.py` runs the production `Analyzer` (24 sampled frames, face
detection, fusion head, mean plus top-quartile aggregation) on 60 Celeb-DF v2 videos
from the official test list and 30 Google DFD videos of held-out actors (21 to 28),
balanced real/fake.

| Videos | AUC | Accuracy | Precision | Recall | n |
|---|---|---|---|---|---|
| Google DFD | 0.951 | 76.7% | 68.2% | 100% | 30 |
| Celeb-DF v2 | 0.713 | 65.0% | 68.0% | 56.7% | 60 |
| All | 0.761 | 68.9% | 68.1% | 71.1% | 90 |

Accuracy uses the 0.5 threshold. 20% of videos fall in the inconclusive band (40% to
60%). Mean processing time was 18 seconds per video on a shared 2-core CPU. Source:
`reports/video_metrics.json` and the per-video `reports/video_eval.csv`. With 90
videos these numbers carry wide error bars (roughly plus or minus 0.1 AUC).

## Closest known cases (retrieval, added in v1.1)

Next to the verdict, the app lists the 5 training faces most similar to the upload.
The index covers the 3,440 training crops; the projection is described in
[ARCHITECTURE.md](ARCHITECTURE.md). To check that the neighbours carry real signal, a
plain 5-nearest-neighbour vote was scored on the same held-out test split
(`reports/similar_index.json`):

| Test faces | 5-NN vote AUC | Fusion head AUC |
|---|---|---|
| 140k Faces (StyleGAN) | 0.992 | 0.999 |
| DeepFakeFace | 0.931 | 0.968 |
| Google DFD | 0.579 | 0.848 |
| Celeb-DF v2 | 0.460 | 0.586 |
| All | 0.810 | 0.920 |

Retrieval is strong for generated and diffusion faces and weak for video face swaps,
where the nearest faces tend to be the same actors rather than the same kind of
manipulation. That is why it is shown as supporting evidence and never decides the
verdict on its own.

## Unfamiliar-image guard (added in v1.2)

Retrieval also tells the app when a face is unlike anything it was trained on. If even
the closest known face is a weak match, the fusion score is not trusted: the face and
the result are reported as **inconclusive** with an "Unfamiliar image" note, and the
raw score is still shown.

The cut-off comes from training data only. For each training face we take its most
similar training face from a different person or video, and the threshold is the 2nd
percentile of those similarities (0.576). On the held-out test split
(`reports/similar_index.json`):

| Test faces | Share | Fusion head accuracy |
|---|---|---|
| Flagged as unfamiliar | 27 of 1,320 (2.0%) | 66.7% |
| All others | 1,293 (98.0%) | 83.1% |

So the guard only fires on a small set of faces where the model really is less
reliable. AUC is unchanged because the probabilities are not altered.

It is a partial fix, not a cure. In a manual check with two out-of-domain photos, it
caught a portrait from a recent AI generator that the model had scored as 11% fake
(closest-case similarity 0.50, flagged). It did not catch a real concert photo with
heavy red stage lighting that the model scored as 97% fake (similarity 0.65, above the
threshold).

## Limitations and known failure modes

* **Celeb-DF v2 is hard.** AUC 0.59 on single frames. Celeb-DF was built to remove
  the visible blending artifacts that earlier detectors rely on
  ([Li et al., 2020](https://arxiv.org/abs/1909.12962)), and none of the five public
  backbones handles it well (best: 0.68). It is the clearest target for the GPU
  fine-tune.
* **New generators.** The detectors learned artifacts of today's GANs, diffusion
  models and face swaps. Generators released later may not leave the same traces;
  cross-generator generalization is an open research problem
  ([Pei et al., 2024](https://arxiv.org/abs/2407.06174)).
* **Compression and resizing** remove high-frequency evidence. The test data is mostly
  high quality; screenshots and messaging-app re-uploads will score less reliably.
* **Faces only.** If no face is found, the whole image is scored and the result says so.
  Small faces (under 40 px) are skipped.
* **Demographics were not audited.** The datasets skew toward celebrities and Western
  faces, so accuracy across skin tones, ages and genders is unknown.
* **Scores are not calibrated probabilities across domains.** That is why the UI
  shows an "inconclusive" band between 40% and 60%.
* **Unusual photography.** Strong coloured lighting, filters and heavy colour grading
  are rare in the training data and can push a real face toward "fake". The
  unfamiliar-image guard catches some of these cases, not all.

## Ethical considerations

A false "fake" can damage a real person's reputation, and a false "real" can lend
credibility to a forgery. The interface therefore shows uncertainty, explains the
evidence, and prints a disclaimer on every PDF report. Uploaded files are processed
in memory and discarded; results live in an in-memory cache of the last 64 analyses
and are lost on restart.
