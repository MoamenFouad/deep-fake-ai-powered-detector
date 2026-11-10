# Datasets

DeepShield is trained and evaluated on four public datasets that cover the three
main families of face forgery: fully generated faces, diffusion edits, and
face-swap videos. Mixing them matters: a detector trained on one family often
fails on the others ([Pei et al., 2024](https://arxiv.org/abs/2407.06174)).

| Key | Dataset | Real | Fake | Mirror used |
|---|---|---|---|---|
| `ffhq_stylegan` | 140k Real and Fake Faces | Flickr-Faces-HQ photos | StyleGAN generated faces | [TheKernel01/140k-Real-and-Fake-Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) |
| `dff` | DeepFakeFace ([Song et al., 2023](https://arxiv.org/abs/2309.02218)) | IMDB-WIKI celebrity photos | Stable Diffusion text-to-image, Stable Diffusion inpainting, InsightFace swaps | [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) |
| `celebdf` | Celeb-DF v2 ([Li et al., 2020](https://arxiv.org/abs/1909.12962)) | YouTube interviews | Improved face-swap deepfakes | [gonnerthetooner/deepfake_video_data](https://huggingface.co/datasets/gonnerthetooner/deepfake_video_data) |
| `dfd` | Google/Jigsaw DeepFakeDetection, part of FaceForensics++ ([Rössler et al., 2019](https://arxiv.org/abs/1901.08971)) | Paid actors | Face-swap deepfakes of the same actors | same mirror, `google_dfd/` |

## Sampling and splits

`ml/scripts/prepare_data.py` streams only what it needs (parquet row groups and
individual zip members via HTTP range requests), so nothing close to the full
70 GB has to be downloaded.

| Source | Train (per class) | Test (per class) | Unit |
|---|---|---|---|
| ffhq_stylegan | 600 images | 250 images | image |
| dff | 400 images | 150 images | image |
| celebdf | 120 videos x 4 frames | 40 videos x 4 frames | video |
| dfd | 60 videos x 4 frames | 25 videos x 4 frames | video |

Leakage controls, which are the most common mistake in student deepfake projects:

* **Celeb-DF:** the official `List_of_testing_videos.txt` defines the test set, and
  every identity that appears in it is removed from training.
* **DFD:** split by actor. Actors 01-20 train, 21-28 test.
* **DeepFakeFace:** the real and fake archives contain the same people, so the split
  is done by identity hash. A person is never in both train and test.
* **140k Faces:** the official train and test parquet files are used.
* A further 20% of training groups is held out as a validation split for model
  selection, so the test set is only touched once.

Each sample goes through the production face detector and cropper. Video frames
without a detectable face are skipped (the 12 most evenly spaced frames from the
first 12 seconds are tried; the first 4 with a face are kept).

## Licenses and responsible use

These datasets are for research. Celeb-DF and FaceForensics++ require agreeing to
their terms of use; if you redistribute anything, request access from the original
authors first. The 140k set derives from FFHQ (Creative Commons BY-NC-SA 4.0).
DeepShield ships no dataset images except two sample faces, both from the
140k Real and Fake Faces test split (one FFHQ photo, one StyleGAN face).

## Rebuilding

```bash
pip install -r ml/requirements.txt
python ml/scripts/prepare_data.py --out data/crops          # ~5,000 crops, ~25 min
python ml/scripts/train_fusion.py --data data/crops          # features + head, ~60 min on CPU
python ml/scripts/eval_videos.py --per-class 30              # end-to-end video test, ~15 min
python ml/scripts/build_similar_index.py                     # closest-known-cases index, seconds
python ml/scripts/make_figures.py
```

Use `--scale 0.1` on `prepare_data.py` for a 5-minute smoke run.
