# Datasets

DeepShield v2 is built from **12,863 face crops from 20 public sources**. A detector
trained on one family of fakes usually fails on the others
([Pei et al., 2024](https://arxiv.org/abs/2407.06174)), so the data deliberately mixes
old and new generators, face swaps, studio portraits and ordinary photos.

| Split | Crops | Role |
|---|---|---|
| Train | 8,075 (plus 16,150 augmented copies) | Fits the fusion head |
| Test | 2,385 | Held-out people, videos and files from the 13 training sources |
| Unseen | 2,403 | 7 sources that are **never** trained on, only tested |

## Sources used for training and testing

Fake faces:

| Key | What it is | Train | Test | From |
|---|---|---|---|---|
| `ffhq_stylegan` | StyleGAN generated faces | 600 | 250 | [140k Real and Fake Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces) |
| `dff` | Stable Diffusion text-to-image, Stable Diffusion inpainting, InsightFace swaps | 400 | 150 | DeepFakeFace ([Song et al., 2023](https://arxiv.org/abs/2309.02218)), [OpenRL/DeepFakeFace](https://huggingface.co/datasets/OpenRL/DeepFakeFace) |
| `celebdf` | Celebrity face-swap videos | 480 | 160 | Celeb-DF v2 ([Li et al., 2020](https://arxiv.org/abs/1909.12962)) |
| `dfd` | Actor face-swap videos | 240 | 100 | Google/Jigsaw DeepFakeDetection, part of FaceForensics++ ([Rössler et al., 2019](https://arxiv.org/abs/1901.08971)) |
| `flux_faces` | FLUX.1-dev faces generated from FFHQ and CelebA-HQ captions | 405 | 95 | [bitmind/ffhq-256___FLUX.1-dev_training_faces](https://huggingface.co/datasets/bitmind/ffhq-256___FLUX.1-dev_training_faces), [bitmind/celeb-a-hq___FLUX.1-dev_training_faces](https://huggingface.co/datasets/bitmind/celeb-a-hq___FLUX.1-dev_training_faces) |
| `sdxl_faces` | Stable Diffusion XL ([Podell et al., 2023](https://arxiv.org/abs/2307.01952)) faces generated from FFHQ captions | 400 | 100 | [bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces](https://huggingface.co/datasets/bitmind/ffhq-256___stable-diffusion-xl-base-1.0_training_faces) |
| `sfhq_t2i` | 1024 px portraits from FLUX.1-dev, FLUX.1-pro and SDXL | 822 | 178 | SFHQ-T2I, [bitmind/SyntheticFacesHighQuality-T2I](https://huggingface.co/datasets/bitmind/SyntheticFacesHighQuality-T2I) |
| `faceswap` | Face swaps on CelebA photos | 414 | 86 | [bitmind/face-swap](https://huggingface.co/datasets/bitmind/face-swap) |

Real faces:

| Key | What it is | Train | Test | From |
|---|---|---|---|---|
| `ffhq_stylegan` | Flickr-Faces-HQ photos | 600 | 250 | 140k Real and Fake Faces |
| `dff` | IMDB-WIKI celebrity photos | 400 | 150 | DeepFakeFace |
| `celebdf` | YouTube interview frames | 480 | 160 | Celeb-DF v2 |
| `dfd` | Frames of the paid actors | 240 | 100 | Google DFD |
| `celeba` | CelebA photos, the same size and format as the face swaps | 402 | 98 | the real half of the Collaborative Diffusion set in [DF40](https://huggingface.co/datasets/ManhQuangAI/df-40-test-full) |
| `celebahq` | CelebA-HQ 1024 px photos ([Karras et al., 2018](https://arxiv.org/abs/1710.10196)) | 570 | 130 | [bitmind/celeb-a-hq](https://huggingface.co/datasets/bitmind/celeb-a-hq) |
| `coco` | Faces found in everyday COCO photos ([Lin et al., 2014](https://arxiv.org/abs/1405.0312)) | 536 | 114 | [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) |
| `flickr` | Faces found in Flickr30k photos of people | 517 | 133 | [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) |
| `pexels` | Faces found in Pexels stock photos: professional lighting, colour grading, props | 569 | 131 | [ujin-song/pexels-image-1M](https://huggingface.co/datasets/ujin-song/pexels-image-1M) |

The everyday and stock photos are there for a reason. The first version only knew
tidy, front-facing dataset portraits, and it flagged unusual real photos as fake.

## Sources that are only tested, never trained on

These measure what a detector is really for: a kind of fake or photo it has not met.

| Key | What it is | Crops | From |
|---|---|---|---|
| `midjourney` | Midjourney faces (fake) | 400 | DF40 ([Yan et al., 2024](https://arxiv.org/abs/2406.13495)) |
| `stylegan3` | StyleGAN3 faces (fake) ([Karras et al., 2021](https://arxiv.org/abs/2106.12423)) | 300 | [34data/StyleGAN3_1024x1024_images](https://huggingface.co/datasets/34data/StyleGAN3_1024x1024_images) |
| `collabdiff` | Collaborative Diffusion faces (fake) ([Huang et al., 2023](https://arxiv.org/abs/2304.10530)) | 300 | DF40 |
| `heygen` | Frames of HeyGen talking-head avatars (fake), 3 per clip | 153 | DF40 |
| `youtube` | Frames of the real clips that ship with the HeyGen set, 3 per clip | 150 | DF40 |
| `wfir` | whichfaceisreal.com: StyleGAN faces and real photos | 300 + 300 | DF40 |
| `lfw` | Labeled Faces in the Wild press photos (real), one per person | 500 | [bitmind/lfw](https://huggingface.co/datasets/bitmind/lfw) |

## How the crops are made

`ml/scripts/prepare_data.py` builds the four core sources and
`ml/scripts/extend_data.py` adds the rest. Both stream or download only what they need
and run every image through the app's own face detector and crop (largest face, 25%
margin). For photos of scenes (COCO, Flickr30k, Pexels) an image is kept only when a
face is found with high confidence and is large enough to judge.

### Splits and leakage controls

* **Celeb-DF:** the official `List_of_testing_videos.txt` defines the test set, and
  every identity that appears in it is removed from training.
* **DFD:** split by actor. Actors 01 to 20 train, 21 to 28 test.
* **DeepFakeFace:** the real and fake archives contain the same people, so the split
  is done by identity hash. A person is never in both train and test.
* **140k Faces:** the official train and test files are used.
* **The newer sources:** split 80/20 by a hash of the file name, so a file is in one
  split only. HeyGen and YouTube frames are grouped by clip.
* **Unseen sources** never enter training, model selection or calibration.

### Augmented copies (training only)

Every training crop gets two extra copies, so the head cannot lean on resolution,
compression or colour:

* a **harsh** copy with one or two of: JPEG quality 30 to 75, shrinking to 64 to 160 px
  and enlarging again, Gaussian blur, noise, and a colour cast with changed gamma,
  contrast and saturation (stage lights, filters);
* a **shared online** copy: scaled down to 160 to 512 px and saved as JPEG quality 70
  to 95, which is what almost every photo on the web has been through.

Test and unseen crops are never augmented. `ml/scripts/eval_robustness.py` degrades a
sample of them separately to measure the effect.

## Real-world images (version 3)

`ml/scripts/collect_wild.py` reads Parquet row groups straight from the Hugging Face Hub
and keeps only images with a person in the caption and a face of at least 64 px.

| Source | License | Use |
|---|---|---|
| [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) core train | CC BY-NC 4.0 | Training (AI images from about 70 generators, real LAION and Pexels photos) |
| OpenFake core test | CC BY-NC 4.0 | Test only (newest generators) |
| OpenFake reddit test | CC BY-NC 4.0 | Split 50/50 by file hash: half training, half test |
| [detection-datasets/coco](https://huggingface.co/datasets/detection-datasets/coco) shards 3 to 9 / 10 to 13 | CC BY 4.0 annotations, Flickr image terms | Training / test |

Because of OpenFake's non-commercial license, the version 3 heads are for research and
education only. Images are not redistributed; only the trained heads and a retrieval index
of numbers and labels are in this repository.

## Licenses and responsible use

These datasets are for research and education, and each keeps the license shown on its
own page. Celeb-DF and FaceForensics++ require agreeing to their terms of use; FFHQ is
Creative Commons BY-NC-SA 4.0; COCO and Flickr30k photos belong to their Flickr
authors; Pexels photos are under the [Pexels License](https://www.pexels.com/license/).
DeepShield does not redistribute any dataset: the repository holds only the trained
head (1,155 numbers) and a search index of 64-number fingerprints with labels.

The six sample photos in the app are three Pexels stock photos and three AI portraits
(FLUX.1 and SDXL from SFHQ-T2I, Midjourney from DF40). None of them is in the training
split.

## Rebuilding

```bash
pip install -r ml/requirements.txt
python ml/scripts/prepare_data.py --out data/crops     # the four core sources
python ml/scripts/extend_data.py --out data/crops      # newer generators, everyday photos, unseen sources
python ml/scripts/extend_data.py --out data/crops --augment
python ml/scripts/train_fusion.py --data data/crops    # features and head; about 3 hours on a 2-core CPU
python ml/scripts/build_similar_index.py               # closest-known-cases index
python ml/scripts/eval_robustness.py                   # degraded copies of held-out faces
python ml/scripts/eval_videos.py --per-class 30        # end-to-end video test
python ml/scripts/make_figures.py
```

Use `--scale 0.1` on the two data scripts for a quick smoke run.
