<div align="center">

# 🛡️ DeepShield: Is This Face Real?

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

[![CI](https://github.com/MoamenFouad/deep-fake-ai-powered-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/MoamenFouad/deep-fake-ai-powered-detector/actions/workflows/ci.yml)
![Test AUC](https://img.shields.io/badge/test_AUC-0.965-0d0d0d)
![Version](https://img.shields.io/badge/version-2.0.0-ffd400)
![License](https://img.shields.io/badge/license-MIT-0d0d0d)

> **Drop a photo or a video. See if the face is real. See why.**

</div>

DeepShield is an AI-powered deepfake detector built for the DEPI Machine Learning track.
Upload a photo or a short video and it finds every face, checks it with two pretrained
deepfake detectors fused by a model we trained on 20 public sources, and shows the
evidence: **where** in the face (Grad-CAM heatmap), **when** in the video (per-frame
timeline), and **which known fakes it resembles** (closest training cases). Every check
can be exported as a PDF forensic report.

<p align="center">
  <img src="docs/images/demo.gif" width="860" alt="Picking a sample portrait, the verdict appearing, and dragging the evidence-map slider">
</p>

---

## 🚀 Features

### 🧠 Detection
- **Two frozen expert detectors:** [CommunityForensics ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) and [SigLIP deepfake-detector v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1), each trained on different kinds of fakes
- **Fusion head** trained on 8,075 faces from 13 sources, from StyleGAN to FLUX.1 and SDXL, plus everyday and stock photos: logistic regression over 1,154 features (both embeddings plus both scores) that lifts held-out test AUC from 0.853 (simple average) to **0.965**
- **A separate video head** trained on video frames, because compression and motion blur fooled the photo head
- **YuNet face detection** with a 25% context crop, so every face in a group photo gets its own score
- **Three-way verdict:** real, inconclusive or likely manipulated, instead of a forced yes/no

### 🔍 Explainability
- **Evidence map:** Grad-CAM heatmap with a drag slider to compare it against the original face
- **Detector agreement:** both backbones' raw scores, with a note when the fusion head overrides them
- **Closest known cases (retrieval):** each face is matched against 8,075 labeled training faces and the 5 nearest are shown with their kind of fake ("5 of 5 are StyleGAN faces"). Only 64-number fingerprints and labels are stored, never images
- **Unfamiliar-image guard:** when even the closest known face is a weak match, the result is reported as inconclusive instead of a confident wrong answer (fires on 2% of test faces, where accuracy drops to 74%)
- **Video timeline:** 24 sampled frames scored and plotted; click a point to jump the video there

### ⚡ Experience
- Analysis starts the moment a file is chosen; drop a file anywhere or paste with Ctrl+V / Cmd+V
- Six built-in sample photos (three real, three AI) with the answer revealed after each check
- Recent checks in the tab, one-click summary copy
- **PDF forensic report** for every analysis
- Fully responsive, keyboard accessible, files checked in memory and never stored

---

## 📸 Screenshots

<p align="center"><img src="docs/images/home.png" width="900" alt="DeepShield home page"></p>

| AI-generated face | Real photo |
|---|---|
| *A Midjourney portrait caught at 99.3%; all 5 nearest cases are FLUX.1 and SDXL fakes* | *A Pexels stock photo at 19.6%; the nearest cases are real stock and celebrity photos* |
| ![Fake result](docs/images/result-fake.png) | ![Real result](docs/images/result-real.png) |

| Video face swap | PDF forensic report |
|---|---|
| *A held-out Google DFD face swap: 24-frame timeline, caught at 77.5% although both raw detectors lean real* | *Verdict, crops, heatmaps, backbone scores and closest cases* |
| ![Video result](docs/images/result-video.png) | ![PDF report](docs/images/pdf-report.png) |

| On a phone | Accuracy, live from the test set |
|---|---|
| ![Mobile](docs/images/mobile.png) | ![Accuracy section](docs/images/accuracy.png) |

---

## 📊 Results

Every number below comes from data that was **never used for training or for any
modelling choice**: held-out people, videos and files from the 13 training sources, plus
7 sources the model never saw at all ([data](docs/DATASETS.md), [model card](docs/MODEL_CARD.md)).

| Held-out photos (face crops) | n | DeepShield v2 | v1 | Best single detector |
|---|---|---|---|---|
| Original four test sets (StyleGAN, DeepFakeFace, Celeb-DF, DFD) | 1,320 | **0.931** | 0.920 | 0.816 |
| Newer sources (FLUX, SDXL, face swaps, everyday and stock photos) | 1,065 | **0.992** | 0.858 | 0.885 |
| **All held-out test faces** | 2,385 | **0.965** (89.1% accuracy) | 0.893 | 0.857 |
| Never-trained sources (Midjourney, StyleGAN3, HeyGen, LFW, ...) | 2,403 | 0.969 (91.1% accuracy) | 0.919 | 0.994 |

AUC: 1.0 is perfect, 0.5 is a coin flip. On the never-trained sources one backbone on its
own ranks slightly better than the fused model; everywhere else the fusion wins clearly.

| Kind of fake | Caught | Kind of real photo | Correctly passed |
|---|---|---|---|
| SDXL, Midjourney, Collaborative Diffusion | 100% | Flickr everyday photos | 99.3% |
| FLUX.1 faces and portraits | 99.0 to 99.4% | Pexels stock photos | 97.7% |
| StyleGAN and StyleGAN3 | 97.4 to 98.7% | CelebA-HQ portraits | 94.6% |
| Face swaps (CelebA) | 88.4% | YouTube clips | 93.3% |
| Celeb-DF v2 video frames | 67.5% | LFW news photos | 82.0% |
| HeyGen talking-head avatars | **39.2%**, the weak spot | | |

**Robust to everyday edits.** On 600 held-out faces, AUC stays at 0.932 after "shared
online" resizing and JPEG, and 0.898 after messaging-app compression (v1: 0.809 and 0.696).

**Whole videos** (90 held-out videos through the app pipeline, video head):

| Videos | AUC | v1 AUC |
|---|---|---|
| Celeb-DF v2, official test list (60) | **0.817** | 0.713 |
| Google DFD, held-out actors (30) | 0.907 | 0.951 |
| All (90) | **0.778** | 0.761 |

One video in three is answered "inconclusive" rather than guessed; when the app does
answer, it is right 79.7% of the time.

<p align="center"><img src="reports/accuracy_by_source.png" width="720" alt="Share of held-out faces judged correctly per source, v2 against v1"></p>

---

## 🏗️ Architecture

```
Photo or video (web UI or POST /api/analyze)
│
▼
YuNet face detector ──────────────► square crop, 25% context
│                                   (videos: up to 24 sampled frames)
│
├─── CommunityForensics ViT ──────► 384-d embedding + fake score
├─── SigLIP detector v1 ──────────► 768-d embedding + fake score
│
▼
Fusion head (logistic regression, 1,154 features)
  photos: photo head, videos: video head
│
├─── Verdict ─────────────────────► real / inconclusive / fake
│     image: most suspicious face, video: mean + top-quartile
├─── Grad-CAM ────────────────────► evidence heatmap per face
├─── Retrieval (PCA-64, cosine) ──► 5 closest labeled training faces
└─── ReportLab ───────────────────► PDF forensic report
```

How the model was built:

1. **Data from 20 sources:** old and new generators, face swaps, studio portraits and everyday snapshots, so the model cannot cheat by spotting a photo style. Seven sources are held back entirely to test generators it never saw.
2. **Let the data pick the backbones:** nine public detectors and encoders were compared with leave-one-source-out validation on the training split. CommunityForensics ViT + SigLIP v1 stayed best.
3. **Light fine-tune:** a 1,155-parameter fusion head learns to combine them, chosen by how well it handles a whole domain it was not trained on, then calibrated. It trains in seconds on a CPU.
4. **A head for videos:** frames of videos get their own head trained on video frames, with a whole-video calibration.
5. **Retrieval:** a 5-nearest-neighbour vote alone reaches 0.90 AUC, so it is shown as supporting evidence, and it flags images unlike anything in training as inconclusive.

Design decisions and diagram: [ARCHITECTURE.md](docs/ARCHITECTURE.md). Version history: [CHANGELOG.md](CHANGELOG.md).

### Tech Stack

**Backend**

| Technology | Purpose |
|---|---|
| FastAPI + Uvicorn | REST API and static web app |
| Pydantic | Typed request and response schemas |
| ReportLab | PDF forensic reports |
| Python 3.12 | Runtime |

**AI and ML**

| Technology | Purpose |
|---|---|
| PyTorch + Transformers | Running the two ViT / SigLIP backbones |
| scikit-learn | Fusion head, PCA for the retrieval index, metrics |
| OpenCV YuNet | Face detection |
| Grad-CAM | Evidence heatmaps |
| NumPy | Cosine nearest-neighbour search over 8,075 fingerprints |

**Frontend**

| Technology | Purpose |
|---|---|
| HTML, CSS, vanilla JavaScript | No build step, loads instantly |
| Archivo (variable font) | Self-hosted typography |
| Inline SVG | Gauge, timeline and accuracy charts |

**Infrastructure**

| Technology | Purpose |
|---|---|
| Docker / Docker Compose | One-command run; image published to the GitHub Container Registry |
| GitHub Actions | Lint and tests on every push |
| Hugging Face Hub | Free download of the two backbone models |
| Cloudflare Tunnel | Free public link to the app running on a laptop |
| Git LFS | Model files, fonts and images |

---

## 🛠️ Installation and Setup

**Prerequisites:** Python 3.12+. No GPU, Docker or cloud account needed. About 700 MB of model weights download on first start.

**1. Clone the repository**

```bash
git clone https://github.com/MoamenFouad/deep-fake-ai-powered-detector.git
cd deep-fake-ai-powered-detector
```

**2. Install and run**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 7860
# App runs on http://localhost:7860
```

Prefer containers? `docker compose up --build` does the same.

**3. Run the tests**

```bash
pip install -r requirements-dev.txt
ruff check . && DS_STUB_MODELS=1 pytest -q
```

Tests use a stub model, so they run in seconds without downloads.

---

## 🌍 Hosting

The app needs about **2 GB of RAM** and no GPU. Where it fits:

| Option | Cost | Fits? |
|---|---|---|
| Your laptop + [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) | Free, no account | Yes, while the laptop is on |
| [Azure for Students](https://learn.microsoft.com/en-us/azure/education-hub/about-azure-for-students) + Container Apps | Free $100 credit, no card | Yes |
| Render, 1 CPU / 2 GB instance | $25 per month | Yes |
| Render free instance | Free | **No**, it has 512 MB of RAM ([Render pricing](https://render.com/pricing)) |

Every push to `main` publishes a ready image, `ghcr.io/moamenfouad/deep-fake-ai-powered-detector`,
that runs on any of them. Step-by-step guides: [DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## 🔌 API Reference

Interactive docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analyze` | Image (JPG, PNG, WebP, BMP, up to 10 MB) or video (MP4, MOV, AVI, WebM, MKV, up to 60 MB) |
| GET | `/api/report/{id}` | PDF report for a previous analysis |
| GET | `/api/health` | Model status, engine and retrieval index |
| GET | `/api/metrics` | The test metrics shown on the page |

```bash
curl -F "file=@photo.jpg" http://localhost:7860/api/analyze
```

```json
{
  "id": "3f2c...",
  "media_type": "image",
  "verdict": "fake",
  "fake_probability": 0.997,
  "confidence": 0.99,
  "faces": [{"box": {"x": 41, "y": 52, "w": 170, "h": 170}, "fake_probability": 0.997,
             "model_scores": {"CommunityForensics-DeepfakeDet-ViT": 0.99, "deepfake-detector-model-v1": 0.98},
             "similar": [{"label": "fake", "kind": "stylegan", "similarity": 0.89}]}],
  "engine": "fusion-head"
}
```

---

## 🔁 Reproduce the Model

```bash
pip install -r ml/requirements.txt
python ml/scripts/prepare_data.py --out data/crops             # the four core datasets
python ml/scripts/extend_data.py --out data/crops              # 16 more sources
python ml/scripts/extend_data.py --out data/crops --augment    # augmented training copies
python ml/scripts/train_fusion.py --data data/crops            # select, train and test the photo head
python ml/scripts/train_video_head.py                          # the video head
python ml/scripts/build_similar_index.py                       # closest-known-cases index
python ml/scripts/eval_robustness.py                           # everyday-edit robustness
python ml/scripts/eval_videos.py --per-class 30                # whole-video evaluation
python ml/scripts/make_figures.py
```

Notebooks: [data exploration](ml/notebooks/01_data_exploration.ipynb),
[training and evaluation](ml/notebooks/02_training_and_evaluation.ipynb),
[optional GPU fine-tune](ml/notebooks/03_full_finetune_gpu.ipynb).

## 📁 Project Structure

```
app/            FastAPI service, web UI (app/static), services (detector, classifier, analyzer, similar, report)
ml/             data preparation, training, evaluation scripts and notebooks
models/         photo head, video head and retrieval index (backbone weights download at start)
reports/        metrics and figures from the held-out test
tests/          pytest suite (runs with a stub model, no downloads)
docs/           architecture, datasets, deployment, model card, screenshots
```

---

## ⚠️ Limitations

DeepShield is a screening aid, not proof. It is weakest on talking-head avatars
(HeyGen), on high-quality face swaps (Celeb-DF), on generators released after its
training data, and on unusual photos such as strong coloured stage lighting. The
unfamiliar-image guard catches some of these cases, not all. It judges faces only: a
real face placed into a fake scene still reads as real. Read the
[model card](docs/MODEL_CARD.md) before relying on a result.

## 🙏 Credits

- **Backbones:** [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) ([Park and Owens, CVPR 2025](https://arxiv.org/abs/2411.04125)) and [deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1)
- **Face detector:** [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Datasets:** 20 public sources, each credited in [DATASETS.md](docs/DATASETS.md), including [140k Real and Fake Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces), [DeepFakeFace](https://arxiv.org/abs/2309.02218), [Celeb-DF v2](https://arxiv.org/abs/1909.12962), [FaceForensics++ / Google DFD](https://arxiv.org/abs/1901.08971), SFHQ-T2I, CelebA-HQ, COCO, Flickr30k and Pexels
- **Sample photos:** three from [Pexels](https://www.pexels.com/license/) (free to use), three AI portraits from SFHQ-T2I and DF40
- **Font:** [Archivo](https://github.com/Omnibus-Type/Archivo) by Omnibus-Type (SIL Open Font License)

## 👤 Author

Built by **Moamen Ahmed Fouad** for the DEPI Machine Learning track (2026).
[GitHub @MoamenFouad](https://github.com/MoamenFouad)

## 📄 License

MIT, see [LICENSE](LICENSE). Model weights and datasets keep their own licenses.
