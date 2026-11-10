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
![Test AUC](https://img.shields.io/badge/test_AUC-0.920-0d0d0d)
![Version](https://img.shields.io/badge/version-1.2.0-ffd400)
![License](https://img.shields.io/badge/license-MIT-0d0d0d)

> **Drop a photo or a video. See if the face is real. See why.**

</div>

DeepShield is an AI-powered deepfake detector built for the DEPI Machine Learning track.
Upload a photo or a short video and it finds every face, checks it with two pretrained
deepfake detectors fused by a model we trained on four public datasets, and shows the
evidence: **where** in the face (Grad-CAM heatmap), **when** in the video (per-frame
timeline), and **which known fakes it resembles** (closest training cases). Every check
can be exported as a PDF forensic report.

<p align="center">
  <img src="docs/images/demo.gif" width="860" alt="Choosing the AI face sample, the verdict appearing, and dragging the evidence-map slider">
</p>

---

## 🚀 Features

### 🧠 Detection
- **Two frozen expert detectors:** [CommunityForensics ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) and [SigLIP deepfake-detector v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1), each trained on different kinds of fakes
- **Fusion head** trained on our data: logistic regression over 1,154 features (both embeddings plus both scores) that lifts test AUC from 0.852 (simple average) to **0.920**
- **YuNet face detection** with a 25% context crop, so every face in a group photo gets its own score
- **Three-way verdict:** real, inconclusive or likely manipulated, instead of a forced yes/no

### 🔍 Explainability
- **Evidence map:** Grad-CAM heatmap with a drag slider to compare it against the original face
- **Detector agreement:** both backbones' raw scores, with a note when the fusion head overrides them
- **Closest known cases (retrieval):** each face is matched against 3,440 labeled training faces and the 5 nearest are shown with their kind of fake ("5 of 5 are StyleGAN faces"). Only 64-number fingerprints and labels are stored, never images
- **Unfamiliar-image guard:** when even the closest known face is a weak match, the result is reported as inconclusive instead of a confident wrong answer (fires on 2% of test faces, where accuracy drops to 67%)
- **Video timeline:** 24 sampled frames scored and plotted; click a point to jump the video there

### ⚡ Experience
- Analysis starts the moment a file is chosen; drop a file anywhere or paste with Ctrl+V / Cmd+V
- Built-in samples, recent checks in the tab, one-click summary copy
- **PDF forensic report** for every analysis
- Fully responsive, keyboard accessible, files checked in memory and never stored

---

## 📸 Screenshots

<p align="center"><img src="docs/images/home.png" width="900" alt="DeepShield home page"></p>

| AI-generated face | Real photo |
|---|---|
| *Caught at 99.9%, both detectors agree, 5 of 5 nearest cases are StyleGAN fakes* | *0.6% manipulation probability; all 5 nearest cases are real FFHQ photos* |
| ![Fake result](docs/images/result-fake.png) | ![Real result](docs/images/result-real.png) |

| Video face swap | PDF forensic report |
|---|---|
| *24-frame timeline; the fusion head catches a swap both raw detectors missed* | *Verdict, crops, heatmaps, backbone scores and closest cases* |
| ![Video result](docs/images/result-video.png) | ![PDF report](docs/images/pdf-report.png) |

| On a phone | Accuracy, live from the test set |
|---|---|
| ![Mobile](docs/images/mobile.png) | ![Accuracy section](docs/images/accuracy.png) |

---

## 📊 Results

Held-out test set: **1,320 face crops** from four public datasets. No identity, actor or
source video appears in both train and test ([details](docs/DATASETS.md)).

| Model | AUC | Accuracy | F1 |
|---|---|---|---|
| **DeepShield (fused)** | **0.920** | **82.7%** | **0.816** |
| Average of the same two backbones | 0.852 | 74.7% | 0.678 |
| Best single pretrained model | 0.816 | 72.5% | 0.644 |

| Dataset (type of fake) | AUC |
|---|---|
| 140k Real and Fake Faces (StyleGAN) | 0.999 |
| DeepFakeFace (Stable Diffusion, InsightFace swaps) | 0.968 |
| Google DFD (video face swaps) | 0.848 |
| Celeb-DF v2 (high-quality video face swaps) | 0.586, the known weak spot |

**Whole videos, end to end** (90 held-out videos through the full app pipeline):

| Videos | AUC | Accuracy at 0.5 |
|---|---|---|
| Google DFD, held-out actors (30) | 0.951 | 76.7% (every fake caught) |
| Celeb-DF v2, official test list (60) | 0.713 | 65.0% |
| All (90) | 0.761 | 68.9% |

Combining 24 frames helps: Celeb-DF rises from 0.59 per frame to 0.71 per video.

<p align="center"><img src="reports/per_dataset_auc.png" width="720" alt="AUC per dataset, fused model vs best single pretrained model"></p>

Full method, numbers and limitations: [Model card](docs/MODEL_CARD.md).

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
│
├─── Verdict ─────────────────────► real / inconclusive / fake
│     image: most suspicious face, video: mean + top-quartile
├─── Grad-CAM ────────────────────► evidence heatmap per face
├─── Retrieval (PCA-64, cosine) ──► 5 closest labeled training faces
└─── ReportLab ───────────────────► PDF forensic report
```

How the model was built:

1. **Benchmark:** five public deepfake detectors from the Hugging Face Hub were tested on our data. The best scored 0.82 AUC; two were near coin-flip.
2. **Let the data pick the backbones:** every pair was scored on a validation split carved out of training data. CommunityForensics ViT + SigLIP v1 won.
3. **Light fine-tune:** a 1,155-parameter fusion head learns to combine them. It trains in seconds on a CPU and adds 0.07 AUC over averaging.
4. **Retrieval:** a 5-nearest-neighbour vote alone reaches 0.81 AUC, so it is shown as supporting evidence while the verdict stays with the fusion head.

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
| NumPy | Cosine nearest-neighbour search over 3,440 fingerprints |

**Frontend**

| Technology | Purpose |
|---|---|
| HTML, CSS, vanilla JavaScript | No build step, loads instantly |
| Archivo (variable font) | Self-hosted typography |
| Inline SVG | Gauge, timeline and accuracy charts |

**Infrastructure**

| Technology | Purpose |
|---|---|
| Docker / Docker Compose | One-command run |
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

## 🌍 Share a Public Link

The app runs on a normal laptop CPU, and a free
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
turns it into a public HTTPS link that opens on any phone. No account or credit card needed.

```bash
brew install cloudflared                        # once (macOS)
cloudflared tunnel --url http://localhost:7860  # while the app is running
```

Full guide: [DEPLOYMENT.md](docs/DEPLOYMENT.md).

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
python ml/scripts/prepare_data.py --out data/crops   # stream and crop 4,760 faces
python ml/scripts/train_fusion.py --data data/crops  # benchmark, select, train, test
python ml/scripts/eval_videos.py --per-class 30      # whole-video evaluation
python ml/scripts/build_similar_index.py             # closest-known-cases index
python ml/scripts/make_figures.py
```

Notebooks: [data exploration](ml/notebooks/01_data_exploration.ipynb),
[training and evaluation](ml/notebooks/02_training_and_evaluation.ipynb),
[optional GPU fine-tune](ml/notebooks/03_full_finetune_gpu.ipynb).

## 📁 Project Structure

```
app/            FastAPI service, web UI (app/static), services (detector, classifier, analyzer, similar, report)
ml/             data preparation, training, evaluation scripts and notebooks
models/         trained fusion head and retrieval index (backbone weights download at start)
reports/        metrics and figures from the held-out test
tests/          pytest suite (runs with a stub model, no downloads)
docs/           architecture, datasets, deployment, model card, screenshots
```

---

## ⚠️ Limitations

DeepShield is a screening aid, not proof. It is weakest on high-quality face swaps
(Celeb-DF), on heavily compressed re-uploads, and on generators released after its
training data, and unusual photos such as strong coloured stage lighting can fool it.
The unfamiliar-image guard catches some of these cases, not all. It judges faces only: a real face placed into a fake scene still reads as
real. Read the [model card](docs/MODEL_CARD.md) before relying on a result.

## 🙏 Credits

- **Backbones:** [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) ([Park and Owens, CVPR 2025](https://arxiv.org/abs/2411.04125)) and [deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1)
- **Face detector:** [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Datasets:** [140k Real and Fake Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces), [DeepFakeFace](https://arxiv.org/abs/2309.02218), [Celeb-DF v2](https://arxiv.org/abs/1909.12962), [FaceForensics++ / Google DFD](https://arxiv.org/abs/1901.08971)
- **Font:** [Archivo](https://github.com/Omnibus-Type/Archivo) by Omnibus-Type (SIL Open Font License)

## 👤 Author

Built by **Moamen Ahmed Fouad** for the DEPI Machine Learning track (2026).
[GitHub @MoamenFouad](https://github.com/MoamenFouad)

## 📄 License

MIT, see [LICENSE](LICENSE). Model weights and datasets keep their own licenses.
