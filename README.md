---
title: DeepShield
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Detect AI-generated and face-swapped faces in photos and videos
---

# DeepShield: AI-powered deepfake detector

DeepShield checks whether a face in a photo or video is real or AI-manipulated,
shows **where** the evidence is (Grad-CAM heatmap) and **when** it appears in a
video (per-frame timeline), and exports a PDF forensic report. The interface works
in English and Arabic.

Built by **Moamen Ahmed Fouad** for the DEPI Machine Learning track (2026).

**Live demo:** `https://<your-hf-username>-deepshield.hf.space` (see [Deployment](docs/DEPLOYMENT.md))

## Results

Held-out test set: 1,320 face crops from four public datasets, never seen during
training or model selection.

| | AUC | Accuracy | F1 |
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

**Whole videos, end to end** (90 held-out videos through the full app pipeline: frame
sampling, face detection, fusion, aggregation):

| Videos | AUC | Accuracy at 0.5 |
|---|---|---|
| Google DFD, held-out actors (30) | 0.951 | 76.7% (every fake caught) |
| Celeb-DF v2, official test list (60) | 0.713 | 65.0% |
| All (90) | 0.761 | 68.9% |

Combining 24 frames helps: Celeb-DF rises from 0.59 per frame to 0.71 per video.
One video in five lands in the "inconclusive" band instead of getting a forced answer.

<p align="center"><img src="reports/per_dataset_auc.png" width="720" alt="AUC per dataset, fused model vs best single pretrained model"></p>

Full numbers, method and limitations: [Model card](docs/MODEL_CARD.md).

## How it works

```
upload ─► face detection (YuNet) ─► square crop with 25% context
       ─► two frozen pretrained detectors (ViT-S + SigLIP)
       ─► fusion head trained on our data (logistic regression, 1,154 features)
       ─► image: most suspicious face  |  video: 24 frames, mean + top-quartile
       ─► verdict (real / inconclusive / fake) + Grad-CAM + timeline + PDF
```

1. **Five public deepfake detectors** from the Hugging Face Hub were benchmarked.
   Alone, the best one scored 0.82 AUC, and two of them were near coin-flip on our data.
2. **The data picked the backbones.** Every pair was scored on a validation split
   carved out of training data. CommunityForensics ViT + SigLIP v1 won.
3. **Light fine-tune:** a 1,155-parameter fusion head learns how to combine their
   embeddings and scores. It trains in seconds on a CPU and adds 0.07 AUC over simple
   averaging.
4. **No leakage:** identities, actors and source videos never appear in both train
   and test ([details](docs/DATASETS.md)).

Architecture diagram and design decisions: [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
git clone https://github.com/MoamenFouad/deep-fake-ai-powered-detector.git
cd deep-fake-ai-powered-detector

# Docker
docker compose up --build            # http://localhost:7860

# or Python 3.11+
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 7860
```

The first start downloads about 700 MB of model weights.

## API

Interactive docs at `/docs`.

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
             "model_scores": {"CommunityForensics-DeepfakeDet-ViT": 0.99, "deepfake-detector-model-v1": 0.98}}],
  "engine": "fusion-head"
}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Image (JPG, PNG, WebP, BMP, up to 10 MB) or video (MP4, MOV, AVI, WebM, MKV, up to 60 MB) |
| `GET /api/report/{id}` | PDF report for a previous analysis |
| `GET /api/health` | Model status and engine |
| `GET /api/metrics` | The test metrics shown on the page |

## Reproduce the model

```bash
pip install -r ml/requirements.txt
python ml/scripts/prepare_data.py --out data/crops   # stream and crop 4,760 faces
python ml/scripts/train_fusion.py --data data/crops  # benchmark, select, train, test
python ml/scripts/eval_videos.py --per-class 30      # whole-video evaluation
python ml/scripts/make_figures.py
```

Notebooks: [data exploration](ml/notebooks/01_data_exploration.ipynb),
[training and evaluation](ml/notebooks/02_training_and_evaluation.ipynb),
[optional GPU fine-tune](ml/notebooks/03_full_finetune_gpu.ipynb).

## Project layout

```
app/            FastAPI service, web UI (app/static), services (detector, classifier, analyzer, report)
ml/             data preparation, training, evaluation scripts and notebooks
models/         trained fusion head (backbone weights download at build time)
reports/        metrics and figures from the held-out test
tests/          pytest suite (runs with a stub model, no downloads)
docs/           architecture, datasets, deployment, model card
```

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && DS_STUB_MODELS=1 pytest -q
```

## Limitations

DeepShield is a screening aid, not proof. It is weakest on high-quality face swaps
(Celeb-DF), on heavily compressed re-uploads, and on generators released after its
training data. Read the [model card](docs/MODEL_CARD.md) before relying on a result.

## Credits

* Backbones: [CommunityForensics-DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
  ([Park and Owens, CVPR 2025](https://arxiv.org/abs/2411.04125)) and
  [deepfake-detector-model-v1](https://huggingface.co/prithivMLmods/deepfake-detector-model-v1).
* Face detector: [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
* Datasets: [140k Real and Fake Faces](https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces),
  [DeepFakeFace](https://arxiv.org/abs/2309.02218), [Celeb-DF v2](https://arxiv.org/abs/1909.12962),
  [FaceForensics++ / Google DFD](https://arxiv.org/abs/1901.08971).
* Fonts: IBM Plex Sans and IBM Plex Sans Arabic (SIL Open Font License).

## License

MIT, see [LICENSE](LICENSE). Model weights and datasets keep their own licenses.
