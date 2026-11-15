# Changelog

## 2.0.0 (November 2026)

### Model
- **Bigger, newer training data.** 12,863 face crops from 20 public sources instead
  of 4,760 from 4. New fakes: FLUX.1, SDXL and SFHQ-T2I portraits, CelebA face swaps.
  New real photos: CelebA-HQ, COCO, Flickr30k and Pexels stock photos, so studio
  lighting and everyday snapshots are both covered. Seven more sources (Midjourney,
  StyleGAN3, Collaborative Diffusion, HeyGen avatars, LFW, YouTube, whichfaceisreal)
  are kept out of training and only tested. Every training crop also has two
  augmented copies (harsh degradations, and "shared online" resizing plus JPEG).
- **Retrained photo head** with leave-one-domain-out model selection and Platt
  calibration. Held-out test AUC 0.965 on 2,385 faces (v1 head: 0.893); 0.931 on the
  original four test sets (v1: 0.920); 0.969 on the never-trained sources (v1: 0.919).
- **New video head.** The photo head started to read video compression as a sign of
  a fake, so videos now use a second head trained only on frames of training videos,
  with its own whole-video calibration. 90 held-out videos: AUC 0.778 (v1: 0.761),
  Celeb-DF v2 0.817 (v1: 0.713).
- **Robustness report** (`ml/scripts/eval_robustness.py`): AUC after resizing, JPEG,
  blur, messaging-app compression and coloured lighting, all higher than v1.
- Grad-CAM now explains the fused score through both backbones and says which side
  (fake or real) the highlighted areas support.
- Closest-known-cases index rebuilt on the 8,075 training faces; unfamiliar-image
  threshold re-measured (0.62).

### App
- Six new sample photos (three Pexels stock photos, FLUX, Midjourney and SDXL
  portraits, none of them used in training) with an "answer" reveal after each check.
- Faceless images are always reported as inconclusive.
- Safer uploads: images over 60 megapixels or under 16 px are refused, very large
  photos are scaled down before decoding, 16-bit images are handled, video frames are
  capped at 1280 px so long 4K clips cannot exhaust memory, broken frame-rate metadata
  is ignored.
- The API answers 503 with `Retry-After` while the models load, and the page retries
  by itself.
- Each result reports the analyzed image size and which side the heatmap explains;
  file names are escaped in the PDF report.

### Hosting
- A ready-made Docker image is published to the GitHub Container Registry on every
  push, and the container listens on `$PORT` when a host sets it.
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) compares hosts. Render's free instance (512 MB)
  is too small for two vision transformers; Azure for Students (free credit, no card)
  and a laptop with a Cloudflare tunnel both work.

## 1.2.0 (November 2026)

### New
- **Unfamiliar-image guard.** If even the closest known training face is a weak match
  (similarity below 0.576, the 2nd percentile measured on training faces), the face is
  outside what the model has learned. The result is reported as inconclusive with an
  "Unfamiliar image" note instead of a confident score. On the test split it fires on
  2.0% of faces, where the fusion head's accuracy is 66.7% versus 83.1% elsewhere.
  Each face in `POST /api/analyze` has an `unfamiliar` flag, and the PDF report
  explains it.

### Changed
- Version 1.2.0.

## 1.1.0 (November 2026)

### New
- **Closest known cases.** Every face is matched against the 3,440 labeled training
  faces and the 5 most similar are shown with their label and kind of fake
  (for example "5 of the 5 most similar known faces are fakes, mostly StyleGAN face").
  This is the retrieval half of RAG. The index stores numbers and labels only, never
  images (`ml/scripts/build_similar_index.py`, `app/services/similar.py`).
- **Redesigned web app.** An "evidence board" look: black on white, heavy wide
  headlines, and evidence-marker yellow for the numbered exhibits (verdict, evidence
  map, closest known cases, timeline). The verdict is a full-colour block with a
  real / unsure / fake gauge.
- **Faster flow.** Analysis starts as soon as a file is chosen. Drop a file anywhere
  on the page, or paste an image with Ctrl+V (Cmd+V on a Mac).
- **Recent checks.** Thumbnails of this session's results; click one to reopen it.
  Kept in the browser tab only.
- **Interactive video timeline.** Click a point to jump the video to that moment.
- **Detector agreement** explains when the fusion head overrides the two detectors'
  raw scores, and a **Copy summary** button copies a one-line result.
- The PDF report lists the closest known cases.
- `GET /api/health` reports `similar_cases`; each face in `POST /api/analyze` has a
  `similar` list.

### Changed
- The interface is English only.
- Font: Archivo (variable width and weight), self-hosted under the SIL Open Font
  License, replaces IBM Plex.
- Version 1.1.0.
- Hosting: the app is shared from a laptop with a free Cloudflare quick tunnel. The
  Hugging Face Spaces deploy workflow was removed because Docker Spaces on the free
  CPU tier now require a paid plan.

## 1.0.0 (November 2026)
- First release: two pretrained detectors with a trained fusion head (0.920 test AUC),
  Grad-CAM evidence maps, video timeline, PDF reports, Docker image and CI.
