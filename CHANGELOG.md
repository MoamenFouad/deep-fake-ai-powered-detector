# Changelog

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

## 1.0.0 (November 2026)
- First release: two pretrained detectors with a trained fusion head (0.920 test AUC),
  Grad-CAM evidence maps, video timeline, PDF reports, Docker image, CI and Hugging
  Face Spaces deployment.
