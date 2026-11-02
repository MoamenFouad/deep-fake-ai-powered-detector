# Deployment

DeepShield ships as one Docker image. The default target is **Hugging Face Spaces**
(free CPU tier, public URL, no credit card). The same image runs anywhere Docker runs.

## Hugging Face Spaces (recommended)

### 1. One-time setup on Hugging Face (5 minutes)

1. Create a free account at [huggingface.co/join](https://huggingface.co/join).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space) and fill in:
   * **Space name:** `deepshield`
   * **License:** MIT
   * **SDK:** Docker, then **Blank** template
   * **Hardware:** CPU basic (free)
   * **Visibility:** Public
3. Create an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
   **Create new token**, type **Write**, name it `github-deploy`. Copy it once;
   Hugging Face will not show it again. Never paste it into code, chats or commits.

### 2a. Automatic deploys from GitHub (recommended)

Every push to `main` that passes CI is pushed to the Space by
`.github/workflows/deploy-hf.yml`.

1. In GitHub, open the repo, then **Settings > Secrets and variables > Actions**.
2. **Secrets** tab, **New repository secret**: name `HF_TOKEN`, value: the token.
3. **Variables** tab, **New repository variable**: name `HF_SPACE`, value
   `<your-hf-username>/deepshield`.
4. Push to `main` (or run the workflow manually from the **Actions** tab).
5. Watch the build in the Space's **Logs** tab. The first build takes 8 to 12
   minutes because it bakes about 700 MB of model weights into the image. Later
   builds reuse cached layers.

### 2b. Manual push (alternative)

```bash
brew install git-lfs        # macOS; once per machine
git lfs install
git remote add space https://huggingface.co/spaces/<your-hf-username>/deepshield
git push space main         # username: your HF username, password: the token
```

### 3. Check it

* `https://<your-hf-username>-deepshield.hf.space/` shows the app.
* `.../api/health` returns `"models_loaded": true` and `"engine": "fusion-head"`
  about 30 seconds after the container starts.
* `.../docs` is the interactive API documentation (Swagger UI).

Free Spaces sleep after 48 hours without traffic and wake on the next visit
(about 30 to 60 seconds). Open the link a few minutes before you present.

### Why the README starts with a YAML block

Spaces read the front matter at the top of `README.md` (`sdk: docker`,
`app_port: 7860`, title, emoji, colors). GitHub shows it as a small table; that
is expected.

## Local

```bash
# Docker
docker compose up --build        # http://localhost:7860

# Or plain Python
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

## Configuration

All settings are environment variables with sensible defaults (`app/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `DS_MODELS` | the two chosen backbones | Comma-separated Hugging Face model ids |
| `DS_HEAD_PATH` | `models/fusion_head.npz` | Fusion head weights |
| `DS_STUB_MODELS` | `0` | `1` uses a deterministic stub (tests, CI) |
| `DS_MAX_IMAGE_MB` | `10` | Image upload limit |
| `DS_MAX_VIDEO_MB` | `60` | Video upload limit |
| `DS_VIDEO_FRAMES` | `24` | Frames sampled per video |
| `DS_VIDEO_MAX_SECONDS` | `120` | Only the first N seconds are sampled |
| `DS_FAKE_THRESHOLD` | `0.5` | Decision threshold |
| `DS_UNCERTAIN_MARGIN` | `0.1` | Half-width of the "inconclusive" band |

## Performance on the free tier

Measured on 2 vCPUs: a photo with one face takes about 2 seconds including the
Grad-CAM map; a 24-frame video takes about 15 to 25 seconds. Memory use is about
1.5 GB with both backbones loaded (the free tier has 16 GB).
