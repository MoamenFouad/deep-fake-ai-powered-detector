# Running and sharing DeepShield

DeepShield runs on any machine with Python 3.12. No GPU, Docker or cloud account is
needed. The two backbone models are downloaded once, for free, from the
[Hugging Face Hub](https://huggingface.co/) the first time the app starts (about
700 MB) and are cached after that.

## 1. Run it on your computer

```bash
git clone https://github.com/MoamenFouad/deep-fake-ai-powered-detector.git
cd deep-fake-ai-powered-detector

python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

uvicorn app.main:app --port 7860
```

Open <http://localhost:7860>. About 30 seconds after start-up,
<http://localhost:7860/api/health> shows `"models_loaded": true`.

## 2. Share a public link (Cloudflare Tunnel)

A [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
gives the running app a public HTTPS address. It is free and needs no account.

```bash
# once
brew install cloudflared             # macOS. Other systems: see the Cloudflare link above

# every time, in a second terminal while the app is running
cloudflared tunnel --url http://localhost:7860
```

It prints a link like `https://random-words.trycloudflare.com`. Anyone can open it,
including from a phone. The link works while your computer and both terminals stay
open, and a new link is generated every time you start the tunnel.

Before a presentation: start the app, start the tunnel, open the link on your phone
once to check it, and keep the laptop plugged in with sleep turned off.

## 3. Docker (optional)

The repository also includes a `Dockerfile` for anyone who prefers containers or
wants to host the app on a server:

```bash
docker compose up --build            # http://localhost:7860
```

## Configuration

All settings are environment variables with sensible defaults (`app/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `DS_MODELS` | the two chosen backbones | Comma-separated Hugging Face model ids |
| `DS_HEAD_PATH` | `models/fusion_head.npz` | Fusion head weights |
| `DS_SIMILAR_INDEX` | `models/similar_index.npz` | Closest-known-cases index |
| `DS_SIMILAR_K` | `5` | Number of closest cases shown per face |
| `DS_STUB_MODELS` | `0` | `1` uses a deterministic stub (tests, CI) |
| `DS_MAX_IMAGE_MB` | `10` | Image upload limit |
| `DS_MAX_VIDEO_MB` | `60` | Video upload limit |
| `DS_VIDEO_FRAMES` | `24` | Frames sampled per video |
| `DS_VIDEO_MAX_SECONDS` | `120` | Only the first N seconds are sampled |
| `DS_FAKE_THRESHOLD` | `0.5` | Decision threshold |
| `DS_UNCERTAIN_MARGIN` | `0.1` | Half-width of the "inconclusive" band |

## Performance

Measured on 2 CPU cores: a photo with one face takes about 2 seconds including the
Grad-CAM map, and a 24-frame video takes about 15 to 25 seconds. Memory use is about
1.5 GB with both backbones loaded.
