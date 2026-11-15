# Running, sharing and hosting DeepShield

DeepShield runs on any machine with Python 3.12 and about 2 GB of free memory. No GPU
is needed. The two backbone models are downloaded once, for free, from the
[Hugging Face Hub](https://huggingface.co/) the first time the app starts (about
700 MB) and are cached after that.

## How much the app needs

Measured on a 2-core CPU machine with both backbones loaded:

| | Memory (RSS) |
|---|---|
| After the first photo | about 1.15 GB |
| Peak while analyzing a video | about 1.45 GB |

A photo with one face takes 1 to 2 seconds including the evidence map, and a 24-frame
video takes 15 to 25 seconds. So a host needs **2 GB of RAM and at least 1 CPU core**.
That number decides which hosting options below are realistic.

## Which host fits

| Option | Cost | Card needed | Fits the app? |
|---|---|---|---|
| Your laptop + Cloudflare quick tunnel | Free | No | Yes. The link works while the laptop is on |
| Azure for Students + Container Apps | Free ($100 student credit) | No | Yes, with 1 vCPU / 2 GiB or more |
| Google Cloud Run | Free tier, then pay as you go | Yes | Yes, with 2 GiB memory |
| Render, 2 GB instance | $25 per month | Yes | Yes |
| Render free instance | Free | No | **No.** It has 512 MB of RAM and the app needs about 1.5 GB |
| Hugging Face Spaces (Docker) | Free accounts report they can no longer create Docker Spaces on the free CPU tier | Yes | Only on a paid plan |

Render's free instance is too small for two vision transformers, and its first instance
with 2 GB costs $25 a month ([Render pricing](https://render.com/pricing)). For a
student project the two free routes are the laptop tunnel and Azure for Students.

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
<http://localhost:7860/api/health> shows `"models_loaded": true`. Until then the page
waits and retries by itself.

## 2. Share a public link from your laptop (Cloudflare Tunnel)

A [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
gives the running app a public HTTPS address. It is free and needs no account.

```bash
# once, macOS without Homebrew (Apple silicon; on an Intel Mac use the -amd64 file)
mkdir -p ~/bin && cd ~/bin
curl -L -o cloudflared.tgz https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64.tgz
tar -xzf cloudflared.tgz && rm cloudflared.tgz && chmod +x cloudflared
# with Homebrew instead: brew install cloudflared

# every time, in a second terminal while the app is running
~/bin/cloudflared tunnel --url http://localhost:7860
```

It prints a link like `https://random-words.trycloudflare.com`. Anyone can open it,
including from a phone. The link works while your computer and both terminals stay
open, and a new link is generated every time you start the tunnel.

Before a presentation: start the app, start the tunnel, open the link on your phone
once to check it, and keep the laptop plugged in with sleep turned off.

## 3. Host it permanently on Azure (free with a student account)

[Azure for Students](https://azure.microsoft.com/en-us/free/students) gives full-time
students a $100 credit with no credit card; you verify with your university email.
[Azure Container Apps](https://azure.microsoft.com/en-us/pricing/details/container-apps/)
also has a monthly free grant of 180,000 vCPU-seconds and 360,000 GiB-seconds, and it
scales to zero when nobody is using the app.

The repository publishes a ready-made Docker image to the GitHub Container Registry on
every push to `main` (`.github/workflows/docker.yml`), with the model weights baked in.

1. Make the image public once: on GitHub open your profile, **Packages**,
   `deep-fake-ai-powered-detector`, **Package settings**, **Change visibility**, Public.
2. Sign up for Azure for Students with your university email.
3. Open the [Azure Cloud Shell](https://shell.azure.com) (a terminal in the browser,
   nothing to install) and run:

```bash
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az containerapp up \
  --name deepshield \
  --resource-group deepshield-rg \
  --location westeurope \
  --image ghcr.io/moamenfouad/deep-fake-ai-powered-detector:latest \
  --ingress external --target-port 7860

az containerapp update \
  --name deepshield --resource-group deepshield-rg \
  --cpu 2 --memory 4Gi --min-replicas 0 --max-replicas 1
```

The first command prints the public address
(`https://deepshield.<random>.westeurope.azurecontainerapps.io`). With
`--min-replicas 0` the app sleeps when idle and the first visit after a pause takes
about a minute while the container starts and the models load; the page shows a
"warming up" message and retries by itself. Use `--min-replicas 1` on presentation day
to keep it warm, and set it back to 0 afterwards so the credit lasts.

To update the hosted app after a new push:

```bash
az containerapp update --name deepshield --resource-group deepshield-rg \
  --image ghcr.io/moamenfouad/deep-fake-ai-powered-detector:latest
```

To stop paying anything at all: `az group delete --name deepshield-rg`.

## 4. Any other container host

The same image runs anywhere that offers 2 GB of RAM:

```bash
docker run -p 7860:7860 ghcr.io/moamenfouad/deep-fake-ai-powered-detector:latest
# or build it yourself
docker compose up --build
```

The container listens on `$PORT` when the host sets it (Cloud Run, Render, Azure) and
on 7860 otherwise. On Render choose **New, Web Service, Existing image**, paste the
image address and pick the 2 GB instance.

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
