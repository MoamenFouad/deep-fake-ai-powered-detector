"""Provenance: what the file itself says about where it came from.

Many AI image tools now label their output. ChatGPT and Google's Gemini (Nano Banana)
sign images with C2PA Content Credentials, Google and Adobe also write the IPTC
"digital source type" into the XMP metadata, and local Stable Diffusion front ends
store the generation settings in PNG text chunks. When such a label survives, it is far
more reliable than any pixel detector. Labels are easy to remove (screenshots and
messaging apps strip them), so a missing label proves nothing.
"""

from __future__ import annotations

import io
import json
import logging
import re
from dataclasses import dataclass, field

from PIL import Image

log = logging.getLogger(__name__)

# IPTC digital source types for AI output.
# https://cv.iptc.org/newscodes/digitalsourcetype/
AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia": "generated",
    "compositeWithTrainedAlgorithmicMedia": "edited",
    "algorithmicMedia": "generated",
}
SOURCE_TYPE_RE = re.compile(
    rb"digitalsourcetype/(trainedAlgorithmicMedia|compositeWithTrainedAlgorithmicMedia|algorithmicMedia)\b")

# Generator names written into EXIF Software or PNG text by common tools.
AI_SOFTWARE = re.compile(
    r"midjourney|dall[\s·-]?e|stable[\s-]?diffusion|novelai|firefly|imagen|gemini|chatgpt|openai|"
    r"leonardo|ideogram|flux|comfyui|automatic1111|invokeai|fooocus|krea|runway|kling|seedream|recraft",
    re.I,
)
SD_PNG_KEYS = ("parameters", "prompt", "workflow", "sd-metadata", "invokeai_metadata", "Dream")

# Well-known names inside C2PA claim generators, for a readable label.
KNOWN_ISSUERS = [
    ("openai", "OpenAI (ChatGPT)"),
    ("chatgpt", "OpenAI (ChatGPT)"),
    ("google", "Google"),
    ("gemini", "Google"),
    ("adobe", "Adobe"),
    ("firefly", "Adobe Firefly"),
    ("microsoft", "Microsoft"),
    ("bing", "Microsoft"),
    ("meta", "Meta"),
    ("samsung", "Samsung"),
    ("truepic", "Truepic"),
]


@dataclass
class Provenance:
    """What the file's own metadata says. `ai` is True only when a label says AI made it."""

    ai: bool = False
    kind: str | None = None          # "generated" or "edited"
    source: str | None = None        # "c2pa", "xmp", "exif" or "png"
    tool: str | None = None          # who made or signed it, when known
    signed: bool = False             # True for a C2PA manifest whose signature checks out
    camera: str | None = None        # camera make and model from EXIF, if any
    details: list[str] = field(default_factory=list)


def _issuer(text: str) -> str | None:
    low = text.lower()
    for key, name in KNOWN_ISSUERS:
        if key in low:
            return name
    return None


def _c2pa(data: bytes, mime: str) -> Provenance | None:
    try:
        import c2pa  # optional dependency
    except ImportError:
        return None
    try:
        reader = c2pa.Reader(mime, io.BytesIO(data))
        store = json.loads(reader.json())
    except Exception:  # no manifest, or a format c2pa cannot read
        return None
    manifests = store.get("manifests") or {}
    active = manifests.get(store.get("active_manifest") or "", {})
    if not active:
        return None
    state = str(store.get("validation_state") or "")
    failures = [s for s in store.get("validation_status") or [] if "untrusted" not in str(s.get("code", ""))]
    signed = state.lower() in ("valid", "trusted") or not failures

    generator = active.get("claim_generator") or ""
    info = active.get("claim_generator_info") or []
    if isinstance(info, list):
        generator += " " + " ".join(str(i.get("name", "")) for i in info if isinstance(i, dict))
    issuer = (active.get("signature_info") or {}).get("issuer") or ""
    # The signer's certificate says who vouches for the file; the generator name is free text.
    tool = _issuer(issuer) or issuer or _issuer(generator) or (generator.strip().split("/")[0] or None)

    kind = None
    # Look through every manifest in the chain: an edit of an AI image keeps the AI ingredient.
    for manifest in manifests.values():
        for assertion in manifest.get("assertions") or []:
            if not str(assertion.get("label", "")).startswith("c2pa.actions"):
                continue
            for action in (assertion.get("data") or {}).get("actions") or []:
                dst = str(action.get("digitalSourceType", ""))
                for key, value in AI_SOURCE_TYPES.items():
                    if dst.endswith("/" + key):
                        kind = kind or value
    return Provenance(ai=kind is not None, kind=kind, source="c2pa", tool=tool, signed=signed,
                      details=[f"Content Credentials signed by {tool or 'an unknown issuer'}"])


def _mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:12] in (b"ftypavif", b"ftypheic", b"ftypmif1"):
        return "image/avif" if data[8:12] == b"avif" else "image/heic"
    return None


def inspect(data: bytes) -> Provenance:
    """Read Content Credentials, XMP, EXIF and PNG text. Never raises."""
    try:
        return _inspect(data)
    except Exception:  # metadata is a bonus; a broken tag must never break the check
        log.exception("Provenance check failed")
        return Provenance()


def _inspect(data: bytes) -> Provenance:
    mime = _mime(data)
    result = _c2pa(data, mime) if mime else None
    if result is not None and result.ai:
        return result
    found = result or Provenance()

    # IPTC digital source type in XMP (Google, Adobe and others write it even without C2PA).
    m = SOURCE_TYPE_RE.search(data[:4_000_000])
    if m:
        found.ai, found.kind, found.source = True, AI_SOURCE_TYPES[m.group(1).decode()], found.source or "xmp"
        found.details.append("The image metadata marks it as made with generative AI")

    try:
        img = Image.open(io.BytesIO(data))
        exif = img.getexif()
        make, model, software = (str(exif.get(t, "")).strip() for t in (0x010F, 0x0110, 0x0131))
        if make or model:
            found.camera = " ".join(p for p in (make, model if not model.startswith(make) else "") if p) or model
        texts = dict(getattr(img, "text", {}) or {})
    except Exception:
        software, texts = "", {}

    if not found.ai and software and AI_SOFTWARE.search(software):
        found.ai, found.kind, found.source, found.tool = True, "generated", "exif", software[:60]
        found.details.append(f"The EXIF software field names {software[:60]}")
    if not found.ai:
        keys = [k for k in texts if k in SD_PNG_KEYS]
        blob = " ".join(str(v)[:2000] for v in texts.values())
        if keys or AI_SOFTWARE.search(blob):
            m2 = AI_SOFTWARE.search(blob)
            found.ai, found.kind, found.source = True, "generated", "png"
            found.tool = m2.group(0) if m2 else "Stable Diffusion"
            found.details.append("The PNG stores image-generation settings")
    return found
