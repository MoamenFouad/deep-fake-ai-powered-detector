"""Provenance: labels that a file carries about its own origin."""

import io

import numpy as np
from PIL import Image, PngImagePlugin

from app.services.provenance import inspect


def _jpeg(**kw) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 90, 60)).save(buf, "JPEG", **kw)
    return buf.getvalue()


def test_plain_photo_has_no_label():
    p = inspect(_jpeg())
    assert not p.ai and p.source is None and p.camera is None


def test_camera_exif_is_reported_but_is_not_a_verdict():
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Apple", "iPhone 15 Pro"
    p = inspect(_jpeg(exif=exif))
    assert not p.ai and p.camera == "Apple iPhone 15 Pro"


def test_iptc_digital_source_type_in_xmp_marks_ai():
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
           b'Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
           b'</rdf:RDF></x:xmpmeta>')
    p = inspect(_jpeg(xmp=xmp))
    assert p.ai and p.kind == "generated" and p.source == "xmp"


def test_ai_edit_is_reported_as_edited():
    xmp = b'<x:xmpmeta>digitalsourcetype/compositeWithTrainedAlgorithmicMedia</x:xmpmeta>'
    p = inspect(_jpeg(xmp=xmp))
    assert p.ai and p.kind == "edited"


def test_stable_diffusion_png_parameters_mark_ai():
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "portrait photo, Steps: 30, Sampler: Euler a")
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "PNG", pnginfo=info)
    p = inspect(buf.getvalue())
    assert p.ai and p.source == "png"


def test_ai_software_in_exif_marks_ai():
    exif = Image.Exif()
    exif[0x0131] = "Midjourney"
    p = inspect(_jpeg(exif=exif))
    assert p.ai and p.source == "exif"


def test_garbage_never_raises():
    assert not inspect(b"not an image at all").ai
    assert not inspect(np.random.default_rng(0).bytes(5000)).ai


def test_analyzer_trusts_an_ai_label(client):
    xmp = b'<x:xmpmeta>digitalsourcetype/trainedAlgorithmicMedia</x:xmpmeta>'
    r = client.post("/api/analyze", files={"file": ("x.jpg", _jpeg(xmp=xmp), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "fake" and "provenance_ai" in body["notes"]
    assert body["provenance"]["ai"] and body["provenance"]["source"] == "xmp"


def test_c2pa_content_credentials_mark_ai():
    # Signed with a throwaway test certificate; the manifest records an AI "created" action.
    from pathlib import Path

    data = (Path(__file__).parent / "fixtures" / "c2pa_ai_test_signer.jpg").read_bytes()
    p = inspect(data)
    assert p.ai and p.source == "c2pa" and p.kind == "generated"
    # The signer named in the certificate is reported, not the free-text generator name.
    assert p.tool == "DeepShield Test"
