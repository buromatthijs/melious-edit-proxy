"""
Melious edit proxy for Open WebUI.

Open WebUI sends image edits to the OpenAI-style endpoint POST /v1/images/edits
(multipart form, images in field `image` or `image[]`). Melious only offers
POST /v1/images/generations, where edits are done with `input_images`.

This proxy translates /images/edits into a Melious /images/generations call.
All other /v1/* requests are passed through unchanged, so the proxy can also be
used as the base URL for normal image generation.

The Authorization header from Open WebUI (your Melious API key) is forwarded as-is;
the proxy stores no secrets.
"""

import base64
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

MELIOUS_BASE_URL = os.getenv("MELIOUS_BASE_URL", "https://api.melious.ai/v1").rstrip("/")
TIMEOUT = float(os.getenv("TIMEOUT_SECONDS", "180"))
# "raw" = plain base64, "dataurl" = data:image/png;base64,...
INPUT_IMAGE_FORMAT = os.getenv("INPUT_IMAGE_FORMAT", "dataurl")
# Optional: "png", "jpeg" or "webp". Empty = provider default (JPEG for FLUX).
OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "")
DEFAULT_EDIT_MODEL = os.getenv("DEFAULT_EDIT_MODEL", "flux-2-klein-9b")
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "20"))

# Sizes Melious accepts in the `size` field; anything else is sent as width/height.
MELIOUS_SIZES = {"256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("melious-edit-proxy")

app = FastAPI(title="Melious edit proxy")
client = httpx.AsyncClient(timeout=TIMEOUT)


def _auth_headers(request: Request) -> dict:
    auth = request.headers.get("authorization")
    if not auth:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    return {"Authorization": auth}


def _size_fields(size: str | None) -> dict:
    if not size or size == "auto":
        return {}
    if size in MELIOUS_SIZES:
        return {"size": size}
    try:
        width, height = (int(v) for v in size.lower().split("x"))
        return {"width": width, "height": height}
    except ValueError:
        log.warning("Ignoring unknown size %r", size)
        return {}


def _encode(data: bytes, content_type: str | None) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    if INPUT_IMAGE_FORMAT == "dataurl":
        return f"data:{content_type or 'image/png'};base64,{b64}"
    return b64


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/images/edits")
async def image_edits(request: Request):
    form = await request.form()

    prompt = form.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Field 'prompt' is required")

    uploads = form.getlist("image") + form.getlist("image[]")
    if not uploads:
        raise HTTPException(status_code=400, detail="No image found in 'image' or 'image[]'")

    input_images = []
    for upload in uploads:
        if isinstance(upload, str):
            # Already base64 or a data URL
            input_images.append(upload.split(",", 1)[1] if upload.startswith("data:") and INPUT_IMAGE_FORMAT == "raw" else upload)
            continue
        data = await upload.read()
        if len(data) > MAX_IMAGE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Image larger than {MAX_IMAGE_MB} MB")
        input_images.append(_encode(data, upload.content_type))

    if form.get("mask") is not None:
        log.info("Mask received but not supported by Melious; ignoring it")

    payload = {
        "model": form.get("model") or DEFAULT_EDIT_MODEL,
        "prompt": prompt,
        "input_images": input_images,
        "response_format": "b64_json",
        **_size_fields(form.get("size")),
    }
    if form.get("n"):
        payload["n"] = int(form.get("n"))
    if OUTPUT_FORMAT:
        payload["output_format"] = OUTPUT_FORMAT

    log.info("Edit: model=%s images=%d", payload["model"], len(input_images))

    try:
        r = await client.post(
            f"{MELIOUS_BASE_URL}/images/generations",
            json=payload,
            headers=_auth_headers(request),
        )
    except httpx.HTTPError as e:
        log.error("Melious request failed: %s", e)
        raise HTTPException(status_code=502, detail="Could not reach Melious")

    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(path: str, request: Request):
    """Forward everything else (e.g. /images/generations, /models) unchanged."""
    headers = _auth_headers(request)
    if ct := request.headers.get("content-type"):
        headers["Content-Type"] = ct
    try:
        r = await client.request(
            request.method,
            f"{MELIOUS_BASE_URL}/{path}",
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )
    except httpx.HTTPError as e:
        log.error("Passthrough failed: %s", e)
        return JSONResponse({"error": {"message": "Could not reach Melious"}}, status_code=502)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))
