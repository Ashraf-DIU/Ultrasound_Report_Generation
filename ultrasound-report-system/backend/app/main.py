"""FastAPI inference service for the ultrasound report generator.

Endpoints
    GET  /            short JSON description
    GET  /health      {"status": "loading" | "ready" | "error", ...}
    GET  /api/info    model and corpus details shown on the website
    POST /api/report  multipart form, field "file": one ultrasound image
                      (a single frame or a whole printed sheet of frames)

Run locally:
    cd backend
    BUNDLE_DIR=/path/to/web_bundle uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import base64
import gc
import logging
import os
import threading
import time
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import preprocess as pp
from .bundle import load_bundle
from .encoder import ImageEncoder, LinearProbe
from .rag import ConsensusReporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "12"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff",
                 "image/webp", "application/octet-stream"}

DISCLAIMER = (
    "Research prototype for a final-year project. This draft is generated "
    "automatically by matching the image to similar past cases. It is NOT a "
    "diagnosis and must not be used for patient care. A qualified "
    "radiologist or sonologist must review every report.")

app = FastAPI(title="Ultrasound Report Generator API", version="1.0.0")

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Lets every Vercel preview URL of your project through when set, e.g.
    # ALLOWED_ORIGIN_REGEX=https://ultrasound-report.*\.vercel\.app
    allow_origin_regex=os.environ.get("ALLOWED_ORIGIN_REGEX") or None,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

STATE = {"status": "loading", "error": None, "loaded_at": None}
MODEL = {}
_infer_lock = threading.Lock()    # one inference at a time: keeps memory flat


def _load():
    try:
        t0 = time.time()
        b = load_bundle()
        cfg = b.config
        pp.CFG.update({k: v for k, v in cfg.get("preprocess", {}).items() if k in pp.CFG})
        MODEL["bundle"] = b
        MODEL["encoder"] = ImageEncoder(
            b.onnx_path,
            mean=cfg.get("image_mean", (0.48145466, 0.4578275, 0.40821073)),
            std=cfg.get("image_std", (0.26862954, 0.26130258, 0.27577711)),
            image_size=int(cfg.get("image_size", 224)),
            threads=int(os.environ.get("ORT_THREADS", "1")))
        MODEL["rag"] = ConsensusReporter(
            b.studies, b.embeddings, b.ontology,
            k=int(os.environ.get("RAG_K", cfg.get("k", 5))),
            tau=float(os.environ.get("RAG_TAU", cfg.get("tau", 0.4))))
        MODEL["probe"] = LinearProbe(b.probe) if b.probe else None
        gc.collect()
        STATE.update(status="ready", loaded_at=time.time())
        log.info("Model ready in %.1fs", time.time() - t0)
    except Exception as e:                         # noqa: BLE001
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")
        log.error("Model failed to load:\n%s", traceback.format_exc())


@app.on_event("startup")
def _startup():
    # Load in the background so /health answers immediately while the bundle
    # downloads (Render marks the deploy live once /health responds).
    threading.Thread(target=_load, daemon=True).start()


def _thumb(frame: np.ndarray, size: int = 160) -> str:
    small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode() if ok else ""


@app.get("/")
def root():
    return {"service": "Ultrasound Report Generator API", "status": STATE["status"],
            "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": STATE["status"], "error": STATE["error"]}


@app.get("/api/info")
def info():
    if STATE["status"] != "ready":
        return {"status": STATE["status"], "error": STATE["error"]}
    b = MODEL["bundle"]
    cfg = b.config
    return {
        "status": "ready",
        "model_name": cfg.get("model_name"),
        "weights": cfg.get("weights", "fp32"),
        "corpus_size": len(b.studies),
        "k": MODEL["rag"].k,
        "tau": MODEL["rag"].tau,
        "probe_targets": MODEL["probe"].targets if MODEL["probe"] else [],
        "bundle_version": cfg.get("bundle_version"),
        "created": cfg.get("created"),
        "embedding_check": cfg.get("embedding_check"),
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/report")
async def report(file: UploadFile = File(...)):
    if STATE["status"] == "loading":
        raise HTTPException(503, "The model is still loading. Try again in a minute.")
    if STATE["status"] == "error":
        raise HTTPException(500, f"The model failed to load: {STATE['error']}")
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(415, "Please upload an image file (JPG or PNG).")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_MB:.0f} MB.")

    try:
        bgr = pp.decode_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    del data

    t0 = time.time()
    with _infer_lock:
        frames, boxes = pp.sheet_to_frames(bgr, max_frames=MAX_FRAMES)
        frames_in = [pp.jpeg_roundtrip(f) for f in frames]
        z, _ = MODEL["encoder"].embed_study(frames_in)
        out = MODEL["rag"].report(z)
        cls = MODEL["probe"].predict(z) if MODEL["probe"] else []
    elapsed = time.time() - t0

    cfg = MODEL["bundle"].config
    warnings = []
    ood_thr = cfg.get("ood_threshold")
    ood = ood_thr is not None and out["top_similarity"] < float(ood_thr)
    if ood:
        warnings.append(
            "This image looks unlike every study in the reference corpus. It "
            "may not be an ultrasound image, or it may be a type of scan the "
            "system was not built for. Treat the draft below as unreliable.")
    if len(frames) == 1 and boxes[0] == (0, 0, bgr.shape[1], bgr.shape[0]):
        warnings.append("No separate ultrasound frames were detected; the whole image was used.")

    findings_text = " ".join(f"{f['organ']}: {f['text']}" for f in out["findings"])
    return {
        "report": {
            "exam_type": out["exam_type"],
            "findings": out["findings"],
            "findings_text": findings_text,
            "impression": out["impression"],
        },
        "consensus": out["consensus"],
        "classifier": cls,
        "similar_cases": out["similar_cases"],
        "frames": [{"index": i, "box": list(map(int, bx)), "thumbnail": _thumb(f)}
                   for i, (f, bx) in enumerate(zip(frames, boxes))],
        "quality": {
            "n_frames": len(frames),
            "top_similarity": round(out["top_similarity"], 4),
            "ood_threshold": ood_thr,
            "out_of_distribution": bool(ood),
            "warnings": warnings,
        },
        "method": {
            "k": MODEL["rag"].k,
            "tau": MODEL["rag"].tau,
            "model_name": cfg.get("model_name"),
            "corpus_size": len(MODEL["bundle"].studies),
            "seconds": round(elapsed, 2),
        },
        "disclaimer": DISCLAIMER,
    }
