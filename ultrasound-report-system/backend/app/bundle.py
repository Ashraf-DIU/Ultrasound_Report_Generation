"""Finds, downloads and loads the model bundle made by kaggle/export_web_bundle.py.

The bundle is one zip (or an unzipped folder) holding:

    visual.onnx            BiomedCLIP ViT-B/16 image tower (ONNX, fp32 or int8)
    study_embeddings.npy   (N, 512) L2-normalised study embeddings of the corpus,
                           computed with the SAME visual.onnx file
    studies.json           one record per study: id, impression, findings,
                           exam type, ontology labels
    ontology.json          the 34-class rule ontology from notebook 00
    probe.json             linear-probe weights + thresholds (optional)
    config.json            K, tau, image mean/std, preprocessing values, etc.

Where it comes from, in order of preference:
    BUNDLE_DIR   a local folder (for running on your own computer)
    BUNDLE_URL   a direct download link to the zip (Hugging Face or GitHub
                 Release). Add HF_TOKEN if the Hugging Face repo is private.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("bundle")

REQUIRED = ["visual.onnx", "study_embeddings.npy", "studies.json",
            "ontology.json", "config.json"]


@dataclass
class Bundle:
    root: Path
    config: dict
    ontology: dict
    studies: list
    embeddings: np.ndarray
    probe: dict | None = None
    onnx_path: Path = field(default=None)


def _find_root(folder: Path) -> Path:
    """The zip may contain the files at the top level or inside one folder."""
    if (folder / "config.json").exists():
        return folder
    for p in folder.rglob("config.json"):
        if (p.parent / "visual.onnx").exists():
            return p.parent
    raise FileNotFoundError(f"No bundle (config.json + visual.onnx) found under {folder}")


def _download(url: str, dest: Path) -> None:
    headers = {"User-Agent": "ultrasound-report-api/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    log.info("Downloading bundle from %s", url.split("?")[0])
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    log.info("Downloaded %.1f MB", dest.stat().st_size / 1e6)


def locate_bundle() -> Path:
    local = os.environ.get("BUNDLE_DIR")
    if local:
        p = Path(local)
        if p.is_file() and p.suffix == ".zip":
            return _unzip(p)
        return _find_root(p)

    url = os.environ.get("BUNDLE_URL")
    if not url:
        raise RuntimeError(
            "No model bundle configured. Set BUNDLE_URL (a direct link to "
            "web_bundle.zip) or BUNDLE_DIR (a local folder). See DEPLOYMENT_GUIDE.md.")

    cache = Path(os.environ.get("BUNDLE_CACHE", Path(tempfile.gettempdir()) / "us_bundle"))
    marker = cache / ".source"
    if marker.exists() and marker.read_text() == url:
        try:
            return _find_root(cache)
        except FileNotFoundError:
            pass
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    zpath = cache / "bundle.zip"
    _download(url, zpath)
    root = _unzip(zpath, cache)
    zpath.unlink(missing_ok=True)
    marker.write_text(url)
    return root


def _unzip(zpath: Path, into: Path | None = None) -> Path:
    into = into or zpath.with_suffix("")
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.namelist():           # refuse paths that escape the folder
            target = (into / m).resolve()
            if not str(target).startswith(str(into.resolve())):
                raise ValueError(f"Unsafe path in bundle zip: {m}")
        zf.extractall(into)
    return _find_root(into)


def load_bundle() -> Bundle:
    root = locate_bundle()
    missing = [f for f in REQUIRED if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle at {root} is missing {missing}")

    config = json.loads((root / "config.json").read_text())
    ontology = json.loads((root / "ontology.json").read_text())
    studies = json.loads((root / "studies.json").read_text())
    Z = np.load(root / "study_embeddings.npy").astype(np.float32)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    if len(Z) != len(studies):
        raise ValueError(f"{len(Z)} embeddings but {len(studies)} studies in bundle")
    probe = None
    if (root / "probe.json").exists():
        probe = json.loads((root / "probe.json").read_text())
    log.info("Bundle loaded: %d studies, dim %d, model %s",
             len(studies), Z.shape[1], config.get("model_name"))
    return Bundle(root=root, config=config, ontology=ontology, studies=studies,
                  embeddings=Z, probe=probe, onnx_path=root / "visual.onnx")
