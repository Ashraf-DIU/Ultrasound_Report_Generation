# =============================================================================
# export_web_bundle.py  —  builds web_bundle.zip for the website's API
#
# Run this on KAGGLE as a new notebook (one code cell, paste the whole file).
#
# Before running, in the Kaggle notebook editor:
#   1. Settings -> Internet: ON           (needed to download BiomedCLIP)
#   2. Settings -> Accelerator: None/CPU  (GPU is fine too, not needed)
#   3. Add Input -> Your Work / Notebooks, and attach the OUTPUT of:
#        02-preprocessing-splits   (master_split.csv, frames.csv, frames.zip,
#                                   ontology.json, label_tiers.json, ...)
#        04-biomedclip-multitask   (embeddings/study_image_embeddings.npy)
#      If your notebook 09 (export bundle) output already contains these
#      files you can attach it instead; the script searches every input.
#
# What it does
#   a. Loads BiomedCLIP exactly as notebook 04 does (open_clip, hf-hub).
#   b. Exports the image tower to ONNX and checks it against PyTorch.
#   c. Optionally makes an int8 copy (4x smaller, fits Render's free plan).
#   d. Re-embeds every corpus frame with the ONNX model the API will use, so
#      the retrieval index and the live queries come from the same weights,
#      and compares the result with notebook 04's study_image_embeddings.npy.
#   e. Fits the image linear probe (Table 4, "BiomedCLIP probe (image)").
#   f. Writes config.json, studies.json, ontology.json, probe.json and zips
#      everything to /kaggle/working/web_bundle.zip.
#   g. Optionally uploads the zip to a Hugging Face model repo (see bottom).
# =============================================================================
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def pipq(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)


pipq("open_clip_torch", "onnx", "onnxruntime", "onnxscript")

import cv2                                   # noqa: E402
import onnxruntime as ort                    # noqa: E402
import open_clip                             # noqa: E402
import torch                                 # noqa: E402

# ----------------------------------------------------------------------------
# Settings you may change
# ----------------------------------------------------------------------------
WEIGHTS = "int8"        # "int8" (~90 MB, for Render free 512 MB) or "fp32" (~345 MB)
K, TAU = 5, 0.4         # consensus fusion settings from the paper (Table 3)
OUT = Path("/kaggle/working/web_bundle")
ZIP_PATH = Path("/kaggle/working/web_bundle.zip")
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

# Optional: upload to Hugging Face. Add a Kaggle secret named HF_TOKEN
# (Add-ons -> Secrets) holding a token with WRITE access, and set the repo.
HF_UPLOAD_REPO = ""     # e.g. "your-hf-username/ultrasound-rag-bundle"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
torch.set_grad_enabled(False)


# ----------------------------------------------------------------------------
# 1. Find the notebook outputs (same resolver as notebook 04)
# ----------------------------------------------------------------------------
def _rank(p):
    nums = re.findall(r"[/\\](\d{2})[-_]", str(p))
    return max((int(n) for n in nums), default=-1)


def find_file(fname, required=True):
    hits = sorted(Path("/kaggle/input").rglob(fname), key=_rank, reverse=True)
    if hits:
        return hits[0]
    if required:
        raise FileNotFoundError(
            f"'{fname}' not found in any attached input. Attach notebook 02's and "
            "notebook 04's outputs (Add Input -> Notebooks).")
    return None


df = pd.read_csv(find_file("master_split.csv"))
frames = pd.read_csv(find_file("frames.csv"))
ontology = json.loads(find_file("ontology.json").read_text())
tiers = json.loads(find_file("label_tiers.json").read_text())
Ydf = pd.read_csv(find_file("labels_multihot.csv"))
pre_cfg_path = find_file("preprocess_config.json", required=False)
pre_cfg = json.loads(pre_cfg_path.read_text()) if pre_cfg_path else {}
nb4_emb_path = find_file("study_image_embeddings.npy", required=False)
print(f"{len(df)} studies, {len(frames)} frames, {len(ontology)} ontology classes")


# Re-anchor frame paths (notebook 04, cell 3)
def resolve_frames_root(frames_df):
    sample = Path(str(frames_df["frame_path"].iloc[0]))
    if sample.exists():
        return None
    rel = Path(sample.parent.name) / sample.name
    candidates = [Path("/kaggle/working/frames")]
    for depth in ("*/frames", "*/*/frames", "*/*/*/frames", "*/frames/*"):
        candidates += sorted(Path("/kaggle/input").glob(depth))
    for c in candidates:
        if (c / rel).exists():
            return c
    hits = sorted(Path("/kaggle/input").rglob("frames.zip"))
    if hits:
        dest = Path("/kaggle/working/frames")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(hits[0]) as zf:
            zf.extractall(dest)
        if (dest / rel).exists():
            return dest
    raise FileNotFoundError("Frames not found. Attach notebook 02's output (frames/ or frames.zip).")


_root = resolve_frames_root(frames)
if _root is not None:
    frames["frame_path"] = [str(_root / Path(p).parent.name / Path(p).name)
                            for p in frames["frame_path"]]
assert all(os.path.exists(p) for p in frames["frame_path"]), "some frame files are missing"

# ----------------------------------------------------------------------------
# 2. Load BiomedCLIP (notebook 04, cell 6)
# ----------------------------------------------------------------------------
clip_model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP)
clip_model = clip_model.eval().float().cpu()

# Read the exact normalisation from the preprocess transform
mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
for t in getattr(preprocess, "transforms", []):
    if t.__class__.__name__ == "Normalize":
        mean, std = tuple(float(x) for x in t.mean), tuple(float(x) for x in t.std)
print("normalisation mean/std:", mean, std)


class VisualTower(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m.encode_image(pixel_values)      # un-normalised, (N, 512)


tower = VisualTower(clip_model).eval()

# ----------------------------------------------------------------------------
# 3. Export to ONNX and verify
# ----------------------------------------------------------------------------
fp32_path = OUT / "visual_fp32.onnx"
dummy = torch.randn(1, 3, 224, 224)
export_kwargs = dict(input_names=["pixel_values"], output_names=["embeddings"],
                     dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
                     opset_version=17, do_constant_folding=True)
try:
    torch.onnx.export(tower, dummy, str(fp32_path), dynamo=False, **export_kwargs)
except TypeError:                                     # older torch: no dynamo arg
    torch.onnx.export(tower, dummy, str(fp32_path), **export_kwargs)
print(f"ONNX fp32: {fp32_path.stat().st_size/1e6:.0f} MB")


def to_tensor(bgr):
    """Identical to backend/app/encoder.py::_to_tensor."""
    if bgr.shape[:2] != (224, 224):
        bgr = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]
    m = np.array(mean, np.float32).reshape(1, 3, 1, 1)
    s = np.array(std, np.float32).reshape(1, 3, 1, 1)
    return ((x - m) / s).astype(np.float32)


def cos(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


sample_paths = frames["frame_path"].tolist()[:8]
xs = np.concatenate([to_tensor(cv2.imread(p)) for p in sample_paths])
ref = tower(torch.from_numpy(xs)).numpy()
sess32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
got32 = sess32.run(None, {"pixel_values": xs})[0]
c32 = cos(ref, got32)
print(f"PyTorch vs ONNX fp32 cosine: min {c32.min():.6f}")
assert c32.min() > 0.9999, "ONNX export does not match PyTorch"

if WEIGHTS == "int8":
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model_path = OUT / "visual.onnx"
    # MatMul/Gemm only: the transformer weights are ~99% of the size, and
    # leaving the patch-embedding Conv in fp32 avoids the ConvInteger op that
    # some onnxruntime builds do not implement.
    quantize_dynamic(str(fp32_path), str(model_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    fp32_path.unlink()
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    c8 = cos(ref, sess.run(None, {"pixel_values": xs})[0])
    print(f"PyTorch vs ONNX int8 cosine: min {c8.min():.4f}, mean {c8.mean():.4f}")
else:
    model_path = OUT / "visual.onnx"
    fp32_path.rename(model_path)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
print(f"visual.onnx: {model_path.stat().st_size/1e6:.0f} MB ({WEIGHTS})")

# ----------------------------------------------------------------------------
# 4. Re-embed every frame with the bundled ONNX model -> study embeddings
#    (notebook 04, cell 8: renormalised mean of L2-normalised frames)
# ----------------------------------------------------------------------------
t0 = time.time()
Zframe = []
for i, p in enumerate(frames["frame_path"]):
    img = cv2.imread(p)
    assert img is not None, f"cannot read {p}"
    Zframe.append(sess.run(None, {"pixel_values": to_tensor(img)})[0][0])
    if i % 100 == 0:
        print(f"  embedded {i}/{len(frames)}", end="\r")
Zframe = np.stack(Zframe).astype(np.float32)
Zframe /= np.linalg.norm(Zframe, axis=1, keepdims=True) + 1e-8

idx_by_study = frames.groupby("study_id").indices
Z = np.zeros((len(df), Zframe.shape[1]), np.float32)
for i, sid in enumerate(df["study_id"].values):
    ix = idx_by_study.get(sid, [])
    assert len(ix), f"study {sid} has no frames"
    v = Zframe[list(ix)].mean(0)
    Z[i] = v / (np.linalg.norm(v) + 1e-8)
np.save(OUT / "study_embeddings.npy", Z)
print(f"\nStudy embeddings {Z.shape} in {time.time()-t0:.0f}s")

# Compare with notebook 04's embeddings (what the paper's numbers used)
emb_check = None
if nb4_emb_path is not None:
    Z4 = np.load(nb4_emb_path).astype(np.float32)
    Z4 /= np.linalg.norm(Z4, axis=1, keepdims=True) + 1e-8
    if Z4.shape == Z.shape:
        c = cos(Z, Z4)

        def topk(M, k=5):
            S = M @ M.T
            np.fill_diagonal(S, -np.inf)
            return np.argsort(-S, axis=1)[:, :k]

        overlap = np.mean([len(set(a) & set(b)) / 5 for a, b in zip(topk(Z), topk(Z4))])
        emb_check = {"mean_cosine_vs_nb04": round(float(c.mean()), 5),
                     "min_cosine_vs_nb04": round(float(c.min()), 5),
                     "top5_neighbour_overlap_vs_nb04": round(float(overlap), 4)}
        print("Agreement with notebook 04 embeddings:", emb_check)
        if c.mean() < 0.97:
            print("WARNING: embeddings differ noticeably from notebook 04. Check that "
                  "the attached frames are the same ones notebook 04 used.")

# ----------------------------------------------------------------------------
# 5. Study records + ontology
# ----------------------------------------------------------------------------
def extract_labels(text):
    t = str(text).lower()
    return sorted(lab for lab, spec in ontology.items()
                  if any(re.search(p, t) for p in spec.get("include", []))
                  and not any(re.search(p, t) for p in spec.get("exclude", [])))


def clean(v):
    return "" if pd.isna(v) else str(v)


studies = [{
    "study_id": clean(r["study_id"]),
    "impression": clean(r.get("impression")),
    "findings": clean(r.get("findings")),
    "exam_type": clean(r.get("exam_type")) or None,
    "labels": extract_labels(r.get("impression", "")),
} for _, r in df.iterrows()]
(OUT / "studies.json").write_text(json.dumps(studies, ensure_ascii=False))
(OUT / "ontology.json").write_text(json.dumps(ontology, indent=2))

# ----------------------------------------------------------------------------
# 6. Linear probe on image embeddings (notebook 04, cell 14) with thresholds
#    chosen on out-of-fold predictions (paper: thresholds fit on training folds)
# ----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score          # noqa: E402

TARGETS = [t for t in tiers["trainable"] + tiers["marginal"] if t in Ydf.columns]
Y = Ydf[TARGETS].values.astype(int)
folds = df["fold"].values
oof = np.zeros(Y.shape, np.float32)
for f in sorted(set(folds)):
    tr, te = folds != f, folds == f
    for j in range(len(TARGETS)):
        if Y[tr, j].sum() < 2:
            oof[te, j] = Y[tr, j].mean()
            continue
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        clf.fit(Z[tr], Y[tr, j])
        oof[te, j] = clf.predict_proba(Z[te])[:, 1]

thresholds, aucs = [], {}
for j, t in enumerate(TARGETS):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(Y[:, j], (oof[:, j] >= g).astype(int), zero_division=0) for g in grid]
    thresholds.append(float(grid[int(np.argmax(f1s))]))
    if 0 < Y[:, j].sum() < len(Y):
        aucs[t] = round(float(roc_auc_score(Y[:, j], oof[:, j])), 3)
print("probe out-of-fold AUC:", aucs)

coef, intercept = [], []
for j in range(len(TARGETS)):
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    clf.fit(Z, Y[:, j])                      # final model on the whole corpus
    coef.append(clf.coef_[0].tolist())
    intercept.append(float(clf.intercept_[0]))
(OUT / "probe.json").write_text(json.dumps({
    "targets": TARGETS, "coef": coef, "intercept": intercept,
    "thresholds": thresholds, "oof_auc": aucs}))

# ----------------------------------------------------------------------------
# 7. Out-of-distribution threshold: an upload whose best match is less
#    similar than almost any corpus study's best match gets a warning.
# ----------------------------------------------------------------------------
S = Z @ Z.T
np.fill_diagonal(S, -np.inf)
top1 = S.max(1)
ood_threshold = float(np.percentile(top1, 1) - 0.05)
print(f"leave-one-out top-1 similarity: min {top1.min():.3f}, median "
      f"{np.median(top1):.3f}; OOD warning below {ood_threshold:.3f}")

# ----------------------------------------------------------------------------
# 8. config.json + zip
# ----------------------------------------------------------------------------
preprocess_keys = ["img_size", "min_panel_area_frac", "min_panel_side", "close_kernel",
                   "banner_frac", "clahe_clip", "clahe_grid", "speckle_filter"]
config = {
    "bundle_version": 1,
    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model_name": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (image tower)",
    "weights": WEIGHTS,
    "image_size": 224,
    "image_mean": list(mean),
    "image_std": list(std),
    "k": K,
    "tau": TAU,
    "ood_threshold": ood_threshold,
    "corpus_size": len(df),
    "n_frames": int(len(frames)),
    "preprocess": {k: pre_cfg[k] for k in preprocess_keys if k in pre_cfg},
    "embedding_check": emb_check,
}
(OUT / "config.json").write_text(json.dumps(config, indent=2))

if ZIP_PATH.exists():
    ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(OUT.iterdir()):
        zf.write(p, arcname=p.name)
print("\nBundle contents:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:24s} {p.stat().st_size/1e6:8.2f} MB")
print(f"\nWrote {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1e6:.0f} MB)")
print("Download it from the notebook's Output panel (right side), or let the "
      "next cell upload it to Hugging Face.")

# ----------------------------------------------------------------------------
# 9. Optional: upload to Hugging Face Hub
# ----------------------------------------------------------------------------
if HF_UPLOAD_REPO:
    pipq("huggingface_hub")
    from huggingface_hub import HfApi
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(HF_UPLOAD_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(ZIP_PATH), path_in_repo="web_bundle.zip",
                    repo_id=HF_UPLOAD_REPO, repo_type="model")
    print("\nUploaded. Your BUNDLE_URL is:")
    print(f"  https://huggingface.co/{HF_UPLOAD_REPO}/resolve/main/web_bundle.zip")
