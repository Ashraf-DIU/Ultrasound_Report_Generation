"""BiomedCLIP image tower running on ONNX Runtime (CPU).

ONNX Runtime is used instead of PyTorch so the API fits in the 512 MB of
memory on Render's free plan. The export script checks that the ONNX model
gives the same embeddings as the PyTorch model used in notebook 04.
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

# BiomedCLIP / OpenAI CLIP normalisation. The bundle's config.json overrides
# these with the exact values read from open_clip's preprocess transform.
DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageEncoder:
    def __init__(self, onnx_path, mean=DEFAULT_MEAN, std=DEFAULT_STD,
                 image_size=224, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Arena off: lower peak memory on small instances, small speed cost.
        so.enable_cpu_mem_arena = False
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.mean = np.array(mean, np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, np.float32).reshape(1, 3, 1, 1)
        self.size = image_size

    def _to_tensor(self, frames_bgr: list) -> np.ndarray:
        """Same as open_clip's BiomedCLIP preprocess on a 224x224 frame:
        resize (no-op) -> centre crop (no-op) -> RGB -> [0,1] -> normalise."""
        out = []
        for f in frames_bgr:
            if f.shape[:2] != (self.size, self.size):
                f = cv2.resize(f, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            out.append(rgb.transpose(2, 0, 1))
        x = np.stack(out)
        return ((x - self.mean) / self.std).astype(np.float32)

    def embed_frames(self, frames_bgr: list) -> np.ndarray:
        """(N, D) L2-normalised frame embeddings (notebook 04, embed_images)."""
        feats = []
        for f in frames_bgr:                       # one at a time: lowest memory
            x = self._to_tensor([f])
            y = self.sess.run(None, {self.input_name: x})[0]
            feats.append(y[0])
        Z = np.stack(feats).astype(np.float32)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

    def embed_study(self, frames_bgr: list) -> tuple[np.ndarray, np.ndarray]:
        """Study embedding = renormalised mean of L2-normalised frame
        embeddings (manuscript Eq. 6)."""
        Zf = self.embed_frames(frames_bgr)
        v = Zf.mean(0)
        return v / (np.linalg.norm(v) + 1e-8), Zf


class LinearProbe:
    """Logistic-regression probe on frozen BiomedCLIP study embeddings
    ("BiomedCLIP probe (image)" in Table 4). Shown as supporting evidence
    only: its macro-F1 is about 0.22, far below the report generator's use
    of retrieved cases."""

    def __init__(self, probe: dict):
        self.targets = probe["targets"]
        self.W = np.array(probe["coef"], np.float32)
        self.b = np.array(probe["intercept"], np.float32)
        self.thr = np.array(probe.get("thresholds", [0.5] * len(self.targets)), np.float32)
        self.auc = probe.get("oof_auc", {})

    def predict(self, z: np.ndarray) -> list:
        p = 1.0 / (1.0 + np.exp(-(self.W @ z + self.b)))
        rows = [{"label": t, "probability": round(float(pi), 4),
                 "threshold": round(float(ti), 4), "positive": bool(pi >= ti),
                 "cv_auc": self.auc.get(t)}
                for t, pi, ti in zip(self.targets, p, self.thr)]
        return sorted(rows, key=lambda r: -r["probability"])
