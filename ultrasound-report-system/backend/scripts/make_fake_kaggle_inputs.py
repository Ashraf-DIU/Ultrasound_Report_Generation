"""Developer test helper: creates a FAKE /kaggle/input tree (synthetic sheets,
reports and NB02/NB04-style artifacts) so the export script and the API can
be tested end to end without Kaggle. The images are drawn shapes, not real
ultrasound, and the reports are invented. Never deploy a bundle built from it.

    python scripts/make_fake_kaggle_inputs.py /tmp/fake_kaggle/input
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake_kaggle/input")
A02 = ROOT / "notebooks/ashrafbin/02-preprocessing-splits/artifacts"
F02 = ROOT / "notebooks/ashrafbin/02-preprocessing-splits/frames"
A04 = ROOT / "notebooks/ashrafbin/04-biomedclip-multitask/artifacts/embeddings"
for d in (A02, F02, A04):
    d.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
CASES = [
    ("Normal USG findings.", "Liver: Normal in size and echotexture. Gall bladder: Normal. Kidneys: Normal.", "Whole abdomen", ["Normal study"]),
    ("Fatty change in liver.", "Liver: Mildly enlarged with increased echogenicity. Gall bladder: Normal. Kidneys: Normal.", "Whole abdomen", ["Fatty liver"]),
    ("Cholelithiasis.", "Liver: Normal. Gall bladder: Contains echogenic calculus with shadow. Kidneys: Normal.", "Whole abdomen", ["Cholelithiasis"]),
    ("Features of cystitis.", "Urinary bladder: Thickened irregular wall. Uterus: Normal.", "Lower abdomen", ["Cystitis"]),
    ("Single alive intrauterine pregnancy of 10 weeks.", "Uterus: Gestational sac with fetal pole and cardiac activity.", "Obstetric profile", ["Normal pregnancy"]),
]
ontology = {
    "Normal study": {"group": "Normal", "include": [r"normal (?:usg |us )?(?:study|findings)"], "exclude": []},
    "Fatty liver": {"group": "Hepatobiliary", "include": [r"fatty change", r"fatty liver"], "exclude": []},
    "Cholelithiasis": {"group": "Hepatobiliary", "include": [r"cholelithias"], "exclude": []},
    "Cystitis": {"group": "Urinary", "include": [r"cystitis"], "exclude": []},
    "Normal pregnancy": {"group": "Obstetric", "include": [r"single alive", r"intrauterine pregnancy"], "exclude": []},
}
targets = [c[3][0] for c in CASES]


def fake_frame(kind, seed):
    r = np.random.default_rng(seed)
    img = np.zeros((300, 400, 3), np.uint8)
    cv2.putText(img, "MEDICAL 3.5MHz", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    pts = np.array([[200, 40], [60, 280], [340, 280]], np.int32)
    cv2.fillPoly(img, [pts], (90, 90, 90))
    noise = r.normal(0, 25, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise * (img > 0), 0, 255).astype(np.uint8)
    cv2.circle(img, (200, 120 + 25 * kind), 20 + 6 * kind, (30 + 40 * kind,) * 3, -1)
    return img


rows, frows, mh = [], [], []
for i in range(60):
    k = i % len(CASES)
    imp, find, exam, labs = CASES[k]
    sid = f"R{i+1}"
    n = 1 + i % 3
    (F02 / sid).mkdir(exist_ok=True)
    for p in range(n):
        fr = cv2.resize(fake_frame(k, i * 10 + p), (224, 168))
        canvas = np.zeros((224, 224, 3), np.uint8)
        canvas[28:196] = fr
        fp = F02 / sid / f"{sid}_p{p:02d}.jpg"
        cv2.imwrite(str(fp), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        frows.append({"study_id": sid, "frame_id": fp.stem,
                      "frame_path": f"/kaggle/working/frames/{sid}/{fp.name}",
                      "panel_index": p, "src_box": "0,0,1,1"})
    rows.append({"study_id": sid, "impression": imp, "findings": find, "exam_type": exam,
                 "fold": i % 5, "dup_group": i})
    mh.append({t: int(t in labs) for t in targets})

pd.DataFrame(rows).to_csv(A02 / "master_split.csv", index=False)
pd.DataFrame(frows).to_csv(A02 / "frames.csv", index=False)
pd.DataFrame(mh).to_csv(A02 / "labels_multihot.csv", index=False)
(A02 / "ontology.json").write_text(json.dumps(ontology))
(A02 / "label_tiers.json").write_text(json.dumps({"trainable": targets, "marginal": [], "rare": []}))
(A02 / "preprocess_config.json").write_text(json.dumps({"img_size": 224, "banner_frac": 0.11, "clahe_clip": 2.5}))
np.save(A04 / "study_image_embeddings.npy", rng.normal(size=(60, 512)).astype(np.float32))

# A multi-frame "sheet" to upload in tests
sheet = np.full((700, 900, 3), 8, np.uint8)
for j, (y, x) in enumerate([(30, 30), (30, 470), (370, 30), (370, 470)]):
    sheet[y:y + 300, x:x + 400] = fake_frame(j % 5, 999 + j)
cv2.imwrite(str(ROOT.parent / "test_sheet.jpg"), sheet)
print("fake inputs at", ROOT, "| test sheet at", ROOT.parent / "test_sheet.jpg")
