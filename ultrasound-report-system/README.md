# Ultrasound Report Generator (US-RAG)

A web system for the paper **"Retrieval-Augmented Consensus for Low-Resource
Ultrasound Reporting"** (Md. Ashraf Bin Alam et al., Daffodil International
University). Upload an ultrasound image and get a draft report built from
the most similar past cases: BiomedCLIP image retrieval plus
similarity-weighted consensus fusion (K = 5, tau = 0.4).

> **Research prototype. Not a medical device and not for clinical use.**
> Every draft must be reviewed by a qualified radiologist.

**To deploy it, follow [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).**

## Folders

```
backend/                 FastAPI inference API (deploy on Render)
  app/preprocess.py      frame decomposition + CLAHE (from notebook 02)
  app/encoder.py         BiomedCLIP image tower on ONNX Runtime
  app/rag.py             retrieval + consensus fusion (from notebook 05)
  app/bundle.py          downloads and loads web_bundle.zip
  app/main.py            HTTP endpoints: /health, /api/info, /api/report
  tests/                 end-to-end API tests
  scripts/               developer test helpers (fake data, random model)
frontend/                Next.js website (deploy on Vercel)
kaggle/
  export_web_bundle.py   run on Kaggle to build web_bundle.zip
render.yaml              Render Blueprint for the API
DEPLOYMENT_GUIDE.md      step-by-step deployment
```

## API in one example

```bash
curl -F file=@sheet.jpg https://YOUR-API.onrender.com/api/report
```

returns `report` (exam type, findings, impression), `similar_cases` (top 5
with similarity and the radiologist's impression), `consensus` (vote per
finding), `classifier` (linear-probe probabilities), `frames` (thumbnails of
the detected frames), `quality` (warnings, out-of-distribution flag) and the
`disclaimer`.

## Testing without the real model

```bash
cd backend
pip install -r requirements.txt pytest httpx pandas open_clip_torch onnx onnxscript scikit-learn
python scripts/make_fake_kaggle_inputs.py /kaggle/input       # synthetic data
python scripts/run_export_with_random_model.py ../kaggle/export_web_bundle.py
BUNDLE_DIR=/kaggle/working/web_bundle TEST_IMAGE=/kaggle/test_sheet.jpg pytest -q
```

The fake bundle uses random weights and invented reports. It only checks
that the code runs. Never deploy it.
