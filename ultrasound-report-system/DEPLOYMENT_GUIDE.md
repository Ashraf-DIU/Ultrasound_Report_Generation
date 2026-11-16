# Deployment guide: Ultrasound Report Generator

This guide takes you from the files in this folder to a live website where
anyone can upload an ultrasound image and get a draft report. No earlier
deployment experience is assumed. Plan for about 1 to 2 hours the first time.

## The big picture

The system has three parts, and each one lives in a different place:

| Part | What it is | Where it runs | Cost |
|---|---|---|---|
| **Model bundle** (`web_bundle.zip`, about 90 MB) | BiomedCLIP image encoder + the embeddings and reports of your 259 studies | Hugging Face (or a GitHub Release) | Free |
| **API** (`backend/`) | Python/FastAPI. Cuts the image into frames, runs BiomedCLIP, finds the 5 most similar studies, votes, writes the report | Render | Free |
| **Website** (`frontend/`) | Next.js. The upload page and the report view | Vercel | Free |

```
 Browser ──upload──> Vercel website ──(browser calls API directly)──> Render API
                                                                        │
                                              downloads once at start   │
                                                                        ▼
                                                   Hugging Face: web_bundle.zip
```

Why two hosts? Vercel is great for websites but its serverless functions are
too small and too short-lived for a 90 MB model. Render runs a normal Python
server that keeps the model in memory.

You will do the steps in this order:

1. Build `web_bundle.zip` on Kaggle
2. Upload the bundle to Hugging Face and get its download link
3. Put the code on GitHub
4. Deploy the API on Render
5. Deploy the website on Vercel
6. Connect the two (CORS) and test

---

## Step 0. Accounts you need

Create these (all free, you can sign in to Render and Vercel with GitHub):

- GitHub: https://github.com
- Hugging Face: https://huggingface.co/join
- Render: https://render.com
- Vercel: https://vercel.com
- Kaggle: you already have this

On your computer you also need **Git** (https://git-scm.com/downloads). Node.js
and Python are only needed if you want to run the project locally (optional,
see the end).

---

## Step 1. Build the model bundle on Kaggle

The API cannot use your Kaggle notebooks directly, so one extra notebook
packages everything the website needs into a single zip file.

1. Open Kaggle and click **Create -> New Notebook**.
2. In the right-hand panel open **Settings**:
   - **Internet**: turn **On** (it downloads BiomedCLIP from Hugging Face,
     exactly as notebook 04 does). You may need a phone-verified Kaggle
     account for this.
   - **Accelerator**: None is fine.
3. Click **Add Input** (right panel) -> **Your Work** -> **Notebooks**, and add
   the output of:
   - `02-preprocessing-splits`
   - `04-biomedclip-multitask`

   If your `09-export-bundle` output already contains `master_split.csv`,
   `frames.csv`, the frames, `ontology.json`, `label_tiers.json`,
   `labels_multihot.csv` and `study_image_embeddings.npy`, attaching just that
   one also works. The script searches every attached input.
4. Open `kaggle/export_web_bundle.py` from this project, copy **all** of it,
   and paste it into the notebook's first code cell.
5. Click **Run All**. It takes about 5 to 10 minutes.
6. Check the printed lines:
   - `PyTorch vs ONNX fp32 cosine: min 1.000000` (should be 0.9999 or more)
   - `PyTorch vs ONNX int8 cosine: ...` (about 0.98 or more is normal)
   - `Agreement with notebook 04 embeddings: {... 'mean_cosine_vs_nb04': ...}`.
     With int8 weights expect around 0.97 to 0.99. A value far below that
     means different frames were attached than the ones notebook 04 used.
   - `Wrote /kaggle/working/web_bundle.zip (about 90 MB)`
7. Download it: in the right panel open **Output**, find
   `web_bundle.zip`, click the three dots, then **Download**.
   (If it does not show, click **Save Version -> Save & Run All**, wait for
   it to finish, then download from the notebook's **Output** tab.)

**About int8 vs fp32.** The script makes an int8 (compressed) copy of the
encoder by default, so the API fits in the 512 MB memory of Render's free
plan. The study embeddings in the bundle are recomputed with that same int8
encoder, so the uploaded image and the reference studies are always compared
in the same space. If you later move to a paid plan (2 GB), change
`WEIGHTS = "int8"` to `WEIGHTS = "fp32"` at the top of the script and rebuild
for embeddings identical to the paper's.

---

## Step 2. Host the bundle on Hugging Face

1. Sign in to Hugging Face, click your profile picture -> **New Model**.
2. Name it, for example `ultrasound-rag-bundle`. Choose **Public** (simplest).
   The dataset is already public on Mendeley, so this exposes nothing new,
   but ask your supervisor if you are unsure. Click **Create model**.
3. Open the **Files and versions** tab -> **Add file** -> **Upload files**,
   drag in `web_bundle.zip`, and click **Commit changes to main**.
4. Your download link (this is your `BUNDLE_URL`) is:

   ```
   https://huggingface.co/YOUR-HF-USERNAME/ultrasound-rag-bundle/resolve/main/web_bundle.zip
   ```

   Paste it in a browser tab. A download should start. If it shows a web page
   instead, the link is wrong (it must say `/resolve/`, not `/blob/`).

**If you made the repository private**, create a token at
https://huggingface.co/settings/tokens (type "Read"), and later add it on
Render as an environment variable named `HF_TOKEN`.

**Shortcut:** you can make the Kaggle notebook upload the zip for you. Add a
Kaggle secret called `HF_TOKEN` (Add-ons -> Secrets) holding a **Write** token,
and set `HF_UPLOAD_REPO = "your-hf-username/ultrasound-rag-bundle"` at the top
of the script before running it.

**Alternative: GitHub Release.** In your GitHub repository (Step 3) click
**Releases -> Create a new release**, tag `v1.0`, attach `web_bundle.zip`
under "Attach binaries", and publish. Right-click the zip link and copy it:
`https://github.com/YOU/REPO/releases/download/v1.0/web_bundle.zip`. That is
your `BUNDLE_URL`. This only works if the repository is public.

Do **not** commit the zip itself into the repository. Git is not meant for
large files and GitHub rejects files over 100 MB.

---

## Step 3. Put the code on GitHub

1. On GitHub click **+ -> New repository**. Name it, for example,
   `ultrasound-report-system`. Public or private both work. Do not add a
   README (this project already has one). Click **Create repository**.
2. On your computer, open a terminal (Windows: "Git Bash"), go into this
   project folder (the one that contains `backend/`, `frontend/` and this
   guide), and run:

   ```bash
   git init
   git add .
   git commit -m "Ultrasound report generator: API and website"
   git branch -M main
   git remote add origin https://github.com/YOUR-GITHUB-USERNAME/ultrasound-report-system.git
   git push -u origin main
   ```

   If Git asks for a password, use a GitHub personal access token
   (GitHub -> Settings -> Developer settings -> Personal access tokens), or
   use GitHub Desktop (https://desktop.github.com) which signs in for you.
3. Refresh the repository page on GitHub. You should see `backend/`,
   `frontend/`, `kaggle/`, `render.yaml` and this guide.

---

## Step 4. Deploy the API on Render

1. Sign in to https://dashboard.render.com with GitHub.
2. Click **New + -> Blueprint**. Connect your GitHub account if asked and
   pick the `ultrasound-report-system` repository. Render reads
   `render.yaml` and proposes one web service called
   `ultrasound-report-api` on the Free plan.
3. Render asks for the values marked "sync: false":
   - `BUNDLE_URL`: the Hugging Face link from Step 2.
   - `ALLOWED_ORIGINS`: type `*` for now. You will tighten it in Step 6.
4. Click **Apply**. The first build takes about 3 to 5 minutes. Watch the
   **Logs** tab. A good start ends with lines like:

   ```
   Downloading bundle from https://huggingface.co/...
   Downloaded 88.4 MB
   Bundle loaded: 259 studies, dim 512, model BiomedCLIP-...
   Model ready in ...s
   ```
5. Copy the service address from the top of the page, for example
   `https://ultrasound-report-api.onrender.com`.
6. Test it: open `https://ultrasound-report-api.onrender.com/health` in your
   browser. You want `{"status":"ready","error":null}`. `"loading"` means
   wait 30 seconds and refresh. `/docs` gives you a page where you can try
   `POST /api/report` with an image directly.

**Manual setup instead of Blueprint** (if you prefer): New + -> Web Service ->
pick the repo, then set Root Directory `backend`, Runtime `Python 3`, Build
Command `pip install -r requirements.txt`, Start Command
`uvicorn app.main:app --host 0.0.0.0 --port $PORT`, Instance Type Free, and add
the environment variables from the table below.

### API environment variables (Render -> your service -> Environment)

| Name | Required | Example | What it does |
|---|---|---|---|
| `BUNDLE_URL` | yes | `https://huggingface.co/you/ultrasound-rag-bundle/resolve/main/web_bundle.zip` | Where the model bundle is downloaded from at start-up |
| `ALLOWED_ORIGINS` | yes | `https://ultrasound-report.vercel.app` | Websites allowed to call the API (comma-separated). `*` allows any |
| `PYTHON_VERSION` | yes | `3.11.11` | Python version Render uses (set by `render.yaml`) |
| `ALLOWED_ORIGIN_REGEX` | no | `https://ultrasound-report.*\.vercel\.app` | Also allow Vercel preview addresses |
| `HF_TOKEN` | no | `hf_...` | Only if the Hugging Face repo is private |
| `ORT_THREADS` | no | `1` | CPU threads for the model. Raise on paid plans |
| `RAG_K`, `RAG_TAU` | no | `5`, `0.4` | Override K and tau (the paper uses 5 and 0.4) |
| `MAX_UPLOAD_MB` | no | `10` | Largest image accepted |

### Things to know about Render's free plan

- **It sleeps.** After 15 minutes without visitors the API shuts down. The
  next visitor waits about 1 minute while it wakes up and re-downloads the
  bundle. The website shows "Waking up the model server" during this time.
  Before a demo or your defence, open the website a few minutes early.
- **It is slow.** The free plan has a small slice of a CPU, so one report
  can take 10 to 30 seconds, mostly in BiomedCLIP. The paid Starter plan is
  much faster.
- **Memory.** It has 512 MB. The API uses about 340 MB with the int8
  bundle, so it fits. The fp32 bundle needs a paid plan.

---

## Step 5. Deploy the website on Vercel

1. Sign in to https://vercel.com with GitHub.
2. Click **Add New... -> Project**, find `ultrasound-report-system`, and
   click **Import**.
3. On the configuration screen:
   - **Root Directory**: click **Edit** and choose `frontend`. (Important:
     the website is in a subfolder.)
   - **Framework Preset**: Next.js (detected automatically).
   - **Environment Variables**: add one:
     - Name: `NEXT_PUBLIC_API_URL`
     - Value: your Render address from Step 4, with no slash at the end,
       e.g. `https://ultrasound-report-api.onrender.com`
4. Click **Deploy**. It takes about 1 minute. Vercel gives you an address
   like `https://ultrasound-report-system.vercel.app`.

If you change `NEXT_PUBLIC_API_URL` later, you must **redeploy** (Deployments
-> three dots on the latest -> Redeploy), because the value is built into
the website at build time.

---

## Step 6. Connect them and test

1. Go back to Render -> your service -> **Environment**, and change
   `ALLOWED_ORIGINS` from `*` to your Vercel address, e.g.
   `https://ultrasound-report-system.vercel.app` (no slash at the end).
   Save. Render restarts the API.
2. Open your Vercel address. The badge under the title should turn green:
   "Model server ready".
3. Upload an ultrasound image (a single frame or a full scanned sheet),
   tick the checkbox, and click **Generate report**.
4. You should see the editable report (examination, findings, impression),
   the frames the system cut out, the 5 most similar past cases with their
   radiologist impressions, the finding vote, and the classifier
   probabilities. **Print / Save as PDF** prints only the report.

From now on, every `git push` to `main` redeploys both the API (Render) and
the website (Vercel) automatically.

---

## Troubleshooting

| What you see | Likely cause | Fix |
|---|---|---|
| Badge stays "Waking up the model server" for minutes | API asleep, still building, or crashed | Open Render -> Logs. Open `/health` yourself |
| `/health` says `"error"` with `No model bundle configured` | `BUNDLE_URL` missing | Add it in Render -> Environment |
| `/health` error mentions `HTTP Error 404` or `401` | Wrong link, or private repo without token | Use the `/resolve/main/` link; add `HF_TOKEN` for private repos |
| `/health` error `BadZipFile` | The link downloads a web page, not the zip | Use `/resolve/` (Hugging Face) or the release asset link (GitHub) |
| Render log "Out of memory (used over 512Mi)" | fp32 bundle on the free plan | Rebuild with `WEIGHTS = "int8"` or upgrade the plan |
| Website says "Could not reach the model server" and the browser console shows a CORS error | `ALLOWED_ORIGINS` does not match the website address | Copy the exact Vercel address (https, no trailing slash) into `ALLOWED_ORIGINS` |
| Website calls `localhost:8000` | `NEXT_PUBLIC_API_URL` not set on Vercel, or not redeployed | Set it, then Redeploy |
| Vercel build fails "Couldn't find any pages or app directory" | Root Directory not set | Project Settings -> General -> Root Directory = `frontend` |
| Warning "This image looks unlike every study in the reference corpus" | Not an ultrasound, or a type of scan the corpus does not contain | Expected behaviour. The draft is unreliable for such images |

---

## Running everything on your own computer (optional)

Needs Python 3.11 and Node.js 20 or newer.

```bash
# 1. API: unzip web_bundle.zip somewhere first
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
BUNDLE_DIR=/path/to/web_bundle uvicorn app.main:app --reload --port 8000
# Windows PowerShell: $env:BUNDLE_DIR="C:\path\to\web_bundle"; uvicorn app.main:app --reload --port 8000

# 2. Website, in a second terminal
cd frontend
cp .env.example .env.local         # it already points at http://localhost:8000
npm install
npm run dev
# open http://localhost:3000
```

---

## How the deployed system matches the paper

| Paper (section) | Code |
|---|---|
| Frame decomposition, banner removal, sector crop, CLAHE, letterbox 224 (3.3) | `backend/app/preprocess.py`, copied from notebook 02 |
| BiomedCLIP ViT-B/16, study embedding = renormalised mean of normalised frames (Eq. 6) | `backend/app/encoder.py`; weights exported by `kaggle/export_web_bundle.py` |
| Top-K inner-product retrieval (Eq. 7), K = 5 | `backend/app/rag.py::retrieve` |
| Similarity-weighted vote, tau = 0.4 (Eq. 8) | `backend/app/rag.py::vote`, copied from notebook 05 |
| Phrasebook realisation; normal impression when empty (3.6) | `backend/app/rag.py::realise` |
| Section-aware findings body (notebook 05, method 6c) | `backend/app/rag.py::findings_body` |
| BiomedCLIP probe (image), Table 4 | `probe.json`, shown as supporting evidence only |

At deployment the whole corpus is the reference set. There are no folds,
because the uploaded image is never part of the corpus. Exactly as the
paper says (Section 5), the output is a draft for a radiologist to accept,
edit or discard. That is why the report fields are editable and the
disclaimer appears on screen and in print.
