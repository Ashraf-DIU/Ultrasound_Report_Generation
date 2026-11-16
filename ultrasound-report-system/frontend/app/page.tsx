"use client";

import { useState } from "react";
import ReportView from "@/components/ReportView";
import ServerStatus from "@/components/ServerStatus";
import Uploader from "@/components/Uploader";
import { generateReport, type ReportResponse } from "@/lib/api";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReportResponse | null>(null);
  const [agreed, setAgreed] = useState(false);

  async function run() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      setResult(await generateReport(file));
    } catch (e) {
      const msg = (e as Error).message;
      setError(
        msg === "Failed to fetch"
          ? "Could not reach the model server. It may still be waking up; wait a minute and try again."
          : msg,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="container">
      <section className="hero no-print">
        <h1>Ultrasound Report Generator</h1>
        <p className="lead">
          Upload an ultrasound image and get a draft report built from the most similar past cases,
          using BiomedCLIP retrieval and similarity-weighted consensus fusion.
        </p>
        <ServerStatus />
      </section>

      <div className="disclaimer no-print">
        <strong>Medical disclaimer.</strong> This is a research prototype from a final-year project. It
        is not a medical device, it does not give a diagnosis, and its output must not be used for patient
        care. Every draft must be checked by a qualified radiologist or sonologist. Do not upload images
        that show a patient&apos;s name or other identifying details.
      </div>

      <section className="card no-print">
        <Uploader
          disabled={busy}
          onFile={(f, url) => {
            setFile(f);
            setPreview(url);
            setResult(null);
            setError(null);
          }}
        />
        {file && (
          <div className="selected">
            <img src={preview} alt="selected" />
            <div>
              <div>
                <strong>{file.name}</strong> <span className="muted">({(file.size / 1024).toFixed(0)} KB)</span>
              </div>
              <label className="check">
                <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
                I understand this is a research tool and not a diagnosis.
              </label>
              <button onClick={run} disabled={busy || !agreed}>
                {busy ? "Generating report…" : "Generate report"}
              </button>
            </div>
          </div>
        )}
        {busy && <div className="progress"><span /></div>}
        {error && <p className="error">{error}</p>}
      </section>

      {result && <ReportView data={result} imageUrl={preview} />}
    </main>
  );
}
