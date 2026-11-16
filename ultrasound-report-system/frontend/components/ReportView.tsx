"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReportResponse } from "@/lib/api";

function pct(x: number) {
  return `${(x * 100).toFixed(0)}%`;
}

export default function ReportView({ data, imageUrl }: { data: ReportResponse; imageUrl: string }) {
  // The draft is editable: the paper frames the system as draft-assist,
  // where a radiologist accepts, edits or discards the suggestion.
  const [patient, setPatient] = useState({ name: "", age: "", sex: "", refBy: "" });
  const [exam, setExam] = useState(data.report.exam_type ?? "");
  const [findings, setFindings] = useState(
    data.report.findings.map((f) => `${f.organ}: ${f.text}`).join("\n"),
  );
  const [impression, setImpression] = useState(data.report.impression);
  const [copied, setCopied] = useState(false);
  const today = useMemo(() => new Date().toLocaleDateString(), []);

  useEffect(() => {
    setExam(data.report.exam_type ?? "");
    setFindings(data.report.findings.map((f) => `${f.organ}: ${f.text}`).join("\n"));
    setImpression(data.report.impression);
  }, [data]);

  const plainText = [
    "ULTRASOUND REPORT (AI-GENERATED DRAFT, NOT FOR CLINICAL USE)",
    patient.name ? `Patient: ${patient.name}` : null,
    patient.age ? `Age: ${patient.age}` : null,
    patient.sex ? `Sex: ${patient.sex}` : null,
    patient.refBy ? `Referred by: ${patient.refBy}` : null,
    `Date: ${today}`,
    exam ? `Examination: ${exam}` : null,
    "",
    "FINDINGS:",
    findings || "(none drafted)",
    "",
    "IMPRESSION:",
    impression,
    "",
    data.disclaimer,
  ]
    .filter((l): l is string => l !== null)
    .join("\n");

  function download() {
    const blob = new Blob([plainText], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `ultrasound-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  async function copy() {
    await navigator.clipboard.writeText(plainText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  const kept = data.consensus.filter((c) => c.kept);

  return (
    <div className="report-wrap">
      {data.quality.warnings.length > 0 && (
        <div className="warn no-print">
          {data.quality.warnings.map((w, i) => (
            <p key={i}>⚠ {w}</p>
          ))}
        </div>
      )}

      {/* ---------------- the printable report ---------------- */}
      <section className="card report" id="report">
        <div className="report-head">
          <div>
            <h2>Ultrasound Report</h2>
            <div className="badge">AI-generated draft. Requires review by a radiologist.</div>
          </div>
          <img src={imageUrl} alt="Uploaded ultrasound" className="report-thumb" />
        </div>

        <div className="grid4">
          <label>
            Patient name
            <input value={patient.name} onChange={(e) => setPatient({ ...patient, name: e.target.value })} placeholder="optional" />
          </label>
          <label>
            Age
            <input value={patient.age} onChange={(e) => setPatient({ ...patient, age: e.target.value })} placeholder="optional" />
          </label>
          <label>
            Sex
            <input value={patient.sex} onChange={(e) => setPatient({ ...patient, sex: e.target.value })} placeholder="optional" />
          </label>
          <label>
            Referred by
            <input value={patient.refBy} onChange={(e) => setPatient({ ...patient, refBy: e.target.value })} placeholder="optional" />
          </label>
        </div>
        <p className="muted small no-print">
          Patient details stay in your browser. They are never sent to the server.
        </p>

        <div className="row">
          <label className="grow">
            Examination
            <input value={exam} onChange={(e) => setExam(e.target.value)} />
          </label>
          <label>
            Date
            <input value={today} readOnly />
          </label>
        </div>

        <label>
          Findings
          <textarea rows={Math.max(4, findings.split("\n").length + 1)} value={findings} onChange={(e) => setFindings(e.target.value)} />
        </label>
        <label>
          Impression
          <textarea className="impression" rows={2} value={impression} onChange={(e) => setImpression(e.target.value)} />
        </label>

        <p className="disclaimer-print">{data.disclaimer}</p>

        <div className="actions no-print">
          <button onClick={() => window.print()}>Print / Save as PDF</button>
          <button className="secondary" onClick={download}>Download .txt</button>
          <button className="secondary" onClick={copy}>{copied ? "Copied" : "Copy text"}</button>
        </div>
      </section>

      {/* ---------------- how the draft was made ---------------- */}
      <div className="no-print">
        <section className="card">
          <h3>Frames the system found in your image</h3>
          <p className="muted small">
            The image is split into its separate ultrasound frames, the console banner is removed, and contrast is
            enhanced (CLAHE), the same way the training images were prepared.
          </p>
          <div className="frames">
            {data.frames.map((f) => (
              <img key={f.index} src={f.thumbnail} alt={`frame ${f.index + 1}`} />
            ))}
          </div>
        </section>

        <section className="card">
          <h3>Most similar past cases (the evidence for this draft)</h3>
          <p className="muted small">
            The {data.method.k} most visually similar studies from the reference corpus of {data.method.corpus_size}{" "}
            radiologist-reported studies. Their findings were voted on, weighted by similarity, and every finding with at
            least {pct(data.method.tau)} of the vote was kept.
          </p>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Case</th>
                <th>Similarity</th>
                <th>Exam</th>
                <th>Radiologist&apos;s impression</th>
              </tr>
            </thead>
            <tbody>
              {data.similar_cases.map((c, i) => (
                <tr key={c.study_id + i}>
                  <td>{i + 1}</td>
                  <td>{c.study_id}</td>
                  <td>{c.similarity.toFixed(3)}</td>
                  <td>{c.exam_type ?? "-"}</td>
                  <td>{c.impression}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="card">
          <h3>Finding vote</h3>
          {data.consensus.length === 0 ? (
            <p className="muted">No findings were voted for, so the normal impression was used.</p>
          ) : (
            <div className="bars">
              {data.consensus.map((c) => (
                <div key={c.label} className={`bar-row ${c.kept ? "kept" : ""}`}>
                  <span className="bar-label">
                    {c.label} {c.kept ? "✓" : ""}
                  </span>
                  <span className="bar">
                    <span className="bar-fill" style={{ width: pct(c.vote) }} />
                    <span className="bar-thr" style={{ left: pct(data.method.tau) }} />
                  </span>
                  <span className="bar-val">{pct(c.vote)}</span>
                </div>
              ))}
            </div>
          )}
          <p className="muted small">
            Kept: {kept.length ? kept.map((k) => k.label).join(", ") : "none (normal impression)"}. The line marks the
            {" "}{pct(data.method.tau)} threshold.
          </p>
        </section>

        {data.classifier.length > 0 && (
          <details className="card">
            <summary>
              <strong>Image classifier probabilities (supporting, low accuracy)</strong>
            </summary>
            <p className="muted small">
              A linear probe on frozen BiomedCLIP image features. In the paper this model reaches only macro-F1 0.22, so
              it is shown for transparency and is not used to write the report. &quot;CV AUC&quot; is its
              cross-validated AUC for that finding.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Finding</th>
                  <th>Probability</th>
                  <th>Threshold</th>
                  <th>CV AUC</th>
                </tr>
              </thead>
              <tbody>
                {data.classifier.map((c) => (
                  <tr key={c.label} className={c.positive ? "pos" : ""}>
                    <td>{c.label}</td>
                    <td>{pct(c.probability)}</td>
                    <td>{pct(c.threshold)}</td>
                    <td>{c.cv_auc != null ? c.cv_auc.toFixed(2) : "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}

        <p className="muted small center">
          Best match similarity {data.quality.top_similarity.toFixed(3)} · {data.quality.n_frames} frame(s) ·{" "}
          {data.method.seconds}s on the server
        </p>
      </div>
    </div>
  );
}
