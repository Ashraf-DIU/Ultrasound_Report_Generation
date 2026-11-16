// Talks to the FastAPI service on Render.
// NEXT_PUBLIC_API_URL is set in Vercel (Settings -> Environment Variables).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export type Finding = { organ: string; text: string };

export type ReportResponse = {
  report: {
    exam_type: string | null;
    findings: Finding[];
    findings_text: string;
    impression: string;
  };
  consensus: { label: string; vote: number; kept: boolean; group: string }[];
  classifier: {
    label: string;
    probability: number;
    threshold: number;
    positive: boolean;
    cv_auc?: number | null;
  }[];
  similar_cases: {
    study_id: string;
    similarity: number;
    impression: string;
    exam_type: string | null;
    labels: string[];
  }[];
  frames: { index: number; box: number[]; thumbnail: string }[];
  quality: {
    n_frames: number;
    top_similarity: number;
    ood_threshold: number | null;
    out_of_distribution: boolean;
    warnings: string[];
  };
  method: { k: number; tau: number; model_name: string; corpus_size: number; seconds: number };
  disclaimer: string;
};

export type Health = { status: "loading" | "ready" | "error" | "offline"; error?: string | null };

export async function getHealth(timeoutMs = 70000): Promise<Health> {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(`${API_URL}/health`, { signal: ctrl.signal, cache: "no-store" });
    if (!r.ok) return { status: "offline", error: `HTTP ${r.status}` };
    return (await r.json()) as Health;
  } catch (e) {
    return { status: "offline", error: (e as Error).message };
  } finally {
    clearTimeout(t);
  }
}

export async function generateReport(file: File): Promise<ReportResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API_URL}/api/report`, { method: "POST", body: fd });
  if (!r.ok) {
    let msg = `Server error (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as ReportResponse;
}
