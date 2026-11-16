"""Retrieval-augmented consensus fusion, copied from notebook 05.

query study embedding
   -> top-K most similar corpus studies (inner product, FAISS IndexFlatIP
      equivalent; at a few hundred studies exact numpy search is identical)
   -> each neighbour votes for its ontology findings, weighted by similarity
   -> findings with normalised vote >= tau are kept           (Eq. 8)
   -> each kept finding is written with its phrasebook sentence
   -> nothing kept -> the canonical normal impression

K = 5 and tau = 0.4 as in the paper (Table 3). At deployment time there are
no folds: the whole corpus is the reference set, which is the "training
folds" of the paper's protocol.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

NORMAL_FALLBACK = "Normal USG findings."


def extract_labels(text: str, ontology: dict) -> set:
    """Notebook 05, cell 11."""
    t = str(text).lower()
    out = set()
    for lab, spec in ontology.items():
        inc = spec.get("include", [])
        exc = spec.get("exclude", [])
        if any(re.search(p, t) for p in inc) and not any(re.search(p, t) for p in exc):
            out.add(lab)
    return out


class ConsensusReporter:
    def __init__(self, studies: list, embeddings: np.ndarray, ontology: dict,
                 k: int = 5, tau: float = 0.4):
        self.studies = studies
        self.Z = embeddings
        self.ontology = ontology
        self.k = k
        self.tau = tau
        self.impressions = [str(s.get("impression") or "") for s in studies]
        self.findings = [str(s.get("findings") or "") for s in studies]
        # Label sets of the reference impressions (notebook 05 re-extracts them
        # with the same rules rather than trusting stored labels).
        self.label_sets = [extract_labels(t, ontology) for t in self.impressions]
        self.book = self._build_phrasebook()

    # -- notebook 05, cell 13: build_phrasebook ---------------------------
    def _build_phrasebook(self) -> dict:
        book = {}
        for lab in self.ontology:
            cands = [self.impressions[j] for j in range(len(self.studies))
                     if lab in self.label_sets[j] and len(self.label_sets[j]) == 1]
            if cands:
                book[lab] = Counter(c.strip() for c in cands).most_common(1)[0][0]
        return book

    def retrieve(self, q: np.ndarray, k: int | None = None):
        k = k or self.k
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = self.Z @ q
        order = np.argsort(-sims)[:k]
        return order, sims[order]

    # -- notebook 05, cell 13: weighted_vote ------------------------------
    def vote(self, idx, sims):
        w = np.clip(np.asarray(sims, dtype=np.float64), 1e-6, None)
        w = w / w.sum()
        score = defaultdict(float)
        for wt, j in zip(w, idx):
            for lab in self.label_sets[j]:
                score[lab] += wt
        kept = {lab for lab, s in score.items() if s >= self.tau}
        return kept, dict(score)

    # -- notebook 05, cell 13: realisation (6b) ---------------------------
    def realise(self, labs: set, idx) -> str:
        if not labs:
            return self.book.get("Normal study", NORMAL_FALLBACK)
        parts = [self.book.get(lab) for lab in sorted(labs) if self.book.get(lab)]
        parts = parts or [self.impressions[idx[0]]]
        if len(parts) == 1:
            return parts[0]
        return "Suggestive of " + "; ".join(
            re.sub(r"^(suggestive of|features of)\s+", "", p.strip().rstrip("."),
                   flags=re.I) for p in parts) + "."

    # -- notebook 05, cell 13: section-aware findings body (6c) ------------
    def findings_body(self, idx) -> list:
        organ_lines = defaultdict(Counter)
        for j in idx:
            for line in re.split(r"(?<=[.])\s+", self.findings[j]):
                m = re.match(r"\s*([A-Z][A-Za-z .\-/]{2,28}?)\s*[:\-]\s*(.+)", line)
                if m:
                    organ_lines[m.group(1).strip().title()][m.group(2).strip()] += 1
        body = []
        for organ, ctr in organ_lines.items():
            if sum(ctr.values()) >= max(2, int(0.4 * max(len(idx), 1))):
                body.append({"organ": organ, "text": ctr.most_common(1)[0][0]})
        return body[:12]

    def exam_type(self, idx, sims) -> str | None:
        score = defaultdict(float)
        for j, s in zip(idx, sims):
            et = self.studies[j].get("exam_type")
            if et:
                score[et] += max(float(s), 1e-6)
        return max(score, key=score.get) if score else None

    def report(self, q: np.ndarray) -> dict:
        idx, sims = self.retrieve(q)
        labs, scores = self.vote(idx, sims)
        impression = self.realise(labs, idx)
        body = self.findings_body(idx)
        groups = {lab: self.ontology.get(lab, {}).get("group", "Other") for lab in scores}
        return {
            "exam_type": self.exam_type(idx, sims),
            "findings": body,
            "impression": impression,
            "consensus": sorted(
                [{"label": lab, "vote": round(float(s), 4), "kept": lab in labs,
                  "group": groups[lab]} for lab, s in scores.items()],
                key=lambda d: -d["vote"]),
            "similar_cases": [
                {"study_id": str(self.studies[j].get("study_id", j)),
                 "similarity": round(float(s), 4),
                 "impression": self.impressions[j],
                 "exam_type": self.studies[j].get("exam_type"),
                 "labels": sorted(self.label_sets[j])}
                for j, s in zip(idx, sims)],
            "top_similarity": float(sims[0]) if len(sims) else 0.0,
        }
