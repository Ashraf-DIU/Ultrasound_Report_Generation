export const metadata = { title: "How it works | Ultrasound Report Generator" };

// Numbers are from the manuscript "Retrieval-Augmented Consensus for
// Low-Resource Ultrasound Reporting" (Tables 4, 7, 8). Update them here if
// the paper changes.
export default function About() {
  return (
    <main className="container narrow">
      <h1>How it works</h1>
      <p className="lead">
        The system follows the pipeline of the paper <em>Retrieval-Augmented Consensus for Low-Resource
        Ultrasound Reporting</em>. Instead of training a text generator on a few hundred reports, it reuses what
        radiologists actually wrote for the most similar past cases.
      </p>

      <ol className="steps">
        <li>
          <strong>Frame decomposition.</strong> A scanned sheet is split into its separate frames using
          connected components and projection-profile grid cuts.
        </li>
        <li>
          <strong>Preprocessing.</strong> The console banner is removed, the ultrasound sector is cropped,
          contrast is enhanced with CLAHE, and each frame is letterboxed to 224×224.
        </li>
        <li>
          <strong>Encoding.</strong> Each frame goes through the BiomedCLIP ViT-B/16 image encoder. The study
          embedding is the re-normalised mean of the L2-normalised frame embeddings.
        </li>
        <li>
          <strong>Retrieval.</strong> The 5 most similar studies are found by inner product over the
          reference corpus (equivalent to a FAISS flat index).
        </li>
        <li>
          <strong>Consensus fusion.</strong> Findings are extracted from each neighbour&apos;s impression with a
          33-class rule ontology. Each neighbour votes with weight proportional to its similarity, and findings
          with at least 40% of the vote are kept.
        </li>
        <li>
          <strong>Realisation.</strong> Each kept finding is written with the radiologists&apos; most frequent
          phrasing for it. If nothing is kept, the normal impression is used. The findings section is assembled
          from organ lines that the neighbours agree on.
        </li>
      </ol>

      <h2>Reported performance</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>CE-F1</th>
            <th>Exact finding set</th>
            <th>BLEU-4</th>
          </tr>
        </thead>
        <tbody>
          <tr className="pos">
            <td>Consensus fusion, K=5 (this system)</td>
            <td>0.411</td>
            <td>37.5%</td>
            <td>0.100</td>
          </tr>
          <tr>
            <td>Retrieve-and-copy, K=1</td>
            <td>0.281</td>
            <td>25.1%</td>
            <td>0.194</td>
          </tr>
          <tr>
            <td>Prefix-conditioned decoder</td>
            <td>0.219</td>
            <td>8.5%</td>
            <td>0.038</td>
          </tr>
          <tr>
            <td>Constant &quot;Normal USG findings.&quot;</td>
            <td>0.317</td>
            <td>32.4%</td>
            <td>0.000</td>
          </tr>
        </tbody>
      </table>
      <p className="muted small">
        CE = clinical efficacy over the finding ontology, five-fold leakage-controlled cross-validation on 259
        studies. Retrieval: Recall@5 0.744, MRR 0.520.
      </p>

      <h2>Limitations</h2>
      <ul>
        <li>The reference corpus is 259 studies from a single centre, so the system only knows that centre&apos;s cases and wording.</li>
        <li>Rare findings (fewer than 5 examples) are almost never produced. The system tends to fall back to &quot;normal&quot;.</li>
        <li>Findings defined by a measurement (for example endometrial thickness or post-void residual) cannot be read from the image.</li>
        <li>No reader study has been done. The drafts&apos; clinical usefulness is unmeasured.</li>
      </ul>

      <div className="disclaimer">
        <strong>Not for clinical use.</strong> Outputs are research drafts for review by a qualified radiologist and
        must never be used on their own for diagnosis or treatment decisions.
      </div>
    </main>
  );
}
