import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Ultrasound Report Generator",
  description:
    "Research prototype: retrieval-augmented consensus report drafting for ultrasound images (BiomedCLIP + RAG).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="nav no-print">
          <div className="container nav-inner">
            <Link href="/" className="brand">
              US-RAG
            </Link>
            <nav>
              <Link href="/">Generate</Link>
              <Link href="/about">How it works</Link>
            </nav>
          </div>
        </header>
        {children}
        <footer className="footer no-print">
          <div className="container">
            Final Year Design Project, Department of CSE, Daffodil International University. Research use only,
            not for clinical diagnosis.
          </div>
        </footer>
      </body>
    </html>
  );
}
