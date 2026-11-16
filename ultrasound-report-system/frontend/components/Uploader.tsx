"use client";

import { useRef, useState } from "react";

const MAX_MB = 10;
const ACCEPT = ["image/jpeg", "image/png", "image/bmp", "image/webp", "image/tiff"];

export default function Uploader({
  disabled,
  onFile,
}: {
  disabled?: boolean;
  onFile: (f: File, previewUrl: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function take(f?: File | null) {
    setErr(null);
    if (!f) return;
    if (f.type && !ACCEPT.includes(f.type)) {
      setErr("Please choose an image file (JPG or PNG).");
      return;
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setErr(`The image is larger than ${MAX_MB} MB.`);
      return;
    }
    onFile(f, URL.createObjectURL(f));
  }

  return (
    <div>
      <div
        className={`drop ${drag ? "drop-active" : ""} ${disabled ? "drop-disabled" : ""}`}
        onClick={() => !disabled && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (!disabled) take(e.dataTransfer.files?.[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      >
        <div className="drop-icon">⬆</div>
        <div>
          <strong>Drop an ultrasound image here</strong> or click to choose one
        </div>
        <div className="muted small">
          A single frame or a scanned/printed sheet with several frames. JPG or PNG, up to {MAX_MB} MB.
        </div>
        <input
          ref={input}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => {
            take(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {err && <p className="error">{err}</p>}
    </div>
  );
}
