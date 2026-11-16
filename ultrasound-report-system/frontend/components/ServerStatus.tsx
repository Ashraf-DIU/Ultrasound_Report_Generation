"use client";

import { useEffect, useState } from "react";
import { getHealth, type Health } from "@/lib/api";

// Render's free plan puts the API to sleep after 15 minutes without traffic.
// The first request wakes it, which takes about a minute. This badge pings
// /health when the page opens so the server is awake by the time a user
// has picked an image.
export default function ServerStatus({ onReady }: { onReady?: (ready: boolean) => void }) {
  const [h, setH] = useState<Health | null>(null);

  useEffect(() => {
    let stop = false;
    let tries = 0;
    async function poll() {
      while (!stop && tries < 40) {
        tries++;
        const res = await getHealth();
        if (stop) return;
        setH(res);
        onReady?.(res.status === "ready");
        if (res.status === "ready" || res.status === "error") return;
        await new Promise((r) => setTimeout(r, 4000));
      }
    }
    poll();
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const s = h?.status;
  const label =
    s === "ready"
      ? "Model server ready"
      : s === "error"
        ? "Model server error"
        : s === "loading"
          ? "Model is loading…"
          : s === "offline"
            ? "Waking up the model server (up to a minute)…"
            : "Connecting to the model server…";

  return (
    <div className={`status status-${s ?? "pending"}`} title={h?.error ?? ""}>
      <span className="dot" /> {label}
      {s === "error" && h?.error ? <span className="status-err"> {h.error}</span> : null}
    </div>
  );
}
