import { useState } from "react";

export default function Hello() {
  const [clicks, setClicks] = useState(0);

  return (
    <div style={{
      background: "#0a0a0f",
      color: "#c8c8d4",
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "'IBM Plex Mono', monospace",
      gap: 24,
    }}>
      <h1 style={{ color: "#00ff9d", fontSize: 48, fontWeight: 800, letterSpacing: 4 }}>
        HELLO WORLD
      </h1>
      <p style={{ color: "#5a5a72", fontSize: 14 }}>
        running as a native app via jxs
      </p>
      <button
        onClick={() => setClicks(c => c + 1)}
        style={{
          background: "#00ff9d",
          color: "#0a0a0f",
          border: "none",
          padding: "12px 32px",
          borderRadius: 6,
          fontSize: 14,
          fontWeight: 700,
          fontFamily: "'IBM Plex Mono', monospace",
          cursor: "pointer",
          letterSpacing: 2,
        }}
      >
        CLICKED {clicks} TIMES
      </button>
    </div>
  );
}
