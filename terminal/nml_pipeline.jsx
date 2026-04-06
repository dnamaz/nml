import { useState, useRef, useEffect } from "react";
import promptsData from "./pipeline_prompts.json";

// ── Server config ─────────────────────────────────────────────────────────────
// Both think and code use the same model via nml_server proxy on port 8082
function _getPort() {
  try { return new URLSearchParams(window.location.search).get("port") || "8082"; }
  catch { return "8082"; }
}
const API_BASE    = `http://localhost:${_getPort()}/v1`;
const THINK_BASE  = API_BASE;
const CODE_BASE   = API_BASE;

const THINK_SYSTEM = `You are an NML v0.10.0 (Neural Machine Language) architecture planner.
NML is an 89-opcode tensor register machine with 32 registers (R0-RV).
Analyze the user's request and produce a concrete implementation plan that a code generator will follow exactly.

Your plan MUST include:

1. TENSOR NAMES — every named tensor the program needs, with shape and purpose.
   Example: @input shape=1,4, @w1 shape=4,8, @b1 shape=1,8, @w2 shape=8,3, @b2 shape=1,3

2. REGISTER LAYOUT — which register holds which tensor.
   Example: R0=input, R1=w1, R2=b1, R3=w2, R4=b2, R5=hidden, R6=logits, R7=output

3. INSTRUCTION SEQUENCE — the exact opcodes in order, one per line, with register operands.
   Example:
   LD    R0 @input
   LD    R1 @w1
   LD    R2 @b1
   MMUL  R5 R0 R1
   MADD  R5 R5 R2
   RELU  R5 R5
   SOFT  R6 R5
   ST    R6 @output
   HALT

Rules:
- MMUL takes 3 registers: MMUL Rdest Rs1 Rs2  (never immediates)
- MADD takes 3 registers: MADD Rdest Rs1 Rs2
- SOFT takes 2 registers: SOFT Rdest Rs
- RELU takes 2 registers: RELU Rdest Rs
- LD takes a register and a named tensor: LD R0 @name
- ST takes a register and a named tensor: ST R0 @name
- Always end with HALT
- Never use immediates (#value) as operands to MMUL, MADD, RELU, SOFT
- TNET runs end-to-end mini-batch SGD: TNET #epochs #lr #loss_type #batch_size
  Where loss_type=0 (MSE) and batch_size=0 means full-batch. Example: TNET #100 #0.0100 #0 #32
  Register layout for TNET: R0=training_data(N×K), R1=w1(K×H), R2=b1(1×H), R3=w2(H×1), R4=b2(1×1), R9=training_labels(N×1)
  TNET writes updated weights to R1-R4 and final loss to R8; no manual BKWD/WUPD needed
  For inference after training: LD @new_input → MMUL → MADD → RELU → MMUL → MADD → ST @output
- BN Rd Rs [Rgamma [Rbeta]] — batch normalization (2D or 4D tensors). Normalizes activations.
  Example: BN R5 R5 R6 R7  (normalize R5 with learned gamma=R6, beta=R7)
- DROP Rd Rs #rate — inverted dropout with given rate. Use #0.0 at inference to disable.
  Example: DROP R5 R5 #0.2  (20% dropout during training)
- SADD Rd Rs Rs2|#imm — scalar add. SSUB Rd Rs Rs2|#imm — scalar subtract.

Be specific. The code generator cannot infer missing details.`;

const CODE_SYSTEM = `You are an NML (Neural Machine Language) assembler. Output only valid NML assembly code. Do not include explanations, markdown, or commentary.`;

// ── NML opcode highlighting ───────────────────────────────────────────────────
const NML_OPCODES = new Set([
  "MMUL","MADD","MSUB","EMUL","EDIV","SDOT","SCLR","SDIV","SADD","SSUB",
  "RELU","SIGM","TANH","SOFT","GELU",
  "LD","ST","MOV","ALLC","LEAF",
  "RSHP","TRNS","SPLT","MERG",
  "CMP","CMPI","CMPF",
  "JMPT","JMPF","JUMP","LOOP","ENDP","CALL","RET",
  "HALT","TRAP","SYNC","SYS","MOD","ITOF","FTOI","BNOT",
  "CONV","POOL","UPSC","PADZ",
  "ATTN","NORM","EMBD",
  "RDUC","WHER","CLMP","CMPR",
  "FFT","FILT",
  "META","FRAG","ENDF","LINK","PTCH","SIGN","VRFY","VOTE","PROJ","DIST","GATH","SCAT",
  "BKWD","WUPD","LOSS","TNET","TNDEEP","TRAIN","INFER","WDECAY","TLOG","BN","DROP",
  "RELUBK","SIGMBK","TANHBK","GELUBK","SOFTBK","MMULBK","CONVBK","POOLBK","NORMBK","ATTNBK",
]);

// ── NML version history — opcodes added per version ──────────────────────────
const NML_VERSIONS = [
  { ver: "v0.2",   total: 28, opcodes: "MMUL MADD MSUB EMUL SDOT SCLR RELU SIGM TANH SOFT LD ST MOV ALLC RSHP TRNS LOOP ENDP SYNC HALT LEAF TACC CMP JMPT JMPF JUMP CMPF CMPI", note: "Initial spec — NN + tree models" },
  { ver: "v0.3",   total: 42, opcodes: "CONV POOL UPSC PADZ ATTN NORM EMBD RDUC WHER CLMP CMPR FFT FILT GELU", note: "NML-V, NML-T, NML-R, NML-S extensions" },
  { ver: "v0.4",   total: 49, opcodes: "SDIV EDIV SPLT MERG CALL RET TRAP", note: "Backward jumps, error codes, --trace" },
  { ver: "v0.5",   total: 49, opcodes: "", note: "Per-tensor dtypes (f32, f64, i32), auto promotion" },
  { ver: "v0.6",   total: 62, opcodes: "META FRAG ENDF LINK PTCH SIGN VRFY VOTE PROJ DIST GATH SCAT", note: "NML-M2M: signing, fragments, consensus" },
  { ver: "v0.6.2", total: 67, opcodes: "SYS MOD ITOF FTOI BNOT", note: "NML-G general-purpose extension" },
  { ver: "v0.7",   total: 82, opcodes: "BKWD WUPD LOSS TNET RELUBK SIGMBK TANHBK GELUBK SOFTBK MMULBK CONVBK POOLBK NORMBK ATTNBK TNDEEP", note: "NML-TR training + 11 backward opcodes" },
  { ver: "v0.8",   total: 85, opcodes: "SADD SSUB TACC", note: "Scalar add/sub, tree accumulate" },
  { ver: "v0.9",   total: 85, opcodes: "TLOG TRAIN INFER WDECAY", note: "Config-driven training, AdamW, forward-only inference" },
  { ver: "v0.10",  total: 89, opcodes: "BN DROP", note: "Batch normalization, inverted dropout, N-layer TNET" },
];

function highlightNML(text) {
  return text.split("\n").map((line, i) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(";")) {
      return <div key={i} style={{ color: "#6a737d", minHeight: "1.4em" }}>{line || " "}</div>;
    }
    const parts = line.split(/(\s+)/);
    return (
      <div key={i} style={{ minHeight: "1.4em" }}>
        {parts.map((part, j) => {
          if (NML_OPCODES.has(part.trim()))
            return <span key={j} style={{ color: "#79b8ff", fontWeight: 600 }}>{part}</span>;
          if (/^@\w+/.test(part.trim()))
            return <span key={j} style={{ color: "#f8c555" }}>{part}</span>;
          if (/^#/.test(part.trim()))
            return <span key={j} style={{ color: "#9ecbff" }}>{part}</span>;
          if (/^R[0-9A-V]$/.test(part.trim()))
            return <span key={j} style={{ color: "#b392f0" }}>{part}</span>;
          return <span key={j}>{part}</span>;
        })}
      </div>
    );
  });
}


function extractThinkContent(text) {
  // Full <think>...</think> block
  const full = text.match(/<think>([\s\S]*?)<\/think>/);
  if (full) return full[1].trim();
  // Closing </think> only — take everything before it
  const closing = text.match(/^([\s\S]*?)<\/think>/);
  if (closing) return closing[1].trim();
  return text.trim();
}

// ── Panel component ───────────────────────────────────────────────────────────
function Panel({ title, subtitle, color, content, isCode, loading, error, children }) {
  const borderColor = color === "think" ? "#388bfd" : "#3fb950";
  const labelColor  = color === "think" ? "#79b8ff" : "#56d364";

  return (
    <div style={{
      flex: 1, display: "flex", flexDirection: "column", minWidth: 0,
      border: `1px solid ${borderColor}33`,
      borderRadius: 8, overflow: "hidden", background: "#0d1117",
    }}>
      {/* Header */}
      <div style={{
        padding: "10px 16px", borderBottom: `1px solid ${borderColor}33`,
        background: "#161b22", display: "flex", alignItems: "center", gap: 10,
      }}>
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: loading ? "#e3b341" : error ? "#f85149" : borderColor,
          boxShadow: loading ? "0 0 6px #e3b341" : "",
          transition: "background 0.3s",
        }} />
        <div>
          <div style={{ color: labelColor, fontWeight: 700, fontSize: 13 }}>{title}</div>
          <div style={{ color: "#8b949e", fontSize: 11 }}>{subtitle}</div>
        </div>
        <div style={{ marginLeft: "auto" }}>
          {children}
        </div>
      </div>

      {/* Content */}
      <div style={{
        flex: 1, overflowY: "auto", padding: 16, fontFamily: "monospace",
        fontSize: 13, lineHeight: 1.6, color: "#e6edf3",
        minHeight: 300, whiteSpace: "pre-wrap",
      }}>
        {loading && (
          <div style={{ color: "#8b949e", fontStyle: "italic" }}>Generating...</div>
        )}
        {error && (
          <div style={{ color: "#f85149" }}>{error}</div>
        )}
        {!loading && !error && content && (
          isCode ? highlightNML(content) : <span style={{ color: "#e6edf3" }}>{content}</span>
        )}
        {!loading && !error && !content && (
          <div style={{ color: "#30363d" }}>Output will appear here...</div>
        )}
      </div>
    </div>
  );
}

// ── Prompt library sidebar ────────────────────────────────────────────────────
function PromptLibrary({ onSelect }) {
  const [prompts, setPrompts]     = useState(null);
  const [expanded, setExpanded]   = useState({});


  useEffect(() => {
    setPrompts(promptsData);
    if (promptsData.categories?.length) setExpanded({ [promptsData.categories[0].name]: true });
  }, []);

  const toggle = name => setExpanded(p => ({ ...p, [name]: !p[name] }));

  return (
    <div style={{
      width: 240, flexShrink: 0, background: "#161b22",
      borderRight: "1px solid #30363d", overflowY: "auto",
      display: "flex", flexDirection: "column",
    }}>
      <div style={{
        padding: "12px 14px", borderBottom: "1px solid #30363d",
        color: "#8b949e", fontSize: 11, fontWeight: 700, letterSpacing: 1,
        textTransform: "uppercase",
      }}>
        Prompt Library
      </div>

      {!prompts && (
        <div style={{ padding: 12, color: "#8b949e", fontSize: 12 }}>Loading...</div>
      )}

      {prompts?.categories?.map(cat => (
        <div key={cat.name}>
          <button
            onClick={() => toggle(cat.name)}
            style={{
              width: "100%", textAlign: "left", padding: "8px 14px",
              background: "none", border: "none", cursor: "pointer",
              color: "#e6edf3", fontSize: 12, fontWeight: 600,
              borderBottom: "1px solid #21262d",
              display: "flex", alignItems: "center", gap: 6,
            }}
          >
            <span style={{ color: "#8b949e" }}>{expanded[cat.name] ? "▾" : "▸"}</span>
            {cat.name}
          </button>
          {expanded[cat.name] && cat.prompts.map((p, i) => (
            <button
              key={i}
              onClick={() => onSelect(p)}
              style={{
                width: "100%", textAlign: "left",
                padding: "6px 14px 6px 26px",
                background: "none", border: "none", cursor: "pointer",
                color: "#8b949e", fontSize: 11, lineHeight: 1.4,
                borderBottom: "1px solid #21262d",
                transition: "color 0.15s, background 0.15s",
              }}
              onMouseEnter={e => { e.target.style.color="#e6edf3"; e.target.style.background="#21262d"; }}
              onMouseLeave={e => { e.target.style.color="#8b949e"; e.target.style.background="none"; }}
            >
              {p}
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function NMLPipeline() {
  const [prompt, setPrompt]               = useState("");
  const [thinkOut, setThinkOut]           = useState("");
  const [codeOut, setCodeOut]             = useState("");
  const [thinkLoading, setThinkLoading]   = useState(false);
  const [codeLoading, setCodeLoading]     = useState(false);
  const [thinkError, setThinkError]       = useState("");
  const [codeError, setCodeError]         = useState("");
  const [showThinkRaw, setShowThinkRaw]   = useState(false);
  const [thinkRaw, setThinkRaw]           = useState("");
  const [execOut, setExecOut]             = useState(null);  // null = hidden
  const [execLoading, setExecLoading]     = useState(false);
  const [execError, setExecError]         = useState("");
  const [showVersions, setShowVersions]   = useState(false);
  const [dataInput, setDataInput]         = useState("");
  const [showData, setShowData]           = useState(false);
  const [showDataHelp, setShowDataHelp]   = useState(false);

  const textareaRef = useRef(null);

  const DATA_TEMPLATE = `; .nml.data — named tensor definitions
; Format: @name shape=rows,cols dtype=f32 data=v1,v2,...

; --- Inference example ---
@input    shape=1,4   dtype=f32  data=0.5,0.3,0.8,0.1
@w1       shape=4,8   dtype=f32  data=0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2,0.1,0.2
@b1       shape=1,8   dtype=f32  data=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
@target   shape=1,3   dtype=f32  data=0.0,1.0,0.0

; --- Training example (TNET) ---
; N=4 samples, K=2 features → training_data shape=4,2
; training_labels shape=4,1  (regression target)
; @w1 shape=2,4  @b1 shape=1,4  @w2 shape=4,1  @b2 shape=1,1
@training_data    shape=4,2  dtype=f32  data=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
@training_labels  shape=4,1  dtype=f32  data=0.3,0.7,1.1,1.5
@w1               shape=2,4  dtype=f32  data=0.1,-0.2,0.3,-0.1,0.2,-0.3,0.1,-0.2
@b1               shape=1,4  dtype=f32  data=0.0,0.0,0.0,0.0
@w2               shape=4,1  dtype=f32  data=0.2,-0.1,0.3,-0.2
@b2               shape=1,1  dtype=f32  data=0.0`;

  const callModel = async (base, system, userMsg, setOut, setLoading, setError, maxTokens = 800) => {
    setLoading(true);
    setError("");
    setOut("");
    try {
      const r = await fetch(`${base}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            { role: "system", content: system },
            { role: "user",   content: userMsg },
          ],
          max_tokens: maxTokens,
          stream: false,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const text = data.choices?.[0]?.message?.content || "";
      setOut(text);
      return text;
    } catch (e) {
      setError(`Error: ${e.message}`);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const runThink = async (p) => {
    const raw = await callModel(THINK_BASE, THINK_SYSTEM, p, setThinkRaw, setThinkLoading, setThinkError, 700);
    if (raw) setThinkOut(extractThinkContent(raw));
    return raw;
  };

  const runCode = async (userMsg) => {
    await callModel(CODE_BASE, CODE_SYSTEM, userMsg, setCodeOut, setCodeLoading, setCodeError, 600);
  };

  const runExecute = async (code) => {
    setExecLoading(true);
    setExecError("");
    setExecOut("");
    try {
      const r = await fetch(`http://localhost:${_getPort()}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nml_program: code, data: dataInput }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const result = await r.json();
      setExecOut(result.output ?? result.error ?? JSON.stringify(result, null, 2));
      if (result.error) setExecError(result.error);
    } catch (e) {
      setExecError(`Error: ${e.message}`);
      setExecOut("");
    } finally {
      setExecLoading(false);
    }
  };

  const runPipeline = async () => {
    if (!prompt.trim()) return;
    setCodeOut("");
    setCodeError("");
    setExecOut(null);

    const raw = await runThink(prompt);
    if (!raw) return;

    const reasoning = extractThinkContent(raw);
    const codePrompt = `Generate NML assembly for: ${prompt}\n\nPlan:\n${reasoning.slice(0, 1000)}`;
    await runCode(codePrompt);
  };

  const sendToCode = async () => {
    if (!thinkOut) return;
    const codePrompt = `Generate NML assembly for: ${prompt}\n\nPlan:\n${thinkOut.slice(0, 1000)}`;
    await runCode(codePrompt);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) runPipeline();
  };

  const handleSelect = (p) => {
    setPrompt(p);
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  const btnStyle = (color, disabled) => ({
    padding: "7px 16px", borderRadius: 6, border: "none",
    cursor: disabled ? "not-allowed" : "pointer",
    fontWeight: 600, fontSize: 13, transition: "opacity 0.15s",
    opacity: disabled ? 0.4 : 1,
    background: color, color: "#fff",
  });

  const smallBtn = (active) => ({
    background: "none", border: `1px solid ${active ? "#388bfd" : "#30363d"}`, borderRadius: 4,
    color: active ? "#79b8ff" : "#8b949e", fontSize: 11, cursor: "pointer", padding: "2px 8px",
  });

  return (
    <div style={{
      display: "flex", height: "100vh", background: "#0d1117",
      fontFamily: "ui-monospace, monospace", color: "#e6edf3", overflow: "hidden",
    }}>
      {/* Sidebar */}
      <PromptLibrary onSelect={handleSelect} />

      {/* Main */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Top bar */}
        <div style={{
          padding: "12px 20px", borderBottom: "1px solid #30363d",
          background: "#161b22", display: "flex", alignItems: "center", gap: 12,
        }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#e6edf3" }}>NML Pipeline</div>
            <div style={{ fontSize: 11, color: "#8b949e" }}>Think → Code → Execute &nbsp;·&nbsp; 89 opcodes &nbsp;·&nbsp; v0.10.0</div>
          </div>
          <button onClick={() => setShowVersions(v => !v)} style={{
            background: showVersions ? "#388bfd22" : "none",
            border: `1px solid ${showVersions ? "#388bfd" : "#30363d"}`,
            borderRadius: 4, color: showVersions ? "#79b8ff" : "#8b949e",
            fontSize: 11, cursor: "pointer", padding: "4px 10px",
          }}>
            {showVersions ? "Hide Versions" : "Opcode Versions"}
          </button>
        </div>

        {showVersions && (
          <div style={{
            maxHeight: 220, overflowY: "auto", padding: "10px 20px",
            borderBottom: "1px solid #30363d", background: "#0d1117",
          }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ color: "#8b949e", textAlign: "left" }}>
                  <th style={{ padding: "4px 8px", borderBottom: "1px solid #21262d", width: 60 }}>Version</th>
                  <th style={{ padding: "4px 8px", borderBottom: "1px solid #21262d", width: 40 }}>Total</th>
                  <th style={{ padding: "4px 8px", borderBottom: "1px solid #21262d" }}>New Opcodes</th>
                  <th style={{ padding: "4px 8px", borderBottom: "1px solid #21262d" }}>Notes</th>
                </tr>
              </thead>
              <tbody>
                {NML_VERSIONS.map((v, i) => (
                  <tr key={i} style={{ color: i === NML_VERSIONS.length - 1 ? "#56d364" : "#e6edf3" }}>
                    <td style={{ padding: "3px 8px", borderBottom: "1px solid #21262d11", fontWeight: 600 }}>{v.ver}</td>
                    <td style={{ padding: "3px 8px", borderBottom: "1px solid #21262d11", color: "#79b8ff" }}>{v.total}</td>
                    <td style={{ padding: "3px 8px", borderBottom: "1px solid #21262d11", fontFamily: "monospace", color: "#f8c555", wordBreak: "break-word" }}>
                      {v.opcodes || "—"}
                    </td>
                    <td style={{ padding: "3px 8px", borderBottom: "1px solid #21262d11", color: "#8b949e" }}>{v.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Prompt input */}
        <div style={{
          padding: "14px 20px", borderBottom: "1px solid #30363d",
          background: "#161b22", display: "flex", flexDirection: "column", gap: 10,
        }}>
          <textarea
            ref={textareaRef}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you want to build in NML... (⌘↵ to run pipeline)"
            rows={3}
            style={{
              width: "100%", background: "#0d1117", color: "#e6edf3",
              border: "1px solid #30363d", borderRadius: 6,
              padding: "10px 12px", fontSize: 13, fontFamily: "inherit",
              resize: "none", outline: "none", boxSizing: "border-box",
            }}
          />
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button onClick={runPipeline} disabled={!prompt.trim() || thinkLoading || codeLoading}
              style={btnStyle("#388bfd", !prompt.trim() || thinkLoading || codeLoading)}>
              ▶ Run Pipeline
            </button>
            <button onClick={() => runThink(prompt)} disabled={!prompt.trim() || thinkLoading}
              style={btnStyle("#1f6feb", !prompt.trim() || thinkLoading)}>
              Think only
            </button>
            <button onClick={sendToCode} disabled={!thinkOut || codeLoading}
              style={btnStyle("#2ea043", !thinkOut || codeLoading)}>
              → Send to Code
            </button>
            <button onClick={() => codeOut && runExecute(codeOut)} disabled={!codeOut || execLoading}
              style={btnStyle("#6e40c9", !codeOut || execLoading)}>
              ⚡ Execute
            </button>
            <button
              onClick={() => setShowData(v => !v)}
              style={{ ...smallBtn(showData), padding: "7px 12px", fontSize: 12 }}>
              {showData ? "Hide Data" : "Data ▾"}
            </button>
            <button
              onClick={() => { setThinkOut(""); setThinkRaw(""); setCodeOut(""); setThinkError(""); setCodeError(""); setExecOut(null); setExecError(""); }}
              style={{ ...btnStyle("#21262d", false), color: "#8b949e", marginLeft: "auto" }}>
              Clear
            </button>
          </div>

          {/* Data input */}
          {showData && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 11, color: "#8b949e" }}>.nml.data — tensor definitions passed to Execute</span>
                <button onClick={() => setShowDataHelp(v => !v)} style={smallBtn(showDataHelp)}>
                  {showDataHelp ? "Hide format" : "Format help"}
                </button>
                <button onClick={() => setDataInput(DATA_TEMPLATE)} style={smallBtn(false)}>
                  Load template
                </button>
              </div>
              {showDataHelp && (
                <pre style={{
                  margin: 0, padding: "10px 12px", background: "#0d1117",
                  border: "1px solid #21262d", borderRadius: 6,
                  fontSize: 11, color: "#8b949e", overflowX: "auto",
                }}>{DATA_TEMPLATE}</pre>
              )}
              <textarea
                value={dataInput}
                onChange={e => setDataInput(e.target.value)}
                placeholder="@input shape=1,4 dtype=f32 data=0.5,0.3,0.8,0.1"
                rows={4}
                style={{
                  width: "100%", background: "#0d1117", color: "#e6edf3",
                  border: "1px solid #30363d", borderRadius: 6,
                  padding: "8px 12px", fontSize: 12, fontFamily: "inherit",
                  resize: "vertical", outline: "none", boxSizing: "border-box",
                }}
              />
            </div>
          )}
        </div>

        {/* Panels */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Think + Code panels */}
          <div style={{ flex: 1, display: "flex", gap: 12, padding: "12px 16px 6px", overflow: "hidden" }}>
            <Panel title="Think Model" subtitle="nml-1.5b-instruct-v0.10.0 · Reasoning"
              color="think" content={showThinkRaw ? thinkRaw : thinkOut}
              isCode={false} loading={thinkLoading} error={thinkError}>
              {thinkOut && (
                <button onClick={() => setShowThinkRaw(v => !v)} style={smallBtn(showThinkRaw)}>
                  {showThinkRaw ? "Show clean" : "Show raw"}
                </button>
              )}
            </Panel>

            <Panel title="Code Model" subtitle="nml-1.5b-instruct-v0.10.0 · NML Assembly"
              color="code" content={codeOut}
              isCode={true} loading={codeLoading} error={codeError}>
              <div style={{ display: "flex", gap: 6 }}>
                {codeOut && (
                  <button onClick={() => navigator.clipboard?.writeText(codeOut)} style={smallBtn(false)}>
                    Copy
                  </button>
                )}
                {codeOut && (
                  <button onClick={() => runExecute(codeOut)} disabled={execLoading}
                    style={{ ...smallBtn(false), color: execLoading ? "#8b949e" : "#b392f0", borderColor: "#6e40c9" }}>
                    {execLoading ? "Running..." : "⚡ Execute"}
                  </button>
                )}
              </div>
            </Panel>
          </div>

          {/* Execution output */}
          {execOut !== null && (
            <div style={{
              height: 180, margin: "0 16px 12px", flexShrink: 0,
              border: `1px solid ${execError ? "#f8514933" : "#6e40c933"}`,
              borderRadius: 8, overflow: "hidden", background: "#0d1117",
            }}>
              <div style={{
                padding: "8px 14px", background: "#161b22",
                borderBottom: `1px solid ${execError ? "#f8514933" : "#6e40c933"}`,
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <div style={{
                  width: 7, height: 7, borderRadius: "50%",
                  background: execLoading ? "#e3b341" : execError ? "#f85149" : "#3fb950",
                }} />
                <span style={{ fontSize: 12, fontWeight: 700, color: execError ? "#f85149" : "#3fb950" }}>
                  Execution Output
                </span>
                <button onClick={() => setExecOut(null)}
                  style={{ marginLeft: "auto", background: "none", border: "none", color: "#8b949e", cursor: "pointer", fontSize: 14 }}>
                  ×
                </button>
              </div>
              <div style={{
                padding: 12, overflowY: "auto", height: "calc(100% - 35px)",
                fontFamily: "monospace", fontSize: 12, color: execError ? "#f85149" : "#e6edf3",
                whiteSpace: "pre-wrap",
              }}>
                {execLoading ? <span style={{ color: "#8b949e" }}>Executing...</span> : (execOut || execError)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
