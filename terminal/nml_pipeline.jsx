import { useState, useRef, useEffect } from "react";
import promptsData from "./pipeline_prompts.json";

// ── Server config ─────────────────────────────────────────────────────────────
// Server ports: think model (4B Q6) and code model (1.5B Q4) run separately.
// nml_server on SERVER_PORT proxies /v1/chat/completions to the code model
// and provides /generate_validated for the full validation pipeline.
function _getPort() {
  try { return new URLSearchParams(window.location.search).get("port") || "8082"; }
  catch { return "8082"; }
}
function _getThinkPort() {
  try { return new URLSearchParams(window.location.search).get("think") || "8084"; }
  catch { return "8084"; }
}
const SERVER_PORT = _getPort();
const API_BASE    = `http://localhost:${SERVER_PORT}/v1`;
// Think model: use dedicated port if available, fallback to server proxy
const THINK_PORT  = _getThinkPort();
const THINK_BASE  = `http://localhost:${THINK_PORT}/v1`;
const CODE_BASE   = API_BASE;

// Auto-detect: if think port is same as server port, use server for both
const THINK_ENDPOINT = THINK_PORT === SERVER_PORT ? API_BASE : THINK_BASE;

const THINK_SYSTEM = `You are an NML v0.10.0 (Neural Machine Language) architecture planner.
NML is an 89-opcode tensor register machine with 32 registers (R0-RV).
Translate the user's request into the SIMPLEST possible NML program. Do not add features, layers, or complexity beyond what was asked.

CRITICAL: Match complexity to the request.
- "add two tensors" → 3-4 instructions (LD, MADD, ST, HALT)
- "dot product and scale" → 4-5 instructions (LD, SDOT, SCLR, ST, HALT)
- "train a network" → use TNET (only when training is explicitly requested)
Do NOT use TNET, TNDEEP, or training opcodes unless the user specifically asks to train.

Your plan MUST include:

1. TENSOR NAMES — named tensors with shape and purpose.
   Example: @input shape=1,4, @weights shape=4,1

2. REGISTER LAYOUT — which register holds what.
   Example: R0=input, R1=weights, R2=result

3. INSTRUCTION SEQUENCE — exact opcodes in order.
   Example:
   LD    R0 @input
   LD    R1 @weights
   SDOT  R2 R0 R1
   ST    R2 @result
   HALT

Opcode quick reference:
  LD/ST: memory load/store with @name
  MADD/MSUB/EMUL/EDIV: element-wise arithmetic (3 registers)
  SADD/SSUB/SDIV/SCLR: scalar operations (Rd Rs #imm)
  MMUL: matrix multiply [M,K]×[K,N] (3 registers, never immediates)
  SDOT: dot product (3 registers)
  RELU/SIGM/TANH/GELU/SOFT: activations (2 registers)
  CLMP: clamp (Rd Rs #min #max)
  TNET: training (Rconfig #epochs) — R0=data, R1=config[n_layers,3], R9=labels
  LOSS: loss function (Rd Rpred Rlabel #type)
  BKWD: backward pass | WUPD: weight update (Rw Rgrad Rlr)
  CONV/POOL/ATTN/NORM/EMBD: vision/transformer ops
  HALT: must be last instruction

Rules:
- MMUL, MADD, MSUB take 3 registers — never immediates
- Always end with HALT
- Keep it minimal. Fewer instructions = better.`;

const CODE_SYSTEM = `You are an NML (Neural Machine Language) assembler. Output only valid NML assembly code. Do not include explanations, markdown, or commentary.

NML opcode reference (Rd=dest, Rs=source, #imm=immediate):
Memory:    LD Rd @name | ST Rs @name | ALLC Rd #shape | MOV Rd Rs
Arithmetic: MADD Rd Rs1 Rs2 | MSUB Rd Rs1 Rs2 | EMUL Rd Rs1 Rs2 | EDIV Rd Rs1 Rs2
            SADD Rd Rs #imm | SSUB Rd Rs #imm | SDIV Rd Rs #imm | SCLR Rd Rs #imm
Matrix:    MMUL Rd Rs1 Rs2 | SDOT Rd Rs1 Rs2 | TRNS Rd Rs | RSHP Rd Rs #shape
Activation: RELU Rd Rs | SIGM Rd Rs | TANH Rd Rs | GELU Rd Rs | SOFT Rd Rs
Vision:    CONV Rd Rs Rkernel #stride #pad | POOL Rd Rs #size #stride | UPSC Rd Rs #factor | PADZ Rd Rs #amount
Transformer: ATTN Rd Rq Rk Rv | NORM Rd Rs Rgamma Rbeta | EMBD Rd Rtable Rindex
Reduction: RDUC Rd Rs #dim #mode | CLMP Rd Rs #min #max | WHER Rd Rcond Rs1 Rs2 | CMPR Rd Rs #op #thresh
Training:  TNET Rconfig #epochs | LOSS Rd Rpred Rlabel #type | BKWD Rgrad Ract Rloss
           WUPD Rw Rgrad Rlr | BN Rd Rs Rgamma Rbeta | DROP Rd Rs #rate | WDECAY Rd #lambda
Backward:  RELUBK/SIGMBK/TANHBK/GELUBK/SOFTBK Rd Rgrad Rin (3 ops)
           MMULBK Rd_di Rd_dw Rgrad Rin Rw | CONVBK Rd_di Rd_dk Rgrad Rin Rk (5 ops)
Control:   HALT | JUMP #off | JMPT #off | JMPF #off | LOOP Rs|#n | ENDP | CALL #off | RET
General:   SYS Rd #code | FILL Rd #rows #cols #val | CMP Rs1 Rs2 | CMPI Rd Rs #imm
Signal:    FFT Rd Rs Rtwiddle | FILT Rd Rs Rkernel
Tree:      LEAF Rd #val | CMPF Rd Rs #feat #thresh

Rules:
- MMUL needs compatible shapes: [M,K] × [K,N] → [M,N]
- TNET register layout: R0=data, R1=config[n_layers,3], R9=labels. Config rows=[in,out,activation(0=relu)]
- Always end with HALT
- Use @named tensors with LD/ST for data file slots`;

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
  const [validStatus, setValidStatus]     = useState(null);  // null | {valid, attempts, stage, errors, history}
  const [validLoading, setValidLoading]   = useState(false);

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

  const callModel = async (base, system, userMsg, setOut, setLoading, setError, maxTokens = 800, model = undefined) => {
    setLoading(true);
    setError("");
    setOut("");
    try {
      const payload = {
        messages: [
          { role: "system", content: system },
          { role: "user",   content: userMsg },
        ],
        max_tokens: maxTokens,
        stream: false,
      };
      if (model) payload.model = model;

      const r = await fetch(`${base}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      const msg = data.choices?.[0]?.message || {};
      // Ollama think models put reasoning in a separate field
      const text = msg.content || msg.reasoning || "";
      setOut(text);
      return JSON.stringify(msg);  // return full message for reasoning extraction
    } catch (e) {
      setError(`Error: ${e.message}`);
      return null;
    } finally {
      setLoading(false);
    }
  };

  const runThink = async (p) => {
    // Ollama think models need the model name; detect by port
    const isOllama = THINK_PORT === "11434" || THINK_PORT === 11434;
    const model = isOllama ? "qwen3.5:4b" : undefined;  // base model — trained nml-think GGUF has tokenizer issues
    const raw = await callModel(THINK_BASE, THINK_SYSTEM, p, setThinkRaw, setThinkLoading, setThinkError, 700, model);
    if (raw) {
      // Ollama returns {content, reasoning} — extract reasoning if content is empty
      try {
        const msg = JSON.parse(raw);
        const reasoning = msg.reasoning || msg.content || raw;
        setThinkOut(extractThinkContent(reasoning));
      } catch {
        setThinkOut(extractThinkContent(raw));
      }
    }
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
      const r = await fetch(`http://localhost:${SERVER_PORT}/execute`, {
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


  const runValidated = async (codePrompt) => {
    setValidLoading(true);
    setValidStatus(null);
    try {
      const r = await fetch(`http://localhost:${SERVER_PORT}/generate_validated`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: codePrompt,
          max_retries: 3,
          max_tokens: 600,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const result = await r.json();
      setValidStatus(result);
      if (result.code) setCodeOut(result.code);
      if (!result.valid && result.grammar_errors?.length)
        setCodeError(`Validation failed: ${result.grammar_errors[0]}`);
      if (!result.valid && result.runtime_errors?.length)
        setCodeError(`Runtime error: ${result.runtime_errors[0]}`);
      return result;
    } catch (e) {
      setCodeError(`Validation error: ${e.message}`);
      return null;
    } finally {
      setValidLoading(false);
    }
  };

  const runPipeline = async () => {
    if (!prompt.trim()) return;
    setCodeOut("");
    setCodeError("");
    setExecOut(null);
    setValidStatus(null);

    const raw = await runThink(prompt);
    if (!raw) return;

    const reasoning = extractThinkContent(raw);
    const codePrompt = `Generate NML assembly for: ${prompt}\n\nFollow this plan exactly — do not add extra operations:\n${reasoning.slice(0, 1000)}`;

    // Use validated generation — generates, validates, retries automatically
    await runValidated(codePrompt);
  };

  const sendToCode = async () => {
    if (!thinkOut) return;
    const codePrompt = `Generate NML assembly for: ${prompt}\n\nFollow this plan exactly — do not add extra operations:\n${thinkOut.slice(0, 1000)}`;
    await runValidated(codePrompt);
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
              onClick={() => { setThinkOut(""); setThinkRaw(""); setCodeOut(""); setThinkError(""); setCodeError(""); setExecOut(null); setExecError(""); setValidStatus(null); }}
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
            <Panel title="Think Model" subtitle="nml-4b-think Q6 · Architecture Planning"
              color="think" content={showThinkRaw ? thinkRaw : thinkOut}
              isCode={false} loading={thinkLoading} error={thinkError}>
              {thinkOut && (
                <button onClick={() => setShowThinkRaw(v => !v)} style={smallBtn(showThinkRaw)}>
                  {showThinkRaw ? "Show clean" : "Show raw"}
                </button>
              )}
            </Panel>

            <Panel title="Code Model"
              subtitle={
                validStatus
                  ? `nml-1.5b · ${validStatus.valid ? "Validated" : "Failed"} · ${validStatus.attempts} attempt${validStatus.attempts > 1 ? "s" : ""}`
                  : validLoading
                  ? "nml-1.5b · Validating..."
                  : "nml-1.5b-instruct-v0.10.0 · NML Assembly"
              }
              color="code" content={codeOut}
              isCode={true} loading={codeLoading || validLoading} error={codeError}>
              <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                {/* Validation badge */}
                {validStatus && (
                  <span style={{
                    fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
                    background: validStatus.valid ? "#2ea04322" : "#f8514922",
                    color: validStatus.valid ? "#3fb950" : "#f85149",
                    border: `1px solid ${validStatus.valid ? "#2ea04366" : "#f8514966"}`,
                  }}>
                    {validStatus.valid ? "VALID" : "INVALID"}
                    {validStatus.attempts > 1 && ` (retry ${validStatus.attempts - 1})`}
                  </span>
                )}
                {codeOut && (
                  <button onClick={() => navigator.clipboard?.writeText(codeOut)} style={smallBtn(false)}>
                    Copy
                  </button>
                )}
                {codeOut && (
                  <button onClick={() => runExecute(codeOut)} disabled={execLoading}
                    style={{ ...smallBtn(false), color: execLoading ? "#8b949e" : "#b392f0", borderColor: "#6e40c9" }}>
                    {execLoading ? "Running..." : "Execute"}
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
