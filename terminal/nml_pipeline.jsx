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
// Think model: proxied through NML server to avoid CORS issues
const THINK_PORT  = _getThinkPort();
const THINK_BASE  = `http://localhost:${SERVER_PORT}/v1/think`;
const CODE_BASE   = API_BASE;

const THINK_SYSTEM = `You are an NML v0.10.0 (Neural Machine Language) architecture planner.
NML is an 89-opcode tensor register machine with 32 registers (R0-RV).
Translate the user's request into the SIMPLEST possible NML program. Do not add features, layers, or complexity beyond what was asked.

CRITICAL: Match complexity to the request.
- "add two tensors" → 3-4 instructions (LD, MADD, ST, HALT)
- "dot product and scale" → 4-5 instructions (LD, SDOT, SCLR, ST, HALT)
- "train a network" → use TRAIN+INFER (only when training is explicitly requested)
Do NOT use TRAIN, INFER, or training opcodes unless the user specifically asks to train.

Your plan MUST include:

1. DATA FILE — every named tensor the program needs, in .nml.data format.
   For each tensor explain what it holds and why.
   Use exact @name shape=rows,cols format so it can be copied directly.
   Example:
   @input      shape=1,4   ; one sample, 4 features (temperature, pressure, vibration, speed)
   @weights    shape=4,1   ; learned weights connecting 4 inputs to 1 output
   @bias       shape=1,1   ; output bias (initialize to 0.0)
   @target     shape=1,1   ; expected output for training (e.g., failure probability)

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
  TRAIN: config-driven training (TRAIN RU [@input @labels]) — RV=arch, RU=config[epochs,lr,optimizer,print_every,patience,min_delta]
  INFER: forward pass only (INFER Rd R_input) — uses RV arch + R1-R6 weights from TRAIN
  LOSS: loss function (Rd Rpred Rlabel #type)
  BKWD: backward pass | WUPD: weight update (Rw Rgrad Rlr)
  CONV/POOL/ATTN/NORM/EMBD: vision/transformer ops
  HALT: must be last instruction

Data type constraints:
- Most opcodes operate on f32 (float) tensors
- MOD requires INTEGER values (casts to int internally) — do NOT use with floats
- ITOF converts int→float, FTOI converts float→int
- EMBD Rindex values must be integers (token IDs)
- LOSS #type: 0=MSE, 1=MAE, 2=cross-entropy
- TRAIN/INFER: RV arch descriptor [n_layers, h1, act1, h2, act2, ...]. Activation: 0=relu, 1=sigmoid, 2=tanh
- TRAIN config (RU): [epochs, lr, optimizer(0=SGD,1=Adam), print_every, patience, min_delta]

Shape constraints:
- MMUL: [M,K]×[K,N]→[M,N] — inner dims must match
- CONV 2D: input[H,W] kernel[KH,KW]. 4D: input[N,C,H,W] kernel[Co,Ci,KH,KW]
- ATTN: Q[S,D] K[S,D] must match. V[S,Dv] optional
- LOSS: pred and label must have same shape [N,O]
- TRAIN: RV=[n_layers, h1, act1, h2, act2, ...], R1-R6=weight/bias pairs, R0=input, R9=labels
- SOFT: softmax over last dim, rows sum to 1
- BN: data[N,F], gamma[F], beta[F]

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
Training:  TRAIN RU [@data @labels] | INFER Rd R_input | LOSS Rd Rpred Rlabel #type | BKWD Rgrad Ract Rloss
           WUPD Rw Rgrad Rlr | BN Rd Rs Rgamma Rbeta | DROP Rd Rs #rate | WDECAY Rd #lambda
Backward:  RELUBK/SIGMBK/TANHBK/GELUBK/SOFTBK Rd Rgrad Rin (3 ops)
           MMULBK Rd_di Rd_dw Rgrad Rin Rw | CONVBK Rd_di Rd_dk Rgrad Rin Rk (5 ops)
Control:   HALT | JUMP #off | JMPT #off | JMPF #off | LOOP Rs|#n | ENDP | CALL #off | RET
General:   SYS Rd #code | FILL Rd #rows #cols #val | CMP Rs1 Rs2 | CMPI Rd Rs #imm
Signal:    FFT Rd Rs Rtwiddle | FILT Rd Rs Rkernel
Tree:      LEAF Rd #val | CMPF Rd Rs #feat #thresh

Data type constraints:
- Most opcodes operate on f32 (float) tensors
- MOD requires INTEGER values — do NOT use with floats
- ITOF converts int→float, FTOI converts float→int
- EMBD Rindex values must be integers

Shape constraints:
- MMUL: [M,K]×[K,N]→[M,N] — inner dims must match
- CONV 4D: input[N,C,H,W] kernel[Co,Ci,KH,KW] (Ci must match C)
- ATTN: Q[S,D] K[S,D] must match
- LOSS: pred and label must have same shape
- TRAIN: RV=[n_layers,h1,act1,h2,act2,...], RU=[epochs,lr,optimizer,print_every,patience,min_delta], R0=data, R9=labels

Rules:
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
  { ver: "v0.10",  total: 89, opcodes: "BN DROP", note: "Batch normalization, inverted dropout. TNET/TNDEEP now redirect to TRAIN internally" },
];

function highlightNML(text) {
  return text.split("\n").map((line, i) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith(";")) {
      let commentColor = "#6a737d";
      if (trimmed.includes("✓")) commentColor = "#3fb950";
      else if (trimmed.includes("✗")) commentColor = "#f85149";
      else if (trimmed.startsWith("; ── Attempt")) commentColor = "#8b949e";
      else if (trimmed === "; validating..." || trimmed.includes("retrying")) commentColor = "#e3b341";
      return <div key={i} style={{ color: commentColor, minHeight: "1.4em", fontWeight: trimmed.startsWith("; ──") ? 600 : 400 }}>{line || " "}</div>;
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
  const full = text.match(/<think>([\s\S]*?)<\/think>/);
  if (full) return full[1].trim();
  const closing = text.match(/^([\s\S]*?)<\/think>/);
  if (closing) return closing[1].trim();
  return text.trim();
}

function generateDataTemplate(adviseText, thinkText, promptText) {
  const all = [adviseText || "", thinkText || "", promptText || ""].join("\n");
  const tensors = [];
  const seen = new Set();
  const MAX_DIM = 100;

  // Extract explicit @name shape=R,C lines from advisor/think output
  const explicitRe = /@(\w+)\s+shape\s*=\s*(\d+)\s*,\s*(\d+)/g;
  let m;
  while ((m = explicitRe.exec(all)) !== null) {
    if (!seen.has(m[1])) {
      seen.add(m[1]);
      tensors.push({ name: m[1], rows: Math.min(parseInt(m[2]), MAX_DIM), cols: Math.min(parseInt(m[3]), MAX_DIM) });
    }
  }

  // Extract from table/description patterns like "R0 | Training data | (N, 29)"
  const tableRe = /(?:R\d+|@(\w+))\s*[|:—–-]\s*([^|(\n]+?)\s*[|]\s*\(?(\d+)\s*,\s*(\d+)\)?/g;
  while ((m = tableRe.exec(all)) !== null) {
    const name = m[1] || m[2].trim().toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
    if (name && !seen.has(name)) {
      seen.add(name);
      tensors.push({ name, rows: Math.min(parseInt(m[3]), MAX_DIM), cols: Math.min(parseInt(m[4]), MAX_DIM), desc: m[2].trim() });
    }
  }

  // Extract from prose like "shape=(6,3)" or "[100,29]" near a @name or keyword
  const shapeRe = /(?:@(\w+)|(\w+(?:_\w+)*))\s*(?:shape|Shape)?\s*[=:]\s*[\[(](\d+)\s*,\s*(\d+)[\])]/g;
  while ((m = shapeRe.exec(all)) !== null) {
    const name = m[1] || m[2];
    if (name && !seen.has(name) && !/^[Rr]\d/.test(name)) {
      seen.add(name);
      tensors.push({ name, rows: Math.min(parseInt(m[3]), MAX_DIM), cols: Math.min(parseInt(m[4]), MAX_DIM) });
    }
  }

  if (tensors.length === 0) return null;

  const lines = ["; .nml.data — generated from Advisor/Think output", ";"];

  // Categorize tensors
  const isWeight = (n) => /^[wb]\d|weight|bias|gamma|beta|kernel/i.test(n);
  const isLabel = (n) => /label|target/i.test(n);
  const isConfig = (n) => /config/i.test(n);

  const inputs = tensors.filter(t => !isWeight(t.name) && !isLabel(t.name) && !isConfig(t.name));
  const weights = tensors.filter(t => isWeight(t.name));
  const labels = tensors.filter(t => isLabel(t.name));
  const configs = tensors.filter(t => isConfig(t.name));

  const genData = (rows, cols) => {
    const count = rows * cols;
    return Array.from({ length: count }, () => (Math.random() * 0.4 - 0.2).toFixed(2)).join(",");
  };

  const genZeros = (rows, cols) => {
    return Array(rows * cols).fill("0.0").join(",");
  };

  if (inputs.length) {
    lines.push("; ── Input Data ──");
    for (const t of inputs) {
      if (t.desc) lines.push(`; ${t.desc}`);
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genData(t.rows, t.cols)}`);
    }
  }
  if (labels.length) {
    lines.push("", "; ── Labels / Targets ──");
    for (const t of labels) {
      if (t.desc) lines.push(`; ${t.desc}`);
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genZeros(t.rows, t.cols)}`);
    }
  }
  if (weights.length) {
    lines.push("", "; ── Model Weights (initialize small random or zeros) ──");
    for (const t of weights) {
      if (t.desc) lines.push(`; ${t.desc}`);
      const data = /^b\d|bias/i.test(t.name) ? genZeros(t.rows, t.cols) : genData(t.rows, t.cols);
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${data}`);
    }
  }
  if (configs.length) {
    lines.push("", "; ── Config ──");
    for (const t of configs) {
      if (t.desc) lines.push(`; ${t.desc}`);
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genZeros(t.rows, t.cols)}`);
    }
  }

  return lines.join("\n");
}

function generateDataFromCode(nmlCode, contextText) {
  const tensorRefs = new Set();
  const re = /(?:LD|ST)\s+R\d+\s+@(\w+)/g;
  let m;
  while ((m = re.exec(nmlCode)) !== null) tensorRefs.add(m[1]);
  if (tensorRefs.size === 0) return null;

  const MAX_DIM = 100; // cap dimensions so data is fully executable

  // Build a shape lookup from advisor/think context
  const shapeLookup = {};
  if (contextText) {
    const shapeRe = /@(\w+)\s+shape\s*=\s*(\d+)\s*,\s*(\d+)/g;
    while ((m = shapeRe.exec(contextText)) !== null) {
      shapeLookup[m[1]] = {
        rows: Math.min(parseInt(m[2]), MAX_DIM),
        cols: Math.min(parseInt(m[3]), MAX_DIM),
      };
    }
    const proseRe = /(?:@(\w+)|(\w+(?:_\w+)*))\s*(?:shape|Shape)?\s*[=:]\s*[\[(](\d+)\s*,\s*(\d+)[\])]/g;
    while ((m = proseRe.exec(contextText)) !== null) {
      const name = m[1] || m[2];
      if (name && !shapeLookup[name]) {
        shapeLookup[name] = {
          rows: Math.min(parseInt(m[3]), MAX_DIM),
          cols: Math.min(parseInt(m[4]), MAX_DIM),
        };
      }
    }
  }

  const isWeight = (n) => /^[wb]\d|weight|bias|gamma|beta|kernel/i.test(n);
  const isLabel = (n) => /label|target|train_y/i.test(n);
  const isConfig = (n) => /config/i.test(n);
  const isOutput = (n) => /output|result|prediction|pred/i.test(n);

  const genData = (rows, cols) => {
    const count = rows * cols;
    return Array.from({ length: count }, () => (Math.random() * 0.4 - 0.2).toFixed(2)).join(",");
  };
  const genZeros = (rows, cols) => {
    return Array(rows * cols).fill("0.0").join(",");
  };

  const inputs = [], weights = [], labels = [], configs = [], outputs = [];
  for (const name of tensorRefs) {
    const shape = shapeLookup[name] || { rows: 4, cols: 4 };
    const entry = { name, ...shape };
    if (isOutput(name))      outputs.push(entry);
    else if (isWeight(name)) weights.push(entry);
    else if (isLabel(name))  labels.push(entry);
    else if (isConfig(name)) configs.push(entry);
    else                     inputs.push(entry);
  }

  const lines = ["; .nml.data — generated from NML code tensor references", ";"];

  if (inputs.length) {
    lines.push("; ── Input Data ──");
    for (const t of inputs)
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genData(t.rows, t.cols)}`);
  }
  if (labels.length) {
    lines.push("", "; ── Labels / Targets ──");
    for (const t of labels)
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genZeros(t.rows, t.cols)}`);
  }
  if (weights.length) {
    lines.push("", "; ── Model Weights ──");
    for (const t of weights) {
      const data = /^b\d|bias/i.test(t.name) ? genZeros(t.rows, t.cols) : genData(t.rows, t.cols);
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${data}`);
    }
  }
  if (configs.length) {
    lines.push("", "; ── Config ──");
    for (const t of configs)
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genZeros(t.rows, t.cols)}`);
  }
  if (outputs.length) {
    lines.push("", "; ── Output (placeholder) ──");
    for (const t of outputs)
      lines.push(`@${t.name}  shape=${t.rows},${t.cols}  dtype=f32  data=${genZeros(t.rows, t.cols)}`);
  }

  return lines.join("\n");
}

// ── Panel component ───────────────────────────────────────────────────────────
function Panel({ title, subtitle, color, content, isCode, loading, error, children }) {
  const borderColor = color === "think" ? "#388bfd" : color === "advise" ? "#d29922" : "#3fb950";
  const labelColor  = color === "think" ? "#79b8ff" : color === "advise" ? "#e3b341" : "#56d364";

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
        {loading && !content && (
          <div style={{ color: "#8b949e", fontStyle: "italic" }}>Generating...</div>
        )}
        {error && (
          <div style={{ color: "#f85149" }}>{error}</div>
        )}
        {content && (
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
      width: 280, flexShrink: 0, background: "#161b22",
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
          {expanded[cat.name] && cat.prompts.map((p, i) => {
            const isObj = typeof p === "object";
            const label = isObj ? p.label : p;
            const scenario = isObj ? p.scenario : null;
            const text = isObj ? p.text : p;
            return (
              <button
                key={i}
                onClick={() => onSelect(text)}
                style={{
                  width: "100%", textAlign: "left",
                  padding: "6px 14px 6px 26px",
                  background: "none", border: "none", cursor: "pointer",
                  color: "#8b949e", fontSize: 11, lineHeight: 1.4,
                  borderBottom: "1px solid #21262d",
                  transition: "background 0.15s",
                }}
                onMouseEnter={e => { e.currentTarget.style.background="#21262d"; }}
                onMouseLeave={e => { e.currentTarget.style.background="none"; }}
              >
                <div style={{ color: "#c9d1d9", fontWeight: 500 }}>{label}</div>
                {scenario && (
                  <div style={{ color: "#6e7681", fontSize: 10, marginTop: 2, lineHeight: 1.3 }}>{scenario}</div>
                )}
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ── Main app ──────────────────────────────────────────────────────────────────
export default function NMLPipeline() {
  const [prompt, setPrompt]               = useState("");
  const [adviseOut, setAdviseOut]         = useState("");
  const [adviseLoading, setAdviseLoading] = useState(false);
  const [adviseError, setAdviseError]     = useState("");
  const [adviseMeta, setAdviseMeta]       = useState(null);
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

; --- Training example (TRAIN+INFER) ---
; N=4 samples, K=2 features → training_data shape=4,2
; training_labels shape=4,1  (regression target)
; Architecture: 2 layers (hidden=4 ReLU, output=1 linear)
; RV = [2, 4, 0, 1, 4]  →  ALLC RV [5] 2,4,0,1,4
; RU = [1000, 0.01, 1, 0, 0, 0]  →  ALLC RU [6] 1000,0.01,1,0,0,0
@training_data    shape=4,2  dtype=f32  data=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8
@training_labels  shape=4,1  dtype=f32  data=0.3,0.7,1.1,1.5`;

  const runAdvise = async (desc) => {
    setAdviseLoading(true);
    setAdviseError("");
    setAdviseOut("");
    setAdviseMeta(null);
    try {
      const r = await fetch(`http://localhost:${SERVER_PORT}/advise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: desc }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      if (data.llm_error) {
        setAdviseError(`LLM fallback: ${data.llm_error}`);
      }
      if (data.advice) {
        setAdviseOut(data.advice);
        setAdviseMeta({ source: data.source, model: data.model, types: data.problem_types_matched });
      } else if (data.recommendation) {
        const lines = [
          `**Problem Type:** ${data.problem_type}`,
          `**Recommended:** ${data.recommendation}`,
          `**Why:** ${data.why}`,
          `**NML Pattern:** ${data.nml_pattern}`,
          `**Opcodes:** ${(data.nml_opcodes || []).join(", ")}`,
          `**Lesson:** ${data.lesson_ref}`,
        ];
        if (data.sample) lines.push(`**Sample:** ${data.sample}`);
        if (data.alternatives?.length) {
          lines.push("", "**Alternatives:**");
          data.alternatives.forEach(a => lines.push(`  • ${a.name} — ${a.when} (${a.complexity})`));
        }
        setAdviseOut(lines.join("\n"));
        setAdviseMeta({ source: data.source, types: [data.problem_type] });
      } else {
        setAdviseOut(JSON.stringify(data, null, 2));
      }
      return data;
    } catch (e) {
      setAdviseError(`Error: ${e.message}`);
      return null;
    } finally {
      setAdviseLoading(false);
    }
  };

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
    const raw = await callModel(THINK_BASE, THINK_SYSTEM, p, setThinkRaw, setThinkLoading, setThinkError, 1500, model);
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
    const MAX_RETRIES = 3;
    const MAX_TOKENS = 600;
    setValidLoading(true);
    setValidStatus(null);
    setCodeOut("");
    setCodeError("");

    let log = "";
    const addLog = (line) => { log += line + "\n"; setCodeOut(log); };

    const messages = [
      { role: "system", content: CODE_SYSTEM },
      { role: "user",   content: codePrompt },
    ];

    const history = [];

    try {
      for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        // ── Generate ──
        addLog(`; ── Attempt ${attempt}/${MAX_RETRIES} ── Generating...`);

        const genR = await fetch(`${CODE_BASE}/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages, max_tokens: MAX_TOKENS, stream: false }),
        });
        if (!genR.ok) throw new Error(`Code model HTTP ${genR.status}`);
        const genData = await genR.json();
        const rawText = genData.choices?.[0]?.message?.content || "";

        let code = rawText.replace(/<think>[\s\S]*?<\/think>/g, "");
        code = code.replace(/```[a-z]*\n?|```/g, "").trim();

        log = log.replace(/; ── Attempt \d+\/\d+ ── Generating\.\.\.\n$/, "");
        addLog(`; ── Attempt ${attempt}/${MAX_RETRIES} ──`);
        addLog(code);

        // ── Validate grammar ──
        addLog(`; validating...`);

        const valR = await fetch(`http://localhost:${SERVER_PORT}/validate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ nml_program: code }),
        });
        if (!valR.ok) throw new Error(`Validate HTTP ${valR.status}`);
        const valResult = await valR.json();

        if (valResult.valid) {
          log = log.replace(/; validating\.\.\.\n$/, "");
          addLog(`; ✓ valid`);
          setValidStatus({ valid: true, attempts: attempt, code, history });
          setCodeOut(code);
          return { valid: true, code, attempts: attempt };
        }

        const errors = valResult.errors || [];
        const errorsText = errors.map(e => e.message || JSON.stringify(e)).join("; ");

        // Build structured error feedback for the LLM
        // Extract the failing opcodes so we can include their correct schemas
        const failingOps = new Set();
        const structuredFeedback = errors.map(e => {
          let line = `Line ${e.line}: [${e.type || e.errorType}] ${e.message}`;
          if (e.source) {
            line += `\n  Source: ${e.source}`;
            const op = e.source.trim().split(/\s+/)[0]?.toUpperCase();
            if (op && NML_OPCODES.has(op)) failingOps.add(op);
          }
          if (e.fix) line += `\n  Fix:    ${e.fix}`;
          return line;
        }).join("\n");

        // Build a mini opcode reference for just the failing opcodes
        const OPCODE_SCHEMAS = {
          SOFT: "SOFT Rd Rs  (2 registers: dest, source — softmax over last dim)",
          RELU: "RELU Rd Rs  (2 registers)", SIGM: "SIGM Rd Rs  (2 registers)",
          TANH: "TANH Rd Rs  (2 registers)", GELU: "GELU Rd Rs  (2 registers)",
          MMUL: "MMUL Rd Rs1 Rs2  (3 registers, NO immediates)",
          MADD: "MADD Rd Rs1 Rs2  (3 registers)", MSUB: "MSUB Rd Rs1 Rs2  (3 registers)",
          EMUL: "EMUL Rd Rs1 Rs2  (3 registers)", EDIV: "EDIV Rd Rs1 Rs2  (3 registers)",
          SADD: "SADD Rd Rs #imm  (2 regs + 1 immediate)", SSUB: "SSUB Rd Rs #imm",
          SCLR: "SCLR Rd Rs #imm", SDIV: "SDIV Rd Rs #imm",
          SDOT: "SDOT Rd Rs1 Rs2  (3 registers)",
          CONV: "CONV Rd Rs Rkernel [#stride] [#pad]", POOL: "POOL Rd Rs [#size] [#stride]",
          ATTN: "ATTN Rd Rq Rk [Rv]  (3-4 registers)",
          NORM: "NORM Rd Rs Rgamma Rbeta  (4 registers)",
          LOSS: "LOSS Rd Rpred Rlabel [#type]  (3 regs + optional imm)",
          BKWD: "BKWD Rgrad Ract Rloss  (3 registers)",
          WUPD: "WUPD Rw Rgrad Rlr  (3 registers)",
          TRAIN: "TRAIN RU [@data @labels]  (1 reg + optional @refs)",
          INFER: "INFER Rd [R_input]  (1-2 registers)",
          SOFTBK: "SOFTBK Rd Rgrad Rin  (3 registers)",
          RELUBK: "RELUBK Rd Rgrad Rin  (3 registers)",
          BN: "BN Rd Rs [Rgamma] [Rbeta]  (2-4 registers)",
          DROP: "DROP Rd Rs [#rate]  (2 regs + optional imm)",
          CLMP: "CLMP Rd Rs #min #max  (2 regs + 2 immediates)",
          CMPI: "CMPI Rd Rs #imm  (2 regs + 1 immediate)",
          CMP: "CMP Rd Rs  (2 registers)",
          RDUC: "RDUC Rd Rs [#dim] [#mode]  (2 regs + optional imms)",
        };
        const schemaHints = [...failingOps]
          .filter(op => OPCODE_SCHEMAS[op])
          .map(op => `  ${OPCODE_SCHEMAS[op]}`)
          .join("\n");

        log = log.replace(/; validating\.\.\.\n$/, "");
        addLog(`; ✗ ${errorsText}`);

        history.push({ attempt, stage: "grammar", code, errors: errorsText });

        if (attempt < MAX_RETRIES) {
          addLog(`; retrying with error feedback...\n`);
          messages.push({ role: "assistant", content: rawText });
          const schemaSection = schemaHints
            ? `\n\nCorrect operand schemas for failing opcodes:\n${schemaHints}\n`
            : "";
          messages.push({ role: "user", content:
            `Your NML code had errors. Fix them.\n\nPrevious code:\n${code}\n\nErrors:\n${structuredFeedback}${schemaSection}\nFix ONLY the errors listed above. Output only the corrected NML code.`
          });
        }
      }

      const last = history[history.length - 1] || {};
      setValidStatus({
        valid: false, attempts: MAX_RETRIES, code: last.code || "",
        stage: "grammar", grammar_errors: [last.errors || ""], history,
      });
      setCodeError(`Validation failed after ${MAX_RETRIES} attempts`);
      return null;
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

    // Stage 0: Advise — get ML algorithm recommendation
    const adviceData = await runAdvise(prompt);
    let adviceContext = "";
    if (adviceData) {
      if (adviceData.advice) {
        adviceContext = adviceData.advice.slice(0, 600);
      } else if (adviceData.recommendation) {
        adviceContext = `Algorithm: ${adviceData.recommendation}. ${adviceData.why} NML pattern: ${adviceData.nml_pattern}`;
      }
    }

    // Stage 1: Think — enrich with advisor context
    const thinkPrompt = adviceContext
      ? `${prompt}\n\nML Advisor recommendation:\n${adviceContext}`
      : prompt;
    const raw = await runThink(thinkPrompt);
    if (!raw) return;

    // Stage 2: Code — validated generation
    const reasoning = extractThinkContent(raw);
    const codePrompt = `Generate NML assembly for: ${prompt}\n\nFollow this plan exactly — do not add extra operations:\n${reasoning.slice(0, 1000)}`;
    const codeResult = await runValidated(codePrompt);

    // Stage 3: Auto-generate data template from code tensor references
    if (codeResult?.valid && codeResult.code) {
      const contextText = [adviseOut || "", raw || "", prompt].join("\n");
      const tpl = generateDataFromCode(codeResult.code, contextText);
      if (tpl) {
        setDataInput(tpl);
        setShowData(true);
      }
    }
  };

  const runAdvisePipeline = async () => {
    if (!prompt.trim()) return;
    await runAdvise(prompt);
  };

  const sendToCode = async () => {
    if (!thinkOut) return;
    const codePrompt = `Generate NML assembly for: ${prompt}\n\nFollow this plan exactly — do not add extra operations:\n${thinkOut.slice(0, 1000)}`;
    const codeResult = await runValidated(codePrompt);
    if (codeResult?.valid && codeResult.code) {
      const contextText = [adviseOut || "", thinkOut || "", prompt].join("\n");
      const tpl = generateDataFromCode(codeResult.code, contextText);
      if (tpl) {
        setDataInput(tpl);
        setShowData(true);
      }
    }
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

  // Drawer state: null = all closed, "advise" or "think" = that drawer open
  const [openDrawer, setOpenDrawer] = useState(null);
  const toggleDrawer = (key) => setOpenDrawer(prev => prev === key ? null : key);

  return (
    <div style={{
      display: "flex", height: "100vh", background: "#0d1117",
      fontFamily: "ui-monospace, monospace", color: "#e6edf3", overflow: "hidden",
    }}>
      {/* Left sidebar: Prompt Library */}
      <PromptLibrary onSelect={handleSelect} />

      {/* Main area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Top bar */}
        <div style={{
          padding: "10px 20px", borderBottom: "1px solid #30363d",
          background: "#161b22", display: "flex", alignItems: "center", gap: 12,
        }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#e6edf3" }}>NML Pipeline</div>
            <div style={{ fontSize: 11, color: "#8b949e" }}>Advise → Think → Code → Execute &nbsp;·&nbsp; 89 opcodes &nbsp;·&nbsp; v0.10.0</div>
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
          padding: "12px 20px", borderBottom: "1px solid #30363d",
          background: "#161b22", display: "flex", flexDirection: "column", gap: 8,
        }}>
          <textarea
            ref={textareaRef}
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you want to build in NML... (Ctrl+Enter to run pipeline)"
            rows={2}
            style={{
              width: "100%", background: "#0d1117", color: "#e6edf3",
              border: "1px solid #30363d", borderRadius: 6,
              padding: "8px 12px", fontSize: 13, fontFamily: "inherit",
              resize: "none", outline: "none", boxSizing: "border-box",
            }}
          />
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button onClick={runPipeline} disabled={!prompt.trim() || adviseLoading || thinkLoading || codeLoading}
              style={btnStyle("#388bfd", !prompt.trim() || adviseLoading || thinkLoading || codeLoading)}>
              ▶ Pipeline
            </button>
            <button onClick={runAdvisePipeline} disabled={!prompt.trim() || adviseLoading}
              style={btnStyle("#d29922", !prompt.trim() || adviseLoading)}>
              Advise
            </button>
            <button onClick={() => runThink(prompt)} disabled={!prompt.trim() || thinkLoading}
              style={btnStyle("#1f6feb", !prompt.trim() || thinkLoading)}>
              Think
            </button>
            <button
              onClick={() => { setAdviseOut(""); setAdviseError(""); setAdviseMeta(null); setThinkOut(""); setThinkRaw(""); setCodeOut(""); setThinkError(""); setCodeError(""); setExecOut(null); setExecError(""); setValidStatus(null); setDataInput(""); }}
              style={{ ...btnStyle("#21262d", false), color: "#8b949e", marginLeft: "auto" }}>
              Clear
            </button>
          </div>
        </div>

        {/* Main content: vertical drawer tabs | drawer content | code+data+exec */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

          {/* Vertical drawer tab strip */}
          <div style={{
            display: "flex", flexDirection: "column", background: "#161b22",
            borderRight: "1px solid #30363d", flexShrink: 0,
          }}>
            {[
              { key: "advise", label: "Advisor", color: "#d29922", dot: adviseLoading ? "#e3b341" : adviseError ? "#f85149" : adviseOut ? "#d29922" : "#30363d" },
              { key: "think",  label: "Think",   color: "#388bfd", dot: thinkLoading ? "#e3b341" : thinkError ? "#f85149" : thinkOut ? "#388bfd" : "#30363d" },
            ].map(tab => (
              <button key={tab.key} onClick={() => toggleDrawer(tab.key)} style={{
                writingMode: "vertical-rl", textOrientation: "mixed",
                padding: "14px 8px", background: openDrawer === tab.key ? "#0d1117" : "none",
                border: "none", borderRight: openDrawer === tab.key ? `2px solid ${tab.color}` : "2px solid transparent",
                cursor: "pointer", color: openDrawer === tab.key ? tab.color : "#8b949e",
                fontSize: 12, fontWeight: 600, display: "flex", alignItems: "center", gap: 8,
                transition: "all 0.15s",
              }}>
                <div style={{
                  width: 7, height: 7, borderRadius: "50%", background: tab.dot,
                  boxShadow: (tab.key === "advise" ? adviseLoading : thinkLoading) ? "0 0 6px #e3b341" : "none",
                }} />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Drawer content panel (slides open when a tab is active) */}
          {openDrawer && (
            <div style={{
              width: 400, flexShrink: 0, background: "#0d1117",
              borderRight: "1px solid #30363d", display: "flex", flexDirection: "column",
              overflow: "hidden",
            }}>
              {/* Drawer header */}
              <div style={{
                padding: "10px 14px", background: "#161b22",
                borderBottom: `1px solid ${openDrawer === "advise" ? "#d2992233" : "#388bfd33"}`,
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <div style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: openDrawer === "advise"
                    ? (adviseLoading ? "#e3b341" : adviseError ? "#f85149" : "#d29922")
                    : (thinkLoading ? "#e3b341" : thinkError ? "#f85149" : "#388bfd"),
                }} />
                <div style={{ flex: 1 }}>
                  <div style={{
                    color: openDrawer === "advise" ? "#e3b341" : "#79b8ff",
                    fontWeight: 700, fontSize: 13,
                  }}>
                    {openDrawer === "advise" ? "Advisor" : "Think"}
                  </div>
                  <div style={{ color: "#8b949e", fontSize: 11 }}>
                    {openDrawer === "advise"
                      ? (adviseMeta ? `${adviseMeta.source === "llm" ? adviseMeta.model || "Cloud LLM" : "KB"} · ${(adviseMeta.types || []).join(", ")}` : "Algorithm Selection")
                      : "nml-4b-think · Architecture Planning"
                    }
                  </div>
                </div>
                {/* Action buttons */}
                <div style={{ display: "flex", gap: 6 }}>
                  {openDrawer === "advise" && adviseOut && (
                    <>
                      <button onClick={() => navigator.clipboard?.writeText(adviseOut)} style={smallBtn(false)}>Copy</button>
                      <button onClick={() => {
                        const thinkPrompt = `${prompt}\n\nML Advisor recommendation:\n${adviseOut.slice(0, 600)}`;
                        runThink(thinkPrompt);
                        setOpenDrawer("think");
                      }} disabled={thinkLoading}
                        style={{ ...smallBtn(false), color: "#79b8ff", borderColor: "#388bfd" }}>
                        → Think
                      </button>
                    </>
                  )}
                  {openDrawer === "think" && thinkOut && (
                    <>
                      <button onClick={() => navigator.clipboard?.writeText(thinkOut)} style={smallBtn(false)}>Copy</button>
                      <button onClick={() => setShowThinkRaw(v => !v)} style={smallBtn(showThinkRaw)}>
                        {showThinkRaw ? "Clean" : "Raw"}
                      </button>
                      <button onClick={() => { sendToCode(); setOpenDrawer(null); }} disabled={codeLoading}
                        style={{ ...smallBtn(false), color: "#56d364", borderColor: "#3fb950" }}>
                        → Code
                      </button>
                    </>
                  )}
                </div>
              </div>
              {/* Drawer body */}
              <div style={{
                flex: 1, overflowY: "auto", padding: 14, fontFamily: "monospace",
                fontSize: 12, lineHeight: 1.6, color: "#e6edf3", whiteSpace: "pre-wrap",
              }}>
                {openDrawer === "advise" && (
                  <>
                    {adviseLoading && !adviseOut && <span style={{ color: "#8b949e", fontStyle: "italic" }}>Generating...</span>}
                    {adviseError && <span style={{ color: "#f85149" }}>{adviseError}</span>}
                    {adviseOut && <span>{adviseOut}</span>}
                    {!adviseLoading && !adviseError && !adviseOut && <span style={{ color: "#30363d" }}>Run Advise to get algorithm recommendations...</span>}
                  </>
                )}
                {openDrawer === "think" && (
                  <>
                    {thinkLoading && !thinkOut && <span style={{ color: "#8b949e", fontStyle: "italic" }}>Generating...</span>}
                    {thinkError && <span style={{ color: "#f85149" }}>{thinkError}</span>}
                    {(showThinkRaw ? thinkRaw : thinkOut) && <span>{showThinkRaw ? thinkRaw : thinkOut}</span>}
                    {!thinkLoading && !thinkError && !thinkOut && <span style={{ color: "#30363d" }}>Run Think to plan the architecture...</span>}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Code + Data side by side, Execution below */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            {/* Top row: Code (left) + Data (right) */}
            <div style={{ flex: 1, display: "flex", gap: 0, padding: "12px 16px 6px", overflow: "hidden" }}>
              {/* Code panel */}
              <Panel title="Code"
                subtitle={
                  validStatus
                    ? `nml-1.5b · ${validStatus.valid ? "Validated" : "Failed"} · ${validStatus.attempts} attempt${validStatus.attempts > 1 ? "s" : ""}`
                    : validLoading
                    ? "nml-1.5b · Generate → Validate → Retry..."
                    : "nml-1.5b-instruct · NML Assembly"
                }
                color="code" content={codeOut}
                isCode={true} loading={codeLoading && !validLoading} error={codeError}>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  {validStatus && (
                    <span style={{
                      fontSize: 11, fontWeight: 700, padding: "2px 8px", borderRadius: 4,
                      background: validStatus.valid ? "#2ea04322" : "#f8514922",
                      color: validStatus.valid ? "#3fb950" : "#f85149",
                      border: `1px solid ${validStatus.valid ? "#2ea04366" : "#f8514966"}`,
                    }}>
                      {validStatus.valid ? "VALID" : "INVALID"}
                      {validStatus.attempts > 1 && ` (${validStatus.attempts - 1})`}
                    </span>
                  )}
                  {codeOut && (
                    <button onClick={() => navigator.clipboard?.writeText(codeOut)} style={smallBtn(false)}>Copy</button>
                  )}
                  {codeOut && (
                    <button onClick={() => runExecute(codeOut)} disabled={execLoading}
                      style={{ ...smallBtn(false), color: execLoading ? "#8b949e" : "#b392f0", borderColor: "#6e40c9" }}>
                      {execLoading ? "Running..." : "Execute"}
                    </button>
                  )}
                </div>
              </Panel>

              {/* Data panel (right side) */}
              <div style={{
                width: 360, flexShrink: 0, marginLeft: 10,
                display: "flex", flexDirection: "column",
                border: "1px solid #3fb95033", borderRadius: 8,
                overflow: "hidden", background: "#0d1117",
              }}>
                {/* Data header */}
                <div style={{
                  padding: "10px 14px", borderBottom: "1px solid #3fb95033",
                  background: "#161b22", display: "flex", alignItems: "center", gap: 8,
                }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: dataInput ? "#3fb950" : "#30363d",
                  }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ color: "#56d364", fontWeight: 700, fontSize: 13 }}>Data</div>
                    <div style={{ color: "#8b949e", fontSize: 11 }}>.nml.data tensors</div>
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    {(adviseOut || thinkOut || codeOut) && (
                      <button onClick={() => {
                        const contextText = [adviseOut || "", thinkOut || "", prompt].join("\n");
                        const tpl = codeOut
                          ? (generateDataFromCode(codeOut, contextText) || generateDataTemplate(adviseOut, thinkOut, prompt))
                          : generateDataTemplate(adviseOut, thinkOut, prompt);
                        if (tpl) { setDataInput(tpl); }
                        else { alert("No tensor shapes found. Run the pipeline or Think first."); }
                      }} style={{ ...smallBtn(false), color: "#e3b341", borderColor: "#d29922" }}>
                        Generate
                      </button>
                    )}
                    <button onClick={() => setShowDataHelp(v => !v)} style={smallBtn(showDataHelp)}>
                      {showDataHelp ? "Hide" : "Format"}
                    </button>
                    <button onClick={() => setDataInput(DATA_TEMPLATE)} style={smallBtn(false)}>
                      Template
                    </button>
                  </div>
                </div>
                {/* Data format help */}
                {showDataHelp && (
                  <pre style={{
                    margin: 0, padding: "8px 10px", background: "#161b22",
                    borderBottom: "1px solid #21262d",
                    fontSize: 10, color: "#8b949e", overflowX: "auto", maxHeight: 100,
                  }}>{DATA_TEMPLATE}</pre>
                )}
                {/* Data editor */}
                <textarea
                  value={dataInput}
                  onChange={e => setDataInput(e.target.value)}
                  placeholder={"@input  shape=1,4  dtype=f32  data=0.5,0.3,0.8,0.1\n@w1     shape=4,8  dtype=f32  data=0.1,0.2,..."}
                  style={{
                    flex: 1, width: "100%", background: "#0d1117", color: "#e6edf3",
                    border: "none", padding: "10px 14px", fontSize: 12,
                    fontFamily: "ui-monospace, monospace", lineHeight: 1.6,
                    resize: "none", outline: "none", boxSizing: "border-box",
                  }}
                />
              </div>
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
    </div>
  );
}
