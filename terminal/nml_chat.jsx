import { useState, useRef, useEffect, useCallback } from "react";

function _getPort() {
  try {
    return new URLSearchParams(window.location.search).get("port") || "8082";
  } catch { return "8082"; }
}
const NML_PORT = _getPort();
const API_BASE = `http://localhost:${NML_PORT}/v1`;
const EXEC_BASE = `http://localhost:${NML_PORT}`;

let _wasmModule = null;
let _wasmReady = false;
let _wasmError = null;

async function initWasm() {
  if (_wasmReady) return _wasmModule;
  if (_wasmError) return null;
  try {
    const script = document.createElement("script");
    script.src = "nml.js";
    document.head.appendChild(script);
    await new Promise((resolve, reject) => {
      script.onload = resolve;
      script.onerror = () => reject(new Error("Failed to load nml.js"));
    });
    _wasmModule = await globalThis.NMLModule();
    _wasmReady = true;
    return _wasmModule;
  } catch (e) {
    _wasmError = e.message;
    return null;
  }
}

async function runWasm(nmlCode, dataContent) {
  const mod = await initWasm();
  if (!mod) return { status: "WASM ERROR", errors: [_wasmError || "WASM not available"] };

  let stdout = "", stderr = "";
  const origPrint = mod.print;
  const origPrintErr = mod.printErr;
  mod.print = (t) => { stdout += t + "\n"; };
  mod.printErr = (t) => { stderr += t + "\n"; };

  try {
    mod.FS.writeFile("/tmp/_run.nml", nmlCode);
    const args = ["/tmp/_run.nml"];
    if (dataContent && dataContent.trim()) {
      mod.FS.writeFile("/tmp/_run.nml.data", dataContent);
      args.push("/tmp/_run.nml.data");
    }
    args.push("--max-cycles", "100000");
    mod.callMain(args);

    try { mod.FS.unlink("/tmp/_run.nml"); } catch (_) {}
    try { mod.FS.unlink("/tmp/_run.nml.data"); } catch (_) {}

    const registers = {};
    const memory = {};
    let section = "";
    for (const line of stdout.split("\n")) {
      if (line.includes("=== REGISTERS ===")) { section = "reg"; continue; }
      if (line.includes("=== MEMORY ===")) { section = "mem"; continue; }
      if (line.includes("=== STATS ===")) { section = ""; continue; }
      if (section && line.trim()) {
        const m = line.match(/^\s+(\w+):\s+shape=\[([^\]]+)\]\s*(dtype=\w+\s+)?data=\[([^\]]+)\]/);
        if (m) {
          const obj = { shape: m[2], data: m[4].split(",").map(s => parseFloat(s.trim())) };
          if (section === "reg") registers[m[1]] = obj;
          else memory[m[1]] = obj;
        }
      }
    }
    return { status: "OK", output: stdout, registers, memory, runtime: "wasm" };
  } catch (e) {
    return { status: "RUNTIME ERROR", errors: [e.message], output: stdout, stderr };
  } finally {
    mod.print = origPrint;
    mod.printErr = origPrintErr;
  }
}

const FONT = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Source Code Pro', monospace";
const COLORS = {
  bg: "#0a0a0f",
  panel: "#0f0f18",
  border: "#1a1a2e",
  accent: "#00ff9d",
  accentDim: "#00cc7d",
  accentBg: "rgba(0,255,157,0.06)",
  error: "#ff4444",
  warn: "#ffaa00",
  text: "#c8c8d4",
  textDim: "#5a5a72",
  userBg: "#111128",
  userBorder: "#1a1a3e",
};

const NML_OPCODES = new Set([
  "MMUL","MADD","MSUB","EMUL","EDIV","SDOT","DOT","SCLR","SDIV","SADD","SSUB",
  "RELU","SIGM","TANH","SOFT","GELU",
  "LD","ST","MOV","ALLC","RSHP","TRNS","SPLT","MERG",
  "CMPF","CMP","CMPI","JMPT","JMPF","JUMP","JMP","LOOP","ENDP",
  "CALL","RET","LEAF","TACC","SYNC","HALT","TRAP",
  "CONV","POOL","UPSC","PADZ","ATTN","NORM","EMBD",
  "RDUC","WHER","CLMP","CMPR","FFT","FILT",
  "META","FRAG","ENDF","LINK","VOTE","PROJ","DIST","GATH","SCAT","SCTR",
  "SYS","MOD","ITOF","FTOI","BNOT","SIGN","VRFY","PTCH",
  "BKWD","WUPD","LOSS","TNET","BN","DROP",
  "RELUBK","SIGMBK","TANHBK","GELUBK","SOFTBK",
  "MMULBK","CONVBK","POOLBK","NORMBK","ATTNBK","TNDEEP",
  "TRAIN","INFER","WDECAY","TLOG",
]);
const NML_SYMBOLIC = new Set([
  "×","⊕","⊖","⊗","⊘","·","∗","÷","∔","∸","⌐","σ","τ","Σ","ℊ",
  "↓","↑","←","□","⊤","⊢","⊣","⋈","≶","≺","ϟ",
  "↗","↘","→","↻","↺","∎","∑","⇒","⇐","⏸","◼","⚠",
  "⊛","⊓","⊔","⊡","⊙","‖","⊏","⊥","ϛ","⊻","⊧","⊜",
  "∿","⋐","§","◆","◇","⚖","✦","✓","⟐","⟂","⊃","⊂","⚙",
  "∇","⟳","△","⥁","⊞","≋",
  "⌐ˈ","σˈ","τˈ","ℊˈ","Σˈ","×ˈ","⊛ˈ","⊓ˈ","‖ˈ","⊙ˈ","⥁ˈ",
]);

const SYSTEM_PROMPTS = {
  classic: `You are an NML v0.10.0 code generator. NML is an 89-opcode tensor register machine with 32 registers (R0-RV).

CRITICAL RULES:
- LEAF R0 #42.0 loads constants (# prefix). LD R0 @name loads from memory (@ prefix). Never mix them.
- Every program MUST end with HALT.
- CALL #N at line X lands on X+N+1. Subroutine right after HALT: use CALL #2, NOT #3.
- JMPT/JMPF/JUMP offset math: target = current_line + offset + 1.
- No ADD, SUB, MUL, INCR, or LDI instruction exists. Use TACC for addition, EMUL for element-wise multiply, SCLR for scalar multiply.
- Increment pattern: LEAF RC #1.0 then TACC RD RD RC.
- TLOG #n sets print interval (no register). TRAIN Rs [@input] [@labels] runs config-driven training.
- INFER Rd Rs runs forward pass only (no weight update). WDECAY Rd #lambda applies weight decay.
- BN Rd Rs [Rgamma [Rbeta]] applies batch normalization (2D or 4D tensors).
- DROP Rd Rs #rate applies inverted dropout; use #0.0 at inference time to disable.

ALL 89 OPCODES:
Arithmetic: MMUL MADD MSUB EMUL EDIV SDOT/DOT SCLR SDIV SADD SSUB
Activation: RELU SIGM TANH SOFT GELU
Memory: LD ST MOV ALLC LEAF
Data flow: RSHP TRNS SPLT MERG
Compare: CMP CMPI CMPF
Control: JMPT JMPF JUMP LOOP ENDP CALL RET
System: HALT TRAP SYNC SYS MOD ITOF FTOI BNOT
Vision: CONV POOL UPSC PADZ
Transformer: ATTN NORM EMBD
Reduction: RDUC WHER CLMP CMPR
Signal: FFT FILT
M2M: META FRAG ENDF LINK PTCH SIGN VRFY VOTE PROJ DIST GATH SCAT/SCTR
Training: BKWD WUPD LOSS TNET TNDEEP TRAIN INFER WDECAY TLOG BN DROP
Backward: RELUBK SIGMBK TANHBK GELUBK SOFTBK MMULBK CONVBK POOLBK NORMBK ATTNBK

BACKWARD OPCODE PATTERNS:
- Activation backward: RELUBK Rd Rgrad Rinput (gradient through activation)
- MMULBK Rd_dinput Rd_dweight Rgrad Rinput Rweight (matmul backward, outputs 2 gradients)
- CONVBK Rd_dinput Rd_dkernel Rgrad Rinput Rkernel (conv backward, outputs 2 gradients)
- ATTNBK Rd_dq Rgrad Rq Rk Rv (attention backward, 5 args)
- Training loop: LOOP #N ... forward ... LOSS ... backward ... WUPD ... ENDP

REGISTERS: R0-R9 (general), RA (accumulator), RB (general), RC (scratch), RD (counter), RE (condition flag set by CMP/CMPI/CMPF), RF (stack pointer), RG-RI (gradients), RJ (learning rate), RK-RV (training/hive).`,

  symbolic: `You are an NML v0.10.0 code generator. ALWAYS use symbolic syntax with Greek register names. Never use classic opcode mnemonics.

CRITICAL RULES:
- ∎ ι #42.0 loads constants (# prefix). ↓ ι @name loads from memory (@ prefix). Never mix them.
- Every program MUST end with ◼.
- ⇒ #N at line X lands on X+N+1. Subroutine right after ◼: use ⇒ #2, NOT #3.
- ↗/↘/→ offset math: target = current_line + offset + 1.
- No ADD, SUB, MUL, or INCR exists. Use ∑ for addition, ⊗ for element-wise multiply, ∗ for scalar multiply.
- Increment pattern: ∎ γ #1.0 then ∑ δ δ γ.
- ⊞ (batch norm): ⊞ ι κ [λ [μ]] normalizes tensor. ≋ (dropout): ≋ ι κ #rate applies inverted dropout.

ALL 89 SYMBOLIC OPCODES:
Arithmetic: × (matmul) ⊕ (add) ⊖ (sub) ⊗ (emul) ⊘ (ediv) · (dot) ∗ (scale) ÷ (sdiv) ∔ (sadd) ∸ (ssub)
Activation: ⌐ (relu) σ (sigmoid) τ (tanh) Σ (softmax) ℊ (gelu)
Memory: ↓ (load) ↑ (store) ← (move) □ (alloc) ∎ (leaf/constant)
Data flow: ⊤ (transpose) ⊢ (split) ⊣ (merge)
Compare: ≶ (cmp) ≺/ϟ (cmpi) ⋈ (cmpf)
Control: ↗ (jmpt) ↘ (jmpf) → (jump) ↻ (loop) ↺ (endp) ⇒ (call) ⇐ (ret)
Tree: ∎ (leaf) ∑ (accumulate)
System: ◼ (halt) ⚠ (trap) ⏸ (sync) ⚙ (sys) % (mod) ⊶ (itof) ⊷ (ftoi) ¬ (bnot)
Vision: ⊛ (conv) ⊓ (pool) ⊔ (upsc) ⊡ (padz)
Transformer: ⊙ (attn) ‖ (norm) ⊏ (embd)
Reduction: ⊥/ϛ (rduc) ⊻ (wher) ⊧ (clmp) ⊜ (cmpr)
Signal: ∿ (fft) ⋐ (filt)
M2M: § (meta) ◆ (frag) ◇ (endf) ⊿ (ptch) ✦ (sign) ✓ (vrfy) ⚖ (vote) ⟐ (proj) ⟂ (dist) ⊃ (gath) ⊂ (scat)
Training: ∇ (bkwd) ⟳ (wupd) △ (loss) ⥁ (tnet) ⊞ (bn) ≋ (drop)
Backward: ⌐ˈ (relubk) σˈ (sigmbk) τˈ (tanhbk) ℊˈ (gelubk) Σˈ (softbk) ×ˈ (mmulbk) ⊛ˈ (convbk) ⊓ˈ (poolbk) ‖ˈ (normbk) ⊙ˈ (attnbk) ⥁ˈ (tndeep)

REGISTERS (always use Greek): ι κ λ μ ν ξ ο π ρ ς (R0-R9), α (accumulator), β (general), γ (scratch), δ (counter), φ (flag), ψ (stack), η θ ζ (gradients), ω (learning rate), χ υ ε (training).`,
};

function isNMLCode(text) {
  const lines = text.trim().split("\n").filter(l => l.trim() && !l.trim().startsWith(";"));
  if (lines.length === 0) return false;
  let nmlLines = 0;
  for (const line of lines) {
    const firstToken = line.trim().split(/\s+/)[0];
    if (NML_OPCODES.has(firstToken) || NML_OPCODES.has(firstToken.toUpperCase()) || NML_SYMBOLIC.has(firstToken)) {
      nmlLines++;
    }
  }
  return nmlLines >= lines.length * 0.5 && nmlLines >= 2;
}

function _looksLikeNML(line) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith(";")) return true;
  const first = trimmed.split(/\s+/)[0];
  return NML_OPCODES.has(first) || NML_OPCODES.has(first.toUpperCase()) || NML_SYMBOLIC.has(first);
}

function renderMarkdown(text) {
  const lines = text.split("\n");
  const result = [];
  let inCode = false;
  let codeLines = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("```")) {
      if (inCode) {
        result.push({ type: "code", content: codeLines.join("\n") });
        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    result.push({ type: "text", content: line });
  }
  if (inCode && codeLines.length) {
    result.push({ type: "code", content: codeLines.join("\n") });
  }

  // Second pass: detect consecutive NML lines in text blocks and group them as code
  const merged = [];
  let nmlBuf = [];

  const flushNML = () => {
    if (nmlBuf.length >= 2) {
      merged.push({ type: "code", content: nmlBuf.join("\n") });
    } else {
      for (const l of nmlBuf) merged.push({ type: "text", content: l });
    }
    nmlBuf = [];
  };

  for (const block of result) {
    if (block.type === "code") {
      flushNML();
      merged.push(block);
      continue;
    }
    if (_looksLikeNML(block.content)) {
      nmlBuf.push(block.content);
    } else {
      flushNML();
      merged.push(block);
    }
  }
  flushNML();

  return merged;
}

function MessageContent({ text, onNMLResult }) {
  const blocks = renderMarkdown(text);
  return blocks.map((block, i) => {
    if (block.type === "code") {
      const nml = isNMLCode(block.content);
      return (
        <div key={i} style={{ margin: "8px 0" }}>
          <div style={{ position: "relative" }}>
            <div style={{ position: "absolute", top: 4, right: 4, zIndex: 1, display: "flex", gap: 4 }}>
              <CopyButton text={block.content} />
            </div>
            {nml && (
              <div style={{
                position: "absolute", top: 4, left: 8, fontSize: 8, fontWeight: 700,
                letterSpacing: 2, color: "#00b4ff", opacity: 0.6,
              }}>
                NML
              </div>
            )}
            <pre style={{
              background: COLORS.bg,
              border: `1px solid ${nml ? "rgba(0,180,255,0.3)" : COLORS.border}`,
              borderRadius: 4,
              padding: nml ? "22px 14px 10px" : "10px 14px",
              paddingRight: 80,
              fontSize: 12.5,
              lineHeight: "19px",
              fontFamily: FONT,
              overflowX: "auto",
              whiteSpace: "pre",
              letterSpacing: 0.4,
              margin: 0,
            }}>
              {block.content}
            </pre>
          </div>
          {nml && (
            <div style={{ marginTop: 4 }}>
              <RunButton code={block.content} onResult={onNMLResult} messageText={text} />
            </div>
          )}
        </div>
      );
    }
    const line = block.content;
    if (!line.trim()) return <div key={i} style={{ height: 8 }} />;

    let html = line
      .replace(/\*\*([^*]+)\*\*/g, "<<b>>$1<</b>>")
      .replace(/`([^`]+)`/g, "<<code>>$1<</code>>");

    const parts = html.split(/(<<\/?(?:b|code)>>)/);
    const elements = [];
    let bold = false, code = false;
    for (let j = 0; j < parts.length; j++) {
      const p = parts[j];
      if (p === "<<b>>") { bold = true; continue; }
      if (p === "<</b>>") { bold = false; continue; }
      if (p === "<<code>>") { code = true; continue; }
      if (p === "<</code>>") { code = false; continue; }
      if (!p) continue;
      if (code) {
        elements.push(
          <span key={j} style={{
            background: COLORS.bg, padding: "1px 5px",
            borderRadius: 3, fontFamily: FONT, fontSize: 12,
            border: `1px solid ${COLORS.border}`,
          }}>{p}</span>
        );
      } else if (bold) {
        elements.push(<strong key={j} style={{ color: "#fff" }}>{p}</strong>);
      } else {
        elements.push(<span key={j}>{p}</span>);
      }
    }

    const isBullet = line.match(/^(\s*)[-*]\s/);
    if (isBullet) {
      return (
        <div key={i} style={{ paddingLeft: 16, position: "relative", margin: "2px 0" }}>
          <span style={{ position: "absolute", left: 4, color: COLORS.accent }}>•</span>
          {elements}
        </div>
      );
    }
    return <div key={i} style={{ margin: "2px 0" }}>{elements}</div>;
  });
}

function PipelineDisplay({ result }) {
  const [showNml, setShowNml] = useState(false);
  if (!result || !result.stages) return null;

  const total = result.stages.reduce((s, st) => s + (st.duration_ms || 0), 0);

  return (
    <div style={{
      maxWidth: 720, margin: "0 auto 12px",
      padding: "14px 16px", borderRadius: 6,
      background: "rgba(0,180,255,0.04)",
      border: "1px solid rgba(0,180,255,0.2)",
    }}>
      <div style={{
        fontSize: 9, fontWeight: 700, letterSpacing: 2, marginBottom: 10,
        color: "#00b4ff", display: "flex", alignItems: "center", gap: 8,
      }}>
        PIPELINE
        {result.intent && (
          <span style={{
            background: "rgba(0,180,255,0.1)", padding: "2px 6px",
            borderRadius: 3, fontSize: 9, fontWeight: 400, letterSpacing: 0.5,
          }}>
            {result.intent}
          </span>
        )}
        <span style={{ marginLeft: "auto", color: COLORS.textDim, fontWeight: 400 }}>
          {total}ms
        </span>
      </div>

      {result.stages.map((stage, i) => {
        const ok = stage.status === "success" || stage.status === "complete";
        return (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: 8,
            padding: "4px 0", fontSize: 12, fontFamily: FONT,
          }}>
            <span style={{ fontSize: 14, width: 18, textAlign: "center" }}>
              {ok ? "✓" : "✗"}
            </span>
            <span style={{
              color: ok ? COLORS.text : COLORS.error, flex: 1,
            }}>
              {stage.name || stage.stage}
            </span>
            {stage.duration_ms != null && (
              <span style={{ color: COLORS.textDim, fontSize: 10 }}>
                {stage.duration_ms}ms
              </span>
            )}
          </div>
        );
      })}

      {result.final_output != null && (
        <div style={{
          marginTop: 10, padding: "10px 14px", borderRadius: 4,
          background: COLORS.bg, border: `1px solid ${COLORS.border}`,
          fontSize: 18, fontWeight: 700, color: COLORS.accent,
          letterSpacing: 0.5, textAlign: "center",
        }}>
          {typeof result.final_output === "object"
            ? JSON.stringify(result.final_output, null, 2)
            : String(result.final_output)}
        </div>
      )}

      {result.nml_program && (
        <div style={{ marginTop: 8 }}>
          <button onClick={() => setShowNml(v => !v)} style={{
            background: "transparent", border: `1px solid ${COLORS.border}`,
            color: COLORS.textDim, padding: "3px 8px", borderRadius: 4,
            fontSize: 9, cursor: "pointer", fontFamily: FONT, letterSpacing: 1,
          }}>
            {showNml ? "HIDE NML" : "SHOW NML"}
          </button>
          {showNml && (
            <pre style={{
              marginTop: 6, background: COLORS.bg,
              border: `1px solid ${COLORS.border}`, borderRadius: 4,
              padding: "10px 14px", fontSize: 11, lineHeight: "16px",
              fontFamily: FONT, overflowX: "auto", whiteSpace: "pre",
              color: COLORS.text,
            }}>
              {result.nml_program}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };
  return (
    <button onClick={handleCopy} style={{
      background: copied ? COLORS.accentBg : "transparent",
      border: `1px solid ${copied ? COLORS.accent : COLORS.border}`,
      color: copied ? COLORS.accent : COLORS.textDim,
      padding: "3px 8px", borderRadius: 4, fontSize: 9,
      cursor: "pointer", fontFamily: FONT, letterSpacing: 0.5,
      transition: "all 0.2s",
    }}>
      {copied ? "COPIED" : "COPY"}
    </button>
  );
}

function detectInputs(code) {
  const loads = new Set();
  const stores = new Set();
  for (const line of code.split("\n")) {
    const tokens = line.trim().split(/\s+/);
    const op = tokens[0]?.toUpperCase();
    if ((op === "LD" || tokens[0] === "↓" || op === "LOAD") && tokens.length >= 3) {
      const addr = tokens.find(t => t.startsWith("@"));
      if (addr) loads.add(addr.slice(1));
    }
    if ((op === "ST" || tokens[0] === "↑" || op === "STORE") && tokens.length >= 3) {
      const addr = tokens.find(t => t.startsWith("@"));
      if (addr) stores.add(addr.slice(1));
    }
  }
  return [...loads].filter(a => !stores.has(a));
}

function extractDataFromContext(code, messageText) {
  const dataLines = [];
  const allText = messageText || "";
  for (const line of allText.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("@") && trimmed.includes("shape=") && trimmed.includes("data=")) {
      dataLines.push(trimmed);
    }
  }
  for (const line of code.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("@") && trimmed.includes("shape=") && trimmed.includes("data=")) {
      dataLines.push(trimmed);
    }
  }
  const seen = new Set();
  return dataLines.filter(l => {
    const name = l.split(/\s+/)[0];
    if (seen.has(name)) return false;
    seen.add(name);
    return true;
  });
}

function RunButton({ code, onResult, messageText }) {
  const [state, setState] = useState("idle");
  const [result, setResult] = useState(null);
  const [inputs, setInputs] = useState({});
  const [needsInputs, setNeedsInputs] = useState(null);
  const [showInputs, setShowInputs] = useState(true);
  const [contextData, setContextData] = useState([]);
  const [execMode, setExecMode] = useState("server");
  const [wasmAvailable, setWasmAvailable] = useState(false);

  useEffect(() => {
    initWasm().then(m => { if (m) setWasmAvailable(true); });
  }, []);

  useEffect(() => {
    const dataFromCtx = extractDataFromContext(code, messageText);
    setContextData(dataFromCtx);
    const dataNames = new Set(dataFromCtx.map(l => l.split(/\s+/)[0].slice(1)));
    const needed = detectInputs(code).filter(n => !dataNames.has(n));
    setNeedsInputs(needed.length > 0 ? needed : null);
    const init = {};
    for (const name of needed) init[name] = "";
    setInputs(prev => {
      const merged = { ...init };
      for (const name of needed) {
        if (prev[name]) merged[name] = prev[name];
      }
      return merged;
    });
  }, [code, messageText]);

  const buildDataContent = () => {
    const parts = [...contextData];
    if (needsInputs) {
      for (const name of needsInputs) {
        const val = inputs[name]?.trim() || "0.0";
        const nums = val.split(",").map(v => v.trim());
        const shape = nums.length > 1 ? `1,${nums.length}` : "1";
        parts.push(`@${name} shape=${shape} data=${nums.join(",")}`);
      }
    }
    return parts.join("\n");
  };

  const handleRun = async () => {
    setState("validating");
    setResult(null);
    try {
      const nmlOnly = code.split("\n").filter(l => {
        const t = l.trim();
        return !(t.startsWith("@") && t.includes("shape="));
      }).join("\n");

      const dataContent = buildDataContent();

      if (execMode === "wasm" && wasmAvailable) {
        setState("executing");
        const wasmResult = await runWasm(nmlOnly, dataContent);
        setState(wasmResult.status === "OK" ? "done" : "error");
        setResult(wasmResult);
        if (onResult) onResult(wasmResult, nmlOnly);
        return;
      }

      const valR = await fetch(EXEC_BASE + "/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nml_program: nmlOnly }),
      });
      const valData = await valR.json();

      if (!valData.valid) {
        setState("error");
        const r = { status: "GRAMMAR ERROR", errors: valData.errors?.map(e => e.message) || ["Invalid NML"] };
        setResult(r);
        if (onResult) onResult(r, nmlOnly);
        return;
      }

      setState("executing");
      const execR = await fetch(EXEC_BASE + "/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nml_program: nmlOnly, data: dataContent }),
      });
      const execData = await execR.json();
      setState("done");
      setResult(execData);
      if (onResult) onResult(execData, nmlOnly);
    } catch (e) {
      if (execMode === "server") {
        setState("error");
        const r = { status: "CONNECTION ERROR", errors: [e.message + " — try WASM mode"] };
        setResult(r);
        if (onResult) onResult(r, code);
      }
    }
  };

  const btnColors = {
    idle: { bg: "transparent", border: "#00b4ff", color: "#00b4ff" },
    validating: { bg: "rgba(0,180,255,0.1)", border: "#00b4ff", color: "#00b4ff" },
    executing: { bg: "rgba(0,180,255,0.1)", border: "#00b4ff", color: "#00b4ff" },
    done: { bg: "rgba(0,255,157,0.1)", border: COLORS.accent, color: COLORS.accent },
    error: { bg: "rgba(255,68,68,0.1)", border: COLORS.error, color: COLORS.error },
  };
  const s = btnColors[state];
  const modeLabel = execMode === "wasm" ? "WASM" : "SERVER";
  const btnLabel = { idle: `▶ RUN (${modeLabel})`, validating: "VALIDATING...", executing: "EXECUTING...", done: `▶ RE-RUN (${modeLabel})`, error: "▶ RETRY" };
  const hasDataOrInputs = contextData.length > 0 || needsInputs;

  return (
    <div style={{ display: "inline-flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", gap: 4 }}>
        <button onClick={handleRun} disabled={state === "validating" || state === "executing"} style={{
          background: s.bg, border: `1px solid ${s.border}`, color: s.color,
          padding: "3px 8px", borderRadius: 4, fontSize: 9,
          cursor: state === "validating" || state === "executing" ? "wait" : "pointer",
          fontFamily: FONT, letterSpacing: 0.5, transition: "all 0.2s",
        }}>
          {btnLabel[state]}
        </button>
        <button onClick={() => setExecMode(execMode === "server" ? "wasm" : "server")} style={{
          background: "transparent", border: `1px solid ${execMode === "wasm" ? "#ff9900" : "#555"}`,
          color: execMode === "wasm" ? "#ff9900" : "#555",
          padding: "3px 6px", borderRadius: 4, fontSize: 8,
          cursor: "pointer", fontFamily: FONT, letterSpacing: 0.5,
          opacity: wasmAvailable || execMode === "server" ? 1 : 0.4,
        }} title={wasmAvailable ? "Toggle between server and WASM execution" : "WASM not loaded (need nml.js + nml.wasm)"}>
          {execMode === "wasm" ? "⚡ WASM" : "🖥 SERVER"}
        </button>
        {hasDataOrInputs && (
          <button onClick={() => setShowInputs(v => !v)} style={{
            background: "transparent", border: `1px solid ${COLORS.border}`,
            color: COLORS.textDim, padding: "3px 6px", borderRadius: 4, fontSize: 9,
            cursor: "pointer", fontFamily: FONT,
          }}>
            {showInputs ? "▼ DATA" : "▶ DATA"}
          </button>
        )}
      </div>
      {hasDataOrInputs && showInputs && (
        <div style={{
          padding: "6px 8px", borderRadius: 4,
          background: "rgba(0,180,255,0.04)", border: "1px solid rgba(0,180,255,0.15)",
          fontSize: 10, fontFamily: FONT, minWidth: 220, maxWidth: 360,
          maxHeight: 180, overflowY: "auto",
        }}>
          {contextData.length > 0 && (
            <>
              <div style={{ fontSize: 7, fontWeight: 700, letterSpacing: 2, color: COLORS.accent, marginBottom: 4 }}>
                DATA ({contextData.length})
              </div>
              {contextData.map((line, i) => {
                const name = line.split(/\s+/)[0];
                const rest = line.slice(name.length).trim();
                return (
                  <div key={`d-${i}`} style={{ marginBottom: 2, lineHeight: "14px" }}>
                    <span style={{ color: "#ffcc00" }}>{name}</span>{" "}
                    <span style={{ color: COLORS.textDim, fontSize: 9 }}>{rest}</span>
                  </div>
                );
              })}
            </>
          )}
          {needsInputs && (
            <>
              <div style={{
                fontSize: 7, fontWeight: 700, letterSpacing: 2, color: "#00b4ff",
                marginBottom: 4, marginTop: contextData.length > 0 ? 6 : 0,
              }}>
                INPUTS ({needsInputs.length})
              </div>
              {needsInputs.map(name => (
                <div key={name} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
                  <span style={{ color: "#ffcc00", fontSize: 10, minWidth: 60 }}>@{name}</span>
                  <input
                    value={inputs[name] || ""}
                    onChange={e => setInputs(prev => ({ ...prev, [name]: e.target.value }))}
                    placeholder="0.0"
                    style={{
                      background: COLORS.bg, color: COLORS.text, border: `1px solid ${COLORS.border}`,
                      borderRadius: 3, padding: "2px 6px", fontFamily: FONT, fontSize: 10,
                      flex: 1, minWidth: 60, outline: "none",
                    }}
                  />
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ExecutionResult({ content, defaultCollapsed = true }) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const isError = content.includes("failed") || content.includes("ERROR");

  return (
    <div style={{
      maxWidth: 720, margin: "0 auto 8px",
      borderRadius: 6, overflow: "hidden",
      border: `1px solid ${isError ? "rgba(255,68,68,0.3)" : "rgba(0,255,157,0.3)"}`,
    }}>
      <div
        onClick={() => setCollapsed(c => !c)}
        style={{
          padding: "8px 16px",
          background: isError ? "rgba(255,68,68,0.06)" : "rgba(0,255,157,0.06)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          cursor: "pointer", userSelect: "none",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            fontSize: 9, fontWeight: 700, letterSpacing: 2,
            color: isError ? COLORS.error : COLORS.accent,
          }}>
            EXECUTION OUTPUT
          </span>
          <span style={{ fontSize: 10, color: COLORS.textDim }}>
            {content.includes("cycles") ? content.match(/\(([^)]+)\)/)?.[1] || "" : ""}
          </span>
        </div>
        <span style={{ fontSize: 10, color: COLORS.textDim, fontFamily: FONT }}>
          {collapsed ? "▶ SHOW" : "▼ HIDE"}
        </span>
      </div>
      {!collapsed && (
        <div style={{
          padding: "10px 16px",
          background: COLORS.panel,
          fontSize: 12, lineHeight: "20px", fontFamily: FONT,
        }}>
          <MessageContent text={content} />
        </div>
      )}
    </div>
  );
}

export default function NMLChat() {
  const [messages, setMessages] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("nml-chat-messages")) || [];
    } catch { return []; }
  });
  const [input, setInput] = useState("");
  const [generating, setGenerating] = useState(false);
  const [connected, setConnected] = useState(false);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [showSystem, setShowSystem] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [ragStatus, setRagStatus] = useState(null);
  const [agentStatus, setAgentStatus] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [constrained, setConstrained] = useState(false);
  const [constrainedAvailable, setConstrainedAvailable] = useState(false);
  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const ctrlRef = useRef(null);

  const scrollBottom = useCallback(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, []);

  useEffect(() => { scrollBottom(); }, [messages, streamText, scrollBottom]);

  useEffect(() => {
    try { localStorage.setItem("nml-chat-messages", JSON.stringify(messages)); }
    catch {}
  }, [messages]);

  const handleNMLResult = useCallback(async (result, code) => {
    let summary = "";
    const runtimeTag = result.runtime === "wasm" ? " [WASM]" : "";
    if (result.status === "OK" && result.runtime === "wasm") {
      const memEntries = Object.entries(result.memory || {});
      const outputLines = memEntries.map(([k, v]) =>
        `@${k} = [${v.data.map(n => n.toFixed(4)).join(", ")}]`
      );
      summary = `**Execution result${runtimeTag}**:\n${outputLines.join("\n")}\n\n\`\`\`\n${result.output}\`\`\``;
    } else if (result.status === "HALTED") {
      const outputs = result.outputs || {};
      const outputLines = Object.entries(outputs).map(([k, v]) =>
        `@${k} = ${Array.isArray(v) ? `[${v.map(n => n.toFixed(4)).join(", ")}]` : v.toFixed(4)}`
      );
      summary = `**Execution result** (${result.cycles} cycles, ${result.time_us} µs):\n${outputLines.join("\n")}`;
    } else if (result.errors) {
      summary = `**Execution failed${runtimeTag}**: ${result.errors.join(", ")}`;
    } else {
      summary = `**Execution${runtimeTag}**: ${result.status || "unknown"}${result.stderr ? " — " + result.stderr : ""}`;
    }

    setMessages(prev => [...prev, { role: "execution", content: summary, collapsed: true }]);

    if (result.status === "HALTED" && connected && selectedModel) {
      setGenerating(true);
      setStreamText("");
      const outputDetail = Object.entries(result.outputs || {}).map(([k, v]) =>
        `@${k} = ${Array.isArray(v) ? `[${v.map(n => n.toFixed(4)).join(", ")}]` : v.toFixed(4)}`
      ).join("\n");
      const explainPrompt = `Explain this NML program step by step, then explain each output register value and what it represents.\n\nProgram:\n\`\`\`\n${code}\n\`\`\`\n\nExecution completed in ${result.cycles} cycles (${result.time_us} µs).\n\nOutput values:\n${outputDetail}\n\nFor each output, explain: what computation produced this value, and what it means.`;
      const apiMsgs = [
        { role: "system", content: "You are an NML expert. Walk through the program instruction by instruction, then explain each output register value: what computation produced it, what the numeric value represents, and why it has that value. Be specific about the numbers." },
        { role: "user", content: explainPrompt },
      ];

      let full = "";
      try {
        const r = await fetch(API_BASE + "/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model: selectedModel, messages: apiMsgs, stream: true, max_tokens: 512, temperature: 0.3 }),
        });
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          for (const line of decoder.decode(value).split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6);
            if (data === "[DONE]") break;
            try {
              const token = JSON.parse(data).choices?.[0]?.delta?.content;
              if (token) { full += token; setStreamText(full); }
            } catch {}
          }
        }
      } catch {}

      if (full) {
        setMessages(prev => [...prev, { role: "assistant", content: full, ephemeral: true }]);
      }
      setStreamText("");
      setGenerating(false);
    }
  }, [connected, selectedModel]);

  const checkServer = useCallback(async () => {
    try {
      const r = await fetch(API_BASE + "/models", { signal: AbortSignal.timeout(3000) });
      if (!r.ok) throw new Error();
      const d = await r.json();
      const list = (d.data || []).map(m => m.id);
      setModels(list);
      if (list.length) setSelectedModel(prev => prev || list[0]);
      setConnected(true);

      try {
        const healthR = await fetch(EXEC_BASE + "/health", { signal: AbortSignal.timeout(2000) });
        if (healthR.ok) { const h = await healthR.json(); setConstrainedAvailable(!!h.constrained_decoding); }
      } catch {}

      try {
        const ragR = await fetch(API_BASE + "/rag/status", { signal: AbortSignal.timeout(2000) });
        if (ragR.ok) setRagStatus(await ragR.json());
      } catch { setRagStatus(null); }

      try {
        const agentR = await fetch(API_BASE + "/agents/status", { signal: AbortSignal.timeout(2000) });
        if (agentR.ok) setAgentStatus(await agentR.json());
      } catch { setAgentStatus(null); }
    } catch {
      setConnected(false);
      setModels([]);
      setRagStatus(null);
      setAgentStatus(null);
    }
  }, []);

  useEffect(() => {
    checkServer();
    const iv = setInterval(checkServer, 8000);
    return () => clearInterval(iv);
  }, [checkServer]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || generating) return;

    const userMsg = { role: "user", content: text };
    const newMsgs = [...messages, userMsg];
    setMessages(newMsgs);
    setInput("");
    setGenerating(true);
    setStreamText("");

    const apiMsgs = [];
    if (systemPrompt.trim()) apiMsgs.push({ role: "system", content: systemPrompt.trim() });
    apiMsgs.push(...newMsgs
      .filter(m => (m.role === "user" || m.role === "assistant") && !m.ephemeral)
      .map(m => ({ role: m.role, content: m.content })));

    ctrlRef.current = new AbortController();
    let full = "";
    let pipelineUsed = false;
    let pipeData = null;
    setPipelineResult(null);

    try {
      const pipeR = await fetch(API_BASE + "/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: AbortSignal.timeout(15000),
      });
      if (pipeR.ok) {
        pipeData = await pipeR.json();
        if (pipeData.status !== "error" && pipeData.intent !== "general_chat") {
          setPipelineResult(pipeData);
          pipelineUsed = true;
        }
      }
    } catch { /* pipeline not available, fall through to LLM */ }

    if (!pipeData?.final_output) {
      if (pipelineUsed) {
        apiMsgs.push({
          role: "system",
          content: "The agent pipeline has already computed a structured result for this query. Summarize and explain it to the user.",
        });
      }

      try {
        const r = await fetch(API_BASE + "/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedModel,
            messages: apiMsgs,
            stream: !constrained,
            max_tokens: 2048,
            temperature: 0.7,
            constrained,
          }),
          signal: ctrlRef.current.signal,
        });

        if (constrained) {
          const d = await r.json();
          full = d.choices?.[0]?.message?.content || "";
          setStreamText(full);
        } else {
          const reader = r.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            for (const line of chunk.split("\n")) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6);
              if (data === "[DONE]") break;
              try {
                const token = JSON.parse(data).choices?.[0]?.delta?.content;
                if (token) {
                  full += token;
                  setStreamText(full);
                }
              } catch {}
            }
          }
        }
      } catch (e) {
        if (e.name !== "AbortError") {
          full += "\n\n*[Connection lost]*";
        }
      }
    }

    if (full) {
      setMessages(prev => [...prev, { role: "assistant", content: full }]);
    }
    setStreamText("");
    setGenerating(false);
    inputRef.current?.focus();
  }, [input, generating, messages, systemPrompt, selectedModel, constrained]);

  const stop = useCallback(() => {
    ctrlRef.current?.abort();
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setStreamText("");
    try { localStorage.removeItem("nml-chat-messages"); } catch {}
  }, []);

  return (
    <div style={{
      fontFamily: FONT, background: COLORS.bg, color: COLORS.text,
      height: "100vh", display: "flex", flexDirection: "column", overflow: "hidden",
    }}>
      {/* HEADER */}
      <div style={{
        background: COLORS.panel, borderBottom: `1px solid ${COLORS.border}`,
        padding: "12px 20px", display: "flex", alignItems: "center",
        justifyContent: "space-between", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ color: COLORS.accent, fontSize: 20, fontWeight: 800, letterSpacing: 3 }}>NML</span>
          <span style={{ color: COLORS.textDim, fontSize: 11, letterSpacing: 1 }}>CHAT</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <select
            value={selectedModel}
            onChange={e => setSelectedModel(e.target.value)}
            style={{
              background: COLORS.bg, color: COLORS.text, border: `1px solid ${COLORS.border}`,
              borderRadius: 4, padding: "4px 8px", fontFamily: FONT, fontSize: 10,
              maxWidth: 220, cursor: "pointer",
            }}
          >
            {models.length === 0 && <option value="">No models</option>}
            {models.map(m => (
              <option key={m} value={m}>{m.split("/").pop()}</option>
            ))}
          </select>
          <button onClick={() => setShowSystem(s => !s)} style={{
            background: showSystem ? COLORS.accentBg : "transparent",
            border: `1px solid ${showSystem ? COLORS.accent : COLORS.border}`,
            color: showSystem ? COLORS.accent : COLORS.textDim,
            padding: "4px 10px", borderRadius: 4, fontSize: 10,
            cursor: "pointer", fontFamily: FONT, letterSpacing: 0.5,
          }}>
            SYSTEM
          </button>
          <button onClick={clearChat} style={{
            background: "transparent", border: `1px solid ${COLORS.border}`,
            color: COLORS.textDim, padding: "4px 10px", borderRadius: 4,
            fontSize: 10, cursor: "pointer", fontFamily: FONT, letterSpacing: 0.5,
          }}>
            CLEAR
          </button>
          {ragStatus && (
            <div style={{
              background: COLORS.accentBg, border: `1px solid ${COLORS.accent}`,
              padding: "3px 8px", borderRadius: 4, fontSize: 9,
              color: COLORS.accent, letterSpacing: 1, fontWeight: 700,
            }}>
              RAG {ragStatus.tax_files?.toLocaleString()} TAXES
            </div>
          )}
          {agentStatus && (
            <div style={{
              background: "rgba(0,180,255,0.08)", border: "1px solid #00b4ff",
              padding: "3px 8px", borderRadius: 4, fontSize: 9,
              color: "#00b4ff", letterSpacing: 1, fontWeight: 700,
            }}>
              AGENTS {agentStatus.healthy ?? agentStatus.count ?? 0}
            </div>
          )}
          {constrainedAvailable && (
            <button onClick={() => setConstrained(c => !c)} style={{
              background: constrained ? "rgba(255,170,0,0.1)" : "transparent",
              border: `1px solid ${constrained ? COLORS.warn : COLORS.border}`,
              color: constrained ? COLORS.warn : COLORS.textDim,
              padding: "3px 8px", borderRadius: 4, fontSize: 9,
              cursor: "pointer", fontFamily: FONT, letterSpacing: 1,
              fontWeight: constrained ? 700 : 400, transition: "all 0.2s",
            }}>
              {constrained ? "CFG ON" : "CFG"}
            </button>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 8 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: connected ? COLORS.accent : COLORS.error,
              transition: "background 0.3s",
            }} />
            <span style={{ fontSize: 10, color: COLORS.textDim, letterSpacing: 0.5 }}>
              {connected ? "CONNECTED" : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>

      {/* SYSTEM PROMPT */}
      {showSystem && (
        <div style={{
          background: COLORS.panel, borderBottom: `1px solid ${COLORS.border}`,
          padding: "10px 20px",
        }}>
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            marginBottom: 8,
          }}>
            <div style={{ color: COLORS.textDim, fontSize: 9, letterSpacing: 2 }}>
              SYSTEM PROMPT
            </div>
            <div style={{ display: "flex", gap: 6 }}>
              {Object.keys(SYSTEM_PROMPTS).map(key => {
                const active = systemPrompt === SYSTEM_PROMPTS[key];
                return (
                  <button key={key} onClick={() => {
                    const next = SYSTEM_PROMPTS[key];
                    if (next !== systemPrompt) {
                      setMessages([]);
                      setStreamText("");
                      setPipelineResult(null);
                    }
                    setSystemPrompt(next);
                  }} style={{
                    background: active ? COLORS.accentBg : "transparent",
                    border: `1px solid ${active ? COLORS.accent : COLORS.border}`,
                    color: active ? COLORS.accent : COLORS.textDim,
                    padding: "3px 10px", borderRadius: 4, fontSize: 9,
                    cursor: "pointer", fontFamily: FONT, letterSpacing: 1,
                    fontWeight: active ? 700 : 400, transition: "all 0.2s",
                  }}>
                    {key.toUpperCase()}
                  </button>
                );
              })}
              {systemPrompt && (
                <button onClick={() => setSystemPrompt("")} style={{
                  background: "transparent", border: `1px solid ${COLORS.border}`,
                  color: COLORS.textDim, padding: "3px 10px", borderRadius: 4,
                  fontSize: 9, cursor: "pointer", fontFamily: FONT, letterSpacing: 1,
                }}>
                  CLEAR
                </button>
              )}
            </div>
          </div>
          <textarea
            value={systemPrompt}
            onChange={e => setSystemPrompt(e.target.value)}
            placeholder="Select a preset above or type a custom system prompt..."
            style={{
              width: "100%", background: COLORS.bg, color: COLORS.text,
              border: `1px solid ${COLORS.border}`, borderRadius: 4,
              padding: "8px 12px", fontFamily: FONT, fontSize: 12,
              resize: "vertical", minHeight: 50, lineHeight: "18px",
              outline: "none",
            }}
          />
        </div>
      )}

      {/* CHAT MESSAGES */}
      <div ref={chatRef} style={{ flex: 1, overflowY: "auto", padding: "20px" }}>
        {messages.length === 0 && !streamText && (
          <div style={{
            textAlign: "center", padding: "80px 20px",
            color: COLORS.textDim, fontSize: 12, lineHeight: 2,
          }}>
            <div style={{ color: COLORS.accent, fontSize: 32, fontWeight: 800, letterSpacing: 4, marginBottom: 12 }}>
              NML CHAT
            </div>
            <div>Chat with your locally running MLX models.</div>
            <div style={{ fontSize: 10, opacity: 0.6, marginTop: 8 }}>
              {connected
                ? `Model: ${selectedModel.split("/").pop() || "none"}${ragStatus ? ` • RAG: ${ragStatus.tax_files?.toLocaleString()} tax files` : ""}`
                : `Waiting for server on port ${NML_PORT}...`}
            </div>
            {ragStatus && (
              <div style={{ fontSize: 10, opacity: 0.5, marginTop: 4 }}>
                Pay date: {ragStatus.pay_date} • {ragStatus.tax_types} jurisdiction keys
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => {
          if (msg.role === "execution") {
            return <ExecutionResult key={i} content={msg.content} defaultCollapsed={msg.collapsed} />;
          }
          return (
            <div key={i} style={{
              maxWidth: 720, margin: "0 auto 12px",
              padding: "12px 16px", borderRadius: 6,
              background: msg.role === "user" ? COLORS.userBg : COLORS.panel,
              border: `1px solid ${msg.role === "user" ? COLORS.userBorder : COLORS.border}`,
            }}>
              <div style={{
                fontSize: 9, fontWeight: 700, letterSpacing: 2, marginBottom: 8,
                color: msg.role === "user" ? COLORS.accent : COLORS.textDim,
                display: "flex", justifyContent: "space-between", alignItems: "center",
              }}>
                <span>{msg.role === "user" ? "YOU" : "ASSISTANT"}</span>
                {msg.role === "assistant" && <CopyButton text={msg.content} />}
              </div>
              <div style={{ fontSize: 13, lineHeight: "22px", letterSpacing: 0.3 }}>
                <MessageContent text={msg.content} onNMLResult={handleNMLResult} />
              </div>
            </div>
          );
        })}

        {pipelineResult && <PipelineDisplay result={pipelineResult} />}

        {generating && streamText && (
          <div style={{
            maxWidth: 720, margin: "0 auto 12px",
            padding: "12px 16px", borderRadius: 6,
            background: COLORS.panel, border: `1px solid ${COLORS.border}`,
          }}>
            <div style={{
              fontSize: 9, fontWeight: 700, letterSpacing: 2, marginBottom: 8,
              color: COLORS.textDim,
            }}>
              ASSISTANT
            </div>
            <div style={{ fontSize: 13, lineHeight: "20px" }}>
              <MessageContent text={streamText} onNMLResult={handleNMLResult} />
            </div>
          </div>
        )}

        {generating && !streamText && (
          <div style={{
            maxWidth: 720, margin: "0 auto 12px",
            padding: "12px 16px", borderRadius: 6,
            background: COLORS.panel, border: `1px solid ${COLORS.border}`,
            color: COLORS.textDim, fontSize: 12,
          }}>
            Thinking...
          </div>
        )}
      </div>

      {/* INPUT */}
      <div style={{
        borderTop: `1px solid ${COLORS.border}`, background: COLORS.panel,
        padding: "12px 20px", flexShrink: 0,
      }}>
        <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", gap: 10, alignItems: "flex-end" }}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 150) + "px";
            }}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            placeholder={connected ? "Send a message..." : "Waiting for server..."}
            disabled={!connected}
            rows={1}
            style={{
              flex: 1, background: COLORS.bg, color: COLORS.text,
              border: `1px solid ${COLORS.border}`, borderRadius: 6,
              padding: "10px 14px", fontFamily: FONT, fontSize: 13,
              resize: "none", maxHeight: 150, lineHeight: "20px",
              outline: "none",
            }}
          />
          {generating ? (
            <button onClick={stop} style={{
              background: COLORS.error, color: "#fff", border: "none",
              padding: "10px 20px", borderRadius: 6, fontWeight: 800,
              fontSize: 11, cursor: "pointer", fontFamily: FONT, letterSpacing: 2,
              whiteSpace: "nowrap",
            }}>
              STOP
            </button>
          ) : (
            <button onClick={send} disabled={!connected || !input.trim()} style={{
              background: COLORS.accent, color: COLORS.bg, border: "none",
              padding: "10px 20px", borderRadius: 6, fontWeight: 800,
              fontSize: 11, cursor: "pointer", fontFamily: FONT, letterSpacing: 2,
              whiteSpace: "nowrap",
              opacity: (!connected || !input.trim()) ? 0.3 : 1,
            }}>
              SEND
            </button>
          )}
        </div>
        <div style={{
          maxWidth: 720, margin: "6px auto 0", fontSize: 9,
          color: COLORS.textDim, textAlign: "right", letterSpacing: 1,
        }}>
          ENTER TO SEND • SHIFT+ENTER FOR NEW LINE
        </div>
      </div>
    </div>
  );
}
