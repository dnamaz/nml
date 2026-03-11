import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8082/v1";
const EXEC_BASE = "http://localhost:8082";

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
  return result;
}

function MessageContent({ text }) {
  const blocks = renderMarkdown(text);
  return blocks.map((block, i) => {
    if (block.type === "code") {
      const nml = isNMLCode(block.content);
      return (
        <div key={i} style={{ position: "relative", margin: "8px 0" }}>
          <div style={{ position: "absolute", top: 4, right: 4, zIndex: 1, display: "flex", gap: 4 }}>
            {nml && <RunButton code={block.content} />}
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
            paddingRight: 120,
            fontSize: 12.5,
            lineHeight: "19px",
            fontFamily: FONT,
            overflowX: "auto",
            whiteSpace: "pre",
            letterSpacing: 0.4,
          }}>
            {block.content}
          </pre>
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

const NML_OPCODES = new Set([
  "MMUL","MADD","MSUB","EMUL","EDIV","SDOT","DOT","SCLR","SDIV",
  "RELU","SIGM","TANH","SOFT","GELU",
  "LD","ST","MOV","ALLC","RSHP","TRNS","SPLT","MERG",
  "CMPF","CMP","CMPI","JMPT","JMPF","JUMP","JMP","LOOP","ENDP",
  "CALL","RET","LEAF","TACC","SYNC","HALT","TRAP",
  "CONV","POOL","UPSC","PADZ","ATTN","NORM","EMBD",
  "RDUC","WHER","CLMP","CMPR","FFT","FILT",
  "META","FRAG","ENDF","LINK","VOTE","PROJ","DIST","GATH","SCAT","SCTR",
  "SYS","MOD","ITOF","FTOI","BNOT","SIGN","VRFY","PTCH",
]);
const NML_SYMBOLIC = new Set([
  "×","⊕","⊖","⊗","⊘","·","∗","÷","⌐","σ","τ","Σ","ℊ",
  "↓","↑","←","□","⊞","⊤","⊢","⊣","⋈","≶","≺","ϟ",
  "↗","↘","→","↻","↺","∎","∑","⇒","⇐","⏸","◼","⚠",
  "⊛","⊓","⊔","⊡","⊙","‖","⊏","⊥","ϛ","⊻","⊧","⊜",
  "∿","⋐","§","◆","◇","⚖","✦","✓","⟐","⟂","⊃","⊂","⚙",
]);

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

function RunButton({ code }) {
  const [state, setState] = useState("idle");
  const [result, setResult] = useState(null);

  const handleRun = async () => {
    setState("validating");
    setResult(null);
    try {
      const valR = await fetch(EXEC_BASE + "/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nml_program: code }),
      });
      const valData = await valR.json();

      if (!valData.valid) {
        setState("error");
        setResult({
          status: "GRAMMAR ERROR",
          errors: valData.errors?.map(e => e.message) || ["Invalid NML"],
        });
        return;
      }

      setState("executing");
      const execR = await fetch(EXEC_BASE + "/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nml_program: code, data: "" }),
      });
      const execData = await execR.json();
      setState("done");
      setResult(execData);
    } catch (e) {
      setState("error");
      setResult({ status: "CONNECTION ERROR", errors: [e.message] });
    }
  };

  const btnColors = {
    idle: { bg: "transparent", border: "#00b4ff", color: "#00b4ff", text: "RUN" },
    validating: { bg: "rgba(0,180,255,0.1)", border: "#00b4ff", color: "#00b4ff", text: "VALIDATING..." },
    executing: { bg: "rgba(0,180,255,0.1)", border: "#00b4ff", color: "#00b4ff", text: "EXECUTING..." },
    done: { bg: "rgba(0,255,157,0.1)", border: COLORS.accent, color: COLORS.accent, text: "RAN" },
    error: { bg: "rgba(255,68,68,0.1)", border: COLORS.error, color: COLORS.error, text: "ERROR" },
  };
  const s = btnColors[state];

  return (
    <div>
      <button onClick={handleRun} disabled={state === "validating" || state === "executing"} style={{
        background: s.bg, border: `1px solid ${s.border}`, color: s.color,
        padding: "3px 8px", borderRadius: 4, fontSize: 9,
        cursor: state === "validating" || state === "executing" ? "wait" : "pointer",
        fontFamily: FONT, letterSpacing: 0.5, transition: "all 0.2s",
      }}>
        {state === "idle" ? "▶ RUN" : s.text}
      </button>
      {result && (
        <div style={{
          marginTop: 6, padding: "8px 12px", borderRadius: 4,
          background: result.status === "HALTED" ? "rgba(0,255,157,0.06)" : "rgba(255,68,68,0.06)",
          border: `1px solid ${result.status === "HALTED" ? COLORS.accent : COLORS.error}`,
          fontSize: 11, fontFamily: FONT,
        }}>
          <div style={{
            fontSize: 9, fontWeight: 700, letterSpacing: 2, marginBottom: 4,
            color: result.status === "HALTED" ? COLORS.accent : COLORS.error,
          }}>
            {result.status || "RESULT"}
            {result.cycles != null && <span style={{ fontWeight: 400, marginLeft: 8, color: COLORS.textDim }}>{result.cycles} cycles</span>}
            {result.time_us != null && <span style={{ fontWeight: 400, marginLeft: 8, color: COLORS.textDim }}>{result.time_us} µs</span>}
          </div>
          {result.outputs && Object.keys(result.outputs).length > 0 && (
            <div style={{ color: COLORS.text }}>
              {Object.entries(result.outputs).map(([k, v]) => (
                <div key={k}>
                  <span style={{ color: "#ffcc00" }}>@{k}</span>
                  <span style={{ color: COLORS.textDim }}> = </span>
                  <span style={{ color: COLORS.accent, fontWeight: 700 }}>
                    {Array.isArray(v) ? `[${v.map(n => n.toFixed(4)).join(", ")}]` : v.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          )}
          {result.errors && (
            <div style={{ color: COLORS.error }}>
              {result.errors.map((e, i) => <div key={i}>{e}</div>)}
            </div>
          )}
          {result.stderr && <div style={{ color: COLORS.error, fontSize: 10 }}>{result.stderr}</div>}
        </div>
      )}
    </div>
  );
}

export default function NMLChat() {
  const [messages, setMessages] = useState([]);
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
  const chatRef = useRef(null);
  const inputRef = useRef(null);
  const ctrlRef = useRef(null);

  const scrollBottom = useCallback(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, []);

  useEffect(() => { scrollBottom(); }, [messages, streamText, scrollBottom]);

  const checkServer = useCallback(async () => {
    try {
      const r = await fetch(API_BASE + "/models", { signal: AbortSignal.timeout(3000) });
      if (!r.ok) throw new Error();
      const d = await r.json();
      const list = (d.data || []).map(m => m.id);
      setModels(list);
      if (list.length && !selectedModel) setSelectedModel(list[0]);
      setConnected(true);

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
  }, [selectedModel]);

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
    apiMsgs.push(...newMsgs.map(m => ({ role: m.role, content: m.content })));

    ctrlRef.current = new AbortController();
    let full = "";
    let pipelineUsed = false;
    setPipelineResult(null);

    try {
      const pipeR = await fetch(API_BASE + "/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
        signal: AbortSignal.timeout(15000),
      });
      if (pipeR.ok) {
        const pipeData = await pipeR.json();
        if (pipeData.status !== "error" && pipeData.intent !== "general_chat") {
          setPipelineResult(pipeData);
          pipelineUsed = true;
        }
      }
    } catch { /* pipeline not available, fall through to LLM */ }

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
          stream: true,
          max_tokens: 2048,
          temperature: 0.7,
        }),
        signal: ctrlRef.current.signal,
      });

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
    } catch (e) {
      if (e.name !== "AbortError") {
        full += "\n\n*[Connection lost]*";
      }
    }

    if (full) {
      setMessages(prev => [...prev, { role: "assistant", content: full }]);
    }
    setStreamText("");
    setGenerating(false);
    inputRef.current?.focus();
  }, [input, generating, messages, systemPrompt, selectedModel]);

  const stop = useCallback(() => {
    ctrlRef.current?.abort();
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setStreamText("");
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
          <div style={{ color: COLORS.textDim, fontSize: 9, letterSpacing: 2, marginBottom: 6 }}>
            SYSTEM PROMPT
          </div>
          <textarea
            value={systemPrompt}
            onChange={e => setSystemPrompt(e.target.value)}
            placeholder="You are an NML v0.6.4 code generator. NML uses LEAF for constants, LD for memory, TACC for scalar addition, EMUL for multiply, SCLR for scaling. All programs must end with HALT."
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
                : "Waiting for RAG server on port 8082..."}
            </div>
            {ragStatus && (
              <div style={{ fontSize: 10, opacity: 0.5, marginTop: 4 }}>
                Pay date: {ragStatus.pay_date} • {ragStatus.tax_types} jurisdiction keys
              </div>
            )}
          </div>
        )}

        {messages.map((msg, i) => (
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
              <MessageContent text={msg.content} />
            </div>
          </div>
        ))}

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
              <MessageContent text={streamText} />
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
