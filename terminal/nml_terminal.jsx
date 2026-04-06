import { useState, useRef, useEffect, useCallback } from "react";

// ═══════════════════════════════════════════
// NML EMULATOR CORE
// ═══════════════════════════════════════════

class NMLTensor {
  constructor(shape, data = null) {
    this.shape = Array.isArray(shape) ? shape : [shape];
    const size = this.shape.reduce((a, b) => a * b, 1);
    this.data = data ? Float32Array.from(data) : new Float32Array(size);
  }
  get size() { return this.data.length; }
  clone() { return new NMLTensor(this.shape, this.data); }
  toString() {
    if (this.shape.length === 1 && this.shape[0] <= 8) {
      return `[${Array.from(this.data).map(v => v.toFixed(4)).join(", ")}]`;
    }
    return `Tensor<${this.shape.join("×")}> (${this.size} elements)`;
  }
}

function matmul(a, b) {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    if (a.shape.length === 1 && b.shape.length === 2) {
      a = new NMLTensor([1, a.shape[0]], a.data);
    } else {
      throw new Error(`MMUL requires 2D tensors, got ${a.shape} and ${b.shape}`);
    }
  }
  const [m, k1] = a.shape;
  const [k2, n] = b.shape;
  if (k1 !== k2) throw new Error(`Shape mismatch: ${a.shape} @ ${b.shape}`);
  const out = new NMLTensor([m, n]);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k1; p++) s += a.data[i * k1 + p] * b.data[p * n + j];
      out.data[i * n + j] = s;
    }
  return out;
}

function elementwise(a, b, op) {
  if (a.size !== b.size) throw new Error(`Size mismatch: ${a.size} vs ${b.size}`);
  const out = new NMLTensor(a.shape);
  for (let i = 0; i < a.size; i++) out.data[i] = op(a.data[i], b.data[i]);
  return out;
}

function relu(t) {
  const out = new NMLTensor(t.shape);
  for (let i = 0; i < t.size; i++) out.data[i] = Math.max(0, t.data[i]);
  return out;
}
function sigmoid(t) {
  const out = new NMLTensor(t.shape);
  for (let i = 0; i < t.size; i++) out.data[i] = 1 / (1 + Math.exp(-t.data[i]));
  return out;
}
function tanh_act(t) {
  const out = new NMLTensor(t.shape);
  for (let i = 0; i < t.size; i++) out.data[i] = Math.tanh(t.data[i]);
  return out;
}
function softmax(t) {
  const out = new NMLTensor(t.shape);
  const max = Math.max(...t.data);
  let sum = 0;
  for (let i = 0; i < t.size; i++) { out.data[i] = Math.exp(t.data[i] - max); sum += out.data[i]; }
  for (let i = 0; i < t.size; i++) out.data[i] /= sum;
  return out;
}
function dot(a, b) {
  if (a.size !== b.size) throw new Error(`Dot size mismatch`);
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i] * b.data[i];
  return new NMLTensor([1], [s]);
}

class NMLEmulator {
  constructor() { this.reset(); }
  reset() {
    this.registers = {};
    for (let i = 0; i <= 9; i++) this.registers["R" + i] = null;
    "ABCDEFGHIJKLMNOPQRSTUV".split("").forEach(c => this.registers["R" + c] = null);
    this.memory = {};
    this.pc = 0;
    this.halted = false;
    this.log = [];
    this.cycles = 0;
    this.loopStack = [];
    this.callStack = [];
    this.condFlag = false;
  }

  loadMemory(addr, tensor) { this.memory[addr] = tensor; }

  _reg(name) {
    name = name.toUpperCase();
    if (!(name in this.registers)) throw new Error(`Unknown register: ${name}`);
    return name;
  }

  _getTensor(name) {
    name = this._reg(name);
    if (!this.registers[name]) throw new Error(`Register ${name} is empty`);
    return this.registers[name];
  }

  assemble(source) {
    const SYMBOLIC = {
      "×":"MMUL","⊕":"MADD","⊖":"MSUB","⊗":"EMUL","⊘":"EDIV","·":"SDOT","∗":"SCLR","÷":"SDIV","∔":"SADD","∸":"SSUB",
      "⌐":"RELU","σ":"SIGM","τ":"TANH","Σ":"SOFT","ℊ":"GELU",
      "↓":"LD","↑":"ST","←":"MOV","□":"ALLC","∎":"LEAF",
      "⊤":"TRNS","⊢":"SPLT","⊣":"MERG",
      "⋈":"CMPF","≶":"CMP","≺":"CMPI","ϟ":"CMPI",
      "↗":"JMPT","↘":"JMPF","→":"JUMP","↻":"LOOP","↺":"ENDP",
      "⇒":"CALL","⇐":"RET","∑":"TACC",
      "◼":"HALT","⚠":"TRAP","⏸":"SYNC","⚙":"SYS",
      "∇":"BKWD","⟳":"WUPD","△":"LOSS","⥁":"TNET",
      "⊞":"BN","≋":"DROP",
      "⌐ˈ":"RELUBK","σˈ":"SIGMBK","τˈ":"TANHBK","ℊˈ":"GELUBK","Σˈ":"SOFTBK",
      "×ˈ":"MMULBK","⊛ˈ":"CONVBK","⊓ˈ":"POOLBK","‖ˈ":"NORMBK","⊙ˈ":"ATTNBK","⥁ˈ":"TNDEEP",
      "⊛":"CONV","⊓":"POOL","⊔":"UPSC","⊡":"PADZ",
      "⊙":"ATTN","‖":"NORM","⊏":"EMBD",
      "⊥":"RDUC","ϛ":"RDUC","⊻":"WHER","⊧":"CLMP","⊜":"CMPR",
      "∿":"FFT","⋐":"FILT",
      "§":"META","◆":"FRAG","◇":"ENDF","⚖":"VOTE",
      "⟐":"PROJ","⟂":"DIST","⊃":"GATH","⊂":"SCAT",
      "✦":"SIGN","✓":"VRFY",
    };
    const GREEK = {
      "ι":"R0","κ":"R1","λ":"R2","μ":"R3","ν":"R4","ξ":"R5","ο":"R6","π":"R7","ρ":"R8","ς":"R9",
      "α":"RA","β":"RB","γ":"RC","δ":"RD","φ":"RE","ψ":"RF",
      "η":"RG","θ":"RH","ζ":"RI","ω":"RJ","χ":"RK","υ":"RL","ε":"RM",
    };
    const lines = source.split("\n")
      .map(l => l.replace(/;.*$/, "").trim())
      .filter(l => l.length > 0 && !l.startsWith("#"));
    return lines.map((line, idx) => {
      const parts = line.split(/\s+/);
      const rawOp = parts[0];
      const opcode = (SYMBOLIC[rawOp] || rawOp).toUpperCase();
      const operands = parts.slice(1).map(o => GREEK[o] || o);
      return { opcode, operands, line: idx, source: line };
    });
  }

  execute(program) {
    this.pc = 0;
    this.halted = false;
    this.cycles = 0;
    this.log = [];
    const maxCycles = 10000;

    while (this.pc < program.length && !this.halted && this.cycles < maxCycles) {
      const instr = program[this.pc];
      this.cycles++;
      this._exec(instr, program);
    }

    if (this.cycles >= maxCycles) this.log.push({ type: "error", msg: "Max cycles exceeded" });
    return this.log;
  }

  _exec(instr, program) {
    const { opcode, operands } = instr;
    const ops = operands;

    try {
      switch (opcode) {
        case "MMUL": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = matmul(a, b);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} @ ${ops[2]} → ${this.registers[this._reg(ops[0])]}` });
          break;
        }
        case "MADD": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = elementwise(a, b, (x, y) => x + y);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} + ${ops[2]}` });
          break;
        }
        case "MSUB": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = elementwise(a, b, (x, y) => x - y);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} - ${ops[2]}` });
          break;
        }
        case "EMUL": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = elementwise(a, b, (x, y) => x * y);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} * ${ops[2]}` });
          break;
        }
        case "SDOT": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = dot(a, b);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = dot(${ops[1]}, ${ops[2]}) → ${this.registers[this._reg(ops[0])]}` });
          break;
        }
        case "SCLR": {
          const a = this._getTensor(ops[1]);
          const val = parseFloat(ops[2].replace("#", ""));
          const out = new NMLTensor(a.shape);
          for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] * val;
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} * ${val}` });
          break;
        }
        case "SADD": case "SCALAR_ADD": {
          const a = this._getTensor(ops[1]);
          const val = ops[2].startsWith("#") ? parseFloat(ops[2].replace("#", "")) : this._getTensor(ops[2]).data[0];
          const out = new NMLTensor(a.shape);
          for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] + val;
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} + ${val}` });
          break;
        }
        case "SSUB": case "SCALAR_SUB": {
          const a = this._getTensor(ops[1]);
          const val = ops[2].startsWith("#") ? parseFloat(ops[2].replace("#", "")) : this._getTensor(ops[2]).data[0];
          const out = new NMLTensor(a.shape);
          for (let i = 0; i < a.size; i++) out.data[i] = a.data[i] - val;
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} - ${val}` });
          break;
        }
        case "RELU": {
          const t = this._getTensor(ops[1]);
          this.registers[this._reg(ops[0])] = relu(t);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = relu(${ops[1]})` });
          break;
        }
        case "SIGM": {
          const t = this._getTensor(ops[1]);
          this.registers[this._reg(ops[0])] = sigmoid(t);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = sigmoid(${ops[1]})` });
          break;
        }
        case "TANH": {
          const t = this._getTensor(ops[1]);
          this.registers[this._reg(ops[0])] = tanh_act(t);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = tanh(${ops[1]})` });
          break;
        }
        case "SOFT": {
          const t = this._getTensor(ops[1]);
          this.registers[this._reg(ops[0])] = softmax(t);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = softmax(${ops[1]}) → ${this.registers[this._reg(ops[0])]}` });
          break;
        }
        case "LD": case "LOAD": {
          const src = ops[1];
          if (src.startsWith("@")) {
            const addr = src.replace("@", "");
            if (!(addr in this.memory)) throw new Error(`Address @${addr} not found`);
            this.registers[this._reg(ops[0])] = this.memory[addr].clone();
            this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} ← @${addr} ${this.registers[this._reg(ops[0])]}` });
          } else if (src.startsWith("#") || /^-?\d/.test(src)) {
            const val = parseFloat(src.replace("#", ""));
            this.registers[this._reg(ops[0])] = new NMLTensor([1], [val]);
            this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${val}` });
          } else {
            const t = this._getTensor(src);
            this.registers[this._reg(ops[0])] = t.clone();
            this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${src}` });
          }
          break;
        }
        case "ST": {
          const t = this._getTensor(ops[0]);
          const addr = ops[1].replace("@", "");
          this.memory[addr] = t.clone();
          this.log.push({ type: "exec", op: opcode, msg: `@${addr} ← ${ops[0]} ${t}` });
          break;
        }
        case "MOV": case "COPY": {
          const src = ops[1];
          if (src.startsWith("@")) {
            const addr = src.replace("@", "");
            if (!(addr in this.memory)) throw new Error(`Address @${addr} not found`);
            this.registers[this._reg(ops[0])] = this.memory[addr].clone();
          } else if (src.startsWith("#") || /^-?\d/.test(src)) {
            const val = parseFloat(src.replace("#", ""));
            this.registers[this._reg(ops[0])] = new NMLTensor([1], [val]);
          } else {
            this.registers[this._reg(ops[0])] = this._getTensor(src).clone();
          }
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]}` });
          break;
        }
        case "ALLC": {
          const shape = ops[1].replace("#", "").replace("[", "").replace("]", "").split(",").map(Number);
          this.registers[this._reg(ops[0])] = new NMLTensor(shape);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = alloc(${shape})` });
          break;
        }
        case "RSHP": {
          const t = this._getTensor(ops[1]);
          const shape = ops[2].replace("#", "").replace("[", "").replace("]", "").split(",").map(Number);
          const out = new NMLTensor(shape, t.data);
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = reshape(${ops[1]}, ${shape})` });
          break;
        }
        case "TRNS": {
          const t = this._getTensor(ops[1]);
          if (t.shape.length !== 2) throw new Error("TRNS requires 2D tensor");
          const [r, c] = t.shape;
          const out = new NMLTensor([c, r]);
          for (let i = 0; i < r; i++)
            for (let j = 0; j < c; j++)
              out.data[j * r + i] = t.data[i * c + j];
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = transpose(${ops[1]})` });
          break;
        }
        case "LOOP": {
          const count = parseInt(ops[0].replace("#", ""));
          this.loopStack.push({ start: this.pc, count, current: 0 });
          this.log.push({ type: "exec", op: opcode, msg: `loop ${count}×` });
          break;
        }
        case "ENDP": {
          if (this.loopStack.length === 0) throw new Error("ENDP without LOOP");
          const loop = this.loopStack[this.loopStack.length - 1];
          loop.current++;
          if (loop.current < loop.count) {
            this.pc = loop.start;
            this.log.push({ type: "exec", op: opcode, msg: `loop iter ${loop.current + 1}/${loop.count}` });
          } else {
            this.loopStack.pop();
            this.log.push({ type: "exec", op: opcode, msg: `loop complete` });
          }
          break;
        }
        case "SYNC": {
          this.log.push({ type: "exec", op: opcode, msg: `sync barrier` });
          break;
        }
        case "HALT": {
          this.halted = true;
          this.log.push({ type: "halt", op: opcode, msg: `HALTED after ${this.cycles} cycles` });
          return;
        }
        case "LEAF": case "SET_VALUE": {
          const src = ops[1];
          if (src.startsWith("@")) {
            const addr = src.replace("@", "");
            if (!(addr in this.memory)) throw new Error(`Address @${addr} not found`);
            this.registers[this._reg(ops[0])] = this.memory[addr].clone();
          } else if (/^R[0-9A-Fa-f]$/i.test(src)) {
            this.registers[this._reg(ops[0])] = this._getTensor(src).clone();
          } else {
            const val = parseFloat(src.replace("#", ""));
            this.registers[this._reg(ops[0])] = new NMLTensor([1], [val]);
          }
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]}` });
          break;
        }
        case "TACC": case "ACCUMULATE": {
          let a, b;
          if (ops.length >= 3) {
            a = this._getTensor(ops[1]).data[0];
            b = this._getTensor(ops[2]).data[0];
          } else {
            a = this.registers[this._reg(ops[0])] ? this.registers[this._reg(ops[0])].data[0] : 0;
            b = this._getTensor(ops[1]).data[0];
          }
          this.registers[this._reg(ops[0])] = new NMLTensor([1], [a + b]);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${a} + ${b} = ${a + b}` });
          break;
        }
        case "SDIV": {
          const t = this._getTensor(ops[1]);
          const val = parseFloat(ops[2].replace("#", ""));
          const out = new NMLTensor(t.shape);
          for (let i = 0; i < t.size; i++) out.data[i] = t.data[i] / val;
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} / ${val}` });
          break;
        }
        case "EDIV": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = elementwise(a, b, (x, y) => x / y);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${ops[1]} / ${ops[2]}` });
          break;
        }
        case "CMPI": case "COMPARE_VALUE": {
          let regVal, imm;
          if (ops.length >= 3) {
            regVal = this._getTensor(ops[1]).data[0];
            imm = parseFloat(ops[2].replace("#", ""));
          } else {
            regVal = this._getTensor(ops[0]).data[0];
            imm = parseFloat(ops[1].replace("#", ""));
          }
          this.condFlag = regVal < imm;
          this.log.push({ type: "exec", op: opcode, msg: `flag = ${regVal} < ${imm} → ${this.condFlag}` });
          break;
        }
        case "CMPF": {
          const t = this._getTensor(ops[1]);
          const feat = parseInt(ops[2].replace("#", ""));
          const thresh = parseFloat(ops[3].replace("#", ""));
          this.condFlag = t.data[feat] < thresh;
          this.log.push({ type: "exec", op: opcode, msg: `flag = feat[${feat}] < ${thresh} → ${this.condFlag}` });
          break;
        }
        case "CMP": {
          const a = this._getTensor(ops[0]).data[0];
          const b = this._getTensor(ops[1]).data[0];
          this.condFlag = a < b;
          this.log.push({ type: "exec", op: opcode, msg: `flag = ${a} < ${b} → ${this.condFlag}` });
          break;
        }
        case "JMPT": {
          const offset = parseInt(ops[0].replace("#", ""));
          if (this.condFlag) { this.pc += offset; }
          this.log.push({ type: "exec", op: opcode, msg: `flag=${this.condFlag}, ${this.condFlag ? `jump +${offset}` : "no jump"}` });
          break;
        }
        case "JMPF": {
          const offset = parseInt(ops[0].replace("#", ""));
          if (!this.condFlag) { this.pc += offset; }
          this.log.push({ type: "exec", op: opcode, msg: `flag=${this.condFlag}, ${!this.condFlag ? `jump +${offset}` : "no jump"}` });
          break;
        }
        case "JUMP": case "JMP": {
          const offset = parseInt(ops[0].replace("#", ""));
          this.pc += offset;
          this.log.push({ type: "exec", op: opcode, msg: `jump +${offset}` });
          break;
        }
        case "CALL": {
          const offset = parseInt(ops[0].replace("#", ""));
          if (!this.callStack) this.callStack = [];
          this.callStack.push(this.pc + 1);
          this.pc += offset;
          this.log.push({ type: "exec", op: opcode, msg: `call +${offset}, return to PC=${this.pc + 1}` });
          break;
        }
        case "RET": {
          if (!this.callStack || this.callStack.length === 0) throw new Error("RET with empty call stack");
          this.pc = this.callStack.pop() - 1;
          this.log.push({ type: "exec", op: opcode, msg: `return to PC=${this.pc + 1}` });
          break;
        }
        case "TRAP": {
          const code = ops[0] ? parseInt(ops[0].replace("#", "")) : 1;
          throw new Error(`TRAP #${code}`);
        }
        case "SYS": {
          const code = parseInt(ops[1].replace("#", ""));
          const t = this._getTensor(ops[0]);
          if (code === 0) this.log.push({ type: "exec", op: opcode, msg: `print: ${t.data[0]}` });
          else if (code === 1) this.log.push({ type: "exec", op: opcode, msg: `char: ${String.fromCharCode(t.data[0])}` });
          else this.log.push({ type: "exec", op: opcode, msg: `sys #${code}` });
          break;
        }
        case "TNET": {
          /* Mini-batch SGD training loop.
           * Registers: R0=input(N×K) R1=w1(K×H) R2=b1(1×H)
           *            R3=w2(H×1)   R4=b2(1×1)  R9=target(N×1)
           * Immediates: #epochs #lr #loss_type #batch_size
           * Writes: R1 R2 R3 R4 (updated weights), R8 (final loss) */
          const epochs    = ops[0] ? parseInt(ops[0].replace("#", ""))   : 100;
          const lr        = ops[1] ? parseFloat(ops[1].replace("#", "")) : 0.01;
          const bsArg     = ops[3] ? parseInt(ops[3].replace("#", ""))   : 0;

          const X  = this._getTensor("R0");
          const w1 = this._getTensor("R1");
          const b1 = this._getTensor("R2");
          const w2 = this._getTensor("R3");
          const b2 = this._getTensor("R4");
          const yT = this._getTensor("R9");

          const N = X.shape[0];
          const K = w1.shape[0];
          const H = w1.shape[w1.shape.length - 1];
          const B = (bsArg > 0 && bsArg < N) ? bsArg : Math.min(N, 64);

          const W1 = w1.data.slice(), B1 = b1.data.slice();
          const W2 = w2.data.slice(), B2 = b2.data.slice();

          let finalLoss = 0;
          for (let ep = 0; ep < epochs; ep++) {
            finalLoss = 0;
            const nbatch = Math.ceil(N / B);
            for (let bat = 0; bat < nbatch; bat++) {
              const s = bat * B, Bn = Math.min(B, N - s);
              // Forward
              const pre = new Float32Array(Bn * H);
              const hid = new Float32Array(Bn * H);
              for (let i = 0; i < Bn; i++)
                for (let j = 0; j < H; j++) {
                  let sum = 0;
                  for (let p = 0; p < K; p++) sum += X.data[(s+i)*K+p] * W1[p*H+j];
                  pre[i*H+j] = sum + B1[j];
                  hid[i*H+j] = Math.max(0, pre[i*H+j]);
                }
              const out = new Float32Array(Bn);
              for (let i = 0; i < Bn; i++) {
                let sum = 0;
                for (let j = 0; j < H; j++) sum += hid[i*H+j] * W2[j];
                out[i] = sum + B2[0];
              }
              // Loss (MSE) + d_out
              const dout = new Float32Array(Bn);
              for (let i = 0; i < Bn; i++) {
                const diff = out[i] - yT.data[s+i];
                finalLoss += diff * diff;
                dout[i] = 2 * diff / Bn;
              }
              // d_W2, d_B2
              const dW2 = new Float32Array(H); let dB2 = 0;
              for (let i = 0; i < Bn; i++) { for (let j = 0; j < H; j++) dW2[j] += dout[i] * hid[i*H+j]; dB2 += dout[i]; }
              // d_hidden → ReLU backward → d_pre
              const dPre = new Float32Array(Bn * H);
              for (let i = 0; i < Bn; i++)
                for (let j = 0; j < H; j++)
                  dPre[i*H+j] = pre[i*H+j] > 0 ? dout[i] * W2[j] : 0;
              // d_W1, d_B1
              const dW1 = new Float32Array(K * H); const dB1 = new Float32Array(H);
              for (let i = 0; i < Bn; i++)
                for (let j = 0; j < H; j++) {
                  for (let p = 0; p < K; p++) dW1[p*H+j] += dPre[i*H+j] * X.data[(s+i)*K+p];
                  dB1[j] += dPre[i*H+j];
                }
              // Update weights
              for (let i = 0; i < K*H; i++) W1[i] -= lr * dW1[i];
              for (let i = 0; i < H; i++)   B1[i] -= lr * dB1[i];
              for (let i = 0; i < H; i++)   W2[i] -= lr * dW2[i];
              B2[0] -= lr * dB2;
            }
            finalLoss /= N;
          }

          this.registers[this._reg("R1")] = new NMLTensor(w1.shape, W1);
          this.registers[this._reg("R2")] = new NMLTensor(b1.shape, B1);
          this.registers[this._reg("R3")] = new NMLTensor(w2.shape, W2);
          this.registers[this._reg("R4")] = new NMLTensor(b2.shape, B2);
          this.registers[this._reg("R8")] = new NMLTensor([1], [finalLoss]);
          const batchLabel = bsArg > 0 ? `batch=${B}` : `batch=${B}(auto)`;
          this.log.push({ type: "exec", op: opcode,
            msg: `trained ${epochs} epochs · ${batchLabel} · lr=${lr} → loss=${finalLoss.toFixed(6)}` });
          break;
        }
        case "DOT": {
          const a = this._getTensor(ops[1]);
          const b = this._getTensor(ops[2]);
          this.registers[this._reg(ops[0])] = dot(a, b);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = dot(${ops[1]}, ${ops[2]}) → ${this.registers[this._reg(ops[0])]}` });
          break;
        }
        case "GELU": {
          const t = this._getTensor(ops[1]);
          const out = new NMLTensor(t.shape);
          for (let i = 0; i < t.size; i++) {
            const x = t.data[i];
            out.data[i] = 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
          }
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = gelu(${ops[1]})` });
          break;
        }
        case "MOD": case "MODULO": {
          const a = this._getTensor(ops[1]).data[0];
          const b = this._getTensor(ops[2]).data[0];
          this.registers[this._reg(ops[0])] = new NMLTensor([1], [Math.trunc(a) % Math.trunc(b)]);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ${Math.trunc(a)} % ${Math.trunc(b)}` });
          break;
        }
        case "ITOF": case "INT_TO_FLOAT": {
          const v = this._getTensor(ops[1]).data[0];
          this.registers[this._reg(ops[0])] = new NMLTensor([1], [Math.trunc(v)]);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = itof(${v})` });
          break;
        }
        case "FTOI": case "FLOAT_TO_INT": {
          const v = this._getTensor(ops[1]).data[0];
          this.registers[this._reg(ops[0])] = new NMLTensor([1], [Math.trunc(v)]);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ftoi(${v}) = ${Math.trunc(v)}` });
          break;
        }
        case "BNOT": case "BITWISE_NOT": {
          const v = this._getTensor(ops[1]).data[0];
          this.registers[this._reg(ops[0])] = new NMLTensor([1], [~Math.trunc(v)]);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = ~${Math.trunc(v)}` });
          break;
        }
        case "RDUC": case "REDUCE": {
          const t = this._getTensor(ops[1]);
          const mode = parseInt((ops[2] || "#0").replace("#", ""));
          let val;
          if (mode === 0) val = t.data.reduce((a, b) => a + b, 0);
          else if (mode === 1) val = t.data.reduce((a, b) => a + b, 0) / t.size;
          else if (mode === 2) val = Math.max(...t.data);
          else val = Math.min(...t.data);
          this.registers[this._reg(ops[0])] = new NMLTensor([1], [val]);
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = reduce(${ops[1]}, mode=${mode}) = ${val.toFixed(4)}` });
          break;
        }
        case "BN": case "BATCH_NORM": {
          const src = this._getTensor(ops[1]);
          const out = new NMLTensor(src.shape);
          const mean = src.data.reduce((a, b) => a + b, 0) / src.size;
          const variance = src.data.reduce((a, b) => a + (b - mean) ** 2, 0) / src.size;
          const eps = 1e-5;
          const gamma = ops[2] ? this._getTensor(ops[2]) : null;
          const beta = ops[3] ? this._getTensor(ops[3]) : null;
          for (let i = 0; i < src.size; i++) {
            let val = (src.data[i] - mean) / Math.sqrt(variance + eps);
            if (gamma) val *= gamma.data[i % gamma.size];
            if (beta) val += beta.data[i % beta.size];
            out.data[i] = val;
          }
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = bn(${ops[1]})` });
          break;
        }
        case "DROP": case "DROPOUT": {
          const src = this._getTensor(ops[1]);
          const rate = parseFloat((ops[2] || "#0").replace("#", ""));
          const out = new NMLTensor(src.shape);
          if (rate <= 0) {
            out.data.set(src.data);
          } else {
            const scale = 1.0 / (1.0 - rate);
            for (let i = 0; i < src.size; i++) {
              out.data[i] = Math.random() >= rate ? src.data[i] * scale : 0;
            }
          }
          this.registers[this._reg(ops[0])] = out;
          this.log.push({ type: "exec", op: opcode, msg: `${ops[0]} = dropout(${ops[1]}, rate=${rate})` });
          break;
        }
        case "META": case "METADATA": case "SIGN": case "SIGN_PROGRAM":
        case "FRAG": case "FRAGMENT": case "ENDF": case "END_FRAGMENT":
        case "LINK": case "IMPORT": case "PTCH": case "PATCH":
        case "VRFY": case "VERIFY_SIGNATURE": {
          this.log.push({ type: "exec", op: opcode, msg: `${opcode} (structural, no-op)` });
          break;
        }
        default:
          this.log.push({ type: "exec", op: opcode, msg: `${opcode} (not emulated in browser)` });
          break;
      }
    } catch (e) {
      this.log.push({ type: "error", op: opcode, msg: e.message });
      this.halted = true;
      return;
    }
    this.pc++;
  }
}

// ═══════════════════════════════════════════
// EXAMPLE PROGRAMS
// ═══════════════════════════════════════════

const EXAMPLES = {
  "Dense Layer": {
    code: `; Single dense layer forward pass
; 8 instructions — the entire computation
LD    R0 @input
LD    R1 @weights
LD    R2 @bias
MMUL  R3 R0 R1
MADD  R3 R3 R2
RELU  R3 R3
ST    R3 @output
HALT`,
    memory: {
      input: { shape: [1, 4], data: [1.0, 0.5, -0.3, 0.8] },
      weights: { shape: [4, 3], data: [0.2, -0.1, 0.4, 0.5, 0.3, -0.2, -0.1, 0.6, 0.1, 0.3, -0.4, 0.5] },
      bias: { shape: [1, 3], data: [0.1, -0.1, 0.05] },
    }
  },
  "2-Layer Classifier": {
    code: `; Two-layer neural network classifier
; Intent → IR: "classify 4D input into 2 classes"
LD    R0 @input
LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
MMUL  R5 R0 R1
MADD  R5 R5 R2
RELU  R5 R5
MMUL  R6 R5 R3
MADD  R6 R6 R4
SOFT  R6 R6
ST    R6 @output
HALT`,
    memory: {
      input: { shape: [1, 4], data: [0.8, 0.2, 0.5, -0.1] },
      w1: { shape: [4, 3], data: [0.1, 0.4, -0.2, 0.3, -0.1, 0.5, 0.2, 0.3, 0.1, -0.4, 0.2, 0.3] },
      b1: { shape: [1, 3], data: [0.0, 0.1, -0.1] },
      w2: { shape: [3, 2], data: [0.5, -0.3, 0.2, 0.4, -0.1, 0.6] },
      b2: { shape: [1, 2], data: [0.0, 0.0] },
    }
  },
  "Dot Product": {
    code: `; Compute dot product of two vectors
LD    R0 @vec_a
LD    R1 @vec_b
SDOT  R2 R0 R1
ST    R2 @result
HALT`,
    memory: {
      vec_a: { shape: [4], data: [1.0, 2.0, 3.0, 4.0] },
      vec_b: { shape: [4], data: [0.5, 0.5, 0.5, 0.5] },
    }
  },
  "Activation Comparison": {
    code: `; Compare all activation functions on same input
LD    R0 @input
RELU  R1 R0
SIGM  R2 R0
TANH  R3 R0
SOFT  R4 R0
ST    R1 @relu_out
ST    R2 @sigm_out
ST    R3 @tanh_out
ST    R4 @soft_out
HALT`,
    memory: {
      input: { shape: [5], data: [-2.0, -1.0, 0.0, 1.0, 2.0] },
    }
  },
  "Matrix Transpose": {
    code: `; Transpose and multiply
LD    R0 @matrix
TRNS  R1 R0
MMUL  R2 R0 R1
ST    R2 @result
HALT`,
    memory: {
      matrix: { shape: [2, 3], data: [1, 2, 3, 4, 5, 6] },
    }
  },
  "Subroutine": {
    code: `; Double a value using a subroutine
LEAF  R0 #7.0
CALL  #2
ST    R0 @result
HALT
SCLR  R0 R0 #2.0
RET`,
    memory: {}
  },
  "Fibonacci": {
    code: `; Print first 10 Fibonacci numbers
LEAF  R0 #0.0
LEAF  R1 #1.0
LEAF  RD #0.0
LEAF  R5 #10.0
SYS   R0 #0
TACC  R2 R0 R1
MOV   R0 R1
MOV   R1 R2
LEAF  RC #1.0
TACC  RD RD RC
CMP   RD R5
JMPT  #-8
HALT`,
    memory: {}
  },
  "TNET Training": {
    code: `; Mini-batch SGD — TNET #epochs #lr #loss #batch_size
; 4th immediate sets mini-batch size (0 = full-batch / auto)
LD    R1 @w1
LD    R2 @b1
LD    R3 @w2
LD    R4 @b2
LD    R0 @training_data
LD    R9 @training_labels
TNET  #300 #0.0500 #0 #4
ST    R8 @loss
HALT`,
    memory: {
      training_data:   { shape: [8, 2], data: [0.1,0.2, 0.3,0.4, 0.5,0.1, 0.7,0.8, 0.2,0.9, 0.4,0.3, 0.6,0.5, 0.8,0.1] },
      training_labels: { shape: [8, 1], data: [0, 0, 0, 1, 0, 0, 1, 1] },
      w1: { shape: [2, 4], data: [ 0.1,-0.2, 0.3, 0.1, -0.1, 0.4,-0.2, 0.2] },
      b1: { shape: [1, 4], data: [0, 0, 0, 0] },
      w2: { shape: [4, 1], data: [0.2,-0.1, 0.3, 0.1] },
      b2: { shape: [1, 1], data: [0] },
    }
  },
  "Symbolic": {
    code: `; Symbolic syntax: scale and store
∎  ι  #42.0
∗  κ  ι  #3.14
↑  κ  @result
◼`,
    memory: {}
  },
  "Batch Norm + Dropout": {
    code: `; Regularization: BN + Dropout pipeline
LD    R0 @input
LD    R1 @gamma
LD    R2 @beta
BN    R3 R0 R1 R2
DROP  R3 R3 #0.2
RELU  R3 R3
ST    R3 @output
HALT`,
    memory: {
      input: { shape: [1, 4], data: [0.5, -1.2, 0.8, 2.1] },
      gamma: { shape: [1, 4], data: [1.0, 1.0, 1.0, 1.0] },
      beta: { shape: [1, 4], data: [0.0, 0.0, 0.0, 0.0] },
    }
  },
  "Backward Pass": {
    code: `; Manual backward pass through ReLU
LD    R0 @input
LD    R1 @grad_out
RELU  R2 R0
RELUBK R3 R1 R0
ST    R2 @fwd_out
ST    R3 @grad_in
HALT`,
    memory: {
      input: { shape: [1, 4], data: [-0.5, 0.3, -0.1, 0.8] },
      grad_out: { shape: [1, 4], data: [1.0, 1.0, 1.0, 1.0] },
    }
  },
};

// ═══════════════════════════════════════════
// REACT UI
// ═══════════════════════════════════════════

const CLASSIC_TO_SYMBOLIC = {
  "MMUL":"×","MADD":"⊕","MSUB":"⊖","EMUL":"⊗","EDIV":"⊘","SDOT":"·","SCLR":"∗","SDIV":"÷","SADD":"∔","SSUB":"∸",
  "RELU":"⌐","SIGM":"σ","TANH":"τ","SOFT":"Σ","GELU":"ℊ",
  "LD":"↓","ST":"↑","MOV":"←","ALLC":"□","LEAF":"∎",
  "RSHP":"⊟","TRNS":"⊤","SPLT":"⊢","MERG":"⊣",
  "CMP":"≶","CMPI":"≺","CMPF":"⋈",
  "JMPT":"↗","JMPF":"↘","JUMP":"→","LOOP":"↻","ENDP":"↺",
  "CALL":"⇒","RET":"⇐","TACC":"∑",
  "SYNC":"⏸","HALT":"◼","TRAP":"⚠",
  "CONV":"⊛","POOL":"⊓","UPSC":"⊔","PADZ":"⊡",
  "ATTN":"⊙","NORM":"‖","EMBD":"⊏","GELU":"ℊ",
  "RDUC":"⊥","WHER":"⊻","CLMP":"⊧","CMPR":"⊜",
  "FFT":"∿","FILT":"⋐",
  "META":"§","FRAG":"◆","ENDF":"◇","LINK":"⊕","PTCH":"⊿",
  "SIGN":"✦","VRFY":"✓","VOTE":"⚖","PROJ":"⟐","DIST":"⟂","GATH":"⊃","SCAT":"⊂",
  "BKWD":"∇","WUPD":"⟳","LOSS":"△","TNET":"⥁","TNDEEP":"⥁ˈ",
  "RELUBK":"⌐ˈ","SIGMBK":"σˈ","TANHBK":"τˈ","GELUBK":"ℊˈ","SOFTBK":"Σˈ",
  "MMULBK":"×ˈ","CONVBK":"⊛ˈ","POOLBK":"⊓ˈ","NORMBK":"‖ˈ","ATTNBK":"⊙ˈ",
  "TLOG":"⧖","TRAIN":"⟴","INFER":"⟶","WDECAY":"ω",
  "BN":"⊞","DROP":"≋",
  "SYS":"⚙","MOD":"%","ITOF":"⊶","FTOI":"⊷","BNOT":"¬",
};
const SYMBOLIC_TO_CLASSIC = Object.fromEntries(
  Object.entries(CLASSIC_TO_SYMBOLIC).map(([k, v]) => [v, k])
);
const REGISTER_TO_GREEK = {
  "R0":"ι","R1":"κ","R2":"λ","R3":"μ","R4":"ν","R5":"ξ","R6":"ο","R7":"π","R8":"ρ","R9":"ς",
  "RA":"α","RB":"β","RC":"γ","RD":"δ","RE":"φ","RF":"ψ",
  "RG":"η","RH":"θ","RI":"ζ","RJ":"ω","RK":"χ","RL":"υ","RM":"ε",
};
const GREEK_TO_REGISTER = Object.fromEntries(
  Object.entries(REGISTER_TO_GREEK).map(([k, v]) => [v, k])
);

function convertSyntax(code, toSymbolic) {
  return code.split("\n").map(line => {
    const commentIdx = line.indexOf(";");
    if (commentIdx === 0) return line;
    let codePart = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
    const comment = commentIdx >= 0 ? line.slice(commentIdx) : "";
    const tokens = codePart.split(/(\s+)/);
    const converted = tokens.map(token => {
      const t = token.trim();
      if (!t) return token;
      if (toSymbolic) {
        const upper = t.toUpperCase();
        if (CLASSIC_TO_SYMBOLIC[upper]) return token.replace(t, CLASSIC_TO_SYMBOLIC[upper]);
        if (REGISTER_TO_GREEK[upper]) return token.replace(t, REGISTER_TO_GREEK[upper]);
      } else {
        if (SYMBOLIC_TO_CLASSIC[t]) return token.replace(t, SYMBOLIC_TO_CLASSIC[t]);
        if (GREEK_TO_REGISTER[t]) return token.replace(t, GREEK_TO_REGISTER[t]);
        // Handle backward primed symbols like ⌐ˈ
        if (t.length === 2 && SYMBOLIC_TO_CLASSIC[t]) return token.replace(t, SYMBOLIC_TO_CLASSIC[t]);
      }
      return token;
    });
    return converted.join("") + comment;
  }).join("\n");
}

const OPCODE_DOCS = [
{ cat: "Arithmetic", ops: [
{ op: "MMUL", sym: "×", desc: "Matrix multiply: Rd = Rs1 @ Rs2", schema: "Rd Rs1 Rs2" },
{ op: "MADD", sym: "⊕", desc: "Element-wise add: Rd = Rs1 + Rs2", schema: "Rd Rs1 Rs2" },
{ op: "MSUB", sym: "⊖", desc: "Element-wise subtract: Rd = Rs1 - Rs2", schema: "Rd Rs1 Rs2" },
{ op: "EMUL", sym: "⊗", desc: "Element-wise multiply: Rd = Rs1 * Rs2", schema: "Rd Rs1 Rs2" },
{ op: "EDIV", sym: "⊘", desc: "Element-wise divide: Rd = Rs1 / Rs2", schema: "Rd Rs1 Rs2" },
{ op: "SDOT", sym: "·", desc: "Dot product: Rd = Rs1 · Rs2", schema: "Rd Rs1 Rs2" },
{ op: "SCLR", sym: "∗", desc: "Scalar multiply: Rd = Rs * #imm", schema: "Rd Rs #imm" },
{ op: "SDIV", sym: "÷", desc: "Scalar divide: Rd = Rs1 / Rs2|#imm", schema: "Rd Rs1 Rs2|#imm" },
{ op: "SADD", sym: "∔", desc: "Scalar add: Rd = Rs + Rs2|#imm", schema: "Rd Rs Rs2|#imm" },
{ op: "SSUB", sym: "∸", desc: "Scalar subtract: Rd = Rs - Rs2|#imm", schema: "Rd Rs Rs2|#imm" },
]},
{ cat: "Activation", ops: [
{ op: "RELU", sym: "⌐", desc: "Rectified linear unit: Rd = max(0, Rs)", schema: "Rd Rs" },
{ op: "SIGM", sym: "σ", desc: "Sigmoid: Rd = 1/(1+exp(-Rs))", schema: "Rd Rs" },
{ op: "TANH", sym: "τ", desc: "Hyperbolic tangent: Rd = tanh(Rs)", schema: "Rd Rs" },
{ op: "SOFT", sym: "Σ", desc: "Softmax: Rd = softmax(Rs)", schema: "Rd Rs" },
{ op: "GELU", sym: "ℊ", desc: "Gaussian error linear unit", schema: "Rd Rs" },
]},
{ cat: "Memory", ops: [
{ op: "LD", sym: "↓", desc: "Load tensor from memory", schema: "Rd @name" },
{ op: "ST", sym: "↑", desc: "Store tensor to memory", schema: "Rs @name" },
{ op: "MOV", sym: "←", desc: "Copy register: Rd = Rs", schema: "Rd Rs" },
{ op: "ALLC", sym: "□", desc: "Allocate zero tensor", schema: "Rd #[shape]" },
]},
{ cat: "Data Flow", ops: [
{ op: "RSHP", sym: "⊟", desc: "Reshape tensor", schema: "Rd Rs [#shape]" },
{ op: "TRNS", sym: "⊤", desc: "Transpose tensor", schema: "Rd [Rs]" },
{ op: "SPLT", sym: "⊢", desc: "Split tensor along dim", schema: "Rd Re Rs #dim" },
{ op: "MERG", sym: "⊣", desc: "Merge tensors along dim", schema: "Rd Rs1 Rs2 #dim" },
]},
{ cat: "Comparison", ops: [
{ op: "CMP", sym: "≶", desc: "Compare two registers", schema: "Rs1 Rs2" },
{ op: "CMPI", sym: "≺", desc: "Compare register vs immediate", schema: "Rd Rs #imm" },
{ op: "CMPF", sym: "⋈", desc: "Feature comparison", schema: "Rd Rs #feat #thresh" },
]},
{ cat: "Control Flow", ops: [
{ op: "JMPT", sym: "↗", desc: "Jump if flag true", schema: "#offset" },
{ op: "JMPF", sym: "↘", desc: "Jump if flag false", schema: "#offset" },
{ op: "JUMP", sym: "→", desc: "Unconditional jump", schema: "#offset" },
{ op: "LOOP", sym: "↻", desc: "Begin counted loop", schema: "Rs|#count" },
{ op: "ENDP", sym: "↺", desc: "End of loop body", schema: "" },
]},
{ cat: "Subroutine", ops: [
{ op: "CALL", sym: "⇒", desc: "Call subroutine at offset", schema: "#offset" },
{ op: "RET", sym: "⇐", desc: "Return from subroutine", schema: "" },
]},
{ cat: "Tree", ops: [
{ op: "LEAF", sym: "∎", desc: "Load immediate constant", schema: "Rd #value" },
{ op: "TACC", sym: "∑", desc: "Accumulate: Rd = Rd + Rs", schema: "Rd Rs1 [Rs2]" },
]},
{ cat: "System", ops: [
{ op: "SYNC", sym: "⏸", desc: "Barrier synchronization", schema: "" },
{ op: "HALT", sym: "◼", desc: "Terminate execution", schema: "" },
{ op: "TRAP", sym: "⚠", desc: "Trigger runtime fault", schema: "[#code]" },
]},
{ cat: "Vision", ops: [
{ op: "CONV", sym: "⊛", desc: "2D convolution", schema: "Rd Rs Rk [#s] [#p]" },
{ op: "POOL", sym: "⊓", desc: "Max pooling", schema: "Rd Rs [#size] [#stride]" },
{ op: "UPSC", sym: "⊔", desc: "Upscale tensor", schema: "Rd Rs [#factor]" },
{ op: "PADZ", sym: "⊡", desc: "Zero-pad tensor", schema: "Rd Rs [#amount]" },
]},
{ cat: "Transformer", ops: [
{ op: "ATTN", sym: "⊙", desc: "Multi-head attention", schema: "Rd Rq Rk [Rv]" },
{ op: "NORM", sym: "‖", desc: "Layer normalization", schema: "Rd Rs [Rg] [Rb]" },
{ op: "EMBD", sym: "⊏", desc: "Embedding lookup", schema: "Rd Rtable Rindex" },
]},
{ cat: "Reduction", ops: [
{ op: "RDUC", sym: "⊥", desc: "Reduce along dimension", schema: "Rd Rs [#dim] [#mode]" },
{ op: "WHER", sym: "⊻", desc: "Conditional select", schema: "Rd Rcond Rs1 [Rs2]" },
{ op: "CLMP", sym: "⊧", desc: "Clamp values", schema: "Rd Rs #min #max" },
{ op: "CMPR", sym: "⊜", desc: "Mask comparison", schema: "Rd Rs #op #thresh" },
]},
{ cat: "Signal", ops: [
{ op: "FFT", sym: "∿", desc: "Fast Fourier Transform", schema: "Rd Rs Rtwiddle" },
{ op: "FILT", sym: "⋐", desc: "Apply filter kernel", schema: "Rd Rs Rk [#mode]" },
]},
{ cat: "M2M", ops: [
{ op: "META", sym: "§", desc: "Metadata annotation", schema: "@key \"val\"" },
{ op: "FRAG", sym: "◆", desc: "Begin named fragment", schema: "name" },
{ op: "ENDF", sym: "◇", desc: "End fragment", schema: "" },
{ op: "LINK", sym: "⊕", desc: "Import fragment", schema: "@name" },
{ op: "PTCH", sym: "⊿", desc: "Apply differential patch", schema: "" },
{ op: "SIGN", sym: "✦", desc: "Cryptographically sign", schema: "agent=..." },
{ op: "VRFY", sym: "✓", desc: "Verify signature", schema: "@prog @pubkey" },
{ op: "VOTE", sym: "⚖", desc: "Consensus voting", schema: "Rd Rs #strat [#thr]" },
{ op: "PROJ", sym: "⟐", desc: "Linear projection", schema: "Rd Rs1 Rs2" },
{ op: "DIST", sym: "⟂", desc: "Distance metric", schema: "Rd Rs1 Rs2 [#m]" },
{ op: "GATH", sym: "⊃", desc: "Gather: Rd = Rs[Ridx]", schema: "Rd Rs Rindex" },
{ op: "SCAT", sym: "⊂", desc: "Scatter: Rs -> Rd", schema: "Rs Rd Rindex" },
]},
{ cat: "Training", ops: [
{ op: "BKWD", sym: "∇", desc: "Backward pass: compute gradients", schema: "Rg Ra Rl [Rm]" },
{ op: "WUPD", sym: "⟳", desc: "Weight update: W -= lr * grad", schema: "Rw Rg Rlr [Rmom]" },
{ op: "LOSS", sym: "△", desc: "Compute loss", schema: "Rd Rpred Rlbl [#t]" },
{ op: "TNET", sym: "⥁", desc: "Train network loop", schema: "#ep #lr #loss #bs" },
{ op: "TNDEEP", sym: "⥁ˈ", desc: "N-layer dense training", schema: "#ep #lr #opt" },
{ op: "TLOG", sym: "⧖", desc: "Set log interval", schema: "#n" },
{ op: "TRAIN", sym: "⟴", desc: "Config-driven training", schema: "Rs [@d] [@l]" },
{ op: "INFER", sym: "⟶", desc: "Forward pass only", schema: "Rd R_in" },
{ op: "WDECAY", sym: "ω", desc: "Weight decay", schema: "Rd #lambda" },
{ op: "BN", sym: "⊞", desc: "Batch normalization", schema: "Rd Rs [Rg] [Rb]" },
{ op: "DROP", sym: "≋", desc: "Inverted dropout", schema: "Rd Rs [#rate]" },
]},
{ cat: "Backward", ops: [
{ op: "RELUBK", sym: "⌐ˈ", desc: "ReLU backward", schema: "Rd Rg Rin" },
{ op: "SIGMBK", sym: "σˈ", desc: "Sigmoid backward", schema: "Rd Rg Rin" },
{ op: "TANHBK", sym: "τˈ", desc: "Tanh backward", schema: "Rd Rg Rin" },
{ op: "GELUBK", sym: "ℊˈ", desc: "GELU backward", schema: "Rd Rg Rin" },
{ op: "SOFTBK", sym: "Σˈ", desc: "Softmax backward", schema: "Rd Rg Rin" },
{ op: "MMULBK", sym: "×ˈ", desc: "Matmul backward", schema: "Rdi Rdw Rg Ri Rw" },
{ op: "CONVBK", sym: "⊛ˈ", desc: "Conv backward", schema: "Rdi Rdk Rg Ri Rk" },
{ op: "POOLBK", sym: "⊓ˈ", desc: "Pool backward", schema: "Rd Rg Rf [#s] [#st]" },
{ op: "NORMBK", sym: "‖ˈ", desc: "LayerNorm backward", schema: "Rd Rg Rin" },
{ op: "ATTNBK", sym: "⊙ˈ", desc: "Attention backward", schema: "Rdq Rg Rq Rk Rv" },
]},
{ cat: "General", ops: [
{ op: "SYS", sym: "⚙", desc: "System call", schema: "Rd #code" },
{ op: "MOD", sym: "%", desc: "Modulo: Rd = Rs1 % Rs2", schema: "Rd Rs1 Rs2" },
{ op: "ITOF", sym: "⊶", desc: "Int to float", schema: "Rd Rs" },
{ op: "FTOI", sym: "⊷", desc: "Float to int", schema: "Rd Rs" },
{ op: "BNOT", sym: "¬", desc: "Bitwise NOT", schema: "Rd Rs" },
]},
];

const OPCODE_TOOLTIP = {};
OPCODE_DOCS.forEach(cat => cat.ops.forEach(o => {
  OPCODE_TOOLTIP[o.op] = o;
  if (o.sym) OPCODE_TOOLTIP[o.sym] = o;
}));

const FONT = "'IBM Plex Mono', 'Fira Code', monospace";

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
  opcode: "#00e5ff",
  register: "#ff6bef",
  address: "#ffcc00",
  comment: "#3a3a52",
  halt: "#00ff9d",
};

function syntaxHighlight(line) {
  const commentIdx = line.indexOf(";");
  let code = line, comment = "";
  if (commentIdx >= 0) {
    code = line.slice(0, commentIdx);
    comment = line.slice(commentIdx);
  }
  const parts = code.split(/(\s+)/);
  const highlighted = parts.map((p, i) => {
    const upper = p.toUpperCase().trim();
    if (["MMUL","MADD","MSUB","EMUL","SDOT","DOT","SCLR","SDIV","SADD","SSUB","EDIV","RELU","SIGM","TANH","SOFT","GELU",
         "LD","ST","MOV","ALLC","RSHP","TRNS","SPLT","MERG",
         "CMPF","CMP","CMPI","JMPT","JMPF","JUMP","JMP","LOOP","ENDP",
         "CALL","RET","LEAF","TACC","SYNC","HALT","TRAP",
         "CONV","POOL","UPSC","PADZ","ATTN","NORM","EMBD",
         "RDUC","WHER","CLMP","CMPR","FFT","FILT",
         "META","FRAG","ENDF","LINK","PTCH","SIGN","VRFY","VOTE","PROJ","DIST","GATH","SCAT","SCTR",
         "TNET","BKWD","WUPD","LOSS","TNDEEP",
         "RELUBK","SIGMBK","TANHBK","GELUBK","SOFTBK","MMULBK","CONVBK","POOLBK","NORMBK","ATTNBK",
         "TLOG","TRAIN","INFER","WDECAY","BN","DROP",
         "SYS","MOD","ITOF","FTOI","BNOT"].includes(upper)) {
      const tip = OPCODE_TOOLTIP[upper];
      return <span key={i} style={{ color: COLORS.opcode, fontWeight: 700, position: "relative", cursor: "help" }}
        title={tip ? `${tip.op} (${tip.sym})  ${tip.schema}\n${tip.desc}` : upper}>{p}</span>;
    }
    if ("×⊕⊖⊗⊘·∗÷∔∸⌐στΣℊ↓↑←□∎⊤⊢⊣⋈≶≺ϟ↗↘→↻↺⇒⇐∑◼⚠⏸⚙∇⟳△⥁⊞≋⊛⊓⊔⊡⊙‖⊏⊥ϛ⊻⊧⊜∿⋐§◆◇⚖⟐⟂⊃⊂✦✓%¬⊶⊷⊟⧖⟴⟶ω".includes(p.trim()) ||
        /^[⌐στℊΣ×⊛⊓‖⊙⥁]ˈ$/.test(p.trim())) {
      const tip = OPCODE_TOOLTIP[p.trim()];
      return <span key={i} style={{ color: COLORS.opcode, fontWeight: 700, position: "relative", cursor: "help" }}
        title={tip ? `${tip.op} (${tip.sym})  ${tip.schema}\n${tip.desc}` : p.trim()}>{p}</span>;
    }
    if (/^R[0-9A-Fa-f]$/i.test(p.trim()) || /^[ικλμνξοπρςαβγδφψ]$/i.test(p.trim())) {
      return <span key={i} style={{ color: COLORS.register }}>{p}</span>;
    }
    if (p.trim().startsWith("@")) {
      return <span key={i} style={{ color: COLORS.address }}>{p}</span>;
    }
    if (p.trim().startsWith("#")) {
      return <span key={i} style={{ color: COLORS.warn }}>{p}</span>;
    }
    return <span key={i}>{p}</span>;
  });
  return (
    <>
      {highlighted}
      {comment && <span style={{ color: COLORS.comment }}>{comment}</span>}
    </>
  );
}

export default function NMLTerminal() {
  const [code, setCode] = useState(EXAMPLES["Dense Layer"].code);
  const [output, setOutput] = useState([]);
  const [registers, setRegisters] = useState({});
  const [memoryView, setMemoryView] = useState({});
  const [selectedExample, setSelectedExample] = useState("Dense Layer");
  const [stats, setStats] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("output");
  const [leftTab, setLeftTab] = useState("source"); // "source" | "input"
  const [syntaxMode, setSyntaxMode] = useState("classic"); // "classic" | "symbolic"
  const [showDocs, setShowDocs] = useState(false);
  const [docsFilter, setDocsFilter] = useState("");
  const [hoverTip, setHoverTip] = useState(null); // { x, y, doc }
  const [inputData, setInputData] = useState(EXAMPLES["Dense Layer"].memory || {});
  const hoverTimeout = useRef(null);
  const outputRef = useRef(null);
  const textareaRef = useRef(null);
  const gutterRef = useRef(null);

  useEffect(() => {
    if (outputRef.current) outputRef.current.scrollTop = outputRef.current.scrollHeight;
  }, [output]);


  // Auto-detect LD @name slots from code (debounced) and sync input entries
  const syncTimeout = useRef(null);
  useEffect(() => {
    clearTimeout(syncTimeout.current);
    syncTimeout.current = setTimeout(() => {
      const ldPattern = /(?:^|\n)\s*(?:LD|↓)\s+\S+\s+@(\S+)/gi;
      const slots = new Set();
      let m;
      while ((m = ldPattern.exec(code)) !== null) slots.add(m[1]);
      setInputData(prev => {
        const next = {};
        let changed = false;
        // Keep existing data for slots still in code
        for (const name of slots) {
          if (name in prev) {
            next[name] = prev[name];
          } else {
            next[name] = { shape: [1, 4], data: [0, 0, 0, 0] };
            changed = true;
          }
        }
        // Check if any old slots were removed
        for (const name of Object.keys(prev)) {
          if (!slots.has(name)) changed = true;
        }
        return changed ? next : prev;
      });
    }, 800);
    return () => clearTimeout(syncTimeout.current);
  }, [code]);

  const runProgram = useCallback(() => {
    setIsRunning(true);
    const emu = new NMLEmulator();

    if (inputData) {
      Object.entries(inputData).forEach(([addr, { shape, data }]) => {
        emu.loadMemory(addr, new NMLTensor(shape, data));
      });
    }

    const t0 = performance.now();
    const program = emu.assemble(code);
    const log = emu.execute(program);
    const elapsed = performance.now() - t0;

    const tokenCount = code.split(/\s+/).filter(t => t.length > 0 && !t.startsWith(";")).length;
    const instrCount = program.length;

    setOutput(log);
    setRegisters({ ...emu.registers });
    setMemoryView({ ...emu.memory });
    /* Check if a TNET instruction ran and extract loss for display */
    const tnetEntry = log.find(e => e.op === "TNET");
    const tnetLoss = tnetEntry ? tnetEntry.msg.match(/loss=([\d.]+)/) : null;

    setStats({
      cycles: emu.cycles,
      instructions: instrCount,
      tokens: tokenCount,
      timeMs: elapsed.toFixed(2),
      halted: emu.halted,
      loss: tnetLoss ? tnetLoss[1] : null,
    });

    setTimeout(() => setIsRunning(false), 150);
  }, [code, inputData]);

  const loadExample = (name) => {
    setSelectedExample(name);
    if (name === "_custom") {
      setCode("; New program\nLD    R0 @input\nHALT");
      setInputData({});
      setLeftTab("source");
    } else {
      const raw = EXAMPLES[name].code;
      setCode(syntaxMode === "symbolic" ? convertSyntax(raw, true) : raw);
      setInputData(JSON.parse(JSON.stringify(EXAMPLES[name].memory || {})));
    }
    setOutput([]);
    setRegisters({});
    setMemoryView({});
    setStats(null);
  };

  const lineNumbers = code.split("\n").map((_, i) => i + 1);

  return (
    <div style={{
      fontFamily: FONT,
      background: COLORS.bg,
      color: COLORS.text,
      height: "100vh",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
    }}>
      {/* HEADER */}
      <div style={{
        background: COLORS.panel,
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "12px 20px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ color: COLORS.accent, fontSize: 20, fontWeight: 800, letterSpacing: 3 }}>NML</span>
          <span style={{ color: COLORS.textDim, fontSize: 11, letterSpacing: 1 }}>NEURAL MACHINE LANGUAGE v0.10.0</span>
          <button onClick={() => setShowDocs(!showDocs)} style={{
            background: showDocs ? COLORS.accentBg : "transparent",
            border: `1px solid ${showDocs ? COLORS.accent : COLORS.border}`,
            color: showDocs ? COLORS.accent : COLORS.textDim,
            padding: "4px 10px",
            borderRadius: 4,
            fontSize: 10,
            cursor: "pointer",
            fontFamily: FONT,
            letterSpacing: 0.5,
            transition: "all 0.15s",
          }}>
            {showDocs ? "CLOSE DOCS" : "OPCODES"}
          </button>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={() => {
            const next = syntaxMode === "classic" ? "symbolic" : "classic";
            setCode(convertSyntax(code, next === "symbolic"));
            setSyntaxMode(next);
          }} style={{
            background: syntaxMode === "symbolic" ? COLORS.accentBg : "transparent",
            border: `1px solid ${syntaxMode === "symbolic" ? COLORS.accent : COLORS.border}`,
            color: syntaxMode === "symbolic" ? COLORS.accent : COLORS.textDim,
            padding: "4px 10px",
            borderRadius: 4,
            fontSize: 10,
            cursor: "pointer",
            fontFamily: FONT,
            letterSpacing: 0.5,
            transition: "all 0.15s",
            marginRight: 8,
          }}>
            {syntaxMode === "classic" ? "ABC CLASSIC" : "⊛ SYMBOLIC"}
          </button>
          <span style={{ color: COLORS.textDim, fontSize: 11, marginRight: 8 }}>EXAMPLES:</span>
          {Object.keys(EXAMPLES).map(name => (
            <button key={name} onClick={() => loadExample(name)} style={{
              background: selectedExample === name ? COLORS.accentBg : "transparent",
              border: `1px solid ${selectedExample === name ? COLORS.accent : COLORS.border}`,
              color: selectedExample === name ? COLORS.accent : COLORS.textDim,
              padding: "4px 10px",
              borderRadius: 4,
              fontSize: 10,
              cursor: "pointer",
              fontFamily: FONT,
              letterSpacing: 0.5,
              transition: "all 0.15s",
            }}>
              {name.toUpperCase()}
            </button>
          ))}
          <button onClick={() => loadExample("_custom")} style={{
            background: selectedExample === "_custom" ? COLORS.accentBg : "transparent",
            border: `1px solid ${selectedExample === "_custom" ? COLORS.warn : COLORS.border}`,
            color: selectedExample === "_custom" ? COLORS.warn : COLORS.textDim,
            padding: "4px 10px",
            borderRadius: 4,
            fontSize: 10,
            cursor: "pointer",
            fontFamily: FONT,
            letterSpacing: 0.5,
            transition: "all 0.15s",
          }}>
            + NEW
          </button>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden", position: "relative" }}>

        {/* DOCS SIDEBAR */}
        <div style={{
          position: "absolute",
          left: 0, top: 0, bottom: 0,
          width: 320,
          background: COLORS.panel,
          borderRight: `1px solid ${COLORS.border}`,
          zIndex: 10,
          transform: showDocs ? "translateX(0)" : "translateX(-100%)",
          transition: "transform 0.25s ease",
          display: "flex",
          flexDirection: "column",
          boxShadow: showDocs ? "4px 0 20px rgba(0,0,0,0.5)" : "none",
        }}>
          <div style={{
            padding: "10px 14px",
            borderBottom: `1px solid ${COLORS.border}`,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}>
            <span style={{ color: COLORS.accent, fontSize: 11, fontWeight: 700, letterSpacing: 2, flex: 1 }}>OPCODE REFERENCE</span>
            <span style={{ color: COLORS.textDim, fontSize: 9 }}>89</span>
          </div>
          <div style={{ padding: "6px 14px", borderBottom: `1px solid ${COLORS.border}` }}>
            <input
              value={docsFilter}
              onChange={e => setDocsFilter(e.target.value)}
              placeholder="Filter opcodes..."
              style={{
                width: "100%",
                background: COLORS.bg,
                border: `1px solid ${COLORS.border}`,
                color: COLORS.text,
                padding: "5px 8px",
                borderRadius: 4,
                fontSize: 11,
                fontFamily: FONT,
                outline: "none",
              }}
            />
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: "8px 0" }}>
            {OPCODE_DOCS.map(cat => {
              const filtered = cat.ops.filter(o => {
                if (!docsFilter) return true;
                const q = docsFilter.toLowerCase();
                return o.op.toLowerCase().includes(q) || o.sym.includes(q) || o.desc.toLowerCase().includes(q);
              });
              if (filtered.length === 0) return null;
              return (
                <div key={cat.cat} style={{ marginBottom: 12 }}>
                  <div style={{
                    padding: "4px 14px",
                    color: COLORS.accent,
                    fontSize: 9,
                    fontWeight: 700,
                    letterSpacing: 2,
                    textTransform: "uppercase",
                  }}>{cat.cat}</div>
                  {filtered.map(o => (
                    <div key={o.op} style={{
                      padding: "4px 14px",
                      fontSize: 11,
                      display: "flex",
                      gap: 6,
                      alignItems: "baseline",
                      cursor: "default",
                    }}
                    onMouseEnter={e => e.currentTarget.style.background = COLORS.accentBg}
                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                    >
                      <span style={{ color: COLORS.opcode, fontWeight: 700, minWidth: 52 }}>{o.op}</span>
                      <span style={{ color: COLORS.register, minWidth: 18, textAlign: "center" }}>{o.sym}</span>
                      <span style={{ color: COLORS.textDim, fontSize: 10, flex: 1 }}>{o.desc}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </div>

        {/* LEFT: CODE EDITOR */}
        <div style={{
          width: "50%",
          borderRight: `1px solid ${COLORS.border}`,
          display: "flex",
          flexDirection: "column",
        }}>
          <div style={{
            borderBottom: `1px solid ${COLORS.border}`,
            display: "flex",
            alignItems: "center",
            background: COLORS.panel,
          }}>
            <div style={{ display: "flex", flex: 1 }}>
              {["source", "input"].map(tab => (
                <button key={tab} onClick={() => setLeftTab(tab)} style={{
                  padding: "8px 16px",
                  background: leftTab === tab ? COLORS.accentBg : "transparent",
                  border: "none",
                  borderBottom: leftTab === tab ? `2px solid ${COLORS.accent}` : "2px solid transparent",
                  color: leftTab === tab ? COLORS.accent : COLORS.textDim,
                  fontSize: 10,
                  letterSpacing: 2,
                  cursor: "pointer",
                  fontFamily: FONT,
                  fontWeight: leftTab === tab ? 700 : 400,
                }}>
                  {tab.toUpperCase()}
                </button>
              ))}
            </div>
            <button onClick={runProgram} style={{
              background: isRunning ? COLORS.accentDim : COLORS.accent,
              color: COLORS.bg,
              border: "none",
              padding: "6px 20px",
              borderRadius: 4,
              fontWeight: 800,
              fontSize: 11,
              cursor: "pointer",
              fontFamily: FONT,
              letterSpacing: 2,
              transition: "all 0.15s",
              transform: isRunning ? "scale(0.96)" : "scale(1)",
              marginRight: 12,
            }}>
              {isRunning ? "RUNNING..." : "▶ EXECUTE"}
            </button>
          </div>

          {leftTab === "source" && (
          <div
            ref={el => {
              // Use the scroll container as the single scroll source
              if (el) el._nmlScroller = true;
            }}
            onScroll={e => {
              const st = e.target.scrollTop;
              if (gutterRef.current) gutterRef.current.style.transform = `translateY(-${st}px)`;
            }}
            style={{ flex: 1, overflow: "auto", background: COLORS.bg, position: "relative" }}
          >
            <div style={{ display: "flex", minHeight: "100%" }}>
              {/* Line numbers */}
              <div ref={gutterRef} style={{
                padding: "12px 0",
                textAlign: "right",
                color: COLORS.textDim,
                fontSize: 11,
                userSelect: "none",
                minWidth: 40,
                lineHeight: "20px",
                paddingRight: 8,
                borderRight: `1px solid ${COLORS.border}`,
                position: "sticky",
                left: 0,
                background: COLORS.bg,
              }}>
                {lineNumbers.map(n => <div key={n}>{n}</div>)}
              </div>

              {/* Code area — highlighted lines rendered directly */}
              <div style={{ flex: 1, position: "relative" }}>
                {/* Highlighted code as background */}
                <div style={{
                  padding: "12px 16px",
                  fontSize: 13,
                  lineHeight: "20px",
                  fontFamily: FONT,
                  whiteSpace: "pre",
                  pointerEvents: "none",
                  minHeight: "100%",
                }}>
                  {code.split("\n").map((line, i) => (
                    <div key={i} style={{ height: 20 }}>{syntaxHighlight(line) || " "}</div>
                  ))}
                </div>

                {/* Invisible textarea on top for editing */}
                <textarea
                  ref={textareaRef}
                  value={code}
                  onChange={e => setCode(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); runProgram(); }
                  }}
                  onMouseMove={e => {
                    clearTimeout(hoverTimeout.current);
                    hoverTimeout.current = setTimeout(() => {
                      const ta = textareaRef.current;
                      if (!ta) return;
                      const scrollParent = ta.closest("[style*='overflow']") || ta.parentElement.parentElement.parentElement;
                      const rect = ta.getBoundingClientRect();
                      const lineH = 20;
                      const padTop = 12;
                      const scrollTop = scrollParent ? scrollParent.scrollTop : 0;
                      const row = Math.floor((e.clientY - rect.top - padTop + scrollTop) / lineH);
                      const lines = code.split("\n");
                      if (row < 0 || row >= lines.length) { setHoverTip(null); return; }
                      const line = lines[row].replace(/;.*$/, "");
                      const tokens = line.trim().split(/\s+/);
                      const firstToken = tokens[0] || "";
                      const upper = firstToken.toUpperCase();
                      const doc = OPCODE_TOOLTIP[upper] || OPCODE_TOOLTIP[firstToken];
                      if (doc) {
                        setHoverTip({ x: e.clientX + 12, y: e.clientY + 16, doc });
                      } else {
                        setHoverTip(null);
                      }
                    }, 300);
                  }}
                  onMouseLeave={() => { clearTimeout(hoverTimeout.current); setHoverTip(null); }}
                  spellCheck={false}
                  style={{
                    position: "absolute",
                    top: 0, left: 0,
                    width: "100%",
                    height: "100%",
                    margin: 0,
                    padding: "12px 16px",
                    fontSize: 13,
                    lineHeight: "20px",
                    fontFamily: FONT,
                    background: "transparent",
                    color: "transparent",
                    caretColor: COLORS.accent,
                    border: "none",
                    outline: "none",
                    resize: "none",
                    whiteSpace: "pre",
                    overflow: "hidden",
                  }}
                />
              </div>
            </div>

            {/* Hover tooltip */}
            {hoverTip && (
              <div style={{
                position: "fixed",
                left: hoverTip.x,
                top: hoverTip.y,
                background: "#1a1a2e",
                border: `1px solid ${COLORS.accent}`,
                borderRadius: 6,
                padding: "8px 12px",
                zIndex: 20,
                pointerEvents: "none",
                maxWidth: 300,
                boxShadow: "0 4px 16px rgba(0,0,0,0.6)",
              }}>
                <div style={{ display: "flex", gap: 8, alignItems: "baseline", marginBottom: 4 }}>
                  <span style={{ color: COLORS.opcode, fontWeight: 700, fontSize: 13 }}>{hoverTip.doc.op}</span>
                  <span style={{ color: COLORS.register, fontSize: 14 }}>{hoverTip.doc.sym}</span>
                  <span style={{ color: COLORS.textDim, fontSize: 10 }}>{hoverTip.doc.schema}</span>
                </div>
                <div style={{ color: COLORS.text, fontSize: 11 }}>{hoverTip.doc.desc}</div>
              </div>
            )}
          </div>
          )}

          {leftTab === "input" && (
            <div style={{ flex: 1, overflow: "auto", padding: "12px 16px", background: COLORS.bg }}>
              {Object.keys(inputData).length === 0 ? (
                <div style={{ color: COLORS.textDim, fontSize: 12, textAlign: "center", padding: 20 }}>
                  No input data for this example
                </div>
              ) : (
                <div>
                  {Object.entries(inputData).map(([addr, { shape, data }]) => (
                    <div key={addr} style={{
                      padding: "8px 12px",
                      marginBottom: 6,
                      background: COLORS.accentBg,
                      borderRadius: 4,
                      border: `1px solid ${COLORS.border}`,
                    }}>
                      <div style={{ marginBottom: 4, display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ color: COLORS.address, fontWeight: 700 }}>@{addr}</span>
                        <label style={{ color: COLORS.textDim, fontSize: 10 }}>shape:</label>
                        <input
                          value={shape.join(",")}
                          onChange={e => {
                            const newShape = e.target.value.split(",").map(Number).filter(n => !isNaN(n) && n > 0);
                            if (newShape.length > 0) {
                              setInputData(prev => ({
                                ...prev,
                                [addr]: { ...prev[addr], shape: newShape },
                              }));
                            }
                          }}
                          style={{
                            background: COLORS.bg,
                            border: `1px solid ${COLORS.border}`,
                            color: COLORS.text,
                            padding: "2px 6px",
                            borderRadius: 3,
                            fontSize: 10,
                            fontFamily: FONT,
                            width: 60,
                            outline: "none",
                          }}
                        />
                        <span style={{ color: COLORS.textDim, fontSize: 10 }}>
                          {data.length} elements
                        </span>
                      </div>
                      <textarea
                        value={data.join(", ")}
                        onChange={e => {
                          const newData = e.target.value.split(",").map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
                          setInputData(prev => ({
                            ...prev,
                            [addr]: { ...prev[addr], data: newData },
                          }));
                        }}
                        style={{
                          width: "100%",
                          background: COLORS.bg,
                          border: `1px solid ${COLORS.border}`,
                          color: COLORS.text,
                          padding: "6px 8px",
                          borderRadius: 4,
                          fontSize: 11,
                          fontFamily: FONT,
                          resize: "vertical",
                          minHeight: 32,
                          outline: "none",
                          lineHeight: "18px",
                        }}
                      />
                    </div>
                  ))}
                  <button onClick={() => {
                    setInputData(JSON.parse(JSON.stringify(EXAMPLES[selectedExample].memory || {})));
                  }} style={{
                    background: "transparent",
                    border: `1px solid ${COLORS.border}`,
                    color: COLORS.textDim,
                    padding: "4px 10px",
                    borderRadius: 4,
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: FONT,
                    letterSpacing: 1,
                    marginTop: 4,
                  }}>
                    RESET TO DEFAULT
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Stats bar */}
          {stats && (
            <div style={{
              padding: "8px 16px",
              borderTop: `1px solid ${COLORS.border}`,
              background: COLORS.panel,
              display: "flex",
              gap: 24,
              fontSize: 10,
              color: COLORS.textDim,
              letterSpacing: 1,
            }}>
              <span>INSTRUCTIONS: <span style={{ color: COLORS.accent }}>{stats.instructions}</span></span>
              <span>CYCLES: <span style={{ color: COLORS.accent }}>{stats.cycles}</span></span>
              <span>TOKENS: <span style={{ color: COLORS.accent }}>{stats.tokens}</span></span>
              <span>TIME: <span style={{ color: COLORS.accent }}>{stats.timeMs}ms</span></span>
              {stats.loss && <span>LOSS: <span style={{ color: COLORS.warn }}>{stats.loss}</span></span>}
              <span>STATUS: <span style={{ color: stats.halted ? COLORS.halt : COLORS.error }}>{stats.halted ? "HALTED" : "ERROR"}</span></span>
            </div>
          )}
        </div>

        {/* RIGHT: OUTPUT PANEL */}
        <div style={{ width: "50%", display: "flex", flexDirection: "column" }}>

          {/* Tabs */}
          <div style={{
            display: "flex",
            borderBottom: `1px solid ${COLORS.border}`,
            background: COLORS.panel,
          }}>
            {["output", "registers", "memory"].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                flex: 1,
                padding: "8px",
                background: activeTab === tab ? COLORS.accentBg : "transparent",
                border: "none",
                borderBottom: activeTab === tab ? `2px solid ${COLORS.accent}` : "2px solid transparent",
                color: activeTab === tab ? COLORS.accent : COLORS.textDim,
                fontSize: 10,
                letterSpacing: 2,
                cursor: "pointer",
                fontFamily: FONT,
                fontWeight: activeTab === tab ? 700 : 400,
              }}>
                {tab.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div ref={outputRef} style={{ flex: 1, overflow: "auto", padding: "12px 16px" }}>

            {activeTab === "output" && (
              output.length === 0 ? (
                <div style={{ color: COLORS.textDim, fontSize: 12, padding: 20, textAlign: "center", lineHeight: 1.8 }}>
                  Press <span style={{ color: COLORS.accent, fontWeight: 700 }}>EXECUTE</span> or <span style={{ color: COLORS.accent }}>Ctrl+Enter</span> to run
                  <br />
                  <span style={{ fontSize: 10, opacity: 0.6 }}>89 opcodes. Tri-syntax. Zero ambiguity. Machine-first.</span>
                </div>
              ) : (
                output.map((entry, i) => (
                  <div key={i} style={{
                    padding: "4px 0",
                    fontSize: 12,
                    display: "flex",
                    gap: 8,
                    alignItems: "flex-start",
                    borderBottom: `1px solid ${COLORS.border}22`,
                  }}>
                    <span style={{
                      color: entry.type === "error" ? COLORS.error : entry.type === "halt" ? COLORS.halt : COLORS.opcode,
                      fontWeight: 700,
                      minWidth: 42,
                      fontSize: 10,
                      paddingTop: 2,
                    }}>
                      {entry.op || (entry.type === "error" ? "ERR" : "SYS")}
                    </span>
                    <span style={{
                      color: entry.type === "error" ? COLORS.error : entry.type === "halt" ? COLORS.halt : COLORS.text,
                      wordBreak: "break-all",
                    }}>
                      {entry.msg}
                    </span>
                  </div>
                ))
              )
            )}

            {activeTab === "registers" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                {Object.entries(registers).map(([name, val]) => (
                  <div key={name} style={{
                    padding: "6px 10px",
                    background: val ? COLORS.accentBg : "transparent",
                    borderRadius: 4,
                    border: `1px solid ${val ? COLORS.border : "transparent"}`,
                    fontSize: 11,
                  }}>
                    <span style={{ color: COLORS.register, fontWeight: 700 }}>{name}</span>
                    <span style={{ color: COLORS.textDim }}> = </span>
                    <span style={{ color: val ? COLORS.text : COLORS.textDim, fontSize: 10 }}>
                      {val ? val.toString() : "empty"}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {activeTab === "memory" && (
              Object.keys(memoryView).length === 0 ? (
                <div style={{ color: COLORS.textDim, fontSize: 12, textAlign: "center", padding: 20 }}>
                  No memory addresses written yet
                </div>
              ) : (
                Object.entries(memoryView).map(([addr, tensor]) => (
                  <div key={addr} style={{
                    padding: "8px 12px",
                    marginBottom: 6,
                    background: COLORS.accentBg,
                    borderRadius: 4,
                    border: `1px solid ${COLORS.border}`,
                  }}>
                    <div style={{ marginBottom: 4 }}>
                      <span style={{ color: COLORS.address, fontWeight: 700 }}>@{addr}</span>
                      <span style={{ color: COLORS.textDim, fontSize: 10, marginLeft: 8 }}>
                        shape: [{tensor.shape.join("×")}]
                      </span>
                    </div>
                    <div style={{ fontSize: 11, color: COLORS.text, wordBreak: "break-all" }}>
                      [{Array.from(tensor.data).map(v => v.toFixed(4)).join(", ")}]
                    </div>
                  </div>
                ))
              )
            )}
          </div>
        </div>
      </div>

      {/* FOOTER */}
      <div style={{
        padding: "6px 20px",
        borderTop: `1px solid ${COLORS.border}`,
        background: COLORS.panel,
        display: "flex",
        justifyContent: "space-between",
        fontSize: 9,
        color: COLORS.textDim,
        letterSpacing: 1,
        flexShrink: 0,
      }}>
        <span>89 OPCODES • 32 REGISTERS • TRI-SYNTAX • MINI-BATCH SGD</span>
        <span>NML EMULATOR v0.10.0</span>
      </div>
    </div>
  );
}
