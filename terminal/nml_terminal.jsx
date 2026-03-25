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
      "⊞":"RSHP","⊤":"TRNS","⊢":"SPLT","⊣":"MERG",
      "⋈":"CMPF","≶":"CMP","≺":"CMPI","ϟ":"CMPI",
      "↗":"JMPT","↘":"JMPF","→":"JUMP","↻":"LOOP","↺":"ENDP",
      "⇒":"CALL","⇐":"RET","∑":"TACC",
      "◼":"HALT","⚠":"TRAP","⏸":"SYNC","⚙":"SYS",
      "∇":"BKWD","⟳":"WUPD","△":"LOSS","⥁":"TNET",
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
  "Symbolic": {
    code: `; Symbolic syntax: scale and store
∎  ι  #42.0
∗  κ  ι  #3.14
↑  κ  @result
◼`,
    memory: {}
  }
};

// ═══════════════════════════════════════════
// REACT UI
// ═══════════════════════════════════════════

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
         "SYS","MOD","ITOF","FTOI","BNOT"].includes(upper)) {
      return <span key={i} style={{ color: COLORS.opcode, fontWeight: 700 }}>{p}</span>;
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
  const outputRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (outputRef.current) outputRef.current.scrollTop = outputRef.current.scrollHeight;
  }, [output]);

  const runProgram = useCallback(() => {
    setIsRunning(true);
    const emu = new NMLEmulator();
    const example = EXAMPLES[selectedExample];

    if (example && example.memory) {
      Object.entries(example.memory).forEach(([addr, { shape, data }]) => {
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
    setStats({
      cycles: emu.cycles,
      instructions: instrCount,
      tokens: tokenCount,
      timeMs: elapsed.toFixed(2),
      halted: emu.halted,
    });

    setTimeout(() => setIsRunning(false), 150);
  }, [code, selectedExample]);

  const loadExample = (name) => {
    setSelectedExample(name);
    setCode(EXAMPLES[name].code);
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
          <span style={{ color: COLORS.textDim, fontSize: 11, letterSpacing: 1 }}>NEURAL MACHINE LANGUAGE v0.7.0</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
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
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* LEFT: CODE EDITOR */}
        <div style={{
          width: "50%",
          borderRight: `1px solid ${COLORS.border}`,
          display: "flex",
          flexDirection: "column",
        }}>
          <div style={{
            padding: "8px 16px",
            borderBottom: `1px solid ${COLORS.border}`,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            background: COLORS.panel,
          }}>
            <span style={{ color: COLORS.textDim, fontSize: 10, letterSpacing: 2 }}>SOURCE</span>
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
            }}>
              {isRunning ? "RUNNING..." : "▶ EXECUTE"}
            </button>
          </div>

          <div style={{ flex: 1, display: "flex", overflow: "auto", background: COLORS.bg }}>
            {/* Line numbers */}
            <div style={{
              padding: "12px 0",
              textAlign: "right",
              color: COLORS.textDim,
              fontSize: 11,
              userSelect: "none",
              minWidth: 40,
              lineHeight: "20px",
              paddingRight: 8,
              borderRight: `1px solid ${COLORS.border}`,
            }}>
              {lineNumbers.map(n => <div key={n}>{n}</div>)}
            </div>

            {/* Code display overlay + textarea */}
            <div style={{ flex: 1, position: "relative" }}>
              {/* Syntax-highlighted overlay */}
              <pre style={{
                position: "absolute",
                top: 0, left: 0, right: 0, bottom: 0,
                margin: 0,
                padding: "12px 16px",
                fontSize: 13,
                lineHeight: "20px",
                fontFamily: FONT,
                pointerEvents: "none",
                whiteSpace: "pre",
                overflow: "auto",
              }}>
                {code.split("\n").map((line, i) => (
                  <div key={i}>{syntaxHighlight(line) || " "}</div>
                ))}
              </pre>

              {/* Invisible textarea for editing */}
              <textarea
                ref={textareaRef}
                value={code}
                onChange={e => setCode(e.target.value)}
                onKeyDown={e => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); runProgram(); }
                }}
                spellCheck={false}
                style={{
                  position: "absolute",
                  top: 0, left: 0, right: 0, bottom: 0,
                  width: "100%", height: "100%",
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
                  overflow: "auto",
                }}
              />
            </div>
          </div>

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
                  <span style={{ fontSize: 10, opacity: 0.6 }}>82 opcodes. Tri-syntax. Zero ambiguity. Machine-first.</span>
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
        <span>82 OPCODES • 32 REGISTERS • TRI-SYNTAX • FIXED-WIDTH ENCODING</span>
        <span>NML EMULATOR v0.7.0</span>
      </div>
    </div>
  );
}
