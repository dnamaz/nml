/**
 * NML WASM Runtime — Browser execution wrapper
 *
 * Loads the Emscripten-compiled NML runtime and provides a simple API
 * for executing NML programs in the browser without a server round-trip.
 *
 * Build: make nml-wasm (requires emcc)
 *
 * Usage:
 *   const nml = new NMLRuntime();
 *   await nml.init();
 *   const result = await nml.run("LD R0 @x\nSCLR R1 R0 #2.0\nST R1 @result\nHALT",
 *                                 "@x shape=1 data=5.0");
 *   console.log(result.output);   // register + memory dump
 *   console.log(result.success);  // true/false
 */

class NMLRuntime {
  constructor(wasmPath = "./nml.js") {
    this.wasmPath = wasmPath;
    this.module = null;
    this.ready = false;
  }

  async init() {
    if (this.ready) return;
    const NMLModule = (await import(this.wasmPath)).default;
    this.module = await NMLModule({
      print: (text) => { this._stdout += text + "\n"; },
      printErr: (text) => { this._stderr += text + "\n"; },
    });
    this.ready = true;
  }

  async run(programSource, dataSource = null) {
    if (!this.ready) await this.init();

    this._stdout = "";
    this._stderr = "";

    const FS = this.module.FS;

    FS.writeFile("/tmp/program.nml", programSource);
    const args = ["/tmp/program.nml"];

    if (dataSource) {
      FS.writeFile("/tmp/program.nml.data", dataSource);
      args.push("/tmp/program.nml.data");
    }

    args.push("--max-cycles", "100000");

    let exitCode;
    try {
      exitCode = this.module.callMain(args);
    } catch (e) {
      return {
        success: false,
        output: this._stdout,
        error: this._stderr || e.message,
        exitCode: -1,
      };
    }

    try { FS.unlink("/tmp/program.nml"); } catch (_) {}
    try { FS.unlink("/tmp/program.nml.data"); } catch (_) {}

    return {
      success: exitCode === 0,
      output: this._stdout,
      error: this._stderr,
      exitCode,
    };
  }

  parseRegisters(output) {
    const regs = {};
    const lines = output.split("\n");
    let inRegs = false;
    for (const line of lines) {
      if (line.includes("=== REGISTERS ===")) { inRegs = true; continue; }
      if (line.includes("=== MEMORY ===")) { inRegs = false; continue; }
      if (inRegs) {
        const m = line.match(/^\s+(\w+):\s+shape=\[([^\]]+)\]\s+data=\[([^\]]+)\]/);
        if (m) {
          regs[m[1]] = {
            shape: m[2].split("x").map(Number),
            data: m[3].split(",").map(s => parseFloat(s.trim())),
          };
        }
      }
    }
    return regs;
  }

  parseMemory(output) {
    const mem = {};
    const lines = output.split("\n");
    let inMem = false;
    for (const line of lines) {
      if (line.includes("=== MEMORY ===")) { inMem = true; continue; }
      if (line.includes("=== STATS ===")) { inMem = false; continue; }
      if (inMem) {
        const m = line.match(/^\s+(\w+):\s+shape=\[([^\]]+)\]\s+data=\[([^\]]+)\]/);
        if (m) {
          mem[m[1]] = {
            shape: m[2].split("x").map(Number),
            data: m[3].split(",").map(s => parseFloat(s.trim())),
          };
        }
      }
    }
    return mem;
  }
}

if (typeof module !== "undefined") {
  module.exports = { NMLRuntime };
}
