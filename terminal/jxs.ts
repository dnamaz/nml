#!/usr/bin/env bun
import { Webview } from "webview-bun";
import { unlinkSync, mkdtempSync, rmSync, existsSync } from "fs";
import { resolve, basename, join, dirname } from "path";
import { tmpdir } from "os";

const file = process.argv[2];
if (!file) {
  console.error("Usage: bun jxs.ts <component.jsx>");
  process.exit(1);
}

const componentPath = resolve(file);
if (!existsSync(componentPath)) {
  console.error(`File not found: ${componentPath}`);
  process.exit(1);
}

const name = basename(file, ".jsx");
const buildDir = mkdtempSync(join(tmpdir(), "jxs-"));

const entryFile = join(buildDir, "entry.jsx");
await Bun.write(entryFile, `
import { createRoot } from "react-dom/client";
import App from "${componentPath}";
createRoot(document.getElementById("root")).render(<App />);
`);

const projectModules = join(dirname(componentPath), "node_modules");
const skillDir = import.meta.dir;
const skillModules = join(skillDir, "node_modules");

if (!existsSync(join(skillModules, "react"))) {
  console.log("[jxs] Installing dependencies (one-time)...");
  Bun.spawnSync(["bun", "install"], { cwd: skillDir, stdio: ["ignore", "inherit", "inherit"] });
}

const resolveDir = existsSync(join(projectModules, "react")) ? dirname(componentPath) : skillDir;

const result = await Bun.build({
  entrypoints: [entryFile],
  outdir: buildDir,
  target: "browser",
  minify: true,
  plugins: [{
    name: "resolve-deps",
    setup(build) {
      build.onResolve({ filter: /^react|^react-dom/ }, (args) => {
        const resolved = require.resolve(args.path, { paths: [resolveDir] });
        return { path: resolved };
      });
    },
  }],
});

if (!result.success) {
  console.error("Build failed:");
  for (const log of result.logs) console.error(log);
  rmSync(buildDir, { recursive: true });
  process.exit(1);
}

const htmlPath = join(buildDir, "index.html");
await Bun.write(htmlPath, `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>*{margin:0;padding:0;box-sizing:border-box}body{overflow:hidden;background:#0a0a0f}</style>
</head>
<body>
  <div id="root"></div>
  <script src="./${basename(result.outputs[0].path)}"></script>
</body>
</html>`);

console.log(`[jxs] Launching ${name}...`);
const webview = new Webview();
webview.title = name;
webview.size = { width: 1280, height: 800, hint: 0 };
webview.navigate(`file://${htmlPath}`);
webview.run();

rmSync(buildDir, { recursive: true });
