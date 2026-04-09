# NML Agent Pipeline — Windows Startup Script
#
# Starts the think model, code model, and NML server for the
# validated code generation pipeline.
#
# Models:
#   Think: nml-think-v2 Q8_0 (architecture planning)
#   Code:  nml-1.5b-instruct-v0.10.0 F16 (NML assembly generation)
#
# Usage:
#   .\start_agents.ps1                    # defaults (SYCL on Intel Arc Pro B50)
#   .\start_agents.ps1 -Device Vulkan0    # use Vulkan backend instead
#   .\start_agents.ps1 -LlamaPath C:\other\llama-server.exe

param(
    [string]$Device = "SYCL0",
    [string]$LlamaPath = "C:\llama.cpp\sycl\llama-server.exe",
    [int]$ThinkPort = 8084,
    [int]$CodePort = 8085,
    [int]$ServerPort = 8082,
    [switch]$UseOllama,
    [string]$AdvisorLLM = ""
)

# Intel oneAPI — initialize SYCL runtime for Arc Pro B50
$env:ZES_ENABLE_SYSMAN = "1"
$oneAPISetVars = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if (Test-Path $oneAPISetVars) {
    Write-Host "Initializing oneAPI environment..." -ForegroundColor DarkGray
    # Run setvars.bat in cmd and capture the resulting environment
    $envOutput = cmd /c "`"$oneAPISetVars`" --force > nul 2>&1 && set"
    foreach ($line in $envOutput) {
        if ($line -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
} else {
    Write-Host "WARN: oneAPI setvars.bat not found — SYCL may not work." -ForegroundColor Yellow
    Write-Host "  Expected: $oneAPISetVars" -ForegroundColor Yellow
}

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LlamaDir = Split-Path -Parent $LlamaPath

# ── Model paths ──────────────────────────────────────────────────────────
$ThinkModel = Join-Path $Root "nml-model-training\models\nml-think-v2-merged\nml-think-v2-Q8_0.gguf"
$CodeModel  = Join-Path $Root "nml\domain\output\model\nml-1.5b-instruct-v0.10.0-f16.gguf"

# Fallback: check Q4_K_M version
if (-not (Test-Path $CodeModel)) {
    $CodeModel = Join-Path $Root "nml-model-training\output\nml-1.5b-v0.11.0\nml-1.5b-instruct-v0.10.0-20260406-q4_k_m.gguf"
}

# ── Validation ───────────────────────────────────────────────────────────
if (-not (Test-Path $LlamaPath)) {
    Write-Host "ERROR: llama-server not found at $LlamaPath" -ForegroundColor Red
    Write-Host "  Set -LlamaPath to the correct location." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "NML Agent Pipeline" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host "  Think model:  $ThinkModel" -ForegroundColor Gray
Write-Host "  Code model:   $CodeModel" -ForegroundColor Gray
Write-Host "  Device:       $Device" -ForegroundColor Gray
Write-Host "  Think port:   $ThinkPort" -ForegroundColor Gray
Write-Host "  Code port:    $CodePort" -ForegroundColor Gray
Write-Host "  Server port:  $ServerPort" -ForegroundColor Gray
Write-Host ""

$jobs = @()

# ── Start Think Model ────────────────────────────────────────────────────
if ($UseOllama) {
    Write-Host "Starting Think Model via Ollama on port 11434..." -ForegroundColor Blue
    # Ollama serves on 11434 by default — pull the model if needed
    $ollamaModels = & ollama list 2>&1
    if ($ollamaModels -notmatch "nml-think") {
        Write-Host "  Pulling think model into Ollama..." -ForegroundColor DarkGray
        $modelfilePath = Join-Path $Root "nml-model-training\models\nml-think-v2-merged\Modelfile.think"
        if (Test-Path $modelfilePath) {
            Push-Location (Split-Path $modelfilePath)
            & ollama create nml-think -f Modelfile.think
            Pop-Location
        } else {
            Write-Host "  Modelfile not found, pulling base qwen3.5:4b instead" -ForegroundColor Yellow
            & ollama pull qwen3.5:4b
        }
    }
    # Ollama API is on 11434, think endpoint uses /api/chat
    # Pipeline JSX connects via ?think=11434
    $ThinkPort = 11434
    Write-Host "  Ollama think model ready on port $ThinkPort" -ForegroundColor DarkGray
} elseif (Test-Path $ThinkModel) {
    Write-Host "Starting Think Model (4B Q6) on port $ThinkPort..." -ForegroundColor Blue
    $thinkJob = Start-Process -FilePath $LlamaPath -WorkingDirectory $LlamaDir -ArgumentList @(
        "-m", $ThinkModel,
        "--chat-template", "chatml",
        "-c", "4096",
        "--port", $ThinkPort,
        "--host", "127.0.0.1",
        "-ngl", "99",
        "-dev", $Device
    ) -PassThru -WindowStyle Minimized
    $jobs += $thinkJob
    Write-Host "  Think model PID: $($thinkJob.Id)" -ForegroundColor DarkGray
} else {
    Write-Host "WARN: Think model not found, skipping." -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# ── Start Code Model ────────────────────────────────────────────────────
if (Test-Path $CodeModel) {
    Write-Host "Starting Code Model (1.5B F16) on port $CodePort..." -ForegroundColor Green
    $codeJob = Start-Process -FilePath $LlamaPath -WorkingDirectory $LlamaDir -ArgumentList @(
        "-m", $CodeModel,
        "--chat-template", "chatml",
        "-c", "8192",
        "--port", $CodePort,
        "--host", "127.0.0.1",
        "-ngl", "99",
        "-dev", $Device
    ) -PassThru -WindowStyle Minimized
    $jobs += $codeJob
    Write-Host "  Code model PID: $($codeJob.Id)" -ForegroundColor DarkGray
} else {
    Write-Host "ERROR: Code model not found at $CodeModel" -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 3

# ── Start NML Server ────────────────────────────────────────────────────
$ServerScript = Join-Path $Root "nml\serve\nml_server.py"
Write-Host "Starting NML Server on port $ServerPort..." -ForegroundColor Magenta
Write-Host "  Proxying to code model on port $CodePort" -ForegroundColor DarkGray

$serverArgs = @(
    $ServerScript,
    "--http",
    "--port", $ServerPort,
    "--model", "http://127.0.0.1:${CodePort}"
)
if ($AdvisorLLM) {
    $serverArgs += @("--advisor-llm", $AdvisorLLM)
    Write-Host "  ML Advisor: $AdvisorLLM" -ForegroundColor Yellow
}
$serverJob = Start-Process -FilePath "python" -ArgumentList $serverArgs -PassThru -WindowStyle Minimized
$jobs += $serverJob
Write-Host "  Server PID: $($serverJob.Id)" -ForegroundColor DarkGray

Start-Sleep -Seconds 2

# ── Status ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "All services started." -ForegroundColor Green
Write-Host ""
Write-Host "  Pipeline UI:    http://localhost:$ServerPort" -ForegroundColor Cyan
Write-Host "  Think model:    http://localhost:$ThinkPort" -ForegroundColor Blue
Write-Host "  Code model:     http://localhost:$CodePort" -ForegroundColor Green
if ($AdvisorLLM) {
    Write-Host "  ML Advisor:     $AdvisorLLM" -ForegroundColor Yellow
} else {
    Write-Host "  ML Advisor:     KB-only (use -AdvisorLLM URL for cloud LLM)" -ForegroundColor DarkGray
}
Write-Host "  Advise API:     POST http://localhost:$ServerPort/advise" -ForegroundColor Yellow
Write-Host "  Validated gen:  POST http://localhost:$ServerPort/generate_validated" -ForegroundColor Magenta
Write-Host ""
Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Yellow
Write-Host ""

# ── Wait + cleanup ───────────────────────────────────────────────────────
try {
    while ($true) {
        Start-Sleep -Seconds 5
        $alive = $jobs | Where-Object { -not $_.HasExited }
        if ($alive.Count -eq 0) {
            Write-Host "All services have exited." -ForegroundColor Yellow
            break
        }
    }
} finally {
    Write-Host "Stopping services..." -ForegroundColor Yellow
    foreach ($job in $jobs) {
        if (-not $job.HasExited) {
            Stop-Process -Id $job.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped PID $($job.Id)" -ForegroundColor DarkGray
        }
    }
    Write-Host "Done." -ForegroundColor Green
}
