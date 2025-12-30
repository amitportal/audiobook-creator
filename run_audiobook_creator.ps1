# Audiobook Creator - One-Click Setup and Launch
# This script automates installation and launches the GUI
# Requires: Python 3.12+

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Audiobook Creator - Setup & Launch" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check command existence
function Test-Command {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Step 1: Detect hardware
Write-Host "[1/8] Detecting hardware..." -ForegroundColor Yellow
$gpuInfo = ""
if (Test-Command "nvidia-smi") {
    $gpuInfo = " (NVIDIA GPU detected)"
}
$cpuInfo = Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name
Write-Host "  Hardware: $cpuInfo$gpuInfo" -ForegroundColor Green

# Step 2: Check Python version
Write-Host ""
Write-Host "[2/8] Checking Python installation..." -ForegroundColor Yellow
if (Test-Command "python") {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
    
    $versionMatch = [regex]::Match($pythonVersion, "Python (\d+)\.(\d+)")
    if ($versionMatch.Success) {
        $major = [int]$versionMatch.Groups[1].Value
        $minor = [int]$versionMatch.Groups[2].Value
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 12)) {
            Write-Host "  ERROR: Python 3.12+ required. Current: $pythonVersion" -ForegroundColor Red
            Write-Host "  Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}
else {
    Write-Host "  ERROR: Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.12+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 3: Install uv package manager
Write-Host ""
Write-Host "[3/8] Checking uv package manager..." -ForegroundColor Yellow
if (-not (Test-Command "uv")) {
    Write-Host "  Installing uv..." -ForegroundColor Cyan
    python -m pip install uv --quiet
    Write-Host "  uv installed successfully!" -ForegroundColor Green
}
else {
    Write-Host "  uv already installed" -ForegroundColor Green
}

# Step 4: Install project dependencies
Write-Host ""
Write-Host "[4/8] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes on first run..." -ForegroundColor Cyan
uv sync
uv pip install -e . --quiet
Write-Host "  Dependencies installed!" -ForegroundColor Green

# Step 5: Check/Install FFmpeg
Write-Host ""
Write-Host "[5/8] Checking FFmpeg..." -ForegroundColor Yellow
if (-not (Test-Command "ffmpeg")) {
    Write-Host "  FFmpeg not found. Attempting to install via winget..." -ForegroundColor Cyan
    if (Test-Command "winget") {
        winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
        Write-Host "  FFmpeg installed! You may need to restart your terminal." -ForegroundColor Green
    }
    else {
        Write-Host "  WARNING: Please install FFmpeg manually: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  FFmpeg already installed" -ForegroundColor Green
}

# Step 6: Download models
Write-Host ""
Write-Host "[6/8] Checking TTS models..." -ForegroundColor Yellow
$supertonicPath = "$env:USERPROFILE\.cache\huggingface\supertonic_models"
if (-not (Test-Path $supertonicPath)) {
    Write-Host "  Downloading Supertonic models (~500MB)..." -ForegroundColor Cyan
    if (Test-Command "git") {
        git clone https://huggingface.co/Supertone/supertonic "$supertonicPath"
    }
}

$chatterboxPath = "$env:USERPROFILE\.cache\huggingface\chatterbox_models\onnx"
if (-not (Test-Path "$chatterboxPath\language_model_q4.onnx_data")) {
    Write-Host "  Downloading Chatterbox models (~300MB)..." -ForegroundColor Cyan
    uv run python -c "from audiobook_creator.chatterbox_wrapper import ChatterboxONNX; from pathlib import Path; p = Path.home() / '.cache' / 'huggingface' / 'chatterbox_models' / 'onnx'; c = ChatterboxONNX(str(p)); c._ensure_models()"
}

Write-Host "  Checking Soprano/Mira models (HuggingFace cache)..." -ForegroundColor Cyan
uv run python -c "from huggingface_hub import hf_hub_download, snapshot_download; print('  Verifying Soprano...'); hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth'); print('  Verifying Mira...'); snapshot_download('YatharthS/MiraTTS')"

Write-Host "  Models verified!" -ForegroundColor Green

# Step 7: Final Environment Check
Write-Host ""
Write-Host "[7/8] Testing Hardware Acceleration (OpenVINO/CUDA)..." -ForegroundColor Yellow
uv run python -c "import torch; import onnxruntime as ort; print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  ONNX Providers: {ort.get_available_providers()}'); try: import openvino; print(f'  OpenVINO Devices: {openvino.Core().available_devices}'); except: pass"

# Step 8: Launch GUI
Write-Host ""
Write-Host "[8/8] Launching Audiobook Creator GUI..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete! Starting GUI..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

uv run audiobook-gui

Write-Host ""
Write-Host "GUI closed. Thank you for using Audiobook Creator!" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
