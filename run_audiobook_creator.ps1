# Audiobook Creator - One-Click Setup and Launch
# This script automates installation and launches the GUI
# Requires: Python 3.12+

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

# Step 1: Check Python version
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
if (Test-Command "python") {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version number
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

# Step 2: Install uv package manager
Write-Host ""
Write-Host "[2/6] Checking uv package manager..." -ForegroundColor Yellow
if (-not (Test-Command "uv")) {
    Write-Host "  Installing uv..." -ForegroundColor Cyan
    python -m pip install uv --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to install uv" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "  uv installed successfully!" -ForegroundColor Green
}
else {
    Write-Host "  uv already installed" -ForegroundColor Green
}

# Step 3: Install project dependencies
Write-Host ""
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes on first run..." -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Reinstall package to register entry points
Write-Host "  Registering application entry points..." -ForegroundColor Cyan
# Audiobook Creator - One-Click Setup and Launch
# This script automates installation and launches the GUI
# Requires: Python 3.12+

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

# Step 1: Check Python version
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
if (Test-Command "python") {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version number
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

# Step 2: Install uv package manager
Write-Host ""
Write-Host "[2/6] Checking uv package manager..." -ForegroundColor Yellow
if (-not (Test-Command "uv")) {
    Write-Host "  Installing uv..." -ForegroundColor Cyan
    python -m pip install uv --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to install uv" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "  uv installed successfully!" -ForegroundColor Green
}
else {
    Write-Host "  uv already installed" -ForegroundColor Green
}

# Step 3: Install project dependencies
Write-Host ""
Write-Host "[3/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes on first run..." -ForegroundColor Cyan
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Reinstall package to register entry points
Write-Host "  Registering application entry points..." -ForegroundColor Cyan
uv pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: Entry point registration may have failed" -ForegroundColor Yellow
}
Write-Host "  Dependencies installed!" -ForegroundColor Green

# Step 4: Check/Install FFmpeg
Write-Host ""
Write-Host "[4/6] Checking FFmpeg..." -ForegroundColor Yellow
if (-not (Test-Command "ffmpeg")) {
    Write-Host "  FFmpeg not found. Attempting to install via winget..." -ForegroundColor Cyan
    if (Test-Command "winget") {
        winget install Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  FFmpeg installed! You may need to restart your terminal." -ForegroundColor Green
        }
        else {
            Write-Host "  WARNING: Auto-install failed. Please install manually:" -ForegroundColor Yellow
            Write-Host "    winget install Gyan.FFmpeg" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  WARNING: winget not found. Please install FFmpeg manually:" -ForegroundColor Yellow
        Write-Host "    https://ffmpeg.org/download.html" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  FFmpeg already installed" -ForegroundColor Green
}

# Step 5: Download Supertonic models
Write-Host ""
Write-Host "[5/7] Checking Supertonic TTS models..." -ForegroundColor Yellow
$modelPath = "$env:USERPROFILE\.cache\huggingface\supertonic_models"
if (-not (Test-Path $modelPath)) {
    Write-Host "  Downloading Supertonic models (one-time, ~500MB)..." -ForegroundColor Cyan
    Write-Host "  This may take several minutes..." -ForegroundColor Cyan
    
    if (Test-Command "git") {
        git clone https://huggingface.co/Supertone/supertonic "$modelPath"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Models downloaded successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "  WARNING: Model download failed. You can download manually later." -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  WARNING: git not found. Please install git and run:" -ForegroundColor Yellow
        Write-Host "    git clone https://huggingface.co/Supertone/supertonic $modelPath" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  Models already downloaded" -ForegroundColor Green
}

# Step 6: Download Chatterbox ONNX models
Write-Host ""
Write-Host "[6/7] Checking Chatterbox ONNX models..." -ForegroundColor Yellow
$chatterboxBase = "$env:USERPROFILE\.cache\huggingface\chatterbox_models"
$onnxPath = "$chatterboxBase\onnx"

# Check for a critical file to ensure valid install
if (-not (Test-Path "$onnxPath\language_model_q4.onnx_data")) {
    Write-Host "  Downloading Chatterbox ONNX models (one-time, ~300MB)..." -ForegroundColor Cyan
    Write-Host "  This may take several minutes..." -ForegroundColor Cyan
    
    if (Test-Command "git") {
        $tempPath = "$env:USERPROFILE\.cache\huggingface\chatterbox_temp_dl"
        
        # Clean previous temp attempts
        if (Test-Path $tempPath) {
            Remove-Item $tempPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        # Clone full repository to temp
        Write-Host "  Cloning repository..." -ForegroundColor Cyan
        git clone https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX "$tempPath"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  Installing models..." -ForegroundColor Cyan
            
            # Ensure destination exists
            if (-not (Test-Path $chatterboxBase)) {
                New-Item -ItemType Directory -Path $chatterboxBase -Force | Out-Null
            }
            
            # Move onnx folder
            if (Test-Path "$tempPath\onnx") {
                # Remove existing onnx folder if partial
                if (Test-Path $onnxPath) {
                    Remove-Item $onnxPath -Recurse -Force -ErrorAction SilentlyContinue
                }
                
                Copy-Item "$tempPath\onnx" -Destination $chatterboxBase -Recurse -Force
                Write-Host "  Chatterbox models installed successfully!" -ForegroundColor Green
            }
            else {
                Write-Host "  ERROR: 'onnx' folder not found in downloaded repository." -ForegroundColor Red
            }
            
            # Cleanup temp
            Remove-Item $tempPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "  WARNING: Git clone failed. You can download manually later." -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "  WARNING: git not found. To download Chatterbox models:" -ForegroundColor Yellow
        Write-Host "    Go to: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/tree/main/onnx" -ForegroundColor Yellow
        Write-Host "    Download files to: $onnxPath" -ForegroundColor Yellow
    }
}
else {
    Write-Host "  Models already downloaded" -ForegroundColor Green
}

# Step 7: Launch GUI
Write-Host ""
Write-Host "[7/7] Launching Audiobook Creator GUI..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete! Starting GUI..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Launch the GUI
uv run audiobook-gui

# If GUI exits, show message
Write-Host ""
Write-Host "GUI closed. Thank you for using Audiobook Creator!" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
