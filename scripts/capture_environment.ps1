param([string]$Output = "runs/environment_capture")
$ErrorActionPreference = "Stop"
if (-not $env:CONDA_PREFIX) { throw "Activate the actual experiment Conda environment first." }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
conda env export | Out-File -Encoding utf8 "$Output/conda_environment_full.yml"
if ($LASTEXITCODE -ne 0) { throw "Conda export failed." }
conda list --explicit | Out-File -Encoding utf8 "$Output/conda_explicit.txt"
if ($LASTEXITCODE -ne 0) { throw "Conda explicit export failed." }
python -m pip freeze | Out-File -Encoding utf8 "$Output/pip_freeze.txt"
if ($LASTEXITCODE -ne 0) { throw "pip freeze failed." }
python -c "import sys,platform; print(sys.executable); print(sys.version); print(platform.platform())" | Out-File -Encoding utf8 "$Output/python.txt"
nvidia-smi | Out-File -Encoding utf8 "$Output/nvidia_smi.txt"
Write-Output "Environment records saved: $Output"
