# Set up the macro_gpu_lab GPU venv (Py3.11, torch cu121). ASCII-only on purpose
# (PowerShell 5.1 mangles non-ASCII). Run from the repo root.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

uv venv --python 3.11
$py = Join-Path $repo ".venv\Scripts\python.exe"

# torch from the cu121 index; --native-tls for this box's TLS interception.
uv pip install --python $py torch --index-url https://download.pytorch.org/whl/cu121 --native-tls
uv pip install --python $py -r requirements.txt --native-tls

& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
