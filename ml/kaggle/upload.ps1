# Usage: .\kaggle\upload.ps1 YOUR_KAGGLE_USERNAME [-SkipData]
param(
    [string]$Username = "YOUR_KAGGLE_USERNAME",
    [switch]$SkipData
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$MlDir = Split-Path -Parent $ScriptDir
$Staging = "$env:USERPROFILE\kaggle_staging"

if (-not (Test-Path "$env:USERPROFILE\.kaggle\kaggle.json")) {
    Write-Error "Put kaggle.json in ~/.kaggle/ (kaggle.com -> Settings -> API -> Create Token)"
    exit 1
}

# Find best checkpoint (highest step number)
$CheckpointsDir = "$MlDir\outputs\arabic_v1\checkpoints"
$ckptFiles = Get-ChildItem "$CheckpointsDir\model-*" -File |
    Sort-Object { [int]($_.Name -replace 'model-', '') } -Descending
if ($ckptFiles.Count -eq 0) {
    Write-Error "No checkpoints found in $CheckpointsDir"
    exit 1
}
$BestCkpt = $ckptFiles[0]
$Step = $BestCkpt.Name -replace 'model-', ''
Write-Host "Best checkpoint: $($BestCkpt.Name) (step $Step)"

# Clean staging
if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
New-Item -ItemType Directory -Force "$Staging\checkpoints\files" | Out-Null
New-Item -ItemType Directory -Force "$Staging\code\files" | Out-Null

# Stage checkpoint
Copy-Item $BestCkpt.FullName "$Staging\checkpoints\files\"
Copy-Item "$MlDir\outputs\arabic_v1\config\config.yaml" "$Staging\checkpoints\files\"

# Stage code
Copy-Item -Recurse "$MlDir\src"    "$Staging\code\files\"
Copy-Item -Recurse "$MlDir\config" "$Staging\code\files\"

# Use Python API to upload (CLI 'datasets version' has a bug in v2.1.0)
$skipDataFlag = if ($SkipData) { "True" } else { "False" }
python - << 'PYEOF'
import kaggle, json, sys, os
from pathlib import Path

staging  = Path(os.environ['USERPROFILE']) / 'kaggle_staging'
ml_dir   = Path(r'MLDIR_PLACEHOLDER')
username = 'USERNAME_PLACEHOLDER'
skip_data = SKIPDATA_PLACEHOLDER

api = kaggle.KaggleApi()
api.authenticate()

def write_meta(folder, title, ds_id):
    (folder / 'dataset-metadata.json').write_text(
        json.dumps({'title': title, 'id': ds_id, 'licenses': [{'name': 'CC0-1.0'}]})
    )

def upload(folder, title, ds_id, message):
    write_meta(folder, title, ds_id)
    print(f'--- Uploading {ds_id} ---')
    try:
        api.dataset_create_version(str(folder), message, quiet=False, dir_mode='zip')
        print(f'    Done.')
    except Exception as e:
        print(f'    ERROR: {e}')

if not skip_data:
    data_dir = staging / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    src = ml_dir / 'data' / 'processed'
    dst = data_dir / 'processed'
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(str(src), str(dst))
    upload(data_dir, 'Arabic Handwriting Processed Data',
           f'{username}/arabic-handwriting-data', 'update processed data')
else:
    print('--- Skipping data ---')

upload(staging / 'checkpoints', 'Arabic Handwriting Checkpoints',
       f'{username}/arabic-handwriting-checkpoints', f'model-{STEP_PLACEHOLDER}')

upload(staging / 'code', 'Arabic Handwriting Code',
       f'{username}/arabic-handwriting-code', 'update src')
PYEOF

Write-Host ""
Write-Host "Upload complete. model-$Step is now in the Kaggle checkpoint dataset."
Write-Host "Next: open Kaggle, update notebook patience to [1000,700,500], run all cells."
