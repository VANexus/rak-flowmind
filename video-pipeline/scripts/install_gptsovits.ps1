# Manual GPT-SoVITS installer (conda base is not writable on this machine).
# Uses D:\conda_envs\gptsovits env; ffmpeg already on PATH; cmake from pip.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'D:\conda_envs\gptsovits\python.exe'
$env:Path = "D:\conda_envs\gptsovits;D:\conda_envs\gptsovits\Scripts;D:\conda_envs\gptsovits\Library\bin;" + $env:Path

function Step($msg) { Write-Host "==== $msg" }

Step 'cmake via pip'
& $py -m pip install -q cmake

Step 'torch cu126'
& $py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $py -m pip install torch torchcodec --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'torch install failed' }
} else { Write-Host 'torch already present' }

Step 'download pretrained models'
if (-not (Test-Path 'GPT_SoVITS\pretrained_models\sv')) {
    & curl.exe -sS -L --retry 10 --retry-all-errors --retry-delay 3 -C - --speed-limit 50000 --speed-time 120 -o pretrained_models.zip 'https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip'
    if ($LASTEXITCODE -ne 0) { throw 'pretrained download failed' }
    & tar.exe -xf pretrained_models.zip -C GPT_SoVITS
    Remove-Item pretrained_models.zip -Force
}

Step 'download G2PWModel'
if (-not (Test-Path 'GPT_SoVITS\text\G2PWModel')) {
    & curl.exe -sS -L --retry 10 --retry-all-errors --retry-delay 3 -C - -o G2PWModel.zip 'https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip'
    if ($LASTEXITCODE -ne 0) { throw 'G2PW download failed' }
    & tar.exe -xf G2PWModel.zip -C GPT_SoVITS\text
    Remove-Item G2PWModel.zip -Force
}

Step 'pip deps (extra-req --no-deps, then requirements)'
& $py -m pip install -q -r extra-req.txt --no-deps
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host 'requirements failed, retrying without opencc line'
    (Get-Content requirements.txt) | Where-Object { $_ -notmatch 'opencc' } | Set-Content requirements_tmp.txt
    & $py -m pip install -r requirements_tmp.txt
    if ($LASTEXITCODE -ne 0) { throw 'requirements install failed' }
    & $py -m pip install opencc-python-reimplemented
}

Step 'nltk data'
if (-not (Test-Path 'nltk_data.zip')) {
    & curl.exe -sS -L --retry 10 --retry-all-errors --retry-delay 3 -C - -o nltk_data.zip 'https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip'
}
& tar.exe -xf nltk_data.zip -C D:\conda_envs\gptsovits
Remove-Item nltk_data.zip -Force -ErrorAction SilentlyContinue

Step 'open_jtalk dict'
if (-not (Test-Path 'open_jtalk_dic_utf_8-1.11.tar.gz')) {
    & curl.exe -sS -L --retry 10 --retry-all-errors --retry-delay 3 -C - -o open_jtalk_dic_utf_8-1.11.tar.gz 'https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz'
}
$pj = & $py -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))"
& tar.exe -xf open_jtalk_dic_utf_8-1.11.tar.gz -C $pj
Remove-Item open_jtalk_dic_utf_8-1.11.tar.gz -Force -ErrorAction SilentlyContinue

Step 'smoke test imports'
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
& $py -c "import sys; sys.path.insert(0,'.'); from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config; print('TTS module import OK')"
Write-Host '==== INSTALL DONE'
