# 一键下载本流水线所需的权重与工具（模型不进 git：GitHub 单文件 100MB 硬限制）
# 运行前：先 git clone https://github.com/sczhou/ProPainter.git work\ProPainter
# 用法： powershell -File fetch_models.ps1
$ErrorActionPreference = 'Stop'
$work = 'D:\Desktop\obj114514\work'   # ← 换机器改成你的 work 目录
New-Item -ItemType Directory -Force -Path $work | Out-Null
Set-Location $work

function Get-File($url, $out) {
    Write-Host ">> $out"
    & curl.exe -sS -L --retry 10 --retry-all-errors --retry-delay 3 -C - -o $out $url
    if ($LASTEXITCODE -ne 0) { throw "download failed: $url" }
}

# ---- 1. ProPainter 检查点（camenduru 官方镜像，~190MB）----
$ppw = Join-Path $work 'ProPainter\weights'
New-Item -ItemType Directory -Force -Path $ppw | Out-Null
foreach ($f in 'ProPainter.pth', 'recurrent_flow_completion.pth', 'raft-things.pth') {
    if (-not (Test-Path (Join-Path $ppw $f))) {
        Get-File "https://huggingface.co/camenduru/ProPainter/resolve/main/$f" (Join-Path $ppw $f)
    } else { Write-Host "skip $f (exists)" }
}

# ---- 2. Real-ESRGAN ncnn vulkan（43MB，x4 超分 exe + 模型）----
$ra = Join-Path $work 'realesrgan'
if (-not (Test-Path (Join-Path $ra 'realesrgan-ncnn-vulkan.exe'))) {
    New-Item -ItemType Directory -Force -Path $ra | Out-Null
    Push-Location $ra
    Get-File 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip' 'ra.zip'
    & tar.exe -xf ra.zip; Remove-Item ra.zip -Force
    Pop-Location
} else { Write-Host 'skip realesrgan (exists)' }

# ---- 3. shared 版 ffmpeg DLL（torchcodec/GPT-SoVITS 解码参考音频需要，~219MB）----
$ff = Join-Path $work 'ffmpeg_shared'
if (-not (Test-Path (Join-Path $ff 'ffmpeg-master-latest-win64-gpl-shared\bin\avcodec-63.dll'))) {
    New-Item -ItemType Directory -Force -Path $ff | Out-Null
    Push-Location $ff
    Get-File 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip' 'ff.zip'
    & tar.exe -xf ff.zip; Remove-Item ff.zip -Force
    Pop-Location
} else { Write-Host 'skip ffmpeg_shared (exists)' }

# ---- 4. GPT-SoVITS 预训练模型（~4.3GB，用仓库自带安装脚本，见 scripts\install_gptsovits.ps1）----
Write-Host @'

GPT-SoVITS 模型下载请执行：
  git clone https://github.com/RVC-Boss/GPT-SoVITS.git work\GPT-SoVITS
  # 把本仓库 scripts\install_gptsovits.ps1 复制进去后运行（内部会从 HF 下载全部预训练）
  powershell -File work\GPT-SoVITS\install_custom.ps1

'@
Write-Host '==== fetch done'
