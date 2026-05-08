$ErrorActionPreference = "Stop"

# Script arranged with AI help. I checked the path and commands.
$ColmapExe = "C:\Users\29643\Downloads\colmap-x64-windows-cuda\bin\colmap.exe"
if (-not (Test-Path $ColmapExe)) {
    $ColmapExe = "colmap"
}

$ImagePath = "data\images"
$ColmapPath = "data\colmap_cuda"
$DatabasePath = Join-Path $ColmapPath "database.db"

New-Item -ItemType Directory -Force -Path (Join-Path $ColmapPath "sparse") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ColmapPath "dense") | Out-Null

Write-Host "Using COLMAP: $ColmapExe"

Write-Host "Step 1: feature extraction"
& $ColmapExe feature_extractor `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --ImageReader.camera_model PINHOLE `
    --ImageReader.single_camera 1 `
    --FeatureExtraction.use_gpu 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 2: feature matching"
& $ColmapExe exhaustive_matcher `
    --database_path $DatabasePath `
    --FeatureMatching.use_gpu 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 3: sparse reconstruction"
& $ColmapExe mapper `
    --database_path $DatabasePath `
    --image_path $ImagePath `
    --output_path (Join-Path $ColmapPath "sparse")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 4: save sparse ply"
& $ColmapExe model_converter `
    --input_path (Join-Path $ColmapPath "sparse\0") `
    --output_path (Join-Path $ColmapPath "sparse\sparse.ply") `
    --output_type PLY
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 5: image undistortion"
& $ColmapExe image_undistorter `
    --image_path $ImagePath `
    --input_path (Join-Path $ColmapPath "sparse\0") `
    --output_path (Join-Path $ColmapPath "dense")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 6: patch match stereo"
& $ColmapExe patch_match_stereo `
    --workspace_path (Join-Path $ColmapPath "dense") `
    --PatchMatchStereo.geom_consistency 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 7: stereo fusion"
& $ColmapExe stereo_fusion `
    --workspace_path (Join-Path $ColmapPath "dense") `
    --output_path (Join-Path $ColmapPath "dense\fused.ply")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Done"
Write-Host "Sparse: data\colmap_cuda\sparse\sparse.ply"
Write-Host "Dense:  data\colmap_cuda\dense\fused.ply"
