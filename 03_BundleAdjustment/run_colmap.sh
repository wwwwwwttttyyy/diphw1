#!/bin/bash
# Simple COLMAP pipeline
# Script arranged with AI help. I checked the path and commands.

set -e

IMAGE_PATH="data/images"
COLMAP_PATH="data/colmap_cuda"
DATABASE_PATH="$COLMAP_PATH/database.db"

mkdir -p "$COLMAP_PATH/sparse"
mkdir -p "$COLMAP_PATH/dense"

echo "Step 1: feature extraction"
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 1

echo "Step 2: feature matching"
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH" \
    --FeatureMatching.use_gpu 1

echo "Step 3: sparse reconstruction"
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

echo "Step 4: save sparse ply"
colmap model_converter \
    --input_path "$COLMAP_PATH/sparse/0" \
    --output_path "$COLMAP_PATH/sparse/sparse.ply" \
    --output_type PLY

echo "Step 5: image undistortion"
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$COLMAP_PATH/sparse/0" \
    --output_path "$COLMAP_PATH/dense"

echo "Step 6: patch match stereo"
colmap patch_match_stereo \
    --workspace_path "$COLMAP_PATH/dense" \
    --PatchMatchStereo.geom_consistency 1

echo "Step 7: stereo fusion"
colmap stereo_fusion \
    --workspace_path "$COLMAP_PATH/dense" \
    --output_path "$COLMAP_PATH/dense/fused.ply"

echo "Done"
echo "Sparse: $COLMAP_PATH/sparse/sparse.ply"
echo "Dense:  $COLMAP_PATH/dense/fused.ply"
