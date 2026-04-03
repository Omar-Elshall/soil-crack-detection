#!/bin/bash
# Convert ONNX model to TensorRT engine — run this ON THE JETSON.
# trtexec compiles specifically for the local GPU (Orin Nano SM 8.7).
#
# Usage:
#   bash jetson/build_trt.sh
#
# Output:
#   results/efficientcracknet_fp16.trt

set -e

ONNX="results/efficientcracknet.onnx"
ENGINE="results/efficientcracknet_fp16.trt"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

if [ ! -f "$ONNX" ]; then
    echo "ERROR: $ONNX not found."
    echo "Run jetson/export_onnx.py on your PC first, then scp the file here."
    exit 1
fi

echo "Building TensorRT FP16 engine from $ONNX ..."
echo "This takes 2-5 minutes on first run."

$TRTEXEC \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw \
    --workspace=1024 \
    --iterations=10 \
    --warmUp=500

echo ""
echo "Engine saved to $ENGINE"
echo "Run live inference with:"
echo "  DISPLAY=:0 bash -l -c 'cd ~/soil-crack-detection && python3 jetson/live_inference.py --engine $ENGINE'"
