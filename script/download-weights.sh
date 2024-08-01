#!/bin/bash
set -e
echo "Downloading..."

source .env

download_weights() {
  include=$1
  huggingface-cli download sciphi/triplex \
    --revision d3c3a297391cb5609ff7f5350c426ed0908112da \
    --token $HUGGINGFACE_TOKEN \
    --local-dir ./checkpoints \
    --include="*.$include"
}

download_weights "json"
download_weights "safetensors"
download_weights "model"

echo "Done!"