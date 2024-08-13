#!/bin/bash
set -e
echo "Downloading..."

source .env

download_weights() {
  model=$1
  revision=$2
  include=$3
  huggingface-cli download $model \
    --revision $revision \
    --token $HUGGINGFACE_TOKEN \
    --local-dir ./weights \
    --include="*.$include"
}

triplex_revision="d3c3a297391cb5609ff7f5350c426ed0908112da"
ms_phi_revision="d548c233192db00165d842bf8edff054bb3212f8"

download_weights "sciphi/triplex" "$triplex_revision" "json"
download_weights "sciphi/triplex" "$triplex_revision" "safetensors"
download_weights "sciphi/triplex" "$triplex_revision" "model"
download_weights "microsoft/Phi-3-mini-128k-instruct" "$ms_phi_revision" "py"


echo "Done!"