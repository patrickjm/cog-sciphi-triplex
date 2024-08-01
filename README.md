# cog-sciphi-triplex
Cog wrapper for SchiPhi's Triplex model 

## Prerequisites
Note: This guide written for Mac OS - others might take some phenaigling

- Docker installed
- Cog cli - `brew install cog`
- Huggingface cli - `pip install huggingface_hub`

## Building & Pushing
1. `cp .env.example .env`
2. Fill in the `.env` file with your credentials
3. Run: 
  ```bash
  source .env
  script/download-weights.sh
  echo $REPLICATE_TOKEN | cog login --token-stdin

  cog push r8.im/YOUR_ORG/YOUR_REPO --separate-weights
  ```