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
3. Run `script/download-weights.sh`