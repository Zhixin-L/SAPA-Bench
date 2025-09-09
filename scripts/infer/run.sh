#!/usr/bin/env bash
set -euo pipefail

# ---- 1) Configure environment ----
# DO NOT hardcode real keys in code or scripts committed to Git. Use env vars.
export OPENAI_API_KEY="Your_key"
# Optional: an OpenAI-compatible gateway (otherwise leave empty)
# export OPENAI_BASE_URL="https://api.openai.com/v1"



# ---- 2) Run inference ----
python infer_api.py \
  --input_json /hy-tmp/input.json \
  --output_json /hy-tmp/results.json \
  --log_file /hy-tmp/gpt-4o/log/inference_RA_no_hint.log \
  --base_url your_base_url \
  --model gpt-4o \
  --max_workers 4 \
  --img_max_side 1024 \
  --jpeg_quality 85 \
  --temperature 1.0 \
  --top_p 0.95 \
  --log_level INFO
