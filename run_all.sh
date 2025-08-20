#!/usr/bin/env bash
set -euo pipefail

python run_eval_all.py \
  --models \
    "llm-jp/llm-jp-3-1.8b:sae/SAEs-LLM-jp-3-1.8B-dolma:data/tokenized/llmjp_dolma_test_manifest.json" \
    "allenai/OLMo-2-0425-1B:sae/SAEs-OLMo-2-0425-1B-dolma:data/tokenized/olmo2_dolma_warp_html_test_manifest.json" \
  --n_d 16 --k 32 --nl Scalar --ckpt 988240 --lr 0.001
