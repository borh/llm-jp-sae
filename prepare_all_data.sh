#!/usr/bin/env bash

set -euxo pipefail

time uv run python prepare_data.py \
  --model_name_or_dir llm-jp/llm-jp-3-1.8b \
  --warp_sample_rate 1.0 \
  --label llmjp_

time uv run python prepare_data.py \
  --model_name_or_dir llm-jp/llm-jp-3-1.8b \
  --dolma_sample_rate 1.0 \
  --label llmjp_

time uv run python prepare_data.py \
  --model_name_or_dir allenai/OLMo-2-0425-1B \
  --warp_sample_rate 1.0 \
  --label olmo2_

time uv run python prepare_data.py \
  --model_name_or_dir allenai/OLMo-2-0425-1B \
  --dolma_sample_rate 1.0 \
  --label olmo2_
