#!/usr/bin/env bash
# Downloads the public ASSISTments 2009-2010 skill-builder file (64 MB) into data/.
# Original source and citation page:
#   https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -L -o data/skill_builder_data_corrected.csv \
  "https://huggingface.co/datasets/OloriBern/assistments-dataset/resolve/main/skill_builder_data_corrected.csv"
echo "162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da  data/skill_builder_data_corrected.csv" | shasum -a 256 -c -
