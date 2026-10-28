#!/usr/bin/env bash
# Pin the libraries that change evaluation numbers. Run this once in the environment you
# trust (the GPU machine), commit requirements/constraints.txt, and the action will use it.
set -euo pipefail
mkdir -p requirements
pip freeze | grep -i -E '^(torch|torchao|transformers|lm[-_]eval|accelerate|datasets|numpy|safetensors|huggingface[-_]hub|tokenizers|pydantic|pyyaml|typer)==' > requirements/constraints.txt
echo "Wrote requirements/constraints.txt:"; cat requirements/constraints.txt
