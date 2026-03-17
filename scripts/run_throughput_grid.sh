#!/bin/bash
# Run throughput grid search for Gemma 3 1B and 4B models
# Tests: q4 (W4A16) and q8 (W8A8) at 4K, 8K, 16K, 32K, 64K, 128K context

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../venv/bin/activate"

echo "=== Throughput Grid Search ==="
echo "Models: 1B (W8A8), 4B (W4A16, W8A8)"
echo "Contexts: 4K, 8K, 16K, 32K, 64K, 128K"
echo ""

# Default: run full grid
# Override with command line args

python "${SCRIPT_DIR}/throughput_grid_search.py" "$@"
