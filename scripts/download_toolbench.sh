#!/bin/bash
# Download ToolBench tool data into .toolbench_tmp/toolenv/tools by default.
# Uses sparse checkout to avoid cloning the full repository history.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="${TOOLBENCH_TARGET_DIR:-$PROJECT_ROOT/.toolbench_tmp/toolenv/tools}"
CLONE_DIR="$PROJECT_ROOT/.toolbench_clone"

echo "==> Cloning ToolBench (sparse, depth=1)..."
rm -rf "$CLONE_DIR"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/OpenBMB/ToolBench.git "$CLONE_DIR"

cd "$CLONE_DIR"
git sparse-checkout set data/toolenv/tools

echo "==> Copying tool data to $TARGET_DIR..."
mkdir -p "$TARGET_DIR"
# Preserve existing files unless the caller deletes the target directory first.
cp -rn "$CLONE_DIR/data/toolenv/tools/"* "$TARGET_DIR/" 2>/dev/null || true

# Count what we got
TOOL_COUNT=$(find "$TARGET_DIR" -name "*.json" | wc -l | tr -d ' ')
CAT_COUNT=$(find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

echo "==> Done: $TOOL_COUNT tools across $CAT_COUNT categories"
echo "==> Data at: $TARGET_DIR"
echo "==> Use with: export TOOLGEN_TOOLENV_DIR=$TARGET_DIR"

# Cleanup clone
rm -rf "$CLONE_DIR"
echo "==> Cleaned up clone directory"
