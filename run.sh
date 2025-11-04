#!/usr/bin/env bash

# --- Configuration ---
# Set the session name
SESSION="unblink_engine"

# Define the full path to your project's root directory
# This makes the script runnable from anywhere
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Script Body ---
# Kill any existing tmux session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null

# Create a new, detached tmux session with the first window named "distributor"
tmux new-session -d -s "$SESSION" -n distributor -c "$PROJECT_DIR"

# Send the command to start the distributor service
# Using an environment variable for clarity
DISTRIBUTOR_CMD="bun run index.ts"
tmux send-keys -t "$SESSION:distributor" "$DISTRIBUTOR_CMD" C-m

# --- Worker Panes ---

# Create and run the embedding worker
tmux new-window -t "$SESSION" -n worker_embedding 
EMBEDDING_CMD="WORKER_TYPE=\"embedding\" MAX_LATENCY_MS=\"5000\" uv run python -m worker_embedding"
tmux send-keys -t "$SESSION:worker_embedding" "cd \"$PROJECT_DIR/py\" && $EMBEDDING_CMD" C-m


# Create and run the general VLM worker
tmux new-window -t "$SESSION" -n worker_vlm 
VLM_CMD="WORKER_TYPE=\"vlm\" MAX_LATENCY_MS=\"5000\" uv run python -m worker_vlm"
tmux send-keys -t "$SESSION:worker_vlm" "cd \"$PROJECT_DIR/py\" && $VLM_CMD" C-m


# Create and run the fast embedding worker
tmux new-window -t "$SESSION" -n worker_fast_embedding 
FAST_EMBEDDING_CMD="WORKER_TYPE=\"fast_embedding\" MAX_LATENCY_MS=\"200\" uv run python -m worker_embedding"
tmux send-keys -t "$SESSION:worker_fast_embedding" "cd \"$PROJECT_DIR/py\" && $FAST_EMBEDDING_CMD" C-m

# --- Finalization ---
# Select the 'distributor' window by default
tmux select-window -t "$SESSION:distributor"

# Attach to the newly created tmux session
echo "Attaching to tmux session '$SESSION'. To detach, press Ctrl+B then D."
tmux attach -t "$SESSION"