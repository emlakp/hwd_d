#!/bin/bash

# Create a new tmux session for LSTM training
SESSION_NAME="lstm_train"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the tmux session
tmux send-keys -t $SESSION_NAME "cd /home/akopyane/rl/lumos" C-m
tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "CUDA_LAUNCH_BLOCKING=1 python scripts/train_wm_debug_lstm.py" C-m

echo "LSTM training started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl+b then d"
