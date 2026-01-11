#!/usr/bin/env bash

# Launch script_debug_crssm_task_pred.py inside a tmux session with the repo venv activated.

set -euo pipefail

SESSION_NAME="${1:-task_pred}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv/bin/activate"
TRAIN_SCRIPT="${PROJECT_ROOT}/script_debug_crssm_task_pred.py"

if [[ ! -f "${VENV_PATH}" ]]; then
    echo "Virtual environment not found at ${VENV_PATH}."
    echo "Create it (e.g., python3 -m venv .venv) and install dependencies before rerunning."
    exit 1
fi

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training script not found at ${TRAIN_SCRIPT}."
    exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session \"${SESSION_NAME}\" already exists. Attach with: tmux attach -t ${SESSION_NAME}"
    exit 1
fi

RUN_COMMAND=$(cat <<'EOF'
cd "__PROJECT_ROOT__" && \
source "__VENV_PATH__" && \
python "__TRAIN_SCRIPT__" ; \
echo "" ; \
echo "Training finished in tmux session. Press Ctrl+C to stop or run other commands." ; \
exec bash
EOF
)

RUN_COMMAND="${RUN_COMMAND/__PROJECT_ROOT__/${PROJECT_ROOT}}"
RUN_COMMAND="${RUN_COMMAND/__VENV_PATH__/${VENV_PATH}}"
RUN_COMMAND="${RUN_COMMAND/__TRAIN_SCRIPT__/${TRAIN_SCRIPT}}"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '${RUN_COMMAND}'"
echo "Started tmux session \"${SESSION_NAME}\" running script_debug_crssm_task_pred.py."
echo "Attach with: tmux attach -t ${SESSION_NAME}"
