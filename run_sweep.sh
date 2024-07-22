#!/bin/bash

# Run this script as run_sweep.sh NUMBER_OF_AGENTS SWEEP_ID
# It will run $1 agents on the same sweep.

export WANDB_AGENT_DISABLE_FLAPPING=true
export WANDB_AGENT_MAX_INITIAL_FAILURES=1000

for i in $(seq $1); do
  wandb agent $2 & # &> /dev/null &
done

# Recursively list descendant processes. Necessary because we need to send SIGINT to python, not to wandb agent.
get_descendants() {
  local children=$(ps -o pid= --ppid "$1")

  for pid in $children; do
    get_descendants "$pid"
  done

  echo "$children"
}

handler() {
  kill -s SIGINT $(get_descendants $$)
  wait # Needed here too, because the wait outside this handler terminates because of SIGINT.
}

trap handler SIGINT
# If a SIGINT is received (CTRL+C), send a SIGINT to every agent. This will end currently running experiments with a
# "User abort" tag, but will NOT end the sweep. To end the sweep, either abort every experiment, or use the W&B Dashboard.


wait