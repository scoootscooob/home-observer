#!/usr/bin/env bash
set -euo pipefail
# Bounded warm-up rendezvous. Controller uploads final code/data, then creates this marker.
for attempt in $(seq 1 450); do
  if [ -f /workspace/home-observer-launch-ready ]; then
    set +e
    bash /workspace/home-observer/scripts/runpod_job.sh
    job_exit=$?
    set -e
    printf '%s\n' "$job_exit" > /workspace/home-observer/artifacts/initial-exit
    # The outer supervisor's absolute deadline still applies throughout this rendezvous.
    # Keep weights and artifacts available for bounded controller-led evaluation/debugging.
    while [ ! -f /workspace/home-observer-finish ]; do
      sleep 2
    done
    if [ -f /workspace/home-observer/artifacts/final-exit ]; then
      job_exit=$(cat /workspace/home-observer/artifacts/final-exit)
      case "$job_exit" in ''|*[!0-9]*) exit 76;; esac
    fi
    exit "$job_exit"
  fi
  sleep 2
done
printf '%s\n' 'Final payload was not marked ready within 15 minutes.' >&2
exit 75
