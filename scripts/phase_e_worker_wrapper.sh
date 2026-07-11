#!/usr/bin/env bash
# Graceful-exit wrapper for phase_e_worker.py.
#
# Runs the worker with all its args, then on clean exit checks whether any
# other phase_e_worker.py processes are still alive. If this was the last
# worker, unloads the dedicated ModelAtlas Ollama on 11435 so we don't sit
# there holding ~8 GB of VRAM idle between tiers.
#
# Idempotent: if the Ollama plist is already unloaded, the unload is a no-op.
# Safe: won't touch the default Ollama on 11434 or any other process.
# Diagnostic: logs its decisions to /tmp/modelatlas-phase_e-wrapper.log.
#
# To restart Ollama for the next tier:
#   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.ollama.plist

set -o pipefail

PYTHON="/Users/rohanvinaik/tools/infrastructure/ModelAtlas/.venv/bin/python"
WORKER="/Users/rohanvinaik/tools/infrastructure/ModelAtlas/scripts/phase_e_worker.py"
OLLAMA_PLIST="$HOME/Library/LaunchAgents/com.modelatlas.ollama.plist"
LOG="/tmp/modelatlas-phase_e-wrapper.log"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

# Rotate log at 512 KB.
if [[ -f "$LOG" ]] && [[ $(stat -f%z "$LOG" 2>/dev/null || echo 0) -gt 524288 ]]; then
    mv "$LOG" "$LOG.1"
fi

log "wrapper start (pid $$) — args: $*"

"$PYTHON" "$WORKER" "$@"
RC=$?

log "worker exited rc=$RC"

if [[ $RC -ne 0 ]]; then
    # Non-zero exit — worker crashed or was signalled. launchd's
    # KeepAlive.Crashed=true will restart it. Do NOT unload Ollama.
    log "non-zero exit — leaving Ollama loaded for restart"
    exit $RC
fi

# Clean exit. Give launchd a moment to reap sibling worker processes.
sleep 5

# Am I the last phase_e_worker.py alive?
# Filter to workers only (not the wrapper itself; not this shell).
OTHERS=$(pgrep -f 'phase_e_worker\.py' 2>/dev/null | grep -v "^$$\$" | wc -l | tr -d ' ')
log "sibling phase_e_worker.py processes still alive: $OTHERS"

if [[ "$OTHERS" -gt 0 ]]; then
    log "another worker still running — Ollama stays loaded"
    exit 0
fi

# I was the last worker. Unload the dedicated Ollama if it's loaded.
if launchctl list 2>/dev/null | grep -q '^\S*\s\S*\s*com\.modelatlas\.ollama$'; then
    log "last worker done — unloading com.modelatlas.ollama"
    if launchctl unload -w "$OLLAMA_PLIST" 2>>"$LOG"; then
        log "Ollama unloaded — VRAM released"
    else
        log "WARN: launchctl unload failed"
    fi
else
    log "Ollama plist not loaded — no action"
fi

exit 0
