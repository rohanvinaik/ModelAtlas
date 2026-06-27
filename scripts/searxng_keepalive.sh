#!/usr/bin/env bash
# SearXNG (+ Docker Desktop) self-healing keepalive.
#
# Purpose: Phase E workers on spokes hit the hub's SearXNG via Headscale
# tailnet. If Docker Desktop or the SearXNG container dies, spokes silently
# produce empty records (web_pages_fetched=0). This script detects that
# state and recovers.
#
# Cadence: runs every 5 minutes via com.modelatlas.searxng_keepalive plist.
# Idempotent — safe to invoke at any time. Logs to /tmp/modelatlas-searxng-keepalive.log.
#
# Recovery ladder:
#   1. SearXNG responds healthy        → no-op
#   2. SearXNG unreachable, docker ok  → docker start searxng (recreate if missing)
#   3. SearXNG unreachable, docker dead → open -a Docker (Docker Desktop GUI app)

set -uo pipefail

LOG=/tmp/modelatlas-searxng-keepalive.log
SEARXNG_URL="http://localhost:8888/search?q=keepalive&format=json"
SEARXNG_CONFIG="${SEARXNG_CONFIG:-$HOME/searxng-config}"
CONTAINER=searxng
IMAGE=searxng/searxng

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

probe_once() {
    : > /tmp/.searxng_probe
    local out size=0
    out=$(curl -s --max-time 6 -o /tmp/.searxng_probe -w "%{http_code}" "$SEARXNG_URL" 2>/dev/null || echo "000")
    [[ -f /tmp/.searxng_probe ]] && size=$(wc -c < /tmp/.searxng_probe)
    rm -f /tmp/.searxng_probe
    [[ "$out" == "200" && "$size" -gt 100 ]]
}

probe_searxng() {
    # SearXNG under load drops occasional requests (engine timeouts cause
    # queue backpressure). One failure is normal; two consecutive failures
    # ~2s apart is signal. This keeps us from "recovering" a healthy container.
    probe_once && return 0
    sleep 2
    probe_once
}

docker_alive() { docker info >/dev/null 2>&1; }

start_docker_app() {
    log "Docker daemon unreachable — launching Docker Desktop"
    open -ga Docker 2>/dev/null || { log "ERROR: open -a Docker failed"; return 1; }
    # Wait up to 90s for daemon
    for i in $(seq 1 18); do
        sleep 5
        if docker_alive; then
            log "Docker daemon up after ${i} retries (~$((i*5))s)"
            return 0
        fi
    done
    log "ERROR: Docker daemon did not come up within 90s"
    return 1
}

start_container() {
    # Existing container? Start it.
    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        log "Starting existing $CONTAINER container"
        docker start "$CONTAINER" >>"$LOG" 2>&1
        return $?
    fi
    # Missing — recreate with the durable bind-mount.
    log "Recreating $CONTAINER container from scratch (config=$SEARXNG_CONFIG)"
    docker run -d --name "$CONTAINER" \
        --restart unless-stopped \
        -p 8888:8080 \
        -v "${SEARXNG_CONFIG}:/etc/searxng" \
        "$IMAGE" >>"$LOG" 2>&1
    return $?
}

# ---- main probe + recovery ----

if probe_searxng; then
    # Healthy. Trim log if it gets large.
    if [[ -f "$LOG" && $(wc -c <"$LOG") -gt 1048576 ]]; then
        tail -c 524288 "$LOG" > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
    fi
    exit 0
fi

log "SearXNG unhealthy — entering recovery"

if ! docker_alive; then
    if ! start_docker_app; then
        log "ABORT: Docker Desktop did not start; cannot recover SearXNG"
        exit 1
    fi
fi

if ! start_container; then
    log "ERROR: container start/recreate failed"
    exit 1
fi

# Wait up to 60s for the container to start answering
for i in $(seq 1 12); do
    sleep 5
    if probe_searxng; then
        log "Recovered — SearXNG healthy after ${i} retries (~$((i*5))s)"
        exit 0
    fi
done

log "ERROR: SearXNG container started but is not answering after 60s"
exit 1
