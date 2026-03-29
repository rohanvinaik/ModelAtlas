#!/bin/bash
# Phase E Web Enrichment Dashboard
# Usage: bash scripts/phase_e_dashboard.sh [--watch]

WORK_DIR="$HOME/.cache/model-atlas/phase_e_work"
TOTAL=17843

bold=$'\033[1m'
dim=$'\033[2m'
green=$'\033[32m'
red=$'\033[31m'
cyan=$'\033[36m'
reset=$'\033[0m'

bar() {
    local done=$1 total=$2 width=30
    if [ "$total" -eq 0 ]; then
        printf '%0.s░' $(seq 1 $width)
        printf "   0%%"
        return
    fi
    local pct=$((done * 100 / total))
    local filled=$((done * width / total))
    local empty=$((width - filled))
    printf "${green}"
    for ((j=0; j<filled; j++)); do printf '█'; done
    printf "${dim}"
    for ((j=0; j<empty; j++)); do printf '░'; done
    printf "${reset} %3d%%" "$pct"
}

safe_count() {
    # Return numeric line count from a file, 0 if missing
    local f="$1"
    if [ -f "$f" ]; then
        local n
        n=$(wc -l < "$f" 2>/dev/null)
        echo "${n// /}"
    else
        echo 0
    fi
}

remote_count() {
    # SSH line count, returns 0 on any failure
    local host="$1" path="$2"
    local raw
    raw=$(ssh -o ConnectTimeout=3 -o BatchMode=yes "$host" "wc -l < $path 2>/dev/null || echo 0" 2>/dev/null)
    local n
    n=$(echo "$raw" | tr -dc '0-9' | head -c 10)
    echo "${n:-0}"
}

dashboard() {
    printf '\033[2J\033[H'
    echo "${bold}Phase E Web Enrichment Dashboard${reset}"
    echo "${dim}$(date '+%Y-%m-%d %H:%M:%S')${reset}"
    echo ""

    local grand_done=0

    # --- Local ---
    echo "${bold}LOCAL (laptop)${reset}"
    for i in 0 1; do
        local t=$(safe_count "$WORK_DIR/shard_${i}.jsonl")
        local d=$(safe_count "$WORK_DIR/results_${i}.jsonl")
        grand_done=$((grand_done + d))
        printf "  shard_%d: " "$i"
        bar "$d" "$t"
        printf "  %d/%d\n" "$d" "$t"
    done

    # --- Mac Pro ---
    echo ""
    echo "${bold}MACPRO${reset}"
    for i in 2 3; do
        local t=$(safe_count "$WORK_DIR/shard_${i}.jsonl")
        local d=$(remote_count macpro "/Users/squishfam/phase_e_results_${i}.jsonl")
        grand_done=$((grand_done + d))
        printf "  shard_%d: " "$i"
        bar "$d" "$t"
        printf "  %d/%d\n" "$d" "$t"
    done

    # --- Homebridge ---
    echo ""
    echo "${bold}HOMEBRIDGE${reset}"
    for i in 4 5; do
        local t=$(safe_count "$WORK_DIR/shard_${i}.jsonl")
        local d=$(remote_count homebridge "~/phase_e_results_${i}.jsonl")
        grand_done=$((grand_done + d))
        printf "  shard_%d: " "$i"
        bar "$d" "$t"
        printf "  %d/%d\n" "$d" "$t"
    done

    # --- Summary ---
    local remaining=$((TOTAL - grand_done))
    echo ""
    echo "${bold}TOTAL${reset}"
    printf "  "
    bar "$grand_done" "$TOTAL"
    printf "  %d/%d models  ${dim}(%d remaining)${reset}\n" "$grand_done" "$TOTAL" "$remaining"
    echo ""
}

if [ "$1" = "--watch" ] || [ "$1" = "-w" ]; then
    while true; do
        dashboard
        echo "${dim}Refreshing every 60s... (Ctrl+C to stop)${reset}"
        sleep 60
    done
else
    dashboard
fi
