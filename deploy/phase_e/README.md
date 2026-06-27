# Phase E worker deployment

Workers run via `nohup` right now (won't survive reboot). Plists are
staged on every machine for GUI-session activation when convenient.

## Current runtime — nohup processes

| Where | Shards | Model | Banks | Throttle | Started |
|-------|--------|-------|-------|----------|---------|
| hub (local) | 0, 1 | qwen3.5:4b | ALL 8 | `--delay 5`, Nice=5 | nohup, no auto-restart |
| macpro | 2, 3 | qwen2.5:3b | CAPABILITY,QUALITY | `--delay 5`, Nice=20 | nohup, no auto-restart |
| homebridge | 4, 5 | qwen2.5:3b | CAPABILITY,QUALITY | `--delay 5`, Nice=20 | nohup, no auto-restart |

Logs: `/tmp/modelatlas-phase_e-<host>-<n>.log` on each machine.

## Status check

```bash
# Hub
ps -ef | grep phase_e_worker | grep -v grep
tail /tmp/modelatlas-phase_e-local-{0,1}.log

# Remote
ssh macpro 'ps -ef | grep phase_e_worker | grep -v grep'
ssh homebridge 'ps -ef | grep phase_e_worker | grep -v grep'
```

## To make persistent (survive reboot)

`launchctl load` fails over SSH with SIGABRT (modern macOS requires Aqua
session for user LaunchAgents). To activate the staged plists:

1. **On the hub**: open Terminal in your normal GUI session, run

   ```bash
   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.phase_e.local.0.plist
   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.phase_e.local.1.plist
   ```

2. **On macpro and homebridge**: connect via Screen Sharing (or have the
   user open Terminal locally on each), run

   ```bash
   # macpro
   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.phase_e.macpro.2.plist
   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.phase_e.macpro.3.plist

   # homebridge
   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.phase_e.homebridge.4.plist
   launchctl load -w ~/Library/LaunchAgents/com.modelatlas.phase_e.homebridge.5.plist
   ```

3. **Kill the current nohup processes** first to avoid duplicates:

   ```bash
   pkill -f phase_e_worker
   ssh macpro 'pkill -f phase_e_worker'
   ssh homebridge 'pkill -f phase_e_worker'
   ```

Once activated, launchctl will keep them alive with `KeepAlive=Crashed`
and auto-restart on system reboot.

## SearXNG dependency

The container at `localhost:8888` on the hub is the search backend.
Workers connect via `--searxng`. After the 2026-05-21 reconfiguration:

- Config file: `~/searxng-config/settings.yml` (durable, NOT in /tmp)
- Container restart policy: `unless-stopped`
- All default engines enabled; per-engine auto-suspension handles
  individual rate-limit failures

If SearXNG breaks again, check:

```bash
docker ps | grep searxng
docker logs searxng 2>&1 | tail -40
curl -s "http://localhost:8888/search?q=test&format=json" | head -c 500
```

## Sync wrapper

`scripts/sync_and_reconcile.sh` (runs Mon 09:00 via launchd) now pulls
Phase E result files alongside C-phase results. Files land in
`~/.cache/model-atlas/phase_e_work/phase_e_results_<n>.jsonl`. Phase E
merge (`phase_e_postprocess.py`) is **not** automated — run it manually
when ready.
