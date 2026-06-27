# ModelAtlas deployment — launchd setup

This directory holds `launchd` plists for running the persistent-knowledge
maintenance routines on the hub machine (macOS).

## Available plists

| Plist | Cadence | What it does |
|-------|---------|--------------|
| `com.modelatlas.sync.plist` | Weekly (Mon 09:00) | rsync worker JSONL from spokes, reconcile into the canonical DB, run coherence audit, rotate audit log if needed. Wraps `scripts/sync_and_reconcile.sh`. |
| `com.modelatlas.coherence.plist` | Daily (08:00) | Standalone read-only coherence audit. Optional — the weekly sync plist already runs the audit. Load this if you want daily drift surfacing. |
| `com.modelatlas.searxng_keepalive.plist` | Every 5 minutes | Probes `http://localhost:8888`. If unhealthy: starts the SearXNG container; if Docker daemon is dead, launches Docker Desktop. Wraps `scripts/searxng_keepalive.sh`. Critical for spoke Phase E workers — they hit SearXNG via tailnet and silently produce empty records if it's down. |
| `com.modelatlas.ingest.plist` | Continuous (KeepAlive) | **Pre-existing** background ingest daemon. Not part of the persistent-knowledge work. ⚠ The committed path is stale (`/Users/rohan/...`) — fix to `/Users/rohanvinaik/...` before loading. |

## Install (manual)

The standard pattern: symlink the plist into `~/Library/LaunchAgents/`,
then `launchctl load` it. Symlinking (rather than copying) means edits
to the in-repo plist propagate automatically.

```bash
# From the repo root
REPO="$(pwd)"
mkdir -p ~/Library/LaunchAgents

# Weekly sync + reconcile + coherence + rotation
ln -sf "${REPO}/deploy/com.modelatlas.sync.plist" ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.modelatlas.sync.plist

# (Optional) Daily standalone coherence audit
ln -sf "${REPO}/deploy/com.modelatlas.coherence.plist" ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.modelatlas.coherence.plist

# SearXNG / Docker self-healing keepalive (every 5 minutes)
ln -sf "${REPO}/deploy/com.modelatlas.searxng_keepalive.plist" ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.modelatlas.searxng_keepalive.plist
```

Verify a job is loaded:

```bash
launchctl list | grep modelatlas
```

Trigger a job immediately (don't wait for the schedule):

```bash
launchctl start com.modelatlas.sync
```

Check logs:

```bash
tail -f /tmp/modelatlas-sync.log /tmp/modelatlas-sync.err
```

## Uninstall

```bash
launchctl unload -w ~/Library/LaunchAgents/com.modelatlas.sync.plist
rm ~/Library/LaunchAgents/com.modelatlas.sync.plist
```

The same shape applies to `com.modelatlas.coherence.plist`.

## Customising the cadence

The plists ship with conservative defaults:

- **Sync**: Mondays at 09:00 local (Hub time). Change `Weekday`,
  `Hour`, `Minute` in `StartCalendarInterval`.
- **Coherence (standalone)**: Daily at 08:00.

Both routines are idempotent. Running more often than the default is
safe — the reconciler skips already-processed lines via line-hash, the
coherence audit is read-only, and rotation is a no-op below threshold.

## Spoke configuration (workers on other hosts)

Spokes are not configured by these plists. The hub fetches their output
via `rsync` inside `scripts/sync_and_reconcile.sh`. The expected spoke
layout:

```
~/model-atlas/data/incoming/<spoke-hostname>/
    phase_<n>_<YYYY-MM-DD>.jsonl
```

The wrapper script's `HOSTS=("macpro" "homebridge")` array names the
SSH hosts to pull from. SSH config (Headscale alias, key auth) must be
in place before the sync plist runs — `launchctl` runs with the user's
environment, including `~/.ssh/config`.

To add a new spoke:

1. Add its hostname to the `HOSTS` array in `scripts/sync_and_reconcile.sh`.
2. Ensure SSH key-based auth works from the hub.
3. Confirm the spoke writes JSONL to
   `~/model-atlas/data/incoming/<its-hostname>/`. Hostname casing
   matters — use `socket.gethostname()` lower-cased to avoid silent
   zero-row syncs (doc §64).

## What is NOT deployed by these plists

- **Spoke workers**. Each spoke runs its own launchd/cron schedule for
  emitting JSONL. The hub plists assume that's in place.
- **Worker code itself**. The spokes need a checked-out copy or rsync'd
  scripts.
- **Initial data sync**. First-time install needs a manual reconcile to
  populate `reconciler_processed`. Run
  `scripts/sync_and_reconcile.sh --dry-run` first to verify.

## See also

- `docs/admin.md` — the audit-logged write primitives.
- `docs/reconciler.md` — the JSONL → DB applicator.
- `docs/coherence.md` — what the audit surfaces.
- `PERSISTENT_KNOWLEDGE_GROUNDED_DATABASES.md` §45-§50, §62-§68 — the
  hub-and-spoke topology and detection of silent sync failure.
