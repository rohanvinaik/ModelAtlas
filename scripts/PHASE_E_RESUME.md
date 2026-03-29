# Phase E Web Enrichment — Worker Status & Resume

## Deployment (2026-03-29)

17,843 models with 100+ downloads, split across 6 shards.
SearXNG on local laptop (Docker, port 8888), accessed by all workers.

| Shard | Machine | Model | Banks | Status |
|-------|---------|-------|-------|--------|
| 0 | local laptop | qwen3.5:4b | ALL 8 | RUNNING |
| 1 | local laptop | qwen3.5:4b | ALL 8 | RUNNING |
| 2 | macpro | qwen2.5:3b | CAPABILITY,QUALITY | RUNNING |
| 3 | macpro | qwen2.5:3b | CAPABILITY,QUALITY | RUNNING |
| 4 | homebridge | qwen2.5:3b | CAPABILITY,QUALITY | RUNNING |
| 5 | homebridge | qwen2.5:3b | CAPABILITY,QUALITY | RUNNING |

**Rate estimates:**
- Local: ~2 models/min (2 workers, qwen3.5:4b with Metal GPU, 8 banks)
- Remote: ~0.5 models/min each (Xeon CPU, 2 banks only, 300s timeout)
- Combined: ~4 models/min → ~75 hours for full corpus

## Architecture

```
SearXNG (Docker, laptop:8888)
   ↑ JSON API (no rate limit)
   |
   ├── local worker 0 → Ollama qwen3.5:4b (localhost)
   ├── local worker 1 → Ollama qwen3.5:4b (localhost)
   ├── macpro worker 2 → Ollama qwen2.5:3b (macpro:11434)
   ├── macpro worker 3 → Ollama qwen2.5:3b (macpro:11434)
   ├── homebridge worker 4 → Ollama qwen2.5:3b (homebridge:11434)
   └── homebridge worker 5 → Ollama qwen2.5:3b (homebridge:11434)
```

Remotes reach SearXNG via Headscale (100.64.0.1:8888).
Each machine runs its own Ollama instance.
Workers use `--snippets-only` (search snippets, no page fetching).

## Start Commands

### Prerequisites
```bash
# SearXNG must be running on the local laptop
docker start searxng  # or see setup below
```

### SearXNG Setup (one-time)
```bash
mkdir -p /tmp/searxng
cat > /tmp/searxng/settings.yml << 'EOF'
use_default_settings: true
search:
  formats: [html, json]
server:
  limiter: false
  secret_key: "modelatlas-local"
EOF

docker run -d --name searxng -p 8888:8080 \
  -v /tmp/searxng/settings.yml:/etc/searxng/settings.yml \
  searxng/searxng
```

### Local laptop (shards 0+1, all banks)
```bash
python scripts/phase_e_worker.py \
    --input ~/.cache/model-atlas/phase_e_work/shard_0.jsonl \
    --output ~/.cache/model-atlas/phase_e_work/results_0.jsonl \
    --model qwen3.5:4b --searxng http://localhost:8888 \
    --snippets-only --delay 0.2 --max-pages 2 --timeout 8 --resume &

python scripts/phase_e_worker.py \
    --input ~/.cache/model-atlas/phase_e_work/shard_1.jsonl \
    --output ~/.cache/model-atlas/phase_e_work/results_1.jsonl \
    --model qwen3.5:4b --searxng http://localhost:8888 \
    --snippets-only --delay 0.2 --max-pages 2 --timeout 8 --resume &
```

### Mac Pro (shards 2+3, CAPABILITY+QUALITY only)
```bash
ssh macpro
nohup python3 /Users/squishfam/phase_e_worker.py \
    --input /Users/squishfam/phase_e_shard_2.jsonl \
    --output /Users/squishfam/phase_e_results_2.jsonl \
    --model qwen2.5:3b --searxng http://100.64.0.1:8888 \
    --snippets-only --delay 0.3 --max-pages 2 --timeout 300 \
    --banks CAPABILITY,QUALITY --resume \
    > /Users/squishfam/phase_e_shard_2.log 2>&1 &

nohup python3 /Users/squishfam/phase_e_worker.py \
    --input /Users/squishfam/phase_e_shard_3.jsonl \
    --output /Users/squishfam/phase_e_results_3.jsonl \
    --model qwen2.5:3b --searxng http://100.64.0.1:8888 \
    --snippets-only --delay 0.3 --max-pages 2 --timeout 300 \
    --banks CAPABILITY,QUALITY --resume \
    > /Users/squishfam/phase_e_shard_3.log 2>&1 &
```

### Homebridge (shards 4+5, CAPABILITY+QUALITY only)
```bash
ssh homebridge
nohup python3 ~/phase_e_worker.py \
    --input ~/phase_e_shard_4.jsonl \
    --output ~/phase_e_results_4.jsonl \
    --model qwen2.5:3b --searxng http://100.64.0.1:8888 \
    --snippets-only --delay 0.3 --max-pages 2 --timeout 300 \
    --banks CAPABILITY,QUALITY --resume \
    > ~/phase_e_shard_4.log 2>&1 &

nohup python3 ~/phase_e_worker.py \
    --input ~/phase_e_shard_5.jsonl \
    --output ~/phase_e_results_5.jsonl \
    --model qwen2.5:3b --searxng http://100.64.0.1:8888 \
    --snippets-only --delay 0.3 --max-pages 2 --timeout 300 \
    --banks CAPABILITY,QUALITY --resume \
    > ~/phase_e_shard_5.log 2>&1 &
```

## Monitor

```bash
bash scripts/phase_e_dashboard.sh          # one-shot
bash scripts/phase_e_dashboard.sh --watch   # auto-refresh every 60s
```

## Pull & Merge

```bash
scp macpro:/Users/squishfam/phase_e_results_2.jsonl ~/.cache/model-atlas/phase_e_work/results_2.jsonl
scp macpro:/Users/squishfam/phase_e_results_3.jsonl ~/.cache/model-atlas/phase_e_work/results_3.jsonl
scp homebridge:~/phase_e_results_4.jsonl ~/.cache/model-atlas/phase_e_work/results_4.jsonl
scp homebridge:~/phase_e_results_5.jsonl ~/.cache/model-atlas/phase_e_work/results_5.jsonl

# Dry-run first
python -m model_atlas.ingest --merge-e ~/.cache/model-atlas/phase_e_work/results_*.jsonl --merge-e-dry-run

# Actual merge
python -m model_atlas.ingest --merge-e ~/.cache/model-atlas/phase_e_work/results_*.jsonl
```

## Multi-Pass Strategy

**Pass 1 (current):** `--snippets-only` — fast, uses SearXNG snippets only. Gets ~60% of available web signal.

**Pass 2 (future):** Full page fetch for models where Pass 1 found promising snippets but thin content. Re-export only those models, run with page fetching enabled.

**Pass 3 (future):** Remaining banks on remote machines after local completes all 8.

## Notes

- SearXNG must be running on laptop for all workers (including remote)
- Remote workers reach SearXNG via Headscale at 100.64.0.1:8888
- macpro/homebridge are old Xeon CPUs — use qwen2.5:3b + 2 banks only
- Local laptop (Apple Silicon) runs qwen3.5:4b + all 8 banks
- `think: false` in native Ollama API is critical for qwen3.5 (saves 10x+ time)
- All results at confidence=0.4, merge QC prevents overwriting higher-confidence data
