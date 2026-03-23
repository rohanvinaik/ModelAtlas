# Resuming Pipeline Workers

Workers were paused on 2026-03-06. All results pulled locally and merged. This documents how to restart each worker from where it left off.

## Progress at pause

| Worker | Machine | Done | Total | % |
|--------|---------|------|-------|---|
| C1 extended | macpro | 15,099 | 19,498 | 77.4% |
| C2 shard_0 | macpro | 7,133 | 7,841 | 91.0% |
| C3 shard_1 | homebridge | 6,458 | 7,840 | 82.3% |

## Data locations (local)

All input shards and partial results are saved locally:

```
~/.cache/model-atlas/
  phase_c_work/
    shard_0.jsonl          # C2 input (7,841 models)
    results_0.jsonl        # C2 output (7,133 done)
    results_1.jsonl        # C2 shard_1 output (COMPLETE, 5,093)
  phase_c1_work/
    models_for_c1.jsonl    # C1 input (19,498 models)
    results_c1_extended.jsonl  # C1 output (15,099 done)
  phase_c3_work/
    c3_shard_1.jsonl       # C3 input (7,840 models)
    results_c3_1.jsonl     # C3 output (6,458 done)
```

## Restart: C2 shard_0 (macpro)

Only 708 models remaining. Uses `--resume` (reads existing output, skips done models).

```bash
# Upload latest results back to macpro
scp ~/.cache/model-atlas/phase_c_work/shard_0.jsonl \
    ~/.cache/model-atlas/phase_c_work/results_0.jsonl \
    macpro:/Users/squishfam/phase_c_work/

# SSH in and restart with --resume
ssh macpro "cd /Users/squishfam/phase_c_work && \
  python phase_c_worker.py \
    --input shard_0.jsonl \
    --output results_0.jsonl \
    --resume"
```

Note: `phase_c_worker.py` should already be on macpro at `/Users/squishfam/phase_c_work/`. If not:

```bash
scp scripts/phase_c_worker.py macpro:/Users/squishfam/phase_c_work/
```

## Restart: C1 extended (macpro)

4,399 models remaining. The C1 extended script is `phase_c1_extended.py`.

```bash
# Upload latest results
scp ~/.cache/model-atlas/phase_c1_work/models_for_c1.jsonl \
    ~/.cache/model-atlas/phase_c1_work/results_c1_extended.jsonl \
    macpro:/Users/squishfam/phase_c1/

# Rename to match expected filenames on macpro
ssh macpro "cd /Users/squishfam/phase_c1 && \
  mv results_c1_extended.jsonl results_c1.jsonl"

# Restart (the script has built-in resume via reading existing output)
ssh macpro "cd /Users/squishfam/phase_c1 && \
  python phase_c1_extended.py \
    --output results_c1.jsonl \
    --from-ids models_for_c1.jsonl"
```

Note: The script should already be on macpro at `/Users/squishfam/phase_c1/phase_c1_extended.py`.

## Restart: C3 shard_1 (homebridge)

1,382 models remaining. Uses `--resume`.

```bash
# Upload latest results
scp ~/.cache/model-atlas/phase_c3_work/c3_shard_1.jsonl \
    ~/.cache/model-atlas/phase_c3_work/results_c3_1.jsonl \
    homebridge:~/phase_c/

# Restart
ssh homebridge "cd ~/phase_c && \
  python phase_c3_worker.py \
    --input c3_shard_1.jsonl \
    --output results_c3_1.jsonl \
    --resume"
```

Note: `phase_c3_worker.py` should already be on homebridge at `~/phase_c/`.

## After completion

Once workers finish, pull results and merge:

```bash
# Pull
scp macpro:/Users/squishfam/phase_c_work/results_0.jsonl ~/.cache/model-atlas/phase_c_work/
scp macpro:/Users/squishfam/phase_c1/results_c1.jsonl ~/.cache/model-atlas/phase_c1_work/results_c1_extended.jsonl
scp homebridge:~/phase_c/results_c3_1.jsonl ~/.cache/model-atlas/phase_c3_work/

# Merge
uv run python -m model_atlas.ingest --merge-c2 ~/.cache/model-atlas/phase_c_work/results_0.jsonl
uv run python -m model_atlas.ingest --merge-c1 ~/.cache/model-atlas/phase_c1_work/results_c1_extended.jsonl
uv run python -m model_atlas.ingest --merge-c3 ~/.cache/model-atlas/phase_c3_work/results_c3_1.jsonl
```

## Pending work after all three complete

1. **Export C3 for shard_0** — C2 shard_0 needs its own quality gate pass:
   ```bash
   uv run python -m model_atlas.ingest --export-c3 1
   # Deploy to macpro and run phase_c3_worker.py
   ```

2. **Summary selection** — pick best summary per model:
   ```bash
   uv run python -m model_atlas.ingest --select-summaries
   ```

3. **Cut new release** with updated network.db
