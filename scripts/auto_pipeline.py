#!/usr/bin/env python3
"""Autonomous pipeline orchestrator for ModelAtlas.

Runs the C2 → Summary Selection → C3 → D1 → D2 → D3 loop
across macpro and homebridge, polling for worker completion.

Usage:
    python scripts/auto_pipeline.py [--heal-budget 200] [--dry-run]

Requires: SSH key auth to macpro and homebridge.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

# --- Config ---
MACHINES = {
    "macpro": {
        "ssh": "macpro",
        "user": "squishfam",
        "workdir": "/Users/squishfam/auto_pipeline",
        "python": "/usr/bin/python3",
    },
    "homebridge": {
        "ssh": "homebridge",
        "user": "homebridge",
        "workdir": "/Users/homebridge/auto_pipeline",
        "python": "/usr/bin/python3",
    },
}

LOCAL_CACHE = Path.home() / ".cache" / "model-atlas"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLL_INTERVAL = 120  # seconds between progress checks
WORKERS = {
    "c2": "phase_c_worker.py",
    "c3": "phase_c3_worker.py",
    "d3": "phase_d_worker.py",
}


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ssh(machine: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    m = MACHINES[machine]
    return subprocess.run(
        ["ssh", m["ssh"], cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def scp_to(machine: str, local: str, remote: str) -> None:
    m = MACHINES[machine]
    subprocess.run(
        ["scp", local, f"{m['ssh']}:{remote}"],
        check=True,
        capture_output=True,
        timeout=120,
    )


def scp_from(machine: str, remote: str, local: str) -> bool:
    """Pull a file from remote. Returns True on success, False if missing."""
    m = MACHINES[machine]
    r = subprocess.run(
        ["scp", f"{m['ssh']}:{remote}", local],
        capture_output=True,
        timeout=120,
    )
    return r.returncode == 0


def ensure_workdirs() -> None:
    """Create work directories on remote machines."""
    for name, m in MACHINES.items():
        ssh(name, f"mkdir -p {m['workdir']}")
    log("Remote work directories ready")


def deploy_worker(machine: str, worker_name: str) -> None:
    """SCP a standalone worker script to a machine."""
    m = MACHINES[machine]
    local_path = PROJECT_ROOT / "src" / "model_atlas" / worker_name
    scp_to(machine, str(local_path), f"{m['workdir']}/{worker_name}")


def start_worker(
    machine: str,
    worker_name: str,
    input_file: str,
    output_file: str,
    log_file: str,
    extra_args: str = "",
) -> None:
    m = MACHINES[machine]
    cmd = (
        f"cd {m['workdir']} && nohup {m['python']} {worker_name} "
        f"--input {input_file} --output {output_file} {extra_args} "
        f"> {log_file} 2>&1 &"
    )
    ssh(machine, cmd)
    log(f"  Started {worker_name} on {machine}")


def check_ollama(machine: str) -> bool:
    """Verify Ollama is responding on a machine."""
    r = ssh(
        machine,
        "curl -s -o /dev/null -w '%{http_code}' http://localhost:11434/v1/models",
    )
    return r.stdout.strip() == "200"


def poll_workers(targets: dict[str, tuple[str, int]]) -> None:
    """Poll until all workers finish.

    targets: {machine: (results_file_path, expected_count)}
    Waits at least 2 poll cycles before accepting worker exit as completion.
    """
    min_polls = 2
    polls_done = 0
    while True:
        polls_done += 1
        all_done = True
        for machine, (results_path, expected) in targets.items():
            r = ssh(machine, f"wc -l {results_path} 2>/dev/null || echo '0 none'")
            count = int(r.stdout.strip().split()[0])
            running = ssh(
                machine, "ps aux | grep 'phase_.*worker\\.py' | grep -v grep | wc -l"
            )
            procs = int(running.stdout.strip())
            pct = 100 * count / expected if expected > 0 else 0
            log(f"  {machine}: {count}/{expected} ({pct:.0f}%) | workers: {procs}")
            if procs > 0:
                all_done = False
            elif count < expected and polls_done <= min_polls:
                # Worker might still be starting up
                all_done = False
        if all_done:
            # Log warning if any machine finished with very few results
            for machine, (results_path, expected) in targets.items():
                r = ssh(machine, f"wc -l {results_path} 2>/dev/null || echo '0 none'")
                count = int(r.stdout.strip().split()[0])
                if count < expected * 0.1:
                    log(
                        f"  WARNING: {machine} only produced {count}/{expected} results — worker may have crashed"
                    )
            log("  All workers finished")
            return
        time.sleep(POLL_INTERVAL)


def run_local_python(code: str) -> str:
    """Run Python code locally via uv run, return stdout."""
    r = subprocess.run(
        ["uv", "run", "python", "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        log(f"  ERROR: {r.stderr.strip()}")
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


# --- Pipeline stages ---


def stage_c2(dry_run: bool = False) -> bool:
    """Export C2, deploy, run, pull, merge. Returns True if work was done."""
    log("=== Stage: C2 (vibe extraction) ===")

    # Check how many need C2
    out = run_local_python("""
from model_atlas import db
conn = db.get_connection(); db.init_db(conn)
total = conn.execute('SELECT COUNT(*) FROM models').fetchone()[0]
has = conn.execute("SELECT COUNT(*) FROM model_metadata WHERE key='qwen_summary'").fetchone()[0]
print(f"{total - has}")
conn.close()
""")
    remaining = int(out)
    if remaining == 0:
        log("  No models need C2, skipping")
        return False
    log(f"  {remaining} models need C2")
    if dry_run:
        return True

    # Export 2 shards
    out = run_local_python("""
from model_atlas import db
from model_atlas.ingest_phase_c import export_c2
conn = db.get_connection(); db.init_db(conn)
r = export_c2(conn, num_shards=2, min_likes=10)
print(r)
conn.close()
""")
    log(f"  Exported: {out}")
    exported = int(out)
    if exported == 0:
        log("  Nothing exported (min_likes filter?), skipping")
        return False
    per_shard = (exported + 1) // 2

    # Deploy workers and shards
    c2_work = LOCAL_CACHE / "phase_c_work"
    for name in MACHINES:
        deploy_worker(name, WORKERS["c2"])

    shard_map = {"macpro": "shard_0.jsonl", "homebridge": "shard_1.jsonl"}
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        scp_to(machine, str(c2_work / shard), f"{m['workdir']}/{shard}")
        start_worker(machine, WORKERS["c2"], shard, f"results_{shard}", "c2.log")

    # Poll
    targets = {}
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        targets[machine] = (f"{m['workdir']}/results_{shard}", per_shard)
    poll_workers(targets)

    # Pull and merge
    merge_files = []
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        local_path = c2_work / f"auto_results_{shard}"
        if scp_from(machine, f"{m['workdir']}/results_{shard}", str(local_path)):
            merge_files.append(str(local_path))
        else:
            log(f"  WARNING: Could not pull results from {machine}")

    if not merge_files:
        log("  ERROR: No results to merge")
        return False

    file_list = ", ".join(f"'{f}'" for f in merge_files)
    out = run_local_python(f"""
from model_atlas import db
from model_atlas.ingest_phase_c import merge_c2
conn = db.get_connection(); db.init_db(conn)
r = merge_c2(conn, [{file_list}])
print(r)
conn.close()
""")
    log(f"  C2 merge: {out}")
    return True


def stage_summary_selection() -> None:
    log("=== Stage: Summary Selection ===")
    out = run_local_python("""
from model_atlas import db
from model_atlas.ingest_phase_c import select_summaries
conn = db.get_connection(); db.init_db(conn)
r = select_summaries(conn)
print(r)
conn.close()
""")
    log(f"  {out}")


def stage_c3(dry_run: bool = False) -> bool:
    """Export C3, deploy, run, pull, merge."""
    log("=== Stage: C3 (quality gate) ===")

    out = run_local_python("""
from model_atlas import db
conn = db.get_connection(); db.init_db(conn)
has_vibe = set(r[0] for r in conn.execute("SELECT model_id FROM model_metadata WHERE key='vibe_summary'"))
has_quality = set(r[0] for r in conn.execute("SELECT model_id FROM model_metadata WHERE key='quality_score'"))
print(len(has_vibe - has_quality))
conn.close()
""")
    remaining = int(out)
    if remaining == 0:
        log("  No models need C3, skipping")
        return False
    log(f"  {remaining} models need C3")
    if dry_run:
        return True

    out = run_local_python("""
from model_atlas import db
from model_atlas.ingest_phase_c import export_c3
conn = db.get_connection(); db.init_db(conn)
r = export_c3(conn, num_shards=2)
print(r)
conn.close()
""")
    exported = int(out)
    log(f"  Exported {exported} for C3")
    if exported == 0:
        return False
    per_shard = (exported + 1) // 2

    c3_work = LOCAL_CACHE / "phase_c3_work"
    shard_map = {"macpro": "shard_0.jsonl", "homebridge": "shard_1.jsonl"}
    for name in MACHINES:
        deploy_worker(name, WORKERS["c3"])
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        scp_to(machine, str(c3_work / shard), f"{m['workdir']}/{shard}")
        start_worker(
            machine, WORKERS["c3"], shard, f"results_c3_{shard}", "c3.log", "--resume"
        )

    targets = {}
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        targets[machine] = (f"{m['workdir']}/results_c3_{shard}", per_shard)
    poll_workers(targets)

    merge_files = []
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        local_path = c3_work / f"auto_results_c3_{shard}"
        if scp_from(machine, f"{m['workdir']}/results_c3_{shard}", str(local_path)):
            merge_files.append(str(local_path))
        else:
            log(f"  WARNING: Could not pull results from {machine}")

    if not merge_files:
        log("  ERROR: No results to merge")
        return False

    file_list = ", ".join(f"'{f}'" for f in merge_files)
    out = run_local_python(f"""
from model_atlas import db
from model_atlas.ingest_phase_c import merge_c3
conn = db.get_connection(); db.init_db(conn)
r = merge_c3(conn, [{file_list}])
print(r)
conn.close()
""")
    log(f"  C3 merge: {out}")
    return True


def stage_d1() -> None:
    log("=== Stage: D1 (deterministic audit) ===")
    out = run_local_python("""
from model_atlas import db, db_ingest
from model_atlas.phase_d_audit import audit_c2
nconn = db.get_connection(); db.init_db(nconn)
iconn = db_ingest.get_connection(); db_ingest.init_db(iconn)
r = audit_c2(nconn, iconn)
print(f"{r.total_audited} audited, {r.total_mismatches} mismatches, types={r.per_type_counts}")
nconn.close(); iconn.close()
""")
    log(f"  {out}")


def stage_d2() -> None:
    log("=== Stage: D2 (dictionary expansion) ===")
    spec = PROJECT_ROOT / "data" / "expansions" / "domain_specialization.yaml"
    if not spec.exists():
        log("  No expansion spec found, skipping")
        return
    out = run_local_python(f"""
from model_atlas import db
from model_atlas.phase_d_expand import expand_dictionary
conn = db.get_connection(); db.init_db(conn)
r = expand_dictionary(conn, '{spec}', dry_run=False)
print(f"linked={{r.models_linked}}, per_label={{r.per_label}}")
conn.close()
""")
    log(f"  {out}")


def stage_d3(heal_budget: int = 200, dry_run: bool = False) -> bool:
    """Export D3, deploy, run, pull, merge."""
    log(f"=== Stage: D3 (healing, budget={heal_budget}) ===")

    out = run_local_python("""
from model_atlas import db
conn = db.get_connection(); db.init_db(conn)
r = conn.execute("SELECT COUNT(DISTINCT model_id) FROM model_metadata WHERE key='audit_score' AND CAST(value AS REAL) < 0.7").fetchone()[0]
healed = conn.execute("SELECT COUNT(*) FROM correction_events").fetchone()[0]
print(f"{r},{healed}")
conn.close()
""")
    needs_heal, already_healed = (int(x) for x in out.split(","))
    log(f"  {needs_heal} need healing, {already_healed} already healed")
    if needs_heal == 0:
        return False
    if dry_run:
        return True

    actual_budget = min(heal_budget, needs_heal)

    out = run_local_python(f"""
from model_atlas import db, db_ingest
from model_atlas.phase_d_heal import export_d3
nconn = db.get_connection(); db.init_db(nconn)
iconn = db_ingest.get_connection(); db_ingest.init_db(iconn)
r = export_d3(nconn, iconn, tier='local', budget={actual_budget}, num_shards=2, seed=42)
print(f"{{r.run_id}},{{r.total_exported}}")
nconn.close(); iconn.close()
""")
    run_id, exported = out.split(",")
    exported = int(exported)
    log(f"  Exported {exported} for healing (run_id={run_id})")
    if exported == 0:
        return False
    per_shard = (exported + 1) // 2

    d3_work = LOCAL_CACHE / "phase_d_work"
    shard_map = {
        "macpro": "d3_local_shard_0.jsonl",
        "homebridge": "d3_local_shard_1.jsonl",
    }
    for name in MACHINES:
        deploy_worker(name, WORKERS["d3"])
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        scp_to(machine, str(d3_work / shard), f"{m['workdir']}/{shard}")
        start_worker(machine, WORKERS["d3"], shard, f"results_d3_{shard}", "d3.log")

    targets = {}
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        targets[machine] = (f"{m['workdir']}/results_d3_{shard}", per_shard)
    poll_workers(targets)

    merge_files = []
    for machine, shard in shard_map.items():
        m = MACHINES[machine]
        local_path = d3_work / f"auto_results_d3_{shard}"
        if scp_from(machine, f"{m['workdir']}/results_d3_{shard}", str(local_path)):
            merge_files.append(str(local_path))
        else:
            log(f"  WARNING: Could not pull results from {machine}")

    if not merge_files:
        log("  ERROR: No results to merge")
        return False

    file_list = ", ".join(f"'{f}'" for f in merge_files)
    out = run_local_python(f"""
from model_atlas import db
from model_atlas.phase_d_heal import merge_d3
conn = db.get_connection(); db.init_db(conn)
r = merge_d3(conn, [{file_list}], run_id='{run_id}')
print(r)
conn.close()
""")
    log(f"  D3 merge: {out}")
    return True


def stage_d4() -> None:
    log("=== Stage: D4 (export DPO training data) ===")
    d4_dir = LOCAL_CACHE / "phase_d_training"
    d4_dir.mkdir(parents=True, exist_ok=True)
    out = run_local_python(f"""
from model_atlas import db
from model_atlas.phase_d_training import export_training_data
conn = db.get_connection(); db.init_db(conn)
r = export_training_data(conn, '{d4_dir}/training_data.jsonl')
print(f"{{r.total_examples}} examples (by_tier={{r.by_tier}})")
conn.close()
""")
    log(f"  {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelAtlas auto-pipeline")
    parser.add_argument(
        "--heal-budget",
        type=int,
        default=200,
        help="Max models per D3 healing pass (default 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without doing it",
    )
    parser.add_argument(
        "--skip-c2",
        action="store_true",
        help="Skip C2 stage (e.g. workers already running)",
    )
    parser.add_argument("--skip-c3", action="store_true", help="Skip C3 stage")
    parser.add_argument(
        "--from-stage",
        choices=["c2", "summary", "c3", "d1", "d2", "d3", "d4"],
        default="c2",
        help="Start from this stage (default: c2)",
    )
    args = parser.parse_args()

    stages = ["c2", "summary", "c3", "d1", "d2", "d3", "d4"]
    start_idx = stages.index(args.from_stage)

    log("ModelAtlas auto-pipeline starting")
    ensure_workdirs()

    did_c2 = did_c3 = did_d3 = False

    # C2 → Summary Selection → C3 → D1 → D2 → D3 → D4
    if start_idx <= 0 and not args.skip_c2:
        did_c2 = stage_c2(dry_run=args.dry_run)
    if start_idx <= 1:
        stage_summary_selection()
    if start_idx <= 2 and not args.skip_c3:
        did_c3 = stage_c3(dry_run=args.dry_run)
    if start_idx <= 3:
        stage_d1()
    if start_idx <= 4:
        stage_d2()
    if start_idx <= 5:
        did_d3 = stage_d3(heal_budget=args.heal_budget, dry_run=args.dry_run)
    if start_idx <= 6:
        stage_d4()

    log("=== Pipeline complete ===")
    log(f"  C2: {'ran' if did_c2 else 'skipped'}")
    log(f"  C3: {'ran' if did_c3 else 'skipped'}")
    log(f"  D3: {'ran' if did_d3 else 'skipped'}")


if __name__ == "__main__":
    main()
