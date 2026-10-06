"""Query Slurm status for job ids stored by submit_slurm_matrix.py."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def job_ids_from_registry(path: Path) -> list[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("registry must be a JSON list")
    ids = []
    for record in records:
        job_id = str(record.get("job_id", ""))
        if not re.fullmatch(r"[0-9]+", job_id):
            raise ValueError(f"invalid job id in registry: {job_id!r}")
        ids.append(job_id)
    return sorted(set(ids), key=int)


def run(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(command)}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def parse_pipe_rows(text: str, columns: list[str]) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.strip().split("|")
        if len(values) != len(columns):
            continue
        rows.append(dict(zip(columns, values)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    ids = job_ids_from_registry(Path(args.registry))
    if not ids:
        raise SystemExit("registry contains no jobs")
    joined = ",".join(ids)
    queue = run(["squeue", "-h", "-j", joined, "-o", "%i|%T|%M|%R"])
    # Some clusters disable Slurm accounting storage; degrade to queue-only
    # reporting instead of failing, and record why accounting is absent.
    try:
        accounting = run([
            "sacct", "-n", "-P", "-j", joined,
            "--format=JobIDRaw,State,Elapsed,ExitCode,MaxRSS,AllocTRES",
        ])
        accounting_note = None
        accounting_rows = parse_pipe_rows(
            accounting, ["job_id", "state", "elapsed", "exit_code",
                         "max_rss", "allocated_tres"])
    except RuntimeError as error:
        accounting_note = str(error)
        accounting_rows = []
    report = {
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_ids": ids,
        "queue": parse_pipe_rows(
            queue, ["job_id", "state", "elapsed", "reason_or_node"]),
        "accounting": accounting_rows,
        "accounting_note": accounting_note,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
