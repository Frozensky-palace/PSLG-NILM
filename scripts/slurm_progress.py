"""Report per-job progress for a submitted Slurm matrix (guide §6.5).

Reads the JSON registry written by submit_slurm_matrix.py, maps each record
to its artifact run directory and classifies it:

- DONE      run dir contains the done-file (default validation_metrics.json)
- RUNNING   job id is in squeue with state RUNNING/COMPLETING
- PENDING   job id is in squeue with state PENDING
- FAILED    not queued and no done-file (a restart usually follows)
- NO RUN DIR  queued but the artifact dir does not exist yet

Prints a human table plus a final ``ALL N JOBS DONE`` marker; optionally
writes the report as JSON. Replaces the ad-hoc ~/c3_status.sh.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DONE_FILENAME = "validation_metrics.json"
DEFAULT_RUN_GLOB = "s2p_{arm}_r{ratio}_s{seed}_{job_id}"


def load_registry(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("registry must be a non-empty JSON list")
    for record in records:
        if "job_id" not in record:
            raise ValueError(f"registry record missing job_id: {record!r}")
    return records


def queue_states(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    joined = ",".join(job_ids)
    completed = subprocess.run(
        ["squeue", "-h", "-j", joined, "-o", "%i|%T"],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"squeue failed: {completed.stderr.strip()}")
    states: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "|" not in line:
            continue
        job_id, state = line.strip().split("|", 1)
        states[job_id] = state.upper()
    return states


def exports_of(record: dict) -> dict[str, str]:
    return {key: str(value)
            for key, value in record.get("exports", {}).items()}


def find_run_dir(artifacts_root: Path, run_glob: str,
                 mapping: dict[str, str]) -> Path | None:
    """Newest matching dir, preferring one that already has the done-file.

    Arm case is ambiguous (sbatch used upper-case PSLG_ARM values in dir
    names while reports lowercase): try every case variant of the arm and
    merge candidates.
    """
    arm = mapping.get("arm", "")
    candidates: list[Path] = []
    for arm_variant in dict.fromkeys(
            [arm, arm.upper(), arm.lower()] if arm else [arm]):
        if not arm_variant and "{arm}" in run_glob:
            continue
        pattern = run_glob.format(**{**mapping, "arm": arm_variant})
        candidates.extend(artifacts_root.glob(pattern))
    unique = sorted(set(candidates), key=lambda p: (p.stat().st_mtime, p))
    if not unique:
        return None
    for candidate in unique:
        if (candidate / DONE_FILENAME).exists():
            return candidate
    return unique[-1]


def classify(records: list[dict], states: dict[str, str],
             artifacts_root: Path, run_glob: str) -> list[dict]:
    report = []
    for record in records:
        job_id = str(record["job_id"])
        exports = exports_of(record)
        state = states.get(job_id)
        if state in ("RUNNING", "COMPLETING"):
            status = "RUNNING"
        elif state in ("PENDING", "CONFIGURING"):
            status = "PENDING"
        else:
            status = None
        run_dir = find_run_dir(artifacts_root, run_glob, {
            "arm": exports.get("PSLG_ARM", "").lower(),
            "ratio": exports.get("PSLG_RATIO", ""),
            "seed": exports.get("PSLG_SEED", ""),
            "job_id": job_id,
        })
        if run_dir is not None and (run_dir / DONE_FILENAME).exists():
            label = "DONE"
        elif status is not None:
            label = status
        elif run_dir is None:
            label = "NO RUN DIR"
        else:
            label = "FAILED"
        report.append({
            "job_id": job_id,
            "arm": exports.get("PSLG_ARM", "?"),
            "seed": exports.get("PSLG_SEED", "?"),
            "queue_state": state or "-",
            "status": label,
            "run_dir": str(run_dir) if run_dir else "",
        })
    return report


def render(report: list[dict]) -> str:
    lines = ["job_id    arm  seed  queue      status     run_dir"]
    for row in report:
        lines.append(
            f"{row['job_id']:<9} {row['arm']:<4} {row['seed']:<5} "
            f"{row['queue_state']:<10} {row['status']:<10} "
            f"{Path(row['run_dir']).name if row['run_dir'] else '-'}")
    done = sum(1 for row in report if row["status"] == "DONE")
    lines.append(f"progress: {done}/{len(report)} DONE")
    if done == len(report):
        lines.append(f"ALL {len(report)} JOBS DONE")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--artifacts-root", required=True)
    ap.add_argument("--run-glob", default=DEFAULT_RUN_GLOB,
                    help="run dir name pattern; placeholders {arm} {ratio} "
                         "{seed} {job_id}; arms are lowercased")
    ap.add_argument("--output", default=None, help="optional JSON report")
    args = ap.parse_args()

    records = load_registry(Path(args.registry))
    states = queue_states([str(r["job_id"]) for r in records])
    report = classify(records, states, Path(args.artifacts_root),
                      args.run_glob)
    rendered = render(report)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False)
                          + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
