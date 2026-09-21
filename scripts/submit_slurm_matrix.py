"""Preview or submit a Slurm job matrix without shell-string execution.

The matrix YAML contains a template path, common exported variables and one
mapping per job.  Dry-run is the default.  ``--submit`` is deliberately
required before ``sbatch`` is called.  Submitted job ids and exact exported
values are appended to a JSON registry for later auditing.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def load_matrix(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("matrix root must be a mapping")
    if not isinstance(data.get("jobs"), list) or not data["jobs"]:
        raise ValueError("matrix must contain a non-empty jobs list")
    data.setdefault("common", {})
    if not isinstance(data["common"], dict):
        raise ValueError("common must be a mapping")
    return data


def normalize_exports(values: dict) -> dict[str, str]:
    exports: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        value = str(raw_value)
        if not ENV_NAME.fullmatch(key):
            raise ValueError(f"invalid environment name: {key!r}")
        if any(char in value for char in (",", "\n", "\r", "\x00")):
            raise ValueError(
                f"unsafe value for {key}: commas/newlines are not allowed")
        exports[key] = value
    return exports


def build_command(template: Path, exports: dict[str, str],
                  dependency: str | None = None) -> list[str]:
    if not template.is_file():
        raise FileNotFoundError(template)
    export_text = "ALL," + ",".join(
        f"{key}={value}" for key, value in sorted(exports.items()))
    command = ["sbatch", "--parsable", f"--export={export_text}"]
    if dependency:
        if not re.fullmatch(r"(?:afterok|afterany|afternotok):[0-9:]+",
                            dependency):
            raise ValueError(f"invalid Slurm dependency: {dependency}")
        command.append(f"--dependency={dependency}")
    command.append(str(template))
    return command


def append_registry(path: Path, records: list[dict]) -> None:
    existing: list[dict] = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"registry is not a JSON list: {path}")
        existing = loaded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing + records, indent=2,
                               ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--registry", default="server_job_registry.json")
    ap.add_argument("--submit", action="store_true",
                    help="actually call sbatch; omission is a safe dry-run")
    ap.add_argument("--dependency", default=None,
                    help="optional afterok:123 or afterany:123 dependency")
    args = ap.parse_args()

    matrix_path = Path(args.matrix).resolve()
    matrix = load_matrix(matrix_path)
    template = Path(matrix["template"])
    if not template.is_absolute():
        template = (matrix_path.parent / template).resolve()
    common = normalize_exports(matrix["common"])
    records = []
    for index, job in enumerate(matrix["jobs"], start=1):
        if not isinstance(job, dict):
            raise ValueError(f"job {index} must be a mapping")
        exports = {**common, **normalize_exports(job)}
        if args.submit and any("CHANGE_ME" in value
                               for value in exports.values()):
            raise SystemExit(
                f"job {index} still contains CHANGE_ME; edit a server-local "
                "matrix copy before submission")
        command = build_command(template, exports, args.dependency)
        display = " ".join(command)
        if not args.submit:
            print(f"[dry-run {index}] {display}")
            continue
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(
                f"sbatch failed for job {index}: {completed.stderr.strip()}")
        job_id = completed.stdout.strip().split(";")[0]
        record = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "template": str(template),
            "matrix": str(matrix_path),
            "exports": exports,
            "dependency": args.dependency,
        }
        records.append(record)
        print(f"[submitted {index}] job_id={job_id}")
    if args.submit:
        append_registry(Path(args.registry), records)
        print(f"[registry] appended {len(records)} jobs -> {args.registry}")


if __name__ == "__main__":
    main()
