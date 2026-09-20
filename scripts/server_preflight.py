"""Server-side preflight gate before any formal C1/C2/C3 job (guide §5.2).

Checks, in order, that: the repo root exists with the expected frozen commit
and a clean tree; every data manifest verifies file-by-file; staged manifests
contain zero test paths; the output root is writable; and the required
framework imports. Any hard failure exits non-zero so Slurm jobs stop before
wasting compute.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_git(args: list[str], root: Path) -> str:
    completed = subprocess.run(["git", *args], cwd=str(root),
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: "
                           f"{completed.stderr.strip()}")
    return completed.stdout.strip()


def check_commit(root: Path, expected: str) -> str:
    actual = run_git(["rev-parse", "HEAD"], root)
    if expected and actual != expected:
        raise RuntimeError(
            f"commit mismatch: server {actual} != frozen {expected}")
    status = run_git(["status", "--short"], root)
    if status:
        raise RuntimeError(f"working tree is not clean:\n{status}")
    return actual


def check_manifest(manifest_path: Path, repo_root: Path) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.verify_server_manifest import sha256_of_file

    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    failures = []
    for record in manifest["files"]:
        path = repo_root / record["path"]
        if not path.exists() or path.stat().st_size != record["bytes"]:
            failures.append(record["path"])
            continue
        if sha256_of_file(path) != record["sha256"]:
            failures.append(record["path"])
    if failures:
        raise RuntimeError(
            f"{manifest_path}: {len(failures)}/{manifest['file_count']} files "
            f"failed verification, first: {failures[0]}")
    test_paths = [record["path"] for record in manifest["files"]
                  if "test" in Path(record["path"]).parts]
    if test_paths:
        raise RuntimeError(f"{manifest_path} contains test paths: "
                           f"{test_paths[:3]}")
    return {"manifest": str(manifest_path),
            "files": manifest["file_count"],
            "test_path_count": len(test_paths)}


def check_writable(output_root: Path) -> None:
    probe = output_root / ".preflight_write_probe"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()


def check_imports(framework: str) -> None:
    import importlib

    for module in ("yaml", "numpy", "pandas", "sklearn"):
        importlib.import_module(module)
    if framework in ("tensorflow", "both"):
        importlib.import_module("tensorflow")
    if framework in ("torch", "both"):
        importlib.import_module("torch")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--expected-commit", default="",
                    help="frozen commit the server must be standing on")
    ap.add_argument("--manifest", action="append", default=[],
                    help="data manifest to verify; repeatable")
    ap.add_argument("--output-root", required=True,
                    help="directory this job will write into")
    ap.add_argument("--framework", choices=("tensorflow", "torch", "both"),
                    default="both")
    ap.add_argument("--skip-imports", action="store_true",
                    help="only check repo/manifests/output (no framework)")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()

    report = {"repo_root": str(root), "checks": []}
    try:
        commit = check_commit(root, args.expected_commit)
        report["checks"].append({"name": "commit", "ok": True,
                                 "commit": commit})
        for manifest in args.manifest:
            info = check_manifest(Path(manifest), root)
            report["checks"].append({"name": "manifest", "ok": True, **info})
        check_writable(Path(args.output_root))
        report["checks"].append({"name": "writable_output", "ok": True})
        if not args.skip_imports:
            check_imports(args.framework)
            report["checks"].append({"name": "imports", "ok": True,
                                     "framework": args.framework})
    except Exception as error:  # noqa: BLE001 - gate must report, then fail
        report["checks"].append({"name": "failed", "ok": False,
                                 "error": str(error)})
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[preflight] FAILED: {error}", file=sys.stderr)
        sys.exit(1)
    report["passed"] = True
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("[preflight] PASSED")


if __name__ == "__main__":
    main()
