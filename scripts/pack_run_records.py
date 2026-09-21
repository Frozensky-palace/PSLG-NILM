"""Whitelist-pack run records into a transfer tarball (guide §7.15).

Generalizes the ad-hoc ~/pack_c1.sh: explicit source lists only, so stray
files (kernel images, caches) can never sneak into a records archive.

Examples:
  python scripts/pack_run_records.py --output out/c1_records.tar.gz \\
    --file "$REC/*.json":smoke --file "$REC/*.txt":smoke \\
    --dir "$ART/c1_detsec_smoke":detsec \\
    --file ~/pslg_manifests/c1_state_verify.json:manifests

Every --file argument is GLOB:DEST_DIR (copy each match flat into the
destination dir inside the archive); every --dir argument is SRC:DEST_NAME
(copied recursively). Writes "<output>.sha256" with the archive digest and
refuses to overwrite an existing archive unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_files(staging: Path, spec: str) -> int:
    glob_part, _, dest_name = spec.rpartition(":")
    if not glob_part or not dest_name:
        raise SystemExit(f"--file must be GLOB:DEST_DIR, got: {spec}")
    dest = staging / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    pattern = Path(glob_part)
    matches = sorted(pattern.parent.glob(pattern.name))
    if not matches:
        raise SystemExit(f"--file pattern matched nothing: {glob_part}")
    copied = 0
    for match in matches:
        if match.is_file():
            shutil.copy2(match, dest / match.name)
            copied += 1
    return copied


def stage_dir(staging: Path, spec: str) -> int:
    src_part, _, dest_name = spec.rpartition(":")
    if not src_part or not dest_name:
        raise SystemExit(f"--dir must be SRC:DEST_NAME, got: {spec}")
    source = Path(src_part)
    if not source.is_dir():
        raise SystemExit(f"--dir source is not a directory: {source}")
    shutil.copytree(source, staging / dest_name)
    return sum(1 for _ in (staging / dest_name).rglob("*") if _.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="tar.gz path to create")
    ap.add_argument("--file", action="append", default=[],
                    help="GLOB:DEST_DIR; repeatable")
    ap.add_argument("--dir", action="append", default=[],
                    help="SRC_DIR:DEST_NAME; repeatable, copied recursively")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing archive")
    args = ap.parse_args()

    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing archive: {output} "
                         "(pass --force)")
    if not args.file and not args.dir:
        raise SystemExit("nothing to pack: give at least one --file/--dir")

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary) / "stage"
        staging.mkdir()
        staged = 0
        for spec in args.file:
            staged += stage_files(staging, spec)
        for spec in args.dir:
            staged += stage_dir(staging, spec)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            for member in sorted(staging.rglob("*")):
                if member.is_file():
                    archive.add(member, arcname=member.relative_to(staging)
                                .as_posix())
    digest = sha256_file(output)
    Path(f"{output}.sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"[pack] staged files={staged} archive={output}")
    print(f"[pack] sha256={digest}")


if __name__ == "__main__":
    main()
