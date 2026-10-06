"""Safely unpack a manifest bundle without overwriting different files."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.package_manifest_bundle import sha256_file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()
    archive_path = Path(args.archive).resolve()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    repo_root = Path(args.repo_root).resolve()
    expected = {record["path"]: record for record in manifest["files"]}

    extracted = skipped = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()
                   if member.isfile()}
        unexpected = sorted(set(members) - set(expected))
        missing = sorted(set(expected) - set(members))
        if unexpected or missing:
            raise SystemExit(
                f"archive/manifest mismatch: unexpected={unexpected[:3]}, "
                f"missing={missing[:3]}")
        for relative_text, record in expected.items():
            relative = PurePosixPath(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"unsafe path: {relative_text}")
            destination = repo_root.joinpath(*relative.parts).resolve()
            destination.relative_to(repo_root)
            if destination.exists():
                if (destination.is_file()
                        and destination.stat().st_size == record["bytes"]
                        and sha256_file(destination) == record["sha256"]):
                    skipped += 1
                    continue
                raise SystemExit(
                    f"refusing to overwrite different file: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(members[relative_text])
            if source is None:
                raise SystemExit(f"cannot read archive member: {relative_text}")
            with source, destination.open("wb") as stream:
                shutil.copyfileobj(source, stream)
            if (destination.stat().st_size != record["bytes"]
                    or sha256_file(destination) != record["sha256"]):
                destination.unlink(missing_ok=True)
                raise SystemExit(f"extracted file failed hash: {relative_text}")
            extracted += 1
    print(f"[unpack] extracted={extracted} already_matching={skipped}")


if __name__ == "__main__":
    main()
