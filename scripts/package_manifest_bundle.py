"""Create a tar.gz containing exactly the files listed by a data manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_files(manifest_path: Path, repo_root: Path) -> list[tuple[Path, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = []
    for record in manifest["files"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path: {record['path']}")
        source = repo_root.joinpath(*relative.parts).resolve()
        source.relative_to(repo_root.resolve())
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != record["bytes"]:
            raise ValueError(f"byte count mismatch: {relative}")
        if sha256_file(source) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        files.append((source, relative.as_posix()))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest_path = Path(args.manifest).resolve()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    files = validated_files(manifest_path, repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for source, relative in files:
            archive.add(source, arcname=relative, recursive=False)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"[package] files={len(files)} bytes={output.stat().st_size}")
    print(f"[package] sha256={digest}")
    print(f"[package] output={output}")


if __name__ == "__main__":
    main()
