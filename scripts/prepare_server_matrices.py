"""Create server-local C1/C2/C3 matrices with real paths and commit.

The tracked YAML files remain shareable templates. This command writes filled
copies outside the Git worktree, so users never need to edit tracked files or
replace CHANGE_ME by hand.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def git_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root,
        capture_output=True, text=True, check=True).stdout.strip()


def load_template(project_root: Path, name: str) -> dict:
    path = project_root / "config" / "server" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    print(f"[matrix] wrote {path}")


def materialize(project_root: Path, output_dir: Path,
                manifest_root: Path, artifact_root: Path,
                nilm_experiment_dir: Path, ratio: str = "0p5") -> list[Path]:
    commit = git_commit(project_root)
    shared = {"PSLG_PROJECT_ROOT": str(project_root)}

    c1 = load_template(project_root, "c1_gpu_smoke_matrix.yaml")
    c1["template"] = str(project_root / "slurm" / "c1_gpu_smoke.sbatch")
    c1["common"].update({**shared,
                         "PSLG_ARTIFACT_ROOT": str(artifact_root)})

    c2 = load_template(project_root, "c2_detsec_matrix.yaml")
    c2["template"] = str(project_root / "slurm" / "c2_detsec_pc.sbatch")
    c2["common"].update({
        **shared,
        "PSLG_FROZEN_COMMIT": commit,
        "PSLG_MANIFEST": str(
            manifest_root / "c1_state_discovery_trainonly_manifest.json"),
        "PSLG_RUN_ROOT": str(artifact_root),
    })

    c3 = load_template(project_root, "c3_seq2point_matrix.yaml")
    c3["template"] = str(project_root / "slurm" / "c3_seq2point.sbatch")
    c3["common"].update({
        **shared,
        "PSLG_FROZEN_COMMIT": commit,
        "PSLG_MANIFEST": str(
            manifest_root / "c1_nilm_b0_b2_trainval_manifest.json"),
        "PSLG_EXPERIMENT_DIR": str(nilm_experiment_dir),
        "PSLG_RUN_ROOT": str(artifact_root),
        "PSLG_RATIO": ratio,
    })

    outputs = []
    for name, data in (("c1_gpu_smoke.server.yaml", c1),
                       ("c2_detsec.server.yaml", c2),
                       (f"c3_seq2point_r{ratio}.server.yaml", c3)):
        path = output_dir / name
        write_yaml(path, data)
        outputs.append(path)
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--manifest-root", required=True)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--nilm-experiment-dir", required=True)
    ap.add_argument("--ratio", default="0p5",
                    help="run-id label such as 0p5, 1p0 or 2p0")
    args = ap.parse_args()
    materialize(
        Path(args.project_root).resolve(), Path(args.output_dir).resolve(),
        Path(args.manifest_root).resolve(), Path(args.artifact_root).resolve(),
        Path(args.nilm_experiment_dir).resolve(), args.ratio)


if __name__ == "__main__":
    main()
