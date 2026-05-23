import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List


COLMAP_EXE = r"E:\3DGS\gaussian-splatting-main\3dgs_tools\colmap\bin\colmap.exe"


VIEW_RE = re.compile(r"Processing view\s+(\d+)\s*/\s*(\d+)\s+for\s+(.+)$")


def fmt(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def print_stage(title: str, idx: int, total: int) -> None:
    line = "=" * 78
    print(f"\n{line}\n[{idx}/{total}] {title}\n{line}")


def run_stream(cmd: List[str], cwd: Path) -> None:
    """Run a command and stream its output without modifying parameters."""
    print("\nCommand:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    p = subprocess.run(cmd, cwd=str(cwd))
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with return code {p.returncode}.")


def run_patchmatch_with_progress(cmd: List[str], cwd: Path) -> None:
    """
    Run patch_match_stereo and display progress by parsing COLMAP's own log line:
    'Processing view i / N for XXXX.jpg'.
    This does not change COLMAP computation; it only reads stdout.
    """
    print("\nCommand:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))

    start = time.time()
    last_i = None
    last_t = 0.0

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        print(line)

        m = VIEW_RE.search(line)
        if m:
            i = int(m.group(1))
            n = int(m.group(2))
            fname = m.group(3).strip()

            now = time.time()
            # throttle progress printing
            if i != last_i and (now - last_t) >= 0.2:
                elapsed = now - start
                frac = i / max(1, n)
                eta = (elapsed / frac - elapsed) if frac > 1e-9 else float("inf")
                print(f"[PatchMatch] {i}/{n}  ({frac*100:6.2f}%)  elapsed={fmt(elapsed)}  eta={fmt(eta)}  current={fname}")
                last_i = i
                last_t = now

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"patch_match_stereo failed with return code {rc}.")

    print(f"[PatchMatch] completed in {fmt(time.time() - start)}.")


def infer_sparse_model(sparse_dir: Path) -> Path:
    # mapper usually outputs sparse/0, sparse/1, ...
    models = sorted([p for p in sparse_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not models:
        raise FileNotFoundError(f"No sparse model directory found in: {sparse_dir}")
    return models[0]


def main():
    parser = argparse.ArgumentParser(
        description="COLMAP pipeline (default settings): images -> sparse -> dense -> fused.ply"
    )
    parser.add_argument("--images", required=True, help=r'Images directory, e.g. E:\...\images')
    parser.add_argument("--overwrite", action="store_true", help="Delete database/sparse/dense and rerun")
    args = parser.parse_args()

    images_dir = Path(args.images).resolve()
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    workspace = images_dir.parent
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    fused_ply = dense_dir / "fused.ply"

    # optional cleanup
    if args.overwrite:
        if database.exists():
            database.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        if dense_dir.exists():
            shutil.rmtree(dense_dir)

    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Total stages for high-level progress
    # SfM: 3 stages, Dense: 3 stages
    TOTAL = 6
    stage = 0

    t0 = time.time()

    # 1) feature_extractor (DEFAULT settings)
    stage += 1
    print_stage("SfM: feature extraction (default settings)", stage, TOTAL)
    run_stream([COLMAP_EXE, "feature_extractor",
                "--database_path", str(database),
                "--image_path", str(images_dir)], cwd=workspace)

    # 2) exhaustive_matcher (DEFAULT settings)
    stage += 1
    print_stage("SfM: exhaustive matching (default settings)", stage, TOTAL)
    run_stream([COLMAP_EXE, "exhaustive_matcher",
                "--database_path", str(database)], cwd=workspace)

    # 3) mapper (DEFAULT settings)
    stage += 1
    print_stage("SfM: sparse reconstruction (mapper, default settings)", stage, TOTAL)
    run_stream([COLMAP_EXE, "mapper",
                "--database_path", str(database),
                "--image_path", str(images_dir),
                "--output_path", str(sparse_dir)], cwd=workspace)

    sparse_model = infer_sparse_model(sparse_dir)

    # 4) image_undistorter (DEFAULT settings)
    stage += 1
    print_stage("MVS: image undistortion (default settings)", stage, TOTAL)
    run_stream([COLMAP_EXE, "image_undistorter",
                "--image_path", str(images_dir),
                "--input_path", str(sparse_model),
                "--output_path", str(dense_dir)], cwd=workspace)

    # 5) patch_match_stereo (DEFAULT settings, progress parsed from stdout)
    stage += 1
    print_stage("MVS: PatchMatch stereo (default settings, progress from COLMAP output)", stage, TOTAL)
    run_patchmatch_with_progress([COLMAP_EXE, "patch_match_stereo",
                                  "--workspace_path", str(dense_dir)], cwd=workspace)

    # 6) stereo_fusion (DEFAULT settings)
    stage += 1
    print_stage("MVS: stereo fusion -> fused point cloud (default settings)", stage, TOTAL)
    run_stream([COLMAP_EXE, "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused_ply)], cwd=workspace)

    if not fused_ply.exists():
        raise FileNotFoundError(f"Expected output not found: {fused_ply}")

    print("\nResult")
    print(f"  workspace : {workspace}")
    print(f"  sparse    : {sparse_model}")
    print(f"  dense ply : {fused_ply}")
    print(f"  runtime   : {fmt(time.time() - t0)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nError:", e)
        sys.exit(1)
