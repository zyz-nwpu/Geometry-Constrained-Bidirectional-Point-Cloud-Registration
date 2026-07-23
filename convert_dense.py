import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


VIEW_PATTERN = re.compile(r"Processing view\s+(\d+)\s*/\s*(\d+)\s+for\s+(.+)$")


def format_duration(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def print_stage(title, index, total):
    separator = "=" * 78
    print(f"\n{separator}\n[{index}/{total}] {title}\n{separator}")


def command_text(command):
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def run_command(command, workspace):
    print("\nCommand:")
    print(command_text(command))
    process = subprocess.run(command, cwd=str(workspace))
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with return code {process.returncode}.")


def run_patch_match(command, workspace):
    print("\nCommand:")
    print(command_text(command))

    start_time = time.time()
    last_index = None
    last_update = 0.0

    process = subprocess.Popen(
        command,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    if process.stdout is None:
        raise RuntimeError("Unable to read COLMAP output.")

    for line in process.stdout:
        line = line.rstrip("\n")
        print(line)
        match = VIEW_PATTERN.search(line)
        if not match:
            continue

        index = int(match.group(1))
        total = int(match.group(2))
        current = match.group(3).strip()
        now = time.time()

        if index != last_index and now - last_update >= 0.2:
            elapsed = now - start_time
            fraction = index / max(1, total)
            remaining = elapsed / fraction - elapsed if fraction > 1e-9 else 0.0
            print(
                f"[PatchMatch] {index}/{total} "
                f"({fraction * 100:6.2f}%) "
                f"elapsed={format_duration(elapsed)} "
                f"eta={format_duration(remaining)} "
                f"current={current}"
            )
            last_index = index
            last_update = now

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"patch_match_stereo failed with return code {return_code}.")

    print(f"[PatchMatch] completed in {format_duration(time.time() - start_time)}.")


def infer_sparse_model(sparse_dir):
    models = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    if not models:
        raise FileNotFoundError(f"No sparse model directory found in {sparse_dir}")
    return models[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run COLMAP sparse and dense reconstruction.")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument(
        "--colmap",
        default=os.environ.get("COLMAP_EXE") or shutil.which("colmap"),
        help="COLMAP executable. Defaults to COLMAP_EXE or the colmap command on PATH.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_colmap_executable(value):
    if not value:
        raise FileNotFoundError(
            "COLMAP was not found. Add it to PATH, set COLMAP_EXE, or pass --colmap."
        )

    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    executable = shutil.which(str(value))
    if executable:
        return executable

    raise FileNotFoundError(f"COLMAP executable not found: {value}")


def remove_previous_outputs(database, sparse_dir, dense_dir):
    if database.exists():
        database.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    if dense_dir.exists():
        shutil.rmtree(dense_dir)


def main():
    args = parse_args()
    images_dir = args.images.resolve()
    colmap_exe = resolve_colmap_executable(args.colmap)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    workspace = images_dir.parent
    database = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    dense_dir = workspace / "dense"
    fused_ply = dense_dir / "fused.ply"

    if args.overwrite:
        remove_previous_outputs(database, sparse_dir, dense_dir)

    sparse_dir.mkdir(parents=True, exist_ok=True)

    stages = [
        (
            "SfM feature extraction",
            [colmap_exe, "feature_extractor", "--database_path", str(database), "--image_path", str(images_dir)],
            run_command,
        ),
        (
            "SfM exhaustive matching",
            [colmap_exe, "exhaustive_matcher", "--database_path", str(database)],
            run_command,
        ),
        (
            "SfM sparse reconstruction",
            [
                colmap_exe,
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_dir),
            ],
            run_command,
        ),
    ]

    start_time = time.time()
    total_stages = 6

    for index, (title, command, runner) in enumerate(stages, start=1):
        print_stage(title, index, total_stages)
        runner(command, workspace)

    sparse_model = infer_sparse_model(sparse_dir)

    dense_stages = [
        (
            "MVS image undistortion",
            [
                colmap_exe,
                "image_undistorter",
                "--image_path",
                str(images_dir),
                "--input_path",
                str(sparse_model),
                "--output_path",
                str(dense_dir),
            ],
            run_command,
        ),
        (
            "MVS PatchMatch stereo",
            [colmap_exe, "patch_match_stereo", "--workspace_path", str(dense_dir)],
            run_patch_match,
        ),
        (
            "MVS stereo fusion",
            [colmap_exe, "stereo_fusion", "--workspace_path", str(dense_dir), "--output_path", str(fused_ply)],
            run_command,
        ),
    ]

    for offset, (title, command, runner) in enumerate(dense_stages, start=4):
        print_stage(title, offset, total_stages)
        runner(command, workspace)

    if not fused_ply.exists():
        raise FileNotFoundError(f"Expected output not found: {fused_ply}")

    print("\nResult")
    print(f"workspace : {workspace}")
    print(f"sparse    : {sparse_model}")
    print(f"dense ply : {fused_ply}")
    print(f"runtime   : {format_duration(time.time() - start_time)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nError: {error}")
        sys.exit(1)
