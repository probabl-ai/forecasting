#!/usr/bin/env python

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


KERNEL_NAME = "pixi-doc-python"
KERNEL_DISPLAY_NAME = "Python 3 (pixi doc)"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Filename to exclude. Can be passed multiple times.",
    )
    return parser


def iter_scripts(source_dir, excluded_names):
    for script_path in sorted(source_dir.glob("*.py")):
        if script_path.name in excluded_names:
            continue
        if script_path.stem.endswith("_lib"):
            continue
        yield script_path


def ensure_kernel_registered():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--prefix",
            sys.prefix,
            "--name",
            KERNEL_NAME,
            "--display-name",
            KERNEL_DISPLAY_NAME,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_jupytext(script_path, execute):
    command = [sys.executable, "-m", "jupytext", "--to", "notebook"]
    if execute:
        command.extend(["--set-kernel", KERNEL_NAME])
        command.append("--execute")
    command.append(str(script_path))
    subprocess.run(command, check=True)


def move_notebooks(source_dir, output_dir):
    for notebook_path in sorted(source_dir.glob("*.ipynb")):
        shutil.move(str(notebook_path), output_dir / notebook_path.name)


def main():
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / "content" / "python_files"
    output_dir = repo / "content" / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts = list(iter_scripts(source_dir, set(args.exclude)))
    if args.dry_run:
        for script_path in scripts:
            print(script_path.relative_to(repo))
        return

    if args.execute:
        ensure_kernel_registered()

    for script_path in scripts:
        run_jupytext(script_path, execute=args.execute)

    move_notebooks(source_dir, output_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)