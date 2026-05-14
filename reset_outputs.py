import argparse
import shutil
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# Generated outputs from the current pipeline
GENERATED_FILES = [
    SCRIPT_DIR / "labels_psp.csv",
    SCRIPT_DIR / "recipe_index.json",
]

GENERATED_DIRS = [
    SCRIPT_DIR / "cleaned_recipes",
    SCRIPT_DIR / "syn_designs",
    SCRIPT_DIR / "graph_dataset",
]

# Runtime technology mapping cache created by synthesis.py
RUNTIME_TECH_DIR = Path(tempfile.gettempdir()) / "abc_tech_lib"


def gather_targets():
    files = [path for path in GENERATED_FILES if path.exists()]

    generated_dirs = []
    for directory in GENERATED_DIRS:
        if directory.exists():
            generated_dirs.append(directory)

    runtime_items = []
    if RUNTIME_TECH_DIR.exists():
        runtime_items.append(RUNTIME_TECH_DIR)

    return files, generated_dirs, runtime_items


def print_plan(files, generated_dirs, runtime_items):
    print("Will remove the following generated artifacts:")

    if files:
        for file_path in files:
            print(f"- file: {file_path}")

    if generated_dirs:
        for item in generated_dirs:
            print(f"- dir:  {item}")

    if runtime_items:
        for item in runtime_items:
            print(f"- dir:  {item}  (runtime tech cache)")

    if not files and not generated_dirs and not runtime_items:
        print("- nothing to remove")


def remove_targets(files, generated_dirs, runtime_items):
    removed = 0

    for file_path in files:
        file_path.unlink(missing_ok=True)
        removed += 1

    for item in sorted(generated_dirs, key=lambda p: len(p.parts), reverse=True):
        shutil.rmtree(item, ignore_errors=True)
        removed += 1

    for item in runtime_items:
        shutil.rmtree(item, ignore_errors=True)
        removed += 1

    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove generated pipeline outputs to start fresh."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    args = parser.parse_args()

    files, generated_dirs, runtime_items = gather_targets()
    print_plan(files, generated_dirs, runtime_items)

    if args.dry_run:
        print("Dry run complete. No files were deleted.")
        return

    if not args.yes:
        response = input("Proceed with deletion? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            print("Aborted. No files were deleted.")
            return

    removed_count = remove_targets(files, generated_dirs, runtime_items)
    print(f"Done. Removed {removed_count} item(s).")
    print("Deleted output directories: cleaned_recipes, syn_designs, graph_dataset")


if __name__ == "__main__":
    main()
