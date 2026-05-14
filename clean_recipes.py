from pathlib import Path


# Edit this set if you want to allow more synthesis commands.
ALLOWED_COMMANDS = {
    "balance",
    "b",
    "rewrite",
    "rw",
    "refactor",
    "rf",
    "resub",
    "rs",
    "st",
    "dc2",
    "dch",
    "if",
    "ifraig",
    "fraig",
    "resyn",
    "resyn2",
    "compress2rs",
    "choice",
}

INPUT_FOLDER = "recipes"
OUTPUT_FOLDER = "cleaned_recipes"


def strip_comment(line: str) -> str:
    comment_pos = line.find("#")
    if comment_pos >= 0:
        return line[:comment_pos]
    return line


def is_synthesis_command(line: str) -> bool:
    tokens = line.strip().split()
    if not tokens:
        return False
    return tokens[0].lower() in ALLOWED_COMMANDS


def clean_recipe_file(file_path: Path, output_path: Path) -> tuple[int, int]:
    original_lines = file_path.read_text(encoding="utf-8").splitlines()

    kept_lines = []
    removed_count = 0

    for raw_line in original_lines:
        candidate = strip_comment(raw_line).strip()
        if not candidate:
            continue

        if is_synthesis_command(candidate):
            kept_lines.append(" ".join(candidate.split()))
        else:
            removed_count += 1

    new_text = "\n".join(kept_lines)
    if new_text:
        new_text += "\n"
    output_path.write_text(new_text, encoding="utf-8")

    return len(kept_lines), removed_count


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    recipe_dir = script_dir / INPUT_FOLDER
    cleaned_dir = script_dir / OUTPUT_FOLDER
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    recipe_files = sorted(recipe_dir.glob("*.script"))

    if not recipe_files:
        print(f"No .script files found in {recipe_dir}")
        return

    print(f"Cleaning {len(recipe_files)} recipe file(s) in {recipe_dir}...")
    for recipe in recipe_files:
        cleaned_recipe = cleaned_dir / recipe.name
        kept, removed = clean_recipe_file(recipe, cleaned_recipe)
        print(f"- {recipe.name}: kept {kept}, removed {removed} -> {cleaned_recipe}")

    print("Done.")


if __name__ == "__main__":
    main()
