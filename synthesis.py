import os
import csv
import json
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

# --- CONFIGURATION ---
# Use WSL to run the Linux ABC binary
USE_WSL = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Directory containing your original designs
INPUT_DIR = str(SCRIPT_DIR / "orig_designs")

# Directory containing your cleaned synthesis recipes (.script files)
RECIPE_DIR = str(SCRIPT_DIR / "cleaned_recipes")

# Directory where the optimized .bench files will be saved
OUTPUT_DIR = str(SCRIPT_DIR / "syn_designs")

# Label/metadata artifacts
LABEL_CSV_PATH = str(SCRIPT_DIR / "labels_psp.csv")
RECIPE_INDEX_PATH = str(SCRIPT_DIR / "recipe_index.json")

# Technology mapping configuration
# The entire dataset uses one fixed technology library.
TECH_LIB_SOURCE_PATH = SCRIPT_DIR.parent / "tech_lib" / "cadence.genlib"
TECH_LIB_NAME = TECH_LIB_SOURCE_PATH.name
TECH_RUNTIME_DIR = Path(tempfile.gettempdir()) / "abc_tech_lib"
TECH_LIB_PATH = TECH_RUNTIME_DIR / TECH_LIB_NAME
TECH_SUPER_PATH = TECH_LIB_PATH.with_suffix(".super")
TECH_MAP_COMMAND = "map -B 0.9"
TECH_SUPER_BUILD_COMMAND = "super -I 5 -L 2"

# If True, delete optimized BENCH files after extracting power labels.
DELETE_OPTIMIZED_OUTPUTS = False

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


POWER_PATTERN = re.compile(r"power\s*=\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

def to_wsl_path(windows_path, quote=True):
    """Convert a Windows path to a WSL path."""
    # Convert absolute paths like C:\path to /mnt/c/path
    # Convert relative paths to absolute first, then convert
    abs_path = os.path.abspath(windows_path)
    wsl_path = abs_path.replace("\\", "/").replace("C:", "/mnt/c").replace("D:", "/mnt/d").replace("E:", "/mnt/e")
    if quote:
        # Wrap in double quotes to handle spaces in file paths passed to ABC
        return f'"{wsl_path}"'
    return wsl_path


def write_recipe_index(recipes):
    recipe_rows = []
    for recipe_filename in recipes:
        recipe_id = Path(recipe_filename).stem
        recipe_path = Path(RECIPE_DIR) / recipe_filename
        commands = []
        for raw_line in recipe_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line:
                commands.append(line)

        recipe_rows.append(
            {
                "recipe_id": recipe_id,
                "recipe_file": str(recipe_path),
                "num_commands": len(commands),
                "commands": commands,
            }
        )

    Path(RECIPE_INDEX_PATH).write_text(json.dumps(recipe_rows, indent=2), encoding="utf-8")


def load_existing_labels() -> dict[str, dict]:
    existing = {}
    csv_path = Path(LABEL_CSV_PATH)
    if not csv_path.exists():
        return existing

    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            output_name = row.get("output_name", "").strip()
            if not output_name:
                continue
            try:
                power_switch = float(row.get("power_switch", ""))
            except (TypeError, ValueError):
                power_switch = None

            existing[output_name] = {
                "design_id": row.get("design_id", ""),
                "recipe_id": row.get("recipe_id", ""),
                "output_name": output_name,
                "power_switch": power_switch,
                "tech_lib": row.get("tech_lib", TECH_LIB_NAME),
                "tech_lib_path": row.get("tech_lib_path", str(TECH_LIB_SOURCE_PATH)),
                "tech_runtime_path": row.get("tech_runtime_path", str(TECH_LIB_PATH)),
                "tech_super_path": row.get("tech_super_path", str(TECH_SUPER_PATH)),
                "tech_map": row.get("tech_map", TECH_MAP_COMMAND),
                "tech_lib_command": row.get("tech_lib_command", ""),
            }

    return existing


def extract_power_switch(stats_stdout):
    match = POWER_PATTERN.search(stats_stdout)
    if not match:
        return None
    return float(match.group(1))


def ensure_tech_super_library():
    if not TECH_LIB_SOURCE_PATH.exists():
        raise FileNotFoundError(f"Technology library not found: {TECH_LIB_SOURCE_PATH}")

    TECH_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if (not TECH_LIB_PATH.exists()) or TECH_LIB_PATH.stat().st_mtime < TECH_LIB_SOURCE_PATH.stat().st_mtime:
        shutil.copy2(TECH_LIB_SOURCE_PATH, TECH_LIB_PATH)

    if TECH_SUPER_PATH.exists() and TECH_SUPER_PATH.stat().st_mtime >= TECH_LIB_PATH.stat().st_mtime:
        return

    if USE_WSL:
        abc_path = to_wsl_path(str(REPO_ROOT / "abc"), quote=False)
        tech_lib_wsl = to_wsl_path(str(TECH_LIB_PATH), quote=False)
        command = f'{TECH_SUPER_BUILD_COMMAND} {tech_lib_wsl}'
        subprocess.run(["wsl", abc_path, "-c", command], capture_output=True, text=True, check=True)
    else:
        abc_path = str(REPO_ROOT / "abc")
        command = f'{TECH_SUPER_BUILD_COMMAND} {os.path.abspath(TECH_LIB_PATH)}'
        subprocess.run([abc_path, "-c", command], capture_output=True, text=True, check=True)


def collect_power_stats(output_file):
    ensure_tech_super_library()

    tech_lib_command = f'read_library {to_wsl_path(str(TECH_LIB_PATH), quote=False) if USE_WSL else os.path.abspath(TECH_LIB_PATH)}'
    tech_super_command = f'read_super {to_wsl_path(str(TECH_SUPER_PATH), quote=False) if USE_WSL else os.path.abspath(TECH_SUPER_PATH)}'

    def build_stats_commands(output_path):
        return f"read_bench {output_path}; strash; {tech_lib_command}; {tech_super_command}; {TECH_MAP_COMMAND}; print_stats -p"

    if USE_WSL:
        abc_path = to_wsl_path(str(REPO_ROOT / "abc"), quote=False)
        output_file_wsl = to_wsl_path(output_file)
        stats_commands = build_stats_commands(output_file_wsl)
        process = subprocess.run(["wsl", abc_path, "-c", stats_commands], capture_output=True, text=True, check=True)
    else:
        abc_path = str(REPO_ROOT / "abc")
        output_file_abs = f'"{os.path.abspath(output_file)}"'
        stats_commands = build_stats_commands(output_file_abs)
        process = subprocess.run([abc_path, "-c", stats_commands], capture_output=True, text=True, check=True)

    combined_output = f"{process.stdout}\n{process.stderr}".strip()
    return combined_output, tech_lib_command

def optimize_design(design_file, recipe_file, existing_labels_by_output):
    """
    Runs ABC to load a design, apply a recipe, and save the optimized AIG.
    """
    design_name = os.path.splitext(os.path.basename(design_file))[0]
    recipe_name = os.path.splitext(os.path.basename(recipe_file))[0]
    
    # Define the output filename (e.g., multiplier_abc0_opt.bench)
    output_file = os.path.join(OUTPUT_DIR, f"{design_name}_{recipe_name}_opt.bench")
    
    output_stem = Path(output_file).stem

    if os.path.exists(output_file):
        existing = existing_labels_by_output.get(output_stem)
        if existing is not None and existing.get("power_switch") is not None:
            print(f"Skipping {output_stem}: optimized output and label already exist")
            return existing

        try:
            stats_stdout, tech_lib_command = collect_power_stats(output_file)
            power_switch = extract_power_switch(stats_stdout)
            if power_switch is None:
                print(f"  Warning: power not found for existing output {output_file}")
                return None

            row = {
                "design_id": design_name,
                "recipe_id": recipe_name,
                "output_name": output_stem,
                "power_switch": power_switch,
                "tech_lib": TECH_LIB_NAME,
                "tech_lib_path": str(TECH_LIB_SOURCE_PATH),
                "tech_runtime_path": str(TECH_LIB_PATH),
                "tech_super_path": str(TECH_SUPER_PATH),
                "tech_map": TECH_MAP_COMMAND,
                "tech_lib_command": tech_lib_command,
            }
            return row
        except subprocess.CalledProcessError as e:
            print(f"  Error extracting power for existing output {output_stem}:")
            print(e.stdout)
            print(e.stderr)
            return None

    # Convert paths for WSL if needed
    if USE_WSL:
        design_file_wsl = to_wsl_path(design_file)
        recipe_file_wsl = to_wsl_path(recipe_file)
        output_file_wsl = to_wsl_path(output_file)
        abc_path_wsl = to_wsl_path(str(REPO_ROOT / "abc"), quote=False)
    else:
        design_file_wsl = design_file
        recipe_file_wsl = recipe_file
        output_file_wsl = output_file
        abc_path_wsl = str(REPO_ROOT / "abc")
    
    # Construct the ABC command string:
    # 1. read_bench: Load the input
    # 2. strash: Convert to AIG (standardize the graph)
    # 3. source: Run the synthesis recipe
    # 4. write_bench: Save the optimized AIG back to a file
    abc_commands = (
        f"read_bench {design_file_wsl}; "
        f"strash; "
        f"source {recipe_file_wsl}; "
        f"write_bench {output_file_wsl}"
    )
    
    # Execute ABC
    print(f"Applying {recipe_name} to {design_name}...")
    try:
        if USE_WSL:
            # Run through WSL
            process = subprocess.run(
                ["wsl", abc_path_wsl, "-c", abc_commands],
                capture_output=True, 
                text=True,
                check=True
            )
        else:
            # Run directly
            process = subprocess.run(
                [abc_path_wsl, "-c", abc_commands],
                capture_output=True, 
                text=True,
                check=True
            )

        if not os.path.exists(output_file):
            print(f"  Error: synthesis did not create output file: {output_file}")
            if process.stdout:
                print(process.stdout)
            if process.stderr:
                print(process.stderr)
            return None

        stats_stdout, tech_lib_command = collect_power_stats(output_file)
        power_switch = extract_power_switch(stats_stdout)
        if power_switch is None:
            print(f"  Warning: power not found for {output_file}")
            return None

        print(f"  Successfully saved: {output_file}  |  power_switch={power_switch}")

        row = {
            "design_id": design_name,
            "recipe_id": recipe_name,
            "output_name": Path(output_file).stem,
            "power_switch": power_switch,
            "tech_lib": TECH_LIB_NAME,
            "tech_lib_path": str(TECH_LIB_SOURCE_PATH),
            "tech_runtime_path": str(TECH_LIB_PATH),
            "tech_super_path": str(TECH_SUPER_PATH),
            "tech_map": TECH_MAP_COMMAND,
            "tech_lib_command": tech_lib_command,
        }

        if DELETE_OPTIMIZED_OUTPUTS:
            try:
                os.remove(output_file)
            except FileNotFoundError:
                pass

        return row
    except subprocess.CalledProcessError as e:
        print(f"  Error processing {design_name} with {recipe_name}:")
        print(e.stdout)
        print(e.stderr)
        return None


def write_labels_csv(rows):
    headers = ["design_id", "recipe_id", "output_name", "power_switch", "tech_lib", "tech_lib_path", "tech_runtime_path", "tech_super_path", "tech_map", "tech_lib_command"]
    with open(LABEL_CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# --- MAIN EXECUTION LOOP ---
def main():
    # Get all bench files
    designs = [f for f in os.listdir(INPUT_DIR) if f.endswith(".bench")]
    # Get all recipe scripts
    recipes = [f for f in os.listdir(RECIPE_DIR) if f.endswith(".script")]

    print(f"Found {len(designs)} designs and {len(recipes)} recipes.")
    print(f"Using tech library: {TECH_LIB_SOURCE_PATH} with mapping command: {TECH_MAP_COMMAND}")

    write_recipe_index(recipes)
    existing_labels_by_output = load_existing_labels()

    label_rows = []
    for d_file in designs:
        for r_file in recipes:
            input_path = os.path.join(INPUT_DIR, d_file)
            recipe_path = os.path.join(RECIPE_DIR, r_file)
            
            row = optimize_design(input_path, recipe_path, existing_labels_by_output)
            if row is not None:
                label_rows.append(row)

    write_labels_csv(label_rows)
    print(f"Wrote {len(label_rows)} label rows to: {LABEL_CSV_PATH}")
    print(f"Wrote recipe index to: {RECIPE_INDEX_PATH}")

if __name__ == "__main__":
    main()
