from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from daksh.power_model.inference import load_predictor


GRAPH_FEATURE_KEYS = [
    "num_nodes",
    "num_edges",
    "num_primary_inputs",
    "num_primary_outputs",
    "num_latches",
    "num_and_nodes",
    "max_depth",
    "avg_fanin",
    "avg_fanout",
]


def graph_vector_from_json(graph_json_path: Path) -> list[float]:
    payload = json.loads(graph_json_path.read_text(encoding="utf-8"))
    graph_features = payload.get("graph_features", {})
    return [float(graph_features.get(key, 0.0)) for key in GRAPH_FEATURE_KEYS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CSV with AIG/output_name, recipe, actual power (from labels), "
            "and predicted power (from model checkpoint)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "power_model" / "runs" / "final_train_42" / "best_model.pt"),
        help="Path to trained checkpoint (.pt)",
    )
    parser.add_argument(
        "--labels-csv",
        default=str(Path(__file__).resolve().parent / "labels_psp.csv"),
        help="CSV containing design_id, recipe_id, output_name, power_switch",
    )
    parser.add_argument(
        "--graph-dataset-dir",
        default=str(Path(__file__).resolve().parent / "graph_dataset"),
        help="Directory containing graph JSON files named design__recipe.json",
    )
    parser.add_argument(
        "--output-csv",
        default=str(Path(__file__).resolve().parent / "prediction_vs_actual.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    labels_csv_path = Path(args.labels_csv)
    graph_dataset_dir = Path(args.graph_dataset_dir)
    output_csv_path = Path(args.output_csv)

    predictor = load_predictor(checkpoint_path)

    output_rows: list[dict[str, object]] = []
    missing_graph = 0
    unknown_recipe = 0

    with open(labels_csv_path, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            design_id = row.get("design_id", "").strip()
            recipe_id = row.get("recipe_id", "").strip()
            output_name = row.get("output_name", "").strip()
            actual_power_str = row.get("power_switch", "").strip()

            if not design_id or not recipe_id:
                continue

            graph_json_path = graph_dataset_dir / f"{design_id}__{recipe_id}.json"
            if not graph_json_path.exists():
                missing_graph += 1
                continue

            try:
                actual_power = float(actual_power_str)
            except ValueError:
                continue

            try:
                features = graph_vector_from_json(graph_json_path)
                predicted_power = predictor.score(features, recipe_id)
            except KeyError:
                unknown_recipe += 1
                continue

            output_rows.append(
                {
                    "aig": output_name or f"{design_id}_{recipe_id}_opt",
                    "design_id": design_id,
                    "recipe": recipe_id,
                    "actual_power_ps_p": actual_power,
                    "predicted_power": predicted_power,
                }
            )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = [
            "aig",
            "design_id",
            "recipe",
            "actual_power_ps_p",
            "predicted_power",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_csv_path}")
    if missing_graph:
        print(f"Skipped {missing_graph} rows due to missing graph JSON")
    if unknown_recipe:
        print(f"Skipped {unknown_recipe} rows due to recipe IDs unknown to checkpoint")


if __name__ == "__main__":
    main()
