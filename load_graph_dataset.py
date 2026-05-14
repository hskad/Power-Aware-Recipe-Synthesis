import argparse
import json
import random
from pathlib import Path


def load_json_records(dataset_dir: Path):
    records = []
    for json_file in sorted(dataset_dir.glob("*.json")):
        payload = json.loads(json_file.read_text(encoding="utf-8"))

        x = [
            [
                node["type_id"],
                node["fanin"],
                node["fanout"],
                node["depth"],
            ]
            for node in payload.get("nodes", [])
        ]
        edge_index = payload.get("edges", [])
        y = payload.get("target")

        if y is None:
            label = payload.get("label", {})
            y = label.get("power_switch")

        if y is None:
            raise ValueError(f"Missing target/power_switch in {json_file}")

        records.append(
            {
                "id": json_file.stem,
                "design_id": payload.get("design_id", ""),
                "recipe_id": payload.get("recipe_id", ""),
                "x": x,
                "edge_index": edge_index,
                "y": float(y),
                "recipe": payload.get("recipe", {}),
            }
        )

    return records


def split_by_design(records, train_ratio=0.8, val_ratio=0.1, seed=7):
    design_ids = sorted({record["design_id"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(design_ids)

    n_total = len(design_ids)
    if n_total == 0:
        return [], [], []

    n_train = max(1, int(n_total * train_ratio))
    n_val = int(n_total * val_ratio)

    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    train_designs = set(design_ids[:n_train])
    val_designs = set(design_ids[n_train : n_train + n_val])
    test_designs = set(design_ids[n_train + n_val :])

    train = [record for record in records if record["design_id"] in train_designs]
    val = [record for record in records if record["design_id"] in val_designs]
    test = [record for record in records if record["design_id"] in test_designs]

    return train, val, test


def to_torch(records):
    try:
        import torch
    except ImportError:
        return None

    tensor_records = []
    for record in records:
        x = torch.tensor(record["x"], dtype=torch.float32)

        if record["edge_index"]:
            edge_index = torch.tensor(record["edge_index"], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        y = torch.tensor([record["y"]], dtype=torch.float32)

        tensor_records.append(
            {
                "id": record["id"],
                "design_id": record["design_id"],
                "recipe_id": record["recipe_id"],
                "x": x,
                "edge_index": edge_index,
                "y": y,
            }
        )

    return tensor_records


def main():
    parser = argparse.ArgumentParser(description="Load recipe-conditioned graph dataset rows")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parent / "graph_dataset"), help="Directory containing JSON dataset rows")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for design-level split")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    records = load_json_records(dataset_dir)

    if not records:
        print(f"No JSON files found in {dataset_dir}")
        return

    train, val, test = split_by_design(records, seed=args.seed)
    print(f"Loaded {len(records)} rows from {dataset_dir}")
    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    first = records[0]
    print(
        "Sample -> "
        f"id={first['id']}, design={first['design_id']}, recipe={first['recipe_id']}, "
        f"nodes={len(first['x'])}, edges={len(first['edge_index'])}, y={first['y']}"
    )

    tensor_records = to_torch(records)
    if tensor_records is None:
        print("PyTorch not installed: returning Python records only.")
    else:
        sample = tensor_records[0]
        print(
            "Tensor sample -> "
            f"x={tuple(sample['x'].shape)}, edge_index={tuple(sample['edge_index'].shape)}, y={tuple(sample['y'].shape)}"
        )


if __name__ == "__main__":
    main()
