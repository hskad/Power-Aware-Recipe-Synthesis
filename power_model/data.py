import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch


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


@dataclass
class Sample:
    sample_id: str
    design_id: str
    recipe_id: str
    recipe_idx: int
    graph_features: list[float]
    target: float


def _load_recipe_index(recipe_index_path: Path) -> dict[str, int]:
    if not recipe_index_path.exists():
        return {}

    rows = json.loads(recipe_index_path.read_text(encoding="utf-8"))
    recipe_ids = sorted({row.get("recipe_id", "") for row in rows if row.get("recipe_id")})
    return {recipe_id: idx for idx, recipe_id in enumerate(recipe_ids)}


def _graph_feature_vector(payload: dict) -> list[float]:
    graph_features = payload.get("graph_features", {})
    values = []
    for key in GRAPH_FEATURE_KEYS:
        values.append(float(graph_features.get(key, 0.0)))
    return values


def load_samples(dataset_dir: Path, recipe_index_path: Path) -> tuple[list[Sample], dict[str, int]]:
    recipe_to_idx = _load_recipe_index(recipe_index_path)
    samples: list[Sample] = []

    for json_file in sorted(dataset_dir.glob("*.json")):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        recipe_id = payload.get("recipe_id", "")
        if recipe_id not in recipe_to_idx:
            recipe_to_idx[recipe_id] = len(recipe_to_idx)

        y = payload.get("target")
        if y is None:
            y = payload.get("label", {}).get("power_switch")
        if y is None:
            raise ValueError(f"Missing target in {json_file}")

        samples.append(
            Sample(
                sample_id=json_file.stem,
                design_id=payload.get("design_id", ""),
                recipe_id=recipe_id,
                recipe_idx=recipe_to_idx[recipe_id],
                graph_features=_graph_feature_vector(payload),
                target=float(y),
            )
        )

    return samples, recipe_to_idx


def load_design_graph_features(dataset_dir: Path, design_id: str) -> tuple[str, list[float]]:
    for json_file in sorted(dataset_dir.glob("*.json")):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        if payload.get("design_id", "") == design_id:
            return json_file.stem, _graph_feature_vector(payload)

    raise FileNotFoundError(f"No graph dataset entry found for design_id={design_id}")


def split_by_design(samples: list[Sample], train_ratio: float, val_ratio: float, seed: int):
    design_ids = sorted({sample.design_id for sample in samples})
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

    train = [sample for sample in samples if sample.design_id in train_designs]
    val = [sample for sample in samples if sample.design_id in val_designs]
    test = [sample for sample in samples if sample.design_id in test_designs]
    return train, val, test


def list_design_ids(samples: list[Sample]) -> list[str]:
    return sorted({sample.design_id for sample in samples})


def leave_one_design_out_split(samples: list[Sample], held_out_design_id: str) -> tuple[list[Sample], list[Sample]]:
    train_pool = [sample for sample in samples if sample.design_id != held_out_design_id]
    held_out = [sample for sample in samples if sample.design_id == held_out_design_id]
    return train_pool, held_out


def split_support_query(samples: list[Sample], k_shot: int, seed: int) -> tuple[list[Sample], list[Sample]]:
    if k_shot <= 0:
        raise ValueError("k_shot must be positive")
    if len(samples) <= k_shot:
        raise ValueError(f"Need more than k_shot samples. Got {len(samples)} samples and k_shot={k_shot}.")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    support_idx = set(indices[:k_shot])
    support = [samples[idx] for idx in range(len(samples)) if idx in support_idx]
    query = [samples[idx] for idx in range(len(samples)) if idx not in support_idx]
    return support, query


def _fit_standardizer(rows: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(rows, dtype=torch.float32)
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return mean, std


def _transform_target(y: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "none":
        return y
    if target_transform == "log1p":
        return torch.log1p(torch.clamp(y, min=0.0))
    raise ValueError(f"Unsupported target_transform: {target_transform}")


def to_tensors(
    samples: list[Sample],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_transform: str = "none",
):
    if not samples:
        return None

    x = torch.tensor([sample.graph_features for sample in samples], dtype=torch.float32)
    x = (x - feature_mean) / feature_std

    recipe_idx = torch.tensor([sample.recipe_idx for sample in samples], dtype=torch.long)
    y_raw = torch.tensor([sample.target for sample in samples], dtype=torch.float32).unsqueeze(1)
    y = _transform_target(y_raw, target_transform)

    return {
        "ids": [sample.sample_id for sample in samples],
        "design_ids": [sample.design_id for sample in samples],
        "recipe_ids": [sample.recipe_id for sample in samples],
        "x": x,
        "recipe_idx": recipe_idx,
        "y": y,
        "y_raw": y_raw,
    }


def prepare_splits(
    dataset_dir: Path,
    recipe_index_path: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    target_transform: str = "none",
):
    samples, recipe_to_idx = load_samples(dataset_dir, recipe_index_path)
    train_samples, val_samples, test_samples = split_by_design(samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    if not train_samples:
        raise ValueError("Training split is empty. Add more designs or adjust split ratios.")

    feature_mean, feature_std = _fit_standardizer([sample.graph_features for sample in train_samples])

    train = to_tensors(train_samples, feature_mean, feature_std, target_transform=target_transform)
    val = to_tensors(val_samples, feature_mean, feature_std, target_transform=target_transform)
    test = to_tensors(test_samples, feature_mean, feature_std, target_transform=target_transform)

    return {
        "recipe_to_idx": recipe_to_idx,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train": train,
        "val": val,
        "test": test,
    }
