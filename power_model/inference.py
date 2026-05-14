from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from daksh.power_model.data import _fit_standardizer, load_design_graph_features, load_samples
from daksh.power_model.model import PowerPredictor


def _stabilize_transformed_prediction(y_pred: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "log1p":
        return torch.clamp(y_pred, min=-10.0, max=16.0)
    return y_pred


def _invert_target_transform(y_pred: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "none":
        return y_pred
    if target_transform == "log1p":
        return torch.expm1(_stabilize_transformed_prediction(y_pred, target_transform))
    raise ValueError(f"Unsupported target_transform: {target_transform}")


@dataclass
class LoadedPredictor:
    model: PowerPredictor
    device: torch.device
    recipe_to_idx: dict[str, int]
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    target_transform: str

    def score(self, graph_features: list[float], recipe_id: str) -> float:
        if recipe_id not in self.recipe_to_idx:
            raise KeyError(f"Unknown recipe_id={recipe_id}")

        x = torch.tensor([graph_features], dtype=torch.float32, device=self.device)
        x = (x - self.feature_mean.to(self.device)) / self.feature_std.to(self.device)
        recipe_idx = torch.tensor([self.recipe_to_idx[recipe_id]], dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x, recipe_idx)
            y_pred = _invert_target_transform(y_pred, self.target_transform)
        return float(y_pred.item())


def load_predictor(checkpoint_path: Path, device: torch.device | None = None) -> LoadedPredictor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    recipe_to_idx = checkpoint["recipe_to_idx"]
    feature_mean = checkpoint["feature_mean"]
    feature_std = checkpoint["feature_std"]
    target_transform = config.get("target_transform", "none")

    hidden_dim = int(config.get("hidden_dim", 128))
    recipe_emb_dim = int(config.get("recipe_emb_dim", 16))
    dropout = float(config.get("dropout", 0.2))

    model = PowerPredictor(
        graph_feature_dim=feature_mean.shape[0],
        recipe_vocab_size=len(recipe_to_idx),
        hidden_dim=hidden_dim,
        recipe_emb_dim=recipe_emb_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LoadedPredictor(
        model=model,
        device=device,
        recipe_to_idx=recipe_to_idx,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_transform=target_transform,
    )


def predict_design_recipe(checkpoint_path: Path, dataset_dir: Path, design_id: str, recipe_id: str) -> float:
    predictor = load_predictor(checkpoint_path)
    _, graph_features = load_design_graph_features(dataset_dir, design_id)
    return predictor.score(graph_features, recipe_id)