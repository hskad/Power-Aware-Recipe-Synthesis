import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from daksh.power_model.data import (
    leave_one_design_out_split,
    list_design_ids,
    load_samples,
    prepare_splits,
    split_by_design,
    split_support_query,
    to_tensors,
    _fit_standardizer,
)
from daksh.power_model.model import PowerPredictor


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    rmse = math.sqrt(torch.mean((y_true - y_pred) ** 2).item())

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    if ss_tot.item() == 0:
        r2 = 0.0
    else:
        r2 = (1.0 - (ss_res / ss_tot)).item()

    return {"mae": mae, "rmse": rmse, "r2": r2}


def invert_target_transform(y: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "none":
        return y
    if target_transform == "log1p":
        # Keep exponentiation numerically stable for outlier predictions.
        return torch.expm1(torch.clamp(y, min=-10.0, max=16.0))
    raise ValueError(f"Unsupported target_transform: {target_transform}")


def stabilize_transformed_prediction(y_pred: torch.Tensor, target_transform: str) -> torch.Tensor:
    if target_transform == "log1p":
        return torch.clamp(y_pred, min=-10.0, max=16.0)
    return y_pred


def evaluate(model, split_tensors, device, target_transform: str):
    if split_tensors is None:
        return None, None, None

    model.eval()
    with torch.no_grad():
        x = split_tensors["x"].to(device)
        recipe_idx = split_tensors["recipe_idx"].to(device)
        y_true_raw = split_tensors["y_raw"].to(device)
        y_pred_transformed = stabilize_transformed_prediction(model(x, recipe_idx), target_transform)
        y_pred_raw = invert_target_transform(y_pred_transformed, target_transform)

    return metrics(y_true_raw, y_pred_raw), y_true_raw.cpu(), y_pred_raw.cpu()


def write_predictions(path: Path, split_tensors, y_true: torch.Tensor, y_pred: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "design_id", "recipe_id", "y_true", "y_pred"])
        for idx, sample_id in enumerate(split_tensors["ids"]):
            writer.writerow(
                [
                    sample_id,
                    split_tensors["design_ids"][idx],
                    split_tensors["recipe_ids"][idx],
                    float(y_true[idx].item()),
                    float(y_pred[idx].item()),
                ]
            )


def mean_baseline(train_tensors, eval_tensors):
    if eval_tensors is None:
        return None

    mean_value = torch.mean(train_tensors["y_raw"])
    y_pred = torch.full_like(eval_tensors["y_raw"], mean_value)
    return metrics(eval_tensors["y_raw"], y_pred)


def train_model(model, train, val, device, epochs, batch_size, lr, weight_decay, loss_name, target_transform):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()

    train_dataset = TensorDataset(train["x"], train["recipe_idx"], train["y"])
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, rb, yb in train_loader:
            xb = xb.to(device)
            rb = rb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = stabilize_transformed_prediction(model(xb, rb), target_transform)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.shape[0]

        train_loss = running_loss / len(train_dataset)
        val_stats, _, _ = evaluate(model, val, device, target_transform=target_transform)

        if val_stats is None:
            if best_state is None:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif val_stats["mae"] < best_val_mae:
            best_val_mae = val_stats["mae"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            if val_stats is None:
                print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val=NA")
            else:
                print(
                    f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                    f"val_mae={val_stats['mae']:.4f} val_rmse={val_stats['rmse']:.4f} val_r2={val_stats['r2']:.4f}"
                )

    if best_state is not None:
        model.load_state_dict(best_state)


def adapt_on_support(
    model,
    support,
    device,
    fewshot_epochs,
    fewshot_lr,
    fewshot_weight_decay,
    loss_name,
    target_transform,
    tune_recipe_embedding=False,
):
    if support is None:
        return

    for param in model.graph_encoder.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    for param in model.recipe_emb.parameters():
        param.requires_grad = tune_recipe_embedding

    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return

    optimizer = torch.optim.Adam(params, lr=fewshot_lr, weight_decay=fewshot_weight_decay)
    criterion = nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()

    dataset = TensorDataset(support["x"], support["recipe_idx"], support["y"])
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=True)

    model.train()
    for _ in range(fewshot_epochs):
        for xb, rb, yb in loader:
            xb = xb.to(device)
            rb = rb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = stabilize_transformed_prediction(model(xb, rb), target_transform=target_transform)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()


def aggregate_metric_dicts(metric_list):
    if not metric_list:
        return None

    keys = ["mae", "rmse", "r2"]
    mean_metrics = {key: sum(row[key] for row in metric_list) / len(metric_list) for key in keys}
    std_metrics = {
        f"{key}_std": (statistics.pstdev([row[key] for row in metric_list]) if len(metric_list) > 1 else 0.0)
        for key in keys
    }
    return {**mean_metrics, **std_metrics}


def run_fewshot_lodo(args):
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    recipe_index_path = Path(args.recipe_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, recipe_to_idx = load_samples(dataset_dir, recipe_index_path)
    design_ids = list_design_ids(samples)
    if len(design_ids) < 3:
        raise ValueError("Need at least 3 designs for leave-one-design-out few-shot evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for held_out_design in design_ids:
        train_pool_samples, held_out_samples = leave_one_design_out_split(samples, held_out_design)
        if len(held_out_samples) <= args.k_shot:
            print(f"Skipping {held_out_design}: only {len(held_out_samples)} samples for k_shot={args.k_shot}")
            continue

        train_samples, val_samples, _ = split_by_design(
            train_pool_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        if not train_samples:
            print(f"Skipping {held_out_design}: empty train split")
            continue

        feature_mean, feature_std = _fit_standardizer([sample.graph_features for sample in train_samples])
        train = to_tensors(train_samples, feature_mean, feature_std, target_transform=args.target_transform)
        val = to_tensors(val_samples, feature_mean, feature_std, target_transform=args.target_transform)
        held_out = to_tensors(held_out_samples, feature_mean, feature_std, target_transform=args.target_transform)

        if train is None or held_out is None:
            print(f"Skipping {held_out_design}: invalid tensors")
            continue

        model = PowerPredictor(
            graph_feature_dim=train["x"].shape[1],
            recipe_vocab_size=len(recipe_to_idx),
            hidden_dim=args.hidden_dim,
            recipe_emb_dim=args.recipe_emb_dim,
            dropout=args.dropout,
        ).to(device)

        train_model(
            model=model,
            train=train,
            val=val,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_name=args.loss,
            target_transform=args.target_transform,
        )

        zero_shot_stats, _, _ = evaluate(model, held_out, device, target_transform=args.target_transform)

        base_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        repeat_query_metrics = []
        repeat_support_metrics = []
        repeat_query_maes = []

        for repeat_idx in range(args.fewshot_repeats):
            support_samples, query_samples = split_support_query(
                held_out_samples,
                k_shot=args.k_shot,
                seed=args.lodo_seed + (repeat_idx * 10007) + sum(ord(ch) for ch in held_out_design),
            )

            support = to_tensors(support_samples, feature_mean, feature_std, target_transform=args.target_transform)
            query = to_tensors(query_samples, feature_mean, feature_std, target_transform=args.target_transform)

            if support is None or query is None:
                continue

            adapted_model = PowerPredictor(
                graph_feature_dim=train["x"].shape[1],
                recipe_vocab_size=len(recipe_to_idx),
                hidden_dim=args.hidden_dim,
                recipe_emb_dim=args.recipe_emb_dim,
                dropout=args.dropout,
            ).to(device)
            adapted_model.load_state_dict(base_state)

            adapt_on_support(
                model=adapted_model,
                support=support,
                device=device,
                fewshot_epochs=args.fewshot_epochs,
                fewshot_lr=args.fewshot_lr,
                fewshot_weight_decay=args.fewshot_weight_decay,
                loss_name=args.loss,
                target_transform=args.target_transform,
                tune_recipe_embedding=args.fewshot_tune_recipe_embedding,
            )

            fewshot_stats, query_true, query_pred = evaluate(
                adapted_model,
                query,
                device,
                target_transform=args.target_transform,
            )
            support_stats, _, _ = evaluate(adapted_model, support, device, target_transform=args.target_transform)

            if fewshot_stats is None or support_stats is None:
                continue

            repeat_query_metrics.append(fewshot_stats)
            repeat_support_metrics.append(support_stats)
            repeat_query_maes.append(fewshot_stats["mae"])

            if query_true is not None and query_pred is not None:
                write_predictions(
                    output_dir / f"fewshot_pred_query_{held_out_design}_repeat{repeat_idx}.csv",
                    query,
                    query_true,
                    query_pred,
                )

        if not repeat_query_metrics or not repeat_support_metrics:
            print(f"Skipping {held_out_design}: no valid few-shot repeats")
            continue

        fewshot_stats = aggregate_metric_dicts(repeat_query_metrics)
        support_stats = aggregate_metric_dicts(repeat_support_metrics)

        results[held_out_design] = {
            "k_shot": args.k_shot,
            "n_support": len(support_samples),
            "n_query": len(query_samples),
            "fewshot_repeats": args.fewshot_repeats,
            "zero_shot": zero_shot_stats,
            "fewshot_query": fewshot_stats,
            "fewshot_support": support_stats,
            "fewshot_query_mae_per_repeat": repeat_query_maes,
        }

        print(
            f"[{held_out_design}] zero_shot_mae={zero_shot_stats['mae']:.4f} "
            f"fewshot_query_mae={fewshot_stats['mae']:.4f} n_support={len(support_samples)}"
        )

    if not results:
        raise ValueError("No few-shot LODO results generated")

    zero_maes = [entry["zero_shot"]["mae"] for entry in results.values() if entry["zero_shot"] is not None]
    few_maes = [entry["fewshot_query"]["mae"] for entry in results.values() if entry["fewshot_query"] is not None]

    summary = {
        "mode": "fewshot_lodo",
        "k_shot": args.k_shot,
        "design_count": len(results),
        "zero_shot_mean_mae": sum(zero_maes) / len(zero_maes) if zero_maes else None,
        "fewshot_mean_mae": sum(few_maes) / len(few_maes) if few_maes else None,
        "improvement_pct": (
            (sum(zero_maes) - sum(few_maes)) / sum(zero_maes) * 100.0
            if zero_maes and sum(zero_maes) > 0
            else None
        ),
        "per_design": results,
        "config": vars(args),
    }

    summary_path = output_dir / "fewshot_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved few-shot summary: {summary_path}")
    print(f"zero_shot_mean_mae={summary['zero_shot_mean_mae']}")
    print(f"fewshot_mean_mae={summary['fewshot_mean_mae']}")
    print(f"improvement_pct={summary['improvement_pct']}")


def main():
    parser = argparse.ArgumentParser(description="Train a power-only predictor (no ranking metrics)")
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parents[1] / "graph_dataset"))
    parser.add_argument("--recipe-index", default=str(Path(__file__).resolve().parents[1] / "recipe_index.json"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "runs"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="log1p")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--recipe-emb-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval-mode", choices=["standard", "fewshot_lodo"], default="standard")
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--fewshot-epochs", type=int, default=20)
    parser.add_argument("--fewshot-lr", type=float, default=1e-3)
    parser.add_argument("--fewshot-weight-decay", type=float, default=0.0)
    parser.add_argument("--fewshot-repeats", type=int, default=3)
    parser.add_argument("--lodo-seed", type=int, default=101)
    parser.add_argument("--fewshot-tune-recipe-embedding", action="store_true")
    args = parser.parse_args()

    if args.eval_mode == "fewshot_lodo":
        run_fewshot_lodo(args)
        return

    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    recipe_index_path = Path(args.recipe_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_splits(
        dataset_dir=dataset_dir,
        recipe_index_path=recipe_index_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        target_transform=args.target_transform,
    )

    train = prepared["train"]
    val = prepared["val"]
    test = prepared["test"]

    if train is None:
        raise ValueError("Training split is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PowerPredictor(
        graph_feature_dim=train["x"].shape[1],
        recipe_vocab_size=len(prepared["recipe_to_idx"]),
        hidden_dim=args.hidden_dim,
        recipe_emb_dim=args.recipe_emb_dim,
        dropout=args.dropout,
    ).to(device)

    train_model(
        model=model,
        train=train,
        val=val,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        target_transform=args.target_transform,
    )

    train_stats, train_true, train_pred = evaluate(model, train, device, target_transform=args.target_transform)
    val_stats, val_true, val_pred = evaluate(model, val, device, target_transform=args.target_transform)
    test_stats, test_true, test_pred = evaluate(model, test, device, target_transform=args.target_transform)

    baseline_val = mean_baseline(train, val)
    baseline_test = mean_baseline(train, test)

    summary = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "config": vars(args),
    }

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "recipe_to_idx": prepared["recipe_to_idx"],
            "feature_mean": prepared["feature_mean"],
            "feature_std": prepared["feature_std"],
            "config": vars(args),
        },
        output_dir / "best_model.pt",
    )

    if train_true is not None and train_pred is not None:
        write_predictions(output_dir / "pred_train.csv", train, train_true, train_pred)
    if val_true is not None and val_pred is not None:
        write_predictions(output_dir / "pred_val.csv", val, val_true, val_pred)
    if test_true is not None and test_pred is not None:
        write_predictions(output_dir / "pred_test.csv", test, test_true, test_pred)

    print("Final metrics (power prediction only):")
    print(f"train -> {train_stats}")
    print(f"val   -> {val_stats}")
    print(f"test  -> {test_stats}")
    print(f"baseline val  -> {baseline_val}")
    print(f"baseline test -> {baseline_test}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
