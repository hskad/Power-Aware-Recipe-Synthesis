from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from daksh.power_model.data import load_design_graph_features
from daksh.power_model.inference import load_predictor
from daksh.recipe_search_utils import (
    allowed_commands,
    baseline_recipe_ids,
    load_recipe_catalog,
    mutate_recipe,
    nearest_recipe_id,
    normalize_recipe,
    random_recipe,
    recipe_id_to_file,
    recipe_script_text,
)
from daksh.synthesis import optimize_design


@dataclass
class SearchStep:
    step: int
    temperature: float
    current_recipe_signature: str
    current_proxy_recipe_id: str
    current_score: float
    candidate_recipe_signature: str
    candidate_proxy_recipe_id: str
    candidate_score: float
    accepted: bool


def _temperature(start_temp: float, end_temp: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end_temp
    ratio = step / max(1, total_steps - 1)
    return start_temp * ((end_temp / start_temp) ** ratio)


def run_simulated_annealing(
    design_id: str,
    graph_features: list[float],
    predictor,
    recipe_catalog,
    initial_commands: list[str],
    recipe_length: int,
    steps: int,
    start_temp: float,
    end_temp: float,
    seed: int,
):
    rng = random.Random(seed)
    command_pool = allowed_commands()

    current_commands = normalize_recipe(initial_commands, recipe_length)
    current_proxy_recipe_id, _ = nearest_recipe_id(current_commands, recipe_catalog, recipe_length)
    current_score = predictor.score(graph_features, current_proxy_recipe_id)

    best_commands = current_commands[:]
    best_proxy_recipe_id = current_proxy_recipe_id
    best_score = current_score
    trace: list[SearchStep] = []

    for step in range(steps):
        temp = _temperature(start_temp, end_temp, step, steps)

        candidate_commands = mutate_recipe(current_commands, rng, command_pool=command_pool)
        candidate_proxy_recipe_id, _ = nearest_recipe_id(candidate_commands, recipe_catalog, recipe_length)
        candidate_score = predictor.score(graph_features, candidate_proxy_recipe_id)

        accepted = False
        if candidate_score <= current_score:
            accepted = True
        else:
            delta = candidate_score - current_score
            accept_prob = math.exp(-delta / max(temp, 1e-8))
            accepted = rng.random() < accept_prob

        if accepted:
            current_commands = candidate_commands
            current_proxy_recipe_id = candidate_proxy_recipe_id
            current_score = candidate_score

        if current_score < best_score:
            best_commands = current_commands[:]
            best_proxy_recipe_id = current_proxy_recipe_id
            best_score = current_score

        trace.append(
            SearchStep(
                step=step,
                temperature=temp,
                current_recipe_signature=";".join(current_commands),
                current_proxy_recipe_id=current_proxy_recipe_id,
                current_score=current_score,
                candidate_recipe_signature=";".join(candidate_commands),
                candidate_proxy_recipe_id=candidate_proxy_recipe_id,
                candidate_score=candidate_score,
                accepted=accepted,
            )
        )

    return {
        "design_id": design_id,
        "best_recipe_id": best_proxy_recipe_id,
        "best_recipe_commands": best_commands,
        "best_predicted_power": best_score,
        "final_recipe_id": current_proxy_recipe_id,
        "final_recipe_commands": current_commands,
        "final_predicted_power": current_score,
        "trace": [asdict(step) for step in trace],
    }


def validate_with_abc(design_file: Path, recipe_catalog, recipe_id: str):
    recipe_file = Path(recipe_id_to_file(recipe_catalog, recipe_id))
    return optimize_design(str(design_file), str(recipe_file), existing_labels_by_output={})


def validate_generated_recipe_with_abc(design_file: Path, generated_script_path: Path):
    return optimize_design(str(design_file), str(generated_script_path), existing_labels_by_output={})


def main():
    parser = argparse.ArgumentParser(description="Simulated annealing recipe search guided by the power predictor")
    parser.add_argument("--design-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parent / "graph_dataset"))
    parser.add_argument("--recipe-index", default=str(Path(__file__).resolve().parent / "recipe_index.json"))
    parser.add_argument("--orig-designs-dir", default=str(Path(__file__).resolve().parent / "orig_designs"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "sa_runs"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start-temp", type=float, default=5.0)
    parser.add_argument("--end-temp", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-recipe-id", default="abc1")
    parser.add_argument("--recipe-length", type=int, default=20)
    args = parser.parse_args()

    if args.recipe_length <= 0:
        raise ValueError("recipe_length must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe_catalog = load_recipe_catalog(Path(args.recipe_index))
    recipe_ids = sorted(recipe_catalog.keys())
    baseline_map = baseline_recipe_ids()

    _, graph_features = load_design_graph_features(Path(args.dataset_dir), args.design_id)
    predictor = load_predictor(Path(args.checkpoint))

    initial_recipe_id = args.initial_recipe_id if args.initial_recipe_id in recipe_ids else baseline_map["syn"]
    if initial_recipe_id in recipe_catalog:
        initial_commands = normalize_recipe(recipe_catalog[initial_recipe_id].commands, args.recipe_length)
    else:
        initial_commands = random_recipe(args.recipe_length, random.Random(args.seed), command_pool=allowed_commands())

    search_result = run_simulated_annealing(
        design_id=args.design_id,
        graph_features=graph_features,
        predictor=predictor,
        recipe_catalog=recipe_catalog,
        initial_commands=initial_commands,
        recipe_length=args.recipe_length,
        steps=args.steps,
        start_temp=args.start_temp,
        end_temp=args.end_temp,
        seed=args.seed,
    )
    search_summary = {k: v for k, v in search_result.items() if k != "trace"}

    design_file = Path(args.orig_designs_dir) / f"{args.design_id}.bench"
    generated_scripts_dir = output_dir / "generated_scripts"
    generated_scripts_dir.mkdir(parents=True, exist_ok=True)
    best_script_path = generated_scripts_dir / f"{args.design_id}_sa_best.script"
    best_script_path.write_text(recipe_script_text(search_result["best_recipe_commands"]), encoding="utf-8")

    candidate_validation = validate_generated_recipe_with_abc(design_file, best_script_path)
    candidate_validation_source = "generated"
    if candidate_validation is None:
        candidate_validation = validate_with_abc(design_file, recipe_catalog, search_result["best_recipe_id"])
        candidate_validation_source = "proxy_fallback"

    baseline_results = {}
    for baseline_name, recipe_id in baseline_map.items():
        baseline_results[baseline_name] = validate_with_abc(design_file, recipe_catalog, recipe_id)

    summary = {
        "design_id": args.design_id,
        "search": search_summary,
        "search_proxy_recipe_id": search_result["best_recipe_id"],
        "search_generated_script": str(best_script_path),
        "candidate_validation": candidate_validation,
        "candidate_validation_source": candidate_validation_source,
        "baseline_validation": baseline_results,
        "config": vars(args),
    }

    summary_path = output_dir / f"{args.design_id}_sa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved summary: {summary_path}")
    print(
        f"Best predicted proxy recipe: {search_result['best_recipe_id']} "
        f"-> {search_result['best_predicted_power']:.4f}"
    )
    print(f"Best generated command count: {len(search_result['best_recipe_commands'])}")
    candidate_power = None if candidate_validation is None else candidate_validation.get("power_switch")
    print(f"Best actual candidate power: {candidate_power} ({candidate_validation_source})")


if __name__ == "__main__":
    main()
