from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from daksh.clean_recipes import ALLOWED_COMMANDS


@dataclass(frozen=True)
class RecipeEntry:
    recipe_id: str
    recipe_file: str
    commands: list[str]


def load_recipe_catalog(recipe_index_path: Path) -> dict[str, RecipeEntry]:
    rows = json.loads(recipe_index_path.read_text(encoding="utf-8"))
    catalog: dict[str, RecipeEntry] = {}
    for row in rows:
        recipe_id = row.get("recipe_id", "").strip()
        if not recipe_id:
            continue
        commands = row.get("commands", [])
        catalog[recipe_id] = RecipeEntry(
            recipe_id=recipe_id,
            recipe_file=row.get("recipe_file", ""),
            commands=list(commands),
        )
    return catalog


def baseline_recipe_ids() -> dict[str, str]:
    # Standard ABC syn baselines mapped to the repository's canonical scripts.
    return {
        "syn": "abc1",
        "syn2": "abc2",
        "syn3": "abc3",
    }


def neighbor_recipe_id(current_recipe_id: str, recipe_ids: list[str], rng: random.Random) -> str:
    if not recipe_ids:
        raise ValueError("recipe_ids cannot be empty")
    if len(recipe_ids) == 1:
        return recipe_ids[0]

    candidates = [recipe_id for recipe_id in recipe_ids if recipe_id != current_recipe_id]
    if not candidates:
        candidates = recipe_ids[:]
    return rng.choice(candidates)


def recipe_id_to_file(recipe_catalog: dict[str, RecipeEntry], recipe_id: str) -> str:
    if recipe_id not in recipe_catalog:
        raise KeyError(f"Unknown recipe_id={recipe_id}")
    return recipe_catalog[recipe_id].recipe_file


def allowed_commands() -> list[str]:
    # Conservative AIG-safe subset for generated scripts.
    # These local transforms avoid mode-switching commands that can leave the
    # network in a non-AIG state during arbitrary mutation order.
    preferred = {
        "balance",
        "b",
        "rewrite",
        "rw",
        "refactor",
        "rf",
        "resub",
        "rs",
    }
    return sorted(cmd for cmd in ALLOWED_COMMANDS if cmd in preferred)


def normalize_recipe(commands: list[str], recipe_length: int, fill_command: str = "balance") -> list[str]:
    if recipe_length <= 0:
        raise ValueError("recipe_length must be positive")

    pool = set(allowed_commands())
    clean = []
    for cmd in commands:
        token = cmd.strip()
        if not token:
            continue
        if token in pool:
            clean.append(token)
        else:
            clean.append(fill_command)
    if len(clean) >= recipe_length:
        return clean[:recipe_length]

    padded = clean[:]
    padded.extend([fill_command] * (recipe_length - len(clean)))
    return padded


def random_recipe(recipe_length: int, rng: random.Random, command_pool: list[str] | None = None) -> list[str]:
    if command_pool is None:
        command_pool = allowed_commands()
    if not command_pool:
        raise ValueError("command_pool cannot be empty")
    return [rng.choice(command_pool) for _ in range(recipe_length)]


def mutate_recipe(commands: list[str], rng: random.Random, command_pool: list[str] | None = None) -> list[str]:
    if command_pool is None:
        command_pool = allowed_commands()
    if not commands:
        raise ValueError("commands cannot be empty")
    if not command_pool:
        raise ValueError("command_pool cannot be empty")

    mutated = commands[:]
    move = rng.choice(["replace", "swap"]) if len(mutated) > 1 else "replace"

    if move == "replace":
        idx = rng.randrange(len(mutated))
        mutated[idx] = rng.choice(command_pool)
    else:
        i, j = rng.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]

    return mutated


def recipe_distance(a: list[str], b: list[str]) -> int:
    if len(a) != len(b):
        raise ValueError("recipe_distance requires equal-length recipes")
    return sum(1 for left, right in zip(a, b) if left != right)


def nearest_recipe_id(
    commands: list[str],
    recipe_catalog: dict[str, RecipeEntry],
    recipe_length: int,
) -> tuple[str, int]:
    if not recipe_catalog:
        raise ValueError("recipe_catalog cannot be empty")

    target = normalize_recipe(commands, recipe_length)
    best_recipe_id = ""
    best_distance = None

    for recipe_id in sorted(recipe_catalog.keys()):
        entry = recipe_catalog[recipe_id]
        candidate = normalize_recipe(entry.commands, recipe_length)
        dist = recipe_distance(target, candidate)
        if best_distance is None or dist < best_distance:
            best_recipe_id = recipe_id
            best_distance = dist

    return best_recipe_id, int(best_distance)


def recipe_script_text(commands: list[str]) -> str:
    return "\n".join(commands).strip() + "\n"