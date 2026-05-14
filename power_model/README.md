# Power Predictor Module (Isolated)

This module is a standalone power-prediction baseline and does not modify existing OpenABC model files.

## Objective (Phase 2–3)

Build a stable power predictor by:
1. Reusing the Bullseye-style QoR architecture from `OpenABC-master/models/qor/SynthNetV1`
2. Retraining the output head for scalar `power_switch` regression
3. Evaluating power prediction quality only (not recipe ranking)
4. Validating reproducibility across multiple random seeds

## What it does

- Loads graph dataset rows from `daksh/graph_dataset`
- Uses recipe conditioning via `recipe_id`
- Trains a recipe-conditioned regressor for `power_switch` (switched capacitance)
- Reports power-prediction-focused metrics:
  - **Primary**: `MAE` (global and per-design)
  - **Primary**: `RMSE` (global and per-design)
  - **Optional auxiliary**: `R2` (explained variance) for interpretability

No recipe-ranking metrics (e.g., top-k similarity, Spearman rank) are computed in this phase.

## Architecture

- **Base**: Bullseye QoR-style recipe-conditioned encoder
- **Input**: graph-level feature embeddings + recipe embedding
- **Output head**: single scalar `power_switch` regression
- **Reference**: `OpenABC-master/models/qor/SynthNetV1/model.py` (retargeted for power regression)

## Training Protocol

- **Data split**: Design-level (all recipes of a design stay in train/val/test together)
- **Checkpoint selection**: Lowest validation MAE
- **Reproducibility**: Run with fixed seed (logged in config); target 3-run variance validation
- **Hyperparameters logged**: seed, learning rate, epochs, batch size, train/val/test split ratios

### Few-shot mode for new designs

- Use `--eval-mode fewshot_lodo` to run leave-one-design-out evaluation.
- For each held-out design:
  - Train a base model on all other designs.
  - Measure zero-shot error on the held-out design.
  - Randomly sample `k` support points from that held-out design.
  - Adapt only the prediction head (graph encoder frozen).
  - Evaluate on the remaining query points.
- This estimates how much a small calibration set can help on unseen designs.

## Files

- `data.py` — dataset loading, design-level split, feature normalization
- `model.py` — recipe-conditioned regressor (Bullseye QoR base architecture)
- `train.py` — training/evaluation loop, MAE/RMSE reporting per design, checkpoint management
- `__init__.py` — module exports

## Run

From repo root:

```bash
# Standard run with default hyperparameters
python -m daksh.power_model.train --seed 42 --output-dir daksh/power_model/runs/exp_seed42

# Quick validation (smoke test, 5 epochs)
python -m daksh.power_model.train --seed 42 --epochs 5 --output-dir daksh/power_model/runs/smoke

# Reproducibility check: run 3 seeds back-to-back
for seed in 42 43 44; do
  python -m daksh.power_model.train --seed $seed --output-dir daksh/power_model/runs/exp_seed$seed
done

# Few-shot LODO evaluation (k=3 support points per held-out design)
python -m daksh.power_model.train \
  --eval-mode fewshot_lodo \
  --seed 42 \
  --lodo-seed 101 \
  --k-shot 3 \
  --epochs 120 \
  --fewshot-epochs 20 \
  --fewshot-lr 1e-3 \
  --output-dir daksh/power_model/runs/fewshot_k3
```

## Outputs

In `--output-dir`:

- `config.json` — seed, learning rate, epochs, batch size, split ratios
- `metrics_summary.json` — train/val/test MAE, RMSE, R² global and per-design
- `best_model.pt` — checkpoint selected by lowest validation MAE
- `pred_train.csv` — predictions on training set (with per-design breakdown)
- `pred_val.csv` — predictions on validation set (with per-design breakdown)
- `pred_test.csv` — predictions on test set (with per-design breakdown)
- `training_log.txt` or `tensorboard/` — loss curves (optional)

Few-shot mode additionally writes:

- `fewshot_results.json` — per-design zero-shot vs few-shot metrics and aggregate MAE summary
- `fewshot_pred_query_<design>.csv` — query predictions after few-shot adaptation per held-out design

## Acceptance Criteria (Phase 2–3 Gate)

1. **Stable convergence**: Validation MAE plateau across 3+ runs with consistent trajectories
2. **Meaningful gain**: Model outperforms baseline (e.g., predict train mean power) by >15–20% MAE
3. **Error consistency**: No severe collapse on specific designs (e.g., <2× average error spread)
4. **Reproducibility**: Low variance (<5%) across 3 runs with different seeds

## Next Steps (After Predictor Acceptance)

1. Freeze best checkpoint and preprocessing config
2. Begin simulated annealing (SA) integration as separate phase:
   - Use frozen predictor to score recipes during SA search
   - Rank recipes by predicted power
3. Re-introduce ranking metrics and recipe-ranking evaluation at SA phase

## Notes

- This is a baseline predictor isolated from OpenABC's existing ranking models
- Focus is power prediction quality, not recipe search quality
- Recipe conditioning input path is preserved but not evaluated for ranking in this phase
- The trained model will serve as the scoring function for simulated annealing recipe search (Phase 4)
