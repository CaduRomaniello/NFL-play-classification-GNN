# optuna_optimization.py
import json
import os
import optuna
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.models.gcn_trainer import GNNTrainer
from src.utils.logger import Logger


EDGE_STRATEGIES = ["CLOSEST-", "QB-CLOSEST-", "DELAUNAY", "GABRIEL", "RNG", "MST"]
N_TRIALS = 100
N_JOBS_PER_STUDY = 2  # parallel trials within each strategy
MAX_STRATEGY_WORKERS = 3  # how many strategies to run in parallel


def create_config_from_trial(base_config, trial, edge_strategy):
    """Create config with Optuna-suggested hyperparameters"""
    config = base_config
    config.EDGE_STRATEGY = edge_strategy

    # Suggest GCN hyperparameters
    config.GCN.HIDDEN_CHANNELS = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
    config.GCN.HIDDEN_LAYERS = trial.suggest_int("hidden_layers", 1, 4)
    # config.GCN.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    # config.GCN.DROPOUT = trial.suggest_float("dropout", 0.1, 0.5)
    # config.GCN.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config.GCN.WARMUP_EPOCHS = trial.suggest_int("warmup_epochs", 5, 30)

    return config


def create_objective(edge_strategy):
    """Create an objective function for a specific edge strategy"""
    def objective(trial):
        # Load base config
        with open("config.json", "r") as f:
            config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

        # Apply Optuna suggestions with fixed edge strategy
        config = create_config_from_trial(config, trial, edge_strategy)
        config.DOWN_SAMPLE = False
        # Don't fix the seed — let each trial have natural randomness
        # so Optuna finds hyperparams that are robust across initializations
        config.RANDOM_SEED = trial.number  # different seed per trial

        # Build data
        data_pipeline = DataPipeline(config)
        pass_graphs, rush_graphs = data_pipeline.execute()

        # Train model
        trainer = GNNTrainer(config)
        train_graphs, val_graphs, test_graphs = trainer.split_and_prepare_data(pass_graphs, rush_graphs)
        results = trainer.train_model(pass_graphs, rush_graphs)

        test_accuracy = results['best_gcn_results']['accuracy']
        return test_accuracy

    return objective


def run_strategy(strategy):
    """Run optimization for a single edge strategy (executed in a separate process)"""
    print(f"\n{'='*60}")
    print(f"[PID {os.getpid()}] Optimizing edge strategy: {strategy}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        study_name=f"gcn_optimization_{strategy}",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )

    study.optimize(
        create_objective(strategy),
        n_trials=N_TRIALS,
        n_jobs=N_JOBS_PER_STUDY,
        show_progress_bar=True
    )

    print(f"\nBest trial for {strategy}:")
    print(f"  Value (val_acc): {study.best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    return strategy, {
        "best_accuracy": study.best_trial.value,
        "best_params": study.best_trial.params
    }


def main():
    all_results = {}

    with ProcessPoolExecutor(max_workers=MAX_STRATEGY_WORKERS) as executor:
        futures = {
            executor.submit(run_strategy, strategy): strategy
            for strategy in EDGE_STRATEGIES
        }

        for future in as_completed(futures):
            strategy = futures[future]
            try:
                strategy_name, result = future.result()
                all_results[strategy_name] = result
            except Exception as e:
                print(f"Strategy {strategy} failed: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY - Best results per edge strategy:")
    print(f"{'='*60}")
    for strategy, result in sorted(all_results.items(), key=lambda x: x[1]["best_accuracy"], reverse=True):
        print(f"  {strategy}: {result['best_accuracy']:.4f}")

    # Save all results
    with open("best_params.json", "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()