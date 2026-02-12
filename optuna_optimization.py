# optuna_optimization.py
import json
import optuna
from types import SimpleNamespace
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.models.gcn_trainer import GNNTrainer
from src.utils.logger import Logger


def create_config_from_trial(base_config, trial):
    """Create config with Optuna-suggested hyperparameters"""
    # Copy base config
    config = base_config
    
    config.EDGE_STRATEGY = trial.suggest_categorical(
        "edge_strategy", 
        ["CLOSEST-", "QB-CLOSEST-", "DELAUNAY", "GABRIEL", "RNG", "MST"]
    )
    
    # Suggest GCN hyperparameters
    config.GCN.HIDDEN_CHANNELS = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
    config.GCN.HIDDEN_LAYERS = trial.suggest_int("hidden_layers", 1, 4)
    config.GCN.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config.GCN.DROPOUT = trial.suggest_float("dropout", 0.1, 0.5)
    config.GCN.WEIGHT_DECAY = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    config.GCN.WARMUP_EPOCHS = trial.suggest_int("warmup_epochs", 5, 30)
    
    
    return config


def objective(trial):
    """Optuna objective function"""
    # Load base config
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    # Apply Optuna suggestions
    config = create_config_from_trial(config, trial)
    # config.EDGE_STRATEGY = "RNG"  # Fix strategy for optimization
    config.DOWN_SAMPLE = False
    
    # Build data (can cache this outside the objective for efficiency)
    data_pipeline = DataPipeline(config)
    pass_graphs, rush_graphs = data_pipeline.execute()
    
    # Train model
    trainer = GNNTrainer(config)
    train_graphs, val_graphs, test_graphs = trainer.split_and_prepare_data(pass_graphs, rush_graphs)
    
    # Get validation accuracy (you'd need to modify train_model to return this)
    results = trainer.train_model(pass_graphs, rush_graphs)
    
    # Return metric to optimize
    # best_gcn_results comes from classification_report(output_dict=True)
    # which has keys: 'accuracy', 'Rush', 'Pass', 'macro avg', 'weighted avg'
    test_accuracy = results['best_gcn_results']['accuracy']
    
    return test_accuracy


def main():
    # Create study
    study = optuna.create_study(
        direction="maximize",  # maximize accuracy
        study_name="gcn_optimization",
        storage="sqlite:///optuna_study.db",  # persist results
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    
    # Print results
    print(f"\nBest trial:")
    print(f"  Value (val_acc): {study.best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params
    with open("best_params.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)


if __name__ == "__main__":
    main()