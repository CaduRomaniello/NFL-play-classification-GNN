# import optuna

# # Load existing study
# study = optuna.load_study(
#     study_name="gcn_optimization",
#     storage="sqlite:///optuna_study.db"
# )

# # Best trial
# print(f"Best accuracy: {study.best_value:.4f}")
# print(f"Best params: {study.best_params}")

# # All trials
# for trial in study.trials:
#     print(f"Trial {trial.number}: {trial.value} - {trial.params}")

# # Get as DataFrame (very useful!)
# df = study.trials_dataframe()
# print(df)
# df.to_csv("optuna_results.csv")  # Export to CSV

# # =============================================================================================

import optuna

study = optuna.load_study(
    study_name="gcn_optimization",
    storage="sqlite:///optuna_study.db"
)

# Built-in visualizations
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

# Optimization history
fig = plot_optimization_history(study)
fig.show()

# Parameter importance
fig = plot_param_importances(study)
fig.show()

# Parallel coordinate plot
fig = plot_parallel_coordinate(study)
fig.show()

# Slice plot (effect of each parameter)
fig = plot_slice(study)
fig.show()

# # =============================================================================================

# import optuna

# def main():
#     study = optuna.load_study(
#         study_name="gcn_optimization",
#         storage="sqlite:///optuna_study.db"
#     )
    
#     print(f"Number of finished trials: {len(study.trials)}")
#     print(f"\n{'='*50}")
#     print("BEST TRIAL:")
#     print(f"{'='*50}")
#     print(f"  Value (accuracy): {study.best_value:.4f}")
#     print(f"  Params:")
#     for key, value in study.best_params.items():
#         print(f"    {key}: {value}")
    
#     print(f"\n{'='*50}")
#     print("TOP 5 TRIALS:")
#     print(f"{'='*50}")
    
#     # Sort trials by value
#     sorted_trials = sorted(
#         [t for t in study.trials if t.value is not None],
#         key=lambda t: t.value,
#         reverse=True
#     )
    
#     for i, trial in enumerate(sorted_trials[:5]):
#         print(f"\n#{i+1} - Trial {trial.number}: accuracy={trial.value:.4f}")
#         for key, value in trial.params.items():
#             print(f"    {key}: {value}")

# if __name__ == "__main__":
#     main()