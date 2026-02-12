import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EDGE_STRATEGIES = ["CLOSEST-", "QB-CLOSEST-", "DELAUNAY", "GABRIEL", "RNG", "MST"]
STORAGE = "sqlite:///optuna_study.db"


def load_studies():
    """Load all edge strategy studies from the database"""
    studies = {}
    for strategy in EDGE_STRATEGIES:
        try:
            study = optuna.load_study(
                study_name=f"gcn_optimization_{strategy}",
                storage=STORAGE,
            )
            studies[strategy] = study
            print(f"Loaded {strategy}: {len(study.trials)} trials, best={study.best_value:.4f}")
        except Exception as e:
            print(f"Could not load {strategy}: {e}")
    return studies


def plot_best_accuracy_comparison(studies):
    """Bar chart comparing best accuracy across all edge strategies"""
    strategies = []
    accuracies = []
    for strategy, study in sorted(studies.items(), key=lambda x: x[1].best_value, reverse=True):
        strategies.append(strategy)
        accuracies.append(study.best_value)

    fig = go.Figure(data=[
        go.Bar(
            x=strategies,
            y=accuracies,
            text=[f"{a:.4f}" for a in accuracies],
            textposition="auto",
            marker_color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"],
        )
    ])
    fig.update_layout(
        title="Best Accuracy per Edge Strategy",
        xaxis_title="Edge Strategy",
        yaxis_title="Best Accuracy",
        yaxis_range=[min(accuracies) - 0.02, max(accuracies) + 0.02] if accuracies else None,
    )
    fig.show()


def plot_optimization_history_all(studies):
    """Overlay optimization history for all strategies"""
    fig = go.Figure()
    for strategy, study in studies.items():
        trials = [t for t in study.trials if t.value is not None]
        trials.sort(key=lambda t: t.number)
        values = [t.value for t in trials]

        # Running best
        best_values = []
        current_best = -float("inf")
        for v in values:
            current_best = max(current_best, v)
            best_values.append(current_best)

        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=best_values,
            mode="lines",
            name=strategy,
        ))

    fig.update_layout(
        title="Optimization History (Running Best) - All Strategies",
        xaxis_title="Trial",
        yaxis_title="Best Accuracy",
    )
    fig.show()


def plot_trial_values_all(studies):
    """Scatter plot of all trial values for each strategy"""
    fig = go.Figure()
    for strategy, study in studies.items():
        trials = [t for t in study.trials if t.value is not None]
        trials.sort(key=lambda t: t.number)
        fig.add_trace(go.Scatter(
            x=[t.number for t in trials],
            y=[t.value for t in trials],
            mode="markers",
            name=strategy,
            opacity=0.6,
        ))

    fig.update_layout(
        title="All Trial Accuracies - All Strategies",
        xaxis_title="Trial Number",
        yaxis_title="Accuracy",
    )
    fig.show()


def plot_best_params_table(studies):
    """Table showing best params for each strategy"""
    all_params = set()
    for study in studies.values():
        all_params.update(study.best_params.keys())
    all_params = sorted(all_params)

    header = ["Strategy", "Accuracy"] + all_params
    rows = {col: [] for col in header}

    for strategy, study in sorted(studies.items(), key=lambda x: x[1].best_value, reverse=True):
        rows["Strategy"].append(strategy)
        rows["Accuracy"].append(f"{study.best_value:.4f}")
        for param in all_params:
            val = study.best_params.get(param, "N/A")
            if isinstance(val, float):
                rows[param].append(f"{val:.6f}")
            else:
                rows[param].append(str(val))

    fig = go.Figure(data=[go.Table(
        header=dict(values=header, fill_color="paleturquoise", align="left"),
        cells=dict(values=[rows[col] for col in header], fill_color="lavender", align="left"),
    )])
    fig.update_layout(title="Best Parameters per Edge Strategy")
    fig.show()


def plot_per_strategy_details(studies):
    """Show parameter importance and parallel coordinate for each strategy"""
    for strategy, study in studies.items():
        completed = [t for t in study.trials if t.value is not None]
        if len(completed) < 2:
            print(f"Skipping {strategy} — not enough completed trials ({len(completed)})")
            continue

        print(f"\n--- {strategy} ---")

        try:
            fig = plot_param_importances(study)
            fig.update_layout(title=f"Parameter Importance — {strategy}")
            fig.show()
        except Exception as e:
            print(f"  Could not plot param importances for {strategy}: {e}")

        try:
            fig = plot_parallel_coordinate(study)
            fig.update_layout(title=f"Parallel Coordinate — {strategy}")
            fig.show()
        except Exception as e:
            print(f"  Could not plot parallel coordinate for {strategy}: {e}")

        try:
            fig = plot_slice(study)
            fig.update_layout(title=f"Slice Plot — {strategy}")
            fig.show()
        except Exception as e:
            print(f"  Could not plot slice for {strategy}: {e}")


def plot_accuracy_distribution(studies):
    """Box plot of accuracy distributions across strategies"""
    fig = go.Figure()
    for strategy, study in studies.items():
        values = [t.value for t in study.trials if t.value is not None]
        fig.add_trace(go.Box(y=values, name=strategy))

    fig.update_layout(
        title="Accuracy Distribution per Edge Strategy",
        yaxis_title="Accuracy",
    )
    fig.show()


def print_summary(studies):
    """Print text summary of all results"""
    print(f"\n{'='*70}")
    print("SUMMARY — All Edge Strategies")
    print(f"{'='*70}")
    for strategy, study in sorted(studies.items(), key=lambda x: x[1].best_value, reverse=True):
        completed = len([t for t in study.trials if t.value is not None])
        failed = len([t for t in study.trials if t.value is None])
        print(f"\n  {strategy}:")
        print(f"    Best accuracy: {study.best_value:.4f}")
        print(f"    Completed trials: {completed}, Failed: {failed}")
        print(f"    Best params: {study.best_params}")

    # Export to CSV
    import pandas as pd
    rows = []
    for strategy, study in studies.items():
        for trial in study.trials:
            row = {"strategy": strategy, "trial": trial.number, "accuracy": trial.value}
            row.update(trial.params)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("optuna_results_all_strategies.csv", index=False)
    print(f"\nResults exported to optuna_results_all_strategies.csv")


def main():
    studies = load_studies()
    if not studies:
        print("No studies found in the database.")
        return

    print_summary(studies)
    plot_best_accuracy_comparison(studies)
    plot_optimization_history_all(studies)
    plot_trial_values_all(studies)
    plot_accuracy_distribution(studies)
    plot_best_params_table(studies)
    plot_per_strategy_details(studies)


if __name__ == "__main__":
    main()
