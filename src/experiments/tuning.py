"""Otimizacao de hiperparametros das baselines (RF, MLP e DeepSets) com Optuna.

Protocolo (igual para todos os alvos):
  * divisao da semente de tuning (padrao 0, fora das sementes de avaliacao 1..30);
  * treino no conjunto de treino e objetivo = Macro F1 no conjunto de VALIDACAO
    (o teste nao e usado no tuning);
  * amostrador TPE (optuna.samplers.TPESampler) com semente fixa;
  * estudos salvos em SQLite, entao a execucao pode ser interrompida e retomada.

Os melhores hiperparametros de cada alvo vao para o arquivo de parametros
(padrao fair_params.json), em <FAMILIA>.tuned.<representacao>.
"""
import copy
from datetime import datetime

import optuna

from src.experiments import common
from src.models.set_features import extract_play, build_matrix, REPRESENTATIONS, STRICT_REPRESENTATIONS
from src.utils.logger import Logger


def rf_space(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 500]),
        "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8]),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.25, 0.5]),
    }


def mlp_space(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    width = trial.suggest_categorical("width", [64, 128, 256])
    return {
        "hidden_layer_sizes": [width] * n_layers,
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "max_iter": 1000,
    }


def deepsets_space(trial, base):
    params = dict(base)
    params.update({
        "HIDDEN_CHANNELS": trial.suggest_categorical("HIDDEN_CHANNELS", [64, 128, 256]),
        "PHI_LAYERS": trial.suggest_int("PHI_LAYERS", 1, 3),
        "RHO_LAYERS": trial.suggest_int("RHO_LAYERS", 0, 2),
        "POOLING": trial.suggest_categorical("POOLING", ["mean", "max", "mean+max"]),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.0, 0.5),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-4, 1e-2, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-3, log=True),
    })
    return params


def parse_target(target):
    """'rf:concat_team_y' -> ('RF', 'concat_team_y'); 'deepsets' -> ('DEEPSETS', None)"""
    family, _, rep = target.partition(":")
    family = family.upper()
    if family == "DEEPSETS":
        return family, None
    if family not in ("RF", "MLP"):
        raise ValueError(f"Unknown target {target}")
    if rep in ("all", "strict"):
        return family, rep
    if rep not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{rep}'. Options: {REPRESENTATIONS}")
    return family, rep


def expand_targets(targets):
    out = []
    for t in targets:
        family, rep = parse_target(t)
        if rep == "all":
            out.extend((family, r) for r in REPRESENTATIONS)
        elif rep == "strict":
            out.extend((family, r) for r in STRICT_REPRESENTATIONS)
        else:
            out.append((family, rep))
    return out


def tune(args):
    config = common.load_config(args.config)
    config.RANDOM_SEED = args.tuning_seed
    params_file = common.load_params(args.params)

    pass_g, rush_g = common.load_graphs(config, args.base_strategy)
    split = common.make_split(len(pass_g), len(rush_g), args.tuning_seed, config)
    train_g, val_g, _ = common.apply_split(split, pass_g, rush_g)

    plays = None
    for family, rep in expand_targets(args.targets):
        name = f"{family}-{rep}" if rep else family
        study = optuna.create_study(
            direction="maximize",
            study_name=f"tuning_{name}_seed{args.tuning_seed}",
            storage=args.storage,
            sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
            load_if_exists=True,
        )
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = args.trials - done
        Logger.info(f"Tuning {name}: {done} trials done, {max(remaining, 0)} remaining")

        if family == "DEEPSETS":
            base = params_file["DEEPSETS"]["matched"]

            def objective(trial):
                p = deepsets_space(trial, base)
                trial.set_user_attr("model_params", p)
                res = common.run_torch_model(config, "deepsets", train_g, val_g, val_g, deepsets_params=p)
                trial.set_user_attr("stopped_epoch", res["stopped_epoch"])
                return res["macro_f1"]
        else:
            if plays is None:
                plays = {"P": [extract_play(g) for g in pass_g], "R": [extract_play(g) for g in rush_g]}
            tr, va, _ = common.apply_split(split, plays["P"], plays["R"])
            X_tr, y_tr = build_matrix(tr, rep)
            X_va, y_va = build_matrix(va, rep)
            space = rf_space if family == "RF" else mlp_space

            def objective(trial, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, space=space, family=family):
                p = space(trial)
                trial.set_user_attr("model_params", p)
                res = common.run_sklearn_model(family, p, args.tuning_seed, X_tr, y_tr, X_va, y_va, timing=False)
                return res["macro_f1"]

        if remaining > 0:
            study.optimize(objective, n_trials=remaining)

        best = study.best_trial
        Logger.info(f"Best {name}: val macro F1 = {best.value:.4f} | {best.user_attrs['model_params']}")

        params_file = common.load_params(args.params)  # recarrega: outro processo pode ter escrito
        if family == "DEEPSETS":
            params_file["DEEPSETS"]["tuned"] = best.user_attrs["model_params"]
        else:
            params_file.setdefault(family, {}).setdefault("tuned", {})[rep] = best.user_attrs["model_params"]
        params_file.setdefault("_tuning_info", {})[name] = {
            "best_val_macro_f1": best.value,
            "n_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "tuning_seed": args.tuning_seed,
            "sampler": f"TPESampler(seed={args.sampler_seed})",
            "optuna_version": optuna.__version__,
            "study_name": study.study_name,
            "storage": args.storage,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        common.save_params(args.params, params_file)
