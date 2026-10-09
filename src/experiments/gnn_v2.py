"""Protocolo novo (v2): tuning e execucao de GCN, GraphSAGE e DeepSets em condicoes iguais.

Tuning (igual ao das baselines RF/MLP):
  * divisao da semente 0 (fora das sementes de avaliacao 1..29);
  * objetivo = Macro F1 na VALIDACAO (o teste nao e usado);
  * optuna.samplers.TPESampler(seed=42), 20 trials por estudo (padrao de --trials);
  * mesmo espaco de busca para os tres modelos (o peso das arestas so existe nas GNNs);
  * um estudo por (modelo, topologia); DeepSets tem um estudo so (nao usa arestas).

Arquivos proprios para nao conflitar com o protocolo anterior:
  gnn_params.json, optuna_gnn.db e <out>/runs_v2/.
"""
import fcntl
import json
import os
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

import optuna
import torch

from src.experiments import common
from src.models.trainer_v2 import graphs_to_data, train_v2
from src.utils.logger import Logger

CONVS = {"gcn": "GCN", "sage": "SAGE"}
FAMILY = {"gcn": "GCN", "sage": "GraphSAGE", "none": "DeepSets"}
DEEPSETS_STRATEGY = "MST"  # o DeepSets ignora as arestas; o cache so serve para ler os atributos


# ----------------------------------------------------------------------------
# Arquivo de parametros (com trava, pois 3 processos podem escrever)
# ----------------------------------------------------------------------------
@contextmanager
def _locked_params(path):
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        params = common.load_params(path)
        yield params
        common.save_params(path, params)
        fcntl.flock(lock, fcntl.LOCK_UN)


def train_cfg(config):
    return config.GNN_V2


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
class DataCache:
    """Mantem em memoria so a topologia em uso (economiza RAM com 3 processos)"""

    def __init__(self, config):
        self.config = config
        self.strategy = None
        self.data = None

    def get(self, strategy):
        if strategy != self.strategy:
            self.data = None
            pass_g, rush_g = common.load_graphs(self.config, strategy)
            Logger.info(f"Converting {strategy} graphs to PyG (edges in both directions)...")
            self.data = (graphs_to_data(pass_g), graphs_to_data(rush_g))
            del pass_g, rush_g
            self.strategy = strategy
        return self.data

    def split(self, strategy, seed):
        pass_d, rush_d = self.get(strategy)
        split = common.make_split(len(pass_d), len(rush_d), seed, self.config)
        return common.apply_split(split, pass_d, rush_d)


# ----------------------------------------------------------------------------
# Tuning
# ----------------------------------------------------------------------------
def search_space(trial, conv):
    p = {
        "CONV": conv,
        "HIDDEN_CHANNELS": trial.suggest_categorical("HIDDEN_CHANNELS", [64, 128, 256]),
        "CONV_LAYERS": trial.suggest_int("CONV_LAYERS", 1, 3),
        "POOLING": trial.suggest_categorical("POOLING", ["mean", "max", "mean+max"]),
        "POST_LAYERS": trial.suggest_int("POST_LAYERS", 0, 2),
        "DROPOUT": trial.suggest_float("DROPOUT", 0.0, 0.5),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1e-4, 1e-2, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-3, log=True),
        "EDGE_WEIGHT": "none",
    }
    if conv != "none":
        p["EDGE_WEIGHT"] = trial.suggest_categorical("EDGE_WEIGHT", ["none", "inverse"])
    return p


def parse_targets(targets):
    """gcn:MST, sage:all, deepsets -> [(conv, strategy)]"""
    out = []
    for t in targets:
        name, _, strat = t.partition(":")
        name = name.lower()
        if name == "deepsets":
            out.append(("none", DEEPSETS_STRATEGY))
            continue
        if name not in CONVS:
            raise ValueError(f"Unknown target {t} (use gcn:<topologia>, sage:<topologia> ou deepsets)")
        strategies = common.ALL_STRATEGIES if strat in ("", "all") else [strat]
        for s in strategies:
            if s not in common.ALL_STRATEGIES:
                raise ValueError(f"Unknown topology {s}. Options: {common.ALL_STRATEGIES}")
            out.append((name, s))
    return out


def model_key(conv, strategy):
    return ("DEEPSETS", "tuned") if conv == "none" else (CONVS[conv], strategy)


def _warn_if_cpu():
    if not torch.cuda.is_available():
        Logger.warning("CUDA nao disponivel: o treino vai rodar na CPU e sera MUITO mais lento. "
                       "Rode na maquina com GPU NVIDIA.")


def tune(args):
    config = common.load_config(args.config)
    torch.set_num_threads(args.threads)
    _warn_if_cpu()
    cache = DataCache(config)

    for conv, strategy in parse_targets(args.targets):
        name = "DeepSets" if conv == "none" else f"{CONVS[conv]}-{strategy}"
        # Trava: varios processos criando o mesmo banco SQLite ao mesmo tempo dao conflito de esquema
        with open(args.params + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            study = optuna.create_study(
                direction="maximize",
                study_name=f"gnn_{name}_seed{args.tuning_seed}",
                storage=optuna.storages.RDBStorage(args.storage, engine_kwargs={"connect_args": {"timeout": 120}}),
                sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
                load_if_exists=True,
            )
            fcntl.flock(lock, fcntl.LOCK_UN)
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = args.trials - done
        Logger.info(f"Tuning {name}: {done} trials done, {max(remaining, 0)} remaining")

        if remaining > 0:
            train_d, val_d, _ = cache.split(strategy, args.tuning_seed)

            def objective(trial):
                p = search_space(trial, conv)
                trial.set_user_attr("model_params", p)
                res = train_v2(SimpleNamespace(**p), train_cfg(config), train_d, val_d, None, args.tuning_seed)
                trial.set_user_attr("best_epoch", res["best_epoch"])
                trial.set_user_attr("stopped_epoch", res["stopped_epoch"])
                trial.set_user_attr("train_time_s", res["train_time_s"])
                return res["best_val_macro_f1"]

            study.optimize(objective, n_trials=remaining)

        best = study.best_trial
        Logger.info(f"Best {name}: val macro F1 = {best.value:.4f} | {best.user_attrs['model_params']}")
        group, key = model_key(conv, strategy)
        with _locked_params(args.params) as params:
            params.setdefault(group, {})[key] = best.user_attrs["model_params"]
            params.setdefault("_tuning_info", {})[name] = {
                "best_val_macro_f1": best.value,
                "n_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "tuning_seed": args.tuning_seed,
                "sampler": f"TPESampler(seed={args.sampler_seed})",
                "optuna_version": optuna.__version__,
                "study_name": study.study_name,
                "storage": args.storage,
                "training": common.namespace_to_dict(train_cfg(config)),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }


# ----------------------------------------------------------------------------
# Execucao (sementes de avaliacao)
# ----------------------------------------------------------------------------
def _result_path(out, name, seed):
    return os.path.join(out, "runs_v2", name, f"seed_{seed:03d}.json")


def _save(out, name, seed, res, extra):
    path = _result_path(out, name, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = common.summarize(res["test_labels"], res["test_preds"])
    result.update({k: v for k, v in res.items() if k != "standardizer"})
    result.update(extra, model=name, seed=seed, protocol="v2",
                  finished_at=datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as f:
        json.dump(common.to_jsonable(result), f)
    Logger.info(f"[{name} | seed {seed}] test macro F1 = {result['macro_f1']:.4f} -> {path}")


def run(args):
    config = common.load_config(args.config)
    torch.set_num_threads(args.threads)
    _warn_if_cpu()
    params = common.load_params(args.params)
    seeds = common.parse_seeds(args.seeds)
    models = [m.lower() for m in args.models]
    cache = DataCache(config)

    # (nome, conv, topologia, parametros)
    jobs = []
    for conv in ("gcn", "sage"):
        if conv not in models:
            continue
        for s in args.strategies:
            p = params.get(CONVS[conv], {}).get(s)
            if p is None:
                Logger.warning(f"No tuned params for {CONVS[conv]}-{s} in {args.params}; skipping (run tune-gnn first).")
                continue
            jobs.append((f"{CONVS[conv]}-{s.rstrip('-')}", conv, s, p))
    if "deepsets" in models:
        if "tuned" in params.get("DEEPSETS", {}):
            jobs.append(("DeepSets-tuned", "none", DEEPSETS_STRATEGY, params["DEEPSETS"]["tuned"]))
        else:
            Logger.warning("No tuned params for DeepSets; skipping DeepSets-tuned (run tune-gnn deepsets first).")
        gcn_ref = params.get("GCN", {}).get(args.matched_reference)
        if gcn_ref is not None:
            # Ablacao: a GCN otimizada da topologia de referencia, sem arestas
            matched = dict(gcn_ref, CONV="none", EDGE_WEIGHT="none")
            jobs.append(("DeepSets-matched", "none", DEEPSETS_STRATEGY, matched))
        else:
            Logger.warning(f"No tuned params for GCN-{args.matched_reference}; skipping DeepSets-matched.")

    # Topologia por fora para manter so um conjunto de grafos na memoria
    order = []
    for s in dict.fromkeys(j[2] for j in jobs):
        order += [j for j in jobs if j[2] == s]

    for name, conv, strategy, p in order:
        for seed in seeds:
            if not args.overwrite and os.path.exists(_result_path(args.out, name, seed)):
                continue
            train_d, val_d, test_d = cache.split(strategy, seed)
            Logger.info(f"===== {name} | seed {seed} =====")
            res = train_v2(SimpleNamespace(**p), train_cfg(config), train_d, val_d, test_d, seed)
            _save(args.out, name, seed, res, {
                "family": FAMILY[conv], "strategy": strategy if conv != "none" else None,
                "representation": "graph" if conv != "none" else "set", "variant": "v2",
                "params": p, "training": common.namespace_to_dict(train_cfg(config)),
            })
    Logger.info("Done.")
