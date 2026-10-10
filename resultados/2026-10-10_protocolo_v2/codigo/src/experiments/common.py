"""Funcoes compartilhadas pelos experimentos de comparacao justa (pedidos da banca).

Pontos principais:
  * Os grafos de cada topologia sao construidos uma unica vez e guardados em
    cache (data/graphs/cache). O pre-processamento de cada semana e feito uma
    vez so e reaproveitado por todas as topologias.
  * A divisao treino/validacao/teste de cada semente reproduz exatamente a
    divisao usada pela GCN nos resultados do texto (GNNTrainer.split_and_prepare_data
    logo apos set_seed) e e a MESMA para todos os modelos (GCN, DeepSets, RF, MLP).
"""
import copy
import json
import os
import pickle
import random
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.utils.logger import Logger

CACHE_DIR = os.path.join("data", "graphs", "cache")
ALL_STRATEGIES = ["MST", "RNG", "CLOSEST-", "GABRIEL", "QB-CLOSEST-", "DELAUNAY"]
TARGET_NAMES = ["Rush", "Pass"]


# ----------------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [namespace_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    return obj


def load_params(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_params(path, params):
    with open(path, "w") as f:
        json.dump(params, f, indent=4)


def parse_seeds(text):
    """'1-30' -> [1..30]; '1,2,5' -> [1,2,5]; '1-10,20' -> ..."""
    seeds = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


# ----------------------------------------------------------------------------
# Grafos
# ----------------------------------------------------------------------------
def _cache_file(config, strategy):
    name = f"{strategy}_N{config.N}_qb{int(config.QB_LINK)}_ds{int(config.DOWN_SAMPLE)}_w{'-'.join(map(str, config.FILES.WEEKS_TO_READ))}.pkl"
    return os.path.join(CACHE_DIR, name)


def build_graph_cache(config, strategies, overwrite=False):
    """Constroi (uma vez) os grafos de todas as topologias pedidas e salva em cache."""
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.data.graph_builder import GraphBuilder
    from src.data.graph_strategies.strategy_factory import GraphStrategyFactory

    todo = [s for s in strategies if overwrite or not os.path.exists(_cache_file(config, s))]
    if not todo:
        Logger.info("All requested graph caches already exist.")
        return
    os.makedirs(CACHE_DIR, exist_ok=True)

    configs = {}
    for s in todo:
        c = copy.deepcopy(config)
        c.EDGE_STRATEGY = s
        configs[s] = c

    loader = DataLoader(config)
    preprocessor = DataPreprocessor(configs[todo[0]])
    graphs = {s: ([], []) for s in todo}

    for week in config.FILES.WEEKS_TO_READ:
        start = time.time()
        games, player_play, players, plays = loader.load_auxiliar_nfl_files()
        week_tracking = loader.load_week_data(week)
        plays_p, tracking_p, connections = preprocessor.execute(games, player_play, players, plays, week_tracking)

        for s in todo:
            if s != todo[0]:
                strategy = GraphStrategyFactory.create_strategy(configs[s])
                connections = strategy.calculate_connections(tracking_p.copy(), players)
            pass_g, rush_g = GraphBuilder(configs[s]).execute(plays_p, tracking_p, connections, downSample=config.DOWN_SAMPLE)
            graphs[s][0].extend(pass_g)
            graphs[s][1].extend(rush_g)
        Logger.info(f"Week {week} processed in {time.time() - start:.1f}s")

    for s in todo:
        with open(_cache_file(config, s), "wb") as f:
            pickle.dump(graphs[s], f)
        Logger.info(f"Saved {s}: pass={len(graphs[s][0])}, rush={len(graphs[s][1])} -> {_cache_file(config, s)}")


def load_graphs(config, strategy):
    path = _cache_file(config, strategy)
    if not os.path.exists(path):
        Logger.info(f"Graph cache for {strategy} not found; building it now...")
        build_graph_cache(config, [strategy])
    with open(path, "rb") as f:
        pass_graphs, rush_graphs = pickle.load(f)
    return pass_graphs, rush_graphs


# ----------------------------------------------------------------------------
# Divisao dos dados
# ----------------------------------------------------------------------------
def make_split(n_pass, n_rush, seed, config):
    """Indices de treino/validacao/teste identicos aos da GCN no texto.

    GNNTrainer.train_model chama set_seed(seed) (que comeca por random.seed(seed))
    e em seguida random.shuffle nas listas de passe e de corrida. random.shuffle
    so depende do tamanho da lista e do estado do gerador, entao embaralhar
    listas de indices do mesmo tamanho gera a mesma permutacao.
    """
    rng_state = random.getstate()
    random.seed(seed)
    p = list(range(n_pass))
    r = list(range(n_rush))
    random.shuffle(p)
    random.shuffle(r)
    random.setstate(rng_state)

    tr, va = config.DATASET.TRAIN_SPLIT, config.DATASET.VALIDATION_SPLIT
    p_tr, p_va = int(tr * n_pass), int(tr * n_pass) + int(va * n_pass)
    r_tr, r_va = int(tr * n_rush), int(tr * n_rush) + int(va * n_rush)
    return {
        "train": ([("P", i) for i in p[:p_tr]] + [("R", i) for i in r[:r_tr]]),
        "val": ([("P", i) for i in p[p_tr:p_va]] + [("R", i) for i in r[r_tr:r_va]]),
        "test": ([("P", i) for i in p[p_va:]] + [("R", i) for i in r[r_va:]]),
    }


def apply_split(split, pass_items, rush_items):
    pick = lambda part: [pass_items[i] if c == "P" else rush_items[i] for c, i in split[part]]
    return pick("train"), pick("val"), pick("test")


# ----------------------------------------------------------------------------
# Avaliacao
# ----------------------------------------------------------------------------
def summarize(labels, preds):
    report = classification_report(labels, preds, target_names=TARGET_NAMES, output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "report": report,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def result_path(out_dir, model_name, seed):
    return os.path.join(out_dir, "runs", model_name, f"seed_{seed:03d}.json")


def save_result(out_dir, model_name, seed, result):
    path = result_path(out_dir, model_name, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = dict(result, model=model_name, seed=seed, finished_at=datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as f:
        json.dump(to_jsonable(result), f)
    Logger.info(f"[{model_name} | seed {seed}] macro F1 = {result['macro_f1']:.4f} -> {path}")


# ----------------------------------------------------------------------------
# Modelos em grafo (GCN) e em conjunto (DeepSets) - mesmo GNNTrainer
# ----------------------------------------------------------------------------
def deepsets_train_params(ds):
    """Namespace no formato esperado por GNNTrainer.train_on_split"""
    return SimpleNamespace(EPOCHS=ds.EPOCHS, LEARNING_RATE=ds.LEARNING_RATE, WEIGHT_DECAY=ds.WEIGHT_DECAY,
                           OPTIMIZER=ds.OPTIMIZER, WARMUP_EPOCHS=ds.WARMUP_EPOCHS,
                           EARLY_STOP_PATIENCE=ds.EARLY_STOP_PATIENCE, MIN_DELTA=ds.MIN_DELTA)


def run_torch_model(config, kind, train_g, val_g, test_g, deepsets_params=None):
    """kind = 'gcn' ou 'deepsets'. Para DeepSets, deepsets_params e um dict com os hiperparametros."""
    from src.models.gcn_trainer import GNNTrainer
    from src.models.deepsets import DeepSets

    config = copy.deepcopy(config)
    trainer = GNNTrainer(config)
    if kind == "gcn":
        out = trainer.train_on_split(train_g, val_g, test_g)
        params = namespace_to_dict(config.GCN)
    elif kind == "deepsets":
        config.DEEPSETS = SimpleNamespace(**deepsets_params)
        out = trainer.train_on_split(train_g, val_g, test_g, model_factory=DeepSets,
                                     train_params=deepsets_train_params(config.DEEPSETS))
        params = dict(deepsets_params)
    else:
        raise ValueError(kind)

    result = summarize(out["test_labels"], out["test_preds"])
    result.update({
        "params": params,
        "n_parameters": out["model"].get_num_parameters(),
        "stopped_epoch": out["stopped_epoch"],
        "early_stopped": out["early_stopped"],
        "train_time_s": out["train_time_s"],
        "inference": out["inference"],
        "test_keys": out["test_keys"],
        "test_labels": out["test_labels"],
        "test_preds": out["test_preds"],
        "test_proba": out["test_proba"],
    })
    return result


# ----------------------------------------------------------------------------
# Baselines vetoriais (RF e MLP)
# ----------------------------------------------------------------------------
def make_sklearn_model(family, params, seed):
    params = dict(params)
    if family == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if family == "MLP":
        if "hidden_layer_sizes" in params:
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        return MLPClassifier(random_state=seed, **params)
    raise ValueError(family)


def _sklearn_inference(model, X_test, n_samples=200):
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # uma jogada por vez: paralelismo so atrapalha
    rows = X_test[:n_samples]
    for row in rows[:20]:
        model.predict(row.reshape(1, -1))
    start = time.perf_counter()
    for row in rows:
        model.predict(row.reshape(1, -1))
    per_play_ms = (time.perf_counter() - start) * 1000 / len(rows)
    start = time.perf_counter()
    model.predict(X_test)
    return {"per_play_ms": per_play_ms, "full_test_s": time.perf_counter() - start,
            "n_test": len(X_test), "device": "cpu"}


def run_sklearn_model(family, params, seed, X_train, y_train, X_eval, y_eval, keys_eval=None, timing=True):
    """Treina RF ou MLP. O MLP recebe os dados padronizados (StandardScaler ajustado no treino)."""
    if family == "MLP":
        scaler = StandardScaler().fit(X_train)
        X_train, X_eval = scaler.transform(X_train), scaler.transform(X_eval)

    model = make_sklearn_model(family, params, seed)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    preds = model.predict(X_eval)
    proba = model.predict_proba(X_eval)[:, 1]

    result = summarize(y_eval, preds)
    result.update({
        "params": params,
        "train_time_s": train_time,
        "test_keys": keys_eval,
        "test_labels": y_eval,
        "test_preds": preds,
        "test_proba": proba,
    })
    if family == "MLP":
        result["n_iter"] = int(model.n_iter_)
    if timing:
        result["inference"] = _sklearn_inference(model, X_eval)
    return result
