"""Executa a comparacao justa: para cada semente, a MESMA divisao para todos os modelos.

Nomes dos modelos nos resultados:
    GCN-<topologia>               GCN do texto (mesma configuracao)
    DeepSets-matched              DeepSets com os hiperparametros da GCN (= GCN sem arestas)
    DeepSets-tuned                DeepSets com hiperparametros do Optuna
    RF-<repr>-default             RF com a configuracao do texto (200 arvores)
    RF-<repr>-tuned               RF com hiperparametros do Optuna
    MLP-<repr>-default / -tuned   idem para o MLP

Cada execucao vira um JSON em <out>/runs/<modelo>/seed_XXX.json. Execucoes ja
existentes sao puladas, entao da para interromper e retomar, ou dividir o
trabalho em varios processos (por exemplo, sementes ou modelos diferentes).
"""
import copy
import os

from src.experiments import common
from src.models.set_features import extract_play, build_matrix
from src.utils.logger import Logger


def _todo(out_dir, name, seed, overwrite):
    return overwrite or not os.path.exists(common.result_path(out_dir, name, seed))


def run(args):
    config = common.load_config(args.config)
    params = common.load_params(args.params)
    seeds = common.parse_seeds(args.seeds)
    models = set(m.lower() for m in args.models)
    out = args.out

    graphs = {}

    def get_graphs(strategy):
        if strategy not in graphs:
            graphs[strategy] = common.load_graphs(config, strategy)
        return graphs[strategy]

    # Variantes de DeepSets e das baselines vetoriais
    ds_variants = [v for v in args.deepsets_variants if v in params.get("DEEPSETS", {})]
    for v in set(args.deepsets_variants) - set(ds_variants):
        Logger.warning(f"DeepSets variant '{v}' not found in {args.params} (run the tune step first). Skipping.")

    vector_jobs = []  # (familia, repr, variante, params)
    for family in ("RF", "MLP"):
        if family.lower() not in models:
            continue
        fam = params.get(family, {})
        for rep in args.reprs:
            if rep in args.default_reprs:
                vector_jobs.append((family, rep, "default", fam["default"]))
            if not args.no_tuned:
                if rep in fam.get("tuned", {}):
                    vector_jobs.append((family, rep, "tuned", fam["tuned"][rep]))
                else:
                    Logger.warning(f"No tuned params for {family}-{rep} in {args.params} (run the tune step first).")

    plays = None
    base = args.base_strategy

    for seed in seeds:
        cfg = copy.deepcopy(config)
        cfg.RANDOM_SEED = seed
        pass_b, rush_b = get_graphs(base)
        split = common.make_split(len(pass_b), len(rush_b), seed, cfg)
        Logger.info(f"===== Seed {seed}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} =====")

        # GCN (uma por topologia)
        if "gcn" in models:
            for strategy in args.strategies:
                name = f"GCN-{strategy.rstrip('-')}"
                if not _todo(out, name, seed, args.overwrite):
                    continue
                pg, rg = get_graphs(strategy)
                assert (len(pg), len(rg)) == (len(pass_b), len(rush_b)), "Topologies have different numbers of plays"
                tr, va, te = common.apply_split(split, pg, rg)
                res = common.run_torch_model(cfg, "gcn", tr, va, te)
                res.update(family="GCN", strategy=strategy, representation="graph", variant="text")
                common.save_result(out, name, seed, res)

        # DeepSets (ignora as arestas; usa os mesmos atributos de no da GCN)
        if "deepsets" in models:
            tr, va, te = common.apply_split(split, pass_b, rush_b)
            for variant in ds_variants:
                name = f"DeepSets-{variant}"
                if not _todo(out, name, seed, args.overwrite):
                    continue
                res = common.run_torch_model(cfg, "deepsets", tr, va, te, deepsets_params=params["DEEPSETS"][variant])
                res.update(family="DeepSets", strategy=None, representation="set", variant=variant)
                common.save_result(out, name, seed, res)

        # RF e MLP com cada representacao vetorial
        pending = [j for j in vector_jobs if _todo(out, f"{j[0]}-{j[1]}-{j[2]}", seed, args.overwrite)]
        if pending:
            if plays is None:
                plays = {"P": [extract_play(g) for g in pass_b], "R": [extract_play(g) for g in rush_b]}
            tr, va, te = common.apply_split(split, plays["P"], plays["R"])
            matrices = {}
            for family, rep, variant, p in pending:
                if rep not in matrices:
                    matrices[rep] = (build_matrix(tr, rep), build_matrix(te, rep))
                (X_tr, y_tr), (X_te, y_te) = matrices[rep]
                name = f"{family}-{rep}-{variant}"
                Logger.info(f"Training {name} (seed {seed}, {X_tr.shape[1]} features)...")
                res = common.run_sklearn_model(family, p, seed, X_tr, y_tr, X_te, y_te,
                                               keys_eval=[q["key"] for q in te])
                res.update(family=family, strategy=None, representation=rep, variant=variant,
                           n_features=int(X_tr.shape[1]))
                common.save_result(out, name, seed, res)

    Logger.info("Done.")
