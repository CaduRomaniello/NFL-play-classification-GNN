"""Experimentos pedidos pela banca: comparacao justa entre GCN e baselines.

Subcomandos (ver EXPERIMENTOS_BANCA.md):
    build     constroi o cache de grafos de todas as topologias
    tune      otimiza hiperparametros de RF / MLP / DeepSets (Optuna, validacao)
    run       executa todas as sementes com a mesma divisao para todos os modelos
    tune-gnn  protocolo v2: otimiza GCN / GraphSAGE / DeepSets (Optuna, validacao)
    run-gnn   protocolo v2: executa GCN / GraphSAGE / DeepSets nas sementes de avaliacao
    analyze   gera tabelas, boxplot, Friedman/Nemenyi, Wilcoxon e matrizes de confusao
"""
import argparse

from src.experiments import common
from src.models.set_features import REPRESENTATIONS

DEFAULT_CONFIG = "config_comparacao_justa.json"
DEFAULT_PARAMS = "fair_params.json"
DEFAULT_GNN_PARAMS = "gnn_params.json"
DEFAULT_OUT = "output/comparacao_justa"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the graph cache")
    b.add_argument("--strategies", nargs="+", default=common.ALL_STRATEGIES)
    b.add_argument("--overwrite", action="store_true")

    t = sub.add_parser("tune", help="tune RF/MLP/DeepSets hyperparameters on the validation set")
    t.add_argument("targets", nargs="+",
                   help="rf:<repr>, mlp:<repr>, rf:all, mlp:all, rf:strict, mlp:strict or deepsets. "
                        f"Representations: {', '.join(REPRESENTATIONS)}")
    t.add_argument("--trials", type=int, default=50)
    t.add_argument("--tuning-seed", type=int, default=0, help="split seed used for tuning (default 0)")
    t.add_argument("--sampler-seed", type=int, default=42)
    t.add_argument("--storage", default="sqlite:///optuna_baselines.db")
    t.add_argument("--params", default=DEFAULT_PARAMS)
    t.add_argument("--base-strategy", default="MST", help="graph cache used to read node features")

    r = sub.add_parser("run", help="run the fair comparison")
    r.add_argument("--seeds", default="1-29", help="seeds 1-29 match the text's GCN runs in output/results")
    r.add_argument("--models", nargs="+", default=["deepsets", "rf", "mlp"],
                   help="by default only the baselines; the text's GCN runs are imported in 'analyze'")
    r.add_argument("--strategies", nargs="+", default=["MST", "RNG"], help="topologies for the GCN")
    r.add_argument("--reprs", nargs="+", default=REPRESENTATIONS, choices=REPRESENTATIONS)
    r.add_argument("--default-reprs", nargs="+", default=["mean"],
                   help="representations that also run with the text's RF/MLP configuration")
    r.add_argument("--no-tuned", action="store_true", help="only run the text's RF/MLP configuration")
    r.add_argument("--deepsets-variants", nargs="+", default=["matched", "tuned"])
    r.add_argument("--base-strategy", default="MST", help="graph cache used by DeepSets/RF/MLP (edges ignored)")
    r.add_argument("--params", default=DEFAULT_PARAMS)
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--overwrite", action="store_true")

    tg = sub.add_parser("tune-gnn", help="protocol v2: tune GCN / GraphSAGE / DeepSets on the validation set")
    tg.add_argument("targets", nargs="+",
                    help="gcn:<topology>, sage:<topology>, gcn:all, sage:all or deepsets. "
                         f"Topologies: {', '.join(common.ALL_STRATEGIES)}")
    tg.add_argument("--trials", type=int, default=20)
    tg.add_argument("--tuning-seed", type=int, default=0)
    tg.add_argument("--sampler-seed", type=int, default=42)
    tg.add_argument("--storage", default="sqlite:///optuna_gnn.db")
    tg.add_argument("--params", default=DEFAULT_GNN_PARAMS)
    tg.add_argument("--threads", type=int, default=4, help="CPU threads per process (3 processes -> 4 each)")

    rg = sub.add_parser("run-gnn", help="protocol v2: run GCN / GraphSAGE / DeepSets on the evaluation seeds")
    rg.add_argument("--seeds", default="1-29")
    rg.add_argument("--models", nargs="+", default=["gcn", "sage", "deepsets"])
    rg.add_argument("--strategies", nargs="+", default=common.ALL_STRATEGIES)
    rg.add_argument("--matched-reference", default="MST",
                    help="DeepSets-matched = the tuned GCN of this topology without edges")
    rg.add_argument("--params", default=DEFAULT_GNN_PARAMS)
    rg.add_argument("--out", default=DEFAULT_OUT)
    rg.add_argument("--threads", type=int, default=4)
    rg.add_argument("--overwrite", action="store_true")

    a = sub.add_parser("analyze", help="aggregate and test the results")
    a.add_argument("--out", default=DEFAULT_OUT)
    a.add_argument("--old-gcn-dir", default=None,
                   help="also import the text's GCN runs (e.g. output/results); same split, paired by seed")
    a.add_argument("--reference", default=None, help="reference model for Wilcoxon (default: best GCN)")
    a.add_argument("--only", nargs="+", default=None, help="restrict the analysis to these model names")
    a.add_argument("--include-old-in-tests", action="store_true",
                   help="keep the old-protocol GCN/DeepSets runs in the statistical tests when v2 results exist")

    args = parser.parse_args()

    if args.cmd == "build":
        common.build_graph_cache(common.load_config(args.config), args.strategies, overwrite=args.overwrite)
    elif args.cmd == "tune":
        from src.experiments.tuning import tune
        tune(args)
    elif args.cmd == "run":
        from src.experiments.runner import run
        run(args)
    elif args.cmd == "tune-gnn":
        from src.experiments.gnn_v2 import tune
        tune(args)
    elif args.cmd == "run-gnn":
        from src.experiments.gnn_v2 import run
        run(args)
    elif args.cmd == "analyze":
        from src.experiments.analysis import analyze
        analyze(args)


if __name__ == "__main__":
    main()
