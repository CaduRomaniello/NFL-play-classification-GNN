"""Analise dos resultados da comparacao justa.

Gera em <out>/analise:
    resumo.csv / resumo.tex     Macro F1 (media, desvio, min, max), acuracia, F1 por classe,
                                tempo de inferencia por jogada e numero de execucoes
    boxplot_macro_f1.png/pdf    distribuicao do Macro F1 por modelo
    friedman_nemenyi.txt        teste de Friedman, ranks medios e Nemenyi (todas as configs)
    nemenyi_cd.png/pdf          diagrama de diferenca critica
    wilcoxon_vs_ref.csv         Wilcoxon pareado (por semente) contra o modelo de referencia,
                                com correcao de Holm
    matrizes_confusao.png/pdf   matrizes de confusao medias (normalizadas por linha)
    relatorio.md                resumo legivel de tudo acima
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.experiments.common import ALL_STRATEGIES
from src.utils.logger import Logger

plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

FAMILY_COLOR = {"GCN": "#2a78d6", "GraphSAGE": "#0f4c8a", "DeepSets": "#8e44ad", "RF": "#eb6834", "MLP": "#1baf7a"}
OLD_TAGS = ("(texto)", "(antigo)")  # resultados do protocolo antigo da GCN/DeepSets


def br(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def load_runs(out_dir, old_gcn_dir=None):
    rows, preds = [], {}
    files = sorted(glob.glob(os.path.join(out_dir, "runs", "*", "seed_*.json")))
    files_v2 = sorted(glob.glob(os.path.join(out_dir, "runs_v2", "*", "seed_*.json")))
    for f in files + files_v2:
        with open(f) as fh:
            j = json.load(fh)
        rep = j["report"]
        name = j["model"]
        # Com o protocolo v2 presente, os modelos em PyTorch do protocolo anterior viram "(antigo)"
        if files_v2 and j.get("protocol") != "v2" and j["family"] in ("GCN", "DeepSets"):
            name = f"{name} (antigo)"
        rows.append({
            "model": name, "family": j["family"], "seed": j["seed"], "protocol": j.get("protocol", "v1"),
            "representation": j.get("representation"), "variant": j.get("variant"),
            "macro_f1": j["macro_f1"], "accuracy": j["accuracy"],
            "f1_rush": rep["Rush"]["f1-score"], "f1_pass": rep["Pass"]["f1-score"],
            "recall_rush": rep["Rush"]["recall"], "recall_pass": rep["Pass"]["recall"],
            "per_play_ms": j.get("inference", {}).get("per_play_ms", np.nan),
            "device": j.get("inference", {}).get("device"),
            "train_time_s": j.get("train_time_s", np.nan),
            "cm": np.array(j["confusion_matrix"], dtype=float),
        })
        preds[(name, j["seed"])] = (j.get("test_keys"), j.get("test_labels"), j.get("test_preds"))

    # Resultados antigos da GCN (output/results). A divisao usada pela GCN no texto e a
    # mesma reproduzida aqui (ver common.make_split), entao sao pareaveis por semente.
    if old_gcn_dir:
        for strategy in ALL_STRATEGIES:  # ignora first_tests e outras pastas antigas
            for f in sorted(glob.glob(os.path.join(old_gcn_dir, strategy, "*.json"))):
                with open(f) as fh:
                    j = json.load(fh)
                g = j["best_gcn_results"]
                name = f"GCN-{j['config']['EDGE_STRATEGY'].rstrip('-')} (texto)"
                rows.append({
                    "model": name, "family": "GCN", "seed": j["config"]["RANDOM_SEED"], "protocol": "texto",
                    "representation": "graph", "variant": "texto",
                    "macro_f1": g["macro avg"]["f1-score"], "accuracy": g["accuracy"],
                    "f1_rush": g["Rush"]["f1-score"], "f1_pass": g["Pass"]["f1-score"],
                    "recall_rush": g["Rush"]["recall"], "recall_pass": g["Pass"]["recall"],
                    "per_play_ms": np.nan, "device": None, "train_time_s": np.nan,
                    "cm": np.array(g["confusion_matrix"], dtype=float),
                })
    if not rows:
        raise SystemExit(f"No results found in {out_dir}/runs")
    df = pd.DataFrame(rows)
    dup = df.duplicated(["model", "seed"], keep="last")
    if dup.any():
        Logger.warning(f"Dropping {dup.sum()} duplicated (model, seed) rows")
        df = df[~dup]
    return df, preds


def to_markdown(table, index=True):
    """Tabela markdown simples (evita depender do pacote tabulate)"""
    t = table.reset_index() if index else table
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def summary_table(df):
    g = df.groupby("model")
    summ = pd.DataFrame({
        "family": g["family"].first(),
        "n": g["macro_f1"].size(),
        "macro_f1_mean": g["macro_f1"].mean(),
        "macro_f1_std": g["macro_f1"].std(ddof=1),
        "macro_f1_min": g["macro_f1"].min(),
        "macro_f1_max": g["macro_f1"].max(),
        "accuracy_mean": g["accuracy"].mean(),
        "f1_rush_mean": g["f1_rush"].mean(),
        "f1_pass_mean": g["f1_pass"].mean(),
        "recall_rush_mean": g["recall_rush"].mean(),
        "recall_pass_mean": g["recall_pass"].mean(),
        "per_play_ms_mean": g["per_play_ms"].mean(),
        "device": g["device"].first(),
        "train_time_s_mean": g["train_time_s"].mean(),
    })
    return summ.sort_values("macro_f1_mean", ascending=False)


def write_latex(summ, path):
    lines = []
    for model, r in summ.iterrows():
        ms = "--" if np.isnan(r.per_play_ms_mean) else br(r.per_play_ms_mean, 2)
        lines.append(f"        {model.replace('_', chr(92) + '_')} & {br(r.macro_f1_mean)} & {br(r.macro_f1_std)} & "
                     f"{br(r.macro_f1_min)} & {br(r.macro_f1_max)} & {ms} \\\\")
    tex = r"""\begin{table}[htbp]
    \centering
    \caption{Macro F1-Score no conjunto de teste com a mesma divis\~ao de dados para todos os modelos.}
    \label{tab:comparacao_justa}
    \begin{tabular}{lccccc}
        \hline
        Modelo & M\'edia & Desvio-padr\~ao & M\'inimo & M\'aximo & Infer\^encia (ms/jogada) \\
        \hline
%s
        \hline
    \end{tabular}
\end{table}
""" % "\n".join(lines)
    with open(path, "w") as f:
        f.write(tex)


def boxplot(df, summ, path):
    order = list(summ.index[::-1])  # melhor em cima
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(order) + 1.2))
    data = [df[df.model == m].macro_f1.values for m in order]
    try:
        bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False)
    except TypeError:
        bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], order):
        c = FAMILY_COLOR.get(summ.loc[m, "family"], "gray")
        patch.set_facecolor(c); patch.set_alpha(0.5); patch.set_edgecolor(c)
    for med in bp["medians"]:
        med.set_color("black")
    rng = np.random.default_rng(0)
    for i, (v, m) in enumerate(zip(data, order), start=1):
        ax.scatter(v, i + rng.uniform(-0.12, 0.12, len(v)), s=5, alpha=0.7,
                   color=FAMILY_COLOR.get(summ.loc[m, "family"], "gray"), linewidths=0, zorder=3)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)
    ax.set_xlabel("Macro F1-Score no conjunto de teste")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: br(x, 2)))
    ax.grid(axis="x", color="#e4e3df")
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def friedman_nemenyi(df, models, out_dir):
    """Friedman + Nemenyi nas sementes em que todos os modelos foram executados."""
    wide = df[df.model.isin(models)].pivot(index="seed", columns="model", values="macro_f1")
    wide = wide.dropna()
    lines = []
    if wide.shape[0] < 2 or wide.shape[1] < 3:
        lines.append(f"Not enough complete data for Friedman (seeds={wide.shape[0]}, models={wide.shape[1]}).")
        return wide, "\n".join(lines)

    chi2, p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()
    k, n = wide.shape[1], wide.shape[0]
    lines.append(f"Friedman: chi2 = {chi2:.2f}, p = {p:.3g} (k = {k} modelos, N = {n} sementes)")
    lines.append("")
    lines.append("Ranks medios (1 = melhor):")
    for m, r in ranks.items():
        lines.append(f"  {r:6.2f}  {m}")

    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(wide.values)
        nem.index = nem.columns = wide.columns
        nem.to_csv(os.path.join(out_dir, "nemenyi_pvalues.csv"))
        # q_0.05 de Nemenyi = amplitude studentizada (infinitos g.l.) / sqrt(2); ex.: k = 8 -> 3,031
        q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        if q:
            cd = q * np.sqrt(k * (k + 1) / (6 * n))
            lines.append("")
            lines.append(f"Nemenyi: CD = {cd:.3f} (q_0.05 = {q:.3f}, k = {k}, N = {n})")
        lines.append("p-valores de Nemenyi salvos em nemenyi_pvalues.csv")

        if q:
            cd_diagram(ranks, cd, os.path.join(out_dir, "nemenyi_cd"))
    except Exception as e:  # scikit-posthocs ausente ou versao diferente
        lines.append(f"(Nemenyi/CD diagram skipped: {e})")
    return wide, "\n".join(lines)


def cd_diagram(ranks, cd, path):
    """Diagrama de diferenca critica (Demsar, 2006). Modelos ligados por uma barra
    tem diferenca de rank medio menor que a CD (nao diferem pelo teste de Nemenyi).
    Implementacao propria: a do scikit-posthocs trava com muitos modelos."""
    ranks = ranks.sort_values()
    names, r = list(ranks.index), ranks.values
    k = len(r)
    half = (k + 1) // 2
    fig_h = 0.28 * half + 1.6
    fig, ax = plt.subplots(figsize=(10, fig_h))
    lo, hi = 1, max(k, int(np.ceil(r.max())))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-(half + 1.5), 1.6)
    ax.axis("off")
    ax.hlines(0, lo, hi, color="black", lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, 0, 0.15, color="black", lw=1)
        ax.text(t, 0.3, str(t), ha="center", va="bottom", fontsize=8)
    ax.hlines(1.2, lo, lo + cd, color="black", lw=1.5)
    ax.text(lo + cd / 2, 1.3, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (name, x) in enumerate(zip(names, r)):
        left = i < half
        row = (i if left else k - 1 - i) + 1
        y = -row * 0.9
        xt = lo - 0.4 if left else hi + 0.4
        ax.plot([x, x, xt], [0, y, y], color="#52514e", lw=0.8)
        ax.text(xt, y, f"{name} ({x:.2f})", ha="right" if left else "left", va="center", fontsize=8)
    # grupos maximos de modelos consecutivos com diferenca < CD
    groups = []
    for i in range(k):
        j = i
        while j + 1 < k and r[j + 1] - r[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    for g, (a, b) in enumerate(groups):
        y = -0.45 - 0.2 * g
        ax.hlines(y, r[a] - 0.05, r[b] + 0.05, color="black", lw=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def holm(pvals):
    pvals = np.asarray(pvals)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_vs_ref(df, ref, models):
    wide = df[df.model.isin(models)].pivot(index="seed", columns="model", values="macro_f1")
    rows = []
    for m in models:
        if m == ref:
            continue
        pair = wide[[ref, m]].dropna()
        if len(pair) < 5:
            continue
        diff = pair[ref] - pair[m]
        try:
            p = stats.wilcoxon(pair[ref], pair[m]).pvalue
        except ValueError:
            p = 1.0
        rows.append({"modelo": m, "n_sementes": len(pair), "dif_media": diff.mean(), "dif_mediana": diff.median(),
                     "ref_vence": int((diff > 0).sum()), "p_wilcoxon": p})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_holm"] = holm(res["p_wilcoxon"].values)
        res = res.sort_values("dif_media")
    return res


def confusion_matrices(df, models, path):
    models = [m for m in models if m in set(df.model)]
    if not models:
        return
    cols = min(4, len(models))
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, m in zip(axes.flat, models):
        cm = np.mean(np.stack(df[df.model == m].cm.values), axis=0)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.axis("on")
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{br(norm[i, j] * 100, 1)}%\n({cm[i, j]:.0f})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Corrida", "Passe"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Corrida", "Passe"])
        ax.set_xlabel("Predito"); ax.set_ylabel("Real")
        ax.set_title(m, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)


def analyze(args):
    out_dir = os.path.join(args.out, "analise")
    os.makedirs(out_dir, exist_ok=True)
    df, _ = load_runs(args.out, args.old_gcn_dir)
    if args.only:
        df = df[df.model.isin(args.only)]

    summ = summary_table(df)
    summ.to_csv(os.path.join(out_dir, "resumo.csv"), float_format="%.4f")
    write_latex(summ, os.path.join(out_dir, "resumo.tex"))
    boxplot(df, summ, os.path.join(out_dir, "boxplot_macro_f1"))

    # Para os testes estatisticos usamos, por padrao, todas as configuracoes exceto as GCN "(texto)"
    # importadas, que duplicam as GCN reexecutadas quando ambas existem.
    # Testes estatisticos: sem duplicar configuracoes. Se ha resultados do protocolo v2, as GCN/DeepSets
    # do protocolo antigo ("(texto)", "(antigo)") ficam so na tabela; senao, "(texto)" sai apenas quando
    # a mesma topologia foi reexecutada.
    has_v2 = (df.protocol == "v2").any()
    if args.include_old_in_tests or not has_v2:
        test_models = [m for m in summ.index if not (m.endswith("(texto)") and m.replace(" (texto)", "") in summ.index)]
    else:
        test_models = [m for m in summ.index if not m.endswith(OLD_TAGS)]
    wide, fr_text = friedman_nemenyi(df, test_models, out_dir)
    with open(os.path.join(out_dir, "friedman_nemenyi.txt"), "w") as f:
        f.write(fr_text + "\n")

    ref = args.reference or next((m for m in test_models if summ.loc[m, "family"] in ("GCN", "GraphSAGE")), test_models[0])
    wil = wilcoxon_vs_ref(df, ref, test_models)
    wil.to_csv(os.path.join(out_dir, "wilcoxon_vs_ref.csv"), index=False, float_format="%.5f")

    # Matrizes de confusao: melhor de cada familia + configuracoes "texto"
    cm_models = []
    for fam in ["GCN", "GraphSAGE", "DeepSets", "RF", "MLP"]:
        cm_models += list(summ[summ.family == fam].index[:2])
    confusion_matrices(df, cm_models, os.path.join(out_dir, "matrizes_confusao"))

    # Relatorio
    show = summ[["n", "macro_f1_mean", "macro_f1_std", "macro_f1_min", "macro_f1_max", "accuracy_mean",
                 "f1_rush_mean", "f1_pass_mean", "per_play_ms_mean", "device", "train_time_s_mean"]]
    md = ["# Comparacao justa - resultados", "",
          f"Execucoes lidas de `{args.out}/runs`" + (f" + GCN antigas de `{args.old_gcn_dir}`" if args.old_gcn_dir else ""), "",
          "## Resumo (Macro F1 no teste)", "", to_markdown(show), "",
          "## Friedman / Nemenyi", "", "```", fr_text, "```", "",
          f"## Wilcoxon pareado por semente vs `{ref}` (correcao de Holm)", "",
          to_markdown(wil, index=False) if not wil.empty else "(sem pares suficientes)", ""]
    with open(os.path.join(out_dir, "relatorio.md"), "w") as f:
        f.write("\n".join(md))

    pd.set_option("display.width", 200)
    print(show.to_string(float_format=lambda x: f"{x:.4f}"))
    print()
    print(fr_text)
    print()
    if not wil.empty:
        print(f"Wilcoxon vs {ref}:")
        print(wil.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    Logger.info(f"Analysis saved to {out_dir}")
