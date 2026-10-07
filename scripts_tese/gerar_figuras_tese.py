"""
Gera as tabelas (LaTeX) e figuras usadas na versao final da dissertacao
a partir dos resultados salvos em output/results.

Uso (na raiz do repositorio):
    python scripts_tese/gerar_figuras_tese.py
    python scripts_tese/gerar_figuras_tese.py --results output/results --out output/tese_final

Saidas em --out:
    tab_resultados_f1.tex        Tabela 4.1 (media, desvio-padrao, minimo e maximo do Macro F1)
    tab_metricas_por_classe.tex  Recall e F1 por classe (GCN-MST, RF, MLP)
    tab_espaco_busca.tex         Tabela 3.3 (espaco de busca do Optuna)
    fig_boxplot_macro_f1.pdf/png Distribuicao do Macro F1 por configuracao
    fig_matrizes_confusao.pdf/png Matrizes de confusao medias no teste
    resumo_estatisticas.csv      Numeros usados nas tabelas
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ESTRATEGIAS = ["MST", "RNG", "CLOSEST-", "GABRIEL", "QB-CLOSEST-", "DELAUNAY"]
NOMES = {"MST": "MST", "RNG": "RNG", "CLOSEST-": "CLOSEST", "GABRIEL": "GABRIEL",
         "QB-CLOSEST-": "QB-CLOSEST", "DELAUNAY": "DELAUNAY"}

# Cores (paleta validada para daltonismo): GCN azul, RF laranja, MLP verde-agua
COR = {"GCN": "#2a78d6", "RF": "#eb6834", "MLP": "#1baf7a"}
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.edgecolor": TINTA_2,
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def br(x, casas=4):
    """Numero com virgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def carregar(results_dir):
    """Le os JSONs (ignora first_tests). Retorna lista de dicts por execucao."""
    linhas = []
    for estr in ESTRATEGIAS:
        for f in sorted(glob.glob(os.path.join(results_dir, estr, "*.json"))):
            with open(f) as fh:
                j = json.load(fh)
            linhas.append({
                "estrategia": estr,
                "seed": j["config"]["RANDOM_SEED"],
                "gcn": j["best_gcn_results"],
                "rf": j["rf_results"],
                "mlp": j["mlp_results"],
            })
    if not linhas:
        raise SystemExit(f"Nenhum resultado encontrado em {results_dir}")
    return linhas


def f1(res):
    return res["macro avg"]["f1-score"]


def series_f1(linhas):
    """Macro F1 por configuracao. GCN: 29 execucoes por topologia.
    RF: uma execucao por semente (o RF da o mesmo resultado nas 6 topologias).
    MLP: todas as execucoes (29 sementes x 6 topologias)."""
    s = {}
    for estr in ESTRATEGIAS:
        s[f"GCN - {NOMES[estr]}"] = np.array([f1(l["gcn"]) for l in linhas if l["estrategia"] == estr])
    s["RF"] = np.array([f1(l["rf"]) for l in linhas if l["estrategia"] == ESTRATEGIAS[0]])
    s["MLP"] = np.array([f1(l["mlp"]) for l in linhas])
    return s


def tabela_resultados(s, out):
    ordem = sorted(s, key=lambda k: -s[k].mean())
    linhas_tex = []
    csv = ["configuracao;n;media;desvio;minimo;maximo"]
    for k in ordem:
        v = s[k]
        linhas_tex.append(f"        {k} & {br(v.mean())} & {br(v.std(ddof=1))} & {br(v.min())} & {br(v.max())} \\\\")
        csv.append(f"{k};{len(v)};{v.mean():.4f};{v.std(ddof=1):.4f};{v.min():.4f};{v.max():.4f}")
    n_gcn = len(s["GCN - MST"])
    tex = r"""\begin{table}[htbp]
    \centering
    \caption{Macro F1-Score no conjunto de teste por modelo e topologia de grafo (GCN e RF: %d execu\c{c}\~oes; MLP: %d execu\c{c}\~oes, somando as seis topologias).}
    \label{tab:resultados_f1}
    \begin{tabular}{lcccc}
        \hline
        Modelo / topologia & M\'edia & Desvio-padr\~ao & M\'inimo & M\'aximo \\
        \hline
%s
        \hline
    \end{tabular}
\end{table}
""" % (n_gcn, len(s["MLP"]), "\n".join(linhas_tex))
    with open(os.path.join(out, "tab_resultados_f1.tex"), "w") as fh:
        fh.write(tex)
    with open(os.path.join(out, "resumo_estatisticas.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")


def tabela_por_classe(linhas, out, estr="MST"):
    sel = [l for l in linhas if l["estrategia"] == estr]
    rows = []
    for nome, chave in [(f"GCN - {NOMES[estr]}", "gcn"), ("RF", "rf"), ("MLP", "mlp")]:
        vals = []
        for classe in ["Rush", "Pass"]:
            for met in ["recall", "f1-score"]:
                vals.append(np.mean([l[chave][classe][met] for l in sel]))
        rows.append(f"        {nome} & " + " & ".join(br(v, 3) for v in vals) + r" \\")
    tex = r"""\begin{table}[htbp]
    \centering
    \caption{Recall e F1-Score m\'edios por classe no conjunto de teste (%d execu\c{c}\~oes, topologia %s para a GCN).}
    \label{tab:metricas_por_classe}
    \begin{tabular}{lcccc}
        \hline
        & \multicolumn{2}{c}{Corrida} & \multicolumn{2}{c}{Passe} \\
        Modelo & Recall & F1-Score & Recall & F1-Score \\
        \hline
%s
        \hline
    \end{tabular}
\end{table}
""" % (len(sel), NOMES[estr], "\n".join(rows))
    with open(os.path.join(out, "tab_metricas_por_classe.tex"), "w") as fh:
        fh.write(tex)


def tabela_espaco_busca(out):
    tex = r"""\begin{table}[htbp]
    \centering
    \caption{Espa\c{c}o de busca de hiperpar\^ametros da GCN no Optuna.}
    \label{tab:espaco_busca}
    \begin{tabular}{ll}
        \hline
        Hiperpar\^ametro & Valores \\
        \hline
        Canais ocultos & \{128, 256\} \\
        Camadas ocultas & \{1, 2, 3\} \\
        Taxa de aprendizado & $10^{-3}$ a $10^{-1}$ (escala logar\'itmica) \\
        Dropout & 0,1 a 0,5 \\
        \textit{Weight decay} & $10^{-6}$ a $10^{-3}$ (escala logar\'itmica) \\
        \'Epocas de \textit{warmup} & \{10, 20, 30\} \\
        Subamostragem (\textit{down sample}) & \{sim, n\~ao\} \\
        Liga\c{c}\~ao do QB (\textit{qb link}) & \{sim, n\~ao\} \\
        \hline
    \end{tabular}
\end{table}
"""
    with open(os.path.join(out, "tab_espaco_busca.tex"), "w") as fh:
        fh.write(tex)


def figura_boxplot(s, out):
    ordem = sorted(s, key=lambda k: np.median(s[k]))  # de baixo para cima
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    dados = [s[k] for k in ordem]
    cores = [COR["GCN"] if k.startswith("GCN") else COR[k] for k in ordem]
    kw = dict(widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=TINTA, linewidth=1.6),
                    whiskerprops=dict(color=TINTA_2, linewidth=1),
                    capprops=dict(color=TINTA_2, linewidth=1),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="white",
                                    markeredgecolor=TINTA_2, linewidth=0.8))
    try:  # matplotlib >= 3.10
        bp = ax.boxplot(dados, orientation="horizontal", **kw)
    except TypeError:  # versoes anteriores
        bp = ax.boxplot(dados, vert=False, **kw)
    for patch, c in zip(bp["boxes"], cores):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
        patch.set_linewidth(1.2)
    # pontos individuais (jitter) por cima, para mostrar as execucoes
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate(zip(dados, cores), start=1):
        y = i + rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(v, y, s=6, color=c, alpha=0.7, linewidths=0, zorder=3)
    ax.set_yticks(range(1, len(ordem) + 1))
    ax.set_yticklabels(ordem)
    ax.set_xlabel("Macro F1-Score no conjunto de teste")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: br(x, 2)))
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_boxplot_macro_f1.{ext}"))
    plt.close(fig)


def figura_matrizes(linhas, out, estr="MST"):
    sel = [l for l in linhas if l["estrategia"] == estr]
    modelos = [(f"GCN - {NOMES[estr]}", "gcn"), ("Random Forest", "rf"), ("MLP", "mlp")]
    classes = ["Corrida", "Passe"]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("azul", ["#f2f6fc", "#2a78d6", "#123f78"])
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
    for ax, (nome, chave) in zip(axes, modelos):
        cm = np.mean([np.array(l[chave]["confusion_matrix"], dtype=float) for l in sel], axis=0)
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                cor_txt = "white" if pct[i, j] > 0.55 else TINTA
                ax.text(j, i, f"{br(pct[i, j] * 100, 1)}%\n({cm[i, j]:.0f})",
                        ha="center", va="center", color=cor_txt, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
        ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
        ax.set_xlabel("Classe prevista")
        ax.set_title(nome, fontsize=12, color=TINTA)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Classe real")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_matrizes_confusao.{ext}"))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="output/results")
    p.add_argument("--out", default="output/tese_final")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    linhas = carregar(a.results)
    s = series_f1(linhas)
    tabela_resultados(s, a.out)
    tabela_por_classe(linhas, a.out)
    tabela_espaco_busca(a.out)
    figura_boxplot(s, a.out)
    figura_matrizes(linhas, a.out)
    print(f"{len(linhas)} execucoes lidas. Arquivos gerados em {a.out}")


if __name__ == "__main__":
    main()
