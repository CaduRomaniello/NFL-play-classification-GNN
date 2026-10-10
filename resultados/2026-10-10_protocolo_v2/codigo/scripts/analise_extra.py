"""Analise complementar dos resultados v2: testes pareados das ablacoes, topologias,
hiperparametros escolhidos, tempo de inferencia e analise de erros por tipo de jogada.
Gera figuras (PNG/PDF) e tabelas (CSV/JSON) em output/comparacao_justa/analise/extra/.

Uso: python analise_extra.py <raiz com output/ e gnn_params.json> [caminho do plays.csv]"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = sys.argv[1]
OUT = os.path.join(ROOT, "output/comparacao_justa/analise/extra")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 10, "savefig.bbox": "tight", "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
COLOR = {"GCN": "#2a78d6", "GraphSAGE": "#4a3aa7", "DeepSets": "#e87ba4", "RF": "#eda100", "MLP": "#1baf7a",
         "texto": "#9a9890"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def br(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------- carga
rows, preds = [], {}
for f in sorted(glob.glob(os.path.join(ROOT, "output/comparacao_justa/runs*/*/seed_*.json"))):
    j = json.load(open(f))
    v2 = "runs_v2" in f
    name = j["model"] if (v2 or j["family"] not in ("GCN", "DeepSets")) else j["model"] + " (antigo)"
    rows.append({"model": name, "family": j["family"], "seed": j["seed"], "macro_f1": j["macro_f1"],
                 "f1_rush": j["report"]["Rush"]["f1-score"] if "report" in j else np.nan,
                 "per_play_ms": j.get("inference", {}).get("per_play_ms", np.nan),
                 "device": j.get("inference", {}).get("device"), "params": j.get("params"),
                 "best_epoch": j.get("best_epoch"), "stopped_epoch": j.get("stopped_epoch"),
                 "n_parameters": j.get("n_parameters")})
    if j.get("test_keys"):
        preds[(name, j["seed"])] = (j["test_keys"], j["test_labels"], j["test_preds"])
OLD = ["MST", "RNG", "CLOSEST-", "GABRIEL", "QB-CLOSEST-", "DELAUNAY"]  # sem output/results/first_tests
for f in sorted(f for s_ in OLD for f in glob.glob(os.path.join(ROOT, "output/results", s_, "*.json"))):
    j = json.load(open(f))
    g = j["best_gcn_results"]
    rows.append({"model": f"GCN-{j['config']['EDGE_STRATEGY'].rstrip('-')} (texto)", "family": "GCN",
                 "seed": j["config"]["RANDOM_SEED"], "macro_f1": g["macro avg"]["f1-score"],
                 "f1_rush": g["Rush"]["f1-score"]})
df = pd.DataFrame(rows)
wide = df.pivot_table(index="seed", columns="model", values="macro_f1")
summ = df.groupby("model").macro_f1.agg(["mean", "std", "count"])
fam = df.groupby("model").family.first()


# ---------------------------------------------------------------- testes pareados
def paired(a, b, label):
    d = (wide[a] - wide[b]).dropna()
    p = wilcoxon(d).pvalue
    return {"comparacao": label, "A": a, "B": b, "media_A": wide[a].mean(), "media_B": wide[b].mean(),
            "dif_media": d.mean(), "A_vence": int((d > 0).sum()), "n": len(d), "p_wilcoxon": p}


TOPOS = ["MST", "RNG", "CLOSEST", "GABRIEL", "QB-CLOSEST", "DELAUNAY"]
tests = [
    paired("GCN-MST", "GCN-MST (texto)", "Efeito das correcoes na GCN (MST)"),
    paired("GCN-MST", "DeepSets-matched", "Efeito das arestas na GCN (mesmos hiperparametros)"),
    paired("SAGE-RNG", "DeepSets-tuned", "Melhor GraphSAGE x DeepSets otimizado"),
    paired("SAGE-RNG", "GCN-RNG", "GraphSAGE x GCN (RNG)"),
    paired("DeepSets-tuned", "DeepSets-tuned (antigo)", "DeepSets v2 x protocolo anterior"),
    paired("DeepSets-tuned", "MLP-concat_xy-tuned", "DeepSets x melhor baseline estritamente justa"),
    paired("DeepSets-tuned", "MLP-concat_team_role-tuned", "DeepSets x melhor baseline com lado de campo"),
    paired("GCN-RNG", "MLP-concat_team_role-tuned", "Melhor GCN x MLP-concat_team_role"),
    paired("MLP-concat_xy-tuned", "MLP-mean-default", "MLP: concatenacao ordenada x media (texto)"),
]
for t in TOPOS:
    tests.append(paired(f"GCN-{t}", f"GCN-{t} (texto)", f"Correcoes: GCN-{t}"))
    tests.append(paired(f"SAGE-{t}", f"GCN-{t}", f"SAGE x GCN: {t}"))
    tests.append(paired(f"SAGE-{t}", "DeepSets-tuned", f"SAGE x DeepSets-tuned: {t}"))
tests = pd.DataFrame(tests)
# Holm dentro da familia de testes
order = tests.p_wilcoxon.argsort().values
m = len(tests)
holm = np.empty(m)
running = 0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * tests.p_wilcoxon.iloc[i]))
    holm[i] = running
tests["p_holm"] = holm
tests.to_csv(os.path.join(OUT, "testes_pareados.csv"), index=False, float_format="%.6g")

# ---------------------------------------------------------------- hiperparametros escolhidos
gp = json.load(open(os.path.join(ROOT, "gnn_params.json")))
hp = []
for grp, label in (("GCN", "GCN"), ("SAGE", "GraphSAGE")):
    for topo, p in gp[grp].items():
        hp.append({"modelo": f"{label}-{topo.rstrip('-')}", **{k: p[k] for k in p if k != "CONV"},
                   "val_f1": gp["_tuning_info"][f"{grp}-{topo}"]["best_val_macro_f1"]})
p = gp["DEEPSETS"]["tuned"]
hp.append({"modelo": "DeepSets-tuned", **{k: p[k] for k in p if k != "CONV"},
           "val_f1": gp["_tuning_info"]["DeepSets"]["best_val_macro_f1"]})
hp = pd.DataFrame(hp)
hp.to_csv(os.path.join(OUT, "hiperparametros_v2.csv"), index=False, float_format="%.6g")

# epocas e parametros
ep = df[df.best_epoch.notna()].groupby("model").agg(best_epoch=("best_epoch", "median"),
                                                    stopped_epoch=("stopped_epoch", "median"),
                                                    n_parameters=("n_parameters", "first"))
ep.to_csv(os.path.join(OUT, "epocas_parametros.csv"))

# ---------------------------------------------------------------- tempo de inferencia
inf = df[df.per_play_ms.notna()].groupby("model").agg(per_play_ms=("per_play_ms", "median"),
                                                      device=("device", "first"))
inf.to_csv(os.path.join(OUT, "inferencia.csv"), float_format="%.4f")


# ---------------------------------------------------------------- figura 1: progressao
def dotplot(ax, models, labels, colors):
    y = np.arange(len(models))[::-1]
    for yi, mname, c in zip(y, models, colors):
        vals = wide[mname].dropna().values
        jit = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.36
        ax.scatter(vals, yi + jit, s=9, color=c, alpha=0.45, linewidths=0, zorder=2)
        mu, sd = vals.mean(), vals.std(ddof=1)
        ax.plot([mu - sd, mu + sd], [yi, yi], color=INK, lw=1.4, zorder=3)
        ax.scatter([mu], [yi], s=46, color=c, edgecolors="white", linewidths=1.6, zorder=4)
        ax.text(mu + sd + 0.003, yi, br(mu), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


prog = [
    ("RF-mean-default", "RF, média dos jogadores (texto)", "RF"),
    ("MLP-mean-default", "MLP, média dos jogadores (texto)", "MLP"),
    ("MLP-concat_xy-tuned", "MLP, concatenação ordenada por x,y", "MLP"),
    ("MLP-concat_team_role-tuned", "MLP, ordenada por lado e função*", "MLP"),
    ("GCN-MST (texto)", "GCN-MST do texto", "texto"),
    ("GCN-MST", "GCN-MST corrigida (v2)", "GCN"),
    ("DeepSets-matched", "GCN-MST v2 sem arestas", "DeepSets"),
    ("SAGE-MST", "GraphSAGE-MST (v2)", "GraphSAGE"),
    ("SAGE-RNG", "GraphSAGE-RNG (v2), melhor GNN", "GraphSAGE"),
    ("DeepSets-tuned", "DeepSets otimizado (v2)", "DeepSets"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
dotplot(ax, [p[0] for p in prog], [p[1] for p in prog], [COLOR[p[2]] for p in prog])
ax.set_xlabel("Macro F1 no teste (29 sementes; ponto = média, barra = ±1 desvio-padrão)")
ax.set_xlim(0.65, 0.875)
fig.text(0.01, -0.02, "* usa o lado de campo (ataque/defesa) na ordenação, informação que a GNN não recebe.",
         fontsize=7.5, color=INK2)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig1_progressao.{ext}"))
plt.close(fig)

# ---------------------------------------------------------------- figura 2: topologias
fig, ax = plt.subplots(figsize=(7.6, 3.6))
x = np.arange(len(TOPOS))
for k, (suffix, label, c, off) in enumerate([(" (texto)", "GCN do texto", COLOR["texto"], -0.22),
                                             ("", "GCN v2", COLOR["GCN"], 0.0),
                                             ("SAGE", "GraphSAGE v2", COLOR["GraphSAGE"], 0.22)]):
    names = [f"SAGE-{t}" if suffix == "SAGE" else f"GCN-{t}{suffix}" for t in TOPOS]
    mu = np.array([summ.loc[n, "mean"] for n in names])
    sd = np.array([summ.loc[n, "std"] for n in names])
    ax.errorbar(x + off, mu, yerr=sd, fmt="o", ms=6.5, color=c, ecolor=c, elinewidth=1.4, capsize=0,
                markeredgecolor="white", markeredgewidth=1.2, label=label, zorder=3)
ds = summ.loc["DeepSets-tuned"]
ax.axhspan(ds["mean"] - ds["std"], ds["mean"] + ds["std"], color=COLOR["DeepSets"], alpha=0.13, lw=0, zorder=1)
ax.axhline(ds["mean"], color=COLOR["DeepSets"], lw=1.4, ls="--", zorder=2)
ax.text(-0.4, ds["mean"] + ds["std"] + 0.001, "faixa: DeepSets otimizado (sem arestas), média ± desvio", ha="left", va="bottom",
        fontsize=8, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(TOPOS, fontsize=8.5)
ax.set_ylabel("Macro F1 no teste")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig2_topologias.{ext}"))
plt.close(fig)

# ---------------------------------------------------------------- figura 3: diferencas pareadas
pairs = [("GCN-MST", "GCN-MST (texto)", "GCN-MST v2 − GCN-MST do texto\n(efeito das correções)", COLOR["GCN"]),
         ("GCN-MST", "DeepSets-matched", "GCN-MST v2 − mesma rede sem arestas\n(efeito das arestas na GCN)", COLOR["GCN"]),
         ("SAGE-RNG", "GCN-RNG", "GraphSAGE-RNG − GCN-RNG\n(efeito do tipo de agregação)", COLOR["GraphSAGE"]),
         ("SAGE-RNG", "DeepSets-tuned", "GraphSAGE-RNG − DeepSets otimizado\n(arestas, ambos otimizados)", COLOR["GraphSAGE"])]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, (a, b, label, c) in enumerate(pairs):
    d = (wide[a] - wide[b]).dropna().values
    yi = len(pairs) - 1 - i
    jit = (np.random.default_rng(1).random(len(d)) - 0.5) * 0.34
    ax.scatter(d, yi + jit, s=12, color=c, alpha=0.55, linewidths=0, zorder=2)
    ax.plot([np.percentile(d, 25), np.percentile(d, 75)], [yi, yi], color=INK, lw=2.2, zorder=3)
    ax.scatter([np.median(d)], [yi], s=46, color=c, edgecolors="white", linewidths=1.6, zorder=4)
    wins = int((d > 0).sum())
    ax.text(0.0445, yi, f"A vence em {wins}/{len(d)}", va="center", fontsize=8.5, color=INK2)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(pairs))[::-1])
ax.set_yticklabels([p[2] for p in pairs], fontsize=8.5)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.027, 0.058)
ax.set_xlabel("Diferença de Macro F1 por semente (A − B)\nponto grande = mediana; barra = intervalo interquartil")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig3_diferencas_pareadas.{ext}"))
plt.close(fig)

# ---------------------------------------------------------------- analise de erros
PLAYS_CSV = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data/raw/plays.csv")  # dados do Big Data Bowl (nao versionados)
plays = pd.read_csv(PLAYS_CSV,
                    usecols=["gameId", "playId", "down", "yardsToGo", "quarter", "offenseFormation",
                             "playAction", "pff_runPassOption", "dropbackType", "rushLocationType",
                             "receiverAlignment", "gameClock", "preSnapHomeScore", "preSnapVisitorScore"])
plays = plays.set_index(["gameId", "playId"])
ERR_MODELS = ["DeepSets-tuned", "SAGE-RNG", "GCN-MST", "MLP-concat_xy-tuned", "RF-mean-default"]
recs = []
for mname in ERR_MODELS:
    for seed in range(1, 30):
        keys, labels, pr = preds[(mname, seed)]
        for (g, pid), yt, yp in zip(keys, labels, pr):
            recs.append((mname, seed, g, pid, yt, yp))
er = pd.DataFrame(recs, columns=["model", "seed", "gameId", "playId", "label", "pred"])
er["correct"] = (er.label == er.pred).astype(int)
er = er.join(plays, on=["gameId", "playId"])


def bucket_ytg(v):
    return "1–2" if v <= 2 else ("3–6" if v <= 6 else ("7–10" if v <= 10 else "11+"))


er["ytg"] = er.yardsToGo.apply(bucket_ytg)
er["classe"] = np.where(er.label == 1, "Passe", "Corrida")
er["tipo"] = np.select(
    [(er.label == 1) & (er.playAction == True), (er.label == 1), (er.label == 0)],
    ["Passe com play action", "Passe sem play action", "Corrida"], "")
er["rpo"] = np.where(er.pff_runPassOption == 1, "RPO", "Não RPO")


def acc_by(col, models=ERR_MODELS, min_n=150):
    t = er[er.model.isin(models)].groupby([col, "model"]).correct.agg(["mean", "count"]).reset_index()
    n = er[er.model == models[0]].groupby(col).size()
    t = t[t[col].isin(n[n >= min_n].index)]
    return t.pivot(index=col, columns="model", values="mean")[models], n


tables = {}
for col in ["tipo", "rpo", "offenseFormation", "down", "ytg", "classe"]:
    t, n = acc_by(col)
    t["n_previsoes"] = n.reindex(t.index)
    tables[col] = t
    t.to_csv(os.path.join(OUT, f"acuracia_por_{col}.csv"), float_format="%.4f")

# dificuldade por jogada (modelo mais forte), agregando sementes onde a jogada caiu no teste
best = er[er.model == "DeepSets-tuned"].groupby(["gameId", "playId"]).agg(
    acc=("correct", "mean"), n=("correct", "size"), label=("label", "first"),
    playAction=("playAction", "first"), rpo=("rpo", "first"), formation=("offenseFormation", "first"))
all5 = er.groupby(["gameId", "playId"]).correct.mean()
best["acc_todos"] = all5.reindex(best.index)
diff = {
    "jogadas_unicas": int(len(best)),
    "sempre_acertadas_deepsets": float((best.acc == 1).mean()),
    "sempre_erradas_deepsets": float((best.acc == 0).mean()),
    "sempre_erradas_todos_5": float((best.acc_todos == 0).mean()),
    "sempre_acertadas_todos_5": float((best.acc_todos == 1).mean()),
}
hard = best[best.acc_todos == 0]
diff["dificeis_pct_passe"] = float((hard.label == 1).mean())
diff["dificeis_pct_play_action"] = float(((hard.label == 1) & (hard.playAction == True)).mean())
diff["dificeis_pct_rpo"] = float((hard.rpo == "RPO").mean())
diff["geral_pct_passe"] = float((best.label == 1).mean())
diff["geral_pct_play_action"] = float(((best.label == 1) & (best.playAction == True)).mean())
diff["geral_pct_rpo"] = float((best.rpo == "RPO").mean())
hard_form = hard.formation.value_counts(normalize=True).head(5).to_dict()
all_form = best.formation.value_counts(normalize=True).to_dict()
diff["dificeis_formacoes"] = {k: [v, all_form.get(k)] for k, v in hard_form.items()}
json.dump(diff, open(os.path.join(OUT, "dificuldade.json"), "w"), indent=2, ensure_ascii=False)

# figura 4: acuracia por tipo de jogada e por formacao
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
short = {"DeepSets-tuned": ("DeepSets", COLOR["DeepSets"]), "SAGE-RNG": ("GraphSAGE-RNG", COLOR["GraphSAGE"]),
         "GCN-MST": ("GCN-MST v2", COLOR["GCN"]), "MLP-concat_xy-tuned": ("MLP-concat_xy", COLOR["MLP"]),
         "RF-mean-default": ("RF-média (texto)", COLOR["RF"])}
for ax, col, title, order in [
        (axes[0], "tipo", "Por tipo de jogada", ["Corrida", "Passe sem play action", "Passe com play action"]),
        (axes[1], "offenseFormation", "Por formação ofensiva", None)]:
    t = tables[col].drop(columns="n_previsoes")
    if order is None:
        order = list(tables[col].sort_values("n_previsoes", ascending=False).index)
    t = t.loc[order]
    y = np.arange(len(order))[::-1]
    for k, mname in enumerate(ERR_MODELS):
        lab, c = short[mname]
        ax.scatter(t[mname].values, y + (k - 2) * 0.13, s=26, color=c, edgecolors="white", linewidths=0.8,
                   label=lab, zorder=3)
    nlab = tables[col]["n_previsoes"].loc[order]
    ax.set_yticks(y)
    ax.set_yticklabels([f"{o}\n(n={int(n_)})" for o, n_ in zip(order, nlab)], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel("Taxa de acerto")
axes[0].set_xlim(0.35, 1.0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.99), fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"fig4_erros_por_tipo.{ext}"))
plt.close(fig)

# ---------------------------------------------------------------- saida resumida
pd.set_option("display.width", 200)
print(tests[["comparacao", "media_A", "media_B", "dif_media", "A_vence", "n", "p_holm"]].to_string(index=False))
print()
print(hp.to_string(index=False))
print()
print(ep.to_string())
print()
print(inf.to_string())
print()
for k, t in tables.items():
    print(t.round(3).to_string())
    print()
print(json.dumps(diff, indent=1, ensure_ascii=False))
