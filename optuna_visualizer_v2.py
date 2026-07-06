import json
import os
import math

from optuna.importance import FanovaImportanceEvaluator

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
# STORAGE = "sqlite:///optuna/fourth/optuna_study.db"
STORAGE = "sqlite:///zzz_first/optuna_study.db"

# --- Variável para controle global do tamanho da fonte ---
# Aumentado para 24 para legibilidade no artigo
FONT_SIZE = 24 


def load_studies():
    """Carrega todos os estudos de estratégias de aresta do banco de dados"""
    studies = {}
    for strategy in EDGE_STRATEGIES:
        try:
            study = optuna.load_study(
                study_name=f"gcn_optimization_{strategy}",
                storage=STORAGE,
            )
            studies[strategy] = study
            print(f"Carregado {strategy}: {len(study.trials)} tentativas, melhor={study.best_value:.4f}")
        except Exception as e:
            print(f"Não foi possível carregar {strategy}: {e}")
    return studies


def plot_combined_slice(studies):
    """Gráfico de dispersão de cada parâmetro vs acurácia, combinando todas as estratégias"""
    import pandas as pd

    # Coleta todas as tentativas em uma lista plana
    rows = []
    for strategy, study in studies.items():
        for trial in study.trials:
            if trial.value is None:
                continue
            row = {"strategy": strategy, "accuracy": trial.value}
            row.update(trial.params)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Todas as colunas de hiperparâmetros (exclui metadados)
    param_cols = [c for c in df.columns if c not in ("strategy", "accuracy")]

    n = len(param_cols)
    cols = 3
    rows_count = (n + cols - 1) // cols

    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows_count, cols=cols, subplot_titles=param_cols)

    for i, param in enumerate(param_cols):
        row = i // cols + 1
        col = i % cols + 1
        sub_df = df[["strategy", "accuracy", param]].dropna()

        for strategy in sub_df["strategy"].unique():
            s = sub_df[sub_df["strategy"] == strategy]
            fig.add_trace(
                go.Scatter(
                    x=s[param],
                    y=s["accuracy"],
                    mode="markers",
                    name=strategy,
                    showlegend=(i == 0),  # mostra a legenda apenas uma vez
                    opacity=0.6,
                    marker=dict(size=6),
                ),
                row=row, col=col,
            )

        fig.update_xaxes(title_text=param, row=row, col=col)
        fig.update_yaxes(title_text="Acurácia", row=row, col=col)

    fig.update_layout(
        title="Parâmetro vs Acurácia — Todas as Estratégias Combinadas",
        height=350 * rows_count,
        font=dict(size=FONT_SIZE)
    )
    
    # Atualiza a fonte dos subtítulos de cada subplot
    for annotation in fig['layout']['annotations']: 
        annotation['font'] = dict(size=FONT_SIZE)
        
    fig.show()


def plot_overall_optuna_details(studies):
    """Cria um estudo global unificado para avaliar parâmetros independente da estratégia."""
    print("\n--- Gerando Análise Geral de Parâmetros ---")
    
    # 1. Cria um estudo global temporário na memória
    overall_study = optuna.create_study(direction="maximize")
    
    # 2. Coleta todas as execuções completas (trials) de todos os estudos
    all_trials = []
    for strategy, study in studies.items():
        for trial in study.trials:
            # Pega apenas os trials que finalizaram com sucesso (possuem valor)
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append(trial)
    
    # 3. Injeta todos os trials no estudo global
    if not all_trials:
        print("Nenhuma tentativa completa encontrada para análise geral.")
        return
        
    overall_study.add_trials(all_trials)

    print(f"Estudo global criado com sucesso contendo {len(overall_study.trials)} tentativas.")

    # --- Gráficos Globais ---
    
    # A. Importância Geral dos Parâmetros
    try:
        # Criamos o avaliador com uma seed fixa (ex: 42) para garantir a reprodutibilidade
        evaluator = FanovaImportanceEvaluator(seed=42)
        
        # Passamos o avaliador personalizado para a função de plotagem
        fig = plot_param_importances(overall_study, evaluator=evaluator)
        
        fig.update_layout(title="Importância Geral dos Parâmetros (Todas as Estratégias)", font=dict(size=FONT_SIZE))
        fig.show()
    except Exception as e:
        print(f"Erro ao gerar importância geral dos parâmetros: {e}")

    # B. Coordenadas Paralelas Geral (com redução de ticks e correção de rotação)
    try:
        fig = plot_parallel_coordinate(overall_study)
        
        # Ajusta título, fonte e adiciona margens para que os rótulos não sejam cortados
        fig.update_layout(
            title="Coordenadas Paralelas Geral (Todas as Estratégias)", 
            font=dict(size=FONT_SIZE),
            margin=dict(t=100, b=80, l=60, r=60) # Aumenta margem superior/inferior
        )
        
        # Força o ângulo dos rótulos para 0 (horizontal)
        fig.update_traces(labelangle=0)
        
        # Forma segura de reduzir a quantidade de ticks no eixo Y (dimensões do parcoords)
        if fig.data and fig.data[0].type == 'parcoords':
            for dim in fig.data[0].dimensions:
                # Verifica se a dimensão tem tickvals configurados
                if getattr(dim, 'tickvals', None) is not None:
                    tickvals = list(dim.tickvals)
                    ticktext = list(getattr(dim, 'ticktext', None) or tickvals)
                    
                    n_ticks = len(tickvals)
                    max_ticks = 5  # Limite máximo de marcações desejadas no eixo
                    
                    if n_ticks > max_ticks:
                        step = math.ceil(n_ticks / max_ticks)
                        # Mantém valores intercalados, sempre incluindo o primeiro e o último
                        indices = list(range(0, n_ticks, step))
                        if (n_ticks - 1) not in indices:
                            indices.append(n_ticks - 1)
                            
                        # Atualiza a dimensão in-place usando .update()
                        dim.update(
                            tickvals=[tickvals[i] for i in indices],
                            ticktext=[ticktext[i] for i in indices]
                        )

        fig.show()
    except Exception as e:
        print(f"Erro ao gerar coordenadas paralelas geral: {e}")

    # C. Fatias (Slice) Geral - Dividido em 3 gráficos (3, 3 e 2 parâmetros)
    try:
        # Extrair todos os nomes de parâmetros únicos
        all_params = set()
        for t in overall_study.trials:
            all_params.update(t.params.keys())
        all_params = sorted(list(all_params))
        
        # Fatiar a lista de parâmetros de 3 em 3
        chunk_size = 3
        chunks = [all_params[i:i + chunk_size] for i in range(0, len(all_params), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            fig = plot_slice(overall_study, params=chunk)
            fig.update_layout(
                title=f"Gráfico de Fatias Geral - Parte {i+1} (Todas as Estratégias)", 
                font=dict(size=FONT_SIZE)
            )
            # Ajuste extra para a fonte do subtítulo do Plotly
            for annotation in fig['layout']['annotations']: 
                annotation['font'] = dict(size=FONT_SIZE)
            fig.show()
            
    except Exception as e:
        print(f"Erro ao gerar gráfico de fatias geral: {e}")

def plot_best_accuracy_comparison(studies):
    """Gráfico de barras comparando a melhor acurácia entre as estratégias de arestas"""
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
            textfont=dict(size=FONT_SIZE)
        )
    ])
    fig.update_layout(
        title="Melhor Acurácia por Estratégia de Aresta",
        xaxis_title="Estratégia de Aresta",
        yaxis_title="Melhor Acurácia",
        yaxis_range=[min(accuracies) - 0.02, max(accuracies) + 0.02] if accuracies else None,
        font=dict(size=FONT_SIZE)
    )
    fig.show()


def plot_optimization_history_all(studies):
    """Sobrepõe o histórico de otimização de todas as estratégias"""
    fig = go.Figure()
    for strategy, study in studies.items():
        trials = [t for t in study.trials if t.value is not None]
        trials.sort(key=lambda t: t.number)
        values = [t.value for t in trials]

        # Melhor valor contínuo
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
        title="Histórico de Otimização (Melhor Atual) - Todas as Estratégias",
        xaxis_title="Tentativa",
        yaxis_title="Melhor Acurácia",
        font=dict(size=FONT_SIZE)
    )
    fig.show()


def plot_trial_values_all(studies):
    """Gráfico de dispersão com os valores de todas as tentativas para cada estratégia"""
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
        title="Acurácia de Todas as Tentativas - Todas as Estratégias",
        xaxis_title="Número da Tentativa",
        yaxis_title="Acurácia",
        font=dict(size=FONT_SIZE)
    )
    fig.show()


def plot_best_params_table(studies):
    """Tabela mostrando os melhores parâmetros para cada estratégia"""
    all_params = set()
    for study in studies.values():
        all_params.update(study.best_params.keys())
    all_params = sorted(all_params)

    header = ["Estratégia", "Acurácia"] + all_params
    rows = {col: [] for col in header}

    for strategy, study in sorted(studies.items(), key=lambda x: x[1].best_value, reverse=True):
        rows["Estratégia"].append(strategy)
        rows["Acurácia"].append(f"{study.best_value:.4f}")
        for param in all_params:
            val = study.best_params.get(param, "N/A")
            if isinstance(val, float):
                rows[param].append(f"{val:.6f}")
            else:
                rows[param].append(str(val))

    # Calcula a altura da linha dinamicamente com base no FONT_SIZE
    row_height = int(FONT_SIZE * 1.8)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header, 
            fill_color="paleturquoise", 
            align="left", 
            font=dict(size=FONT_SIZE),
            height=row_height + 10  # Cabeçalho ligeiramente maior
        ),
        cells=dict(
            values=[rows[col] for col in header], 
            fill_color="lavender", 
            align="left", 
            font=dict(size=FONT_SIZE),
            height=row_height       # Altura das células de dados
        ),
    )])
    
    fig.update_layout(
        title="Melhores Parâmetros por Estratégia de Aresta", 
        font=dict(size=FONT_SIZE),
        margin=dict(l=20, r=20, t=60, b=20) # Ajusta margens para evitar cortes laterais
    )
    fig.show()


def plot_per_strategy_details(studies):
    """Mostra a importância dos parâmetros e coordenadas paralelas por estratégia"""
    for strategy, study in studies.items():
        completed = [t for t in study.trials if t.value is not None]
        if len(completed) < 2:
            print(f"Ignorando {strategy} — não há tentativas completas suficientes ({len(completed)})")
            continue

        print(f"\n--- {strategy} ---")

        try:
            fig = plot_param_importances(study)
            fig.update_layout(title=f"Importância dos Parâmetros — {strategy}", font=dict(size=FONT_SIZE))
            fig.show()
        except Exception as e:
            print(f"  Não foi possível plotar a importância dos parâmetros para {strategy}: {e}")

        try:
            fig = plot_parallel_coordinate(study)
            fig.update_layout(title=f"Coordenadas Paralelas — {strategy}", font=dict(size=FONT_SIZE))
            fig.show()
        except Exception as e:
            print(f"  Não foi possível plotar as coordenadas paralelas para {strategy}: {e}")

        try:
            fig = plot_slice(study)
            fig.update_layout(title=f"Gráfico de Fatias — {strategy}", font=dict(size=FONT_SIZE))
            fig.show()
        except Exception as e:
            print(f"  Não foi possível plotar o gráfico de fatias para {strategy}: {e}")


def plot_accuracy_distribution(studies):
    """Box plot da distribuição de acurácia entre estratégias"""
    fig = go.Figure()
    for strategy, study in studies.items():
        values = [t.value for t in study.trials if t.value is not None]
        fig.add_trace(go.Box(y=values, name=strategy))

    fig.update_layout(
        title="Distribuição de Acurácia por Estratégia de Aresta",
        yaxis_title="Acurácia",
        font=dict(size=FONT_SIZE)
    )
    fig.show()


def print_summary(studies):
    """Imprime um resumo em texto de todos os resultados"""
    print(f"\n{'='*70}")
    print("RESUMO — Todas as Estratégias de Aresta")
    print(f"{'='*70}")
    for strategy, study in sorted(studies.items(), key=lambda x: x[1].best_value, reverse=True):
        completed = len([t for t in study.trials if t.value is not None])
        failed = len([t for t in study.trials if t.value is None])
        print(f"\n  {strategy}:")
        print(f"    Melhor acurácia: {study.best_value:.4f}")
        print(f"    Tentativas concluídas: {completed}, Falhas: {failed}")
        print(f"    Melhores parâmetros: {study.best_params}")

    # Exportar para CSV
    import pandas as pd
    rows = []
    for strategy, study in studies.items():
        for trial in study.trials:
            row = {"strategy": strategy, "trial": trial.number, "accuracy": trial.value}
            row.update(trial.params)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("optuna_results_all_strategies.csv", index=False)
    print(f"\nResultados exportados para optuna_results_all_strategies.csv")
    

def export_best_overall_config(studies):
    best_strategy = None
    best_study = None
    best_accuracy = -1
    
    # Encontra o estudo com a maior acurácia
    for strategy, study in studies.items():
        if study.best_value > best_accuracy:
            best_accuracy = study.best_value
            best_strategy = strategy
            best_study = study

    if best_study:
        # Carrega a configuração base
        config_path = "config.json"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        if "GCN" not in config:
            config["GCN"] = {}

        # Mescla os melhores parâmetros
        for key, value in best_study.best_params.items():
            # Deixar em maiúsculas mantém a consistência com o padrão de variáveis
            config["GCN"][key.upper()] = value

        # Salva explicitamente qual estratégia foi a melhor
        config["GCN"]["EDGE_STRATEGY"] = best_strategy

        # Garante que o diretório exista e escreve a nova configuração
        out_path = "optuna/fourth/best_config.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(config, f, indent=4)
            
        print(f"\nSalva a melhor configuração geral (Acurácia: {best_accuracy:.4f}) em {out_path}.")


def main():
    studies = load_studies()
    if not studies:
        print("Nenhum estudo encontrado no banco de dados.")
        return

    print_summary(studies)
    export_best_overall_config(studies)
    plot_best_accuracy_comparison(studies)
    plot_optimization_history_all(studies)
    plot_trial_values_all(studies)
    plot_accuracy_distribution(studies)
    plot_best_params_table(studies)
    plot_combined_slice(studies)
    # plot_per_strategy_details(studies)
    
    plot_overall_optuna_details(studies)


if __name__ == "__main__":
    main()