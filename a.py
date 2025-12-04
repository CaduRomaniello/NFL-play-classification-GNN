import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

# Importações do seu projeto
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.graph_strategies.minimun_spanning_tree import MSTStrategy

def run_mst_visualization(game_id=2022091200, play_id=64):
    """
    Executa e exibe apenas a visualização da Minimum Spanning Tree.
    
    Args:
        game_id: ID do jogo a visualizar
        play_id: ID da jogada a visualizar
    """
    print(f"Executando MST para jogo {game_id}, jogada {play_id}")
    
    # Carregar configuração
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    # Carregar dados
    print("Carregando dados...")
    data_loader = DataLoader(config)
    games, player_play, players, plays = data_loader.load_auxiliar_nfl_files()
    
    # Carregar apenas a semana que contém o jogo desejado
    week_to_load = 1  # Ajustar conforme necessário
    tracking_data = data_loader.load_week_data(week_to_load)
    
    # Pré-processamento
    print("Pré-processando dados...")
    preprocessor = DataPreprocessor(config)
    plays = plays[plays['gameId'].isin(tracking_data['gameId'])].copy()
    plays = preprocessor._calc_possession_team_point_diff(plays, games).copy()
    plays = preprocessor._verify_plays_result(plays).copy()
    plays, tracking_data = preprocessor._verify_invalid_values(plays, tracking_data)
    
    # Filtrar apenas os dados da jogada específica
    tracking_data = tracking_data[tracking_data['frameType'] == 'SNAP']
    tracking_data = preprocessor._merge_player_info(players, tracking_data)
    play_tracking_data = tracking_data[(tracking_data['gameId'] == game_id) & 
                                      (tracking_data['playId'] == play_id)]
    
    if play_tracking_data.empty:
        print(f"Nenhum dado encontrado para jogo {game_id}, jogada {play_id}")
        return
    
    # Inicializar estratégia MST
    print("Calculando MST...")
    mst_strategy = MSTStrategy(config=config)
    mst_graph = mst_strategy.calculate_connections(play_tracking_data, players)
    
    # Criar figura e desenhar
    print("Desenhando MST...")
    fig, ax = plt.subplots(figsize=(12, 6.33))
    mst_strategy.draw(mst_graph, game_id, play_id, play_tracking_data, ax=ax)
    
    # Adicionar título
    ax.set_title(f"Minimum Spanning Tree - Jogo {game_id}, Jogada {play_id}", 
                fontsize=16, fontweight='bold')
    
    # Exibir
    plt.tight_layout()
    plt.show()
    
    print("Visualização concluída!")
    return fig, ax

if __name__ == "__main__":
    run_mst_visualization(game_id=2022091200, play_id=64)