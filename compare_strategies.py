import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from matplotlib.gridspec import GridSpec

# Importações do seu projeto
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.visualization import createFootballField
from src.data.graph_strategies.delaunay import DelaunayStrategy
from src.data.graph_strategies.gabriel import GabrielStrategy
from src.data.graph_strategies.relative_neighborhood import RNGStrategy
from src.data.graph_strategies.minimun_spanning_tree import MSTStrategy
from src.data.graph_strategies.closest_n import ClosestNStrategy
from src.data.graph_strategies.qb_closest_n import QBClosestNStrategy

def draw_football_field(ax):
    """Desenha um campo de futebol no eixo fornecido."""
    import matplotlib.patches as patches
    
    # Criar campo
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                            edgecolor='r', facecolor='gray', zorder=0)
    ax.add_patch(rect)
    
    # plot field lines
    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
            80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
            [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
            53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
            color='white')
    
    # Endzones
    ez1 = patches.Rectangle((0, 0), 10, 53.3,
                            linewidth=0.1,
                            edgecolor='r',
                            facecolor='black',
                            alpha=0.2,
                            zorder=0)
    ez2 = patches.Rectangle((110, 0), 120, 53.3,
                            linewidth=0.1,
                            edgecolor='r',
                            facecolor='brown',
                            alpha=0.2,
                            zorder=0)
    ax.add_patch(ez1)
    ax.add_patch(ez2)
    
    # set axis limits
    ax.set_xlim(0, 120)
    ax.set_ylim(-5, 58.3)
    ax.axis('off')
    
    # plot line numbers
    for x in range(20, 110, 10):
        numb = x
        if x > 50:
            numb = 120 - x
        ax.text(x, 5, str(numb - 10),
                horizontalalignment='center',
                fontsize=20,
                color='white')
        ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                horizontalalignment='center',
                fontsize=20,
                color='white', rotation=180)
    
    # checking the size of image to plot hash marks for each yd line
    hash_range = range(11, 110)
    
    # plot hash marks
    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

def compare_graph_strategies(game_id=2022091200, play_id=64, save_path=None):
    """
    Compara visualmente diferentes estratégias de grafo para uma jogada específica.
    
    Args:
        game_id: ID do jogo a visualizar
        play_id: ID da jogada a visualizar
        save_path: Caminho para salvar a imagem (opcional)
    """
    print(f"Comparando estratégias de grafo para jogo {game_id}, jogada {play_id}")
    
    # Carregar configuração
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    
    # Carregar dados
    print("Carregando dados...")
    data_loader = DataLoader(config)
    games, player_play, players, plays = data_loader.load_auxiliar_nfl_files()
    
    # Carregar apenas a semana que contém o jogo desejado
    week_to_load = 1
    tracking_data = data_loader.load_week_data(week_to_load)
    
    # Pré-processamento básico necessário
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
    
    # Calculate zoom limits based on player positions
    player_data = play_tracking_data[play_tracking_data['displayName'] != 'football']
    if not player_data.empty:
        x_coords = player_data['x'].values
        y_coords = player_data['y'].values
        
        margin_x = 5
        margin_y = 3
        
        zoom_x_min = max(0, np.min(x_coords) - margin_x)
        zoom_x_max = min(120, np.max(x_coords) + margin_x)
        zoom_y_min = max(0, np.min(y_coords) - margin_y)
        zoom_y_max = min(53.3, np.max(y_coords) + margin_y)
    else:
        zoom_x_min, zoom_x_max = 0, 120
        zoom_y_min, zoom_y_max = 0, 53.3
    
    # Criar figura para comparação
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3, figure=fig)
    
    # Inicializar estratégias
    strategies = [
        (DelaunayStrategy(config=config), "Triangulação de Delaunay", gs[0, 0]),
        (GabrielStrategy(config=config), "Gabriel Graph", gs[0, 1]),
        (RNGStrategy(config=config), "Relative Neighborhood Graph", gs[0, 2]),
        (MSTStrategy(config=config), "Minimum Spanning Tree", gs[1, 0]),
        (ClosestNStrategy(config=config), f"Closest-{config.N}", gs[1, 1]),
        (QBClosestNStrategy(config=config), f"QB-Closest-{config.N}", gs[1, 2])
    ]
    
    # Suprimir exibição plt.show() nas funções draw
    plt.ioff()
    
    # Calcular e visualizar cada estratégia
    axes = []
    for strategy, title, position in strategies:
        print(f"Calculando e desenhando {title}...")
        
        # Calcular conexões
        connections = strategy.calculate_connections(play_tracking_data, players)
        
        # Criar subplot
        ax = fig.add_subplot(position)
        
        # Modificar temporariamente a função show para não exibir a figura
        original_show = plt.show
        plt.show = lambda: None
        
        # Desenhar estratégia (ela já desenha o campo)
        strategy.draw(connections, game_id, play_id, play_tracking_data, ax=ax)
        
        # Restaurar função show
        plt.show = original_show
        
        # APLICAR ZOOM APENAS NESTE SUBPLOT
        ax.set_xlim(zoom_x_min, zoom_x_max)
        ax.set_ylim(zoom_y_min, zoom_y_max)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adicionar título
        ax.set_title(title, fontsize=22, fontweight='bold')
        axes.append(ax)
    
    # Ajustar layout e adicionar título principal
    fig.suptitle(f"Comparação de Estratégias de Grafo - Jogo {game_id}, Jogada {play_id}", 
                fontsize=22, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Salvar ou mostrar
    if save_path:
        print(f"Salvando imagem em {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    print("Comparação concluída!")
    return fig, axes

# Modificar a função createFootballField para aceitar um eixo existente
def patch_createFootballField():
    """Modifica a função createFootballField para aceitar um eixo existente."""
    import types
    from src.utils.visualization import createFootballField as original_createFootballField
    
    def new_createFootballField(linenumbers=True,
                            endzones=True,
                            highlight_line=False,
                            highlight_line_number=50,
                            highlighted_name='Line of Scrimmage',
                            fifty_is_los=False,
                            figsize=(12, 6.33),
                            ax=None):
        """Versão modificada que permite reutilizar um eixo existente."""
        import matplotlib.patches as patches
        
        if ax is None:
            # Criar nova figura e eixo
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            # Usar eixo existente
            fig = ax.figure
        
        # Criar campo
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                edgecolor='r', facecolor='gray', zorder=0)
        ax.add_patch(rect)
        
        # plot field lines
        ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                color='white')
        
        # plot line of scrimmage at 50 yd line if fifty_is_los is True
        if fifty_is_los:
            ax.plot([60, 60], [0, 53.3], color='gold')
            ax.text(62, 50, '<- Player Yardline at Snap', color='gold')
        
        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='black',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='brown',
                                    alpha=0.2,
                                    zorder=0)
            ax.add_patch(ez1)
            ax.add_patch(ez2)
        
        # set axis limits
        ax.set_xlim(0, 120)
        ax.set_ylim(-5, 58.3)
        ax.axis('off')
        
        # plot line numbers
        if linenumbers:
            for x in range(20, 110, 10):
                numb = x
                if x > 50:
                    numb = 120 - x
                ax.text(x, 5, str(numb - 10),
                        horizontalalignment='center',
                        fontsize=22,
                        color='white')
                ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                        horizontalalignment='center',
                        fontsize=22,
                        color='white', rotation=180)
        
        # checking the size of image to plot hash marks for each yd line
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)
        
        # plot hash marks
        for x in hash_range:
            ax.plot([x, x], [0.4, 0.7], color='white')
            ax.plot([x, x], [53.0, 52.5], color='white')
            ax.plot([x, x], [22.91, 23.57], color='white')
            ax.plot([x, x], [29.73, 30.39], color='white')
        
        # highlight line of scrimmage
        if highlight_line:
            hl = highlight_line_number + 10
            ax.plot([hl, hl], [0, 53.3], color='yellow')
            ax.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                    color='yellow')
        
        return fig, ax
    
    # Substituir a função original
    import src.utils.visualization
    src.utils.visualization.createFootballField = new_createFootballField

if __name__ == "__main__":
    # Modificar a função createFootballField para aceitar um eixo existente
    patch_createFootballField()
    
    # Criar diretório para salvar resultados
    os.makedirs("output/comparisons", exist_ok=True)
    
    # Comparar diferentes jogadas
    game_ids = [2022091200]
    play_ids = [64, 85, 109]  # Vários IDs de jogadas para comparar
    
    for game_id in game_ids:
        for play_id in play_ids:
            save_path = f"output/comparisons/comparison_game_{game_id}_play_{play_id}.png"
            compare_graph_strategies(game_id, play_id, save_path)