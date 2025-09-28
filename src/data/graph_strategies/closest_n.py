from matplotlib import cm, pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from src.data.graph_strategies.graph_strategy import GraphStrategy
from src.utils.logger import Logger
from src.utils.visualization import createFootballField

class ClosestNStrategy(GraphStrategy):
    """Estratégia que conecta cada jogador aos N jogadores mais próximos."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_connections(self, tracking_data: pd.DataFrame, players: pd.DataFrame = None) -> dict:
        Logger.info(f'Calculating connections using {self.config.N} closest players strategy...')
        distances = {}
        grouped_by_game = tracking_data.groupby('gameId')
        
        for game_id, game_group in grouped_by_game:
            distances[game_id] = {}
            grouped_by_play = game_group.groupby('playId')
            
            for play_id, play_group in grouped_by_play:
                distances[game_id][play_id] = {}
                
                coords = play_group[['x', 'y']].values
                dist_matrix = cdist(coords, coords, metric='euclidean')
                
                distances[game_id][play_id]['dist_df'] = pd.DataFrame(
                    dist_matrix, index=play_group['nflId'], columns=play_group['nflId'])
                
                distances[game_id][play_id]['sorted_distances'] = distances[game_id][play_id]['dist_df'].apply(
                    lambda row: row.sort_values().values.tolist(), axis=1)
                distances[game_id][play_id]['sorted_players'] = distances[game_id][play_id]['dist_df'].apply(
                    lambda row: row.sort_values().index.tolist(), axis=1)
                
                distances[game_id][play_id]['connections'] = self._calc_n_closest_players(
                    distances[game_id][play_id]['sorted_distances'],
                    distances[game_id][play_id]['sorted_players'],
                    players, play_group)
            
        return distances
    
    def _calc_n_closest_players(self, sorted_distances, sorted_players, players, tracking_data):
        all_dist = {}
        for index, value in sorted_distances.items():
            closest_players = []
            for i in range(1, len(sorted_players.loc[index])):
                closest_players.append({
                    'nflId': sorted_players.loc[index][i],
                    'distance': value[i]
                })
                if len(closest_players) == self.config.N:
                    break

            all_dist[index] = closest_players
            
        return all_dist
    
    @staticmethod
    def get_strategy_name() -> str:
        return "CLOSEST-N"
    
    def draw(self, graph_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame, ax=None):
        """
        Desenha as conexões dos N jogadores mais próximos para uma jogada específica.
        
        Args:
            graph_dict: Dicionário com os dados das conexões
            game_id: ID do jogo
            play_id: ID da jogada
            tracking_data: DataFrame com dados de rastreamento
            ax: Eixo matplotlib para desenhar (opcional)
        """

        Logger.info(f'Drawing closest-N graph for game {game_id}, play {play_id}...')
        
        if ax is None:
            fig, ax = createFootballField() 
            
        if play_id not in graph_dict[game_id]:
            Logger.warning(f"No closest-N data for game {game_id}, play {play_id}")
            return
        
        graph_data = graph_dict[game_id][play_id]
        
        if 'connections' not in graph_data or not graph_data['connections']:
            Logger.warning(f"No connections in closest-N for game {game_id}, play {play_id}")
            return
        
        # Draw each connection
        for player_id, connections in graph_data['connections'].items():
            player_data = tracking_data[tracking_data['nflId'] == player_id]
            
            if not player_data.empty:
                p1_x = player_data['x'].values[0]
                p1_y = player_data['y'].values[0]
                
                for connection in connections:
                    connected_player_id = connection['nflId']
                    connected_player_data = tracking_data[tracking_data['nflId'] == connected_player_id]
                    
                    if not connected_player_data.empty:
                        p2_x = connected_player_data['x'].values[0]
                        p2_y = connected_player_data['y'].values[0]
                        
                        # Draw the connection
                        ax.plot([p1_x, p2_x], [p1_y, p2_y], 'k-', alpha=1.0, linewidth=1.5, zorder=50)

        # Gerar cores para cada time automaticamente
        distinct_colors = [
            '#e6194B',  # Vermelho
            '#3cb44b',  # Verde
            '#4363d8',  # Azul
            '#f58231',  # Laranja
            '#911eb4',  # Roxo
            '#42d4f4',  # Ciano
            '#f032e6',  # Magenta
            '#bfef45',  # Lima
            '#fabed4',  # Rosa
            '#469990',  # Verde-azulado
            '#dcbeff',  # Lavanda
            '#9A6324',  # Marrom
            '#fffac8',  # Bege
            '#800000',  # Bordô
            '#aaffc3',  # Menta
            '#808000',  # Oliva
            '#ffd8b1',  # Pêssego
            '#000075',  # Azul marinho
            '#a9a9a9',  # Cinza médio
        ]

        unique_teams = tracking_data['club'].unique()
        
        # Remover 'football' se estiver presente para tratá-lo separadamente
        if 'football' in unique_teams:
            unique_teams = unique_teams[unique_teams != 'football']
            
        # Criar dicionário de cores para cada time
        colors = {}
        for i, team in enumerate(unique_teams):
            colors[team] = distinct_colors[i % len(distinct_colors)]
        
        # Adicionar cor específica para a bola
        colors['football'] = '#8B4513'  # Um marrom mais escuro para a bola

        # Plotar os jogadores
        for index, player in tracking_data.iterrows():
            x = player['x']
            y = player['y']
            s = player['displayName']
            
            if s == 'football':
                # Opcional: plotar a bola de forma diferente
                ax.scatter(x, y, color=colors['football'], marker='o', s=100, zorder=100)
                continue
                
            ax.scatter(x, y, color=colors.get(player['club'], 'black'), zorder=100)
            
            # Adicionar rótulo com o número do jogador ou posição (opcional)
            # ax.text(x, y, f"{player['jerseyNumber']}", fontsize=8, ha='center', va='center', color='white')

        plt.show()