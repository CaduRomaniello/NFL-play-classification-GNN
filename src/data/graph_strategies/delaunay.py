from matplotlib import cm, pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from src.data.graph_strategies.graph_strategy import GraphStrategy
from src.utils.logger import Logger
from src.utils.visualization import createFootballField

class DelaunayStrategy(GraphStrategy):
    """Estratégia que conecta jogadores usando triangulação de Delaunay."""
    
    def calculate_connections(self, tracking_data: pd.DataFrame, players: pd.DataFrame = None) -> dict:
        Logger.info('Calculating connections using Delaunay triangulation...')
        triangulations = {}
        grouped_by_game = tracking_data.groupby('gameId')
        
        for game_id, game_group in grouped_by_game:
            triangulations[game_id] = {}
            grouped_by_play = game_group.groupby('playId')
            
            for play_id, play_group in grouped_by_play:
                triangulations[game_id][play_id] = {}
                
                # Extract player coordinates and IDs
                points = play_group[['x', 'y']].values
                player_ids = play_group['nflId'].values
                
                # Compute Delaunay triangulation
                try:
                    tri = Delaunay(points)
                    
                    # Store original points and player IDs
                    triangulations[game_id][play_id]['points'] = points
                    triangulations[game_id][play_id]['player_ids'] = player_ids
                    triangulations[game_id][play_id]['triangulation'] = tri
                    
                    # Calculate the connections for each player
                    triangulations[game_id][play_id]['connections'] = self._calc_player_connections(
                        tri, player_ids, points)
                
                except Exception as e:
                    Logger.error(f"Error computing triangulation for game {game_id}, play {play_id}: {e}")
                    triangulations[game_id][play_id]['connections'] = {}
            
        return triangulations
    
    def _calc_player_connections(self, tri, player_ids, points):
        connections = {}
        
        # Create a dictionary to map from point index to player ID
        idx_to_id = {i: player_ids[i] for i in range(len(player_ids))}
        
        # For each player, find all connected players in the triangulation
        for i in range(len(player_ids)):
            player_id = player_ids[i]
            connections[player_id] = []
            
            # Find simplices (triangles) containing this point
            point_neighbors = set()
            for simplex in tri.simplices:
                if i in simplex:
                    # Add all other points in this simplex
                    for j in simplex:
                        if j != i:
                            point_neighbors.add(j)
            
            # Convert point indices to player IDs and calculate distances
            for neighbor_idx in point_neighbors:
                neighbor_id = idx_to_id[neighbor_idx]
                distance = np.linalg.norm(points[i] - points[neighbor_idx])
                connections[player_id].append({
                    'nflId': neighbor_id,
                    'distance': distance
                })
        
        return connections
    
    @staticmethod
    def get_strategy_name() -> str:
        return "DELAUNAY"
    
    def draw(self, graph_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame, ax=None):
        """
        Desenha a triangulação de Delaunay para uma jogada específica.
        
        Args:
            graph_dict: Dicionário com os dados da triangulação
            game_id: ID do jogo
            play_id: ID da jogada
            tracking_data: DataFrame com dados de rastreamento
            ax: Eixo matplotlib para desenhar (opcional)
        """

        Logger.info(f'Drawing Delaunay graph for game {game_id}, play {play_id}...')
        
        if ax is None:
            fig, ax = createFootballField() 
            
        if play_id not in graph_dict[game_id]:
            Logger.warning(f"No Delaunay data for game {game_id}, play {play_id}")
            return
        
        graph_data = graph_dict[game_id][play_id]
        
        if 'triangulation' not in graph_data or graph_data['triangulation'] is None:
            Logger.warning(f"No triangulation data for game {game_id}, play {play_id}")
            return
        
        # Draw each Delaunay triangle
        for simplex in graph_data['triangulation'].simplices:
            # Get player IDs for this triangle
            players = [graph_data['player_ids'][i] for i in simplex]
            # Get coordinates for each player
            coords = []
            for player_id in players:
                player_data = tracking_data[tracking_data['nflId'] == player_id]
                if not player_data.empty:
                    coords.append((player_data['x'].values[0], player_data['y'].values[0]))
            
            # Draw the triangle if we have all coordinates
            if len(coords) == 3:
                ax.plot([coords[0][0], coords[1][0], coords[2][0], coords[0][0]], 
                        [coords[0][1], coords[1][1], coords[2][1], coords[0][1]], 
                        'k-', alpha=1.0, linewidth=1.5, zorder=50)
                
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