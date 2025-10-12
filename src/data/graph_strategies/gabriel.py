from matplotlib import cm, pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from src.data.graph_strategies.graph_strategy import GraphStrategy
from src.utils.logger import Logger
from src.utils.visualization import createFootballField

class GabrielStrategy(GraphStrategy):
    """Estratégia que conecta jogadores usando Gabriel Graph.
    
    O Gabriel Graph é um subgrafo da triangulação de Delaunay onde uma aresta existe
    se o círculo com diâmetro na aresta não contém outros pontos.
    """
    
    def __init__(self, config):
        self.config = config
    
    def calculate_connections(self, tracking_data: pd.DataFrame, players: pd.DataFrame = None) -> dict:
        Logger.info('Calculating connections using Gabriel Graph...')
        gabriel_graphs = {}
        grouped_by_game = tracking_data.groupby('gameId')
        
        for game_id, game_group in grouped_by_game:
            gabriel_graphs[game_id] = {}
            grouped_by_play = game_group.groupby('playId')
            
            for play_id, play_group in grouped_by_play:
                gabriel_graphs[game_id][play_id] = {}
                
                # Extract player coordinates and IDs
                points = play_group[['x', 'y']].values
                player_ids = play_group['nflId'].values
                
                # First compute Delaunay triangulation
                try:
                    tri = Delaunay(points)
                    
                    # Store original points and player IDs
                    gabriel_graphs[game_id][play_id]['points'] = points
                    gabriel_graphs[game_id][play_id]['player_ids'] = player_ids
                    
                    # Calculate Gabriel Graph edges by filtering Delaunay edges
                    gabriel_edges = []
                    for simplex in tri.simplices:
                        for i in range(len(simplex)):
                            for j in range(i+1, len(simplex)):
                                p1_idx = simplex[i]
                                p2_idx = simplex[j]
                                # Check if this edge satisfies Gabriel Graph criterion
                                if self._is_gabriel_edge(p1_idx, p2_idx, points):
                                    gabriel_edges.append((p1_idx, p2_idx))
                    
                    gabriel_graphs[game_id][play_id]['edges'] = gabriel_edges
                    
                    # Calculate the connections for each player
                    gabriel_graphs[game_id][play_id]['connections'], gabriel_graphs[game_id][play_id]['edges'] = self._calc_player_connections(
                        gabriel_edges, player_ids, points, play_group, players)
                    
                except Exception as e:
                    Logger.error(f"Error computing Gabriel Graph for game {game_id}, play {play_id}: {e}")
                    gabriel_graphs[game_id][play_id]['edges'] = []
                    gabriel_graphs[game_id][play_id]['connections'] = {}
            
        return gabriel_graphs
    
    def _is_gabriel_edge(self, p1_idx: int, p2_idx: int, points: np.ndarray) -> bool:
        """
        Check if an edge between points p1 and p2 satisfies the Gabriel Graph criterion:
        No other point can be inside the circle with diameter p1-p2.
        
        Returns True if the edge belongs to the Gabriel Graph, False otherwise.
        """
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        
        # Calculate the center of the circle (midpoint of p1-p2)
        center = (p1 + p2) / 2
        
        # Calculate the radius (half the distance between p1 and p2)
        radius = np.linalg.norm(p1 - p2) / 2
        
        # Check if any other point is inside this circle
        for i in range(len(points)):
            if i != p1_idx and i != p2_idx:
                # Calculate distance from this point to the center
                dist_to_center = np.linalg.norm(points[i] - center)
                # If distance is less than radius, point is inside circle
                if dist_to_center < radius:
                    return False
        
        # If we get here, no points were inside the circle
        return True
    
    def _calc_player_connections(self, gabriel_edges: list, player_ids: np.ndarray, points: np.ndarray, play_group: pd.DataFrame, players: pd.DataFrame) -> tuple:
        """
        Calculate connections between players based on Gabriel Graph.
        Returns a dictionary mapping each player to their connected players and the updated edges list.
        """
        connections = {}
        edges = list(gabriel_edges)  # Criar uma cópia da lista de arestas
        
        # Initialize empty lists for each player
        for i, player_id in enumerate(player_ids):
            connections[player_id] = []
        
        # Add connections based on Gabriel edges
        for edge in gabriel_edges:
            p1_idx, p2_idx = edge
            p1_id = player_ids[p1_idx]
            p2_id = player_ids[p2_idx]
            
            # Calculate distance
            distance = np.linalg.norm(points[p1_idx] - points[p2_idx])
            
            # Add bidirectional connections
            connections[p1_id].append({
                'nflId': p2_id,
                'distance': distance
            })
            
            connections[p2_id].append({
                'nflId': p1_id,
                'distance': distance
            })
        
        # Add QB connections if QB_LINK is enabled
        if hasattr(self.config, 'QB_LINK') and self.config.QB_LINK and players is not None:
            # Iterate through all rows from play_group to find the QB
            for index, player in play_group.iterrows():
                nflId = player['nflId']
                player_info = players.loc[players['nflId'] == nflId]
                
                if not player_info.empty:
                    position = player_info['position'].values[0]
                    
                    if position == 'QB':
                        qb_id = player['nflId']
                        qb_idx = np.where(player_ids == qb_id)[0][0]
                        
                        # Connect QB to all other players
                        for index2, teammate in play_group.iterrows():
                            teammate_id = teammate['nflId']
                            
                            if teammate_id != qb_id:
                                teammate_idx = np.where(player_ids == teammate_id)[0][0]
                                
                                # Calculate distance
                                distance = np.linalg.norm(
                                    np.array([player['x'], player['y']]) - 
                                    np.array([teammate['x'], teammate['y']])
                                )
                                
                                # Check if this connection already exists
                                already_connected = any(
                                    conn['nflId'] == teammate_id 
                                    for conn in connections[qb_id]
                                )
                                
                                if not already_connected:
                                    # Add QB -> Teammate connection
                                    connections[qb_id].append({
                                        'nflId': teammate_id,
                                        'distance': distance
                                    })
                                    
                                    # Add Teammate -> QB connection
                                    connections[teammate_id].append({
                                        'nflId': qb_id,
                                        'distance': distance
                                    })
                                    
                                    # Add edge to the edges list
                                    edge_tuple = (qb_idx, teammate_idx)
                                    reverse_edge = (teammate_idx, qb_idx)
                                    
                                    if edge_tuple not in edges and reverse_edge not in edges:
                                        edges.append(edge_tuple)
                        
                        break  # Found QB, no need to continue
        
        return connections, edges
    
    @staticmethod
    def get_strategy_name() -> str:
        return "GABRIEL"
    
    def draw(self, graph_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame, ax=None):
        """
        Desenha o Gabriel Graph para uma jogada específica.
        
        Args:
            graph_dict: Dicionário com os dados do Gabriel Graph
            game_id: ID do jogo
            play_id: ID da jogada
            tracking_data: DataFrame com dados de rastreamento
            ax: Eixo matplotlib para desenhar (opcional)
        """

        Logger.info(f'Drawing Gabriel Graph for game {game_id}, play {play_id}...')
        
        if ax is None:
            fig, ax = createFootballField() 
            
        if play_id not in graph_dict[game_id]:
            Logger.warning(f"No Gabriel Graph data for game {game_id}, play {play_id}")
            return
        
        graph_data = graph_dict[game_id][play_id]
        
        if 'edges' not in graph_data or not graph_data['edges']:
            Logger.warning(f"No edges in Gabriel Graph for game {game_id}, play {play_id}")
            return
        
        # Draw each Gabriel edge
        for edge in graph_data['edges']:
            p1_idx, p2_idx = edge
            p1_id = graph_data['player_ids'][p1_idx]
            p2_id = graph_data['player_ids'][p2_idx]
            
            # Get player coordinates
            p1_data = tracking_data[tracking_data['nflId'] == p1_id]
            p2_data = tracking_data[tracking_data['nflId'] == p2_id]
            
            if not p1_data.empty and not p2_data.empty:
                x1, y1 = p1_data['x'].values[0], p1_data['y'].values[0]
                x2, y2 = p2_data['x'].values[0], p2_data['y'].values[0]
                
                # Draw the edge
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=1.0, linewidth=1.5, zorder=50)

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