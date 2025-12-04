from matplotlib import cm, pyplot as plt
import pandas as pd
import numpy as np
from src.data.graph_strategies.graph_strategy import GraphStrategy
from src.utils.logger import Logger
from src.utils.visualization import createFootballField

class MSTStrategy(GraphStrategy):
    """Estratégia que conecta jogadores usando Minimum Spanning Tree (MST).
    
    A MST conecta todos os jogadores com a menor distância total possível,
    sem formar ciclos. Utiliza o algoritmo de Kruskal.
    """
    
    def __init__(self, config):
        self.config = config
    
    def calculate_connections(self, tracking_data: pd.DataFrame, players: pd.DataFrame = None) -> dict:
        Logger.info('Calculating connections using Minimum Spanning Tree...')
        mst_graphs = {}
        grouped_by_game = tracking_data.groupby('gameId')
        
        for game_id, game_group in grouped_by_game:
            mst_graphs[game_id] = {}
            grouped_by_play = game_group.groupby('playId')
            
            for play_id, play_group in grouped_by_play:
                mst_graphs[game_id][play_id] = {}
                
                # Extract player coordinates and IDs
                points = play_group[['x', 'y']].values
                player_ids = play_group['nflId'].values
                
                try:
                    # Store original points and player IDs
                    mst_graphs[game_id][play_id]['points'] = points
                    mst_graphs[game_id][play_id]['player_ids'] = player_ids
                    
                    # Calculate all possible edges and their weights
                    edges = []
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            weight = np.linalg.norm(points[i] - points[j])
                            edges.append((i, j, weight))
                    
                    # Sort edges by weight
                    edges.sort(key=lambda x: x[2])
                    
                    # Kruskal's algorithm
                    mst_edges = []
                    parent = list(range(len(points)))
                    
                    # Build MST
                    for i, j, weight in edges:
                        if self._find(parent, i) != self._find(parent, j):  # Check if adding this edge creates a cycle
                            mst_edges.append((i, j))
                            self._union(parent, i, j)
                    
                    mst_graphs[game_id][play_id]['edges'] = mst_edges
                    
                    # Calculate the connections for each player
                    mst_graphs[game_id][play_id]['connections'], mst_graphs[game_id][play_id]['edges'] = self._calc_player_connections(
                        mst_edges, player_ids, points, play_group, players)
                    
                except Exception as e:
                    Logger.error(f"Error computing MST for game {game_id}, play {play_id}: {e}")
                    mst_graphs[game_id][play_id]['edges'] = []
                    mst_graphs[game_id][play_id]['connections'] = {}
            
        return mst_graphs
    
    def _find(self, parent: list, i: int) -> int:
        """Find operation for Union-Find data structure."""
        if parent[i] != i:
            parent[i] = self._find(parent, parent[i])
        return parent[i]
    
    def _union(self, parent: list, i: int, j: int) -> None:
        """Union operation for Union-Find data structure."""
        parent[self._find(parent, i)] = self._find(parent, j)
    
    def _calc_player_connections(self, mst_edges: list, player_ids: np.ndarray, points: np.ndarray, play_group: pd.DataFrame, players: pd.DataFrame) -> tuple:
        """
        Calculate connections between players based on Minimum Spanning Tree.
        Returns a dictionary mapping each player to their connected players and the updated edges list.
        """
        connections = {}
        edges = list(mst_edges)  # Criar uma cópia da lista de arestas
        
        # Initialize empty lists for each player
        for i, player_id in enumerate(player_ids):
            connections[player_id] = []
        
        # Add connections based on MST edges
        for edge in mst_edges:
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
                                    
                                    # Add edge to the edges list (both directions for completeness)
                                    edge_tuple = (qb_idx, teammate_idx)
                                    reverse_edge = (teammate_idx, qb_idx)
                                    
                                    if edge_tuple not in edges and reverse_edge not in edges:
                                        edges.append(edge_tuple)
                        
                        break  # Found QB, no need to continue
        
        return connections, edges
    
    @staticmethod
    def get_strategy_name() -> str:
        return "MST"
    
    def draw(self, graph_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame, ax=None):
        """
        Desenha a Minimum Spanning Tree para uma jogada específica.
        
        Args:
            graph_dict: Dicionário com os dados da MST
            game_id: ID do jogo
            play_id: ID da jogada
            tracking_data: DataFrame com dados de rastreamento
            ax: Eixo matplotlib para desenhar (opcional)
        """

        Logger.info(f'Drawing MST graph for game {game_id}, play {play_id}...')

        if ax is None:
            fig, ax = createFootballField() 
            
        if play_id not in graph_dict[game_id]:
            Logger.warning(f"No MST data for game {game_id}, play {play_id}")
            return
        
        graph_data = graph_dict[game_id][play_id]
        
        if 'edges' not in graph_data or not graph_data['edges']:
            Logger.warning(f"No edges in MST for game {game_id}, play {play_id}")
            return
        
        # Draw each MST edge
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
                
            ax.scatter(x, y, color=colors.get(player['club'], 'black'), s=200, zorder=100)
            
            # Adicionar rótulo com o número do jogador ou posição (opcional)
            # ax.text(x, y, f"{player['jerseyNumber']}", fontsize=8, ha='center', va='center', color='white')

        plt.show()