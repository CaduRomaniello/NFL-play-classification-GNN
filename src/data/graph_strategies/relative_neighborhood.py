from matplotlib import cm, pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from src.data.graph_strategies.graph_strategy import GraphStrategy
from src.utils.logger import Logger
from src.utils.visualization import createFootballField

class RNGStrategy(GraphStrategy):
    """Estratégia que conecta jogadores usando Relative Neighborhood Graph (RNG).
    
    O RNG é um subgrafo do Gabriel Graph onde uma aresta existe se não houver outro ponto
    que esteja mais próximo de ambos os extremos do que eles estão um do outro.
    """

    def __init__(self, config):
        self.config = config

    def calculate_connections(self, tracking_data: pd.DataFrame, players: pd.DataFrame = None) -> dict:
        Logger.info('Calculating connections using Relative Neighborhood Graph...')
        rng_graphs = {}
        grouped_by_game = tracking_data.groupby('gameId')
        
        for game_id, game_group in grouped_by_game:
            rng_graphs[game_id] = {}
            grouped_by_play = game_group.groupby('playId')
            
            for play_id, play_group in grouped_by_play:
                rng_graphs[game_id][play_id] = {}
                
                # Extract player coordinates and IDs
                points = play_group[['x', 'y']].values
                player_ids = play_group['nflId'].values
                
                # First compute Delaunay triangulation (RNG is a subgraph of Delaunay)
                try:
                    tri = Delaunay(points)
                    
                    # Store original points and player IDs
                    rng_graphs[game_id][play_id]['points'] = points
                    rng_graphs[game_id][play_id]['player_ids'] = player_ids
                    
                    # Calculate RNG edges by filtering Delaunay edges
                    rng_edges = []
                    for simplex in tri.simplices:
                        for i in range(len(simplex)):
                            for j in range(i+1, len(simplex)):
                                p1_idx = simplex[i]
                                p2_idx = simplex[j]
                                # Check if this edge satisfies RNG criterion
                                if self._is_rng_edge(p1_idx, p2_idx, points):
                                    rng_edges.append((p1_idx, p2_idx))
                    
                    rng_graphs[game_id][play_id]['edges'] = rng_edges
                    
                    # Calculate the connections for each player
                    rng_graphs[game_id][play_id]['connections'], rng_graphs[game_id][play_id]['edges'] = self._calc_player_connections(
                        rng_edges, player_ids, points, play_group, players)
                    
                except Exception as e:
                    Logger.error(f"Error computing RNG for game {game_id}, play {play_id}: {e}")
                    rng_graphs[game_id][play_id]['edges'] = []
                    rng_graphs[game_id][play_id]['connections'] = {}
            
        return rng_graphs
    
    def _is_rng_edge(self, p1_idx: int, p2_idx: int, points: np.ndarray) -> bool:
        """
        Check if an edge between points p1 and p2 satisfies the RNG criterion:
        No other point can be closer to both p1 and p2 than they are to each other.
        
        Returns True if the edge belongs to the RNG, False otherwise.
        """
        p1 = points[p1_idx]
        p2 = points[p2_idx]
        
        # Calculate distance between p1 and p2
        dist_p1p2 = np.linalg.norm(p1 - p2)
        
        # Check if any other point violates RNG condition
        for i in range(len(points)):
            if i != p1_idx and i != p2_idx:
                # Calculate distances from this point to p1 and p2
                dist_p1r = np.linalg.norm(p1 - points[i])
                dist_p2r = np.linalg.norm(p2 - points[i])
                
                # If both distances are less than dist_p1p2, this is not an RNG edge
                if dist_p1r < dist_p1p2 and dist_p2r < dist_p1p2:
                    return False
        
        # If we get here, no points violated the RNG condition
        return True

    def _calc_player_connections(self, rng_edges: list, player_ids: np.ndarray, points: np.ndarray, play_group: pd.DataFrame, players: pd.DataFrame) -> dict:
        """
        Calculate connections between players based on RNG.
        Returns a dictionary mapping each player to their connected players.
        """
        connections = {}
        edges = rng_edges
        
        # Initialize empty lists for each player
        for i, player_id in enumerate(player_ids):
            connections[player_id] = []
        
        # Add connections based on RNG edges
        for edge in rng_edges:
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

        if self.config.QB_LINK:
            #iterate through all rows from play_group
            for index, player in play_group.iterrows():
                # look for the value of column position
                nflId = player['nflId']
                position = players.loc[players['nflId'] == nflId]['position'].values[0]
                if position == 'QB':
                    qb_id = player['nflId']
                    for index2, teammate in play_group.iterrows():
                        if teammate['nflId'] != qb_id:
                            distance = np.linalg.norm(np.array([player['x'], player['y']]) - np.array([teammate['x'], teammate['y']]))
                            connections[qb_id].append({
                                'nflId': teammate['nflId'],
                                'distance': distance
                            })
                            # now i need to add this connection to the edges to
                            edges.append((np.where(player_ids == qb_id)[0][0], np.where(player_ids == teammate['nflId'])[0][0]))
                            connections[teammate['nflId']].append({
                                'nflId': qb_id,
                                'distance': distance
                            })
                            edges.append((np.where(player_ids == teammate['nflId'])[0][0], np.where(player_ids == qb_id)[0][0]))
                    break


        return connections, edges
    
    @staticmethod
    def get_strategy_name() -> str:
        return "RNG"
    
    def draw(self, graph_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame, ax=None):
        """
        Desenha o Relative Neighborhood Graph para uma jogada específica.
        
        Args:
            graph_dict: Dicionário com os dados do RNG
            game_id: ID do jogo
            play_id: ID da jogada
            tracking_data: DataFrame com dados de rastreamento
            ax: Eixo matplotlib para desenhar (opcional)
        """

        Logger.info(f'Drawing RNG graph for game {game_id}, play {play_id}...')
        
        if ax is None:
            fig, ax = createFootballField()
            
        if play_id not in graph_dict[game_id]:
            Logger.warning(f"No RNG data for game {game_id}, play {play_id}")
            return
        
        graph_data = graph_dict[game_id][play_id]
        
        if 'edges' not in graph_data or not graph_data['edges']:
            Logger.warning(f"No edges in RNG for game {game_id}, play {play_id}")
            return
        
        # Draw each RNG edge
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