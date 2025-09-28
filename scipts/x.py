import os
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.spatial import Delaunay
from IPython.display import display
from scipy.spatial.distance import cdist

GAME_ID = 2022091112
PLAY_ID = 1674

# MINIMUM SPANNING TREE (MST)
def calc_mst_connections(tracking_data: pd.DataFrame) -> dict:
    """
    Calculate Minimum Spanning Tree for player positions.
    MST connects all players with minimum total distance, without forming cycles.
    Uses Kruskal's algorithm.
    """
    print("Calculating Minimum Spanning Tree between players...")
    
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
                
                # Find operation for Union-Find
                def find(i):
                    if parent[i] != i:
                        parent[i] = find(parent[i])
                    return parent[i]
                
                # Union operation for Union-Find
                def union(i, j):
                    parent[find(i)] = find(j)
                
                # Build MST
                for i, j, weight in edges:
                    if find(i) != find(j):  # Check if adding this edge creates a cycle
                        mst_edges.append((i, j))
                        union(i, j)
                
                mst_graphs[game_id][play_id]['edges'] = mst_edges
                
                # Calculate the connections for each player
                mst_graphs[game_id][play_id]['connections'] = calc_mst_player_connections(
                    mst_edges, player_ids, points)
                
            except Exception as e:
                print(f"Error computing MST for game {game_id}, play {play_id}: {e}")
                mst_graphs[game_id][play_id]['edges'] = []
                mst_graphs[game_id][play_id]['connections'] = {}
        
    return mst_graphs

def calc_mst_player_connections(mst_edges: list, player_ids: np.ndarray, points: np.ndarray) -> dict:
    """
    Calculate connections between players based on Minimum Spanning Tree.
    Returns a dictionary mapping each player to their connected players.
    """
    connections = {}
    
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
    
    return connections

def draw_mst_graph(mst_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame):
    """
    Draw the Minimum Spanning Tree for a specific play.
    """
    if play_id not in mst_dict[game_id]:
        print(f"No MST data for game {game_id}, play {play_id}")
        return
    
    graph_data = mst_dict[game_id][play_id]
    
    if 'edges' not in graph_data or not graph_data['edges']:
        print(f"No edges in MST for game {game_id}, play {play_id}")
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
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.7, linewidth=2)

# RELATIVE NEIGHBORHOOD GRAPH
def calc_rng_connections(tracking_data: pd.DataFrame) -> dict:
    """
    Calculate Relative Neighborhood Graph (RNG) for player positions.
    RNG is a subgraph of Gabriel Graph where an edge exists if there is no other point
    that is closer to both endpoints than they are to each other.
    """
    print("Calculating Relative Neighborhood Graph between players...")
    
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
                            if is_rng_edge(p1_idx, p2_idx, points):
                                rng_edges.append((p1_idx, p2_idx))
                
                rng_graphs[game_id][play_id]['edges'] = rng_edges
                
                # Calculate the connections for each player
                rng_graphs[game_id][play_id]['connections'] = calc_rng_player_connections(
                    rng_edges, player_ids, points)
                
            except Exception as e:
                print(f"Error computing RNG for game {game_id}, play {play_id}: {e}")
                rng_graphs[game_id][play_id]['edges'] = []
                rng_graphs[game_id][play_id]['connections'] = {}
        
    return rng_graphs

def is_rng_edge(p1_idx: int, p2_idx: int, points: np.ndarray) -> bool:
    """
    Check if an edge between points p1 and p2 satisfies the Relative Neighborhood Graph criterion:
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

def calc_rng_player_connections(rng_edges: list, player_ids: np.ndarray, points: np.ndarray) -> dict:
    """
    Calculate connections between players based on Relative Neighborhood Graph.
    Returns a dictionary mapping each player to their connected players.
    """
    connections = {}
    
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
    
    return connections

def draw_rng_graph(rng_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame):
    """
    Draw the Relative Neighborhood Graph for a specific play.
    """
    if play_id not in rng_dict[game_id]:
        print(f"No RNG data for game {game_id}, play {play_id}")
        return
    
    graph_data = rng_dict[game_id][play_id]
    
    if 'edges' not in graph_data or not graph_data['edges']:
        print(f"No edges in RNG for game {game_id}, play {play_id}")
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
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)

# GABRIEL GRAPH
def calc_gabriel_connections(tracking_data: pd.DataFrame) -> dict:
    """
    Calculate Gabriel Graph for player positions.
    Gabriel Graph is a subgraph of Delaunay triangulation where an edge exists
    if the circle with the edge as diameter contains no other points.
    """
    print("Calculating Gabriel Graph between players...")
    
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
                            if is_gabriel_edge(p1_idx, p2_idx, points):
                                gabriel_edges.append((p1_idx, p2_idx))
                
                gabriel_graphs[game_id][play_id]['edges'] = gabriel_edges
                
                # Calculate the connections for each player
                gabriel_graphs[game_id][play_id]['connections'] = calc_gabriel_player_connections(
                    gabriel_edges, player_ids, points)
                
            except Exception as e:
                print(f"Error computing Gabriel Graph for game {game_id}, play {play_id}: {e}")
                gabriel_graphs[game_id][play_id]['edges'] = []
                gabriel_graphs[game_id][play_id]['connections'] = {}
        
    return gabriel_graphs

def is_gabriel_edge(p1_idx: int, p2_idx: int, points: np.ndarray) -> bool:
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

def calc_gabriel_player_connections(gabriel_edges: list, player_ids: np.ndarray, points: np.ndarray) -> dict:
    """
    Calculate connections between players based on Gabriel Graph.
    Returns a dictionary mapping each player to their connected players.
    """
    connections = {}
    
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
    
    return connections

def draw_gabriel_graph(gabriel_dict: dict, game_id: int, play_id: int, tracking_data: pd.DataFrame):
    """
    Draw the Gabriel Graph for a specific play.
    """
    if play_id not in gabriel_dict[game_id]:
        print(f"No Gabriel Graph data for game {game_id}, play {play_id}")
        return
    
    graph_data = gabriel_dict[game_id][play_id]
    
    if 'edges' not in graph_data or not graph_data['edges']:
        print(f"No edges in Gabriel Graph for game {game_id}, play {play_id}")
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
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)

# DEALAUNAY TRIANGULATION
def calc_delaunay_connections(tracking_data: pd.DataFrame) -> dict:
    """
    Calculate Delaunay triangulation for player positions.
    Returns a dictionary with connections between players based on the triangulation.
    """
    print("Calculating Delaunay triangulation between players...")
    
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
                
                # Store the triangulation object
                triangulations[game_id][play_id]['triangulation'] = tri
                
                # Store original points and player IDs
                triangulations[game_id][play_id]['points'] = points
                triangulations[game_id][play_id]['player_ids'] = player_ids
                
                # Calculate the connections for each player
                triangulations[game_id][play_id]['connections'] = calc_player_connections(tri, player_ids, points)
                
                # Optionally, also store distance information
                triangulations[game_id][play_id]['distances'] = calc_connection_distances(tri, player_ids, points)
            
            except Exception as e:
                print(f"Error computing triangulation for game {game_id}, play {play_id}: {e}")
                # If triangulation fails (e.g., coplanar points), store empty results
                triangulations[game_id][play_id]['triangulation'] = None
                triangulations[game_id][play_id]['connections'] = {}
        
    return triangulations

def calc_player_connections(tri: Delaunay, player_ids: np.ndarray, points: np.ndarray) -> dict:
    """
    Calculate connections between players based on Delaunay triangulation.
    Returns a dictionary mapping each player to their connected players.
    """
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

def calc_connection_distances(tri: Delaunay, player_ids: np.ndarray, points: np.ndarray) -> pd.DataFrame:
    """
    Calculate distances between connected players in the Delaunay triangulation.
    Returns a sparse distance matrix as a DataFrame.
    """
    # Initialize an empty distance matrix
    n = len(player_ids)
    distances = np.zeros((n, n))
    
    # Fill with infinities (representing no direct connection)
    distances.fill(np.inf)
    
    # Set diagonal to zero (distance to self)
    np.fill_diagonal(distances, 0)
    
    # For each edge in the triangulation, compute the distance
    edges = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                # Add edge (smaller index first)
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    
    # Calculate distances for each edge
    for i, j in edges:
        dist = np.linalg.norm(points[i] - points[j])
        distances[i, j] = dist
        distances[j, i] = dist  # Distance matrix is symmetric
    
    # Convert to DataFrame with player IDs as indices and columns
    dist_df = pd.DataFrame(distances, index=player_ids, columns=player_ids)
    
    return dist_df

######################################
def createFootballField(linenumbers=True,
                          endzones=True,
                          highlight_line=False,
                          highlight_line_number=50,
                          highlighted_name='Line of Scrimmage',
                          fifty_is_los=False,
                          figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """

    # create figure
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='gray', zorder=0)

    # create axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    # plot field lines
    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    
    # plot line of scrimmage at 50 yd line if fifty_is_los is True
    if fifty_is_los:
        plt.plot([60, 60], [0, 53.3], color='gold')
        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='green',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='purple',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)

    # set axis limits
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')

    # plot line numbers
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
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
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
        
    return fig, ax

def play_result(play: pd.Series) -> int:
    playType = None
    if not pd.isna(play['qbSpike']) and play['qbSpike']:
        playType = None
    elif not pd.isna(play['qbKneel']) and play['qbKneel']:
        playType = None
    elif not pd.isna(play['qbSneak']) and play['qbSneak']:
        playType = None
    elif play['passResult'] == 'R':
        playType = None
    elif not pd.isna(play['rushLocationType']):
        playType = 0
    elif not pd.isna(play['passLocationType']):
        playType = 1
    elif not pd.isna(play['passResult']):
        playType = 1
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        display(play)
        raise ValueError("Can't determine play type")
    
    return playType

def calc_n_closest_players(sorted_distances: list, sorted_players: list, n: int) -> dict:
    all_dist = {}
    for index, value in sorted_distances.items():
        closest_players = []
        for i in range(1, len(sorted_players.loc[index])):
            closest_players.append({
                'nflId': sorted_players.loc[index][i],
                'distance': value[i]
            })
            if len(closest_players) == n:
                break
        all_dist[index] = closest_players
        
    return all_dist

def calc_distance_between_players(tracking_data: pd.DataFrame, n: int = 2) -> dict:
    print("Calculating distance between players...")
    
    distances = {}
    grouped_by_game = tracking_data.groupby('gameId')
    
    for game_id, game_group in grouped_by_game:
        distances[game_id] = {}
        grouped_by_play = game_group.groupby('playId')
        
        for play_id, play_group in grouped_by_play:
            distances[game_id][play_id] = {}
            
            coords = play_group[['x', 'y']].values
            dist_matrix = cdist(coords, coords, metric='euclidean')
            
            distances[game_id][play_id]['dist_df'] = pd.DataFrame(dist_matrix, index=play_group['nflId'], columns=play_group['nflId'])
            
            distances[game_id][play_id]['sorted_distances'] = distances[game_id][play_id]['dist_df'].apply(lambda row: row.sort_values().values.tolist(), axis=1)
            distances[game_id][play_id]['sorted_players'] = distances[game_id][play_id]['dist_df'].apply(lambda row: row.sort_values().index.tolist(), axis=1)
            
            distances[game_id][play_id]['n_closest_players'] = calc_n_closest_players(distances[game_id][play_id]['sorted_distances'], distances[game_id][play_id]['sorted_players'], n)
        
    return distances

cur_path = os.path.os.getcwd()
data_path = os.path.abspath(os.path.join(cur_path, './data/raw/'))

games = pd.read_csv(os.path.join(data_path, 'games.csv'))
player_play = pd.read_csv(os.path.join(data_path, 'player_play.csv'))
players = pd.read_csv(os.path.join(data_path, 'players.csv'))
plays = pd.read_csv(os.path.join(data_path, 'plays.csv'))

tracking_data = pd.DataFrame()
for week in [1]:
    tracking_data = pd.concat([tracking_data, pd.read_csv(os.path.join(data_path, f'tracking_week_{week}.csv'))])
    
print('Data read successfully')

plays = plays[plays['gameId'].isin(tracking_data['gameId'])]
plays['playResult'] = plays.apply(lambda x: play_result(x), axis=1)

pd.set_option('display.max_columns', None)
games = games[games['gameId'] == GAME_ID]
display(games[games['gameId'] == GAME_ID].head(1))

plays = plays[plays['playId'] == PLAY_ID]
display(plays.head(1))

tracking_data = tracking_data[(tracking_data['gameId'] == GAME_ID) & (tracking_data['playId'] == PLAY_ID)]
tracking_data = tracking_data[tracking_data['frameType'] == 'SNAP']

football = tracking_data[tracking_data['club'] == 'football'].copy()
display(football)
tracking_data = tracking_data[tracking_data['club'] != 'football']
display(tracking_data)

# dist_dict = calc_distance_between_players(tracking_data, 2) #! uncoment
delaunay_dict = calc_delaunay_connections(tracking_data)
gabriel_dict = calc_gabriel_connections(tracking_data)
rng_dict = calc_rng_connections(tracking_data)
mst_dict = calc_mst_connections(tracking_data)

direction = tracking_data.iloc[0]['playDirection']
absoluteYardlineNumber = plays.iloc[0]['absoluteYardlineNumber']
yardline_number = plays.iloc[0]['yardlineNumber']

if direction == 'left':
    highlight_line_number = absoluteYardlineNumber - 10
else:
    highlight_line_number = 110 - absoluteYardlineNumber
    
if highlight_line_number > 50:
    check_line = 50 - (50 - highlight_line_number)
    
if check_line == yardline_number:
    raise ValueError('Line of scrimmage is not correct')

coords = tracking_data[['x', 'y']].values
dist_matrix = cdist(coords, coords, metric='euclidean')

dist_df = pd.DataFrame(dist_matrix, index=tracking_data['nflId'], columns=tracking_data['nflId'])

sorted_distances = dist_df.apply(lambda row: row.sort_values().values.tolist(), axis=1)
sorted_players = dist_df.apply(lambda row: row.sort_values().index.tolist(), axis=1)

all_dist = {}
for index, value in sorted_distances.items():
    closest_players = []
    for i in range(1, len(sorted_players.loc[index])):
        # closest_players.append((sorted_players.loc[index][i], value[i]))
        closest_players.append({
            'nflId': sorted_players.loc[index][i],
            'distance': value[i]
        })
        if len(closest_players) == 2:
            break
    all_dist[index] = closest_players
    
fig, ax = createFootballField(highlight_line=True, highlight_line_number=absoluteYardlineNumber - 10)

colors = {'MIN': 'purple',
            'GB': 'green',
            'football': '#7b3f00'}
    
plt.scatter(football['x'].values[0], football['y'].values[0], color=colors[football['club'].values[0]])

# N == 2
# for key, value in all_dist.items():
#         key_x = tracking_data[tracking_data['nflId'] == key]['x'].values[0]
#         key_y = tracking_data[tracking_data['nflId'] == key]['y'].values[0]
#         for player in value:
#             player_x = tracking_data[tracking_data['nflId'] == player['nflId']]['x'].values[0]
#             player_y = tracking_data[tracking_data['nflId'] == player['nflId']]['y'].values[0]
#             plt.plot([key_x, player_x], [key_y, player_y], color='black', linewidth=1)

# DEALAUNAY GRAPH
# if delaunay_dict[GAME_ID][PLAY_ID]['triangulation'] is not None:
#     for simplex in delaunay_dict[GAME_ID][PLAY_ID]['triangulation'].simplices:
#         # Get player IDs for this triangle
#         players = [delaunay_dict[GAME_ID][PLAY_ID]['player_ids'][i] for i in simplex]
#         # Get coordinates for each player
#         coords = []
#         for player_id in players:
#             player_data = tracking_data[tracking_data['nflId'] == player_id]
#             if not player_data.empty:
#                 coords.append((player_data['x'].values[0], player_data['y'].values[0]))
        
#         # Draw the triangle if we have all coordinates
#         if len(coords) == 3:
#             plt.plot([coords[0][0], coords[1][0], coords[2][0], coords[0][0]], 
#                      [coords[0][1], coords[1][1], coords[2][1], coords[0][1]], 
#                      'k-', alpha=0.3)

# GABRIEL GRAPH
# draw_gabriel_graph(gabriel_dict, GAME_ID, PLAY_ID, tracking_data)

# RNG GRAPHdraw_rng_graph(rng_dict, GAME_ID, PLAY_ID, tracking_data)
# draw_rng_graph(rng_dict, GAME_ID, PLAY_ID, tracking_data)

# MST GRAPH
draw_mst_graph(mst_dict, GAME_ID, PLAY_ID, tracking_data)
            
for index, player in tracking_data.iterrows():
    x = player['x']
    y = player['y']
    s = player['displayName']
    if s == 'football':
        continue
    plt.scatter(x, y, color=colors[player['club']], zorder=100)
    
homeAbbr = games.iloc[0]['homeTeamAbbr']
visitorAbbr = games.iloc[0]['visitorTeamAbbr']
homePoints = plays.iloc[0]['preSnapHomeScore']
visitorPoints = plays.iloc[0]['preSnapVisitorScore']
quarter = plays.iloc[0]['quarter']
down = plays.iloc[0]['down']
yardsToGo = plays.iloc[0]['yardsToGo']
gameClock = plays.iloc[0]['gameClock']
description = plays.iloc[0]['playDescription']

downSufix = 'st' if down == 1 else 'nd' if down == 2 else 'rd' if down == 3 else 'th'
quarterSufix = 'st' if quarter == 1 else 'nd' if quarter == 2 else 'rd' if quarter == 3 else 'th'

playType = 'Pass' if plays.iloc[0]['playResult'] == 1 else 'Run' if plays.iloc[0]['playResult'] == 0 else 'Other'

title = f"{homeAbbr} ({homePoints}) vs {visitorAbbr} ({visitorPoints}) - {quarter}{quarterSufix} {gameClock} - {down}{downSufix} & {yardsToGo} ({playType} play)"

wrapped_title = textwrap.fill(title, width=100)
    
plt.title(wrapped_title, fontsize=14, color='black', loc='left', multialignment='left')

plt.show()