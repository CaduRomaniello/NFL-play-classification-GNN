import json
import random
import pandas as pd
import networkx as nx

from IPython.display import display

def graphs_create(plays: pd.DataFrame, tracking_data: pd.DataFrame, distances: dict) -> list[nx.Graph]:
    print("    Creating graphs...")
    # ['gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'possessionTeam', 'gameClock', 'absoluteYardlineNumber', 'offenseFormation', 'receiverAlignment', 'playClockAtSnap', 'possessionTeamPointDiff', 'playResult']
    graphs = []
    for i, play in plays.iterrows():
        gameId = play['gameId']
        playId = play['playId']
        
        graph_attrs = json.loads(play.drop(['gameId', 'playId']).to_json())
        G = nx.Graph(quarter=graph_attrs['quarter'], 
            down=graph_attrs['down'],
            yardsToGo=graph_attrs['yardsToGo'],
            possessionTeam=graph_attrs['possessionTeam'],
            gameClock=graph_attrs['gameClock'],
            absoluteYardlineNumber=graph_attrs['absoluteYardlineNumber'],
            offenseFormation=graph_attrs['offenseFormation'],
            receiverAlignment=graph_attrs['receiverAlignment'],
            playClockAtSnap=graph_attrs['playClockAtSnap'],
            possessionTeamPointDiff=graph_attrs['possessionTeamPointDiff'],
            playResult=graph_attrs['playResult']
        )
        
        for key, value in distances[gameId][playId]['n_closest_players'].items():
            for player in value:
                G.add_edge(key, player['nflId'], weight=player['distance'])
                
        iterator = tracking_data[(tracking_data['gameId'] == gameId) & (tracking_data['playId'] == playId)]
        for key, value in iterator.iterrows():
            relevant_info = ['club', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'height', 'weight', 'position', 'totalDis']
            info_dict = {k: v for k, v in value.items() if k in relevant_info}
            nx.set_node_attributes(G, {value['nflId']: info_dict})
            
        graphs.append(G)
        
    return graphs

def graphs_data_balancer(graphs):
    print("    Balancing graphs...")
    
    pass_graphs = []
    rush_graphs = []
    for graph in graphs:
        if graph.graph['playResult'] == 1:
            pass_graphs.append(graph)
        else:
            rush_graphs.append(graph)
            
    n_pass_graphs = len(pass_graphs)
    n_rush_graphs = len(rush_graphs)
    
    # random.shuffle(pass_graphs)
    # random.shuffle(rush_graphs)
    
    if (n_pass_graphs > n_rush_graphs):
        pass_graphs = pass_graphs[:n_rush_graphs]
    elif (n_pass_graphs < n_rush_graphs):
        rush_graphs = rush_graphs[:n_pass_graphs]
    
    return pass_graphs, rush_graphs