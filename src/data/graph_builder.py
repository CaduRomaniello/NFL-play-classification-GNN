# classe para criar os grafos e fazer um down sample neles

import json
import pandas as pd
import networkx as nx

from src.utils.logger import Logger


class GraphBuilder:
    def __init__(self, config):
        self.config = config

    def execute(self, plays: pd.DataFrame, tracking_data: pd.DataFrame, dist_dict: dict, downSample: bool = True) -> tuple[list[nx.Graph], list[nx.Graph]]:
        graphs = self._graphs_create(plays, tracking_data, dist_dict)
        pass_graphs, rush_graphs = self._graphs_splitter(graphs, downSample)
        return pass_graphs, rush_graphs

    def _graphs_create(self, plays: pd.DataFrame, tracking_data: pd.DataFrame, distances: dict) -> list[nx.Graph]:
        Logger.info('Creating graphs...')
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
            
            for key, value in distances[gameId][playId]['connections'].items():
                for player in value:
                    G.add_edge(key, player['nflId'], weight=player['distance'])
                    
            iterator = tracking_data[(tracking_data['gameId'] == gameId) & (tracking_data['playId'] == playId)]
            for key, value in iterator.iterrows():
                relevant_info = ['club', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'height', 'weight', 'position', 'totalDis']
                info_dict = {k: v for k, v in value.items() if k in relevant_info}
                nx.set_node_attributes(G, {value['nflId']: info_dict})
                
            graphs.append(G)
            
        return graphs

    def _graphs_splitter(self, graphs: list[nx.Graph], downSample: bool) -> tuple[list[nx.Graph], list[nx.Graph]]:
        Logger.info('Splitting graphs...')

        pass_graphs = []
        rush_graphs = []
        for graph in graphs:
            if graph.graph['playResult'] == 1:
                pass_graphs.append(graph)
            else:
                rush_graphs.append(graph)

        if downSample:  
            n_pass_graphs = len(pass_graphs)
            n_rush_graphs = len(rush_graphs)
            
            if (n_pass_graphs > n_rush_graphs):
                pass_graphs = pass_graphs[:n_rush_graphs]
            elif (n_pass_graphs < n_rush_graphs):
                rush_graphs = rush_graphs[:n_pass_graphs]
        
        return pass_graphs, rush_graphs