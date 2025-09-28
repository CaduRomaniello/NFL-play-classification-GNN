# classe para preprocessar os dados
import json
import pandas as pd
import networkx as nx

from IPython.display import display
from scipy.spatial.distance import cdist
from sklearn.calibration import LabelEncoder

from src.utils.logger import Logger
from src.data.graph_strategies.strategy_factory import GraphStrategyFactory

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.strategy = GraphStrategyFactory.create_strategy(config)

    def execute(self, games: pd.DataFrame, player_play: pd.DataFrame, players: pd.DataFrame, plays: pd.DataFrame, tracking_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        Logger.info('Starting data preprocessing execution')
        plays = plays[plays['gameId'].isin(tracking_data['gameId'])].copy()
        plays = self._calc_possession_team_point_diff(plays, games).copy()
        plays = self._verify_plays_result(plays).copy()
        plays, tracking_data = self._verify_invalid_values(plays, tracking_data)
        plays = self._calc_game_clock_to_seconds(plays)

        td_before_snap = tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP')]
        tracking_data = self._calc_total_dis(td_before_snap)
        tracking_data = tracking_data[tracking_data['frameType'] == 'SNAP']

        football = tracking_data[(tracking_data['displayName'] == 'football')]
        tracking_data = self._merge_player_info(players, tracking_data)

        plays = plays.dropna(subset=['playResult']).copy()
        tracking_data['playDirection'] = tracking_data.apply(lambda row: 0 if row['playDirection'] == 'left' else 1, axis=1)

        le_offenseFormation = LabelEncoder()
        le_receiverAlignment = LabelEncoder()
        le_possessionTeam = LabelEncoder()
        plays["offenseFormation"] = le_offenseFormation.fit_transform(plays["offenseFormation"])
        plays["receiverAlignment"] = le_receiverAlignment.fit_transform(plays["receiverAlignment"])
        plays['possessionTeam'] = le_possessionTeam.fit_transform(plays['possessionTeam'])

        le_club = LabelEncoder()
        le_position = LabelEncoder()
        tracking_data['club'] = le_club.fit_transform(tracking_data['club'])
        tracking_data['position'] = le_position.fit_transform(tracking_data['position'])

        connections_dict = self.strategy.calculate_connections(tracking_data, players)

        # self.strategy.draw(connections_dict, 2022091200, 64, tracking_data[(tracking_data['gameId'] == 2022091200) & (tracking_data['playId'] == 64)], ax=None)

        return plays, tracking_data, connections_dict

    def _point_diff(self, p, games: pd.DataFrame):
        game_id = p['gameId']
        home_team = games[games['gameId'] == game_id]['homeTeamAbbr'].values[0]
        possession_team = p['possessionTeam']

        possession_team_point_diff = 0
        if possession_team == home_team:
            possession_team_point_diff = p['preSnapHomeScore'] - p['preSnapVisitorScore']
        else:
            possession_team_point_diff = p['preSnapVisitorScore'] - p['preSnapHomeScore']

        return possession_team_point_diff

    def _calc_possession_team_point_diff(self, plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
        Logger.info('Calculating possession team point difference...')
        plays['possessionTeamPointDiff'] = plays.apply(lambda x: self._point_diff(x, games), axis=1)
        return plays

    def _verify_plays_result(self, plays: pd.DataFrame) -> pd.DataFrame:
        Logger.info('Verifying plays result...')
        
        plays['playResult'] = plays.apply(lambda x: self._play_result(x), axis=1)
        return plays

    def _play_result(self, play: pd.Series) -> int:
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

    def _verify_invalid_values(self, plays: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        Logger.info('Verifying invalid values...')
        plays.fillna({'receiverAlignment': 'EMPTY'}, inplace=True)
        plays.fillna({'offenseFormation': 'EMPTY'}, inplace=True)
        plays.fillna({'playClockAtSnap': 0}, inplace=True)
        
        snap_mask = tracking['frameType'] == 'SNAP'
        tracking.loc[snap_mask, 'o'] = tracking.loc[snap_mask].apply(lambda t: self._o_invalid_values(plays, t), axis=1)
        tracking.loc[snap_mask, 'dir'] = tracking.loc[snap_mask].apply(lambda t: self._dir_invalid_values(plays, t), axis=1)
        
        return plays, tracking

    def _o_invalid_values(self, plays: pd.DataFrame, t) -> int:
        game_id = t['gameId']
        play_id = t['playId']
        
        return 90 if pd.isna(t['o']) and t['playDirection'] == 'right' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
                270 if pd.isna(t['o']) and t['playDirection'] == 'right' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
                270 if pd.isna(t['o']) and t['playDirection'] == 'left' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
                90 if pd.isna(t['o']) and t['playDirection'] == 'left' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else t['o']

    def _dir_invalid_values(self, plays: pd.DataFrame, t) -> int:
        game_id = t['gameId']
        play_id = t['playId']

        return 90 if pd.isna(t['dir']) and t['playDirection'] == 'right' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
                270 if pd.isna(t['dir']) and t['playDirection'] == 'right' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
                270 if pd.isna(t['dir']) and t['playDirection'] == 'left' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
                90 if pd.isna(t['dir']) and t['playDirection'] == 'left' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else t['dir']

    def _calc_game_clock_to_seconds(self, plays: pd.DataFrame) -> pd.DataFrame:
        Logger.info('Calculating game clock to seconds...')
        
        plays['gameClock'] = plays['gameClock'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
        return plays

    def _calc_total_dis(self, plays: pd.DataFrame) -> pd.DataFrame:
        Logger.info('Calculating total distance...')

        grouped = plays.groupby(['gameId', 'playId', 'nflId'])
        total_dis = grouped['dis'].sum().reset_index()
        total_dis.rename(columns={'dis': 'totalDis'}, inplace=True)
        plays = plays.merge(total_dis, on=['gameId', 'playId', 'nflId'])

        return plays
    
    def _merge_player_info(self, players: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
        Logger.info('Merging player info...')

        players['height'] = players['height'].str.split('-').apply(lambda x: round((int(x[0]) * 12 + int(x[1])) * 2.54))
        players['weight'] = round(players['weight'] * 0.45359237)
        
        players_info = players[['nflId', 'height', 'weight', 'position']]
        t = pd.merge(tracking, players_info, on='nflId')
        return t

    # def _calc_distance_between_players(self, tracking_data: pd.DataFrame, players: pd.DataFrame, n: int = 2) -> dict:
    #     Logger.info('Calculating distance between players...')
    #     distances = {}
    #     grouped_by_game = tracking_data.groupby('gameId')
        
    #     for game_id, game_group in grouped_by_game:
    #         distances[game_id] = {}
    #         grouped_by_play = game_group.groupby('playId')
            
    #         for play_id, play_group in grouped_by_play:
    #             distances[game_id][play_id] = {}
                
    #             coords = play_group[['x', 'y']].values
    #             dist_matrix = cdist(coords, coords, metric='euclidean')
                
    #             distances[game_id][play_id]['dist_df'] = pd.DataFrame(dist_matrix, index=play_group['nflId'], columns=play_group['nflId'])
                
    #             distances[game_id][play_id]['sorted_distances'] = distances[game_id][play_id]['dist_df'].apply(lambda row: row.sort_values().values.tolist(), axis=1)
    #             distances[game_id][play_id]['sorted_players'] = distances[game_id][play_id]['dist_df'].apply(lambda row: row.sort_values().index.tolist(), axis=1)
                
    #             distances[game_id][play_id]['n_closest_players'] = self._calc_n_closest_players(distances[game_id][play_id]['sorted_distances'], distances[game_id][play_id]['sorted_players'], n, players, play_group)
            
    #     return distances
    
    # def _calc_n_closest_players(self, sorted_distances: list, sorted_players: list, n: int, players: pd.DataFrame, tracking_data:pd.DataFrame) -> dict:
    #     all_dist = {}
    #     for index, value in sorted_distances.items():
    #         closest_players = []
    #         for i in range(1, len(sorted_players.loc[index])):
    #             if (players.loc[players['nflId'] == index]['position'].values[0] != 'QB'):
    #                 closest_players.append({
    #                     'nflId': sorted_players.loc[index][i],
    #                     'distance': value[i]
    #                 })
    #                 if len(closest_players) == n:
    #                     break
    #             else:
    #                 if tracking_data[tracking_data['nflId'] == index]['club'].values[0] == tracking_data[tracking_data['nflId'] == sorted_players.loc[index][i]]['club'].values[0]:
    #                     closest_players.append({
    #                         'nflId': sorted_players.loc[index][i],
    #                         'distance': value[i]
    #                     })
    #         all_dist[index] = closest_players
            
    #     return all_dist
    