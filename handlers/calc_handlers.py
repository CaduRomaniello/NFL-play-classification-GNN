import pandas as pd
from IPython.display import display
from scipy.spatial.distance import cdist

def calc_possession_team_point_diff(plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    print("    Calculating possession team point difference...")
    
    plays['possessionTeamPointDiff'] = plays.apply(lambda x: point_diff(x, games), axis=1)
    return plays

def point_diff(p, games:pd.DataFrame):
    game_id = p['gameId']
    home_team = games[games['gameId'] == game_id]['homeTeamAbbr'].values[0]
    possession_team = p['possessionTeam']

    possession_team_point_diff = 0
    if possession_team == home_team:
        possession_team_point_diff = p['preSnapHomeScore'] - p['preSnapVisitorScore']
    else:
        possession_team_point_diff = p['preSnapVisitorScore'] - p['preSnapHomeScore']

    return possession_team_point_diff
        
def calc_game_clock_to_seconds(plays: pd.DataFrame) -> pd.DataFrame:
    print("    Calculating game clock to seconds...")
    
    plays['gameClock'] = plays['gameClock'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    return plays

def calc_total_dis(plays: pd.DataFrame) -> pd.DataFrame:
    print("    Calculating total distance...")
    
    # print("Calculating total distance...")

    # total_dis = plays.groupby(['gameId', 'playId', 'nflId'])['dis'].sum()
    # index_vals = plays.set_index(['gameId', 'playId', 'nflId']).index.map(total_dis)
    # plays.loc[:, 'totalDis'] = index_vals.map(total_dis)

    # return plays
    
    # #TODO: check this line of code
    # plays = plays.assign(totalDis=plays.groupby(['gameId', 'playId', 'nflId'])['dis'].transform('sum'))

    grouped = plays.groupby(['gameId', 'playId', 'nflId'])
    total_dis = grouped['dis'].sum().reset_index()
    total_dis.rename(columns={'dis': 'totalDis'}, inplace=True)
    plays = plays.merge(total_dis, on=['gameId', 'playId', 'nflId'])

    return plays

def calc_distance_between_players(tracking_data: pd.DataFrame, n: int = 2) -> dict:
    print("    Calculating distance between players...")
    
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

    coords = tracking_data[['x', 'y']].values
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    dist_df = pd.DataFrame(dist_matrix, index=tracking_data['nflId'], columns=tracking_data['nflId'])
    return dist_df

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