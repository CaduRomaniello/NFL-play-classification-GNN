import os
import gc
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist

# classifying plays based on pass or rush
def pass_or_rush(play):
    if not pd.isna(play['qbSpike']) and play['qbSpike']:
        return 'none'
    elif not pd.isna(play['qbKneel']) and play['qbKneel']:
        return 'none'
    elif not pd.isna(play['qbSneak']) and play['qbSneak']:
        return 'none'
    elif play['passResult'] == 'R':
        return 'none'
    elif not pd.isna(play['rushLocationType']):
        return 'rush'
    elif not pd.isna(play['passLocationType']):
        return 'pass'
    elif not pd.isna(play['passResult']):
        # print("PASS WITHOUT INFO")
        return 'pass'
    else:
        # print("can't determine play type")
        return 'none'
    
# keeping only the n smallest values in a row for Frobenius norm calculation
def keep_n_smallest(row, n):
    row_copy = row.copy()
    row_copy[row_copy == 0] = np.inf
    smallest_indices = row_copy.nsmallest(n).index
    row[~row.index.isin(smallest_indices)] = 0
    return row
    
# getting paths
cur_path = os.path.os.getcwd()
# print(cur_path)
data_path = os.path.abspath(os.path.join(cur_path, './nfl_data/2025/'))
# print(data_path)
distances_path = os.path.abspath(os.path.join(cur_path, './frobenius_norm/'))
# print(distances_path)

# loading data
games = pd.read_csv(os.path.join(data_path, 'games.csv'))
plays = pd.read_csv(os.path.join(data_path, 'plays.csv'))

# filtrando apenas as jogadas de passe ou corrida
print(f'Number of plays before filtering by play type: {len(plays)}')

plays['play_type'] = plays.apply(pass_or_rush, axis=1)
plays = plays[plays['play_type'] != 'none']

print(f'Number of plays after filtering by play type: {len(plays)}')

# adding week to plays
games.sort_values(['week'], ascending=[True], inplace=True)

games_info = games[['gameId', 'week']]
plays = pd.merge(plays, games_info, on='gameId')
plays.sort_values(['week'], ascending=[True], inplace=True)

# filtering plays to only include week 1 |||| remove this when you want to process all weeks
print(f'Number of plays before filtering by week: {len(plays)}')

plays = plays[plays['week'] == 1]

print(f'Number of plays after filtering by week: {len(plays)}')

# separating plays into pass and rush
pass_plays = plays[plays['play_type'] == 'pass']
pass_plays = pass_plays.reset_index(drop=True)

rush_plays = plays[plays['play_type'] == 'rush']
rush_plays = rush_plays.reset_index(drop=True)

print(f'Number of pass plays: {len(pass_plays)}')
print(f'Number of rush plays: {len(rush_plays)}')

# files to process
# files = ['tracking_week_1.csv', 'tracking_week_2.csv', 'tracking_week_3.csv', 'tracking_week_4.csv', 'tracking_week_5.csv', 'tracking_week_6.csv', 'tracking_week_7.csv']
files = ['tracking_week_1.csv']

# reading data
data = pd.DataFrame()
for i, file in enumerate(files):
    data = pd.concat([data, pd.read_csv(data_path + '/' + file)])
    
print(f'Number of rows in tracking data before filtering by frame type: {len(data)}')

data = data[(data['frameType'] == 'SNAP') & (data['displayName'] != "football")]

print(f'Number of rows in tracking data after filtering by frame type: {len(data)}')

# filtering data to only include plays that are in the plays dataframe
plays['game_play_key'] = plays['gameId'].astype(str) + '_' + plays['playId'].astype(str)
data['game_play_key'] = data['gameId'].astype(str) + '_' + data['playId'].astype(str)

print(f'Number of plays before filtering tracking data by plays: {len(data)}')

data = data[data['game_play_key'].isin(plays['game_play_key'])]

print(f'Number of plays after filtering tracking data by plays: {len(data)}')

data.drop('game_play_key', axis=1, inplace=True)
plays.drop('game_play_key', axis=1, inplace=True)

# initializing list to store the results
results = []

start_time = datetime.now()
print(f"Início da execução: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# calculating frobenius norm for all type of edges combinations
for n in range(1, 2):
    print(f'Calculating Frobenius norm for n={n} - {datetime.now() - start_time} elapsed')
    # initializing list to store Frobenius norms
    norms = []
    # calculating frobenius norm
    for i, pass_play in pass_plays.iterrows():
        iteration_time = datetime.now()
        print(f'    Processing pass play {i}/{len(pass_plays)} - {iteration_time - start_time} elapsed')
        if i == 2:
            break
        
        # filtering pass play data
        play_data = data[(data['gameId'] == pass_play['gameId']) & (data['playId'] == pass_play['playId'])]
        
        if play_data.empty:
            continue
        
        # calculating distance matrix for pass play
        pass_coords = play_data[['x', 'y']].values
        pass_dist_matrix = cdist(pass_coords, pass_coords, metric='euclidean')
        
        # keeping only the 2 smallest distances for each player
        pass_dist_df = pd.DataFrame(pass_dist_matrix, index=play_data['nflId'], columns=play_data['nflId'])
        pass_dist_df = pass_dist_df.apply(lambda row: keep_n_smallest(row, n), axis=1)
        pass_dist_matrix = pass_dist_df.values
        
        # iterating through rush plays to calculate Frobenius norm from the difference of distance matrices
        for j, rush_play in rush_plays.iterrows():
            # filtering rush play data
            rush_data = data[(data['gameId'] == rush_play['gameId']) & (data['playId'] == rush_play['playId'])]
            
            if rush_data.empty:
                continue
            
            # calculating distance matrix for rush play
            rush_coords = rush_data[['x', 'y']].values
            rush_dist_matrix = cdist(rush_coords, rush_coords, metric='euclidean')
            
            # keeping only the 2 smallest distances for each player
            rush_dist_df = pd.DataFrame(rush_dist_matrix, index=rush_data['nflId'], columns=rush_data['nflId'])
            rush_dist_df = rush_dist_df.apply(lambda row: keep_n_smallest(row, n), axis=1)
            rush_dist_matrix = rush_dist_df.values
            
            # calculating Frobenius norm
            frobenius_norm = np.linalg.norm(pass_dist_matrix - rush_dist_matrix)
            
            norms.append(frobenius_norm)
            
    results.append({
        'n': n,
        'mean_frobenius_norm': np.mean(norms) if norms else None,
        'std_frobenius_norm': np.std(norms) if norms else None
    })
    
print(f"Fim da execução: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# saving results
df_distances = pd.DataFrame(results)
df_distances.to_csv(os.path.join('./', 'frobenius_norm.csv'))

json_path = os.path.join('./', 'frobenius_norm.json')
with open(json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)