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
    """
    Mantém apenas os n menores valores em uma linha e zera os demais,
    retornando uma nova série sem modificar a original.
    """
    result = row.copy()
    row_copy = row.copy()
    row_copy[row_copy == 0] = np.inf
    smallest_indices = row_copy.nsmallest(n).index
    result[~result.index.isin(smallest_indices)] = 0
    result[result.index.isin(smallest_indices)] = 1
    
    return result  

# ---> start measuring time
start_time = datetime.now()
print(f"[{datetime.now()} - {datetime.now() - start_time}] Início da execução\n")

# ---> getting paths
cur_path = os.path.os.getcwd()
data_path = os.path.abspath(os.path.join(cur_path, 'Mestrado/nfl_data/2025/'))

# --- >loading data
games = pd.read_csv(os.path.join(data_path, 'games.csv'))
plays = pd.read_csv(os.path.join(data_path, 'plays.csv'))

# ---> filtering pass and rush plays
print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of plays before filtering by play type: {len(plays)}')
plays['play_type'] = plays.apply(pass_or_rush, axis=1)
plays = plays[plays['play_type'] != 'none']
print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of plays after filtering by play type: {len(plays)}\n')

# ---> adding week to plays
games.sort_values(['week'], ascending=[True], inplace=True)
games_info = games[['gameId', 'week']]
plays = pd.merge(plays, games_info, on='gameId')
plays.sort_values(['week'], ascending=[True], inplace=True)

# ---> filtering plays to only include week 1 |||| remove this when you want to process all weeks
# print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of plays before filtering by week: {len(plays)}')
# plays = plays[plays['week'] == 1]
# print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of plays after filtering by week: {len(plays)}\n')

# ---> separating plays into pass and rush
print(f'[{datetime.now()} - {datetime.now() - start_time}] Separating plays into pass and rush plays...')
pass_plays = plays[plays['play_type'] == 'pass']
pass_plays = pass_plays.reset_index(drop=True)

rush_plays = plays[plays['play_type'] == 'rush']
rush_plays = rush_plays.reset_index(drop=True)

print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of pass plays: {len(pass_plays)}')
print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of rush plays: {len(rush_plays)}\n')

# ---> files to process
files = ['tracking_week_1.csv', 'tracking_week_2.csv', 'tracking_week_3.csv', 'tracking_week_4.csv', 'tracking_week_5.csv', 'tracking_week_6.csv', 'tracking_week_7.csv', 'tracking_week_8.csv', 'tracking_week_9.csv']
# files = ['tracking_week_1.csv']

# ---> reading data
data = pd.DataFrame()
start_time = datetime.now()
for i, file in enumerate(files):
    print(f'[{datetime.now()} - {datetime.now() - start_time}] Reading file {i+1}/{len(files)}: {file}')
    data = pd.concat([data, pd.read_csv(data_path + '/' + file)])
print(f'[{datetime.now()} - {datetime.now() - start_time}] Finished reading tracking data\n')

# ---> cleaning data
print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of rows in tracking data before filtering by frame type: {len(data)}')
data = data[(data['frameType'] == 'SNAP') & (data['displayName'] != "football")]
print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of rows in tracking data after filtering by frame type: {len(data)}\n')

# ---> filtering data to only include plays that are in the plays dataframe
plays['game_play_key'] = plays['gameId'].astype(str) + '_' + plays['playId'].astype(str)
data['game_play_key'] = data['gameId'].astype(str) + '_' + data['playId'].astype(str)

print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of plays before filtering tracking data by plays: {len(data)}')
data = data[data['game_play_key'].isin(plays['game_play_key'])]
print(f'[{datetime.now()} - {datetime.now() - start_time}] Number of plays after filtering tracking data by plays: {len(data)}\n')

data.drop('game_play_key', axis=1, inplace=True)
plays.drop('game_play_key', axis=1, inplace=True)

# ---> calculating distance matrices for all plays
dist_matrices = {}
print(f'[{datetime.now()} - {datetime.now() - start_time}] Calculating distance matrices for {len(plays)} plays...')
for i, play in plays.iterrows():
    # filtering data for the current play
    play_data = data[(data['gameId'] == play['gameId']) & (data['playId'] == play['playId'])]
    
    if play_data.empty:
        continue
    
    # calculating distance matrix for the current play
    coords = play_data[['x', 'y']].values
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # storing the distance matrix in a dictionary with gameId and playId as key
    dist_matrices[f'{play["gameId"]}_{play["playId"]}'] = {
        "distance_matrix": dist_matrix,
        "nflIds": play_data['nflId']
    }

print(f'[{datetime.now()} - {datetime.now() - start_time}] Finished calculating distance matrices')

# ---> initializing list to store the results
results = []

# ---> calculating frobenius norm for all type of edges combinations
for n in range(1, 22):
    print(f'\n[{datetime.now()} - {datetime.now() - start_time}] Calculating Frobenius norm for n={n}')

    sample_key = list(dist_matrices.keys())[0]  # Pega a primeira chave
    sample_matrix = dist_matrices[sample_key]['distance_matrix']
    total_elements = sample_matrix.size
    zero_elements = (sample_matrix == 0).sum()
    print(f'[{datetime.now()} - {datetime.now() - start_time}]     dist_matrices for n={n} has {total_elements} elements, of which {zero_elements} are zero ({(zero_elements / total_elements) * 100:.2f}% zeros)')

    # filtering dist matrices to contain only the n closest players
    n_closest = {}
    print(f'[{datetime.now()} - {datetime.now() - start_time}]     Keeping only the {n} closest players for each play...')
    for i, play in plays.iterrows():
        # keeping only the n smallest distances for each player
        dist_df = pd.DataFrame(dist_matrices[f"{play['gameId']}_{play['playId']}"]['distance_matrix'],
                               index=dist_matrices[f"{play['gameId']}_{play['playId']}"]['nflIds'],
                               columns=dist_matrices[f"{play['gameId']}_{play['playId']}"]['nflIds'])
        dist_df = dist_df.apply(lambda row: keep_n_smallest(row, n), axis=1)
        n_closest[f"{play['gameId']}_{play['playId']}"] = dist_df.values
    print(f'[{datetime.now()} - {datetime.now() - start_time}]     Finished keeping only the {n} closest players for each play')

    sample_key = list(n_closest.keys())[0]  # Pega a primeira chave
    sample_matrix = n_closest[sample_key]
    total_elements = sample_matrix.size
    zero_elements = (sample_matrix == 0).sum()
    print(f'[{datetime.now()} - {datetime.now() - start_time}]     Sample distance matrix for n={n} has {total_elements} elements, of which {zero_elements} are zero ({(zero_elements / total_elements) * 100:.2f}% zeros)')

    # initializing list to store Frobenius norms
    norms = []
    # calculating frobenius norm
    for i, pass_play in pass_plays.iterrows():
        if i % 100 == 0:
            print(f'[{datetime.now()} - {datetime.now() - start_time}]     Processing pass play {i}/{len(pass_plays)}')
        # if i == 10:
        #     break
        
        pass_dist_matrix = n_closest[f"{pass_play['gameId']}_{pass_play['playId']}"]
        
        # iterating through rush plays to calculate Frobenius norm from the difference of distance matrices
        for j, rush_play in rush_plays.iterrows():
            rush_dist_matrix = n_closest[f"{rush_play['gameId']}_{rush_play['playId']}"]
            
            # calculating Frobenius norm
            frobenius_norm = np.linalg.norm(pass_dist_matrix - rush_dist_matrix)
            norms.append(frobenius_norm)

    results.append({
        'n': n,
        'mean_frobenius_norm': np.mean(norms) if norms else None,
        'std_frobenius_norm': np.std(norms) if norms else None
    })
    print(f'[{datetime.now()} - {datetime.now() - start_time}] Finished calculating Frobenius norm for n={n} with results: {json.dumps(results[-1])}')
    
print(f"\n[{datetime.now()} - {datetime.now() - start_time}] Fim da execução")

# ---> saving results
df_distances = pd.DataFrame(results)
df_distances.to_csv(os.path.join('./', 'Mestrado/frobenius_norm.csv'))

json_path = os.path.join('./', 'Mestrado/frobenius_norm.json')
with open(json_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)