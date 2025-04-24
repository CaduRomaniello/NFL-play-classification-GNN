from contextlib import redirect_stdout
from datetime import datetime
import time
import json
import random

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from handlers.calc_handlers import calc_distance_between_players, calc_game_clock_to_seconds, calc_n_closest_players, calc_possession_team_point_diff, calc_total_dis
from handlers.graph_handlers import graphs_create, graphs_data_balancer
from handlers.merge_handlers import merge_player_info
from handlers.model_handlers_v2 import model_run
# from handlers.model_handlers_ import model_run
from handlers.model_handlers_v2 import convert_nx_to_pytorch_geometric
from handlers.verify_handlers import verify_invalid_values, verify_plays_result
from playground import playground
from IPython.display import display
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from data_handlers.read_files import read2025data
from visualization.create_plot import createFootballField

# the team that have 
PLAY_RELEVANT_COLUMNS = ['gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'possessionTeam', 'gameClock', 'absoluteYardlineNumber', 'offenseFormation', 'receiverAlignment', 'playClockAtSnap', 'possessionTeamPointDiff', 'playResult']
TRACKING_RELEVANTCOLUMNS = ['nflId', 'club', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'height', 'weight', 'position', 'totalDis']
N_CLOSEST_PLAYERS = 2
RANDOM_SEED = 1
NUMBER_OF_ITERS = 5

CONFIG = {
    'RANDOM_SEED': 1,
    'GNN_EPOCHS': 50,
    'GNN_HIDDEN_CHANNELS': 64,
    'GNN_HIDDEN_LAYERS': 3,
    'GNN_LEARNING_RATE': 0.0001,
    'GNN_DROPOUT': 0.5,
    'GNN_WEIGHT_DECAY': 5e-4,
    'RF_ESTIMATORS': 100,
    'MLP_HIDDEN_CHANNELS': 64,
    'MLP_HIDDEN_LAYERS': 2,
    'MLP_MAX_ITER':3000,
    'MLP_LEARNING_RATE': 0.01,
    'MLP_ALPHA': 5e-4,
    'DATASET_SPLIT': 0.8,
    'SHOW_INFO': True,
}

def main():    
    weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rush_graphs = []
    pass_graphs = []
    random_seed = 0
    random.seed(random_seed)
    random_seed = 13
    
    for week in weeks:
        print('--------------------------------------------------------')
        print(f'Getting data for week {week}...')
        week_pass_graphs, week_rush_graphs = getGraphs([week])
        
        pass_graphs.extend(week_pass_graphs)
        rush_graphs.extend(week_rush_graphs)
        print()
        
    for i in range(NUMBER_OF_ITERS):
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = f"output/{timestamp}.txt"
        random_seed += 1
        random.seed(random_seed)
        
        with open(output_filename, "w", encoding="utf-8") as output_file:
            # Redirecionar os logs para o arquivo
            with redirect_stdout(output_file):
                print(f"Iteration {i + 1}/{NUMBER_OF_ITERS}")
                print(f"Output file: {output_filename}")
                
                CONFIG['RANDOM_SEED'] = random_seed  # Exemplo: entre 10 e 100 épocas
                CONFIG['GNN_EPOCHS'] = random.choice([100, 200, 300])  # Exemplo: 32, 64 ou 128
                CONFIG['GNN_HIDDEN_CHANNELS'] = random.choice([32, 64, 128])  # Exemplo: entre 0.0001 e 0.01
                CONFIG['GNN_HIDDEN_LAYERS'] = random.choice([1, 2, 3])
                CONFIG['GNN_LEARNING_RATE'] = random.choice([0.001, 0.0001, 0.00001])  # Exemplo: entre 0.0001 e 0.01
                CONFIG['GNN_DROPOUT'] = random.choice([0.2, 0.3, 0.4, 0.5])
                CONFIG['GNN_WEIGHT_DECAY'] = random.choice([5e-3, 5e-4, 5e-5])
                CONFIG['RF_ESTIMATORS'] = random.choice([50, 100, 150])
                CONFIG['MLP_HIDDEN_CHANNELS'] = random.choice([32, 64, 128]) 
                CONFIG['MLP_HIDDEN_LAYERS'] = random.choice([1, 2, 3]) 
                CONFIG['MLP_LEARNING_RATE'] = random.choice([0.01, 0.001, 0.0001]) 
                CONFIG['MLP_ALPHA'] = random.choice([5e-3, 5e-4, 5e-5])
                
                print("CONFIG values:")
                for key, value in CONFIG.items():
                    print(f"{key}: {value}")
        
                model_run(pass_graphs, rush_graphs, config=CONFIG)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Iteration {i + 1} completed. Logs saved to {output_filename}. Duration: {duration:.2f} seconds")
        
        
        
        

def getGraphs(weeks=[1]):
    #* reading data
    # games, player_play, players, plays, tracking_data = read2025data()
    games, player_play, players, plays, tracking_data = read2025data(weeks=weeks)
    
    #* removing test data
    plays = plays[plays['gameId'].isin(tracking_data['gameId'])]
    
    #* getting possession team point diff
    plays = calc_possession_team_point_diff(plays, games) #! uncomment
        
    #* getting pass or rush play type
    plays = verify_plays_result(plays) #! uncoment
    
    #* verifying invalid values
    plays, tracking_data = verify_invalid_values(plays, tracking_data) #! uncoment
    
    #* calculating game clock to seconds    
    plays = calc_game_clock_to_seconds(plays) #! uncoment
    
    #* adding total distance before snap to tracking data and filtering for events of type 'SNAP'
    td_before_snap = tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP')]
    tracking_data = calc_total_dis(td_before_snap) #! uncoment
    tracking_data = tracking_data[tracking_data['frameType'] == 'SNAP']
    
    #* retrieving football info and transforming players data and adding player info to tracking dataclear
    football = tracking_data[(tracking_data['displayName'] == 'football')]
    tracking_data = merge_player_info(players, tracking_data) #! uncoment
    
    #* dropping rows with nan values for playResult
    plays = plays.dropna(subset=['playResult'])
    
    #* enconding categorical variables
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
    
    # display(plays.head(5))
    # print('--------------------------------------------------------')
    # print('--------------------------------------------------------')
    # print('--------------------------------------------------------')
    # print('--------------------------------------------------------')
    # print('--------------------------------------------------------')
    # print('--------------------------------------------------------')
    # display(tracking_data.head(5))
    
    #! remember to calculate it per gameId and playId and return a dictionary of distances of type:
    #!  {
    #!      'gameId': {
    #!          'playId': {distances_df},
    #!          'playId': {distances_df},
    #!          ...
    #!      }
    #!  }
    #* calculating distances between players, sorting them and finding n closest players
    dist_dict = calc_distance_between_players(tracking_data, N_CLOSEST_PLAYERS) #! uncoment
    
    #* creating graphs
    graphs = graphs_create(plays, tracking_data, dist_dict) #! uncoment
    
    #* filtering plays and tracking_data df for relevant columns
    plays = plays[PLAY_RELEVANT_COLUMNS]
    tracking_data = tracking_data[TRACKING_RELEVANTCOLUMNS]
    
    # print(plays['playResult'].value_counts())
    
    #* Balancing the dataset
    pass_graphs, rush_graphs = graphs_data_balancer(graphs) #! uncoment
    
    #* running the model
    # model_run(pass_graphs, rush_graphs, epochs=300, show_info=True)
    
    # a = convert_nx_to_pytorch_geometric(graphs, include_labels=True)
    
    return pass_graphs, rush_graphs




    


    
    
    
    
def old_main():
    games, player_play, players, plays, tracking_data = read2025data()
    
    # print(games.isna().sum())
    # print()
    # print(player_play.isna().sum())
    # print()
    # print(players.isna().sum())
    # print()
    # print(plays.isna().sum())
    # print()
    # print(tracking_data.isna().sum())
    # print()
    
    game_id = 2022091110
    play_id = 55
    
    #* filtering df's
    game = games[games['gameId'] == game_id]
    play = plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]
    tracking_data = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id)]
    
    yardline_number = play['yardlineNumber'].values[0] #? arrumar aqui caso necessário
    
    #* getting possession team point diff
    home_team = games['homeTeamAbbr'].values[0]
    possession_team = play['possessionTeam'].values[0]
    
    possession_team_point_diff = 0
    if possession_team == home_team:
        possession_team_point_diff = play['preSnapHomeScore'].values[0] - play['preSnapVisitorScore'].values[0]
    else:
        possession_team_point_diff = play['preSnapVisitorScore'].values[0] - play['preSnapHomeScore'].values[0]
        
        
    #* getting pass or rush play type
    passLocationType = play['passLocationType'].values[0]
    rushLocationType = play['rushLocationType'].values[0]
    
    playType = ''
    if pd.isna(play['passLocationType'].values[0]) and pd.isna(play['rushLocationType'].values[0]):
        raise ValueError('play does not have passLocationType or rushLocationType')
    elif not pd.isna(play['passLocationType'].values[0]) and not pd.isna(play['rushLocationType'].values[0]):
        raise ValueError('play has both passLocationType and rushLocationType')
    else:
        playType = 'pass' if pd.isna(play['rushLocationType']).empty else 'rush'
        
    if not pd.isna(play['qbSpike'].values[0]) and play['qbSpike'].values[0]:
        playType = 'none'
    elif not pd.isna(play['qbKneel'].values[0]) and play['qbKneel'].values[0]:
        playType = 'none'
    elif not pd.isna(play['qbSneak'].values[0]) and play['qbSneak'].values[0]:
        playType = 'none'
    elif play['passResult'].values[0] == 'R':
        playType = 'none'
    elif not pd.isna(play['rushLocationType'].values[0]):
        playType = 'rush'
    elif not pd.isna(play['passLocationType'].values[0]):
        playType = 'pass'
    elif not pd.isna(play['passResult'].values[0]):
        print("PASS WITHOUT INFO")
        playType = 'pass'
    else:
        print("can't determine play type")
        playType = 'none'
        
    if playType == 'none':
        raise ValueError('playType is none')
    
    play['playType'] = 0 if playType == 'rush' else 1
    # play['playType'] = 0 if playType == 'rush' else 1
    
    
    #* adding possession team point diff, dealing with nan values, transforming gameClock to seconds
    play['possessionTeamPointDiff'] = possession_team_point_diff
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    display(play['possessionTeamPointDiff'])
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    
    play.fillna({'receiverAlignment': 'EMPTY'}, inplace=True)
    play.fillna({'offenseFormation': 'EMPTY'}, inplace=True)
    play.fillna({'playClockAtSnap': 0}, inplace=True)
    
    
    play['gameClock'] = play['gameClock'].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    
    
    #* filtering play df for relevant columns
    play = play[PLAY_RELEVANT_COLUMNS]

    
    #* adding play type to play df
    play['playType'] = playType
    
    #* verifying if there is only one line_set event
    if tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP') & (tracking_data['event'] == 'line_set')]['frameId'].nunique() != 1:
        raise ValueError('There is more than one line_set event')
    
    #* filtering tracking data for events after line_set
    line_set_frame_id = tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP') & (tracking_data['event'] == 'line_set')]['frameId'].values[0]
    td_before_snap = tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP')]
    td_at_snap = td_before_snap[td_before_snap['frameType'] == 'SNAP']
    
    #* getting football info
    football = td_at_snap[(td_at_snap['displayName'] == 'football')]
    
    #* transforming players data and adding player info to tracking data
    
    #* delaing with nan values for 'o' and 'dir' columns
    td_at_snap['o'] = td_at_snap.apply(lambda row: 90 if pd.isna(row['o']) and row['playDirection'] == 'right' and row['club'] == play['possessionTeam'].values[0] 
                                            else (270 if pd.isna(row['o']) and row['playDirection'] == 'right' and row['club'] != play['possessionTeam'].values[0]
                                                else (270 if pd.isna(row['o']) and row['playDirection'] == 'left' and row['club'] == play['possessionTeam'].values[0] 
                                                    else (90 if pd.isna(row['o']) and row['playDirection'] == 'left' and row['club'] != play['possessionTeam'].values[0]
                                                        else row['o']))), axis=1)

    td_at_snap['dir'] = td_at_snap.apply(lambda row: 90 if pd.isna(row['o']) and row['playDirection'] == 'right' and row['club'] == play['possessionTeam'].values[0] 
                                            else (270 if pd.isna(row['o']) and row['playDirection'] == 'right' and row['club'] != play['possessionTeam'].values[0]
                                                else (270 if pd.isna(row['o']) and row['playDirection'] == 'left' and row['club'] == play['possessionTeam'].values[0] 
                                                    else (90 if pd.isna(row['o']) and row['playDirection'] == 'left' and row['club'] != play['possessionTeam'].values[0]
                                                        else row['o']))), axis=1)
    
    
    # #* enconding categorical variables
    # td_at_snap['playDirection'] = td_at_snap.apply(lambda row: 0 if row['playDirection'] == 'left' else 1, axis=1)
    
    # le_offenseFormation = LabelEncoder()
    # le_receiverAlignment = LabelEncoder()
    # le_possessionTeam = LabelEncoder()
    # plays["offenseFormation_encoded"] = le_offenseFormation.fit_transform(plays["offenseFormation"])
    # plays["receiverAlignment_encoded"] = le_receiverAlignment.fit_transform(plays["receiverAlignment"])
    # plays['possessionTeam_encoded'] = le_possessionTeam.fit_transform(plays['possessionTeam'])
    
    # le_club = LabelEncoder()
    # le_position = LabelEncoder()
    # td_at_snap['club'] = le_club.fit_transform(td_at_snap['club'])
    # td_at_snap['position'] = le_position.fit_transform(td_at_snap['position'])
    
    #* getting football info
    football = td_at_snap[(td_at_snap['displayName'] == 'football')]
    
    #* transforming players data and adding player info to tracking data
    players['height'] = players['height'].str.split('-').apply(lambda x: round((int(x[0]) * 12 + int(x[1])) * 2.54))
    players['weight'] = round(players['weight'] * 0.45359237)
    
    players_info = players[['nflId', 'height', 'weight', 'position']]
    td_at_snap = pd.merge(td_at_snap, players_info, on='nflId')
    
    #* adding total distance before snap to tracking data
    distances = td_before_snap.groupby('nflId')['dis'].sum()
    td_at_snap['totalDis'] = td_at_snap['nflId'].map(distances)
    
    # creating plot
    absoluteYardlineNumber = play['absoluteYardlineNumber'].values[0]
    direction = td_at_snap.iloc[0]['playDirection'] #TODO: fix this because data has been categorized as 0 (left) or 1 (right)

    if direction == 'left':
        highlight_line_number = absoluteYardlineNumber - 10
    else:
        highlight_line_number = 110 - absoluteYardlineNumber
        
    if highlight_line_number > 50:
        check_line = 50 - (50 - highlight_line_number)
        
    if check_line == yardline_number:
        raise ValueError('Line of scrimmage is not correct')
    
    fig, ax = createFootballField(highlight_line=True, highlight_line_number=highlight_line_number)
    
    # calculating distances between players, creating a new dataframe for it, sorting it and creating a list
    coords = td_at_snap[['x', 'y']].values
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    dist_df = pd.DataFrame(dist_matrix, index=td_at_snap['nflId'], columns=td_at_snap['nflId'])
    
    sorted_distances = dist_df.apply(lambda row: row.sort_values().values.tolist(), axis=1)
    sorted_players = dist_df.apply(lambda row: row.sort_values().index.tolist(), axis=1)
    
    # for index, value in sorted_distances.items():
    #     print(f'Player {index}: ', end='')
    #     for j in range(1, 7):
    #         print(f'{sorted_players.loc[index][j]:.2f}, ', end='')
    #     print()
        
    # closest team players
    k_players = 5
    team_dist = {}
    for index, value in sorted_distances.items():
        team = td_at_snap[td_at_snap['nflId'] == index]['club'].values[0]
        closest_players = []
        for i in range(1, len(sorted_players.loc[index])):
            if td_at_snap[td_at_snap['nflId'] == sorted_players.loc[index][i]]['club'].values[0] == team:
                # closest_players.append((sorted_players.loc[index][i], value[i]))
                closest_players.append({
                    'nflId': sorted_players.loc[index][i],
                    'distance': value[i]
                })
            if len(closest_players) == k_players:
                break
        
        team_dist[index] = closest_players
    
    # closest opponent players
    opponent_dist = {}
    for index, value in sorted_distances.items():
        team = td_at_snap[td_at_snap['nflId'] == index]['club'].values[0]
        closest_players = []
        for i in range(1, len(sorted_players.loc[index])):
            if td_at_snap[td_at_snap['nflId'] == sorted_players.loc[index][i]]['club'].values[0] != team:
                # closest_players.append((sorted_players.loc[index][i], value[i]))
                closest_players.append({
                    'nflId': sorted_players.loc[index][i],
                    'distance': value[i]
                })
            if len(closest_players) == k_players:
                break
        opponent_dist[index] = closest_players
        
    # closest all players
    all_dist = {}
    for index, value in sorted_distances.items():
        closest_players = []
        for i in range(1, len(sorted_players.loc[index])):
            # closest_players.append((sorted_players.loc[index][i], value[i]))
            closest_players.append({
                'nflId': sorted_players.loc[index][i],
                'distance': value[i]
            })
            if len(closest_players) == k_players:
                break
        all_dist[index] = closest_players
        
    # print(json.dumps(team_dist, indent=4))
    
    # # plotting lines between players
    # for key, value in team_dist.items():
    #     key_x = td_at_snap[td_at_snap['nflId'] == key]['x'].values[0]
    #     key_y = td_at_snap[td_at_snap['nflId'] == key]['y'].values[0]
    #     for player in value:
    #         player_x = td_at_snap[td_at_snap['nflId'] == player['nflId']]['x'].values[0]
    #         player_y = td_at_snap[td_at_snap['nflId'] == player['nflId']]['y'].values[0]
    #         plt.plot([key_x, player_x], [key_y, player_y], color='lightgreen', linewidth=0.5)
            
    # for key, value in opponent_dist.items():
    #     key_x = td_at_snap[td_at_snap['nflId'] == key]['x'].values[0]
    #     key_y = td_at_snap[td_at_snap['nflId'] == key]['y'].values[0]
    #     for player in value:
    #         player_x = td_at_snap[td_at_snap['nflId'] == player['nflId']]['x'].values[0]
    #         player_y = td_at_snap[td_at_snap['nflId'] == player['nflId']]['y'].values[0]
    #         plt.plot([key_x, player_x], [key_y, player_y], color='lightcoral', linewidth=0.5)
            
    # plotting closest players without team distinction
    for key, value in all_dist.items():
        key_x = td_at_snap[td_at_snap['nflId'] == key]['x'].values[0]
        key_y = td_at_snap[td_at_snap['nflId'] == key]['y'].values[0]
        for player in value:
            player_x = td_at_snap[td_at_snap['nflId'] == player['nflId']]['x'].values[0]
            player_y = td_at_snap[td_at_snap['nflId'] == player['nflId']]['y'].values[0]
            plt.plot([key_x, player_x], [key_y, player_y], color='lightblue', linewidth=0.5)
            
    # creating network graph with attributes, adding all edges, and then adding node attributes
    display(play.head(5))
    graph_attrs = json.loads(play.drop(['gameId', 'playId'], axis=1).to_json(orient='records'))[0]
    G = nx.Graph(quarter=graph_attrs['quarter'], 
                down=graph_attrs['down'],
                yardsToGo=graph_attrs['yardsToGo'],
                possessionTeam=graph_attrs['possessionTeam'],
                gameClock=graph_attrs['gameClock'],
                absoluteYardlineNumber=graph_attrs['absoluteYardlineNumber'],
                offenseFormation=graph_attrs['offenseFormation'],
                receiverAlignment=graph_attrs['receiverAlignment'],
                playClockAtSnap=graph_attrs['playClockAtSnap'],
                possessionTeamPointDiff=graph_attrs['possessionTeamPointDiff'])
    
    for key, value in all_dist.items():
        for player in value:
            G.add_edge(key, player['nflId'], weight=player['distance'])
            
    for key, value in td_at_snap.iterrows():
        relevant_info = ['club', 'displayName', 'playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'height', 'weight', 'position', 'totalDis']
        info_dict = {k: v for k, v in value.items() if k in relevant_info}
        nx.set_node_attributes(G, {value['nflId']: info_dict})
    
    # print(list(G.nodes(data=True)))
    # for n, nbrs in G.adj.items():
    #     for nbr, eattr in nbrs.items():
    #         wt = eattr['weight']
    #         print(f"({n}, {nbr}, {wt:.3})")
    
    print(G.nodes[42401.0])
    print(G.graph)
            
    # ploting players
    colors = {'ARI': 'white',
            'KC': 'red',
            'football': '#7b3f00'}
    
    for index, player in td_at_snap.iterrows():
        x = player['x']
        y = player['y']
        s = player['displayName']
        if s == 'football':
            continue
        plt.scatter(x, y, color=colors[player['club']], zorder=2)

    plt.scatter(football['x'].values[0], football['y'].values[0], color=colors[football['club'].values[0]])
            
    
    plt.show()
    
    
if __name__ == "__main__":
    main()