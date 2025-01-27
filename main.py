import json

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from playground import playground
from scipy.spatial.distance import cdist
from data_handlers.read_files import read2025data
from visualization.create_plot import createFootballField

PLAY_RELEVANT_COLUMNS = ['gameId', 'playId', 'quarter', 'down', 'yardsToGo', 'possessionTeam', 'gameClock', 'absoluteYardlineNumber', 'offenseFormation', 'receiverAlignment', 'playClockAtSnap']

def main():
    games, player_play, players, plays, tracking_data = read2025data()
    
    game_id = 2022091110
    play_id = 55
    
    # filtering df's
    game = games[games['gameId'] == game_id]
    play = plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]
    tracking_data = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id)]
    
    yardline_number = play['yardlineNumber'].values[0]
    
    # getting possession team point diff
    home_team = game['homeTeamAbbr'].values[0]
    possession_team = play['possessionTeam'].values[0]
    
    possession_team_point_diff = 0
    if possession_team == home_team:
        possession_team_point_diff = play['preSnapHomeScore'].values[0] - play['preSnapVisitorScore'].values[0]
    else:
        possession_team_point_diff = play['preSnapVisitorScore'].values[0] - play['preSnapHomeScore'].values[0]
        
    # getting pass or rush play type
    passLocationType = play['passLocationType'].values[0]
    rushLocationType = play['rushLocationType'].values[0]
    
    playType = ''
    if pd.isna(play['passLocationType'].values[0]) and pd.isna(play['rushLocationType'].values[0]):
        raise ValueError('play does not have passLocationType or rushLocationType')
    elif not pd.isna(play['passLocationType'].values[0]) and not pd.isna(play['rushLocationType'].values[0]):
        raise ValueError('play has both passLocationType and rushLocationType')
    else:
        playType = 'pass' if pd.isna(play['rushLocationType']).empty else 'rush'
    
    
    # filtering play df for relevant columns and adding possession team point diff
    play = play[PLAY_RELEVANT_COLUMNS]
    play['possessionTeamPointDiff'] = possession_team_point_diff
    
    # adding play type to play df
    # play['playType'] = playType
    
    # verifying if there is only one line_set event
    if tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP') & (tracking_data['event'] == 'line_set')]['frameId'].nunique() != 1:
        raise ValueError('There is more than one line_set event')
    
    # filtering tracking data for events after line_set
    line_set_frame_id = tracking_data[(tracking_data['frameType'] != 'AFTER_SNAP') & (tracking_data['event'] == 'line_set')]['frameId'].values[0]
    td_before_snap = tracking_data[(tracking_data['frameId'] >= line_set_frame_id) & (tracking_data['frameType'] != 'AFTER_SNAP')]
    td_at_snap = td_before_snap[td_before_snap['frameType'] == 'SNAP'].sort_values('club')
    
    # getting football info
    football = td_at_snap[(td_at_snap['displayName'] == 'football')]
    
    # adding player info to tracking data
    players_info = players[['nflId', 'height', 'weight', 'position']]
    td_at_snap = pd.merge(td_at_snap, players_info, on='nflId')
    
    # adding total distance before snap to tracking data
    distances = td_before_snap.groupby('nflId')['dis'].sum()
    td_at_snap['totalDis'] = td_at_snap['nflId'].map(distances)
    
    # creating plot
    absoluteYardlineNumber = play['absoluteYardlineNumber'].values[0]
    direction = td_at_snap.iloc[0]['playDirection']

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
        
    # 3 closest team players
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
            if len(closest_players) == 2:
                break
        
        team_dist[index] = closest_players
    
    # 3 closest opponent players
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
            if len(closest_players) == 2:
                break
        opponent_dist[index] = closest_players
        
    # 3 closest all players
    all_dist = {}
    for index, value in sorted_distances.items():
        closest_players = []
        for i in range(1, len(sorted_players.loc[index])):
            # closest_players.append((sorted_players.loc[index][i], value[i]))
            closest_players.append({
                'nflId': sorted_players.loc[index][i],
                'distance': value[i]
            })
            if len(closest_players) == 5:
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