import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_handlers.read_files import read2025data
from visualization.create_plot import createFootballField

def playground():
    games, player_play, players, plays, tracking_data = read2025data()
    
    play = plays.loc[(plays['gameId'] == 2022091110) & (plays['playId'] == 55)]
    yards_to_go = play['yardsToGo'].values[0]
    yardline_number = play['yardlineNumber'].values[0]

    game_id = 2022091110
    play_id = 55
    first_play = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id) & (tracking_data['frameType'] == 'SNAP')].sort_values('club')

    new_columns = players[['nflId', 'height', 'weight', 'position']]
    
    pd.set_option('display.max_columns', None)
    result = pd.merge(first_play, new_columns, on='nflId')

    createFootballField(highlight_line=True, highlight_line_number=100-yardline_number), 

    plt.plot([100 - yardline_number - yards_to_go + 10, 100 - yardline_number - yards_to_go + 10], [0, 53.3], color='blue')

    colors = {'ARI': 'black',
            'KC': 'red',
            'football': '#7b3f00'}

    for index, player in result.iterrows():
        x = player['x']
        y = player['y']
        s = player['displayName']
        plt.scatter(x, y, color=colors[player['club']])

    football = tracking_data[(tracking_data['gameId'] == game_id) & (tracking_data['playId'] == play_id) & (tracking_data['frameType'] == 'SNAP') & (tracking_data['displayName'] == 'football')]
    plt.scatter(football['x'].values[0], football['y'].values[0], color=colors[football['club'].values[0]])

    plt.show()
    
if __name__ == "__main__":
    playground()