import os
import pandas as pd

def read2025data(weeks=[1]):
    if not weeks: 
        raise ValueError('read2025data: Weeks must be provided')
    
    
    print('    Reading data...')
    
    cur_path = os.getcwd()
    data_path = os.path.abspath(os.path.join(cur_path, './Mestrado/nfl_data/2025/'))
    
    games = pd.read_csv(os.path.join(data_path, 'games.csv'))
    player_play = pd.read_csv(os.path.join(data_path, 'player_play.csv'))
    players = pd.read_csv(os.path.join(data_path, 'players.csv'))
    plays = pd.read_csv(os.path.join(data_path, 'plays.csv'))

    tracking_data = pd.DataFrame()
    for week in weeks:
        tracking_data = pd.concat([tracking_data, pd.read_csv(os.path.join(data_path, f'tracking_week_{week}.csv'))])
        
    print('    Data read successfully')

    return games, player_play, players, plays, tracking_data