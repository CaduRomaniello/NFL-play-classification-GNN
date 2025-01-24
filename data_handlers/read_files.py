import pandas as pd

def read2025data(weeks=[1]):
    if not weeks: 
        raise ValueError('read2025data: Weeks must be provided')
    
    
    print('Reading data...')
    
    games = pd.read_csv(f'./nfl_data/2025/games.csv')
    player_play = pd.read_csv(f'./nfl_data/2025/player_play.csv')
    players = pd.read_csv(f'./nfl_data/2025/players.csv')
    plays = pd.read_csv(f'./nfl_data/2025/plays.csv')

    tracking_data = pd.DataFrame()
    for week in weeks:
        tracking_data = pd.concat([tracking_data, pd.read_csv(f'./nfl_data/2025/tracking_week_{week}.csv')])
        
    print('Data read successfully')

    return games, player_play, players, plays, tracking_data