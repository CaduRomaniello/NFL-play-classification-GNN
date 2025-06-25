import pandas as pd

def merge_player_info(players: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    print("    Merging player info...")
    
    players['height'] = players['height'].str.split('-').apply(lambda x: round((int(x[0]) * 12 + int(x[1])) * 2.54))
    players['weight'] = round(players['weight'] * 0.45359237)
    
    players_info = players[['nflId', 'height', 'weight', 'position']]
    t = pd.merge(tracking, players_info, on='nflId')
    return t