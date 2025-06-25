import pandas as pd
from tqdm import tqdm
from IPython.display import display

def verify_plays_result(plays: pd.DataFrame) -> pd.DataFrame:
    print("    Verifying plays result...")
    
    plays['playResult'] = plays.apply(lambda x: play_result(x), axis=1)
    return plays

def play_result(play: pd.Series) -> int:
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
        
def verify_invalid_values(plays: pd.DataFrame, tracking: pd.DataFrame) -> pd.DataFrame:
    print("    Verifying invalid values...")
    
    plays.fillna({'receiverAlignment': 'EMPTY'}, inplace=True)
    plays.fillna({'offenseFormation': 'EMPTY'}, inplace=True)
    plays.fillna({'playClockAtSnap': 0}, inplace=True)
    
    tracking[tracking['frameType'] == 'SNAP']['o'] == tracking[tracking['frameType'] == 'SNAP'].apply(lambda t: o_invalid_values(plays, t), axis=1)
    tracking[tracking['frameType'] == 'SNAP']['dir'] == tracking[tracking['frameType'] == 'SNAP'].apply(lambda t: dir_invalid_values(plays, t), axis=1)
    return plays, tracking
    
            
def o_invalid_values(plays: pd.DataFrame, t) -> int:
    game_id = t['gameId']
    play_id = t['playId']
    
    return 90 if pd.isna(t['o']) and t['playDirection'] == 'right' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
            270 if pd.isna(t['o']) and t['playDirection'] == 'right' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
            270 if pd.isna(t['o']) and t['playDirection'] == 'left' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
            90 if pd.isna(t['o']) and t['playDirection'] == 'left' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else t['o']

def dir_invalid_values(plays: pd.DataFrame, t) -> int:
    game_id = t['gameId']
    play_id = t['playId']
    
    return 90 if pd.isna(t['dir']) and t['playDirection'] == 'right' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
            270 if pd.isna(t['dir']) and t['playDirection'] == 'right' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
            270 if pd.isna(t['dir']) and t['playDirection'] == 'left' and t['club'] == plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else \
            90 if pd.isna(t['dir']) and t['playDirection'] == 'left' and t['club'] != plays[(plays['gameId'] == game_id) & (plays['playId'] == play_id)]['possessionTeam'].values[0] else t['dir']
