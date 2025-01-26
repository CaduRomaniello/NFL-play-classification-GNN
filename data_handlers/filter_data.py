def process_data(plays, tracking):
    for _, play in plays.iterrows():
        play_id = play['playId']
        play_data = play
        players_data = tracking[tracking['playId'] == play_id]
        
def process_single_data(plays, tracking, play_id, game_id):
    play_data = plays[(plays['playId'] == play_id) & (plays['gameId'] == game_id)]
    players_data = tracking[tracking['playId'] == play_id]
    
    return play_data, players_data