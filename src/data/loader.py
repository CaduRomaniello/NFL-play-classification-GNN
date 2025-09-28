# crie uma classe que irá ser responsável por carregar todos os arquivos
import os
import pandas as pd

from src.utils.logger import Logger

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_path = config.FILES.INPUT_PATH
        self.weeks_to_read = config.FILES.WEEKS_TO_READ

    def load_week_data(self, week: int) -> pd.DataFrame:
        Logger.info(f'Loading tracking data for week {week}')
        cur_path = os.getcwd()
        data_path = os.path.abspath(os.path.join(cur_path, self.config.FILES.INPUT_PATH, f"tracking_week_{week}.csv"))

        tracking_data = pd.DataFrame()
        tracking_data = pd.concat([tracking_data, pd.read_csv(data_path)])
        return tracking_data

    def load_auxiliar_nfl_files(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Logger.info('Loading auxiliar NFL files')
        cur_path = os.getcwd()
        data_path = os.path.abspath(os.path.join(cur_path, self.config.FILES.INPUT_PATH))

        games = pd.read_csv(os.path.join(data_path, 'games.csv'))
        player_play = pd.read_csv(os.path.join(data_path, 'player_play.csv'))
        players = pd.read_csv(os.path.join(data_path, 'players.csv'))
        plays = pd.read_csv(os.path.join(data_path, 'plays.csv'))

        return games, player_play, players, plays