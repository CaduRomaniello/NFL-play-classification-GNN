# classe para orquestrar o pipeline da leitura, tratamento e construção dos grafos

from src.data.graph_builder import GraphBuilder
from src.utils.logger import Logger
import pandas as pd
import networkx as nx
import pickle
import os
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from IPython.display import display


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.data_preprocessor = DataPreprocessor(config)
        self.graph_builder = GraphBuilder(config)

        self.weeks_to_read = config.FILES.WEEKS_TO_READ
        self.graphs_path = config.FILES.GRAPHS_PATH
        self.rush_graphs = []
        self.pass_graphs = []

    def execute(self) -> tuple[list[nx.Graph], list[nx.Graph]]:
        Logger.info('Starting data pipeline execution')
        
        # Verificar se os grafos já existem
        if self._graphs_exist():
            Logger.info('Loading existing graphs from cache...')
            return self._load_graphs()

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        tracking_data = pd.DataFrame()
        for week in self.weeks_to_read:
            # Etapa 1: Leitura dos dados
            games, player_play, players, plays = self.data_loader.load_auxiliar_nfl_files()

            week_tracking_data = self.data_loader.load_week_data(week)
            tracking_data = pd.concat([tracking_data, week_tracking_data])

            # Etapa 2: Pré-processamento dos dados
            plays, week_tracking_data, dist_dict = self.data_preprocessor.execute(games, player_play, players, plays, week_tracking_data)

            # Etapa 3: Construção dos grafos
            pass_graphs, rush_graphs = self.graph_builder.execute(plays, week_tracking_data, dist_dict, downSample=self.config.DOWN_SAMPLE)

            self.pass_graphs.extend(pass_graphs)
            self.rush_graphs.extend(rush_graphs)

        # Salvar os grafos gerados
        self._save_graphs(self.pass_graphs, self.rush_graphs)

        return self.pass_graphs, self.rush_graphs
    
    def _graphs_exist(self) -> bool:
        """Verifica se os grafos já foram processados e salvos"""
        if self.config.DOWN_SAMPLE:
            pass_file = os.path.join(self.graphs_path, "dowmSampled/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "dowmSampled/rush_graphs.pkl")
        else:
            pass_file = os.path.join(self.graphs_path, "original/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "original/rush_graphs.pkl")

        return os.path.exists(pass_file) and os.path.exists(rush_file)
    
    def _save_graphs(self, pass_graphs: list, rush_graphs: list):
        """Salva os grafos em arquivos pickle"""
        # Criar diretório se não existir
        os.makedirs(self.graphs_path, exist_ok=True)

        if self.config.DOWN_SAMPLE:
            pass_file = os.path.join(self.graphs_path, "dowmSampled/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "dowmSampled/rush_graphs.pkl")
        else:
            pass_file = os.path.join(self.graphs_path, "original/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "original/rush_graphs.pkl")
        
        # Salvar grafos de passe
        with open(pass_file, 'wb') as f:
            pickle.dump(pass_graphs, f)
        
        # Salvar grafos de corrida
        with open(rush_file, 'wb') as f:
            pickle.dump(rush_graphs, f)
        
        Logger.info(f'Saved {len(pass_graphs)} pass graphs to {pass_file}')
        Logger.info(f'Saved {len(rush_graphs)} rush graphs to {rush_file}')
    
    def _load_graphs(self) -> tuple[list[nx.Graph], list[nx.Graph]]:
        """Carrega os grafos salvos"""
        if self.config.DOWN_SAMPLE:
            pass_file = os.path.join(self.graphs_path, "dowmSampled/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "dowmSampled/rush_graphs.pkl")
        else:
            pass_file = os.path.join(self.graphs_path, "original/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "original/rush_graphs.pkl")
        
        # Carregar grafos de passe
        with open(pass_file, 'rb') as f:
            pass_graphs = pickle.load(f)
        
        # Carregar grafos de corrida
        with open(rush_file, 'rb') as f:
            rush_graphs = pickle.load(f)
        
        Logger.info(f'Loaded {len(pass_graphs)} pass graphs from cache')
        Logger.info(f'Loaded {len(rush_graphs)} rush graphs from cache')
        
        return pass_graphs, rush_graphs
    
    def force_rebuild_graphs(self) -> tuple[list[nx.Graph], list[nx.Graph]]:
        """Força a reconstrução dos grafos, ignorando o cache"""
        Logger.info('Forcing graphs rebuild...')
        
        # Remover arquivos de cache se existirem
        if self.config.DOWN_SAMPLE:
            pass_file = os.path.join(self.graphs_path, "dowmSampled/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "dowmSampled/rush_graphs.pkl")
        else:
            pass_file = os.path.join(self.graphs_path, "original/pass_graphs.pkl")
            rush_file = os.path.join(self.graphs_path, "original/rush_graphs.pkl")

        if os.path.exists(pass_file):
            os.remove(pass_file)
        if os.path.exists(rush_file):
            os.remove(rush_file)
        
        # Executar o pipeline normalmente
        return self.execute()