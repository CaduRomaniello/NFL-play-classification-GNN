import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

class GraphStrategy(ABC):
    """Interface para estratégias de construção de grafos."""
    
    @abstractmethod
    def calculate_connections(self, tracking_data: pd.DataFrame, players: pd.DataFrame = None) -> Dict[int, Dict[int, Any]]:
        """
        Calcula as conexões entre jogadores usando a estratégia específica.
        
        Args:
            tracking_data: DataFrame com dados de rastreamento dos jogadores
            players: DataFrame com informações dos jogadores
            
        Returns:
            Um dicionário com as conexões para cada jogo e jogada
        """
        pass
    
    @staticmethod
    def get_strategy_name() -> str:
        """Retorna o nome da estratégia para identificação."""
        pass