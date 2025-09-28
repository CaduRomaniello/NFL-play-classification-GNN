from src.data.graph_strategies.qb_closest_n import QBClosestNStrategy
from src.data.graph_strategies.gabriel import GabrielStrategy
from src.data.graph_strategies.delaunay import DelaunayStrategy
from src.data.graph_strategies.closest_n import ClosestNStrategy
from src.data.graph_strategies.graph_strategy import GraphStrategy
from src.data.graph_strategies.relative_neighborhood import RNGStrategy
from src.data.graph_strategies.minimun_spanning_tree import MSTStrategy

class GraphStrategyFactory:
    """Factory para criar a estratégia de grafo apropriada."""
    
    @staticmethod
    def create_strategy(config) -> GraphStrategy:
        """
        Cria uma estratégia de grafo com base no nome da estratégia.
        
        Args:
            strategy_name: Nome da estratégia (ex: "CLOSEST-2", "DELAUNAY", "GABRIEL", "RNG", "MST")
            
        Returns:
            Uma instância da estratégia solicitada
        """
        if config.EDGE_STRATEGY.startswith("CLOSEST-"):
            return ClosestNStrategy(config)
        if config.EDGE_STRATEGY.startswith("QB-CLOSEST-"):
            return QBClosestNStrategy(config)
        elif config.EDGE_STRATEGY == "DELAUNAY":
            return DelaunayStrategy()
        elif config.EDGE_STRATEGY == "GABRIEL":
            return GabrielStrategy()
        elif config.EDGE_STRATEGY == "RNG":
            return RNGStrategy()
        elif config.EDGE_STRATEGY == "MST":
            return MSTStrategy()
        else:
            raise ValueError(f"Strategy {config.EDGE_STRATEGY} not supported")