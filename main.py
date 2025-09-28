import json
from types import SimpleNamespace
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.logger import Logger

EDGE_STRATEGIES = ["CLOSEST-", "QB-CLOSEST-", "DELAUNAY", "GABRIEL", "RNG", "MST"]
# EDGE_STRATEGIES = ["CLOSEST-", "QB-CLOSEST-"]
DOWNSAMPLE_FACTORS = [True, False]

def main(config):
    for strategy in EDGE_STRATEGIES:
        for i in range(1, 11):
            for downsample in DOWNSAMPLE_FACTORS:
                config.EDGE_STRATEGY = strategy
                config.DOWNSAMPLE = downsample
                config.RANDOM_SEED = i

                # Reading data and creating graphs
                Logger.info(f"Starting NFL GNN data pipeline for {strategy} strategy with downsample={downsample} and seed={i}...")
                
                # Initialize and execute data pipeline
                data_pipeline = DataPipeline(config)
                if config.FORCE_REBUILD:
                    pass_graphs, rush_graphs = data_pipeline.force_rebuild_graphs()
                else:
                    pass_graphs, rush_graphs = data_pipeline.execute()
                
                # Print graph statistics
                Logger.info(f"\nGraph creation completed!")
                Logger.info(f"Pass graphs: {len(pass_graphs)}")
                Logger.info(f"Rush graphs: {len(rush_graphs)}")
                Logger.info(f"Total graphs: {len(pass_graphs) + len(rush_graphs)}")

                # Training pipeline
                Logger.info(f"\nStarting training pipeline...")
                training_pipeline = TrainingPipeline(config)
                results = training_pipeline.execute(pass_graphs, rush_graphs)

                Logger.info(f"\nPipeline completed successfully!")

                # return pass_graphs, rush_graphs, results

if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    main(config)