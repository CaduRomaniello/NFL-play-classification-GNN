import json
from types import SimpleNamespace
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline

def main(config):
    # Reading data and creating graphs
    print("Starting NFL GNN data pipeline...")
    
    # Initialize and execute data pipeline
    data_pipeline = DataPipeline(config)
    if config.FORCE_REBUILD:
        pass_graphs, rush_graphs = data_pipeline.force_rebuild_graphs()
    else:
        pass_graphs, rush_graphs = data_pipeline.execute()
    
    # Print graph statistics
    print(f"\nGraph creation completed!")
    print(f"Pass graphs: {len(pass_graphs)}")
    print(f"Rush graphs: {len(rush_graphs)}")
    print(f"Total graphs: {len(pass_graphs) + len(rush_graphs)}")
    
    # Training pipeline
    print(f"\nStarting training pipeline...")
    training_pipeline = TrainingPipeline(config)
    results = training_pipeline.execute(pass_graphs, rush_graphs)
    
    print(f"\nPipeline completed successfully!")
    
    return pass_graphs, rush_graphs, results

if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    main(config)