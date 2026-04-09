import json
import copy
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.logger import Logger

EDGE_STRATEGIES = ["CLOSEST-", "QB-CLOSEST-", "DELAUNAY", "GABRIEL", "RNG", "MST"]
DOWNSAMPLE_FACTORS = [True, False]


def run_experiment(strategy, seed, downsample, base_config):
    """
    Standalone function to handle a single iteration of the pipeline.
    """
    try:
        # Create a local copy of the config to prevent race conditions
        config = copy.deepcopy(base_config)
        config.EDGE_STRATEGY = strategy
        config.DOWN_SAMPLE = downsample
        config.RANDOM_SEED = seed

        Logger.info(f"Starting NFL GNN data pipeline for {strategy} strategy with downsample={downsample} and seed={seed}...")
        
        # Initialize and execute data pipeline
        data_pipeline = DataPipeline(config)
        if config.FORCE_REBUILD:
            pass_graphs, rush_graphs = data_pipeline.force_rebuild_graphs()
        else:
            pass_graphs, rush_graphs = data_pipeline.execute()
            
        Logger.info(f"Graph creation completed for {strategy} (seed {seed})! Pass: {len(pass_graphs)}, Rush: {len(rush_graphs)}")

        # Training pipeline
        training_pipeline = TrainingPipeline(config)
        results = training_pipeline.execute(pass_graphs, rush_graphs)

        Logger.info(f"Pipeline completed successfully for {strategy} (seed {seed})!")
        
        # Return an identifier alongside results so you know which run finished
        return {"strategy": strategy, "seed": seed, "downsample": downsample, "status": "success", "results": results}
        
    except Exception as e:
        Logger.error(f"Error in {strategy} (seed {seed}): {str(e)}")
        return {"strategy": strategy, "seed": seed, "downsample": downsample, "status": "failed", "error": str(e)}


def main(config):
    tasks = []
    
    # 1. Build a list of all parameter combinations
    for strategy in EDGE_STRATEGIES:
        for i in range(1, 30):
            tasks.append((strategy, i, False, config))
            # for downsample in DOWNSAMPLE_FACTORS:
                # tasks.append((strategy, i, downsample, config))

    Logger.info(f"Total experiments to run: {len(tasks)}")

    # 2. Execute them in parallel
    # NOTE: Adjust max_workers based on your CPU cores and available RAM/GPU Memory
    # A safe starting point is usually 2 or 4 for heavy ML workloads.
    max_parallel_jobs = 2
    
    with ProcessPoolExecutor(max_workers=max_parallel_jobs) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(run_experiment, *task) for task in tasks]
        
        # Process results as they finish (order is not guaranteed)
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                # Handle your success results here if needed
                pass
            else:
                # Handle failures here
                pass

    Logger.info("All parallel executions completed!")


if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    main(config)