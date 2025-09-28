import os
import json
from datetime import datetime

from src.models.gcn_trainer import GNNTrainer
from src.models.baseline_models import BaselineModels
from src.utils.logger import Logger

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.gnn_trainer = GNNTrainer(config)
        self.baseline_models = BaselineModels(config)

    def execute(self, pass_graphs, rush_graphs):
        """Execute complete training pipeline"""
        Logger.info('Starting training pipeline...')

        # Train GNN model
        gnn_results = self.gnn_trainer.train_model(pass_graphs, rush_graphs)
        
        # Create simplified datasets for baseline models
        train_graphs, validation_graphs, test_graphs = self.gnn_trainer.split_and_prepare_data(pass_graphs, rush_graphs)
        
        # Run baseline models
        rf_results, mlp_results = self.baseline_models.run_baselines(train_graphs, test_graphs)
        
        # Combine all results
        all_results = {
            'last_gcn_results': gnn_results['last_gcn_results'],
            'best_gcn_results': gnn_results['best_gcn_results'],
            'rf_results': rf_results,
            'mlp_results': mlp_results,
            'train_losses': gnn_results['train_losses'],
            'val_losses': gnn_results['val_losses'],
            'config': self._config_to_dict(self.config),
        }
        
        # Save results
        self._save_results(all_results)
        
        Logger.info('Training pipeline completed!')
        
        return all_results

    def _config_to_dict(self, config):
        """Convert config SimpleNamespace to dict"""
        if hasattr(config, '__dict__'):
            result = {}
            for key, value in config.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._config_to_dict(value)
                else:
                    result[key] = value
            return result
        return config

    def _save_results(self, results):
        """Save training results to file"""
        # Create results directory
        results_dir = self.config.FILES.RESULTS_PATH + '/' + self.config.EDGE_STRATEGY
        os.makedirs(results_dir, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f'training_results_{timestamp}.json')
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        json_results = convert_for_json(results)
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        Logger.info(f'Results saved to {results_file}')