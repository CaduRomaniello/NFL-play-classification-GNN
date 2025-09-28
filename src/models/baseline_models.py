import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import Logger

class BaselineModels:
    def __init__(self, config):
        self.config = config

    def graph_to_vector(self, G):
        """Convert graph to feature vector"""
        node_features = []
        for node in G.nodes():
            features = [
                float(G.nodes[node].get('x', 0)),
                float(G.nodes[node].get('y', 0)),
                float(G.nodes[node].get('s', 0)),
                float(G.nodes[node].get('a', 0)),
                float(G.nodes[node].get('dis', 0)),
                float(G.nodes[node].get('o', 0)),
                float(G.nodes[node].get('dir', 0)),
                float(G.nodes[node].get('height', 0)),
                float(G.nodes[node].get('weight', 0)),
                float(G.nodes[node].get('position', 0)),
                float(G.nodes[node].get('club', 0)),
                float(G.nodes[node].get('playDirection', 0)),
                float(G.nodes[node].get('totalDis', 0)),
            ]
            node_features.append(features)

        node_features = np.array(node_features)
        mean_node_features = node_features.mean(axis=0)

        graph_features = [
            float(G.graph.get('quarter', 0)),
            float(G.graph.get('down', 0)),
            float(G.graph.get('yardsToGo', 0)),
            float(G.graph.get('absoluteYardlineNumber', 0)),
            float(G.graph.get('playClockAtSnap', 0)),
            float(G.graph.get('possessionTeamPointDiff', 0)),
            float(G.graph.get('possessionTeam', 0)),
            float(G.graph.get('gameClock', 0)),
            float(G.graph.get('offenseFormation', 0)),
            float(G.graph.get('receiverAlignment', 0)),
        ]

        full_vector = np.concatenate([mean_node_features, graph_features])
        label = 1 if G.graph['playResult'] == 1 else 0

        return full_vector, label

    def run_baselines(self, train_graphs, test_graphs):
        """Run Random Forest and MLP baselines"""
        Logger.info("Running baseline models...")
        
        # Converter grafos em vetores
        X_train, y_train = zip(*[self.graph_to_vector(G) for G in train_graphs])
        X_test, y_test = zip(*[self.graph_to_vector(G) for G in test_graphs])

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Normalização para a MLP
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Random Forest
        rf_results = self._train_random_forest(X_train, X_test, y_train, y_test)
        
        # MLP
        mlp_results = self._train_mlp(X_train_scaled, X_test_scaled, y_train, y_test)

        return rf_results, mlp_results

    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        rf_start_time = datetime.now()

        rf = RandomForestClassifier(
            n_estimators=self.config.RF.N_ESTIMATORS, 
            random_state=self.config.RANDOM_SEED
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_preds)
        
        Logger.info(f"    🎯 Random Forest Accuracy: {rf_acc:.4f}")
        print(f'    {classification_report(y_test, rf_preds, target_names=["Rush", "Pass"], zero_division=0)}')

        rf_results = classification_report(y_test, rf_preds, target_names=["Rush", "Pass"], output_dict=True, zero_division=0)
        rf_results['confusion_matrix'] = confusion_matrix(y_test, rf_preds)

        rf_end_time = datetime.now()
        Logger.info(f'[{rf_end_time}] Random Forest training finished in {rf_end_time - rf_start_time}')
        
        return rf_results

    def _train_mlp(self, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train MLP model"""
        mlp_start_time = datetime.now()

        mlp = MLPClassifier(
            hidden_layer_sizes=[self.config.MLP.HIDDEN_CHANNELS] * self.config.MLP.HIDDEN_LAYERS,
            max_iter=self.config.MLP.MAX_ITER,
            random_state=self.config.RANDOM_SEED,
            learning_rate_init=self.config.MLP.LEARNING_RATE,
            alpha=self.config.MLP.ALPHA
        )
        mlp.fit(X_train_scaled, y_train)
        mlp_preds = mlp.predict(X_test_scaled)
        mlp_acc = accuracy_score(y_test, mlp_preds)
        
        Logger.info(f"    🤖 MLP Accuracy: {mlp_acc:.4f}")
        print(f'    {classification_report(y_test, mlp_preds, target_names=["Rush", "Pass"], zero_division=0)}')

        mlp_results = classification_report(y_test, mlp_preds, target_names=["Rush", "Pass"], output_dict=True, zero_division=0)
        mlp_results['confusion_matrix'] = confusion_matrix(y_test, mlp_preds)

        mlp_end_time = datetime.now()
        Logger.info(f'[{mlp_end_time}] MLP training finished in {mlp_end_time - mlp_start_time}')
        
        return mlp_results