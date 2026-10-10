import torch
import torch.nn.functional as F
import numpy as np
import random
import time
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.gcn import GCN
from src.utils.logger import Logger

class GNNTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Logger.info(f'Using device: {self.device}')

    def set_seed(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # For cuDNN (GPU) - makes it deterministic but slower
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def convert_nx_to_pytorch_geometric(self, graphs, include_labels=True):
        """Convert NetworkX graphs to PyTorch Geometric format"""
        Logger.info(" Converting graphs to PyTorch Geometric format...")
        data_list = []
        
        for G in graphs:
            # Mapeamento de IDs dos jogadores para índices numéricos
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            
            # Extrair características dos nós
            node_features = []
            for node in G.nodes():
                node_data = G.nodes[node]
                # Como os dados já estão codificados, extraímos diretamente
                features = []
                
                # Coordenadas x, y
                features.append(float(node_data.get('x', 0)))
                features.append(float(node_data.get('y', 0)))
                
                # Velocidade, aceleração, distância
                features.append(float(node_data.get('s', 0)))  # velocidade
                features.append(float(node_data.get('a', 0)))  # aceleração
                features.append(float(node_data.get('dis', 0)))  # distância percorrida
                
                # Orientação e direção
                features.append(float(node_data.get('o', 0)))  # orientação
                features.append(float(node_data.get('dir', 0)))  # direção
                
                # Características físicas
                features.append(float(node_data.get('height', 0)))
                features.append(float(node_data.get('weight', 0)))
                
                # Posição do jogador (já codificada)
                features.append(float(node_data.get('position', 0)))
                
                # Clube (já codificado)
                features.append(float(node_data.get('club', 0)))
                
                # Direção da jogada
                features.append(float(node_data.get('playDirection', 0)))
                
                # Total de distância percorrida
                features.append(float(node_data.get('totalDis', 0)))
                
                # Features gerais do grafo
                features.append(float(G.graph['quarter']))
                features.append(float(G.graph['down']))
                features.append(float(G.graph['yardsToGo']))
                features.append(float(G.graph['absoluteYardlineNumber']))
                features.append(float(G.graph['playClockAtSnap']))
                features.append(float(G.graph['possessionTeamPointDiff']))
                features.append(float(G.graph['possessionTeam']))
                features.append(float(G.graph['gameClock']))
                features.append(float(G.graph['offenseFormation']))
                features.append(float(G.graph['receiverAlignment']))
                
                node_features.append(features)
            
            # Converter para tensor
            x = torch.tensor(node_features, dtype=torch.float)
            
            # Extrair arestas
            edge_indices = []
            edge_weights = []
            for src, dst, data in G.edges(data=True):
                # Converter IDs dos nós para índices numéricos
                src_idx = node_mapping[src]
                dst_idx = node_mapping[dst]
                
                edge_indices.append([src_idx, dst_idx])
                edge_weights.append(data.get('weight', 1.0))  # Peso da aresta (distância entre jogadores)
            
            # Converter para o formato PyTorch Geometric 
            # edge_index deve ser um tensor de tamanho [2, num_edges]
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
            
            # Definir o alvo (y) - classificação binária: 0 para corrida, 1 para passe
            if include_labels:
                play_type = 1 if G.graph.get('playResult') == 1 else 0
                y = torch.tensor([play_type], dtype=torch.long)  # Long tensor para classificação
            else:
                # Para dados de teste reais, não incluímos os labels
                y = torch.tensor([-1], dtype=torch.long)  # Placeholder value
            
            # Criar objeto Data do PyTorch Geometric
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
            )
            
            data_list.append(data)
        
        return data_list

    def split_and_prepare_data(self, pass_graphs, rush_graphs):
        """Split data into train/validation/test sets"""
        # Verificar integridade dos dados
        for i in pass_graphs:
            assert i.graph['playResult'] == 1
        
        for i in rush_graphs:
            assert i.graph['playResult'] == 0
        
        n_pass = len(pass_graphs)
        n_rush = len(rush_graphs)
        Logger.info(f"🔍 DEBUG - Input: Pass={n_pass}, Rush={n_rush}")
        
        # Usar valores do config
        train_split = getattr(self.config.DATASET, 'TRAIN_SPLIT', 0.7)
        val_split = getattr(self.config.DATASET, 'VALIDATION_SPLIT', 0.15)
        test_split = getattr(self.config.DATASET, 'TEST_SPLIT', 0.15)
        
        # Verificar se soma ~1.0
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 0.01:
            Logger.warning(f"⚠️  Dataset splits don't sum to 1.0: {total}")
        
        # ===== DIVISÃO CORRETA: DIVIDIR CADA CLASSE SEPARADAMENTE =====
        
        # Shuffle each class before splitting
        random.shuffle(pass_graphs)
        random.shuffle(rush_graphs)
        
        # Para PASS graphs
        pass_train_end = int(train_split * n_pass)
        pass_val_end = pass_train_end + int(val_split * n_pass)
        
        pass_train = pass_graphs[:pass_train_end]
        pass_val = pass_graphs[pass_train_end:pass_val_end]
        pass_test = pass_graphs[pass_val_end:]
        
        # Para RUSH graphs  
        rush_train_end = int(train_split * n_rush)
        rush_val_end = rush_train_end + int(val_split * n_rush)
        
        rush_train = rush_graphs[:rush_train_end]
        rush_val = rush_graphs[rush_train_end:rush_val_end]
        rush_test = rush_graphs[rush_val_end:]
        
        # Log das divisões
        Logger.info(f"🔍 Pass splits - Train: {len(pass_train)}, Val: {len(pass_val)}, Test: {len(pass_test)}")
        Logger.info(f"🔍 Rush splits - Train: {len(rush_train)}, Val: {len(rush_val)}, Test: {len(rush_test)}")
        
        # Combinar classes
        train_graphs = pass_train + rush_train
        validation_graphs = pass_val + rush_val
        test_graphs = pass_test + rush_test
        
        # Embaralhar
        # random.shuffle(train_graphs)
        # random.shuffle(validation_graphs) 
        # random.shuffle(test_graphs)
        
        Logger.info(f"🔍 Final - Train: {len(train_graphs)}, Val: {len(validation_graphs)}, Test: {len(test_graphs)}")
        
        return train_graphs, validation_graphs, test_graphs

    def _verify_balance(self, graphs, set_name):
        """Verify that the dataset is balanced"""
        n_pass = sum(1 for g in graphs if g.graph['playResult'] == 1)
        n_rush = sum(1 for g in graphs if g.graph['playResult'] == 0)
        assert n_pass == n_rush, f"{set_name} set is not balanced: {n_pass} passes and {n_rush} rushes."

    def print_dataset_info(self, train_graphs, validation_graphs, test_graphs, train_dataset):
        """Print dataset information"""
        if self.config.SHOW_INFO:
            n_pass_train = sum(1 for g in train_graphs if g.graph['playResult'] == 1)
            n_rush_train = sum(1 for g in train_graphs if g.graph['playResult'] == 0)
            n_pass_val = sum(1 for g in validation_graphs if g.graph['playResult'] == 1)
            n_rush_val = sum(1 for g in validation_graphs if g.graph['playResult'] == 0)
            n_pass_test = sum(1 for g in test_graphs if g.graph['playResult'] == 1)
            n_rush_test = sum(1 for g in test_graphs if g.graph['playResult'] == 0)
            
            Logger.info("")
            Logger.info('    ====================')
            Logger.info(f'    Number of graphs: {len(train_graphs) + len(validation_graphs) + len(test_graphs)}')
            Logger.info(f"    Number of train graphs: {len(train_graphs)}")
            Logger.info(f"    Number of validation graphs: {len(validation_graphs)}")
            Logger.info(f"    Number of test graphs: {len(test_graphs)}")
            Logger.info(f'    Percentage of passes in train set: {n_pass_train / len(train_graphs) * 100:.2f}%')
            Logger.info(f'    Percentage of rushes in train set: {n_rush_train / len(train_graphs) * 100:.2f}%')
            Logger.info(f'    Percentage of passes in validation set: {n_pass_val / len(validation_graphs) * 100:.2f}%')
            Logger.info(f'    Percentage of rushes in validation set: {n_rush_val / len(validation_graphs) * 100:.2f}%')
            Logger.info(f'    Percentage of passes in test set: {n_pass_test / len(test_graphs) * 100:.2f}%')
            Logger.info(f'    Percentage of rushes in test set: {n_rush_test / len(test_graphs) * 100:.2f}%')

            data = train_dataset[0]  # Get the first graph object.
            Logger.info("")
            Logger.info(data)
            Logger.info('    =============================================================')

            # Gather some statistics about the first graph.
            Logger.info(f'    Number of nodes: {data.num_nodes}')
            Logger.info(f'    Number of node features: {data.num_node_features}')
            Logger.info(f'    Number of edges: {data.num_edges}')
            Logger.info(f'    Average node degree: {data.num_edges / data.num_nodes:.2f}')
            Logger.info(f'    Has isolated nodes: {data.has_isolated_nodes()}')
            Logger.info(f'    Has self-loops: {data.has_self_loops()}')
            Logger.info(f'    Is undirected: {data.is_undirected()}')

    def train_epoch(self, model, loader, criterion, optimizer):
        """Train one epoch"""
        model.train()
        total_loss = 0
        total_samples = 0

        for data in loader:  # Iterate in batches over the training dataset.
            data = data.to(self.device)  # Move data to device
            batch_size = data.y.size(0)
            total_samples += batch_size
            
            optimizer.zero_grad()  # Clear gradients
            
            out = model(data.x, data.edge_index, data.batch) # WITHOUT EDGE DATA
            # out = model(data.x, data.edge_index, data.batch, data.edge_attr)  # WITH EDGE DATA
            
            loss = criterion(out, data.y)  # Compute loss
            loss.backward()  # Backward pass
            
            # Gradient clipping (opcional, mas ajuda com estabilidade)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()  # Update parameters
            total_loss += loss.item()
        
        # return total_loss / total_samples if total_samples > 0 else 0
        return total_loss / len(loader)

    def evaluate(self, loader, model):
        """Evaluate model"""
        model.eval()

        all_preds = []
        all_labels = []
        correct = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)  # Move data to device
                
                out = model(data.x, data.edge_index, data.batch) # WITHOUT EDGE DATA    
                # out = model(data.x, data.edge_index, data.batch, data.edge_attr)  # WITH EDGE DATA
                
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                all_preds.extend(pred.cpu().numpy())  # Store predictions
                all_labels.extend(data.y.cpu().numpy())  # Store true labels
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.

        accuracy = correct / len(loader.dataset)  # Derive ratio of correct predictions.
        return accuracy, all_preds, all_labels

    def calculate_validation_loss(self, model, validation_loader, criterion):
        """Calculate validation loss"""
        model.eval()
        val_loss = 0
        val_samples = 0

        with torch.no_grad():
            for data in validation_loader:
                data = data.to(self.device)  # Move data to device
                batch_size = data.y.size(0)
                val_samples += batch_size
                
                out = model(data.x, data.edge_index, data.batch) # WITHOUT EDGE DATA
                # out = model(data.x, data.edge_index, data.batch, data.edge_attr) # WITH EDGE DATA
                
                loss = criterion(out, data.y)
                val_loss += loss.item()

        # return val_loss / val_samples if val_samples > 0 else 0
        return val_loss / len(validation_loader)

    def train_model(self, pass_graphs, rush_graphs):
        """Main training method"""
        self.set_seed(self.config.RANDOM_SEED)
        Logger.info("Running GNN model...")
        # Verificar se os dados de entrada estão balanceados
        if len(pass_graphs) != len(rush_graphs):
            Logger.warning(f"⚠️  Input data is imbalanced! Pass: {len(pass_graphs)}, Rush: {len(rush_graphs)}")

        # Split data
        train_graphs, validation_graphs, test_graphs = self.split_and_prepare_data(pass_graphs, rush_graphs)

        return self.train_on_split(train_graphs, validation_graphs, test_graphs)

    def train_on_split(self, train_graphs, validation_graphs, test_graphs, model_factory=None, train_params=None):
        """Treina e avalia um modelo em uma divisao ja definida.

        model_factory: funcao (num_node_features, config) -> torch.nn.Module com a
            mesma assinatura forward(x, edge_index, batch) da GCN. Padrao: GCN.
        train_params: namespace com EPOCHS, LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER,
            WARMUP_EPOCHS, EARLY_STOP_PATIENCE e MIN_DELTA. Padrao: config.GCN.
        """
        self.set_seed(self.config.RANDOM_SEED)
        if model_factory is None:
            model_factory = GCN
        if train_params is None:
            train_params = self.config.GCN

        # Convert to PyTorch Geometric format
        train_dataset = self.convert_nx_to_pytorch_geometric(train_graphs, include_labels=True)
        validation_dataset = self.convert_nx_to_pytorch_geometric(validation_graphs, include_labels=True)
        test_dataset_with_labels = self.convert_nx_to_pytorch_geometric(test_graphs, include_labels=True)
        
        # Print dataset info
        self.print_dataset_info(train_graphs, validation_graphs, test_graphs, train_dataset)
        
        # Create data loaders with smaller batch size
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        test_loader_with_labels = DataLoader(test_dataset_with_labels, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = model_factory(train_dataset[0].num_node_features, self.config).to(self.device)
        Logger.info(f'Model parameters: {model.get_num_parameters():,}')
        
        # Log config
        Logger.info(f"Config: {self.config}")
        
        # Initialize optimizer (AdamW)
        if train_params.OPTIMIZER.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_params.LEARNING_RATE,
                weight_decay=train_params.WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            Logger.info("Using AdamW optimizer")
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=train_params.LEARNING_RATE, 
                weight_decay=train_params.WEIGHT_DECAY
            )
            Logger.info("Using Adam optimizer")
        
        # Initialize scheduler with warmup + cosine annealing
        warmup_epochs = train_params.WARMUP_EPOCHS
        total_epochs = train_params.EPOCHS
        
        # Warmup scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        # Cosine annealing scheduler (after warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        # Combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # Early stopping parameters
        patience = train_params.EARLY_STOP_PATIENCE
        min_delta = train_params.MIN_DELTA
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training metrics
        min_loss = float('inf')
        min_loss_epoch = 0
        max_val_acc = 0
        max_val_acc_epoch = 0
        max_train_acc = 0
        max_train_acc_epoch = 0
        best_model_state = None

        gcn_start_time = datetime.now()
        Logger.info(f'Training GNN model with early stopping (patience={patience})...')
        
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Training loop
        for epoch in range(1, total_epochs + 1):
            # Training step
            total_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            train_losses.append(total_loss)
            
            # Validation step
            val_loss = self.calculate_validation_loss(model, validation_loader, criterion)
            val_losses.append(val_loss)
            
            # Step scheduler
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            scheduler.step()
            
            # Evaluate on train and validation
            train_acc, train_preds, train_labels = self.evaluate(train_loader, model)
            val_acc, val_preds, val_labels = self.evaluate(validation_loader, model)
            
            # Log progress
            if epoch % 10 == 0 or epoch <= warmup_epochs:
                Logger.info(f'    Epoch: {epoch:03d}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, '
                        f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
            
            # Track best metrics
            if total_loss < min_loss:
                min_loss = total_loss
                min_loss_epoch = epoch
            if train_acc > max_train_acc:
                max_train_acc = train_acc
                max_train_acc_epoch = epoch
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                max_val_acc_epoch = epoch
                best_model_state = model.state_dict().copy()
            
            # Early stopping logic
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                Logger.info(f'    ✓ New best validation loss: {val_loss:.4f}')
            else:
                patience_counter += 1
                
            # Check for early stopping
            if patience_counter >= patience:
                Logger.info(f'    🛑 Early stopping triggered at epoch {epoch}')
                Logger.info(f'    Best validation loss: {best_val_loss:.4f}')
                break
        
        gcn_end_time = datetime.now()
        Logger.info(f'Training finished in {gcn_end_time - gcn_start_time}')

        # Print training summary
        Logger.info("")
        Logger.info('    ====================')   
        Logger.info(f"    Min loss: {min_loss:.4f} at epoch {min_loss_epoch}")
        Logger.info(f"    Max train acc: {max_train_acc:.4f} at epoch {max_train_acc_epoch}")
        Logger.info(f"    Max validation acc: {max_val_acc:.4f} at epoch {max_val_acc_epoch}")
        Logger.info(f"    Final LR: {learning_rates[-1]:.6f}")
        Logger.info(f"    Early stopped: {'Yes' if patience_counter >= patience else 'No'}")

        # Test with last model
        test_acc_last, test_preds_last, test_labels_last = self.evaluate(test_loader_with_labels, model)
        Logger.info("    Metrics of last model:")
        Logger.info(f"    Test accuracy: {test_acc_last:.4f}")
        print(f'    {classification_report(test_labels_last, test_preds_last, target_names=["Rush", "Pass"], zero_division=0)}')
        last_gcn_results = classification_report(test_labels_last, test_preds_last, target_names=["Rush", "Pass"], output_dict=True, zero_division=0)
        last_gcn_results['confusion_matrix'] = confusion_matrix(test_labels_last, test_preds_last)
        
        # Test with best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            test_acc_best, test_preds_best, test_labels_best = self.evaluate(test_loader_with_labels, model)
            Logger.info("    Metrics of best model:")
            Logger.info(f"    Test accuracy: {test_acc_best:.4f}")
            print(f'    {classification_report(test_labels_best, test_preds_best, target_names=["Rush", "Pass"], zero_division=0)}')
            best_gcn_results = classification_report(test_labels_best, test_preds_best, target_names=["Rush", "Pass"], output_dict=True, zero_division=0)
            best_gcn_results['confusion_matrix'] = confusion_matrix(test_labels_best, test_preds_best)
        else:
            best_gcn_results = last_gcn_results
        
        Logger.info('    ====================')
        Logger.info("")

        # Predicoes por jogada e tempo de inferencia (modelo no estado final avaliado acima)
        test_preds, test_labels, test_proba = self.predict(test_loader_with_labels, model)
        inference = self.measure_inference_time(model, test_dataset_with_labels)
        
        return {
            'last_gcn_results': last_gcn_results,
            'best_gcn_results': best_gcn_results,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'model': model,
            'best_model_state': best_model_state,
            'early_stopped': patience_counter >= patience,
            'stopped_epoch': min(epoch, total_epochs),
            'train_time_s': (gcn_end_time - gcn_start_time).total_seconds(),
            'test_keys': [(g.graph.get('gameId'), g.graph.get('playId')) for g in test_graphs],
            'test_labels': test_labels,
            'test_preds': test_preds,
            'test_proba': test_proba,
            'inference': inference,
        }

    def predict(self, loader, model):
        """Retorna predicoes, rotulos e probabilidade da classe passe, na ordem do loader"""
        model.eval()
        preds, labels, proba = [], [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = model(data.x, data.edge_index, data.batch)
                p = F.softmax(out, dim=1)[:, 1]
                preds.extend(out.argmax(dim=1).cpu().tolist())
                labels.extend(data.y.cpu().tolist())
                proba.extend(p.cpu().tolist())
        return preds, labels, proba

    def measure_inference_time(self, model, dataset, n_samples=200, batch_size=32):
        """Mede o tempo de inferencia: por jogada (lote de 1) e no conjunto de teste inteiro"""
        model.eval()
        sync = torch.cuda.synchronize if self.device.type == 'cuda' else (lambda: None)
        # Data.to() do PyG altera o objeto no lugar: copia para nao deixar parte do
        # dataset na GPU e parte na CPU (o DataLoader abaixo falharia ao montar os lotes)
        sample = [d.clone() for d in dataset[:n_samples]]
        with torch.no_grad():
            for data in sample[:20]:  # aquecimento
                data = data.to(self.device)
                model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long, device=self.device))
            sync()
            start = time.perf_counter()
            for data in sample:
                data = data.to(self.device)
                model(data.x, data.edge_index, torch.zeros(data.num_nodes, dtype=torch.long, device=self.device))
            sync()
            per_play_ms = (time.perf_counter() - start) * 1000 / len(sample)

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            sync()
            start = time.perf_counter()
            for data in loader:
                data = data.to(self.device)
                model(data.x, data.edge_index, data.batch)
            sync()
            full_test_s = time.perf_counter() - start
        return {'per_play_ms': per_play_ms, 'full_test_s': full_test_s,
                'n_test': len(dataset), 'device': self.device.type}