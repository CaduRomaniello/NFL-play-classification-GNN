import random
import torch
import numpy as np
import torch.nn.functional as F

## GNN
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# MLP and RF
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

class GCN(torch.nn.Module):
    def __init__(self, node_features, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def convert_nx_to_pytorch_geometric(graphs, include_labels=True):
    print(" Converting graphs to PyTorch Geometric format...")
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
            
            # Features gerias do grafo
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

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for data in loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        total_loss += loss.item()  # Accumulate the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return total_loss
    # print(f'Loss: {total_loss / len(loader):.4f}')

def test(loader, model):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def model_run(pass_graphs, rush_graphs, show_info=False, epochs=100, random_seed=1):
    print("Running model...")
    
    n_graphs = len(pass_graphs)
    train_split = int(0.8 * n_graphs)
    
    # Embaralhar os grafos para evitar qualquer viés
    # indices = np.random.permutation(n_graphs)
    # train_indices = indices[:train_split]
    # test_indices = indices[train_split:]
    
    #! ASSERTS
    #! ASSERTS
    #! ASSERTS
    for i in pass_graphs:
        assert i.graph['playResult'] == 1
    
    for i in rush_graphs:
        assert i.graph['playResult'] == 0
    #! ASSERTS
    #! ASSERTS
    #! ASSERTS
    
    train_graphs = pass_graphs[:train_split] + rush_graphs[:train_split]
    random.shuffle(train_graphs)
    
    test_graphs = pass_graphs[train_split:] + rush_graphs[train_split:]
    random.shuffle(test_graphs)
    
    #! ASSERTS
    #! ASSERTS
    #! ASSERTS
    n_pass_test = 0
    n_rush_test = 0
    for i in train_graphs:
        if i.graph['playResult'] == 1:
            n_pass_test += 1
        else:
            n_rush_test += 1
    
    assert n_pass_test == n_rush_test, f"Train set is not balanced: {n_pass_test} passes and {n_rush_test} rushes."
    
    n_pass = 0
    n_rush = 0
    for i in test_graphs:
        if i.graph['playResult'] == 1:
            n_pass += 1
        else:
            n_rush += 1
    
    assert n_pass == n_rush, f"Test set is not balanced: {n_pass} passes and {n_rush} rushes."
    #! ASSERTS
    #! ASSERTS
    #! ASSERTS
    
    if show_info:
        dataset = convert_nx_to_pytorch_geometric(train_graphs, include_labels=True)

        print()
        # print(f'Dataset: {dataset}:')
        print('====================')
        print(f'Number of graphs: {len(dataset)}')
        print(f'Percentage of passes in train set: {n_pass_test / len(train_graphs) * 100:.2f}%')
        print(f'Percentage of rushes in train set: {n_rush_test / len(train_graphs) * 100:.2f}%')
        print(f'Percentage of passes in test set: {n_pass / len(test_graphs) * 100:.2f}%')
        print(f'Percentage of rushes in test set: {n_rush / len(test_graphs) * 100:.2f}%')

        data = dataset[0]  # Get the first graph object.

        print()
        print(data)
        print('=============================================================')

        # Gather some statistics about the first graph.
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of node features: {data.num_node_features}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
    
    print(f"Número de grafos para treino: {len(train_graphs)}")
    print(f"Número de grafos para teste: {len(test_graphs)}")
    
    # Converter para o formato PyTorch Geometric
    train_dataset = convert_nx_to_pytorch_geometric(train_graphs, include_labels=True)
    
    # Para teste real, não incluir os labels
    test_dataset_with_labels = convert_nx_to_pytorch_geometric(test_graphs, include_labels=True)
    test_dataset_no_labels = convert_nx_to_pytorch_geometric(test_graphs, include_labels=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader_with_labels = DataLoader(test_dataset_with_labels, batch_size=64, shuffle=False)
    test_loader_no_labels = DataLoader(test_dataset_no_labels, batch_size=64, shuffle=False)
    
    model = GCN(train_dataset[0].num_node_features, hidden_channels=64)
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    min_loss = 9999999999
    min_loss_epoch = 0
    max_test_acc = 0
    max_test_acc_epoch = 0
    max_train_acc = 0
    max_train_acc_epoch = 0
    for epoch in range(1, epochs + 1):
        total_loss = train(model, train_loader, criterion, optimizer)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader_with_labels, model)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Total Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
            
        if total_loss < min_loss:
            min_loss = total_loss
            min_loss_epoch = epoch
        if train_acc > max_train_acc:
            max_train_acc = train_acc
            max_train_acc_epoch = epoch
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_test_acc_epoch = epoch
         
    print()
    print('====================')   
    print(f"Min loss: {min_loss:.4f} at epoch {min_loss_epoch}")
    print(f"Max train acc: {max_train_acc:.4f} at epoch {max_train_acc_epoch}")
    print(f"Max test acc: {max_test_acc:.4f} at epoch {max_test_acc_epoch}")
    
    print('====================')
    print()
    
    run_baselines(train_graphs, test_graphs, random_seed=random_seed)
            
##############################################################################            
##############################################################################            
##############################################################################
################################# MLP and RF #################################            
##############################################################################
##############################################################################
##############################################################################

def graph_to_vector(G):
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

def run_baselines(train_graphs, test_graphs, random_seed=1):
    # Converter grafos em vetores
    X_train, y_train = zip(*[graph_to_vector(G) for G in train_graphs])
    X_test, y_test = zip(*[graph_to_vector(G) for G in test_graphs])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Normalização para a MLP
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"🎯 Random Forest Accuracy: {rf_acc:.4f}")

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),  # Estrutura da rede
        max_iter=3000,                # Número máximo de iterações
        random_state=random_seed,     # Seed para reprodutibilidade
        learning_rate_init=0.01,     # Taxa de aprendizado inicial
        alpha=5e-4                    # Regularização L2 (equivalente ao weight decay)
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_preds = mlp.predict(X_test_scaled)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    print(f"🤖 MLP Accuracy: {mlp_acc:.4f}")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
# epochs = 300     
# hidden_channels = 64   
# ====================
# Number of graphs: 4514
# Percentage of passes in train set: 50.00%
# Percentage of rushes in train set: 50.00%
# Percentage of passes in test set: 50.00%
# Percentage of rushes in test set: 50.00%

# Data(x=[22, 23], edge_index=[2, 33], edge_attr=[33, 1], y=[1])
# =============================================================
# Number of nodes: 22
# Number of node features: 23
# Number of edges: 33
# Average node degree: 1.50
# Has isolated nodes: False
# Has self-loops: False
# Is undirected: False
# Número de grafos para treino: 4514
# Número de grafos para teste: 1130
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
# GCN(
#   (conv1): GCNConv(23, 64)
#   (conv2): GCNConv(64, 64)
#   (conv3): GCNConv(64, 64)
#   (lin): Linear(in_features=64, out_features=2, bias=True)
# )
# Epoch: 010, Total Loss: 95.7003, Train Acc: 0.6130, Test Acc: 0.5894
# Epoch: 010, Total Loss: 95.7003, Train Acc: 0.6130, Test Acc: 0.5894
# Epoch: 020, Total Loss: 60.7194, Train Acc: 0.6292, Test Acc: 0.5991
# Epoch: 020, Total Loss: 60.7194, Train Acc: 0.6292, Test Acc: 0.5991
# Epoch: 030, Total Loss: 52.5385, Train Acc: 0.6422, Test Acc: 0.6133
# Epoch: 040, Total Loss: 48.0741, Train Acc: 0.6568, Test Acc: 0.6372
# Epoch: 050, Total Loss: 45.6817, Train Acc: 0.6690, Test Acc: 0.6478
# Epoch: 060, Total Loss: 44.5494, Train Acc: 0.6770, Test Acc: 0.6637
# Epoch: 070, Total Loss: 43.0357, Train Acc: 0.6887, Test Acc: 0.6664
# Epoch: 080, Total Loss: 42.6312, Train Acc: 0.6952, Test Acc: 0.6947
# Epoch: 090, Total Loss: 41.8511, Train Acc: 0.7043, Test Acc: 0.6885
# Epoch: 100, Total Loss: 41.6240, Train Acc: 0.7091, Test Acc: 0.6973
# Epoch: 110, Total Loss: 40.7680, Train Acc: 0.7111, Test Acc: 0.6805
# Epoch: 120, Total Loss: 39.9294, Train Acc: 0.7202, Test Acc: 0.7106
# Epoch: 130, Total Loss: 39.8498, Train Acc: 0.7071, Test Acc: 0.6602
# Epoch: 140, Total Loss: 39.2862, Train Acc: 0.7346, Test Acc: 0.7088
# Epoch: 150, Total Loss: 38.6072, Train Acc: 0.7350, Test Acc: 0.6938
# Epoch: 160, Total Loss: 38.7405, Train Acc: 0.7379, Test Acc: 0.7195
# Epoch: 170, Total Loss: 37.6862, Train Acc: 0.7147, Test Acc: 0.6646
# Epoch: 180, Total Loss: 37.8350, Train Acc: 0.7459, Test Acc: 0.7035
# Epoch: 190, Total Loss: 37.2289, Train Acc: 0.7466, Test Acc: 0.7106
# Epoch: 200, Total Loss: 37.1274, Train Acc: 0.7545, Test Acc: 0.7159
# Epoch: 210, Total Loss: 37.3058, Train Acc: 0.7441, Test Acc: 0.7327
# Epoch: 220, Total Loss: 36.6572, Train Acc: 0.7390, Test Acc: 0.6832
# Epoch: 230, Total Loss: 36.1989, Train Acc: 0.7590, Test Acc: 0.7292
# Epoch: 240, Total Loss: 36.6783, Train Acc: 0.7550, Test Acc: 0.7292
# Epoch: 250, Total Loss: 36.0354, Train Acc: 0.7574, Test Acc: 0.7310
# Epoch: 260, Total Loss: 36.1430, Train Acc: 0.7570, Test Acc: 0.7168
# Epoch: 270, Total Loss: 35.8075, Train Acc: 0.7612, Test Acc: 0.7230
# Epoch: 280, Total Loss: 36.1513, Train Acc: 0.7563, Test Acc: 0.7124
# Epoch: 290, Total Loss: 35.1511, Train Acc: 0.7481, Test Acc: 0.6956
# Epoch: 300, Total Loss: 35.5563, Train Acc: 0.7607, Test Acc: 0.7257






























# ====================
# Number of graphs: 6672
# Percentage of passes in train set: 50.00%
# Percentage of rushes in train set: 50.00%
# Percentage of passes in test set: 50.00%
# Percentage of rushes in test set: 50.00%

# Data(x=[22, 23], edge_index=[2, 31], edge_attr=[31, 1], y=[1])
# =============================================================
# Number of nodes: 22
# Number of node features: 23
# Number of edges: 31
# Average node degree: 1.41
# Has isolated nodes: False
# Has self-loops: False
# Is undirected: False
# Número de grafos para treino: 6672
# Número de grafos para teste: 1668
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
# GCN(
#   (conv1): GCNConv(23, 64)
#   (conv2): GCNConv(64, 64)
#   (conv3): GCNConv(64, 64)
#   (lin): Linear(in_features=64, out_features=2, bias=True)
# )
# Epoch: 010, Total Loss: 111.0297, Train Acc: 0.6142, Test Acc: 0.6193
# Epoch: 020, Total Loss: 78.2824, Train Acc: 0.6322, Test Acc: 0.6331
# Epoch: 030, Total Loss: 71.4785, Train Acc: 0.6466, Test Acc: 0.6427
# Epoch: 040, Total Loss: 67.8718, Train Acc: 0.6593, Test Acc: 0.6631
# Epoch: 050, Total Loss: 65.6622, Train Acc: 0.6704, Test Acc: 0.6613
# Epoch: 060, Total Loss: 64.4494, Train Acc: 0.6850, Test Acc: 0.6859
# Epoch: 070, Total Loss: 62.2928, Train Acc: 0.6990, Test Acc: 0.6984
# Epoch: 080, Total Loss: 61.0075, Train Acc: 0.7196, Test Acc: 0.7044
# Epoch: 090, Total Loss: 59.8647, Train Acc: 0.7112, Test Acc: 0.6960
# Epoch: 100, Total Loss: 58.6954, Train Acc: 0.7365, Test Acc: 0.7242
# Epoch: 110, Total Loss: 57.4261, Train Acc: 0.7412, Test Acc: 0.7194
# Epoch: 120, Total Loss: 57.6877, Train Acc: 0.7424, Test Acc: 0.7302
# Epoch: 130, Total Loss: 56.5981, Train Acc: 0.7317, Test Acc: 0.7164
# Epoch: 140, Total Loss: 56.2170, Train Acc: 0.7510, Test Acc: 0.7266
# Epoch: 150, Total Loss: 55.6655, Train Acc: 0.7496, Test Acc: 0.7182
# Epoch: 160, Total Loss: 55.5298, Train Acc: 0.7496, Test Acc: 0.7362
# Epoch: 170, Total Loss: 54.8583, Train Acc: 0.7555, Test Acc: 0.7308
# Epoch: 180, Total Loss: 54.8571, Train Acc: 0.7100, Test Acc: 0.6906
# Epoch: 190, Total Loss: 54.2296, Train Acc: 0.7496, Test Acc: 0.7224
# Epoch: 200, Total Loss: 54.0849, Train Acc: 0.7551, Test Acc: 0.7296
# Epoch: 210, Total Loss: 54.3480, Train Acc: 0.7599, Test Acc: 0.7218
# Epoch: 220, Total Loss: 53.8665, Train Acc: 0.7618, Test Acc: 0.7308
# Epoch: 230, Total Loss: 53.5107, Train Acc: 0.7531, Test Acc: 0.7248
# Epoch: 240, Total Loss: 53.9014, Train Acc: 0.7506, Test Acc: 0.7242
# Epoch: 250, Total Loss: 53.1359, Train Acc: 0.7681, Test Acc: 0.7260
# Epoch: 260, Total Loss: 52.9407, Train Acc: 0.7579, Test Acc: 0.7296
# Epoch: 270, Total Loss: 52.6485, Train Acc: 0.7638, Test Acc: 0.7356
# Epoch: 280, Total Loss: 52.5520, Train Acc: 0.7693, Test Acc: 0.7320
# Epoch: 290, Total Loss: 52.3844, Train Acc: 0.7728, Test Acc: 0.7314
# Epoch: 300, Total Loss: 53.3203, Train Acc: 0.7699, Test Acc: 0.7350

# ====================
# Min loss: 51.7727 at epoch 295
# Max train acc: 0.7728 at epoch 290
# Max test acc: 0.7404 at epoch 292










# ==================== (2 files)
# Min loss: 18.6128 at epoch 300
# Max train acc: 0.7618 at epoch 299
# Max test acc: 0.7164 at epoch 296
# ====================

# 🎯 Random Forest Accuracy: 0.7382
# 🤖 MLP Accuracy: 0.6891









# ==================== (6 files)
# Min loss: 51.2389 at epoch 299
# Max train acc: 0.7749 at epoch 298
# Max test acc: 0.7320 at epoch 233
# ====================

# 🎯 Random Forest Accuracy: 0.7548
# 🤖 MLP Accuracy: 0.7050











# ==================== (9 files)
# Min loss: 74.2383 at epoch 296
# Max train acc: 0.7749 at epoch 286
# Max test acc: 0.7662 at epoch 222
# ====================

# 🎯 Random Forest Accuracy: 0.7318
# 🤖 MLP Accuracy: 0.6777

# 🎯 Random Forest Accuracy: 0.7351
# 🤖 MLP Accuracy: 0.7179