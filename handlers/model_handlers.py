import torch
import numpy as np
import torch.nn.functional as F

from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# 1. Converter seus grafos NetworkX para formato PyTorch Geometric
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
        
        # Extrair características globais do grafo
        graph_features = []
        graph_features.append(float(G.graph['quarter']))
        graph_features.append(float(G.graph['down']))
        graph_features.append(float(G.graph['yardsToGo']))
        graph_features.append(float(G.graph['absoluteYardlineNumber']))
        graph_features.append(float(G.graph['playClockAtSnap']))
        graph_features.append(float(G.graph['possessionTeamPointDiff']))
        graph_features.append(float(G.graph['possessionTeam']))
        graph_features.append(float(G.graph['gameClock']))
        graph_features.append(float(G.graph['offenseFormation']))
        graph_features.append(float(G.graph['receiverAlignment']))
        
        # Assumimos que offenseFormation e outras características categóricas
        # já estão codificadas em algum lugar
        # if 'offenseFormation' in G.graph:
        #     graph_features.append(float(G.graph['offenseFormation']))
        
        graph_attr = torch.tensor(graph_features, dtype=torch.float)
        
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
            graph_attr=graph_attr
        )
        
        data_list.append(data)
    
    return data_list

# 2. Implementar o modelo GNN para classificação binária
class FootballGNN(torch.nn.Module):
    def __init__(self, node_features, graph_features, hidden_channels):
        super(FootballGNN, self).__init__()
        
        # Camadas de convolução do grafo para processamento dos nós
        self.conv1 = GCNConv(node_features, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Camada para processar características do grafo
        self.graph_lin = torch.nn.Linear(graph_features, hidden_channels)
        
        # Camadas finais para combinar representações de nós e do grafo
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 2)  # 2 classes: corrida ou passe
    
    def forward(self, data):
        x, edge_index, edge_attr, graph_attr = data.x, data.edge_index, data.edge_attr, data.graph_attr
        
        # Processamento dos nós
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # x = F.relu(self.conv2(x, edge_index, edge_attr))
        # x = F.dropout(x, p=0.2, training=self.training)
        
        # x = F.relu(self.conv3(x, edge_index, edge_attr))
        # x = F.dropout(x, p=0.2, training=self.training)
        
        # Pooling para obter uma representação a nível de grafo
        # Garantir que a saída tenha a forma [1, hidden_channels]
        x = torch.mean(x, dim=0).unsqueeze(0)  # Forma explícita: [1, hidden_channels]
        
        # Processamento das características do grafo
        # Garantir que graph_attr tenha as dimensões corretas [1, num_features]
        if graph_attr.dim() == 1:
            graph_attr = graph_attr.unsqueeze(0)  # Adiciona dimensão de batch
            
        graph_x = F.relu(self.graph_lin(graph_attr))  # [1, hidden_channels]
        
        # Concatenar características dos nós e do grafo
        x = torch.cat([x, graph_x], dim=1)  # [1, hidden_channels * 2]
        
        # Camadas finais
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)  # log_softmax para classificação
    
# 3. Função de treinamento para um único grafo por vez
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 4. Função de avaliação modificada
# @torch.no_grad()
def evaluate(model, data_list, known_labels=True):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    for data in data_list:
        out = model(data)
        pred = out.argmax(dim=1)
        predictions.append(pred.item())
        
        if known_labels and data.y[0] != -1:  # Só verifica acurácia se houver labels válidos
            correct += int((pred == data.y).sum())
            total += 1
    
    # Retorna a acurácia se houver labels conhecidos, caso contrário, apenas as previsões
    if known_labels and total > 0:
        return correct / total, predictions
    else:
        return -1, predictions  # -1 indica que a acurácia não pode ser calculada

# 5. Treinamento do modelo
def train_model(train_data_list, val_data_list, epochs=200):
    print(" Training model...")
    # Determinar o número de características dos nós e do grafo com base no primeiro exemplo
    sample_data = train_data_list[0]
    node_features = sample_data.x.size(1)
    graph_features = sample_data.graph_attr.size(0)
    
    # Inicializar modelo e otimizador
    model = FootballGNN(node_features, graph_features, hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
    
    # Treinamento
    begin = datetime.now()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for data in train_data_list:
            loss = train(model, optimizer, data)
            total_loss += loss
        
        # Avaliar no conjunto de treinamento
        train_acc, _ = evaluate(model, train_data_list)
        
        # Avaliar no conjunto de validação
        val_acc, _ = evaluate(model, val_data_list)
        
        if epoch % 10 == 0:
            end = datetime.now()
            print(f'    Epoch: {epoch:03d}, Loss: {total_loss/len(train_data_list):.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}', f" - Time elapsed: {(end - begin).total_seconds()} seconds")
    
    return model

# 6. Função principal que separa corretamente os dados e executa o treinamento
def model_run(graphs):
    print("Running model...")
    
    # Separação dos dados
    # Reserve alguns grafos para validação e teste
    n_graphs = len(graphs)
    train_split = int(0.7 * n_graphs)
    val_split = int(0.85 * n_graphs)
    
    # Embaralhar os grafos para evitar qualquer viés
    indices = np.random.permutation(n_graphs)
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    assert not set(train_indices).intersection(set(val_indices))
    assert not set(train_indices).intersection(set(test_indices))
    assert not set(val_indices).intersection(set(test_indices))
    
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]
    
    print(f"Número de grafos para treino: {len(train_graphs)}")
    print(f"Número de grafos para validação: {len(val_graphs)}")
    print(f"Número de grafos para teste: {len(test_graphs)}")
    
    # Converter para o formato PyTorch Geometric
    train_data_list = convert_nx_to_pytorch_geometric(train_graphs, include_labels=True)
    val_data_list = convert_nx_to_pytorch_geometric(val_graphs, include_labels=True)
    
    # Para teste real, não incluir os labels
    test_data_list_with_labels = convert_nx_to_pytorch_geometric(test_graphs, include_labels=True)
    test_data_list_no_labels = convert_nx_to_pytorch_geometric(test_graphs, include_labels=False)
    
    labels = [data.y.item() for data in train_data_list]
    print(f"    Distribuição das classes: {np.bincount(labels)} - {np.bincount(labels)[0] / len(labels)}, {np.bincount(labels)[1] / len(labels)}")
    
    # Treinar o modelo
    model = train_model(train_data_list, val_data_list, epochs=200)
    
    # Avaliar no conjunto de teste COM labels (apenas para verificação da performance real)
    test_acc, _ = evaluate(model, test_data_list_with_labels)
    print(f"Acurácia no conjunto de teste (com labels, apenas para verificação): {test_acc:.4f}")
    
    # Fazendo predições nos dados de teste SEM labels (como seria em um ambiente real)
    _, test_predictions = evaluate(model, test_data_list_no_labels, known_labels=False)
    print(f"Predições nos dados de teste (sem acesso aos labels): {test_predictions}")
    
    # Verificar se as predições estão corretas (apenas para fins de desenvolvimento)
    correct_labels = [data.y.item() for data in test_data_list_with_labels]
    correct_count = sum(1 for pred, true in zip(test_predictions, correct_labels) if pred == true)
    print(f"Labels reais dos dados de teste: {correct_labels}")
    print(f"Acurácia real calculada manualmente: {correct_count / len(test_predictions):.4f}")
    
    return model

# Para fazer uma previsão com um novo grafo (sem acesso ao label)
def predict(model, graph):
    print(" Predicting...")
    model.eval()
    # Converter sem incluir o label real
    data = convert_nx_to_pytorch_geometric([graph], include_labels=False)[0]
    with torch.no_grad():
        out = model(data)
    prob = torch.exp(out)
    pred = out.argmax(dim=1).item()
    return pred, prob[0]  # Retorna a classe predita e as probabilidades





# Distribuição das classes: [ 957 1563] - 0.37976190476190474, 0.6202380952380953
# Training model...
#     Epoch: 010, Loss: 0.6973, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 78.062462 seconds
#     Epoch: 020, Loss: 0.6757, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 158.928592 seconds
#     Epoch: 030, Loss: 0.6847, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 237.465591 seconds
#     Epoch: 040, Loss: 0.7206, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 318.684418 seconds
#     Epoch: 050, Loss: 0.7316, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 399.508909 seconds
#     Epoch: 060, Loss: 0.7409, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 480.51046 seconds
#     Epoch: 070, Loss: 0.7697, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 562.088404 seconds
#     Epoch: 080, Loss: 0.6937, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 642.983201 seconds
#     Epoch: 090, Loss: 0.8604, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 724.300169 seconds
#     Epoch: 100, Loss: 0.7021, Train Acc: 0.6167, Val Acc: 0.6241  - Time elapsed: 805.124117 seconds
# Acurácia no conjunto de teste (com labels, apenas para verificação): 0.6185
# Predições nos dados de teste (sem acesso aos labels): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# Labels reais dos dados de teste: [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
# Acurácia real calculada manualmente: 0.6185






# Distribuição das classes: [ 931 1589] - 0.36944444444444446, 0.6305555555555555
#  Training model...
#     Epoch: 010, Loss: 0.6588, Train Acc: 0.6306, Val Acc: 0.5630  - Time elapsed: 77.433606 seconds
#     Epoch: 020, Loss: 0.6050, Train Acc: 0.6444, Val Acc: 0.5981  - Time elapsed: 156.26347 seconds
#     Epoch: 030, Loss: 0.5707, Train Acc: 0.6778, Val Acc: 0.6481  - Time elapsed: 236.715968 seconds
#     Epoch: 040, Loss: 0.5586, Train Acc: 0.6310, Val Acc: 0.6315  - Time elapsed: 315.985741 seconds
#     Epoch: 050, Loss: 0.5579, Train Acc: 0.7071, Val Acc: 0.6648  - Time elapsed: 395.91582 seconds
#     Epoch: 060, Loss: 0.5525, Train Acc: 0.7159, Val Acc: 0.6630  - Time elapsed: 474.583975 seconds
#     Epoch: 070, Loss: 0.5372, Train Acc: 0.7044, Val Acc: 0.6500  - Time elapsed: 552.95288 seconds
#     Epoch: 080, Loss: 0.5488, Train Acc: 0.7230, Val Acc: 0.6870  - Time elapsed: 631.857382 seconds
#     Epoch: 090, Loss: 0.5326, Train Acc: 0.7183, Val Acc: 0.6852  - Time elapsed: 714.826494 seconds
#     Epoch: 100, Loss: 0.5233, Train Acc: 0.7175, Val Acc: 0.7074  - Time elapsed: 798.057369 seconds
# Acurácia no conjunto de teste (com labels, apenas para verificação): 0.6926
# Predições nos dados de teste (sem acesso aos labels): [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
# Labels reais dos dados de teste: [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
# Acurácia real calculada manualmente: 0.6926




# Running model...
# Número de grafos para treino: 6331
# Número de grafos para validação: 1357
# Número de grafos para teste: 1357
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
#  Converting graphs to PyTorch Geometric format...
#     Distribuição das classes: [2471 3860] - 0.3903016900963513, 0.6096983099036487
#  Training model...
#     Epoch: 010, Loss: 0.6356, Train Acc: 0.6478, Val Acc: 0.6632  - Time elapsed: 193.532112 seconds
#     Epoch: 020, Loss: 0.5843, Train Acc: 0.6967, Val Acc: 0.6721  - Time elapsed: 388.857537 seconds
#     Epoch: 030, Loss: 0.5653, Train Acc: 0.7290, Val Acc: 0.7104  - Time elapsed: 585.303837 seconds
#     Epoch: 040, Loss: 0.5651, Train Acc: 0.7354, Val Acc: 0.7200  - Time elapsed: 779.509894 seconds
#     Epoch: 050, Loss: 0.5579, Train Acc: 0.7346, Val Acc: 0.7155  - Time elapsed: 975.308388 seconds
#     Epoch: 060, Loss: 0.5572, Train Acc: 0.7343, Val Acc: 0.7163  - Time elapsed: 1177.959264 seconds
#     Epoch: 070, Loss: 0.5540, Train Acc: 0.7340, Val Acc: 0.7178  - Time elapsed: 1373.546506 seconds
#     Epoch: 080, Loss: 0.5551, Train Acc: 0.7373, Val Acc: 0.7192  - Time elapsed: 1568.289519 seconds
#     Epoch: 090, Loss: 0.5556, Train Acc: 0.7411, Val Acc: 0.7369  - Time elapsed: 1763.303505 seconds
#     Epoch: 100, Loss: 0.5523, Train Acc: 0.7329, Val Acc: 0.7207  - Time elapsed: 1959.124539 seconds
#     Epoch: 110, Loss: 0.5558, Train Acc: 0.7304, Val Acc: 0.7045  - Time elapsed: 2157.751125 seconds
#     Epoch: 120, Loss: 0.5538, Train Acc: 0.7345, Val Acc: 0.7244  - Time elapsed: 2355.577482 seconds
#     Epoch: 130, Loss: 0.5535, Train Acc: 0.7384, Val Acc: 0.7273  - Time elapsed: 2548.856346 seconds
#     Epoch: 140, Loss: 0.5534, Train Acc: 0.7387, Val Acc: 0.7229  - Time elapsed: 2743.838663 seconds
#     Epoch: 150, Loss: 0.5469, Train Acc: 0.7411, Val Acc: 0.7251  - Time elapsed: 2938.118003 seconds
#     Epoch: 160, Loss: 0.5592, Train Acc: 0.7384, Val Acc: 0.7296  - Time elapsed: 3133.815541 seconds
#     Epoch: 170, Loss: 0.5522, Train Acc: 0.7337, Val Acc: 0.7126  - Time elapsed: 3330.992769 seconds
#     Epoch: 180, Loss: 0.5528, Train Acc: 0.7402, Val Acc: 0.7273  - Time elapsed: 3526.029669 seconds
#     Epoch: 190, Loss: 0.5530, Train Acc: 0.7417, Val Acc: 0.7288  - Time elapsed: 3726.229427 seconds
#     Epoch: 200, Loss: 0.5528, Train Acc: 0.7440, Val Acc: 0.7281  - Time elapsed: 3928.954711 seconds
# Acurácia no conjunto de teste (com labels, apenas para verificação): 0.7428
# Predições nos dados de teste (sem acesso aos labels): [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
# Labels reais dos dados de teste: [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
# Acurácia real calculada manualmente: 0.7428


















































# import torch
# import numpy as np
# import torch.nn.functional as F

# from datetime import datetime
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, global_mean_pool

# # 1. Converter seus grafos NetworkX para formato PyTorch Geometric
# def convert_nx_to_pytorch_geometric(graphs):
#     print(" Converting graphs to PyTorch Geometric format...")
#     data_list = []
    
#     for G in graphs:
#         # Mapeamento de IDs dos jogadores para índices numéricos
#         node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        
#         # Extrair características dos nós
#         node_features = []
#         for node in G.nodes():
#             node_data = G.nodes[node]
#             # Como os dados já estão codificados, extraímos diretamente
#             features = []
            
#             # Coordenadas x, y
#             features.append(float(node_data.get('x', 0)))
#             features.append(float(node_data.get('y', 0)))
            
#             # Velocidade, aceleração, distância
#             features.append(float(node_data.get('s', 0)))  # velocidade
#             features.append(float(node_data.get('a', 0)))  # aceleração
#             features.append(float(node_data.get('dis', 0)))  # distância percorrida
            
#             # Orientação e direção
#             features.append(float(node_data.get('o', 0)))  # orientação
#             features.append(float(node_data.get('dir', 0)))  # direção
            
#             # Características físicas
#             features.append(float(node_data.get('height', 0)))
#             features.append(float(node_data.get('weight', 0)))
            
#             # Posição do jogador (já codificada)
#             features.append(float(node_data.get('position', 0)))
            
#             # Clube (já codificado)
#             features.append(float(node_data.get('club', 0)))
            
#             node_features.append(features)
        
#         # Converter para tensor
#         x = torch.tensor(node_features, dtype=torch.float)
        
#         # Extrair arestas
#         edge_indices = []
#         edge_weights = []
#         for src, dst, data in G.edges(data=True):
#             # Converter IDs dos nós para índices numéricos
#             src_idx = node_mapping[src]
#             dst_idx = node_mapping[dst]
            
#             edge_indices.append([src_idx, dst_idx])
#             edge_weights.append(data.get('weight', 1.0))  # Peso da aresta (distância entre jogadores)
        
#         # Converter para o formato PyTorch Geometric 
#         # edge_index deve ser um tensor de tamanho [2, num_edges]
#         edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
#         edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        
#         # Extrair características globais do grafo
#         graph_features = []
#         graph_features.append(float(G.graph['quarter']))
#         graph_features.append(float(G.graph['down']))
#         graph_features.append(float(G.graph['yardsToGo']))
#         graph_features.append(float(G.graph['absoluteYardlineNumber']))
#         graph_features.append(float(G.graph['playClockAtSnap']))
#         graph_features.append(float(G.graph['possessionTeamPointDiff']))
        
#         # Assumimos que offenseFormation e outras características categóricas
#         # já estão codificadas em algum lugar
#         if 'offenseFormation' in G.graph:
#             graph_features.append(float(G.graph['offenseFormation']))
        
#         graph_attr = torch.tensor(graph_features, dtype=torch.float)
        
#         # Definir o alvo (y) - classificação binária: 0 para corrida, 1 para passe
#         # Aqui vamos assumir que existe uma variável no grafo chamada 'playType' 
#         # que indica se foi corrida ou passe
#         play_type = 1 if G.graph.get('playType', 'run') == 'pass' else 0
#         y = torch.tensor([play_type], dtype=torch.long)  # Long tensor para classificação
        
#         # Criar objeto Data do PyTorch Geometric
#         data = Data(
#             x=x,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             y=y,
#             graph_attr=graph_attr
#         )
        
#         data_list.append(data)
    
#     return data_list

# # 2. Implementar o modelo GNN para classificação binária
# class FootballGNN(torch.nn.Module):
#     def __init__(self, node_features, graph_features, hidden_channels):
#         super(FootballGNN, self).__init__()
        
#         # Camadas de convolução do grafo para processamento dos nós
#         self.conv1 = GCNConv(node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
#         # Camada para processar características do grafo
#         self.graph_lin = torch.nn.Linear(graph_features, hidden_channels)
        
#         # Camadas finais para combinar representações de nós e do grafo
#         self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
#         self.lin2 = torch.nn.Linear(hidden_channels, 2)  # 2 classes: corrida ou passe
    
#     def forward(self, data):
#         # print(" Forward pass...")
#         x, edge_index, edge_attr, graph_attr = data.x, data.edge_index, data.edge_attr, data.graph_attr
        
#         # Processamento dos nós
#         x = F.relu(self.conv1(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.2, training=self.training)
        
#         x = F.relu(self.conv2(x, edge_index, edge_attr))
#         x = F.dropout(x, p=0.2, training=self.training)
        
#         x = F.relu(self.conv3(x, edge_index, edge_attr))
        
#         # Pooling para obter uma representação a nível de grafo
#         # Garantir que a saída tenha a forma [1, hidden_channels]
#         x = torch.mean(x, dim=0).unsqueeze(0)  # Forma explícita: [1, hidden_channels]
        
#         # Processamento das características do grafo
#         # Garantir que graph_attr tenha as dimensões corretas [1, num_features]
#         if graph_attr.dim() == 1:
#             graph_attr = graph_attr.unsqueeze(0)  # Adiciona dimensão de batch
            
#         graph_x = F.relu(self.graph_lin(graph_attr))  # [1, hidden_channels]
        
#         # Concatenar características dos nós e do grafo
#         x = torch.cat([x, graph_x], dim=1)  # [1, hidden_channels * 2]
        
#         # Camadas finais
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.2, training=self.training)
#         x = self.lin2(x)
        
#         return F.log_softmax(x, dim=1)  # log_softmax para classificação
    
# # 3. Função de treinamento para um único grafo por vez
# def train(model, optimizer, data):
#     # print(" Training...")
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out, data.y)
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# # 4. Função de avaliação
# @torch.no_grad()
# def evaluate(model, data_list):
#     # print(" Evaluating...")
#     model.eval()
#     correct = 0
    
#     for data in data_list:
#         out = model(data)
#         pred = out.argmax(dim=1)
#         correct += int((pred == data.y).sum())
    
#     return correct / len(data_list)

# # 5. Treinamento do modelo
# def train_model(data_list, test_data_list, epochs=200):
#     print(" Training model...")
#     # Determinar o número de características dos nós e do grafo com base no primeiro exemplo
#     sample_data = data_list[0]
#     node_features = sample_data.x.size(1)
#     graph_features = sample_data.graph_attr.size(0)
    
#     # Inicializar modelo e otimizador
#     model = FootballGNN(node_features, graph_features, hidden_channels=64)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
#     # Treinamento
#     begin = datetime.now()
#     for epoch in range(1, epochs + 1):
#         total_loss = 0
#         for data in data_list:
#             loss = train(model, optimizer, data)
#             total_loss += loss
        
#         train_acc = evaluate(model, data_list)
#         test_acc = evaluate(model, test_data_list)
        
#         if epoch % 10 == 0:
#             end = datetime.now()
#             print(f'    Epoch: {epoch:03d}, Loss: {total_loss/len(data_list):.4f}, '
#                   f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}', f" - Time elapsed: {(end - begin).total_seconds()} seconds")
    
#     return model

# def model_run(graphs):
#     print("Running model...")
#     # 6. Uso do modelo
#     # Vamos supor que você tenha separado seus dados em treinamento e teste
#     data_list = convert_nx_to_pytorch_geometric(graphs[:-10])  # 800 primeiros para treino
#     test_data_list = convert_nx_to_pytorch_geometric(graphs[-10:])  # Resto para teste

#     # Treinar o modelo
#     model = train_model(data_list, test_data_list)

# # Para fazer uma previsão com um novo grafo
# def predict(model, graph):
#     print(" Predicting...")
#     model.eval()
#     data = convert_nx_to_pytorch_geometric([graph])[0]
#     with torch.no_grad():
#         out = model(data)
#     prob = torch.exp(out)
#     pred = out.argmax(dim=1).item()
#     return pred, prob[0]  # Retorna a classe predita e as probabilidades
















































































# import torch
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.data import DataLoader
# from sklearn.model_selection import train_test_split
# from torch_geometric.nn import GCNConv, global_mean_pool

# class GNN(torch.nn.Module):
#     def __init__(self, node_features, global_features, hidden_dim, num_classes):
#         super(GNN, self).__init__()
#         self.conv1 = GCNConv(node_features, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.fc1 = torch.nn.Linear(384 + 73, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

#     def forward(self, data):
#         # Camadas de convolução no grafo
#         x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
        
#         # Pooling global para agregar informações dos nós
#         x = global_mean_pool(x, batch)
        
#         # Expandir as dimensões de u para que seja compatível com x
#         print(f"x shape: {x.shape}")
#         print(f"u shape: {u.shape}")
#         print()
#         if u.dim() == 1:
#             u = u.view(1, -1).repeat(x.size(0), 1)
#         print(f"x shape: {x.shape}")
#         print(f"u shape: {u.shape}")
        
#         # Concatenar atributos globais
#         x = torch.cat([x, u], dim=1)
        
#         # Camadas totalmente conectadas
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# def model_convert_to_pyg_data(graphs):
#     data_list = []
#     for G in graphs:
#         # Criar um mapeamento de nflId para índices consecutivos
#         nflId_to_index = {nflId: i for i, nflId in enumerate(G.nodes)}

#         # Atributos dos nós (jogadores) na ordem dos índices consecutivos
#         x = []
#         for node in G.nodes:
#             attrs = G.nodes[node]
#             x.append([attrs[key] for key in ['playDirection', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'height', 'weight', 'position', 'totalDis']])

#         # Atualizar os índices em edge_index para usar os índices consecutivos
#         edge_index = []
#         edge_attr = []
#         for u, v, attrs in G.edges(data=True):
#             edge_index.append([nflId_to_index[u], nflId_to_index[v]])
#             edge_attr.append([attrs['weight']])

#         # Atributos globais (jogada)
#         u = [G.graph[key] for key in ['quarter', 'down', 'yardsToGo', 'possessionTeam', 'gameClock', 'absoluteYardlineNumber', 'offenseFormation', 'receiverAlignment', 'playClockAtSnap', 'possessionTeamPointDiff']]

#         # Rótulo
#         y = G.graph['playResult']

#         # Criar objeto Data
#         data = Data(
#             x=torch.tensor(x, dtype=torch.float),
#             edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
#             edge_attr=torch.tensor(edge_attr, dtype=torch.float),
#             u=torch.tensor(u, dtype=torch.float),
#             y=torch.tensor([y], dtype=torch.long)
#         )
#         data_list.append(data)
#     return data_list

# def model_run(graphs):
#     print("Running model...")
    
#     data_list = model_convert_to_pyg_data(graphs)
    
#     train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = GNN(node_features=12, global_features=9, hidden_dim=64, num_classes=2).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     criterion = torch.nn.CrossEntropyLoss()

#     for epoch in range(50):
#         model.train()
#         total_loss = 0
#         for data in train_loader:
#             data = data.to(device)
#             optimizer.zero_grad()
#             out = model(data)
#             loss = criterion(out, data.y)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')
    
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             data = data.to(device)
#             out = model(data)
#             pred = out.argmax(dim=1)
#             correct += (pred == data.y).sum().item()
#             total += data.y.size(0)
#     print(f'Accuracy: {correct / total:.4f}')