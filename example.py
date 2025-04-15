import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn

from torch_geometric.utils import from_networkx

# Converter NetworkX Graph para PyTorch Geometric Data
data = from_networkx(G)

# Exibir estrutura do grafo convertido
print(data)

#! preciso entender melhor isso aqui
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = geom_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = geom_nn.GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Definir modelo
input_dim = data.num_node_features  #? Número de features de entrada, isso aqui é a quantidade de features dos nós?
hidden_dim = 32 #? não entendi o que é isso, seria a qtd de layers?
output_dim = 2  # Exemplo: saída binária (passe ou corrida)

model = GNN(input_dim, hidden_dim, output_dim)
print(model)


# Exemplo de labels (alvo do modelo, playType vale 0 para corrida e 1 para passe)
y = torch.tensor(play['playType'].values, dtype=torch.long)

# Criando otimizador e loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) #! estudar mais sobre otimizadores
loss_fn = nn.CrossEntropyLoss() #! estudar mais sobre loss functions

# Loop de treinamento
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data)  # Forward #? aqui a variável data teria que ser uma lista de grafos (um pra cada jogada)?
    loss = loss_fn(out, y)  # Calcula a perda #! variável y representa os labels, entender pq ele é passado aqui
    loss.backward() # Backward #! entender o que é isso
    optimizer.step() # Atualiza os pesos #! entender o que é isso
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')