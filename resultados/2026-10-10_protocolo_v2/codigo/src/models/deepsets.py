import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class DeepSets(torch.nn.Module):
    """DeepSets (Zaheer et al., 2017): rho(POOL_i phi(x_i)).

    Recebe exatamente a mesma matriz de atributos dos nos usada pela GCN
    (13 atributos do jogador + 10 atributos globais replicados), mas ignora as
    arestas. Com PHI_LAYERS=1, RHO_LAYERS=0 e POOLING="mean" o modelo equivale
    a GCN com um grafo sem arestas (apenas auto-lacos): Linear -> ReLU ->
    pooling medio -> dropout -> Linear. E o controle direto do efeito das arestas.

    A assinatura de forward e a mesma da GCN para reaproveitar o GNNTrainer.
    """

    POOLS = {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}

    def __init__(self, node_features, config):
        super().__init__()
        torch.manual_seed(config.RANDOM_SEED)

        params = config.DEEPSETS
        self.hidden_channels = params.HIDDEN_CHANNELS
        self.phi_layers = params.PHI_LAYERS
        self.rho_layers = params.RHO_LAYERS
        self.dropout = params.DROPOUT
        self.pooling = params.POOLING  # "mean", "max", "sum" ou "mean+max"

        if self.phi_layers < 1:
            raise ValueError("PHI_LAYERS must be at least 1")

        # phi: aplicada a cada jogador de forma independente (pesos compartilhados)
        self.phi = torch.nn.ModuleList()
        self.phi.append(torch.nn.Linear(node_features, self.hidden_channels))
        for _ in range(self.phi_layers - 1):
            self.phi.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))

        pooled_dim = self.hidden_channels * (2 if self.pooling == "mean+max" else 1)

        # rho: aplicada ao vetor agregado da jogada
        self.rho = torch.nn.ModuleList()
        in_dim = pooled_dim
        for _ in range(self.rho_layers):
            self.rho.append(torch.nn.Linear(in_dim, self.hidden_channels))
            in_dim = self.hidden_channels
        self.classifier = torch.nn.Linear(in_dim, 2)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def _pool(self, x, batch):
        if self.pooling == "mean+max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.POOLS[self.pooling](x, batch)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # edge_index e ignorado de proposito: o modelo nao usa a estrutura do grafo
        for i, layer in enumerate(self.phi):
            x = F.relu(layer(x))
            if i < len(self.phi) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._pool(x, batch)

        for layer in self.rho:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
