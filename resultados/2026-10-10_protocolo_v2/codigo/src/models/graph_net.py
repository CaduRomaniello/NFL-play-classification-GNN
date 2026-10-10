import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool, global_max_pool, global_add_pool


class GraphNet(torch.nn.Module):
    """Modelo unico do protocolo novo (v2) para GCN, GraphSAGE e DeepSets.

    Mesma estrutura do modelo original do texto, generalizada:

        [camada de troca de mensagens -> ReLU] x CONV_LAYERS
        -> pooling (media, maximo ou media+maximo)
        -> [dropout -> Linear -> ReLU] x POST_LAYERS
        -> dropout -> Linear (2 classes)

    A unica diferenca entre os modelos e a camada de troca de mensagens (CONV):
        "gcn"   GCNConv: h_i = W * sum_{j in N(i) U {i}} c_ij x_j   (vizinhos e o proprio no misturados)
        "sage"  GraphSAGE com agregacao media: h_i = W1 x_i + W2 * media_{j in N(i)} w_ji x_j
                (o proprio no e os vizinhos tem pesos separados; implementado com GraphConv(aggr="mean")
                do PyG, que e o SAGEConv-mean com suporte a peso nas arestas)
        "none"  DeepSets: h_i = W x_i (cada jogador sozinho, sem arestas)

    Com CONV="gcn", 1 camada, POOLING="mean" e POST_LAYERS=0, e o modelo do texto.
    """

    POOLS = {"mean": global_mean_pool, "max": global_max_pool, "sum": global_add_pool}

    def __init__(self, node_features, params, seed):
        super().__init__()
        torch.manual_seed(seed)

        self.conv_type = params.CONV
        self.hidden = params.HIDDEN_CHANNELS
        self.dropout = params.DROPOUT
        self.pooling = params.POOLING
        self.use_edge_weight = getattr(params, "EDGE_WEIGHT", "none") != "none"

        if params.CONV_LAYERS < 1:
            raise ValueError("CONV_LAYERS must be at least 1")

        self.convs = torch.nn.ModuleList()
        in_dim = node_features
        for _ in range(params.CONV_LAYERS):
            if self.conv_type == "gcn":
                self.convs.append(GCNConv(in_dim, self.hidden))
            elif self.conv_type == "sage":
                self.convs.append(GraphConv(in_dim, self.hidden, aggr="mean"))
            elif self.conv_type == "none":
                self.convs.append(torch.nn.Linear(in_dim, self.hidden))
            else:
                raise ValueError(f"Unknown CONV {self.conv_type}")
            in_dim = self.hidden

        in_dim = self.hidden * (2 if self.pooling == "mean+max" else 1)
        self.post = torch.nn.ModuleList()
        for _ in range(params.POST_LAYERS):
            self.post.append(torch.nn.Linear(in_dim, self.hidden))
            in_dim = self.hidden
        self.classifier = torch.nn.Linear(in_dim, 2)

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def _pool(self, x, batch):
        if self.pooling == "mean+max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.POOLS[self.pooling](x, batch)

    def forward(self, x, edge_index, batch, edge_weight=None):
        w = edge_weight if self.use_edge_weight else None
        for i, conv in enumerate(self.convs):
            if self.conv_type == "none":
                x = conv(x)
            else:
                x = conv(x, edge_index, w)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self._pool(x, batch)

        for layer in self.post:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
