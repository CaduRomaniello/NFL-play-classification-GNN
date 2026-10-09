"""Treino do protocolo novo (v2) para GCN, GraphSAGE e DeepSets.

Diferencas em relacao ao GNNTrainer original (que continua intacto, usado nos resultados do texto):
  1. Arestas nos dois sentidos: cada aresta do grafo nao direcionado vira (i, j) e (j, i).
     No original cada aresta entrava em um sentido so.
  2. Peso das arestas disponivel: edge_weight = 1 / (1 + distancia em jardas). O modelo so o usa
     se EDGE_WEIGHT = "inverse".
  3. Atributos padronizados (media 0, desvio 1) com estatisticas calculadas so nos nos do TREINO.
     Inclui os codigos do LabelEncoder, como o StandardScaler fazia no MLP.
  4. Checkpoint correto: copia profunda dos pesos da epoca com maior Macro F1 na validacao, que e
     o modelo avaliado no teste. No original a copia era rasa e o modelo avaliado era o da ultima epoca.
  5. Nao avalia o conjunto de treino a cada epoca (so era usado em log); nao muda o resultado.

O restante do procedimento e igual ao original: AdamW, lote 32, warmup linear seguido de
cosseno, clipping do gradiente em 1.0, early stopping pela perda de validacao.
"""
import copy
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.graph_net import GraphNet
from src.models.set_features import NODE_FEATURES, GLOBAL_FEATURES
from src.utils.logger import Logger

# O SequentialLR chama internamente step(epoch) no agendador do cosseno ao fim do warmup, e o PyTorch
# avisa que esse uso sera descontinuado. O comportamento e correto (o mesmo do codigo original).
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step\\(\\)` was not necessary")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def graphs_to_data(graphs):
    """Converte grafos NetworkX em Data do PyG com os mesmos 23 atributos por no da GCN original
    (13 do jogador + 10 globais replicados), na mesma ordem e na mesma ordem de nos."""
    out = []
    for G in graphs:
        nodes = list(G.nodes())
        index = {n: i for i, n in enumerate(nodes)}
        glob = [float(G.graph[f]) for f in GLOBAL_FEATURES]
        x = [[float(G.nodes[n].get(f, 0)) for f in NODE_FEATURES] + glob for n in nodes]

        src, dst, dist = [], [], []
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0))
            src += [index[u], index[v]]
            dst += [index[v], index[u]]
            dist += [w, w]

        out.append(Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
            edge_weight=1.0 / (1.0 + torch.tensor(dist, dtype=torch.float)),
            y=torch.tensor([1 if G.graph["playResult"] == 1 else 0], dtype=torch.long),
            game_id=int(G.graph.get("gameId", -1)),
            play_id=int(G.graph.get("playId", -1)),
        ))
    return out


class Standardizer:
    """Media e desvio de cada atributo calculados nos nos do conjunto de treino"""

    def __init__(self, train_data):
        X = torch.cat([d.x for d in train_data], dim=0)
        self.mean = X.mean(dim=0)
        std = X.std(dim=0)
        self.std = torch.where(std < 1e-8, torch.ones_like(std), std)

    def apply(self, data_list):
        out = []
        for d in data_list:
            d2 = d.clone()
            d2.x = (d.x - self.mean) / self.std
            out.append(d2)
        return out


def _forward(model, data):
    return model(data.x, data.edge_index, data.batch, data.edge_weight)


def _evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, labels, proba, total_loss = [], [], [], 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = _forward(model, data)
            if criterion is not None:
                total_loss += criterion(out, data.y).item()
            preds.extend(out.argmax(dim=1).cpu().tolist())
            labels.extend(data.y.cpu().tolist())
            proba.extend(F.softmax(out, dim=1)[:, 1].cpu().tolist())
    return preds, labels, proba, total_loss / max(len(loader), 1)


def _inference_time(model, dataset, device, n_samples=200, batch_size=32):
    model.eval()
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    sample = [d.clone() for d in dataset[:n_samples]]  # Data.to() altera no lugar
    with torch.no_grad():
        for d in sample[:20]:
            d = d.to(device)
            _forward(model, _single_batch(d, device))
        sync()
        start = time.perf_counter()
        for d in sample:
            d = d.to(device)
            _forward(model, _single_batch(d, device))
        sync()
        per_play_ms = (time.perf_counter() - start) * 1000 / len(sample)
        loader = DataLoader([d.clone() for d in dataset], batch_size=batch_size, shuffle=False)
        sync()
        start = time.perf_counter()
        for d in loader:
            _forward(model, d.to(device))
        sync()
        full_test_s = time.perf_counter() - start
    return {"per_play_ms": per_play_ms, "full_test_s": full_test_s, "n_test": len(dataset), "device": device.type}


def _single_batch(d, device):
    d.batch = torch.zeros(d.num_nodes, dtype=torch.long, device=device)
    return d


def train_v2(model_params, train_cfg, train_data, val_data, test_data, seed, measure_inference=True):
    """Treina um GraphNet. model_params: namespace com CONV, HIDDEN_CHANNELS, CONV_LAYERS, POOLING,
    POST_LAYERS, DROPOUT, EDGE_WEIGHT, LEARNING_RATE, WEIGHT_DECAY. train_cfg: EPOCHS, WARMUP_EPOCHS,
    EARLY_STOP_PATIENCE, MIN_DELTA, BATCH_SIZE. test_data pode ser None (tuning)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    scaler = Standardizer(train_data)
    train_data, val_data = scaler.apply(train_data), scaler.apply(val_data)
    test_data = scaler.apply(test_data) if test_data is not None else None

    train_loader = DataLoader(train_data, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_cfg.BATCH_SIZE, shuffle=False)

    model = GraphNet(train_data[0].num_node_features, model_params, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_params.LEARNING_RATE,
                                  weight_decay=model_params.WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    warmup, epochs = train_cfg.WARMUP_EPOCHS, train_cfg.EPOCHS
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-6)],
        milestones=[warmup])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1, best_epoch, best_state = -1.0, 0, None
    best_val_loss, patience_counter = float("inf"), 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(_forward(model, data), data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        preds, labels, _, val_loss = _evaluate(model, val_loader, device, criterion)
        val_f1 = f1_score(labels, preds, average="macro")
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy(model.state_dict())

        if val_loss < best_val_loss - train_cfg.MIN_DELTA:
            best_val_loss, patience_counter = val_loss, 0
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 25 == 0:
            Logger.info(f"    [{model_params.CONV}] epoch {epoch:03d} | val loss {val_loss:.4f} | val macro F1 {val_f1:.4f} "
                        f"| best {best_val_f1:.4f} (epoch {best_epoch}) | {time.time() - start:.0f}s | {device.type}")
        if patience_counter >= train_cfg.EARLY_STOP_PATIENCE:
            break
    train_time = time.time() - start

    model.load_state_dict(best_state)
    result = {
        "best_val_macro_f1": best_val_f1,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "train_time_s": train_time,
        "n_parameters": model.get_num_parameters(),
        "standardizer": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()},
    }
    if test_data is not None:
        test_loader = DataLoader(test_data, batch_size=train_cfg.BATCH_SIZE, shuffle=False)
        preds, labels, proba, _ = _evaluate(model, test_loader, device)
        result.update(test_preds=preds, test_labels=labels, test_proba=proba,
                      test_keys=[(d.game_id, d.play_id) for d in test_data])
        if measure_inference:
            result["inference"] = _inference_time(model, test_data, device)
    Logger.info(f"    [{model_params.CONV}] best val macro F1 {best_val_f1:.4f} at epoch {best_epoch} "
                f"(stopped at {epoch}, {train_time:.0f}s)")
    return result
