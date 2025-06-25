# test_torch_geometric.py
import os
# Comentar/descomentar para testar com GPU/CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

from contextlib import redirect_stdout
from datetime import datetime
import time
import json
import random
import numpy as np
import torch.nn.functional as F
# import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

## GNN
from datetime import datetime
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# MLP and RF
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import networkx as nx

import torch
print(f"PyTorch versão: {torch.__version__}")
print(f"CUDA disponível: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")

try:
    import torch_geometric
    print(f"PyTorch Geometric versão: {torch_geometric.__version__}")
    
    from torch_geometric.data import Data
    print("Criando um objeto Data de exemplo...")
    
    # Criar um grafo simples
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[1], [2], [3]], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    print(f"Grafo criado com {data.num_nodes} nós e {data.num_edges} arestas")
    print("PyTorch Geometric funcionando corretamente!")
    
except Exception as e:
    print(f"Erro ao usar PyTorch Geometric: {e}")


# try:
#     import torch_geometric
#     print("PyTorch Geometric importado com sucesso.")
#     print(f"Versão: {torch_geometric.__version__}")
# except Exception as e:
#     print(f"Erro ao importar: {e}")