try:
    import torch_geometric
    print("PyTorch Geometric importado com sucesso.")
    print(f"Versão: {torch_geometric.__version__}")
except Exception as e:
    print(f"Erro ao importar: {e}")