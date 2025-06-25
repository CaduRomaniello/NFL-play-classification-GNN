import os
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(data, timestamp, n):
    """
    Gera e salva a matriz de confusão como uma imagem PNG.

    Args:
        y_true (list or array): Rótulos verdadeiros.
        y_pred (list or array): Predições do modelo.
        model_name (str): Nome do modelo (ex: "GCN").
        output_dir (str): Diretório onde salvar a imagem.
    """

    best_gcn_matrix = data['best_gcn_results']['confusion_matrix']
    last_gcn_matrix = data['last_gcn_results']['confusion_matrix']
    rf_matrix = data['rf_results']['confusion_matrix']
    mlp_matrix = data['mlp_results']['confusion_matrix']

    cur_path = os.getcwd()
    out_path = os.path.abspath(os.path.join(cur_path, f'Mestrado/eniac/n{n}/{timestamp}'))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Criar as visualizações das matrizes de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=best_gcn_matrix, display_labels=["Rush", "Pass"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - GCN")
    output_path = f"{out_path}/confusion_matrix_best_gcn.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=last_gcn_matrix, display_labels=["Rush", "Pass"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - GCN")
    output_path = f"{out_path}/confusion_matrix_last_gcn.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=rf_matrix, display_labels=["Rush", "Pass"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - Random Forest")
    output_path = f"{out_path}/confusion_matrix_rf.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=mlp_matrix, display_labels=["Rush", "Pass"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - MLP")
    output_path = f"{out_path}/confusion_matrix_mlp.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()