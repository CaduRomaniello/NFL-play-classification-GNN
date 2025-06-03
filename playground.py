import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

# Matriz de confusão (valores reais)
conf_matrix = np.array([[477, 134],
                        [171, 440]])

# Labels
labels = ['Rush', 'Pass']

# Tamanho da fonte (ajustável)
font_size = 30

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=labels, yticklabels=labels, 
            annot_kws={"size": font_size})

plt.title("Confusion Matrix - Random Forest", fontsize=font_size + 2)
plt.xlabel("Predicted label", fontsize=font_size)
plt.ylabel("True label", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
# plt.colorbar(label='', shrink=0.8)

plt.tight_layout()
plt.show()
