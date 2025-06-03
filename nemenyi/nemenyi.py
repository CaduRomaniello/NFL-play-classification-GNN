import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import io

# Dados CSV fornecidos pelo usuário como uma string
csv_data = """GCN,RF,MLP
0.77,0.73,0.69
0.66,0.74,0.67
0.65,0.73,0.69
0.78,0.74,0.67
0.57,0.73,0.73
0.76,0.73,0.69
0.75,0.74,0.68
0.62,0.74,0.73
0.78,0.73,0.73
0.76,0.74,0.73
0.77,0.74,0.68
0.77,0.74,0.65
0.68,0.74,0.69
0.69,0.74,0.70
0.70,0.74,0.68
0.60,0.73,0.74
0.77,0.74,0.73
0.74,0.73,0.68
0.69,0.73,0.67
0.76,0.74,0.68
0.69,0.74,0.74
0.75,0.74,0.68
0.55,0.74,0.70
0.54,0.74,0.70
0.78,0.74,0.69
"""

# Carregar os dados do CSV para um DataFrame do pandas
# O io.StringIO permite ler a string como se fosse um arquivo
df = pd.read_csv('./rush_f1_score.csv', sep=',', header=0)

# Exibir as primeiras linhas do DataFrame para verificar se foi carregado corretamente
print("Dados Carregados:")
print(df.head())
print("-" * 30)

# --- Passo 1: Realizar o Teste de Friedman ---
# O Teste de Friedman verifica se há diferenças estatisticamente significativas
# entre as medianas de três ou mais grupos pareados (nossos modelos, com cada iteração/linha sendo um "bloco").
# Precisamos passar os dados de cada modelo como argumentos separados.
# df['GCN'], df['RF'], df['MLP'] são as colunas de acurácia para cada modelo.

statistic_friedman, p_value_friedman = friedmanchisquare(df['GCN'], df['RF'], df['MLP'])

print(f"Estatística do Teste de Friedman: {statistic_friedman:.4f}")
print(f"Valor-p do Teste de Friedman: {p_value_friedman:.4f}")
print("-" * 30)

# Interpretação do Teste de Friedman:
# Se o valor-p for menor que o nível de significância (alfa, geralmente 0.05),
# isso sugere que há pelo menos uma diferença significativa entre os desempenhos dos modelos.
alpha = 0.05
if p_value_friedman < alpha:
    print("O Teste de Friedman indica uma diferença significativa entre os modelos (p < 0.05).")
    print("Prosseguindo com o Teste de Nemenyi para comparações par a par.")
    print("-" * 30)

    # --- Passo 2: Realizar o Teste de Nemenyi ---
    # O Teste de Nemenyi é um teste post-hoc para identificar quais pares específicos
    # de modelos são significativamente diferentes.
    # A função posthoc_nemenyi_friedman espera os dados em um formato "longo" ou
    # diretamente os dados originais como um array numpy ou lista de listas.
    # No nosso caso, podemos passar os dados do DataFrame diretamente,
    # mas a biblioteca 'scikit-posthocs' geralmente prefere os dados em um formato "melted" (longo)
    # ou pode aceitar um array numpy onde cada coluna é um grupo.
    # Vamos usar os dados como estão, pois a função pode lidar com isso se passarmos
    # os dados como uma matriz numpy.
    
    data_for_nemenyi = df[['GCN', 'RF', 'MLP']].values 
    # .values converte o DataFrame para um array NumPy
    
    # Realiza o teste de Nemenyi.
    # Os resultados são uma matriz de p-valores para cada comparação par a par.
    nemenyi_results = sp.posthoc_nemenyi_friedman(data_for_nemenyi)
    
    # A função posthoc_nemenyi_friedman retorna um DataFrame com os p-valores.
    # Para facilitar a leitura, vamos nomear as colunas e índices com os nomes dos modelos.
    nemenyi_results.columns = ['GCN', 'RF', 'MLP']
    nemenyi_results.index = ['GCN', 'RF', 'MLP']

    print("Resultados do Teste de Nemenyi (matriz de p-valores):")
    print(nemenyi_results)
    print("-" * 30)

    print("Interpretação do Teste de Nemenyi:")
    print(f"Compare os p-valores da matriz acima com o seu nível de significância (alfa = {alpha}).")
    print("Se um p-valor entre dois modelos for < alfa, a diferença entre eles é estatisticamente significativa.")
    print("Valores NaN ou 1.0000 na diagonal são esperados (comparação de um modelo consigo mesmo).")

else:
    print("O Teste de Friedman NÃO indica uma diferença significativa entre os modelos (p >= 0.05).")
    print("O Teste de Nemenyi não é usualmente realizado neste caso, pois não há diferença geral a ser detalhada.")