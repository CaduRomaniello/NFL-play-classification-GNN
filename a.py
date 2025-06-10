import numpy as np

def gerar_matriz_binaria_numpy(linhas, colunas):
    return np.random.randint(0, 2, size=(linhas, colunas))

def complemento(matriz):
    return np.array([[1 - elemento for elemento in linha] for linha in matriz])

n = 20
matriz_gerada = gerar_matriz_binaria_numpy(n, n)

print("\nMatriz binária:")
print(matriz_gerada)


complemento = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        if matriz_gerada[i][j] == 0:
            complemento[i][j] = 1

print("\nComplemento da matriz:")
print(np.array(complemento))

# matriz_gerada = matriz_gerada / np.sum(matriz_gerada)
# complemento = complemento / np.sum(complemento)


print('\nFrobenius matrix: ')
frobenius = complemento - matriz_gerada
frobenius = frobenius/np.sum(frobenius)

frobenius_norm = np.linalg.norm(frobenius)
print(f"\nFrobenius norm: {frobenius_norm}")