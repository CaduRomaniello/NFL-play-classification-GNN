# Protocolo v2: GCN, GraphSAGE e DeepSets em condições iguais

Branch: `experimentos-gnn-justa` (criada a partir de `experimentos-comparacao-justa`).

## Por que existe

A comparação anterior (`EXPERIMENTOS_BANCA.md`) mostrou que o DeepSets-matched, que é a GCN sem arestas, superava a GCN. Ao investigar, apareceram problemas no código que prejudicavam a GCN. Este protocolo corrige esses problemas e reexecuta, nas mesmas condições, todos os modelos que trocam mensagens ou trabalham sobre conjuntos.

| Problema no código original | Correção no v2 |
|---|---|
| Arestas em um sentido só (21 colunas no `edge_index` da MST em vez de 42) | Cada aresta entra nos dois sentidos |
| Distância entre jogadores calculada, mas não usada | Peso `1/(1+distância)`, opcional; o tuning decide (`EDGE_WEIGHT` = `none` ou `inverse`) |
| Hiperparâmetros da GCN escolhidos pela acurácia no **teste** e busca pequena | Mesmo protocolo das baselines: TPE (seed 42), divisão da semente 0, Macro F1 na **validação**, 20 trials, o mesmo espaço de busca para todos |
| "Melhor modelo" era o da última época (cópia rasa dos pesos) | Cópia profunda; o modelo avaliado é o da época com maior Macro F1 na validação |
| Atributos brutos (x até 120, gameClock até 900...), enquanto o MLP recebia dados padronizados | Atributos padronizados (média 0, desvio 1) com estatísticas só do treino, inclusive os códigos do LabelEncoder (como o `StandardScaler` fazia no MLP) |

**Não muda:**
- RF e MLP não precisam rodar de novo: o tuning deles já usou a validação, o MLP já recebia dados padronizados e o RF não depende de escala.
- Os resultados antigos (GCN do texto e GCN/DeepSets do protocolo anterior) não são sobrescritos. Na análise, eles aparecem como `(texto)` e `(antigo)`.
- Divisão dos dados: é a mesma de todos os outros experimentos (sementes 1–29).

## Modelos

Um único modelo, [src/models/graph_net.py](src/models/graph_net.py), com a mesma estrutura do modelo do texto:

```
[troca de mensagens -> ReLU] x CONV_LAYERS -> pooling -> [dropout -> Linear -> ReLU] x POST_LAYERS -> dropout -> Linear
```

A única diferença entre os modelos é a troca de mensagens:
- **GCN:** `GCNConv`, a convolução do texto. O jogador e os vizinhos são misturados.
- **GraphSAGE:** `W1·xᵢ + W2·média(vizinhos)`. O próprio jogador e os vizinhos têm pesos separados, então a rede pode aprender a ignorar os vizinhos. Foi implementado com `GraphConv(aggr="mean")` do PyG, que é o SAGE-mean com suporte a peso nas arestas.
- **DeepSets:** sem troca de mensagens (`W·xᵢ`).

Espaço de busca, igual para os três:

| Hiperparâmetro | Valores |
|---|---|
| Canais ocultos | {64, 128, 256} |
| Camadas de troca de mensagens | 1 a 3 |
| Pooling | média, máximo, média+máximo |
| Camadas após o pooling | 0 a 2 |
| Dropout | 0 a 0,5 |
| Taxa de aprendizado | 1e-4 a 1e-2 (log) |
| Weight decay | 1e-6 a 1e-3 (log) |
| Peso das arestas (só GCN e GraphSAGE) | nenhum, `1/(1+d)` |

O treino é fixo e igual ao do texto (AdamW, lote 32, até 500 épocas, warmup de 20, cosseno, early stopping pela perda de validação), **exceto a paciência do early stopping: 30 épocas em vez de 100**. Esses valores estão em `config_comparacao_justa.json`, seção `GNN_V2`.

Por que a paciência mudou: no código original, o modelo avaliado era o da última época, então uma paciência longa determinava diretamente o modelo final. No v2, o checkpoint guarda a época com o melhor Macro F1 de validação, e esperar 100 épocas só gastava tempo. No primeiro teste com os dados completos, o melhor F1 veio na época 16 e o treino seguiu até a 116, com a perda de validação subindo (86% do tempo desperdiçado). A paciência de 30 vale igualmente para GCN, GraphSAGE e DeepSets.

Configurações executadas (29 sementes cada):
- `GCN-<topologia>` e `SAGE-<topologia>` para as 6 topologias (12 configurações);
- `DeepSets-tuned`, com o tuning próprio do DeepSets;
- `DeepSets-matched`, que é a GCN-MST otimizada sem arestas: ablação direta do efeito das arestas.

Arquivos próprios, para não haver conflito com o protocolo anterior:
- `gnn_params.json`: hiperparâmetros escolhidos;
- `optuna_gnn.db`: estudos do Optuna;
- `output/comparacao_justa/runs_v2/`: resultados.

---

## Como executar

> **Use a máquina com GPU NVIDIA.** Sem GPU (por exemplo, no notebook com Intel Iris Xe), o treino roda na CPU e fica várias vezes mais lento. Os comandos `tune-gnn` e `run-gnn` mostram um aviso quando a CUDA não está disponível, e o log de progresso a cada 25 épocas mostra o dispositivo em uso (`cuda` ou `cpu`). Para conferir antes de começar:
>
> ```bash
> python -c "import torch; print(torch.cuda.is_available())"
> ```

Rode tudo na raiz do repositório, na branch `experimentos-gnn-justa`, com o venv ativo **em cada terminal**:

```bash
source venv/bin/activate
```

Todos os comandos podem ser interrompidos (Ctrl+C) e retomados: basta rodar o mesmo comando de novo. Os trials do Optuna e os resultados das sementes que já terminaram são aproveitados.

### Passo 1: cache das 6 topologias (um terminal, ~30 min, CPU)

O cache da MST já existe nesta máquina. Falta construir os das outras 5:

```bash
python experimentos_banca.py build --strategies RNG CLOSEST- GABRIEL QB-CLOSEST- DELAUNAY
```

O cache guarda os grafos do NetworkX com a distância em cada aresta. A correção dos dois sentidos e o cálculo do peso acontecem na conversão para o PyG, então **não é preciso reconstruir o cache da MST**. Se for rodar em outra máquina, copie a pasta `data/graphs/cache/` ou rode o `build` com as 6 topologias.

### Passo 2: tuning (3 terminais ao mesmo tempo, GPU)

São 13 estudos de 20 trials (padrão de `--trials`), divididos em três grupos de tamanho parecido. Rode um comando em cada terminal:

**Terminal 1:**
```bash
python experimentos_banca.py tune-gnn gcn:MST gcn:RNG gcn:CLOSEST- gcn:GABRIEL
```

**Terminal 2:**
```bash
python experimentos_banca.py tune-gnn gcn:QB-CLOSEST- gcn:DELAUNAY sage:MST sage:RNG
```

**Terminal 3:**
```bash
python experimentos_banca.py tune-gnn deepsets sage:CLOSEST- sage:GABRIEL sage:QB-CLOSEST- sage:DELAUNAY
```

- Cada estudo grava o melhor resultado em `gnn_params.json` assim que termina. Os três processos podem escrever ao mesmo tempo, porque o arquivo é protegido por trava.
- `--threads` (padrão 4) limita as threads de CPU de cada processo, para os três não disputarem os 12 núcleos.
- Para acompanhar, veja o log de cada terminal ou confira o `_tuning_info` em `gnn_params.json`.
- **Espere os 3 terminais terminarem antes do Passo 3.** Se você rodar o Passo 3 antes, as configurações ainda sem tuning são puladas com um aviso. Nesse caso, basta rodar o Passo 3 de novo depois.

### Passo 3: execução nas sementes 1–29 (3 terminais ao mesmo tempo, GPU)

Os três terminais rodam todos os modelos, cada um com um bloco de sementes.

**Terminal 1:**
```bash
python experimentos_banca.py run-gnn --seeds 1-10
```

**Terminal 2:**
```bash
python experimentos_banca.py run-gnn --seeds 11-20
```

**Terminal 3:**
```bash
python experimentos_banca.py run-gnn --seeds 21-29
```

- Cada execução vira `output/comparacao_justa/runs_v2/<modelo>/seed_XXX.json`, com métricas, matriz de confusão, predições por jogada, época escolhida e tempo de inferência.
- Cada processo carrega uma topologia por vez, para economizar RAM.

### Passo 4: análise (um terminal)

```bash
python experimentos_banca.py analyze --old-gcn-dir output/results
```

- A tabela traz todos os modelos: v2, RF/MLP e, para comparação, os antigos marcados com `(texto)` ou `(antigo)`.
- **Os testes estatísticos (Friedman/Nemenyi e Wilcoxon) usam só os modelos v2 e RF/MLP.** Para incluir os antigos, acrescente `--include-old-in-tests`.
- O Wilcoxon compara todos os modelos contra a melhor GNN do v2 (GCN ou GraphSAGE).
- Os arquivos são salvos em `output/comparacao_justa/analise/`.

**Quando terminar, me chame** para analisarmos `analise/` e `gnn_params.json`.

---

## Tempo estimado

Medido no notebook (CPU, um processo, paciência 100): de 2 a 8 s por época, dependendo do tamanho do modelo, e de 6 a 22 minutos por trial. Com a paciência de 30, cada treino deve cair para algo entre 40% e 60% disso.

| Etapa | Treinos | Máquina com GPU, 3 processos (estimativa) | Notebook (CPU), 3 processos (estimativa) |
|---|---|---|---|
| Passo 1 (`build`) | — | ~30 min (CPU) | ~30 min |
| Passo 2 (tuning) | 13 × 20 = 260 | **~5–10 h** | ~1,5–2,5 dias |
| Passo 3 (execução) | 14 × 29 = 406 | **~5–10 h** | ~1,5–2,5 dias |

As estimativas para a GPU usam como referência o DeepSets do protocolo anterior (~1,6 s por época, incluindo a avaliação do treino, que o v2 removeu). Depois dos primeiros trials, o log a cada 25 épocas mostra o tempo acumulado e o dispositivo. Me passe esses números que eu refaço a conta.

O padrão é 20 trials por estudo. Para comparar os modelos, o importante é que todos usem o mesmo número de trials.

## Git

- Os resultados novos ficam em pastas e arquivos próprios (`runs_v2/`, `gnn_params.json`, `optuna_gnn.db`), sem conflito com a outra máquina.
- Se a outra máquina ainda estiver rodando o DeepSets-tuned do protocolo anterior, **não faça `git pull` nela** até terminar. Depois, ela faz commit e push só de `output/comparacao_justa/runs/DeepSets-tuned/`, e esta branch faz o merge sem conflito. Esse resultado aparece na análise como `DeepSets-tuned (antigo)`.
- O arquivo `gnn_params.json.lock` é temporário e está no `.gitignore`.

## Validações feitas (escala reduzida: semana 1, 25 épocas)

1. Os atributos de entrada são idênticos aos da GCN original (mesmos 23 por nó, mesma ordem).
2. Os grafos ficam não direcionados no PyG (42 colunas no `edge_index` da MST contra 21 no original), e o peso é exatamente `1/(1+distância)`.
3. A padronização usa só o treino: média ~0 e desvio 1 no treino, e não exatamente 0 no teste. Os dados originais não são alterados.
4. A GCN com um grafo sem arestas produz a mesma saída que o DeepSets (com os mesmos pesos).
5. O peso das arestas só altera a saída quando `EDGE_WEIGHT = inverse` (GCN e GraphSAGE).
6. Checkpoint: o modelo restaurado reproduz exatamente o melhor Macro F1 de validação, e foi a época 16 de 25, não a última.
7. Três processos de tuning em paralelo e dois de execução, compartilhando o banco e o arquivo de parâmetros. Isso revelou e permitiu corrigir uma condição de corrida na criação do banco SQLite.
8. A análise combina v2, RF/MLP e os resultados antigos corretamente.
