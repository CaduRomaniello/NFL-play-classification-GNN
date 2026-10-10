# Protocolo experimental: bateria de 08 a 10/10/2026 (protocolo v2)

Este documento descreve, em detalhe suficiente para a seção de métodos de um artigo, tudo o que foi configurado e executado nesta bateria de experimentos: dados, atributos, grafos, divisão, modelos, tuning, treino, avaliação, testes estatísticos, máquinas, cronologia, problemas encontrados e limitações. A análise dos resultados está em [RESULTADOS.md](RESULTADOS.md).

Esta pasta é um **snapshot congelado**. Ela contém os resultados brutos, os estudos do Optuna, os hiperparâmetros escolhidos, a configuração, o código exatamente como foi executado e o ambiente. Mudanças futuras no código ou novas execuções no repositório não alteram nada aqui.

---

## 0. Identificação

| Item | Valor |
|---|---|
| Período | 08/10/2026 13h45 a 10/10/2026 05h00 (execução); análise em 10/10/2026 |
| Repositório | `NFL-play-classification-GNN`, branch `experimentos-gnn-justa` |
| Commit de referência | `b72a3b0c361df04c8ed44a64b325d5731263e7e4` |
| Código não commitado no momento do snapshot | só um filtro de aviso em `src/models/trainer_v2.py` (silencia o aviso do PyG sobre `torch-scatter`; não altera resultados). O diff está em `ambiente/git.txt` |
| Integridade | `MANIFEST.sha256` traz o SHA-256 de cada arquivo da pasta |

Para conferir a integridade (Git Bash ou Linux), na raiz desta pasta:

```bash
sha256sum -c MANIFEST.sha256
```

### Estrutura da pasta

| Caminho | Conteúdo |
|---|---|
| `output/comparacao_justa/runs/` | Um JSON por (configuração, semente) do protocolo anterior: RF e MLP (16 configurações) e DeepSets antigo (2 configurações) |
| `output/comparacao_justa/runs_v2/` | Um JSON por (configuração, semente) do protocolo v2: GCN e GraphSAGE nas 6 topologias, DeepSets-tuned e DeepSets-matched |
| `output/results/<topologia>/` | Execuções da GCN usadas no texto da dissertação (29 por topologia), importadas na análise como "(texto)" |
| `output/comparacao_justa/analise/` | Saída do `analyze`: `resumo.csv/.tex`, `relatorio.md`, Friedman/Nemenyi, Wilcoxon, boxplot, diagrama CD, matrizes de confusão |
| `output/comparacao_justa/analise/extra/` | Análise complementar: testes pareados das ablações, topologias, hiperparâmetros, inferência, análise de erros, figuras 1 a 4 e `macro_f1_por_semente.csv` |
| `fair_params.json` | Hiperparâmetros de RF, MLP e DeepSets antigo (padrão do texto e otimizados) |
| `gnn_params.json` | Hiperparâmetros otimizados do protocolo v2 (GCN, GraphSAGE, DeepSets) |
| `optuna_baselines.db` | Estudos do Optuna de RF, MLP e DeepSets antigo |
| `optuna_gnn.db` | Estudos do Optuna do protocolo v2 |
| `optuna_study.db` | Estudos do Optuna da GCN do texto (julho de 2026), guardados para documentar como ela foi otimizada |
| `tuning_historico/` | Todos os trials de todos os estudos em CSV (legível sem o Optuna) e `_resumo_estudos.csv` |
| `config_comparacao_justa.json` | Configuração usada (épocas, paciência, divisão etc.) |
| `codigo/` | Código executado (`src/`, `experimentos_banca.py`, `optuna_optimization.py`), `linux_requirements.txt`, os documentos da época e `scripts/analise_extra.py` |
| `ambiente/` | `pip_freeze.txt` e `git.txt` |
| `relatorio/` | Relatório em HTML com as figuras (o mesmo publicado como artefato) |

Os dados brutos do NFL Big Data Bowl 2025 **não** estão aqui (licença da competição e 8,8 GB). Eles ficam em `data/raw/` do repositório.

---

## 1. Objetivo e histórico

Na defesa, a banca (Prof. Marcos Quiles e Prof. Jadson) apontou que as baselines (RF e MLP) recebiam a **média** dos atributos dos 22 jogadores, enquanto a GCN recebia cada jogador. A vantagem da GCN poderia então vir da perda de informação nas baselines, e não das relações entre jogadores. A banca pediu:
- representações que preservem os jogadores (concatenação ordenada, estatísticas, DeepSets);
- tuning das baselines;
- garantia de que todos os modelos recebem exatamente os mesmos dados;
- análise de erros (matriz de confusão, jogadas fáceis e difíceis), desvio-padrão e boxplot;
- discussão do tempo de inferência.

A bateria teve três etapas:

| Etapa | O que é | Onde está |
|---|---|---|
| **Texto** | GCN da dissertação defendida, 6 topologias, 29 sementes | `output/results/` (só importada, não reexecutada) |
| **Protocolo anterior** (08–09/10) | RF e MLP com 7 representações e tuning na validação; DeepSets com o mesmo treinador da GCN do texto | `output/comparacao_justa/runs/` |
| **Protocolo v2** (09–10/10) | GCN corrigida, GraphSAGE e DeepSets, todos com o mesmo modelo-base, o mesmo treinador e o mesmo protocolo de tuning | `output/comparacao_justa/runs_v2/` |

O protocolo v2 surgiu porque, no protocolo anterior, o DeepSets-matched (a GCN do texto sem arestas) venceu a GCN do texto em 28 de 29 sementes. Ao investigar, apareceram problemas no código da GCN (seção 10). O v2 corrige esses problemas e coloca GCN, GraphSAGE e DeepSets nas mesmas condições. **RF e MLP não precisaram ser reexecutados**: o tuning deles já usava a validação, o MLP já recebia dados padronizados e o RF não depende de escala.

---

## 2. Dados

### 2.1 Fonte e recorte
- NFL Big Data Bowl 2025 (temporada 2022): `games.csv`, `plays.csv`, `players.csv`, `player_play.csv` e `tracking_week_1.csv` a `tracking_week_9.csv`.
- Semanas 1 a 9. Um quadro por jogada: o quadro com `frameType == "SNAP"`.
- A bola (`displayName == "football"`) é removida; ficam os 22 jogadores.

### 2.2 Rótulo (passe = 1, corrida = 0)
Regras em `codigo/src/data/preprocessor.py`, `_play_result`, aplicadas nesta ordem:

| Condição | Rótulo |
|---|---|
| `qbSpike`, `qbKneel` ou `qbSneak` verdadeiro | jogada excluída |
| `passResult == "R"` (scramble do QB) | jogada excluída |
| `rushLocationType` preenchido | corrida |
| `passLocationType` preenchido | passe |
| `passResult` preenchido | passe |

Total: **15.416 jogadas**, sendo **9.313 passes (60,4%)** e **6.103 corridas (39,6%)**. Passes com play action e RPO entram com o rótulo do que de fato aconteceu.

### 2.3 Atributos
Os mesmos para **todos os modelos**: 13 por jogador e 10 globais da jogada.

| Por jogador (13) | Definição |
|---|---|
| `x`, `y` | posição em jardas, nas coordenadas brutas do campo (sem espelhar pelo sentido do ataque) |
| `s`, `a`, `dis` | velocidade, aceleração e distância percorrida no quadro |
| `o`, `dir` | orientação e direção do movimento, em graus (0 a 360). Valores ausentes: 90° ou 270°, conforme o sentido do ataque e o time (`_o_invalid_values`, `_dir_invalid_values`) |
| `height`, `weight` | altura e peso (`players.csv`) |
| `position` | posição do jogador, codificada com `LabelEncoder` |
| `club` | time do jogador, codificado com `LabelEncoder` |
| `playDirection` | 0 = esquerda, 1 = direita |
| `totalDis` | soma de `dis` em todos os quadros até o snap (`frameType != "AFTER_SNAP"`) |

| Globais (10) | Definição |
|---|---|
| `quarter`, `down`, `yardsToGo` | quarto, descida e jardas para a primeira descida |
| `absoluteYardlineNumber` | linha de scrimmage em coordenadas absolutas |
| `playClockAtSnap` | relógio de jogada no snap (ausente → 0) |
| `possessionTeamPointDiff` | placar do time com a posse menos o do adversário |
| `possessionTeam` | time com a posse, `LabelEncoder` |
| `gameClock` | relógio do quarto em segundos |
| `offenseFormation`, `receiverAlignment` | formação e alinhamento, `LabelEncoder` (ausente → `"EMPTY"`) |

Detalhes que valem para todos os modelos:
- Os `LabelEncoder` são ajustados **semana a semana** (seção 11).
- Nos modelos em grafo e no DeepSets, os 10 globais são replicados em cada nó, o que dá 23 atributos por nó.
- Dois atributos auxiliares são calculados mas **não entram como coluna em nenhum modelo**: `isOffense` (`club == possessionTeam`) e `positionName`. Eles só servem para ordenar ou agrupar jogadores em três representações das baselines (seção 5.2).

---

## 3. Grafos

Cada jogada vira um grafo com 22 nós (jogadores). As arestas vêm de uma de 6 topologias, calculadas sobre as posições (`x`, `y`) no snap. A distância euclidiana (jardas) é guardada em cada aresta.

| Topologia | Definição |
|---|---|
| `MST` | Árvore geradora mínima do grafo completo ponderado pela distância (21 arestas) |
| `RNG` | Grafo de vizinhança relativa: aresta (i, j) se nenhum outro jogador está mais próximo de i e de j ao mesmo tempo do que eles estão entre si |
| `GABRIEL` | Grafo de Gabriel: aresta (i, j) se nenhum outro jogador está dentro do círculo de diâmetro ij |
| `DELAUNAY` | Triangulação de Delaunay |
| `CLOSEST-` | Cada jogador é ligado aos N = 2 jogadores mais próximos |
| `QB-CLOSEST-` | Como `CLOSEST-` (N = 2), e o QB é ligado a todos os outros jogadores |

Parâmetros: `N = 2`, `QB_LINK = false` e `DOWN_SAMPLE = false` em todas as execuções (o texto também usou esses valores em todas as topologias). Os grafos são construídos uma vez e guardados em cache (`data/graphs/cache/`, fora do git).

**Como as arestas entram no modelo:**
- **GCN do texto e DeepSets antigo:** cada aresta entra em um sentido só (`[origem, destino]`, na ordem de inserção) e sem peso. Na MST, o `edge_index` tem 21 colunas.
- **Protocolo v2:** cada aresta entra nos dois sentidos (42 colunas na MST). O peso `1/(1 + distância)` fica disponível, e o tuning decide se ele é usado.

---

## 4. Divisão dos dados

- Para cada semente, as listas de passes e de corridas são embaralhadas separadamente com `random.seed(semente)` + `random.shuffle` do Python. Depois cada classe é cortada em 70% / 15% / 15% (com truncamento inteiro). A proporção de classes fica igual nos três conjuntos.
- Tamanhos (iguais em todas as sementes):

| Conjunto | Passes | Corridas | Total |
|---|---|---|---|
| Treino | 6.519 | 4.272 | 10.791 |
| Validação | 1.396 | 915 | 2.311 |
| Teste | 1.398 | 916 | 2.314 |

- **Sementes de avaliação: 1 a 29.** São as que existem para a GCN do texto (o texto fala em 30 execuções, mas há 29). **Semente de tuning: 0**, fora das sementes de avaliação.
- **Todos os modelos usam exatamente a mesma divisão em cada semente.** `common.make_split` reproduz a divisão que a GCN do texto usou; isso foi validado jogada a jogada. Assim, os resultados podem ser pareados por semente nos testes estatísticos.

---

## 5. Modelos

### 5.1 GCN do texto (importada, não reexecutada)
- `GCNConv(23 → 256)` → ReLU → pooling médio global → dropout 0,2 → `Linear(256 → 2)`.
- AdamW (lr 0,001, weight decay 5·10⁻⁴), lote 32, até 500 épocas, warmup linear de 20 épocas (fator inicial 0,1) seguido de cosseno, clipping do gradiente em 1,0, early stopping pela perda de validação com paciência de 100.
- O modelo avaliado no teste é o da **última** época (seção 10, item 2). Atributos sem padronização.
- Tuning (julho de 2026, `optuna_study.db`, `codigo/optuna_optimization.py`): 50 trials por topologia; espaço = canais {128, 256}, camadas 1 a 3, `DOWN_SAMPLE` e `QB_LINK`; objetivo = **acurácia no conjunto de teste** (seção 10, item 3). A configuração final do texto foi 256 canais e 1 camada em todas as topologias.

### 5.2 Baselines vetoriais: RF e MLP (protocolo anterior)
Os nós são ordenados por `nflId` antes de montar o vetor, para que o resultado não dependa da topologia. A jogada vira um vetor de tamanho fixo de 7 formas, sempre com os mesmos 13 + 10 atributos:

| Representação | Dimensão | Construção | Grupo |
|---|---|---|---|
| `mean` | 23 | média dos 22 jogadores + globais (a do texto) | estritamente justa |
| `stats` | 62 | média, desvio-padrão, mínimo e máximo de cada atributo + globais | estritamente justa |
| `concat_raw` | 296 | 22 jogadores na ordem de inserção no grafo (arbitrária) + globais | estritamente justa (controle negativo) |
| `concat_xy` | 296 | 22 jogadores ordenados por profundidade no sentido do ataque (`x` espelhado por `playDirection`) e, no empate, por `y` | estritamente justa |
| `stats_team` | 114 | `stats` separado para ataque e defesa | com lado de campo |
| `concat_team_y` | 296 | ataque e depois defesa (11 + 11); dentro de cada time, por `y` no sentido do ataque | com lado de campo |
| `concat_team_role` | 296 | ataque (QB, RB/FB, WR, TE, OL) e defesa (DL, LB, CB, S); dentro de cada função, por `y` | com lado de campo |

"Com lado de campo" quer dizer que a posição no vetor revela se o jogador é do ataque ou da defesa (via `isOffense`), informação que os modelos em grafo e o DeepSets não recebem.

**RF:** `RandomForestClassifier` (scikit-learn 1.7.1), `random_state = semente`, `n_jobs = -1`. Configuração do texto (`RF-mean-default`): 200 árvores, demais parâmetros padrão.

**MLP:** `MLPClassifier` (scikit-learn 1.7.1), `random_state = semente`, entrada padronizada com `StandardScaler` ajustado no treino. Configuração do texto (`MLP-mean-default`): 2 camadas de 64, lr 0,01, `alpha` 5·10⁻⁴, `max_iter` 3000.

**Espaços de busca:**

| RF | Valores |
|---|---|
| `n_estimators` | {100, 200, 500} |
| `max_depth` | {None, 10, 20, 40} |
| `min_samples_leaf` | {1, 2, 4, 8} |
| `max_features` | {"sqrt", "log2", 0,25, 0,5} |

| MLP | Valores |
|---|---|
| número de camadas | 1 a 3 (inteiro) |
| largura (igual em todas as camadas) | {64, 128, 256} |
| `alpha` | 10⁻⁶ a 10⁻¹ (log) |
| `learning_rate_init` | 10⁻⁴ a 10⁻² (log) |
| `batch_size` | {32, 64, 128, 256} |
| fixos | `early_stopping = True`, `validation_fraction = 0,1`, `n_iter_no_change = 20`, `max_iter = 1000` |

Tuning: 50 trials por (modelo, representação). Os hiperparâmetros escolhidos estão em `fair_params.json` (`RF.tuned`, `MLP.tuned`), e o histórico de cada trial em `tuning_historico/`.

### 5.3 DeepSets do protocolo anterior ("antigo")
- `codigo/src/models/deepsets.py`: φ (camadas lineares + ReLU, pesos compartilhados entre jogadores) → pooling (média, máximo ou média+máximo) → ρ (camadas lineares) → `Linear(→ 2)`. Recebe a mesma matriz de 23 atributos por nó e ignora as arestas.
- Mesmo treinador da GCN do texto (`GNNTrainer.train_on_split`): sem padronização, paciência 100, modelo da última época.
- **DeepSets-matched (antigo):** `Linear(23 → 256)` → ReLU → pooling médio → dropout 0,2 → `Linear(256 → 2)`, com os hiperparâmetros da GCN do texto. Equivale à GCN do texto sem arestas.
- **DeepSets-tuned (antigo):** 30 trials. Espaço: canais {64, 128, 256}, camadas de φ 1 a 3, camadas de ρ 0 a 2, pooling {média, máximo, média+máximo}, dropout 0 a 0,5, lr 10⁻⁴ a 10⁻² (log), weight decay 10⁻⁶ a 10⁻³ (log). Escolhido: 128 canais, φ com 3 camadas, ρ com 1, pooling médio, dropout 0,013, lr 0,0065, weight decay 9,3·10⁻⁶ (val. 0,8432).

### 5.4 Protocolo v2: GCN, GraphSAGE e DeepSets
**Modelo único** (`codigo/src/models/graph_net.py`, `GraphNet`), com a mesma estrutura do modelo do texto, generalizada:

```
[troca de mensagens → ReLU (→ dropout entre camadas)] × CONV_LAYERS
→ pooling (média, máximo ou média+máximo)
→ [dropout → Linear → ReLU] × POST_LAYERS
→ dropout → Linear (2 classes)
```

A única diferença entre os três modelos é a camada de troca de mensagens:

| Modelo | Camada (PyTorch Geometric 2.6.1) | Atualização de cada jogador i |
|---|---|---|
| GCN | `GCNConv` (com autolaços e normalização simétrica) | h_i = W · Σ_{j ∈ N(i) ∪ {i}} c_ij · w_ij · x_j |
| GraphSAGE | `GraphConv(aggr="mean")` | h_i = W₁ · x_i + W₂ · média_{j ∈ N(i)} (w_ij · x_j) |
| DeepSets | `Linear` | h_i = W · x_i (sem vizinhos) |

w_ij = 1/(1 + d_ij) quando `EDGE_WEIGHT = inverse` e 1 quando `none`. A GraphSAGE foi implementada com `GraphConv(aggr="mean")`, que é a GraphSAGE com agregação média e aceita peso nas arestas. Inicialização Xavier uniforme nas camadas lineares, viés zero.

**Treinador** (`codigo/src/models/trainer_v2.py`, `train_v2`). Diferenças em relação ao do texto:
1. Arestas nos dois sentidos e peso pela distância disponível.
2. Atributos padronizados (média 0, desvio 1) com estatísticas calculadas **só nos nós do treino**, inclusive os códigos do `LabelEncoder` (como o `StandardScaler` faz no MLP). Atributo com desvio < 10⁻⁸ não é dividido.
3. **Checkpoint correto:** cópia profunda (`copy.deepcopy`) dos pesos da época com **maior Macro F1 na validação**; esse é o modelo avaliado no teste.
4. Early stopping pela **perda de validação**, paciência de **30** épocas e `MIN_DELTA` 0,001.
5. O conjunto de treino não é avaliado a cada época (só servia para log; não altera o resultado).

Iguais ao texto: AdamW (β = 0,9 / 0,999, ε = 10⁻⁸), lote 32, até 500 épocas, warmup linear de 20 épocas (fator inicial 0,1) seguido de cosseno até 10⁻⁶, clipping do gradiente em 1,0 e entropia cruzada sem pesos de classe. Semente: `random`, `numpy`, `torch` e `torch.cuda` fixados por semente, com `cudnn.deterministic = True` e `cudnn.benchmark = False`.

Por que a paciência mudou de 100 para 30: no texto, o modelo avaliado era o da última época, então a paciência longa definia o modelo final. Com o checkpoint correto, esperar 100 épocas só gastava tempo. No primeiro teste com os dados completos, o melhor F1 veio na época 16 e o treino seguiu até a 116. A mudança vale igualmente para os três modelos.

**Espaço de busca (igual para os três):**

| Hiperparâmetro | Valores |
|---|---|
| Canais ocultos | {64, 128, 256} |
| Camadas de troca de mensagens | 1 a 3 |
| Pooling | {média, máximo, média+máximo} |
| Camadas após o pooling | 0 a 2 |
| Dropout | 0 a 0,5 |
| Taxa de aprendizado | 10⁻⁴ a 10⁻² (log) |
| Weight decay | 10⁻⁶ a 10⁻³ (log) |
| Peso das arestas (só GCN e GraphSAGE) | {nenhum, 1/(1+d)} |

**Estudos:** 13, com 20 trials completos cada: GCN × 6 topologias, GraphSAGE × 6 topologias e DeepSets × 1 (sem arestas, então sem topologia). O DeepSets lê os atributos do cache da MST.

**Hiperparâmetros escolhidos** (`gnn_params.json`):

| Configuração | Canais | Camadas | Pooling | Pós-pooling | Dropout | lr | Weight decay | Peso | F1 val. |
|---|---|---|---|---|---|---|---|---|---|
| GCN-MST | 256 | 1 | média | 2 | 0,483 | 4,14·10⁻³ | 8,2·10⁻⁶ | 1/(1+d) | 0,8317 |
| GCN-RNG | 64 | 2 | média+máx. | 1 | 0,106 | 1,05·10⁻⁴ | 6,0·10⁻⁶ | 1/(1+d) | 0,8331 |
| GCN-CLOSEST | 64 | 2 | média+máx. | 1 | 0,286 | 2,33·10⁻⁴ | 2,3·10⁻⁶ | 1/(1+d) | 0,8303 |
| GCN-GABRIEL | 64 | 1 | média | 1 | 0,132 | 5,05·10⁻⁴ | 6,1·10⁻⁶ | 1/(1+d) | 0,8318 |
| GCN-QB-CLOSEST | 64 | 1 | média | 1 | 0,108 | 9,73·10⁻⁴ | 8,9·10⁻⁶ | 1/(1+d) | 0,8288 |
| GCN-DELAUNAY | 64 | 1 | média | 1 | 0,132 | 5,05·10⁻⁴ | 6,1·10⁻⁶ | 1/(1+d) | 0,8272 |
| GraphSAGE-MST | 128 | 2 | média+máx. | 0 | 0,062 | 2,48·10⁻⁴ | 1,4·10⁻⁴ | 1/(1+d) | 0,8484 |
| GraphSAGE-RNG | 128 | 2 | máx. | 0 | 0,151 | 7,59·10⁻⁴ | 3,8·10⁻⁵ | 1/(1+d) | 0,8527 |
| GraphSAGE-CLOSEST | 64 | 3 | média+máx. | 1 | 0,180 | 2,07·10⁻⁴ | 1,7·10⁻⁴ | 1/(1+d) | 0,8521 |
| GraphSAGE-GABRIEL | 128 | 2 | máx. | 1 | 0,326 | 1,12·10⁻⁴ | 1,1·10⁻⁶ | 1/(1+d) | 0,8493 |
| GraphSAGE-QB-CLOSEST | 128 | 2 | máx. | 1 | 0,361 | 8,11·10⁻⁴ | 9,7·10⁻⁴ | 1/(1+d) | 0,8499 |
| GraphSAGE-DELAUNAY | 64 | 3 | média+máx. | 0 | 0,071 | 8,80·10⁻⁴ | 9,1·10⁻⁵ | 1/(1+d) | 0,8562 |
| DeepSets | 256 | 3 | média+máx. | 0 | 0,304 | 2,19·10⁻⁴ | 2,2·10⁻⁶ | — | 0,8516 |

GCN-GABRIEL e GCN-DELAUNAY terminaram com os mesmos hiperparâmetros. Os dois estudos usam o mesmo amostrador com a mesma semente, então propõem a mesma sequência inicial de trials, e o mesmo trial venceu nos dois.

**Configurações executadas nas sementes 1 a 29 (14):**
- `GCN-<topologia>` e `SAGE-<topologia>` nas 6 topologias, com os hiperparâmetros da tabela;
- `DeepSets-tuned`, com os hiperparâmetros do estudo do DeepSets;
- `DeepSets-matched`: os hiperparâmetros da **GCN-MST** otimizada com `CONV = none` e `EDGE_WEIGHT = none`. É a mesma rede sem arestas, ou seja, a ablação direta do efeito das arestas na GCN.

---

## 6. Protocolo de tuning (comum a todas as etapas desta bateria)

| Item | Valor |
|---|---|
| Biblioteca | Optuna 4.7.0 |
| Amostrador | `optuna.samplers.TPESampler(seed=42)`, configuração padrão (10 trials iniciais aleatórios) |
| Divisão | a da semente 0, fora das sementes de avaliação |
| Treino | conjunto de treino da semente 0 |
| Objetivo | maximizar o **Macro F1 no conjunto de validação**. O teste não é usado em nenhuma escolha |
| Trials | RF e MLP: 50 por representação. DeepSets antigo: 30. Protocolo v2: 20 por estudo |
| Armazenamento | SQLite (`optuna_baselines.db`, `optuna_gnn.db`); estudos retomáveis. Só trials `COMPLETE` contam para o total |
| Escolha final | o trial com maior valor de validação |

Trials interrompidos (Ctrl+C) ou com erro ficam registrados como `FAIL` e não contam. Ocorreram em: MLP-concat_raw (2), DeepSets antigo (1, o erro da seção 10, item 7), GCN-MST (1), GCN-CLOSEST- (2), GCN-QB-CLOSEST- (1) e DeepSets v2 (1). Ao retomar, o amostrador é recriado com a mesma semente; por isso, um estudo interrompido pode seguir uma sequência de trials diferente da que teria sem a interrupção.

---

## 7. Avaliação

### 7.1 Métricas
Para cada (configuração, semente), no conjunto de teste (2.314 jogadas): Macro F1 (métrica principal), acurácia, precisão/revocação/F1 por classe e matriz de confusão. Cada JSON guarda também a predição e a probabilidade de passe de cada jogada (`test_keys` = gameId, playId), a época escolhida e a de parada (v2), o tempo de treino e o tempo de inferência.

### 7.2 Testes estatísticos (`codigo/src/experiments/analysis.py`)
- **Configurações nos testes:** as 30 do protocolo v2 e de RF/MLP. As versões "(texto)" e "(antigo)" aparecem nas tabelas, mas ficam fora dos testes globais, para não contar a mesma configuração duas vezes.
- **Friedman:** `scipy.stats.friedmanchisquare` sobre a matriz semente × configuração (29 × 30).
- **Nemenyi:** diferença crítica CD = q_α · √(k(k+1)/(6N)), com q_0,05 = amplitude studentizada (graus de liberdade infinitos) / √2; k = 30 e N = 29, o que dá q = 3,749 e CD = 8,666. Os p-valores par a par vêm de `scikit_posthocs.posthoc_nemenyi_friedman` (scikit-posthocs 0.12.0).
- **Wilcoxon pareado por semente:** `scipy.stats.wilcoxon` (bilateral; com 29 pares e sem empates, usa a distribuição exata), contra a melhor GNN do v2 (GraphSAGE-RNG), com correção de Holm sobre as 29 comparações.
- **Testes das ablações** (`analise/extra/testes_pareados.csv`): mesmo Wilcoxon, com correção de Holm sobre as 27 comparações daquele arquivo.
- Com 29 pares, o menor p bilateral possível no Wilcoxon exato é 2/2²⁹ ≈ 3,7·10⁻⁹. Os p-valores de 1,0·10⁻⁷ nas tabelas são esse piso multiplicado pelo número de testes na correção de Holm (o A vence em 29 de 29 sementes).

### 7.3 Análise de erros (`codigo/scripts/analise_extra.py`)
- Modelos: DeepSets-tuned, SAGE-RNG, GCN-MST (v2), MLP-concat_xy-tuned e RF-mean-default.
- A predição de cada jogada de teste, em cada semente, é unida ao `plays.csv` por (gameId, playId).
- Grupos: classe; passe com e sem `playAction`; `pff_runPassOption` (RPO); `offenseFormation`; `down`; faixas de `yardsToGo` (1–2, 3–6, 7–10, 11+).
- Unidade de contagem: predição (jogada × semente). Cada jogada cai no teste em cerca de 4 das 29 sementes, o que dá 67.106 predições por modelo sobre 15.285 jogadas distintas.
- Dificuldade por jogada: fração de acertos nas sementes em que ela caiu no teste. "Sempre errada pelos 5 modelos" = nenhum dos 5 acertou em nenhuma dessas sementes.

### 7.4 Tempo de inferência
- Redes (GPU): média de 200 jogadas do teste com lote de 1, depois de 20 jogadas de aquecimento, com `torch.cuda.synchronize` antes e depois. Inclui a cópia da jogada para a GPU.
- RF e MLP (CPU): `predict` em uma jogada por vez (200 jogadas, 20 de aquecimento), com o RF em `n_jobs = 1`.
- A tabela usa a mediana das 29 sementes.
- **Ressalva:** as execuções rodaram com 3 processos em paralelo (seção 8). Os tempos são limites superiores e servem para ordem de grandeza.

---

## 8. Ambiente, máquinas e cronologia

### 8.1 Máquina principal (todas as execuções de avaliação)
- AMD Ryzen 7 9800X3D (8 núcleos / 16 threads), 32 GB de RAM, NVIDIA GeForce RTX 5070 (12 GB, driver 617.42), Windows 11.
- Python 3.13.12; PyTorch 2.10.0+cu128 (CUDA 12.8, cuDNN 9.10.02); PyTorch Geometric 2.6.1 (sem `torch-scatter`); scikit-learn 1.7.1; Optuna 4.7.0; NumPy 2.3.2; pandas 2.3.2; SciPy 1.16.1; NetworkX 3.5; scikit-posthocs 0.12.0. Lista completa em `ambiente/pip_freeze.txt`.

### 8.2 Cronologia

| Quando | Máquina | O que |
|---|---|---|
| 08/10, 13h45–17h42 | computador original (Linux) | Tuning RF (7 representações) e MLP `mean`, `stats` e os primeiros 10 trials de `concat_raw` |
| 08/10, 23h18–23h35 | máquina principal | Restante do tuning do MLP |
| 08/10, 23h19 – 09/10, 03h00 | máquina principal (GPU) | Tuning do DeepSets antigo (30 trials) |
| 08/10, 23h50 – 09/10, 00h33 | máquina principal (CPU) | Execução de RF e MLP, 16 configurações × 29 sementes |
| 08/10, 23h59 – 09/10, 10h22 | máquina principal (GPU) | Execução do DeepSets antigo (matched e tuned) |
| 09/10, 11h26–17h44 | notebook (só CPU) | Tuning v2: GCN-MST, GCN-RNG e 17 trials da GCN-CLOSEST- |
| 09/10, 20h01–23h06 | máquina principal (GPU, 3 processos) | Tuning v2: os outros 3 trials da GCN-CLOSEST- e os outros 10 estudos |
| 09/10, 23h15 – 10/10, 04h58 | máquina principal (GPU, 3 processos: sementes 1–10, 11–20 e 21–29) | Execução v2, 14 configurações × 29 sementes |
| 10/10 | máquina principal | Análise (`analyze` e `analise_extra.py`) |

O tuning do v2 começou num notebook sem GPU e terminou na máquina principal; os trials são independentes e usam a mesma divisão e o mesmo código. **Todas as execuções de avaliação** (sementes 1–29) rodaram na máquina principal.

Duração média por trial (de `tuning_historico/_resumo_estudos.csv`): RF 5–44 s, MLP 2,5–83 s, DeepSets antigo 407 s, v2 na GPU 92–141 s (na CPU do notebook, 308–465 s).

---

## 9. Comandos executados

Na raiz do repositório, com o venv ativo:

```bash
# cache de grafos
python experimentos_banca.py build --strategies MST
python experimentos_banca.py build --strategies RNG CLOSEST- GABRIEL QB-CLOSEST- DELAUNAY

# protocolo anterior
python experimentos_banca.py tune rf:all --trials 50
python experimentos_banca.py tune mlp:all --trials 50
python experimentos_banca.py tune deepsets --trials 30
python experimentos_banca.py run --seeds 1-29 --models rf mlp
python experimentos_banca.py run --seeds 1-29 --models deepsets --deepsets-variants matched
python experimentos_banca.py run --seeds 1-29 --models deepsets --deepsets-variants tuned

# protocolo v2 (3 terminais em paralelo em cada etapa)
python experimentos_banca.py tune-gnn gcn:MST gcn:RNG gcn:CLOSEST- gcn:GABRIEL
python experimentos_banca.py tune-gnn gcn:QB-CLOSEST- gcn:DELAUNAY sage:MST sage:RNG
python experimentos_banca.py tune-gnn deepsets sage:CLOSEST- sage:GABRIEL sage:QB-CLOSEST- sage:DELAUNAY
python experimentos_banca.py run-gnn --seeds 1-10
python experimentos_banca.py run-gnn --seeds 11-20
python experimentos_banca.py run-gnn --seeds 21-29

# análise
python experimentos_banca.py analyze --old-gcn-dir output/results
python analise_extra.py . data/raw/plays.csv
```

### Como refazer a análise só com este snapshot
Os resultados estão na mesma estrutura do repositório, então a análise pode ser refeita sem executar nenhum treino. Na pasta `codigo/`:

```bash
python experimentos_banca.py analyze --out <copia>/output/comparacao_justa --old-gcn-dir ../output/results
python scripts/analise_extra.py .. <caminho>/data/raw/plays.csv
```

O `analyze` grava em `<out>/analise/`; use uma cópia de `output/comparacao_justa/` para não sobrescrever o snapshot. Isso foi testado em 10/10: `resumo.csv`, `friedman_nemenyi.txt` e `wilcoxon_vs_ref.csv` saíram idênticos aos desta pasta.

Para refazer os treinos: recrie o ambiente (`ambiente/pip_freeze.txt`), copie `data/raw/` e rode os comandos acima com o código de `codigo/`. Na GPU, pequenas diferenças são esperadas (seção 11).

---

## 10. Problemas encontrados no código e o que foi feito

| # | Problema | Efeito | Situação |
|---|---|---|---|
| 1 | **Divisão dupla:** em `TrainingPipeline.execute`, `split_and_prepare_data` era chamado duas vezes sobre listas já embaralhadas, e RF/MLP do texto rodaram em outra divisão, não na da GCN | Sem viés, mas quebrava o pareamento por semente e contradizia a Seção 3.5 do texto | Corrigido nesta bateria: todos os modelos usam a divisão da GCN (`common.make_split`) |
| 2 | **"Melhor modelo" era o último:** `model.state_dict().copy()` é cópia rasa, e os pesos guardados seguiam sendo atualizados | Na GCN do texto (e no DeepSets antigo), o modelo avaliado foi o da última época | Corrigido no v2 (`deepcopy`, melhor Macro F1 na validação). Mantido no treinador antigo para preservar os números do texto |
| 3 | **Tuning da GCN do texto pela acurácia no teste** (`optuna_optimization.py`) | Vazamento na seleção de hiperparâmetros, a favor da GCN do texto | Corrigido no v2 (validação). Declarar como limitação dos resultados do texto |
| 4 | **Arestas em um sentido só:** `convert_nx_to_pytorch_geometric` adicionava só `[origem, destino]` | Metade da troca de mensagens não acontecia, num sentido arbitrário | Corrigido no v2 |
| 5 | **Distância calculada mas não usada** (`edge_attr` não era passado ao modelo) | A GCN do texto ignorava a distância | Corrigido no v2: peso 1/(1+d) como hiperparâmetro |
| 6 | **Atributos sem padronização** na GCN (x até 120, gameClock até 900), enquanto o MLP recebia dados padronizados | Otimização mais difícil para a GCN | Corrigido no v2 (estatísticas só do treino) |
| 7 | **Erro no tempo de inferência:** `Data.to(device)` do PyG altera o objeto no lugar, e parte do teste ficava na GPU e parte na CPU | O primeiro trial do tuning do DeepSets antigo falhou depois do treino | Corrigido antes de qualquer resultado válido (commit `d49d63e`); o v2 usa cópias |
| 8 | **MLP variava entre topologias com a mesma semente:** a ordem dos nós mudava com a topologia e alterava a soma de ponto flutuante da média | Resultados do MLP do texto não determinísticos entre topologias | Corrigido nesta bateria (nós ordenados por `nflId`) |
| 9 | `fcntl` (só Unix) na trava de arquivos do v2 | O v2 não rodava no Windows | Trocado por `filelock` (commit `b72a3b0`) |
| 10 | Script de análise complementar lia `output/results/first_tests` como GCN-DELAUNAY do texto | Figura de topologias distorcida na primeira versão | Corrigido antes de gerar as figuras finais; o `analysis.py` oficial nunca teve esse problema |

---

## 11. Limitações conhecidas (não corrigidas nesta bateria)

1. **Coordenadas não normalizadas pelo sentido do ataque:** `x`, `y`, `o` e `dir` são brutos, com `playDirection` como atributo. A prática comum em dados de rastreamento é espelhar o campo para que o ataque sempre avance no mesmo sentido e medir `x` a partir da linha de scrimmage.
2. **Ângulos em graus** padronizados como escalares; o usual é seno e cosseno.
3. **Nenhum indicador de lado de campo nos atributos** (ataque ou defesa). Os modelos em grafo e o DeepSets só podem inferir o lado comparando `club` com `possessionTeam`.
4. **Categóricas como inteiros** (`LabelEncoder`), e não one-hot ou embedding. Os codificadores são ajustados **semana a semana**: se o conjunto de categorias muda entre semanas (por exemplo, times em bye a partir da semana 6), o mesmo código representa categorias diferentes. Afeta todos os modelos igualmente. Ainda não foi verificado nos dados quanto isso acontece.
5. **Tuning com orçamento limitado e uma única divisão:** 20 trials no v2 (50 para RF/MLP). O Macro F1 de validação do trial escolhido é levemente otimista, porque o checkpoint também é escolhido pelo máximo do Macro F1 na validação; isso não afeta o teste.
6. **Orçamentos diferentes entre etapas:** RF e MLP com 50 trials por representação, DeepSets antigo com 30 e v2 com 20. Dentro do v2, todos os estudos têm o mesmo orçamento e o mesmo espaço (o DeepSets só não tem a dimensão do peso das arestas).
7. **Não determinismo na GPU:** a agregação do `GCNConv` e do pooling usa operações atômicas; repetir uma execução pode mudar as últimas casas decimais.
8. **Tempo de inferência medido com 3 processos simultâneos** (valores conservadores).
9. **29 sementes**, e não 30 como diz o texto.
10. **Uma temporada** (semanas 1–9 de 2022) e **só o quadro do snap**.
11. **Arquiteturas de GNN testadas: GCN e GraphSAGE.** Mensagens com posição relativa (EdgeConv, GATv2 ou Transformer com atributos de aresta), grafos completos com atenção e arestas tipadas não foram testados.

---

## 12. Texto sugerido para a seção de métodos do artigo

> Todos os modelos foram avaliados em 29 divisões aleatórias dos dados (sementes 1 a 29), com 70% das jogadas de cada classe para treino, 15% para validação e 15% para teste (2.314 jogadas). Em cada semente, todos os modelos usam exatamente a mesma divisão, o que permite comparações pareadas. Os hiperparâmetros foram escolhidos com o Optuna 4.7.0 (`TPESampler`, semente 42), maximizando o Macro F1 no conjunto de validação de uma divisão separada (semente 0), sem uso do conjunto de teste: 50 trials por representação para o Random Forest e o MLP, e 20 trials por modelo e topologia para a GCN, a GraphSAGE e o DeepSets, com o mesmo espaço de busca para os três. As redes compartilham a mesma estrutura (camadas de troca de mensagens, pooling global, camadas densas) e diferem apenas na troca de mensagens: `GCNConv`, GraphSAGE com agregação média ou nenhuma (DeepSets). O treino usou AdamW, lote de 32, até 500 épocas, warmup linear de 20 épocas seguido de decaimento por cosseno, e early stopping pela perda de validação com paciência de 30 épocas; o modelo avaliado no teste é o da época com maior Macro F1 de validação. Os atributos foram padronizados com média e desvio calculados no treino. As arestas entram nos dois sentidos, e o peso 1/(1 + d), com d a distância em jardas, foi tratado como hiperparâmetro. As diferenças foram avaliadas com o teste de Friedman seguido do pós-teste de Nemenyi (α = 0,05) e com o teste de Wilcoxon pareado por semente, com correção de Holm.
