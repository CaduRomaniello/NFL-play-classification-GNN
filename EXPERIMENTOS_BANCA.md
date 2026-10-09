# Experimentos pedidos pela banca: comparação justa GCN × baselines

Branch: `experimentos-comparacao-justa`

Estes experimentos respondem às críticas do Prof. Marcos Quiles sobre a comparação entre a GCN e as baselines.

**Nada muda nos experimentos da GCN.** As instruções abaixo executam **só as baselines** (DeepSets, RF e MLP). Os resultados da GCN usados no texto (`output/results`) entram direto na análise, sem rodar nada de novo. As alterações em `src/data` e `src/models/gcn_trainer.py` não mudam a entrada nem o treino da GCN (validado: os tensores de entrada são idênticos aos do pipeline original).

| Crítica | O que foi implementado |
|---|---|
| As baselines recebiam só a **média** dos 22 jogadores, o que pode ter degradado a informação | 7 representações vetoriais para RF e MLP (tabela abaixo), entre elas a **concatenação ordenada**, que contém exatamente a mesma informação de entrada da GCN (sem as arestas) |
| "Olhar DeepSets. Fazer as duas abordagens: com concatenação e DeepSets" | Modelo **DeepSets** em PyTorch, com a mesma matriz de atributos dos nós da GCN e o mesmo procedimento de treino |
| "Treinar uma MLP para criar uma representação latente dos jogadores" | É o próprio DeepSets: a rede φ aprende uma representação latente de cada jogador antes do pooling |
| "Você fez a otimização dos hiperparâmetros dos baselines?" | Optuna (TPE) para RF, MLP e DeepSets, em cada representação, usando o conjunto de **validação** |
| "Seria preciso garantir que você está fornecendo exatamente os mesmos dados" | Todos os modelos usam **a mesma divisão treino/validação/teste** em cada semente (ver "Problemas encontrados no código", item 1) |
| Jadson: tempo de inferência em tempo real | Cada execução mede o tempo de inferência por jogada (lote de 1) |

### Representações das baselines (RF e MLP)

**As colunas de entrada são sempre as mesmas da GCN:** os 13 atributos por jogador (`x, y, s, a, dis, o, dir, height, weight, position, club, playDirection, totalDis`) e os 10 atributos globais. Nenhum atributo novo entra no vetor de nenhum modelo.

Os dois atributos auxiliares adicionados aos nós (`isOffense` = `club == possessionTeam` e `positionName`) servem **apenas para ordenar ou agrupar** os jogadores. Mesmo assim, há uma sutileza: quando os jogadores são agrupados por lado, a posição de cada um no vetor indica se ele é do ataque ou da defesa. É uma informação implícita que a GCN não recebe de forma explícita. Por isso, as representações estão divididas em dois grupos:

**Estritamente justas** (usam só os atributos que a GCN recebe; são as que devem sustentar a conclusão principal):

| Nome | Dimensão | Descrição |
|---|---|---|
| `mean` | 23 | média dos jogadores (a representação usada no texto) |
| `stats` | 62 | média, desvio-padrão, mínimo e máximo de cada atributo |
| `concat_raw` | 296 | concatenação na ordem arbitrária dos nós (controle negativo: mostra por que a ordenação importa) |
| `concat_xy` | 296 | concatenação dos 22 jogadores ordenados por profundidade no sentido do ataque (`x`, espelhado pela `playDirection`) e, em empate, por `y`. A ordem é calculada só com atributos que já estão na entrada |

**Com informação de lado de campo na ordenação** (análise complementar; se forem citadas no texto, deixe essa diferença explícita):

| Nome | Dimensão | Descrição |
|---|---|---|
| `stats_team` | 114 | `stats` calculado separadamente para ataque e para defesa |
| `concat_team_y` | 296 | ataque e depois defesa; dentro de cada time, da lateral esquerda para a direita |
| `concat_team_role` | 296 | ataque (QB, RB, WR, TE, OL) e defesa (DL, LB, CB, S); dentro de cada função, ordenado por `y` |

O DeepSets não usa nenhum dos dois atributos auxiliares: recebe exatamente a mesma matriz de atributos dos nós da GCN e só deixa de usar as arestas.

### Variantes do DeepSets

* `DeepSets-matched`: Linear(23→256) → ReLU → pooling médio → dropout → Linear(256→2), com os **mesmos hiperparâmetros da GCN**. Isso equivale a uma GCN com um grafo sem arestas (só autolaços). É o controle mais direto para medir o efeito das arestas: a única diferença em relação à GCN é a troca de mensagens entre vizinhos.
* `DeepSets-tuned`: hiperparâmetros escolhidos pelo Optuna (camadas de φ e ρ, largura, pooling média/máximo/média+máximo, dropout, taxa de aprendizado, weight decay).

---

## Como executar

Todos os comandos são rodados na raiz do repositório, com o venv ativo:

```bash
source venv/bin/activate
```

### Passo 1: construir o cache de grafos (só uma vez)

As baselines ignoram as arestas, então basta o cache de uma topologia (MST), usado só para ler os atributos dos jogadores. Os grafos vão para `data/graphs/cache/`, que já está no `.gitignore`.

```bash
python experimentos_banca.py build --strategies MST
```

### Passo 2: otimizar os hiperparâmetros das baselines

O protocolo é o mesmo para todos: divisão da semente 0 (fora das sementes de avaliação 1–29), treino no conjunto de treino e objetivo igual ao **Macro F1 na validação** (o teste não é usado), com `TPESampler(seed=42)`. Os estudos ficam em `optuna_baselines.db`, então dá para interromper e retomar. Os melhores parâmetros são gravados em `fair_params.json`.

```bash
python experimentos_banca.py tune rf:all --trials 50
```

```bash
python experimentos_banca.py tune mlp:all --trials 50
```

```bash
python experimentos_banca.py tune deepsets --trials 30
```

* `rf:all` e `mlp:all` otimizam as 7 representações (50 trials cada). Para otimizar só as estritamente justas, use `rf:strict` e `mlp:strict`. RF e MLP são rápidos (segundos por trial). O DeepSets é o mais lento: cada trial é um treino completo de até 500 épocas.
* Os três comandos podem rodar ao mesmo tempo em terminais separados. Os comandos de RF e MLP usam a CPU e o do DeepSets usa a GPU.
* Também dá para otimizar uma representação só, por exemplo `tune rf:concat_team_y`.

### Passo 3: rodar as baselines (sementes 1–29, mesma divisão da GCN)

```bash
python experimentos_banca.py run --seeds 1-29 --models deepsets rf mlp
```

Esse comando roda:
* DeepSets-matched e DeepSets-tuned;
* RF e MLP com as 7 representações e os parâmetros do Optuna (`-tuned`);
* RF e MLP com `mean` e a configuração do texto (`-default`), que são as baselines da Tabela 4.1 executadas na mesma divisão da GCN.

Para rodar só as representações estritamente justas, acrescente `--reprs mean stats concat_raw concat_xy`.

Cada execução vira um arquivo `output/comparacao_justa/runs/<modelo>/seed_XXX.json`, com métricas, matriz de confusão, predição de cada jogada (gameId/playId, rótulo, predição e probabilidade de passe) e tempo de inferência. **Execuções que já existem são puladas**, então você pode parar (Ctrl+C) e retomar a qualquer momento.

Para aproveitar CPU e GPU ao mesmo tempo, rode em dois terminais:

```bash
python experimentos_banca.py run --seeds 1-29 --models deepsets
```

```bash
python experimentos_banca.py run --seeds 1-29 --models rf mlp
```

Usei as sementes 1–29 porque são as que existem em `output/results` para a GCN. Nenhum desses comandos executa a GCN. Para executá-la, veja o Apêndice A.

Sugestão: depois da semente 1, olhe o tempo gasto (campo `train_time_s` nos JSONs) para estimar o total.

> **Atenção (baselines `-default`):** no código antigo, RF e MLP rodavam em uma divisão diferente da GCN (item 1 da seção "Problemas encontrados no código atual"). Agora eles rodam na divisão da GCN, como o texto descreve. Por isso, `RF-mean-default` e `MLP-mean-default` podem diferir um pouco dos valores de RF e MLP da Tabela 4.1. A GCN não muda.

### Passo 4: análise

```bash
python experimentos_banca.py analyze --old-gcn-dir output/results
```

`--old-gcn-dir` importa as execuções da GCN usadas no texto, identificadas como `GCN-<topologia> (texto)`. As baselines novas usam exatamente a mesma divisão de cada semente que essas execuções, então os resultados podem ser pareados por semente nos testes estatísticos.

A análise salva os arquivos em `output/comparacao_justa/analise/`:
* `relatorio.md`: resumo de tudo;
* `resumo.csv` / `resumo.tex`: Macro F1 (média, desvio, mínimo, máximo), acurácia, F1 por classe e tempo de inferência;
* `boxplot_macro_f1.png/pdf`: um ponto por execução;
* `friedman_nemenyi.txt`, `nemenyi_pvalues.csv` e `nemenyi_cd.png/pdf`: diagrama de diferença crítica;
* `wilcoxon_vs_ref.csv`: Wilcoxon pareado por semente contra a melhor GCN, com correção de Holm;
* `matrizes_confusao.png/pdf`: as duas melhores configurações de cada família.

**Quando terminar, me chame de novo** para analisarmos `output/comparacao_justa/analise/` e `fair_params.json`.

---

## Problemas encontrados no código atual (importantes para o texto)

1. **As baselines não eram avaliadas na mesma divisão da GCN.** Em `TrainingPipeline.execute`, `train_model` embaralha e divide os dados (divisão da GCN). Depois, `split_and_prepare_data` é chamado **de novo** sobre as listas já embaralhadas, e o resultado é outra divisão, que foi a usada pelo RF e pelo MLP. O texto (Seção 3.5) diz "as mesmas divisões de treinamento, validação e teste que a GCN", o que não é verdade para os resultados atuais. Isso não introduz viés (as duas divisões são aleatórias e estratificadas), mas quebra o pareamento por semente que os testes de Friedman/Nemenyi supõem. **No código novo, todos os modelos usam a divisão da GCN** (validado: a divisão gerada é idêntica, jogada a jogada, à que a GCN usou).
2. **O "melhor modelo" da GCN é, na prática, o último.** Em `gcn_trainer.py`, `best_model_state = model.state_dict().copy()` faz uma cópia rasa: os tensores continuam sendo os do modelo, que seguem sendo atualizados. Por isso, em todos os JSONs, `best_gcn_results` é igual a `last_gcn_results`. **Não corrigi**, para não mudar os números da GCN no texto. O DeepSets usa o mesmo treinador, então o procedimento é idêntico para os dois. Para corrigir no futuro, basta usar `copy.deepcopy(model.state_dict())`.
3. **O Optuna da GCN otimizou a acurácia no conjunto de teste** (`optuna_optimization.py`: `results['best_gcn_results']['accuracy']`). Isso é vazamento na seleção de hiperparâmetros e favorece a GCN. Nas baselines novas, o tuning usa a validação. Vale citar isso como limitação no texto ou reotimizar a GCN usando a validação.
4. **Por que o MLP variava entre topologias com a mesma semente** (comentário "a MLP também deveria ter comportamento determinístico"): a ordem dos nós no grafo muda com a topologia (os nós entram pelas arestas). Com isso, a média dos atributos é somada em ordens diferentes e muda nas últimas casas decimais, o que basta para alterar o treino do MLP. No código novo, os nós são ordenados por `nflId`, e RF/MLP rodam uma vez por semente, o que os torna determinísticos.
5. `output/results` tem **29 sementes** (1–29) por topologia, mas o texto fala em 30 execuções. As instruções usam as sementes 1–29 para casar com essas execuções. Na análise com `--old-gcn-dir`, os testes estatísticos usam só as sementes presentes em todos os modelos.
6. (Observação) `position`, `club`, `possessionTeam`, `offenseFormation` e `receiverAlignment` são codificados com `LabelEncoder` **semana a semana**. Se o conjunto de categorias mudar entre semanas, o mesmo código pode representar coisas diferentes. Isso afeta todos os modelos igualmente, mas é bom ter em mente.
7. Na GPU, a GCN (agregação com operações atômicas no `GCNConv`) pode não ser bit a bit determinística. Isso só importa se você decidir rodar a GCN de novo (Apêndice A).

## Arquivos novos ou alterados

* `experimentos_banca.py`: ponto de entrada (`build`, `tune`, `run`, `analyze`)
* `config_comparacao_justa.json`: configuração usada nos resultados do texto (500 épocas, paciência 100, etc.)
* `fair_params.json`: hiperparâmetros das baselines (`default` = texto; `tuned` = preenchido pelo passo 2)
* `src/experiments/common.py`: cache de grafos, divisão, execução dos modelos
* `src/experiments/tuning.py`, `runner.py`, `analysis.py`
* `src/models/deepsets.py`: modelo DeepSets
* `src/models/set_features.py`: representações vetoriais das baselines
* `src/models/gcn_trainer.py`: novo `train_on_split` (aceita uma divisão pronta e outro modelo). `train_model` tem o mesmo comportamento de antes
* `src/data/preprocessor.py` e `graph_builder.py`: atributos auxiliares `isOffense` e `positionName` nos nós e `gameId`/`playId` no grafo. **Não entram nas features da GCN** (validado: tensores idênticos aos do pipeline original)

---

## Apêndice A: executar a GCN (opcional)

**Não é necessário para os experimentos da banca.** A análise usa as execuções da GCN que já estão em `output/results`. Este apêndice serve para rodar a GCN de novo com o código desta branch, por exemplo para conferir a reprodutibilidade ou gerar as predições de cada jogada, que os JSONs antigos não guardam e podem ser úteis para a análise de erros pedida pelo Prof. Jadson.

A GCN usa a mesma configuração do texto (`config_comparacao_justa.json`: 1 camada, 256 canais, AdamW, lr 0,001, 500 épocas, paciência 100) e a mesma divisão de cada semente.

### Em que momento executar

| Etapa | Quando |
|---|---|
| A.1 Cache das outras topologias | **Logo depois do Passo 1.** Pode rodar em paralelo com o Passo 2 (usa só a CPU) |
| A.2 Treino da GCN | **Depois do Passo 1 e de A.1, e antes do Passo 4 (análise).** Não depende do Passo 2 (a GCN não usa os hiperparâmetros otimizados das baselines). Como usa a GPU, o ideal é rodar **depois do Passo 3** ou intercalado com ele, para não disputar a GPU com o DeepSets |
| A.3 Análise | No lugar do Passo 4 (ou repetindo o Passo 4) |

### A.1 Construir o cache das demais topologias

```bash
python experimentos_banca.py build --strategies RNG CLOSEST- GABRIEL QB-CLOSEST- DELAUNAY
```

O pré-processamento de cada semana é feito uma vez só e reaproveitado pelas 5 topologias (aprox. 30 min no total). Se for rodar só MST e RNG, use `--strategies RNG`.

### A.2 Treinar a GCN

Para as duas melhores topologias do texto:

```bash
python experimentos_banca.py run --seeds 1-29 --models gcn --strategies MST RNG
```

Para as 6 topologias:

```bash
python experimentos_banca.py run --seeds 1-29 --models gcn --strategies MST RNG CLOSEST- GABRIEL QB-CLOSEST- DELAUNAY
```

* Os resultados vão para `output/comparacao_justa/runs/GCN-<topologia>/seed_XXX.json`, na mesma pasta das baselines. Como no Passo 3, execuções que já existem são puladas, então dá para interromper e retomar.
* Cada execução leva alguns minutos na GPU (até 500 épocas). Confira o `train_time_s` da primeira semente para estimar o total.

### A.3 Análise com a GCN executada de novo

```bash
python experimentos_banca.py analyze --old-gcn-dir output/results
```

* A tabela de resumo mostra lado a lado `GCN-MST` (executada de novo) e `GCN-MST (texto)` (execuções antigas), o que permite conferir se os números reproduzem os do texto. Pequenas diferenças são esperadas por causa do não determinismo da GPU (item 7 da seção de problemas).
* Nos testes estatísticos, quando uma topologia foi executada de novo, a versão "(texto)" sai do Friedman/Nemenyi para não contar a mesma configuração duas vezes. As topologias que não foram executadas de novo continuam entrando pela versão "(texto)".
* Para usar só as execuções novas, rode sem `--old-gcn-dir`.
