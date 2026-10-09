# Contexto para o Claude (transferência de sessão)

Resumo da conversa de 08/10/2026 em que os experimentos pedidos pela banca foram implementados. **Leia este arquivo junto com o `EXPERIMENTOS_BANCA.md`**, que tem o passo a passo de execução e a descrição de cada experimento.

## Quem é o usuário e o que está fazendo

- Carlos Eduardo (Cadu), mestrando. A dissertação é sobre classificação de jogadas da NFL (passe × corrida) com GNN (GCN) sobre grafos dos 22 jogadores no snap (dados do NFL Big Data Bowl 2025, semanas 1–9). O texto está em português, em LaTeX.
- A defesa já aconteceu. A banca (Prof. Marcos Quiles e Prof. Jadson) pediu correções. As correções de texto foram feitas em outra conversa. O orientador (Vander) pediu para rodar os experimentos sugeridos pelo Prof. Marcos, para incluir na versão final e em um artigo futuro.
- Responder sempre em **português**.
- **Não fazer commit nem push sem o usuário pedir.**

## A crítica principal do Prof. Marcos

As baselines (RF e MLP) recebiam a **média** dos atributos dos 22 jogadores, enquanto a GCN recebia os jogadores individualmente. Assim, a vantagem da GCN poderia vir da perda de informação nas baselines, e não das relações entre jogadores. Ele sugeriu:
- ordenar e concatenar os jogadores;
- usar estatísticas (média, desvio, mínimo, máximo);
- usar DeepSets;
- otimizar os hiperparâmetros das baselines;
- garantir que todos os modelos recebam exatamente os mesmos dados.

## Decisões tomadas com o usuário (não mudar sem perguntar)

1. **Nada muda nos experimentos da GCN.** A GCN não é executada de novo. A análise importa as execuções antigas de `output/results` (sementes 1–29, 6 topologias) com `--old-gcn-dir`. Rodar a GCN de novo é opcional e está no Apêndice A do `EXPERIMENTOS_BANCA.md`.
2. **Só as baselines são executadas:** DeepSets (variantes `matched` e `tuned`), RF e MLP.
3. **As baselines passam a usar a mesma divisão da GCN.** No código antigo, elas usavam outra divisão, por causa de um bug descrito abaixo. O usuário aceitou que `RF-mean-default` e `MLP-mean-default` fiquem um pouco diferentes da Tabela 4.1.
4. **Comparação justa = nenhum atributo a mais em um modelo do que no outro.** Todos recebem as mesmas 13 features por jogador (`x, y, s, a, dis, o, dir, height, weight, position, club, playDirection, totalDis`) e as mesmas 10 globais. Os atributos auxiliares `isOffense` e `positionName` **não entram como coluna** em nenhum modelo. Eles servem só para ordenar ou agrupar, mas nesse caso a posição no vetor indica o lado do jogador (ataque ou defesa). Por isso, as representações estão divididas em dois grupos:
   - **Estritamente justas** (a conclusão principal se apoia nelas): `mean`, `stats`, `concat_raw` (controle negativo, ordem arbitrária) e `concat_xy` (ordenada por profundidade e lateral no sentido do ataque, usando só `x`, `y` e `playDirection`).
   - **Com informação de lado de campo** (análise complementar; deixar a ressalva explícita no texto): `stats_team`, `concat_team_y` e `concat_team_role`.
5. **DeepSets-matched** = Linear(23→256) → ReLU → pooling médio → dropout → Linear(256→2), com os mesmos hiperparâmetros e o mesmo treinador da GCN. Isso equivale a uma GCN sem arestas: é o controle direto do efeito das arestas.
6. **Tuning das baselines:** Optuna 4.7.0 com `TPESampler(seed=42)`, divisão da semente 0, objetivo = Macro F1 na **validação**, 50 trials para RF/MLP em cada representação e 30 para o DeepSets. Os resultados ficam em `optuna_baselines.db` e `fair_params.json`, que estão versionados no git.
7. **Sementes de avaliação: 1–29**, para casar com `output/results`. O texto diz "30 execuções", mas existem 29.

## Problemas encontrados no código original (já usado no texto)

1. **Divisão dupla:** em `TrainingPipeline.execute`, `split_and_prepare_data` é chamado duas vezes sobre as mesmas listas, que são embaralhadas no lugar. Com isso, RF e MLP foram avaliados em outra divisão, e não na da GCN, ao contrário do que diz a Seção 3.5 do texto. Não há viés, mas o pareamento por semente nos testes de Friedman/Nemenyi fica quebrado. No código novo, `common.make_split` reproduz exatamente a divisão da GCN (validado jogada a jogada).
2. **O "melhor modelo" da GCN é o último:** `best_model_state = model.state_dict().copy()` faz uma cópia rasa, então `best_gcn_results == last_gcn_results` em todos os JSONs. Não foi corrigido, para manter os números do texto. O DeepSets usa o mesmo treinador, então o procedimento é o mesmo para os dois.
3. **O Optuna da GCN otimizou a acurácia no TESTE** (`optuna_optimization.py`). Isso é vazamento na seleção de hiperparâmetros e favorece a GCN. Vale citar como limitação.
4. **As arestas da GCN entram sem peso e em um sentido só:**
   - O modelo é chamado como `model(data.x, data.edge_index, data.batch)`. O `edge_attr`, com as distâncias, é calculado mas não é usado.
   - `convert_nx_to_pytorch_geometric` adiciona cada aresta uma única vez (`[src, dst]`, sem a volta). Resultado: o grafo no PyG fica direcionado (`is_undirected() = False`; a MST tem 21 colunas no `edge_index` em vez de 42). Na `GCNConv`, a mensagem vai só da origem para o destino, e o sentido de cada aresta é arbitrário (ordem de inserção).
   - Não foi corrigido. Para o texto: descrever como limitação e trabalho futuro (arestas nos dois sentidos, com a distância como peso).
   - **Ainda não foi registrado no `EXPERIMENTOS_BANCA.md`.**
5. **Por que a MLP variava com a semente fixa:** a ordem dos nós muda com a topologia, e isso muda a soma de ponto flutuante da média. No código novo, os nós são ordenados por `nflId`, então RF e MLP ficam determinísticos.
6. `position`, `club`, `possessionTeam`, `offenseFormation` e `receiverAlignment` são codificados com `LabelEncoder` semana a semana, então o mesmo código pode significar categorias diferentes em semanas diferentes. Isso afeta todos os modelos igualmente.
7. Na GPU, a `GCNConv` pode não ser bit a bit determinística. Só importa se a GCN for executada de novo.

## Estado em 08/10/2026, ~23h

- `build --strategies MST`: concluído (`data/graphs/cache/MST_...pkl`, que não está no git; copiar ou rodar o `build` de novo, ~15 min).
- `tune rf:all`: **concluído** (7 representações).
- `tune mlp:all`: `mean` e `stats` concluídos; `concat_raw` em 10/50 trials; faltam `concat_xy`, `stats_team`, `concat_team_y` e `concat_team_role`.
- `tune deepsets`: não iniciado.
- `run` e `analyze`: não iniciados.
- Tempos medidos: trial de RF ≈ 5 s (`mean`/`stats`) e ≈ 43 s (concatenações). Estimativas: tuning do DeepSets ≈ 4–8 h, `run` do DeepSets ≈ 5–12 h, `run` de RF+MLP ≈ 1,5–2,5 h.
- O usuário vai continuar a execução em **outro computador**: clonar, `git checkout experimentos-comparacao-justa`, copiar `data/raw` (8,8 GB), criar o venv com `linux_requirements.txt` e depois `pip install optuna==4.7.0 scikit-posthocs==0.12.0` (esses dois não estão no requirements). Não rodar o tuning em duas máquinas ao mesmo tempo: o `optuna_baselines.db` divergiria.

## Próximos passos

1. Terminar o tuning: `tune mlp:all --trials 50` e `tune deepsets --trials 30`.
2. `run --seeds 1-29 --models deepsets` e `run --seeds 1-29 --models rf mlp` (podem rodar em paralelo).
3. `analyze --old-gcn-dir output/results`.
4. Quando o usuário trouxer os resultados, analisar `output/comparacao_justa/analise/` (`relatorio.md`, `resumo.csv`, `friedman_nemenyi.txt`, `wilcoxon_vs_ref.csv`, boxplot, diagrama CD) e `fair_params.json`. Perguntas centrais:
   - A GCN (MST/RNG) continua acima das representações **estritamente justas**, especialmente `concat_xy` e DeepSets?
   - Qual é a diferença entre GCN-MST e DeepSets-matched (o efeito das arestas)?
   - Como ficam as representações com informação de lado de campo (`concat_team_role` teve o maior Macro F1 de validação no tuning do RF: 0,816)?
   - Quais afirmações do texto precisam ser suavizadas, como a banca pediu?
5. Ajudar a escrever as seções novas no texto. Descrever o tuning com precisão para reprodutibilidade (método, versão, semente, número de trials, conjunto de validação), como o Prof. Marcos pediu.

## Arquivos relevantes

- `EXPERIMENTOS_BANCA.md`: instruções de execução e problemas encontrados.
- `experimentos_banca.py`: CLI com os comandos `build`, `tune`, `run` e `analyze`.
- `src/experiments/common.py`: cache de grafos, `make_split`, execução dos modelos.
- `src/experiments/tuning.py`, `runner.py`, `analysis.py`.
- `src/models/deepsets.py`, `src/models/set_features.py`.
- `src/models/gcn_trainer.py`: ganhou `train_on_split`; o comportamento de `train_model` não mudou.
- `config_comparacao_justa.json`: configuração usada no texto.
- `fair_params.json` e `optuna_baselines.db`: hiperparâmetros e estudos do tuning.
- Texto da dissertação (PDF): `~/Downloads/2026_BRACIS_Cadu-2.pdf` no computador original.
