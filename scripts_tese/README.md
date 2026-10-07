# Figuras e tabelas da versão final da dissertação

- `gerar_figuras_tese.py`: lê `output/results` (ignora `first_tests`) e gera em `output/tese_final`:
  - `tab_resultados_f1.tex` (nova Tabela 4.1), `tab_metricas_por_classe.tex`, `tab_espaco_busca.tex` (Tabela 3.3)
  - `fig_boxplot_macro_f1.pdf/.png`, `fig_matrizes_confusao.pdf/.png`
  - `resumo_estatisticas.csv`
  - Rodar na raiz do repositório: `python scripts_tese/gerar_figuras_tese.py`
- `diagramas/*.tex`: figuras em TikZ (standalone). Compilar com `pdflatex arquivo.tex`.
  Os PDFs já compilados estão em `output/tese_final`.
  - `fig_campo_snap`: esquema do campo no momento do snap (Seção 2.1.1)
  - `fig_fluxo_metodo`: visão geral do método (início do Capítulo 3)
  - `fig_entradas_modelos`: entrada da GCN e das baselines (Seções 3.4 e 3.5)
