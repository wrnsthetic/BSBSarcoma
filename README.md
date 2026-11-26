# Seleção de CpGs por busca local para classificação de sarcomas

Experimento de seleção de atributos em dados de metilação de DNA para classificação de subtipos de sarcoma. O objetivo é encontrar um subconjunto mínimo de CpGs que permita classificar as 65 classes do dataset com desempenho comparável ao uso do array completo.

## Contexto

O dataset contém perfis de metilação de 1.077 amostras de sarcoma, cada uma com uma das 65 classes diagnósticas. A metilação é representada por beta-values entre 0 e 1 em ~408 mil posições (CpGs). Trabalhar com todas as posições é viável para classificação direta, mas inviável para um painel diagnóstico clínico — daí a motivação para encontrar um subconjunto pequeno e informativo.

A abordagem é: filtrar os CpGs mais variáveis, usar esses como candidatos, e aplicar Hill Climbing estocástico para refinar o subconjunto guiado pelo Macro F1 em cross-validation.

## Resultados

| Método | CpGs | Macro F1 (teste) | Accuracy (teste) |
|---|---|---|---|
| Baseline — top 50 por variância | 50 | 0.8175 | 88.9% |
| Hill Climbing estocástico | 50 | **0.9212** | **94.9%** |

O teste final usa 216 amostras (20% do dataset) que não participaram de nenhuma etapa da seleção ou treinamento.

## Como rodar

Os scripts esperam ser executados a partir da raiz do projeto. Dependências em `requirements.txt`:

```bash
pip install -r requirements.txt
```

**1. Pré-processamento**

Lê o CSV bruto em chunks (arquivo multi-GB), descarta CpGs com muitos valores faltantes, imputa pela mediana e exporta os 5.000 mais variáveis.

```bash
python scripts/preprocess_select_top_variance_cpgs.py
```

Tempo estimado: ~3 minutos. Gera `data/processed/sarcoma_top5000_variable_cpgs_imputed.csv`.

**2. Baseline**

Avalia os 50 CpGs mais variáveis com Random Forest + 5-fold CV, sem nenhuma busca. Também cria o split 80/20 que será reutilizado em todos os passos seguintes.

```bash
python scripts/run_baseline_random_forest.py
```

Tempo estimado: segundos.

**3. Busca local**

Hill Climbing estocástico. Por padrão usa 20 vizinhos por iteração e patience 20. Para continuar a partir de um subconjunto já encontrado, use `--initial-subset-path`.

```bash
python scripts/run_local_search.py
python scripts/run_local_search.py --n-neighbors 40 --patience 30 --initial-subset-path data/processed/local_search_best_subset.csv
```

O progresso é impresso a cada vizinho avaliado e o melhor subconjunto é salvo incrementalmente — pode interromper a qualquer momento com Ctrl+C sem perder o histórico.

Tempo estimado: 1–3 horas dependendo dos parâmetros.

**4. Avaliação final**

Treina nos 80% completos e avalia uma única vez nos 20% isolados. Esse é o resultado para reportar.

```bash
python scripts/evaluate_final_subset.py
```

**5. Gráficos**

Gera as figuras a partir dos relatórios JSON em `reports/` (não precisa do dataset bruto).

```bash
python scripts/plot_results.py
```

Saída em `reports/figures/`: comparação baseline vs Hill Climbing, convergência da busca, F1 por classe, métricas por fold e CpGs mais variáveis.

## Estrutura de pastas

```
all_samples_sarcoma.csv          — dataset bruto (não versionado, arquivo muito grande)
class_samples_training.csv       — rótulos das amostras
csv_do_subset/                   — classes das amostras de validação

scripts/
  utils.py                       — funções de tempo compartilhadas
  preprocess_select_top_variance_cpgs.py
  run_baseline_random_forest.py
  run_local_search.py
  evaluate_final_subset.py
  plot_results.py                — gráficos a partir de reports/
  inspect_sarcoma_dataset.py     — inspeção exploratória inicial, não faz parte do pipeline

data/
  processed/                     — outputs do pré-processamento e dos experimentos
  exports/                       — listas de CpGs para envio a coautores

reports/                         — relatórios JSON de cada etapa
  figures/                       — gráficos gerados por plot_results.py

docs/
  Fluxograma.png
  FluxogramaAtualizado.png
```

## Decisões metodológicas

**Missing values:** CpGs com 5% ou mais de faltantes são descartados (~20 no total). Os demais são imputados pela mediana do próprio CpG. A mediana é preferível à média por ser robusta a outliers em dados de beta-value.

**Seleção inicial:** os 5.000 CpGs mais variáveis entre amostras. Variância é calculada sem olhar para os rótulos, o que evita qualquer tipo de vazamento nessa etapa.

**Split treino/teste:** divisão estratificada 80/20 feita uma única vez antes de qualquer experimento e reutilizada em todos os scripts. Os 20% de teste nunca influenciam nenhuma decisão — nem seleção de CpGs nem ajuste de hiperparâmetros.

**Avaliação interna:** Stratified 5-fold CV nos 80% de treino. O número de folds é limitado ao tamanho da menor classe para evitar erros com classes raras.

**Métrica principal:** Macro F1 — trata todas as 65 classes com o mesmo peso independente do número de amostras. Accuracy é reportada como secundária.

**Busca local:** Hill Climbing estocástico. A cada iteração, sorteia K vizinhos aleatórios (cada um troca 1 CpG do subconjunto atual por um candidato fora dele), avalia todos com 5-fold CV e aceita o melhor se melhorar o F1. A vizinhança completa seria ~247.000 combinações por iteração, inviável com CV; amostrar K vizinhos é prática padrão para esse tamanho de espaço.

**Classificador:** Random Forest com 300 árvores, `class_weight=balanced_subsample` e `max_features=sqrt`. O peso balanceado é importante dado o desbalanceamento entre classes (de 7 a 59 amostras por classe).

## Referências

Artigos consultados (PDFs não incluídos no repositório):

- Cao (2022)
- Cassidy (2022)
- Develasco (2024)
- Gosh (2022)
- Guyon (2003)
- Hashmi (2024)
- Jager (2025)
- Kanwar (2022)
- Koelsche (2021)
- Lazaros (2026)
- Roohani (2022)
- Ruiztorrubiano (2025)
