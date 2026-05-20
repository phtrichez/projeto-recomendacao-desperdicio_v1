# Sistema Inteligente de Recomendacao para Reducao do Desperdicio Alimentar

Prova de conceito funcional de um sistema de recomendacao de receitas alinhado ao ODS 12. O objetivo e apoiar o aproveitamento de ingredientes disponiveis e reduzir desperdicio alimentar por meio de recomendacoes baseadas em dados da Food.com.

## Escopo Implementado

O projeto implementa:

- EDA da base Food.com Recipes and Interactions;
- baselines de predicao de rating com RMSE e MAE;
- baseline de popularidade para ranking;
- filtragem colaborativa com SVD usando `scipy.sparse.csr_matrix` e `sklearn.decomposition.TruncatedSVD`;
- filtragem baseada em conteudo com ingredientes e `TfidfVectorizer`;
- modelo hibrido combinando score colaborativo, score de conteudo e score de validade;
- avaliacao top-k com Precision@K, Recall@K, F1@K, NDCG@K e HitRate@K;
- aplicacao Streamlit simples para demonstracao.

## Base Utilizada

Foi utilizada a base **Food.com Recipes and Interactions**, disponivel no arquivo local `archive.zip`. Os principais arquivos usados foram:

- `RAW_recipes.csv`;
- `RAW_interactions.csv`.

O arquivo `RAW_recipes.csv` fornece as receitas e ingredientes. O arquivo `RAW_interactions.csv` fornece interacoes entre usuarios e receitas, incluindo ratings.

## Validade dos Alimentos

A base Food.com nao possui informacoes reais sobre estoque domestico, data de compra ou validade dos alimentos do usuario. Portanto, o fator de validade/urgencia foi tratado como uma **entrada simulada ou informada pelo usuario**.

Exemplo usado no codigo:

```python
ingredientes_urgencia = {
    "banana": 5,
    "milk": 4,
    "eggs": 3,
    "flour": 2,
}
```

Nessa escala, `5` representa maior urgencia de consumo.

## Resultados da EDA

Resultados calculados em `results/resumo_eda.csv`:

| Metrica | Valor |
|---|---:|
| Numero de interacoes | 1.132.367 |
| Usuarios unicos | 226.570 |
| Receitas unicas nas interacoes | 231.637 |
| Total de receitas no arquivo | 231.637 |
| Media das avaliacoes | 4,4110 |
| Mediana das avaliacoes | 5,0 |
| Esparsidade aproximada usuario-item | 0,9999784237 |
| Data inicial | 2000-01-25 |
| Data final | 2018-12-20 |
| Periodo com dados | 19 anos |

## Baselines de Predicao de Rating

Resultados calculados em `results/metricas_baseline.csv`:

| Modelo | RMSE | MAE |
|---|---:|---:|
| Media do usuario com fallback global | 1,2631 | 0,7240 |
| Media global | 1,2723 | 0,8523 |
| Media da receita com fallback global | 1,3488 | 0,8252 |

Esses baselines foram avaliados com divisao treino/teste 80/20 e `random_state=42`.

## Amostra de Modelagem para Ranking

Para viabilizar a avaliacao local dos modelos de ranking, foi usada uma amostra filtrada da base. As informacoes foram salvas em `results/resumo_amostra_modelagem.csv`.

| Metrica | Valor |
|---|---:|
| Random state | 42 |
| Interacoes originais | 1.132.367 |
| Minimo de interacoes por usuario | 5 |
| Minimo de avaliacoes por receita | 10 |
| Maximo de usuarios configurado | 800 |
| Maximo de receitas configurado | 1.500 |
| Interacoes na amostra | 1.512 |
| Usuarios na amostra | 395 |
| Receitas na amostra | 97 |
| Media de rating da amostra | 4,5866 |

## Avaliacao Top-K

A avaliacao top-k considera relevantes no teste as receitas com `rating >= 4`. Para cada usuario com itens relevantes no teste, foram geradas recomendacoes top-10 excluindo itens ja vistos no treino.

Resultados calculados em `results/metricas_ranking_modelos.csv`:

| Modelo | Precision@10 | Recall@10 | F1@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|---:|
| Popularidade | 0,0269 | 0,2285 | 0,0471 | 0,1234 | 0,2473 |
| Conteudo TF-IDF | 0,0177 | 0,1456 | 0,0308 | 0,0723 | 0,1658 |
| Colaborativo SVD | 0,0095 | 0,0892 | 0,0171 | 0,0387 | 0,0951 |
| Hibrido | 0,0163 | 0,1458 | 0,0289 | 0,0756 | 0,1630 |

Na amostra avaliada, o baseline de popularidade apresentou os maiores valores de ranking. Esse resultado e plausivel em uma base com ratings muito positivos e forte concentracao de interacoes em receitas populares. O modelo hibrido, por sua vez, combina personalizacao, conteudo e validade simulada, sendo mais alinhado ao objetivo do projeto mesmo quando nao supera o baseline em todas as metricas.

## Como Executar

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Executar EDA:

```powershell
python src/eda.py --data-path "C:\Mackenzie Ciencia Dados\4 Semestre\Projeto aplicado 3\archive.zip" --results-dir results
```

Executar avaliacao de baselines, SVD, TF-IDF, hibrido e ranking:

```powershell
python src/evaluation.py --data-path "C:\Mackenzie Ciencia Dados\4 Semestre\Projeto aplicado 3\archive.zip" --results-dir results --k 10 --max-usuarios 800 --max-receitas 1500
```

Executar recomendador simples por ingredientes:

```powershell
python src/recommender.py --data-path "C:\Mackenzie Ciencia Dados\4 Semestre\Projeto aplicado 3\archive.zip" --results-dir results
```

Executar Streamlit:

```powershell
streamlit run src/app_streamlit.py
```

## Saidas Geradas

- `results/resumo_eda.csv`
- `results/resumo_amostra_modelagem.csv`
- `results/metricas_baseline.csv`
- `results/metricas_ranking_modelos.csv`
- `results/recomendacoes_exemplo.csv`
- `results/recomendacoes_hibridas_exemplo.csv`
- `results/graficos/distribuicao_avaliacoes.png`
- `results/graficos/comparacao_rmse_mae.png`
- `results/graficos/comparacao_precision_recall_ndcg.png`
- `results/graficos/comparacao_modelos_ranking.png`

## Estrutura

```text
projeto-recomendacao-desperdicio/
+-- README.md
+-- requirements.txt
+-- src/
|   +-- data_loader.py
|   +-- eda.py
|   +-- evaluation.py
|   +-- baseline_models.py
|   +-- collaborative_svd.py
|   +-- content_based.py
|   +-- hybrid_recommender.py
|   +-- recommender.py
|   +-- app_streamlit.py
+-- results/
|   +-- resumo_eda.csv
|   +-- metricas_baseline.csv
|   +-- metricas_ranking_modelos.csv
|   +-- recomendacoes_exemplo.csv
|   +-- recomendacoes_hibridas_exemplo.csv
|   +-- graficos/
+-- notebooks/
+-- docs/
```

## Limitacoes

- A avaliacao de ranking foi feita em amostra filtrada por restricao computacional.
- A validade dos alimentos e simulada/informada pelo usuario, pois nao existe na base Food.com.
- O modelo baseado em conteudo usa ingredientes em texto e TF-IDF, sem entendimento semantico profundo.
- O SVD foi treinado em matriz usuario-receita esparsa da amostra, nao no dataset completo.
- Nao foi realizada avaliacao com usuarios reais, teste A/B ou coleta de feedback qualitativo.
