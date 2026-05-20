# Texto para Documento Final - Modulo 4

## Resultados

O projeto desenvolveu uma prova de conceito funcional de um sistema de recomendacao de receitas alinhado ao ODS 12, com foco na reducao do desperdicio alimentar. A solucao foi implementada sobre a base Food.com Recipes and Interactions, utilizando principalmente os arquivos `RAW_recipes.csv` e `RAW_interactions.csv`.

Na etapa de analise exploratoria, foram identificadas 1.132.367 interacoes, 226.570 usuarios unicos e 231.637 receitas unicas nas interacoes. O arquivo de receitas tambem possui 231.637 registros. A media das avaliacoes foi de 4,4110 e a mediana foi 5,0, indicando forte concentracao em ratings altos. O periodo observado vai de 2000-01-25 a 2018-12-20, cobrindo 19 anos. A esparsidade aproximada da matriz usuario-item foi de 0,9999784237.

Para avaliacao dos modelos de ranking, foi preparada uma amostra filtrada da base por restricao computacional. A amostra removeu ratings nulos, manteve usuarios com pelo menos 5 interacoes e receitas com pelo menos 10 avaliacoes, usando `random_state=42`. Com os limites configurados de ate 800 usuarios e 1.500 receitas, a amostra final usada na avaliacao teve 1.512 interacoes, 395 usuarios e 97 receitas.

## Metricas de Avaliacao

Foram utilizadas duas familias de metricas. Para os baselines de predicao de rating, foram calculados RMSE e MAE. Para os modelos de ranking, foram calculadas Precision@10, Recall@10, F1@10, NDCG@10 e HitRate@10.

Na avaliacao de ranking, as interacoes foram divididas por usuario em treino e teste. Foram considerados relevantes no teste os itens com `rating >= 4`. Para cada usuario com pelo menos um item relevante no teste, os modelos geraram uma lista top-10 de recomendacoes, excluindo itens ja vistos no treino. As metricas foram calculadas comparando os itens recomendados com os itens relevantes do teste.

## Comparacao com Baselines

Nos baselines de predicao de rating, a media do usuario com fallback para a media global apresentou o melhor desempenho, com RMSE de 1,2631 e MAE de 0,7240. A media global obteve RMSE de 1,2723 e MAE de 0,8523. A media da receita com fallback global obteve RMSE de 1,3488 e MAE de 0,8252.

Na avaliacao top-k, foram comparados quatro modelos: popularidade, conteudo TF-IDF, colaborativo SVD e hibrido. Os resultados foram:

| Modelo | Precision@10 | Recall@10 | F1@10 | NDCG@10 | HitRate@10 |
|---|---:|---:|---:|---:|---:|
| Popularidade | 0,0269 | 0,2285 | 0,0471 | 0,1234 | 0,2473 |
| Conteudo TF-IDF | 0,0177 | 0,1456 | 0,0308 | 0,0723 | 0,1658 |
| Colaborativo SVD | 0,0095 | 0,0892 | 0,0171 | 0,0387 | 0,0951 |
| Hibrido | 0,0163 | 0,1458 | 0,0289 | 0,0756 | 0,1630 |

O baseline de popularidade apresentou o melhor desempenho nas metricas de ranking da amostra avaliada. Esse resultado indica que, para a amostra e configuracao utilizadas, recomendar receitas populares foi mais efetivo para recuperar itens relevantes no teste. O modelo hibrido nao superou a popularidade nas metricas principais, mas incorporou elementos adicionais alinhados ao objetivo do projeto: personalizacao colaborativa, similaridade de conteudo por ingredientes e urgencia simulada de validade.

## Graficos e Visualizacoes

Foram gerados graficos para apoiar a interpretacao dos resultados. A EDA inclui graficos de distribuicao das avaliacoes, top 10 receitas mais avaliadas, interacoes por ano e distribuicao de interacoes por usuario. Para os modelos, foram gerados graficos comparando RMSE e MAE dos baselines, alem de comparacoes entre Precision@10, Recall@10, NDCG@10 e demais metricas de ranking.

Os principais arquivos de visualizacao sao:

- `results/graficos/distribuicao_avaliacoes.png`;
- `results/graficos/comparacao_rmse_mae.png`;
- `results/graficos/comparacao_precision_recall_ndcg.png`;
- `results/graficos/comparacao_modelos_ranking.png`.

## Discussao dos Resultados

A EDA mostrou que a base Food.com possui grande volume de interacoes e receitas, mas tambem apresenta alta esparsidade usuario-item. Esse comportamento e comum em sistemas de recomendacao, pois cada usuario interage com apenas uma pequena parcela dos itens disponiveis. A concentracao de ratings altos tambem influencia a avaliacao, pois muitos itens relevantes no teste possuem avaliacoes positivas.

O desempenho superior da popularidade no ranking sugere que a amostra avaliada favorece receitas com maior numero de interacoes historicas. Esse resultado nao invalida os modelos personalizados, mas evidencia uma limitacao pratica: em bases com forte popularidade concentrada, modelos simples podem ser competitivos. O SVD colaborativo teve desempenho inferior na amostra usada, possivelmente por causa da esparsidade e do tamanho reduzido da matriz apos os filtros.

O modelo baseado em conteudo com TF-IDF apresentou desempenho intermediario e tem a vantagem de usar ingredientes como informacao interpretavel. O modelo hibrido combinou SVD, conteudo e validade simulada, mas sua principal contribuicao esta no alinhamento ao problema do desperdicio alimentar: ele permite priorizar receitas que aproveitam ingredientes urgentes informados pelo usuario.

E importante destacar que a base Food.com nao contem validade real dos alimentos do usuario. Portanto, o fator validade foi tratado como entrada simulada/informada, e nao como dado observado na base.

## Conclusoes e Trabalhos Futuros

O projeto atingiu o objetivo de construir uma prova de conceito funcional de recomendacao de receitas alinhada ao ODS 12. Foram implementadas analise exploratoria, baselines de rating, filtragem colaborativa com SVD, filtragem baseada em conteudo com TF-IDF, modelo hibrido com fator de validade e avaliacao com metricas de ranking.

Como conclusao, a abordagem demonstra viabilidade tecnica, mas tambem mostra que modelos mais complexos nem sempre superam baselines simples em amostras pequenas e esparsas. A popularidade foi o modelo mais forte nas metricas top-k avaliadas, enquanto o modelo hibrido oferece maior aderencia conceitual ao problema de reduzir desperdicio por meio do aproveitamento de ingredientes disponiveis.

Como trabalhos futuros, recomenda-se: coletar dados reais de despensa e validade informados pelos usuarios; melhorar a normalizacao de ingredientes; testar amostras maiores e estrategias de validacao mais robustas; ajustar pesos do modelo hibrido; incluir diversidade e novidade nas recomendacoes; avaliar a solucao com usuarios reais; e explorar modelos avancados de recomendacao, como fatoracao matricial otimizada, KNN item-item, modelos neurais ou re-ranking orientado a sustentabilidade.

## Referencias

RICCI, F.; ROKACH, L.; SHAPIRA, B. Introduction to Recommender Systems Handbook. In: RICCI, F.; ROKACH, L.; SHAPIRA, B.; KANTOR, P. (org.). Recommender Systems Handbook. Boston: Springer, 2011.

SCIKIT-LEARN. Machine Learning in Python. Disponivel em: https://scikit-learn.org/.

FOOD.COM Recipes and Interactions Dataset. Base de receitas e interacoes utilizada para a prova de conceito.

## Apendices

Link para o GitHub: inserir o link do repositorio final.

Link para o video de apresentacao: inserir o link do video gravado pelo grupo.

Link para o dataset: inserir o link da fonte da base Food.com utilizada no projeto.

Principais artefatos gerados:

- `README.md`;
- `src/`;
- `notebooks/`;
- `results/metricas_baseline.csv`;
- `results/metricas_ranking_modelos.csv`;
- `results/recomendacoes_hibridas_exemplo.csv`;
- `results/graficos/`.
