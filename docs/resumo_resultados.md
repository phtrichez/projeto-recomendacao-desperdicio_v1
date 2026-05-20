# Resumo de Resultados

A analise exploratoria da base Food.com Recipes and Interactions identificou 1.132.367 interacoes, 226.570 usuarios unicos e 231.637 receitas. A media das avaliacoes foi 4,4110 e a mediana foi 5,0. O periodo dos dados vai de 2000-01-25 a 2018-12-20, e a esparsidade aproximada da matriz usuario-item foi 0,9999784237.

Os baselines de predicao de rating foram avaliados com RMSE e MAE. O melhor resultado foi a media do usuario com fallback global, com RMSE de 1,2631 e MAE de 0,7240. A media global obteve RMSE de 1,2723 e MAE de 0,8523. A media da receita com fallback global obteve RMSE de 1,3488 e MAE de 0,8252.

Para os modelos de ranking, foi usada uma amostra filtrada por restricao computacional, com 1.512 interacoes, 395 usuarios e 97 receitas. Foram avaliados popularidade, conteudo TF-IDF, colaborativo SVD e modelo hibrido. A avaliacao usou Precision@10, Recall@10, F1@10, NDCG@10 e HitRate@10, considerando relevantes os itens do teste com rating maior ou igual a 4.

Na amostra avaliada, o modelo de popularidade apresentou os maiores valores de ranking: Precision@10 de 0,0269, Recall@10 de 0,2285, F1@10 de 0,0471, NDCG@10 de 0,1234 e HitRate@10 de 0,2473. O modelo hibrido apresentou Precision@10 de 0,0163, Recall@10 de 0,1458, F1@10 de 0,0289, NDCG@10 de 0,0756 e HitRate@10 de 0,1630.

O modelo hibrido combina score colaborativo, score de conteudo e score de validade. A validade nao existe na base Food.com e foi tratada como entrada simulada/informada pelo usuario. Assim, o projeto deve ser apresentado como prova de conceito funcional e reprodutivel, com resultados reais calculados pelo codigo e limitacoes claramente documentadas.
