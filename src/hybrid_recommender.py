import numpy as np
import pandas as pd


def normalizar_serie(serie):
    serie = serie.astype(float)
    minimo = serie.min()
    maximo = serie.max()
    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(0.0, index=serie.index)
    return (serie - minimo) / (maximo - minimo)


def calcular_score_validade(receitas, ingredientes_urgencia):
    ingredientes_urgencia = {k.lower().strip(): int(v) for k, v in ingredientes_urgencia.items()}
    soma_maxima = sum(ingredientes_urgencia.values())
    if soma_maxima == 0:
        soma_maxima = 1

    rows = []
    for receita in receitas.itertuples():
        encontrados = []
        for ingrediente, urgencia in ingredientes_urgencia.items():
            if any(ingrediente in item for item in receita.ingredients_list):
                encontrados.append(ingrediente)
        soma = sum(ingredientes_urgencia[item] for item in encontrados)
        rows.append(
            {
                "recipe_id": receita.recipe_id,
                "score_validade": soma / soma_maxima,
                "matched_ingredients": ", ".join(sorted(encontrados)),
                "justificativa": "Aproveita ingredientes urgentes: " + ", ".join(sorted(encontrados))
                if encontrados
                else "Nao encontrou ingredientes urgentes informados.",
            }
        )
    return pd.DataFrame(rows).set_index("recipe_id")


class HybridRecommender:
    def __init__(
        self,
        modelo_colaborativo,
        modelo_conteudo,
        receitas,
        ingredientes_urgencia,
        alpha_colaborativo=0.45,
        beta_conteudo=0.35,
        gamma_validade=0.20,
    ):
        self.modelo_colaborativo = modelo_colaborativo
        self.modelo_conteudo = modelo_conteudo
        self.receitas = receitas.drop_duplicates("recipe_id").copy()
        self.ingredientes_urgencia = ingredientes_urgencia
        self.alpha = alpha_colaborativo
        self.beta = beta_conteudo
        self.gamma = gamma_validade
        self.validade = calcular_score_validade(self.receitas, ingredientes_urgencia)

    def recomendar_usuario(self, user_id, k=10):
        colab = self.modelo_colaborativo.scores_usuario(user_id)
        conteudo = self.modelo_conteudo.scores_usuario(user_id)
        validade = self.validade["score_validade"]

        indices = sorted(set(colab.index) | set(conteudo.index) | set(validade.index))
        tabela = pd.DataFrame(index=indices)
        tabela["score_colaborativo"] = colab.reindex(indices).fillna(0)
        tabela["score_conteudo"] = conteudo.reindex(indices).fillna(0)
        tabela["score_validade"] = validade.reindex(indices).fillna(0)

        tabela["score_colaborativo_norm"] = normalizar_serie(tabela["score_colaborativo"])
        tabela["score_conteudo_norm"] = normalizar_serie(tabela["score_conteudo"])
        tabela["score_validade_norm"] = normalizar_serie(tabela["score_validade"])

        tabela["score_final"] = (
            self.alpha * tabela["score_colaborativo_norm"]
            + self.beta * tabela["score_conteudo_norm"]
            + self.gamma * tabela["score_validade_norm"]
        )

        vistos = self.modelo_colaborativo.vistos.get(user_id, set()) | self.modelo_conteudo.vistos.get(user_id, set())
        tabela = tabela.drop(index=list(vistos & set(tabela.index)), errors="ignore")
        tabela = tabela.sort_values("score_final", ascending=False).head(k)
        tabela = tabela.reset_index().rename(columns={"index": "recipe_id"})

        detalhes = self.receitas[["recipe_id", "recipe_name"]].merge(
            self.validade.reset_index()[["recipe_id", "matched_ingredients", "justificativa"]],
            on="recipe_id",
            how="left",
        )
        return tabela.merge(detalhes, on="recipe_id", how="left")

    def recomendar_usuarios(self, usuarios, k=10):
        recomendacoes = {}
        for usuario in usuarios:
            recomendacoes[usuario] = self.recomendar_usuario(usuario, k)["recipe_id"].tolist()
        return recomendacoes
