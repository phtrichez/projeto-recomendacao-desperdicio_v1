import argparse
import ast
from pathlib import Path

import pandas as pd

from eda import ARQUIVO_INTERACOES, ARQUIVO_RECEITAS, ARQUIVO_ZIP, ler_csv_do_zip
from collaborative_svd import CollaborativeSVDRecommender
from content_based import ContentBasedRecommender
from data_loader import carregar_receitas, dividir_treino_teste_por_usuario, preparar_amostra_modelagem
from hybrid_recommender import HybridRecommender


def transformar_texto_em_lista(texto):
    """Transforma a coluna ingredients, que vem como texto, em uma lista Python."""
    try:
        ingredientes = ast.literal_eval(texto)
        return [ingrediente.lower().strip() for ingrediente in ingredientes]
    except (ValueError, SyntaxError):
        return []


def carregar_receitas_e_medias(caminho_zip):
    receitas = ler_csv_do_zip(
        caminho_zip,
        ARQUIVO_RECEITAS,
        colunas=["id", "name", "ingredients"],
    )
    receitas = receitas.rename(columns={"id": "recipe_id", "name": "recipe_name"})
    receitas["ingredients_list"] = receitas["ingredients"].apply(transformar_texto_em_lista)

    interacoes = ler_csv_do_zip(
        caminho_zip,
        ARQUIVO_INTERACOES,
        colunas=["recipe_id", "rating"],
    )
    interacoes["rating"] = pd.to_numeric(interacoes["rating"], errors="coerce")
    interacoes = interacoes.dropna(subset=["rating"])

    medias = (
        interacoes.groupby("recipe_id")["rating"]
        .agg(media_rating="mean", quantidade_avaliacoes="count")
        .reset_index()
    )

    return receitas.merge(medias, on="recipe_id", how="left")


def recomendar_por_ingredientes(receitas, ingredientes_disponiveis, urgencia, top_n=10, min_avaliacoes=20):
    """Recomenda receitas usando ingredientes disponiveis e urgencia simulada de validade."""
    ingredientes_disponiveis = [item.lower().strip() for item in ingredientes_disponiveis]
    receitas = receitas.copy()
    receitas["media_rating"] = receitas["media_rating"].fillna(receitas["media_rating"].mean())
    receitas["quantidade_avaliacoes"] = receitas["quantidade_avaliacoes"].fillna(0).astype(int)
    receitas = receitas[receitas["quantidade_avaliacoes"] >= min_avaliacoes]

    recomendacoes = []
    for receita in receitas.itertuples():
        ingredientes_receita = receita.ingredients_list

        ingredientes_encontrados = []
        for ingrediente in ingredientes_disponiveis:
            for ingrediente_receita in ingredientes_receita:
                if ingrediente in ingrediente_receita:
                    ingredientes_encontrados.append(ingrediente)
                    break

        ingredientes_encontrados = sorted(set(ingredientes_encontrados))
        if len(ingredientes_encontrados) < 2:
            continue

        score_ingredientes = len(ingredientes_encontrados) / len(ingredientes_disponiveis)
        score_rating = receita.media_rating / 5

        soma_urgencia = sum(urgencia.get(ingrediente, 1) for ingrediente in ingredientes_encontrados)
        urgencia_maxima = sum(urgencia.get(ingrediente, 1) for ingrediente in ingredientes_disponiveis)
        score_urgencia = soma_urgencia / urgencia_maxima

        score_final = (0.55 * score_ingredientes) + (0.25 * score_rating) + (0.20 * score_urgencia)

        recomendacoes.append(
            {
                "recipe_id": receita.recipe_id,
                "recipe_name": receita.recipe_name,
                "matched_ingredients": ", ".join(ingredientes_encontrados),
                "matched_count": len(ingredientes_encontrados),
                "available_count": len(ingredientes_disponiveis),
                "urgency_sum": soma_urgencia,
                "ingredient_score": score_ingredientes,
                "urgency_score": score_urgencia,
                "media_rating": receita.media_rating,
                "quantidade_avaliacoes": receita.quantidade_avaliacoes,
                "final_score": score_final,
            }
        )

    recomendacoes = pd.DataFrame(recomendacoes)
    recomendacoes = recomendacoes.sort_values(
        ["final_score", "matched_count", "media_rating", "quantidade_avaliacoes"],
        ascending=False,
    )

    return recomendacoes.head(top_n)


def run_recommendation_example(data_path=ARQUIVO_ZIP, results_dir="results"):
    # A base Food.com nao possui validade real dos alimentos do usuario.
    # A urgencia abaixo e uma entrada simulada: 5 significa mais perto do vencimento.
    ingredientes = ["banana", "milk", "eggs", "flour"]
    urgencia = {"banana": 5, "milk": 4, "eggs": 3, "flour": 2}

    receitas = carregar_receitas_e_medias(data_path)
    recomendacoes = recomendar_por_ingredientes(receitas, ingredientes, urgencia, top_n=10, min_avaliacoes=20)

    pasta_resultados = Path(results_dir)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    recomendacoes.to_csv(pasta_resultados / "recomendacoes_exemplo.csv", index=False)

    return recomendacoes


def run_hybrid_recommendation_example(data_path=ARQUIVO_ZIP, results_dir="results", user_id=None, ingredientes_urgencia=None):
    if ingredientes_urgencia is None:
        ingredientes_urgencia = {"banana": 5, "milk": 4, "eggs": 3, "flour": 2}

    amostra = preparar_amostra_modelagem(data_path, results_dir=results_dir, random_state=42)
    receitas = carregar_receitas(data_path, recipe_ids=amostra["recipe_id"].unique())
    treino, _ = dividir_treino_teste_por_usuario(amostra, random_state=42)

    if user_id is None or user_id not in set(treino["user_id"]):
        user_id = int(treino["user_id"].iloc[0])

    modelo_svd = CollaborativeSVDRecommender(n_components=50, random_state=42).fit(treino)
    modelo_conteudo = ContentBasedRecommender(min_rating_perfil=4).fit(treino, receitas)
    modelo_hibrido = HybridRecommender(
        modelo_svd,
        modelo_conteudo,
        receitas,
        ingredientes_urgencia,
        alpha_colaborativo=0.45,
        beta_conteudo=0.35,
        gamma_validade=0.20,
    )

    recomendacoes = modelo_hibrido.recomendar_usuario(user_id, k=10)
    recomendacoes.insert(0, "user_id", user_id)

    pasta_resultados = Path(results_dir)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    recomendacoes.to_csv(pasta_resultados / "recomendacoes_hibridas_exemplo.csv", index=False)

    return recomendacoes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ARQUIVO_ZIP))
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_recommendation_example(args.data_path, args.results_dir)
