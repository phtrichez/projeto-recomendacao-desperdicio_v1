import numpy as np
import pandas as pd


def estatisticas_baselines(treino):
    media_global = treino["rating"].mean()
    media_usuario = treino.groupby("user_id")["rating"].mean()
    media_receita = treino.groupby("recipe_id")["rating"].mean()
    popularidade = (
        treino.groupby("recipe_id")["rating"]
        .agg(media_rating="mean", quantidade_avaliacoes="count")
        .reset_index()
        .sort_values(["quantidade_avaliacoes", "media_rating"], ascending=False)
    )
    return media_global, media_usuario, media_receita, popularidade


def recomendar_por_popularidade(treino, usuarios, todas_receitas, k=10):
    _, _, _, popularidade = estatisticas_baselines(treino)
    ranking = popularidade["recipe_id"].tolist()
    vistos = treino.groupby("user_id")["recipe_id"].apply(set).to_dict()

    recomendacoes = {}
    for usuario in usuarios:
        itens_vistos = vistos.get(usuario, set())
        recomendacoes[usuario] = [item for item in ranking if item not in itens_vistos][:k]
    return recomendacoes


def recomendar_por_media_receita(treino, usuarios, todas_receitas, k=10):
    _, _, media_receita, popularidade = estatisticas_baselines(treino)
    fallback = popularidade.set_index("recipe_id")["quantidade_avaliacoes"]

    ranking = pd.DataFrame({"recipe_id": list(todas_receitas)})
    ranking["media_rating"] = ranking["recipe_id"].map(media_receita).fillna(media_receita.mean())
    ranking["quantidade_avaliacoes"] = ranking["recipe_id"].map(fallback).fillna(0)
    ranking = ranking.sort_values(["media_rating", "quantidade_avaliacoes"], ascending=False)
    ranking = ranking["recipe_id"].tolist()

    vistos = treino.groupby("user_id")["recipe_id"].apply(set).to_dict()
    return {u: [item for item in ranking if item not in vistos.get(u, set())][:k] for u in usuarios}


def recomendar_por_media_global(treino, usuarios, todas_receitas, k=10):
    # Como a media global da o mesmo score para todos os itens, usa popularidade para desempate.
    return recomendar_por_popularidade(treino, usuarios, todas_receitas, k)


def recomendar_por_media_usuario(treino, usuarios, todas_receitas, k=10):
    # A media do usuario tambem gera o mesmo score para todos os itens daquele usuario.
    # Por isso, a popularidade e usada apenas como desempate de ranking.
    return recomendar_por_popularidade(treino, usuarios, todas_receitas, k)


def prever_ratings_baselines(treino, teste):
    media_global, media_usuario, media_receita, _ = estatisticas_baselines(treino)
    return {
        "media_global": np.full(len(teste), media_global),
        "media_usuario_com_fallback_global": teste["user_id"].map(media_usuario).fillna(media_global).to_numpy(),
        "media_receita_com_fallback_global": teste["recipe_id"].map(media_receita).fillna(media_global).to_numpy(),
    }
