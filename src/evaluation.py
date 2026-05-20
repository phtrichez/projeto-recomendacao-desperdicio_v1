import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baseline_models import prever_ratings_baselines, recomendar_por_popularidade
from collaborative_svd import CollaborativeSVDRecommender
from content_based import ContentBasedRecommender
from data_loader import (
    ARQUIVO_ZIP,
    carregar_interacoes,
    carregar_receitas,
    dividir_treino_teste_por_usuario,
    preparar_amostra_modelagem,
)
from hybrid_recommender import HybridRecommender


INGREDIENTES_URGENCIA_EXEMPLO = {
    "banana": 5,
    "milk": 4,
    "eggs": 3,
    "flour": 2,
}


def calcular_rmse(valores_reais, previsoes):
    erro = valores_reais - previsoes
    return float(np.sqrt(np.mean(erro**2)))


def calcular_mae(valores_reais, previsoes):
    erro = valores_reais - previsoes
    return float(np.mean(np.abs(erro)))


def run_baseline_evaluation(data_path=ARQUIVO_ZIP, results_dir="results", test_size=0.2, random_state=42):
    interacoes = carregar_interacoes(data_path)
    treino, teste = train_test_split(interacoes, test_size=test_size, random_state=random_state)

    previsoes = prever_ratings_baselines(treino, teste)
    resultados = []
    for nome_modelo, previsao in previsoes.items():
        resultados.append(
            {
                "modelo": nome_modelo,
                "rmse": calcular_rmse(teste["rating"], previsao),
                "mae": calcular_mae(teste["rating"], previsao),
                "n_treino": len(treino),
                "n_teste": len(teste),
                "media_global_treino": treino["rating"].mean(),
                "test_size": test_size,
                "random_state": random_state,
            }
        )

    metricas = pd.DataFrame(resultados).sort_values("rmse")

    pasta_resultados = Path(results_dir)
    pasta_graficos = pasta_resultados / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(pasta_resultados / "metricas_baseline.csv", index=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(metricas))
    largura = 0.35
    plt.bar(x - largura / 2, metricas["rmse"], largura, label="RMSE")
    plt.bar(x + largura / 2, metricas["mae"], largura, label="MAE")
    plt.xticks(x, metricas["modelo"], rotation=20, ha="right")
    plt.ylabel("Erro")
    plt.title("Comparacao de RMSE e MAE dos baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pasta_graficos / "comparacao_rmse_mae.png", dpi=160)
    plt.close()

    return metricas


def metricas_topk(recomendacoes, relevantes_por_usuario, k=10):
    linhas = []
    for usuario, relevantes in relevantes_por_usuario.items():
        recomendados = recomendacoes.get(usuario, [])[:k]
        if not relevantes:
            continue

        hits = [1 if item in relevantes else 0 for item in recomendados]
        total_hits = sum(hits)

        precision = total_hits / k
        recall = total_hits / len(relevantes)
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        hit_rate = 1 if total_hits > 0 else 0

        dcg = sum(hit / np.log2(posicao + 2) for posicao, hit in enumerate(hits))
        ideal_hits = min(len(relevantes), k)
        idcg = sum(1 / np.log2(posicao + 2) for posicao in range(ideal_hits))
        ndcg = 0 if idcg == 0 else dcg / idcg

        linhas.append(
            {
                "user_id": usuario,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": f1,
                "ndcg_at_k": ndcg,
                "hitrate_at_k": hit_rate,
            }
        )

    return pd.DataFrame(linhas)


def avaliar_recomendacoes(nome_modelo, recomendacoes, relevantes_por_usuario, k=10):
    por_usuario = metricas_topk(recomendacoes, relevantes_por_usuario, k)
    medias = por_usuario.drop(columns=["user_id"]).mean().to_dict()
    medias["modelo"] = nome_modelo
    medias["k"] = k
    medias["usuarios_avaliados"] = len(por_usuario)
    return medias


def salvar_graficos_ranking(metricas, results_dir):
    pasta_graficos = Path(results_dir) / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    metricas = metricas.copy()

    colunas_principais = ["precision_at_k", "recall_at_k", "ndcg_at_k"]
    metricas.set_index("modelo")[colunas_principais].plot(kind="bar", figsize=(9, 5))
    plt.title("Comparacao de Precision, Recall e NDCG")
    plt.ylabel("Valor medio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "comparacao_precision_recall_ndcg.png", dpi=160)
    plt.close()

    colunas_todas = ["precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k", "hitrate_at_k"]
    metricas.set_index("modelo")[colunas_todas].plot(kind="bar", figsize=(11, 6))
    plt.title("Comparacao dos modelos por metricas de ranking")
    plt.ylabel("Valor medio")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "comparacao_modelos_ranking.png", dpi=160)
    plt.close()


def run_ranking_evaluation(
    data_path=ARQUIVO_ZIP,
    results_dir="results",
    k=10,
    max_usuarios=800,
    max_receitas=1500,
    random_state=42,
):
    amostra = preparar_amostra_modelagem(
        data_path,
        results_dir=results_dir,
        max_usuarios=max_usuarios,
        max_receitas=max_receitas,
        random_state=random_state,
    )
    receitas = carregar_receitas(data_path, recipe_ids=amostra["recipe_id"].unique())
    treino, teste = dividir_treino_teste_por_usuario(amostra, test_size=0.2, random_state=random_state)

    relevantes_por_usuario = (
        teste[teste["rating"] >= 4]
        .groupby("user_id")["recipe_id"]
        .apply(set)
        .to_dict()
    )
    usuarios_avaliacao = sorted(relevantes_por_usuario.keys())
    todas_receitas = sorted(amostra["recipe_id"].unique())

    resultados = []

    rec_popularidade = recomendar_por_popularidade(treino, usuarios_avaliacao, todas_receitas, k=k)
    resultados.append(avaliar_recomendacoes("popularidade", rec_popularidade, relevantes_por_usuario, k))

    modelo_conteudo = ContentBasedRecommender(min_rating_perfil=4).fit(treino, receitas)
    rec_conteudo = modelo_conteudo.recomendar_usuarios(usuarios_avaliacao, k=k)
    resultados.append(avaliar_recomendacoes("conteudo_tfidf", rec_conteudo, relevantes_por_usuario, k))

    modelo_svd = CollaborativeSVDRecommender(n_components=50, random_state=random_state).fit(treino)
    rec_svd = modelo_svd.recomendar_usuarios(usuarios_avaliacao, k=k)
    resultados.append(avaliar_recomendacoes("colaborativo_svd", rec_svd, relevantes_por_usuario, k))

    modelo_hibrido = HybridRecommender(
        modelo_svd,
        modelo_conteudo,
        receitas,
        INGREDIENTES_URGENCIA_EXEMPLO,
        alpha_colaborativo=0.45,
        beta_conteudo=0.35,
        gamma_validade=0.20,
    )
    rec_hibrido = modelo_hibrido.recomendar_usuarios(usuarios_avaliacao, k=k)
    resultados.append(avaliar_recomendacoes("hibrido", rec_hibrido, relevantes_por_usuario, k))

    metricas = pd.DataFrame(resultados)
    colunas = [
        "modelo",
        "k",
        "usuarios_avaliados",
        "precision_at_k",
        "recall_at_k",
        "f1_at_k",
        "ndcg_at_k",
        "hitrate_at_k",
    ]
    metricas = metricas[colunas]

    pasta_resultados = Path(results_dir)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(pasta_resultados / "metricas_ranking_modelos.csv", index=False)
    salvar_graficos_ranking(metricas, results_dir)

    usuario_exemplo = usuarios_avaliacao[0]
    recomendacoes_exemplo = modelo_hibrido.recomendar_usuario(usuario_exemplo, k=k)
    recomendacoes_exemplo.insert(0, "user_id", usuario_exemplo)
    recomendacoes_exemplo.to_csv(pasta_resultados / "recomendacoes_hibridas_exemplo.csv", index=False)

    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ARQUIVO_ZIP))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-usuarios", type=int, default=800)
    parser.add_argument("--max-receitas", type=int, default=1500)
    args = parser.parse_args()

    run_baseline_evaluation(args.data_path, args.results_dir, random_state=42)
    run_ranking_evaluation(
        args.data_path,
        args.results_dir,
        k=args.k,
        max_usuarios=args.max_usuarios,
        max_receitas=args.max_receitas,
        random_state=42,
    )
