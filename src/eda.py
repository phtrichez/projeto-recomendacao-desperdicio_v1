import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARQUIVO_ZIP = Path(r"C:\Mackenzie Ciencia Dados\4 Semestre\Projeto aplicado 3\archive.zip")
ARQUIVO_RECEITAS = "RAW_recipes.csv"
ARQUIVO_INTERACOES = "RAW_interactions.csv"


def ler_csv_do_zip(caminho_zip, nome_csv, colunas=None, datas=None):
    """Le um arquivo CSV que esta dentro do archive.zip."""
    caminho_zip = Path(caminho_zip)

    if not caminho_zip.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_zip}")

    with zipfile.ZipFile(caminho_zip) as zip_file:
        if nome_csv not in zip_file.namelist():
            raise FileNotFoundError(f"{nome_csv} nao foi encontrado dentro do ZIP.")

        with zip_file.open(nome_csv) as arquivo:
            return pd.read_csv(arquivo, usecols=colunas, parse_dates=datas)


def carregar_dados(caminho_zip):
    """Carrega apenas as colunas necessarias para a analise."""
    interacoes = ler_csv_do_zip(
        caminho_zip,
        ARQUIVO_INTERACOES,
        colunas=["user_id", "recipe_id", "date", "rating"],
        datas=["date"],
    )

    receitas = ler_csv_do_zip(
        caminho_zip,
        ARQUIVO_RECEITAS,
        colunas=["id", "name"],
    )

    receitas = receitas.rename(columns={"id": "recipe_id", "name": "recipe_name"})
    interacoes["rating"] = pd.to_numeric(interacoes["rating"], errors="coerce")
    interacoes = interacoes.dropna(subset=["rating"])

    return interacoes, receitas


def calcular_eda(interacoes, receitas):
    """Calcula as tabelas principais da analise exploratoria."""
    total_interacoes = len(interacoes)
    usuarios_unicos = interacoes["user_id"].nunique()
    receitas_unicas = interacoes["recipe_id"].nunique()
    total_receitas = len(receitas)

    pares_usuario_receita = interacoes[["user_id", "recipe_id"]].drop_duplicates().shape[0]
    total_possivel_pares = usuarios_unicos * receitas_unicas
    esparsidade = 1 - (pares_usuario_receita / total_possivel_pares)

    resumo = pd.DataFrame(
        [
            ["numero_interacoes", total_interacoes],
            ["usuarios_unicos", usuarios_unicos],
            ["receitas_unicas_nas_interacoes", receitas_unicas],
            ["total_receitas_arquivo", total_receitas],
            ["media_avaliacoes", interacoes["rating"].mean()],
            ["mediana_avaliacoes", interacoes["rating"].median()],
            ["esparsidade_aproximada_usuario_item", esparsidade],
            ["data_inicio", interacoes["date"].min().date().isoformat()],
            ["data_fim", interacoes["date"].max().date().isoformat()],
            ["periodo_anos", interacoes["date"].dt.year.nunique()],
        ],
        columns=["metrica", "valor"],
    )

    distribuicao_avaliacoes = (
        interacoes["rating"]
        .value_counts()
        .sort_index()
        .rename_axis("rating")
        .reset_index(name="quantidade")
    )

    estatisticas_receitas = (
        interacoes.groupby("recipe_id")["rating"]
        .agg(quantidade_avaliacoes="count", media_rating="mean")
        .reset_index()
        .merge(receitas, on="recipe_id", how="left")
    )

    top_mais_avaliadas = estatisticas_receitas.sort_values(
        "quantidade_avaliacoes", ascending=False
    ).head(10)

    top_melhor_media = (
        estatisticas_receitas[estatisticas_receitas["quantidade_avaliacoes"] >= 20]
        .sort_values(["media_rating", "quantidade_avaliacoes"], ascending=False)
        .head(10)
    )

    interacoes_por_ano = (
        interacoes.assign(ano=interacoes["date"].dt.year)
        .groupby("ano")
        .size()
        .reset_index(name="quantidade_interacoes")
    )

    interacoes_por_usuario = (
        interacoes.groupby("user_id")
        .size()
        .reset_index(name="quantidade_interacoes")
    )

    return {
        "resumo": resumo,
        "distribuicao_avaliacoes": distribuicao_avaliacoes,
        "top_mais_avaliadas": top_mais_avaliadas,
        "top_melhor_media": top_melhor_media,
        "interacoes_por_ano": interacoes_por_ano,
        "interacoes_por_usuario": interacoes_por_usuario,
    }


def salvar_resultados(tabelas, pasta_resultados):
    """Salva CSVs e graficos da EDA."""
    pasta_resultados = Path(pasta_resultados)
    pasta_graficos = pasta_resultados / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    tabelas["resumo"].to_csv(pasta_resultados / "resumo_eda.csv", index=False)
    tabelas["distribuicao_avaliacoes"].to_csv(pasta_resultados / "distribuicao_avaliacoes.csv", index=False)
    tabelas["top_mais_avaliadas"].to_csv(pasta_resultados / "top_10_receitas_mais_avaliadas.csv", index=False)
    tabelas["top_melhor_media"].to_csv(pasta_resultados / "top_10_receitas_melhor_media.csv", index=False)
    tabelas["interacoes_por_ano"].to_csv(pasta_resultados / "interacoes_por_ano.csv", index=False)
    tabelas["interacoes_por_usuario"].to_csv(pasta_resultados / "interacoes_por_usuario.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")

    dist = tabelas["distribuicao_avaliacoes"]
    plt.figure(figsize=(8, 5))
    plt.bar(dist["rating"].astype(str), dist["quantidade"])
    plt.title("Distribuicao das avaliacoes")
    plt.xlabel("Rating")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "distribuicao_avaliacoes.png", dpi=160)
    plt.close()

    top = tabelas["top_mais_avaliadas"].sort_values("quantidade_avaliacoes")
    plt.figure(figsize=(10, 6))
    plt.barh(top["recipe_name"], top["quantidade_avaliacoes"])
    plt.title("Top 10 receitas mais avaliadas")
    plt.xlabel("Quantidade de avaliacoes")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "top_10_receitas_mais_avaliadas.png", dpi=160)
    plt.close()

    por_ano = tabelas["interacoes_por_ano"]
    plt.figure(figsize=(10, 5))
    plt.plot(por_ano["ano"], por_ano["quantidade_interacoes"], marker="o")
    plt.title("Quantidade de interacoes por ano")
    plt.xlabel("Ano")
    plt.ylabel("Quantidade de interacoes")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "interacoes_por_ano.png", dpi=160)
    plt.close()

    por_usuario = tabelas["interacoes_por_usuario"]
    plt.figure(figsize=(9, 5))
    plt.hist(por_usuario["quantidade_interacoes"], bins=60)
    plt.yscale("log")
    plt.title("Distribuicao de interacoes por usuario")
    plt.xlabel("Quantidade de interacoes")
    plt.ylabel("Quantidade de usuarios")
    plt.tight_layout()
    plt.savefig(pasta_graficos / "interacoes_por_usuario.png", dpi=160)
    plt.close()


def run_eda(data_path=ARQUIVO_ZIP, results_dir="results"):
    interacoes, receitas = carregar_dados(data_path)
    tabelas = calcular_eda(interacoes, receitas)
    salvar_resultados(tabelas, results_dir)
    return tabelas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(ARQUIVO_ZIP))
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    run_eda(args.data_path, args.results_dir)
