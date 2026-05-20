import ast
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ARQUIVO_ZIP = Path(r"C:\Mackenzie Ciencia Dados\4 Semestre\Projeto aplicado 3\archive.zip")
RAW_RECIPES = "RAW_recipes.csv"
RAW_INTERACTIONS = "RAW_interactions.csv"
RANDOM_STATE = 42


def ler_csv_do_zip(caminho_zip, nome_csv, colunas=None, datas=None):
    caminho_zip = Path(caminho_zip)
    if not caminho_zip.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_zip}")

    with zipfile.ZipFile(caminho_zip) as zip_file:
        if nome_csv not in zip_file.namelist():
            raise FileNotFoundError(f"{nome_csv} nao encontrado em {caminho_zip}")
        with zip_file.open(nome_csv) as arquivo:
            return pd.read_csv(arquivo, usecols=colunas, parse_dates=datas)


def carregar_interacoes(caminho_zip=ARQUIVO_ZIP):
    interacoes = ler_csv_do_zip(
        caminho_zip,
        RAW_INTERACTIONS,
        colunas=["user_id", "recipe_id", "rating"],
    )
    interacoes["rating"] = pd.to_numeric(interacoes["rating"], errors="coerce")
    return interacoes.dropna(subset=["rating"])


def carregar_receitas(caminho_zip=ARQUIVO_ZIP, recipe_ids=None):
    receitas = ler_csv_do_zip(
        caminho_zip,
        RAW_RECIPES,
        colunas=["id", "name", "ingredients"],
    )
    receitas = receitas.rename(columns={"id": "recipe_id", "name": "recipe_name"})

    if recipe_ids is not None:
        receitas = receitas[receitas["recipe_id"].isin(recipe_ids)].copy()

    receitas["ingredients_list"] = receitas["ingredients"].apply(parse_ingredients)
    receitas["ingredients_text"] = receitas["ingredients_list"].apply(lambda itens: " ".join(itens))
    return receitas


def parse_ingredients(valor):
    try:
        ingredientes = ast.literal_eval(valor)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(ingredientes, list):
        return []
    return [str(item).lower().strip() for item in ingredientes]


def preparar_amostra_modelagem(
    caminho_zip=ARQUIVO_ZIP,
    results_dir="results",
    min_interacoes_usuario=5,
    min_avaliacoes_receita=10,
    max_usuarios=800,
    max_receitas=1500,
    random_state=RANDOM_STATE,
):
    interacoes = carregar_interacoes(caminho_zip)
    total_original = len(interacoes)

    contagem_receitas = interacoes["recipe_id"].value_counts()
    receitas_validas = contagem_receitas[contagem_receitas >= min_avaliacoes_receita].index
    interacoes = interacoes[interacoes["recipe_id"].isin(receitas_validas)]

    contagem_usuarios = interacoes["user_id"].value_counts()
    usuarios_validos = contagem_usuarios[contagem_usuarios >= min_interacoes_usuario].index.to_numpy()

    rng = np.random.default_rng(random_state)
    if max_usuarios and len(usuarios_validos) > max_usuarios:
        usuarios_validos = rng.choice(usuarios_validos, size=max_usuarios, replace=False)
    interacoes = interacoes[interacoes["user_id"].isin(usuarios_validos)]

    if max_receitas and interacoes["recipe_id"].nunique() > max_receitas:
        receitas_mais_frequentes = interacoes["recipe_id"].value_counts().head(max_receitas).index
        interacoes = interacoes[interacoes["recipe_id"].isin(receitas_mais_frequentes)]

    # Garante novamente os filtros depois dos limites de tamanho.
    contagem_usuarios = interacoes["user_id"].value_counts()
    usuarios_validos = contagem_usuarios[contagem_usuarios >= min_interacoes_usuario].index
    interacoes = interacoes[interacoes["user_id"].isin(usuarios_validos)]

    contagem_receitas = interacoes["recipe_id"].value_counts()
    receitas_validas = contagem_receitas[contagem_receitas >= min_avaliacoes_receita].index
    interacoes = interacoes[interacoes["recipe_id"].isin(receitas_validas)].copy()

    resumo = pd.DataFrame(
        [
            ["random_state", random_state],
            ["total_interacoes_original", total_original],
            ["min_interacoes_usuario", min_interacoes_usuario],
            ["min_avaliacoes_receita", min_avaliacoes_receita],
            ["max_usuarios_configurado", max_usuarios],
            ["max_receitas_configurado", max_receitas],
            ["interacoes_amostra", len(interacoes)],
            ["usuarios_amostra", interacoes["user_id"].nunique()],
            ["receitas_amostra", interacoes["recipe_id"].nunique()],
            ["media_rating_amostra", interacoes["rating"].mean()],
        ],
        columns=["metrica", "valor"],
    )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(results_path / "resumo_amostra_modelagem.csv", index=False)

    return interacoes


def dividir_treino_teste_por_usuario(interacoes, test_size=0.2, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    partes_treino = []
    partes_teste = []

    for _, grupo in interacoes.groupby("user_id"):
        grupo = grupo.sample(frac=1, random_state=random_state)
        n_teste = max(1, int(round(len(grupo) * test_size)))
        indices_teste = rng.choice(grupo.index.to_numpy(), size=n_teste, replace=False)
        teste_usuario = grupo.loc[indices_teste]
        treino_usuario = grupo.drop(index=indices_teste)
        partes_treino.append(treino_usuario)
        partes_teste.append(teste_usuario)

    treino = pd.concat(partes_treino).reset_index(drop=True)
    teste = pd.concat(partes_teste).reset_index(drop=True)
    return treino, teste
