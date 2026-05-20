from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import run_hybrid_recommendation_example


st.set_page_config(page_title="Recomendador ODS 12", layout="wide")

st.title("Recomendador de Receitas para Reducao do Desperdicio")
st.caption("Prova de conceito academica com Food.com, SVD, TF-IDF e urgencia simulada de validade.")

user_id = st.text_input("user_id", value="")
ingredientes = st.text_input("Ingredientes disponiveis", value="banana, milk, eggs, flour")
urgencias = st.text_input("Urgencia por ingrediente", value="banana:5, milk:4, eggs:3, flour:2")

st.info(
    "A base Food.com nao possui validade real dos alimentos. "
    "A urgencia e uma entrada simulada/informada pelo usuario."
)


def parse_user_id(valor):
    try:
        return int(valor)
    except ValueError:
        return None


def parse_urgencias(texto_ingredientes, texto_urgencias):
    ingredientes_base = [item.strip().lower() for item in texto_ingredientes.split(",") if item.strip()]
    urgencias_parseadas = {item: 1 for item in ingredientes_base}

    for parte in texto_urgencias.split(","):
        if ":" not in parte:
            continue
        ingrediente, valor = parte.split(":", 1)
        ingrediente = ingrediente.strip().lower()
        try:
            urgencias_parseadas[ingrediente] = max(1, min(5, int(valor.strip())))
        except ValueError:
            urgencias_parseadas[ingrediente] = 1

    return urgencias_parseadas


if st.button("Recomendar"):
    with st.spinner("Gerando recomendacoes..."):
        recomendacoes = run_hybrid_recommendation_example(
            user_id=parse_user_id(user_id),
            ingredientes_urgencia=parse_urgencias(ingredientes, urgencias),
        )
    st.dataframe(
        recomendacoes[
            [
                "recipe_id",
                "recipe_name",
                "score_final",
                "score_colaborativo_norm",
                "score_conteudo_norm",
                "score_validade_norm",
                "matched_ingredients",
                "justificativa",
            ]
        ],
        use_container_width=True,
    )
else:
    caminho_exemplo = Path("results/recomendacoes_hibridas_exemplo.csv")
    if caminho_exemplo.exists():
        st.subheader("Ultimas recomendacoes hibridas geradas")
        st.dataframe(pd.read_csv(caminho_exemplo), use_container_width=True)
