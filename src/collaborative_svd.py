import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class CollaborativeSVDRecommender:
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, treino):
        self.user_ids = sorted(treino["user_id"].unique())
        self.recipe_ids = sorted(treino["recipe_id"].unique())
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(self.recipe_ids)}

        linhas = treino["user_id"].map(self.user_to_idx)
        colunas = treino["recipe_id"].map(self.recipe_to_idx)
        valores = treino["rating"].astype(float)

        matriz = csr_matrix(
            (valores, (linhas, colunas)),
            shape=(len(self.user_ids), len(self.recipe_ids)),
        )

        limite_componentes = max(1, min(matriz.shape) - 1)
        componentes = min(self.n_components, limite_componentes)

        self.modelo = TruncatedSVD(n_components=componentes, random_state=self.random_state)
        self.user_factors = self.modelo.fit_transform(matriz)
        self.item_factors = self.modelo.components_.T
        self.vistos = treino.groupby("user_id")["recipe_id"].apply(set).to_dict()
        return self

    def scores_usuario(self, user_id):
        if user_id not in self.user_to_idx:
            return pd.Series(0.0, index=self.recipe_ids)
        idx = self.user_to_idx[user_id]
        scores = self.user_factors[idx] @ self.item_factors.T
        return pd.Series(scores, index=self.recipe_ids)

    def recomendar_usuario(self, user_id, k=10):
        scores = self.scores_usuario(user_id)
        itens_vistos = self.vistos.get(user_id, set())
        scores = scores.drop(labels=list(itens_vistos & set(scores.index)), errors="ignore")
        return scores.sort_values(ascending=False).head(k).index.tolist()

    def recomendar_usuarios(self, usuarios, k=10):
        return {usuario: self.recomendar_usuario(usuario, k) for usuario in usuarios}
