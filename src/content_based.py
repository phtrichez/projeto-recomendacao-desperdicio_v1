import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, min_rating_perfil=4):
        self.min_rating_perfil = min_rating_perfil

    def fit(self, treino, receitas):
        self.receitas = receitas.drop_duplicates("recipe_id").copy()
        self.recipe_ids = self.receitas["recipe_id"].tolist()
        self.recipe_to_idx = {recipe_id: idx for idx, recipe_id in enumerate(self.recipe_ids)}

        self.vectorizer = TfidfVectorizer(min_df=1)
        self.recipe_matrix = self.vectorizer.fit_transform(self.receitas["ingredients_text"].fillna(""))

        self.vistos = treino.groupby("user_id")["recipe_id"].apply(set).to_dict()
        self.user_profiles = {}

        treino_positivo = treino[treino["rating"] >= self.min_rating_perfil]
        for user_id, grupo in treino_positivo.groupby("user_id"):
            indices = [self.recipe_to_idx[item] for item in grupo["recipe_id"] if item in self.recipe_to_idx]
            if indices:
                perfil = self.recipe_matrix[indices].mean(axis=0)
                self.user_profiles[user_id] = np.asarray(perfil)
        return self

    def scores_usuario(self, user_id):
        if user_id not in self.user_profiles:
            return pd.Series(0.0, index=self.recipe_ids)
        similaridades = cosine_similarity(self.user_profiles[user_id], self.recipe_matrix).ravel()
        return pd.Series(similaridades, index=self.recipe_ids)

    def recomendar_usuario(self, user_id, k=10):
        scores = self.scores_usuario(user_id)
        itens_vistos = self.vistos.get(user_id, set())
        scores = scores.drop(labels=list(itens_vistos & set(scores.index)), errors="ignore")
        return scores.sort_values(ascending=False).head(k).index.tolist()

    def recomendar_usuarios(self, usuarios, k=10):
        return {usuario: self.recomendar_usuario(usuario, k) for usuario in usuarios}
