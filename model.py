import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

class Recommender:
    def __init__(self, data):
        self.data = data
        self.encoder = OneHotEncoder()
        self.features = None
        self.similarity_matrix = None

    def preprocess(self):
        categorical_data = self.data[['subject', 'level', 'type']]
        self.features = self.encoder.fit_transform(categorical_data).toarray()

    def compute_similarity(self):
        self.similarity_matrix = cosine_similarity(self.features)

    def recommend(self, subject, level, type_, top_n=3):
        input_df = pd.DataFrame([[subject, level, type_]],
                                columns=['subject', 'level', 'type'])

        input_vec = self.encoder.transform(input_df).toarray()
        similarity_scores = cosine_similarity(input_vec, self.features)[0]

        top_indices = similarity_scores.argsort()[-top_n:][::-1]

        return self.data.iloc[top_indices]
