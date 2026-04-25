import pandas as pd
from model import Recommender

def train_model():
    data = pd.read_csv('../data/dataset.csv')
    
    model = Recommender(data)
    model.preprocess()
    model.compute_similarity()
    
    return model
