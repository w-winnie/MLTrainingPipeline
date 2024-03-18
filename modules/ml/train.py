# code for training and tuning model
from modules.ml.model.model import init_RandomForestRegressor
from modules.ml.tuning.tuning import tune_random_forest_gridsearch
import pandas as pd

def train_and_tune_model(X_train: pd.DataFrame, y_train: pd.DataFrame, model_config: dict):
    random_forest_model = init_RandomForestRegressor(model_config=model_config)
    # random_forest_model.fit(X_train, y_train)
    best_model = tune_random_forest_gridsearch(random_forest_model, features=X_train, labels=y_train)
    return best_model