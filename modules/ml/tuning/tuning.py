from sklearn.model_selection import GridSearchCV
from modules.config.job_config import training_hyperparameters
import pandas as pd

def tune_random_forest_gridsearch(model, features:pd.DataFrame, labels:pd.DataFrame):
    params = training_hyperparameters
    grid_search = GridSearchCV(estimator=model,
                               param_grid=params,
                               cv = 3, n_jobs=-1, verbose=1, scoring="r2")
    grid_search.fit(features, labels)
    grid_search.best_score_
    rf_best = grid_search.best_estimator_
    return rf_best