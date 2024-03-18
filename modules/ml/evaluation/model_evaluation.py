# To put room for custom metrics rmse errors, mapes, confusion matrix, custom metrics etc.
import pandas as pd

def evaluate_random_forest(trained_model, test_features, test_label, best_model_path):
    test_score =  trained_model.score(test_features, test_label)
    eval_df = pd.DataFrame({
        'test_score': [test_score],
        'parameters': [trained_model.get_params()],
        'model_path': best_model_path
    })
    return eval_df