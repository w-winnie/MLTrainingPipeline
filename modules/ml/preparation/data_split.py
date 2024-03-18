# The idea for a separate file is to accomodate for any custom splitting logic 
# like splitting by unique ids or proportinating data after splits based on combination of attributes

from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List

def prepare_train_test_set(df: pd.DataFrame, training_data_columns: List[str], training_data_labels: str,  test_size: float = 0.33) -> tuple:
    X = df[training_data_columns]
    y = df[training_data_labels]
    
    y = y.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test
