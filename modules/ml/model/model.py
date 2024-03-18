# Could be multiple models or flavors so good to have a separate file
# Also easy to check on

from sklearn.ensemble import RandomForestRegressor
from typing import Dict

def init_RandomForestRegressor(model_config: Dict[str, int]) -> RandomForestRegressor:
    regr = RandomForestRegressor(
        max_depth=model_config['max_depth'], 
        random_state=model_config['random_state'], 
        n_jobs=model_config['n_jobs']
        )
    return regr