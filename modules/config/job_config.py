# job config like instance config, paths, input and output locations, types etc
instance_config = {

}

job_params = {
    'gcs_input_data_path':'',
    'local_input_data_path': '',
    'gcs_model_path':'',
    'gcs_output_data_path':''
}

# Model specific parameters
model_config = {
    'max_depth': 4, 
    'random_state': 0, 
    'n_jobs': -1
}

# idea is to have a schema object/file is to have specific sql/json schema defined for writing output data to a table or db
# because some dbs require that
data_schema = {
    'eval': [
        'test_score',
        'model_path',
        'parameters'
    ]
}

training_data_columns = [
    'dist_to_restaurant', 
    'Hdist_to_restaurant', 
    'avg_Hdist_to_restaurants',	
    'date_day_number', 
    'restaurant_id', 
    'Five_Clusters_embedding', 
    'h3_index',
    'date_hour_number',
    'restaurants_per_index'
]

training_data_labels = [
    'orders_busyness_by_h3_hour'
]

# Sometimes its a better idea to pass training hyperparameters externally as args 
# VertexAI has a tuning service as well - but dataflow doesnt.. 
training_hyperparameters = {
    'max_depth': [4,5],
    'min_samples_leaf': [50,75],
    'n_estimators': [100,150]
}