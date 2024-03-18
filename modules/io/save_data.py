import csv
from google.cloud import storage
from joblib import dump
from datetime import datetime
import pickle
from modules.config.job_config import data_schema

def save_model(model, path, run_env):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.joblib"
    if run_env == 'local':
        save_model_local(model, path, filename)
    elif run_env == 'gcs':
        save_model_gcs(model, path, filename)
    else:
        raise ValueError("Invalid run_env value. It should be either 'local' or 'gcs'.")

def save_pipeline(pipeline, path, run_env):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_prep_pipeline_{timestamp}.joblib"
    if run_env == 'local':
        save_pipeline_local(pipeline, path, filename)
    elif run_env == 'gcs':
        save_pipeline_gcs(pipeline, path, filename)
    else:
        raise ValueError("Invalid run_env value. It should be either 'local' or 'gcs'.")

def save_output_data(df, output_path, schema='eval', run_env='local'):
    columns = data_schema[schema] 
    df = df[columns] 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{schema}_data_{timestamp}.csv"
    if run_env == 'local':
        save_output_data_local(df, output_path, filename)
    elif run_env == 'gcs':
        save_output_data_gcs(df, output_path, filename)
    else:
        raise ValueError("Invalid run_env value. It should be either 'local' or 'gcs'.")

def save_output_data_local(df, path, filename):
    df.to_csv(path + filename)
    print(f"Data saved locally: {path}")
    return f"{path}/{filename}"

def save_output_data_gcs(df, path, filename):
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(path, filename)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(df.to_csv(), content_type='text/csv')
    print(f"Data saved to GCS: {path}")
    return f"{path}/{filename}"

def save_model_local(model, path, filename):
    dump(model, path + filename)
    print(f"Model saved to Local: {path}/{filename}")
    return f"{path}/{filename}"

def save_pipeline_local(pipeline, path, filename):
    dump(pipeline, path + filename)
    print(f"Pipeline saved to Local: {path}/{filename}")
    return f"{path}/{filename}"

def save_model_gcs(model, path, filename):
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(path, filename)
    bucket = storage_client.get_bucket(bucket_name)
    serialized_model = pickle.dumps(model)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(serialized_model)
    print(f"Model saved to GCS: {path}/{filename}")
    return f"{path}/{filename}"

def save_pipeline_gcs(pipeline, path, filename):
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(path, filename)
    bucket = storage_client.get_bucket(bucket_name)
    serialized_pipeline = pickle.dumps(pipeline)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(serialized_pipeline)
    print(f"Pipeline saved to GCS: {path}/{filename}")
    return f"{path}/{filename}"

def parse_gcs_path(gcs_path, filename):
    gcs_path = gcs_path.replace('gs://', '')
    parts = gcs_path.split('/')
    bucket_name = parts[0]
    blob_name = '/'.join(parts[1:])
    blob_name = f"{blob_name}/{filename}"
    print(blob_name)
    return bucket_name, blob_name

def save_mapping_to_csv(mapping, file_path):
    keys = list(mapping.keys())
    values = list(mapping.values())
    if not isinstance(values[0], (list, tuple)):
        values = [[v] for v in values]
    num_columns = len(values[0])
    mapping_data = [tuple([keys[i]] + [values[j][i] for j in range(num_columns)]) for i in range(len(keys))]
    columns = ['ID'] + [f'Value_{i+1}' for i in range(num_columns)]

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns) 
        writer.writerows(mapping_data)