# read data from sources
import pandas as pd
from google.cloud import storage
import io as io_module

def get_data(path, run_env='local'):
    df = pd.DataFrame()
    if run_env == 'local':
        df = get_data_local(path)
    elif run_env == 'gcs':
        df = get_data_gcs(path)
    return df

def get_data_local(file_path):
    df = pd.read_csv(file_path)
    return df

def get_data_gcs(file_path):
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(file_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_string()
    df = pd.read_csv(io_module.BytesIO(data))
    return df

def parse_gcs_path(gcs_path):
    gcs_path = gcs_path.replace('gs://', '')
    parts = gcs_path.split('/')
    bucket_name = parts[0]
    blob_name = '/'.join(parts[1:])
    return bucket_name, blob_name