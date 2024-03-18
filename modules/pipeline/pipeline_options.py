import argparse

def parse_options():
    parser = argparse.ArgumentParser(description='Parse CLI options for the pipeline.')
    parser.add_argument('--input_data_path', type=str, help='GCS input data path')
    parser.add_argument('--model_path', type=str, help='GCS model path')
    parser.add_argument('--output_data_path', type=str, help='GCS output data path')
    parser.add_argument('--run_env', type=str, help='local,gcs')
    args = parser.parse_args()
    return args