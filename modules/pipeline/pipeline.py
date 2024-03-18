# Uses config, io and ml

from modules.io.read_data import get_data
from modules.io.save_data import save_model, save_pipeline, save_output_data
from modules.ml.train import train_and_tune_model
from modules.ml.preparation.data_prep_pipeline import data_prep_pipeline
from modules.ml.preparation.data_split import prepare_train_test_set
from modules.ml.evaluation.model_evaluation import evaluate_random_forest
from modules.config.job_config import training_data_labels, training_data_columns, model_config

def build_pipeline(options=dict):
    # Load Data
    df_raw = get_data(options.input_data_path, options.run_env)

    # DEBUG only: print data - health checks
    print(df_raw.head(2))

    # Train data prep pipeline
    df = data_prep_pipeline.fit_transform(df_raw)

    # Save pipeline
    save_pipeline(data_prep_pipeline, options.model_path, options.run_env)

    # DEBUG only: print data - health checks
    print(df.head(3))
    df.to_csv(options.output_data_path)

    # TODO: Handle splitting - should we split before passing it to pipeline
    #  - i.e. load train and test separately? its not like we are doing a k fold CV
    X_train, X_test, y_train, y_test = prepare_train_test_set(
        df, 
        test_size=0.33, 
        training_data_columns=training_data_columns,
        training_data_labels=training_data_labels
    )

    # Train model 
    model = train_and_tune_model(X_train, y_train, model_config)

    # Save model
    best_model_path = save_model(model, options.model_path, options.run_env)

    # Evaluate model
    eval_data = evaluate_random_forest(model, X_test, y_test, best_model_path)

    # DEBUG only: print data - health checks
    print(eval_data)

    # Save evaluation details
    save_output_data(eval_data, options.output_data_path, 'eval', options.run_env)