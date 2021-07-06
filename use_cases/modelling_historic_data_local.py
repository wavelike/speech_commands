import logging

import mlflow
from mlflow import log_metrics

import numpy as np

from ml_project.data_validation import validate_engineered_data_as_batch, validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.evaluation import evaluate_model, evaluate_predictions
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.historic_data_retrieval import process_historic_data_into_raw_data, retrieve_audio_data
from ml_project.ml_monitoring import create_drift_dashboard
from ml_project.model_export import export_model_artifacts
from ml_project.modelling_process.modelling_process import get_trained_model, get_processed_data, split_features_and_target, split_into_modelling_and_holdout_data
from ml_project.prediction_process import get_predictions
from ml_project.utils import set_mlflow_experiment, setup_logging
from use_cases.use_case_config import config

logger = logging.getLogger('standard')


def execute_modelling_historic_data_local(config):

    ######
    ### setup the run
    logger.info("Setup mlflow experiment")
    set_mlflow_experiment(experiment_name=config.mlflow_experiment)
    ###
    ######

    ######
    ### Spectogram data retrieval
    logger.info("Retrieve spectogram data")
    train_data, val_data, test_data, commands = retrieve_audio_data(config)
    ###
    ######

    ######
    ### Modelling process
    logger.info("Start modelling process")
    model = get_trained_model(config, train_data, val_data, commands)
    ###
    ######

    ######
    ### Evaluate final model on holdout test data
    logger.info("Evaluate holdout_test predictions and final model")

    test_audio = []
    test_labels = []

    for audio, label in test_data:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_data = np.array(test_audio)
    test_labels = np.array(test_labels)

    holdout_test_predictions = get_predictions(config, model, test_data)
    metrics = evaluate_predictions(config, test_labels, holdout_test_predictions)
    log_metrics(metrics)
    ###
    ######

    ######
    ### Model export and logging
    logger.info("Export and log model objects")

    if config.export_model_artifacts:
        model_export = export_model_artifacts(config, model)

        mlflow.sklearn.log_model(model_export, 'model')
        logger.info(f"Model saved in run {mlflow.active_run().info.run_uuid}")
    ###
    ######


if __name__ == '__main__':

    setup_logging('standard')

    execute_modelling_historic_data_local(config)