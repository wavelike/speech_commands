import logging

from ml_project.config import Config
from ml_project.data_validation import validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.evaluation import evaluate_predictions
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.historic_data_retrieval import process_historic_data_into_raw_data, retrieve_audio_data
from ml_project.model_export import load_model_artifacts
from ml_project.modelling_process.data_processing import get_processed_data
from ml_project.modelling_process.modelling_process import split_features_and_target
from ml_project.prediction_process import get_predictions
from ml_project.utils import setup_logging
from use_cases.use_case_config import config

import numpy as np

logger = logging.getLogger('standard')


def execute_predicting_historical_data_local(config: Config):

    ######
    ### Spectogram data retrieval
    logger.info("Retrieve spectogram data")
    train_data, val_data, test_data, commands = retrieve_audio_data(config)
    ###
    ######

    ######
    ### Loading of model artifacts
    logger.info("Loading model artifacts")
    model, _ = load_model_artifacts(export_dir=config.export_dir)
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
    # log_metrics(metrics)
    ###
    ######


if __name__ == '__main__':

    setup_logging('standard')

    execute_predicting_historical_data_local(config)