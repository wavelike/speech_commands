import logging
from typing import Any, Dict

from ml_project.config import Config
from ml_project.data_validation import validate_engineered_data_per_instance, validate_raw_data_per_instance, ProductionData
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.historic_data_retrieval import retrieve_audio_data
from ml_project.model_export import load_model_artifacts
from ml_project.modelling_process.modelling_process import get_processed_data
from ml_project.prediction_process import get_predictions
from ml_project.production_data_retrieval import process_production_input_data_into_raw_data, retrieve_production_data, run_production_simulator
from ml_project.utils import setup_logging
from use_cases.use_case_config import config

import numpy as np

logger = logging.getLogger('standard')


######
### Loading of model artifacts
logger.info("Loading model artifacts")
model, _ = load_model_artifacts(export_dir=config.export_dir)
###
######

def execute_predicting_production_data_local(config: Config, production_data):

    ######
    ###
    logger.info("Get predictions")
    image_array = np.array(ProductionData(**production_data).image_array)
    predictions = get_predictions(config, model, image_array)
    ###
    ######

    return predictions


if __name__ == '__main__':

    setup_logging('standard')

    run_production_simulator(config, execute_predicting_production_data_local)
