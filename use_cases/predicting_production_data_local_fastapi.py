import logging

from ml_project.config import Config
from ml_project.prediction_process import get_server_predictions
from ml_project.production_data_retrieval import run_production_simulator
from ml_project.utils import setup_logging
from use_cases.use_case_config import config

logger = logging.getLogger('standard')


def execute_predicting_production_data_local_server(config: Config, production_data):

    # Production data retrieval
    predictions = get_server_predictions(config, data=production_data)

    return predictions


if __name__ == '__main__':

    setup_logging('standard')

    config.prediction_service_url = "http://127.0.0.1:8000/predict"

    run_production_simulator(config, execute_predicting_production_data_local_server)