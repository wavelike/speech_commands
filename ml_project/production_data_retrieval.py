import json
import logging
from typing import Callable

import numpy as np
import pandas as pd

from ml_project.config import Config
from ml_project.historic_data_retrieval import retrieve_audio_data

logger = logging.getLogger('standard')


def retrieve_production_data(config: Config, json_string: str, ) -> pd.DataFrame:
    """
    Retrieves production data from the relevant data source (dict object, json object)
    :return:
    """

    # from json string
    production_data = pd.DataFrame([json.loads(json_string)])

    return production_data


def process_production_input_data_into_raw_data(config: Config, production_data: pd.DataFrame) -> pd.DataFrame:
    """
    Processing of the production data into raw data that adheres to a given schema (column names, column types, NaN handling, etc.).
    Raw data originating from historic data has the exact same schema as raw_data retrieved from production sources in order to allow unified further processing.

    :param config:
    :param production_data:
    :return:
    """

    # Column selection
    production_data = production_data[config.features]

    return production_data


def run_production_simulator(config: Config, callable_function: Callable):
    '''
    simulates a stream of single data instances - here retrieved from the historic data.
    Each instance is passed to the provided 'callable_function'
    '''

    train_data, val_data, test_data, commands = retrieve_audio_data(config)

    for data, label in test_data:

        #data = np.array([data])

        data = data.numpy()

        data = data.reshape(-1, *data.shape)  # add batch dimension

        data_dict = {'image_array': data.tolist()}

        predictions = callable_function(config, data_dict)

        logger.info("")
        logger.info(f"predictions: {predictions}")
        logger.info("")