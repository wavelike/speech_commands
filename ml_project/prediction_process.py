from typing import Any, Tuple, Dict

import pandas as pd
import requests

from ml_project.config import Config
from ml_project.data_validation import ProductionData
import numpy as np


def get_predictions(config: Config, model: Any, data: np.array) -> np.array:

    predictions = np.argmax(model.predict(data), axis=1)

    return predictions


def get_server_predictions(config: Config, data: Dict) -> Tuple[pd.Series, pd.Series]:

    # validate and parse data
    data_dict = vars(ProductionData(**data))

    response = requests.post(config.prediction_service_url, json=data_dict)

    response_dict = response.json()

    predictions = response_dict['label']

    return predictions