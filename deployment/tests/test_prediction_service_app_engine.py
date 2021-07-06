import json
import logging

import pytest
import requests

from ml_project.data_validation import ProductionData
from ml_project.historic_data_retrieval import retrieve_audio_data
from ml_project.production_data_retrieval import retrieve_production_data
from ml_project.utils import setup_logging

logger = setup_logging('standard')

app_engine_server_url = "https://deployment-316521.ey.r.appspot.com/predict"


@pytest.fixture()
def config_app_engine(config_base):

    return config_base.set_value("prediction_service_url", app_engine_server_url)


def test_predict(config_app_engine):

    train_data, val_data, test_data, commands = retrieve_audio_data(config_app_engine)

    for data, label in train_data.take(1):
        data = data.numpy()

        data = data.reshape(-1, *data.shape)  # add batch dimension

        data_dict = {'image_array': data.tolist()}

        response = requests.post(config_app_engine.prediction_service_url, json=data_dict)

        logger.info("Response:")
        logger.info(response.content)

        assert response.status_code == 200