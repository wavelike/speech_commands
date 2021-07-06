import base64
import json
import logging
import multiprocessing
import time

import pytest
import requests
import uvicorn

import numpy as np

import pickle

from ml_project.historic_data_retrieval import retrieve_audio_data
from ml_project.production_data_retrieval import retrieve_production_data
from ml_project.utils import setup_logging

logger = setup_logging('standard')

url_by_framework = {
    'fastapi': "http://127.0.0.1:8000/predict",
    'flask': "http://127.0.0.1:5000/predict",
}

@pytest.fixture(params=['flask'])
def fastapi_server_setup_config(request, config_base):

    api_framework = request.param

    config_api = config_base.set_value("prediction_service_url", url_by_framework[api_framework])

    if True:
        if api_framework == "fastapi":
            from deployment.prediction_service_fastapi import app as fastapi_app

            #fastapi_app = get_fastapi_app(config_api)
            process = multiprocessing.Process(target=uvicorn.run, args=(fastapi_app,))
            process.daemon = True # daemonize process to assure that child process stops if parent testing process stops unexpectedly
            process.start()

        elif api_framework == "flask":
            from deployment.prediction_service_flask import app as flask_app

            # flask_app = get_flaskapi_app(config_api)
            process = multiprocessing.Process(target=flask_app.run, kwargs={'host': '0.0.0.0'})
            process.daemon = True  # daemonize process to assure that child process stops if parent testing process stops unexpectedly
            process.start()

        else:
            raise(Exception(f"Not supported api_framework parameter provided: {api_framework}"))

    time.sleep(3) # wait a bit to let the server start

    yield config_api

    # teardown
    process.terminate()


def test_predict(fastapi_server_setup_config):

    logger.info("Start")

    train_data, val_data, test_data, commands = retrieve_audio_data(fastapi_server_setup_config)

    for data, label in train_data.take(1):

        data = data.numpy()

        data = data.reshape(-1, *data.shape) # add batch dimension

        data_dict = {'image_array': data.tolist()}

        response = requests.post(fastapi_server_setup_config.prediction_service_url, json=data_dict)

        logger.info("Response:")
        logger.info(response.content)

        assert response.status_code == 200