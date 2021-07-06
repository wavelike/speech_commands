import base64
from typing import Tuple

import pandas as pd
import uvicorn
from fastapi import FastAPI

import numpy as np
import pickle

from ml_project.config import Config
from ml_project.data_validation import ProductionData, validate_engineered_data_per_instance, validate_raw_data_per_instance
from ml_project.feature_engineering.feature_engineering import execute_feature_engineering, get_feature_processes
from ml_project.model_export import load_model_artifacts
from ml_project.modelling_process.data_processing import get_processed_data
from ml_project.prediction_process import get_predictions
from ml_project.production_data_retrieval import process_production_input_data_into_raw_data
from ml_project.utils import get_model_artifacts_dir

app = FastAPI()

# Model_objects loading
model, config = load_model_artifacts(export_dir=get_model_artifacts_dir())


def _get_predictions(config: Config, production_data: np.array) -> Tuple[pd.Series, pd.DataFrame]:

    ######
    ### Retrieve predictions
    predictions = get_predictions(config, model, production_data)
    ###
    ######

    return predictions


@app.post("/predict")
async def predict(data: ProductionData):
    """

    """

    image_array = np.array(data.image_array)
    prediction = _get_predictions(config, production_data=image_array)

    prediction = {
        'label': int(prediction[0]),
    }

    return prediction



if __name__ == "__main__":


    uvicorn.run(app) #, host="0.0.0.0") #, port=8000)

