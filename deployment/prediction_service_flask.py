from typing import Any, Dict, Optional, Tuple

import flask
import pandas as pd
import numpy as np
from flask import Flask, request

from ml_project.config import Config
from ml_project.data_validation import ProductionData
from ml_project.model_export import load_model_artifacts
from ml_project.prediction_process import get_predictions
from ml_project.utils import get_model_artifacts_dir

app = Flask(__name__)


# Model_objects loading
model, config = load_model_artifacts(export_dir=get_model_artifacts_dir())


def _get_predictions(config: Config, production_data: np.array) -> Tuple[pd.Series, pd.DataFrame]:

    ######
    ### Retrieve predictions
    predictions = get_predictions(config, model, production_data)
    ###
    ######

    return predictions


@app.route('/predict', methods=["POST"])
def predict():

    image_array = np.array(ProductionData(**request.get_json()).image_array)
    prediction = _get_predictions(config, production_data=image_array)

    prediction = {
        'label': int(prediction[0]),
    }

    response = flask.jsonify(prediction)

    return response


if __name__ == "__main__":

    app.run(host='0.0.0.0')
