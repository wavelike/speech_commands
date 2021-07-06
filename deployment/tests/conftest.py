import os

import pytest

from ml_project.config import Config
from ml_project.utils import get_project_root


@pytest.fixture(scope="package")
def config_base():

    return Config(historic_or_production_data='historic',
                  local_or_deployed='local',
                  model_parameters={},
                  data_dir=os.path.join(get_project_root(), "deployment/tests/test_data/mini_speech_commands_test"),
                  export_dir=os.path.join(get_project_root(), "deployment/tests/test_data"),
                  n_hyperparameter_optimisation_runs=1,
                  mlflow_experiment='exp1',
                  drift_dashboard_filepath="output/reports/",
                  )
