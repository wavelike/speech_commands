import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import yaml

# TODO: Add pydantic validators here for config settings
from ml_project.utils import get_project_root


@dataclass
class Config:

    historic_or_production_data: str
    local_or_deployed: str

    data_dir: str

    model_parameters: Dict[str, Any]

    modelling_data_percentage: float = 0.8
    holdout_test_data_percentage: float = 0.2

    n_hyperparameter_optimisation_runs: int = 3

    n_folds: int = 5
    optimisation_metric: str = 'acc_avg'
    max_or_min_optimisation_metric: str = 'max'


    export_model_artifacts: bool = True
    export_dir: Optional[str] = None

    prediction_service_url: Optional[str] = None

    mlflow_experiment: str = "default"

    drift_dashboard_filepath: Optional[str] = None

    def __post_init__(self):

        if self.export_dir is None:

            project_config_filepath = os.path.join(get_project_root(), "project_config.yaml")
            with open(project_config_filepath, 'r') as stream:
                project_configs = yaml.safe_load(stream)

            self.export_dir = os.path.join(get_project_root(), project_configs['MODEL_ARTIFACTS_DIR'])


    def set_value(self, field, value):

        assert hasattr(self, field)
        self.__setattr__(field, value)

        return self