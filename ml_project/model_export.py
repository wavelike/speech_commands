import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from keras.models import Sequential, load_model

from ml_project.config import Config
from ml_project.modelling_process.data_processing import PreprocessingObjects

logger = logging.getLogger('standard')


@dataclass
class ModelArtifacts:

    config: Config


def export_model_artifacts(config: Config, model: Sequential) -> Optional[ModelArtifacts]:
    """
    Stores model artifacts as a serialised object
    """

    model_artifacts: Optional[ModelArtifacts]
    if config.export_dir is not None:

        model_artifacts = ModelArtifacts(config=config)

        model_fileapth = os.path.join(config.export_dir, "model.h5")
        model.save(model_fileapth)

        artifact_filepath = os.path.join(config.export_dir, "model_artifacts.pkl")
        with open(artifact_filepath, "wb+") as file:
            pickle.dump(model_artifacts, file)

        logger.info(f"Model artifacts stored to {config.export_dir}")
    else:
        model_artifacts = None

    return model_artifacts


def load_model_artifacts(export_dir: Optional[str]) -> Tuple[Sequential, Config]:
    """
    Loads and returns model artifacts from the provided filepath
    """

    model: Optional[Any]
    preprocessing_objects: Optional[PreprocessingObjects]
    config: Optional[Config]

    if export_dir is not None:

        model_fileapth = os.path.join(export_dir, "model.h5")
        model = load_model(model_fileapth)

        artifact_filepath = os.path.join(export_dir, "model_artifacts.pkl")
        with open(artifact_filepath, "rb") as file:
            model_artifacts: ModelArtifacts = pickle.load(file)

        config = model_artifacts.config
    else:
        logger.warning(f"Export_filepath is None")
        model, config = None, None

    return model, config