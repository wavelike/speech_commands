import logging
import os
import statistics
from typing import Dict, List, Tuple

import mlflow
import pandas as pd
import skopt.space as skopt_space
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import Optimizer
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental import preprocessing

from ml_project.config import Config
from ml_project.evaluation import evaluate_predictions
from ml_project.modelling_process.data_processing import get_processed_data
from ml_project.modelling_process.model_functions import get_model, train_model
from ml_project.prediction_process import get_predictions

import tensorflow as tf

from ml_project.utils import get_project_root

logger = logging.getLogger('standard')


def get_hyperparameter_optimizer() -> Tuple[Optimizer, List[str]]:

    #####
    # Skopt Bayesian Optimizer
    opt_space_hyperparameters = [
        skopt_space.Integer(low=5, high=15, name='max_depth'),
        skopt_space.Integer(low=1, high=50, name='min_samples_leaf'),
        skopt_space.Real(low=0., high=0.3, name='min_weight_fraction_leaf'),
    ]

    hyperparameter_optimizer = Optimizer(
        dimensions=opt_space_hyperparameters,
        base_estimator="ET",  # ExtraTrees for the surrogate model
        #acq_optimizer="sampling",
        n_initial_points=5,
    )

    hyperparameter_optimizer_dim_names = [opt_dim.name for opt_dim in hyperparameter_optimizer.space.dimensions]
    #####

    #####
    # Grid Search
    # TODO
    #####

    return hyperparameter_optimizer, hyperparameter_optimizer_dim_names


def run_cross_validation_loop(config: Config, modelling_data: pd.DataFrame, hyperparameters: Dict) -> Dict:

    cv_fold_assigner = get_cv_fold_assigner(config, modelling_data)

    cv_metrics = {}
    for cross_validation_run in range(config.n_folds):

        # cv data splits into train and validation, split into feature and target data
        train_data, validation_data = split_into_training_and_validation_data(cv_fold_assigner, modelling_data)
        train_x, train_y = split_features_and_target(config, train_data)
        validation_x, validation_y = split_features_and_target(config, validation_data)

        # model-specific data preprocessing
        train_x_processed, validation_x_processed, _ = get_processed_data(config=config,
                                                                        preprocessing_objects=None,  # is None because fresh modelling runs are executed
                                                                        train_x=train_x,
                                                                        validation_x=validation_x)

        # model initialisation and training
        model = get_model(config, hyperparameters)
        model = train_model(config, model, train_x_processed, train_y)

        # predicting on validation data to evaluate cv run
        validation_predictions, validation_prediction_probas = get_predictions(config, model, validation_x_processed)
        metrics = evaluate_predictions(config, validation_y, validation_predictions, validation_prediction_probas)

        # adding run's metric values to their corresponding list in 'cv_metrics' to be eventually averaged
        if len(cv_metrics) != 0:
            cv_metrics = {metric_name: metric_values + [metrics[metric_name]] for metric_name, metric_values in cv_metrics.items()}
        else:
            cv_metrics = {metric_name: [metrics[metric_name]] for metric_name, metric_value in metrics.items()}

    return cv_metrics


def run_hyperparameter_optimisation_cv_loop(config: Config, modelling_data: pd.DataFrame) -> pd.DataFrame:

    leaderboard = pd.DataFrame(columns=['hyperparameters'])
    hyperparameter_optimizer, hyperparameter_optimizer_dim_names = get_hyperparameter_optimizer()

    for hyperparameter_optimisation_run in range(config.n_hyperparameter_optimisation_runs):

        # Retrieve hyperparameters
        hyperparameters_list = hyperparameter_optimizer.ask()
        hyperparameters = {dim_name: hyperparameters_list[ind] for ind, dim_name in enumerate(hyperparameter_optimizer_dim_names)}
        logger.info(f"New hyperparameters in run {hyperparameter_optimisation_run + 1} / {config.n_hyperparameter_optimisation_runs}: {hyperparameters}")

        # Run cross validation and average the resulting metrics
        cv_metrics = run_cross_validation_loop(config, modelling_data, hyperparameters)
        averaged_cv_metrics = {metric_name + "_avg": statistics.mean(metric_values) for metric_name, metric_values in cv_metrics.items()}

        if config.max_or_min_optimisation_metric == "max":
            objective_to_be_minimised = -1 * averaged_cv_metrics[config.optimisation_metric]
        elif config.max_or_min_optimisation_metric == "min":
            objective_to_be_minimised = averaged_cv_metrics[config.optimisation_metric]
        else:
            raise (Exception(f"No valid value for 'config.max_or_min_optimisation_metric' provided: {config.max_or_min_optimisation_metric}"))

        # Inform hyperparameter optimizer about the objective value for the given hyperparameters
        hyperparameter_optimizer.tell(hyperparameters_list, objective_to_be_minimised)

        # Add results of current run to leaderboard
        new_leaderboard_entry = pd.Series({
            'hyperparameters': hyperparameters,
            **averaged_cv_metrics,
            **cv_metrics,
            })

        leaderboard = leaderboard.append(new_leaderboard_entry, ignore_index=True)

    return leaderboard


def split_into_modelling_and_holdout_data(config, data):


    #####
    # chunk split
    holdout_test_starts_index = int(len(data) * config.modelling_data_percentage)
    modelling_data = data.loc[:holdout_test_starts_index]
    holdout_test_data = data.loc[holdout_test_starts_index:]
    #####

    return modelling_data, holdout_test_data


def get_cv_fold_assigner(config: Config, modelling_data):

    #####
    # scikit-learn kFold
    cv_fold_assigner = KFold(n_splits=config.n_folds).split(modelling_data)
    #####

    #####
    cv_fold_assigner = StratifiedKFold(n_splits=config.n_folds).split(modelling_data, y=modelling_data[config.target_col])
    #####

    return cv_fold_assigner


def split_into_training_and_validation_data(cv_fold_assigner, modelling_data):

    # scikit-learn kFolds
    train_indices, validation_indices = next(cv_fold_assigner)
    train_data = modelling_data.loc[train_indices]
    validation_data = modelling_data.loc[validation_indices]

    return train_data, validation_data


def split_features_and_target(config, data) -> Tuple[pd.DataFrame, pd.Series]:

    feature_data = data.drop(columns=[config.target_col])
    target_data = data[config.target_col]

    return feature_data, target_data


def get_best_run_index(config, leaderboard):
    if config.max_or_min_optimisation_metric == "max":
        best_run_index = leaderboard[config.optimisation_metric].argmax()
    elif config.max_or_min_optimisation_metric == "min":
        best_run_index = leaderboard[config.optimisation_metric].argmin()
    else:
        raise (Exception(f"No supported value for 'config.max_or_min_optimisation_metric' provided: {config.max_or_min_optimisation_metric}"))

    return best_run_index


def get_trained_model(config: Config, train_data, val_data, commands):

    AUTOTUNE = tf.data.AUTOTUNE

    # batching trainind and validation set
    batch_size = 64
    train_ds = train_data.batch(batch_size)
    val_ds = val_data.batch(batch_size)

    # reducing latency while training by caching and prefetching data
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in train_data.take(1):
        input_shape = spectrogram.shape

    print('Input shape:', input_shape)
    num_labels = len(commands)

    # Calculate the normalisation statistics based on the train data
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_data.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    plot_model(model, to_file=os.path.join(get_project_root(), 'output/model_architecture.png'), show_shapes=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.model_parameters['epochs'],
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )



    return model





