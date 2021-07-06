import os
import pathlib
from typing import Tuple, List

import pandas as pd
import numpy as np

from ml_project.config import Config
from ml_project.data_validation import raw_data_schema

import tensorflow as tf



def decode_audio(audio_binary) -> tf.Tensor:
    """
    Reads the binary WAV-encoded audio file and converts it into a numerical tensor.
    The audio signals amplitude is normalised to values within the range [-1, 1]
    """

    audio, _ = tf.audio.decode_wav(audio_binary)

    return tf.squeeze(audio, axis=-1)


def get_label(file_path: str) -> str:
    # The label of a file is its parent directory name

      parts = tf.strings.split(file_path, os.path.sep)

      # Note: You'll use indexing here instead of tuple unpacking to enable this
      # to work in a TensorFlow graph.

      return parts[-2]


def get_waveform_and_label(file_path: str):
    # Takes as input a filepath and outputs a numerical tensor representing the audio signal and its corresponding label

    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)

    return waveform, label


def get_spectrogram(waveform):
    """
    Converts waveform data into a spectrogram which shows frequency changes over time (2D image) by applying short-time Fourier Transform (STFT).
    While a single Fourier Transformation converts the audio data into the frequency domain and thus looses all time information, STFT splits the signal into windows of time and
    runs a Fourier Transformation for each window, resulting in a dataset with spectrum information over time in two dimensions.
    """

    # Padding for files with less than 16000 samples in order to give all waveforms the same length and thus the same shape for the 2D output Tensor
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    # Fourier Transformation returns complex numbers, but we are only interested in the magnitude of the frequency and not the phase -> use abs()
    spectrogram = tf.abs(spectrogram)

    return spectrogram

def get_spectrogram_and_label_id_func(commands):

    def get_spectrogram_and_label_id(audio: tf.Tensor, label: str):


        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)

        return spectrogram, label_id

    return get_spectrogram_and_label_id

def get_spectogram_data_from_files(filenames: List[str], commands: np.array):

    AUTOTUNE = tf.data.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)
    waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    # Convert waveform data into spectogram
    spectrogram_data = waveform_ds.map(get_spectrogram_and_label_id_func(commands), num_parallel_calls=AUTOTUNE)

    return spectrogram_data

def retrieve_audio_data(config) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, np.array]:

    data_dir = pathlib.Path(config.data_dir)
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            extract=True,
            cache_dir=data_dir,
            #cache_subdir='data'
        )

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    print('Example file tensor:', filenames[0])

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']

    train_files = filenames[:6400]
    val_files = filenames[6400: 6400 + 800]
    test_files = filenames[-800:]

    print('Training set size', len(train_files))
    print('Validation set size', len(val_files))
    print('Test set size', len(test_files))

    train_data = get_spectogram_data_from_files(train_files, commands)
    val_data = get_spectogram_data_from_files(val_files, commands)
    test_data = get_spectogram_data_from_files(test_files, commands)

    return train_data, val_data, test_data, commands


def process_historic_data_into_raw_data(config: Config, historic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Processing of the historic data into raw data that adheres to a given schema (column names, column types, NaN handling, etc.).
    Raw data originating from historic data has the exact same schema as raw_data retrieved from production sources in order to allow unified further processing.

    :param config:
    :param historic_data:
    :return:
    """

    #####
    ### Column selection
    historic_data = historic_data[config.features + [config.target_col]]
    ###

    #####
    ### NaN imputing for continuous and categorical columns
    historic_data.loc[:, config.cat_cols] = historic_data[config.cat_cols].fillna('<missing>')
    historic_data.loc[:, config.cont_cols] = historic_data[config.cont_cols].fillna(historic_data[config.cont_cols].min() - 999999)
    #####

    return historic_data


