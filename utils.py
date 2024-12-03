import librosa
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm 

def handle_pickle(file_path, data=None, mode="export"):
    """
    Función para exportar o cargar datos usando pickle.

    Args:
        file_path (str): Ruta del archivo pickle.
        data (dict): Datos a exportar (necesario si mode='export').
        mode (str): 'export' para guardar datos, 'load' para cargarlos.

    Returns:
        dict or None:
            Si mode='load', devuelve un diccionario con los datos cargados.
            Si mode='export', devuelve None.
    """
    if mode == "export":
        if data is None:
            raise ValueError("Se necesita 'data' para exportar.")

        # Exportar datos a pickle
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Datos exportados exitosamente a {file_path}")
        return None

    elif mode == "load":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {file_path} no existe.")

        # Cargar datos desde pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Datos cargados exitosamente desde {file_path}")
        return data

    else:
        raise ValueError("El parámetro 'mode' debe ser 'export' o 'load'.")


def get_selected_features(
    direction,
    feature_list=[
        "mel_spectrogram",
        "zcr",
        "rmse",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
        "spectral_rolloff",
        "mfccs",
    ],
    aimed_duration=4,
    sr=22050,
):
    """
    Computes and returns only the specified audio features for a given audio file,
    ensuring each feature is repeated to match the desired duration before concatenation.

    Args:
        direction (str): Path to the audio file.
        feature_list (list of str): List of features to compute. Supported features:
                                    'mel_spectrogram', 'zcr', 'rmse', 'spectral_centroid',
                                    'spectral_bandwidth', 'spectral_contrast',
                                    'spectral_rolloff', 'mfccs'.
        aimed_duration (float): Target duration (in seconds) for the output features.
        sr (int): Sample rate for audio processing.


    Returns:
        np.ndarray: Concatenated array of repeated features, each adjusted to the target duration.
    """
    # Load the audio file
    y, sr = librosa.load(direction, sr=sr)

    # Parameters for feature extraction
    frame_length = int(sr * 0.0232)  # Window size of 23.2 ms
    hop_length = frame_length // 2  # 50% overlap

    # Mapping of feature names to their respective computations
    feature_mapping = {
        "mel_spectrogram": lambda: librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                hop_length=hop_length,
                n_fft=frame_length,
                fmax=sr // 2,
            ).T,
            ref=np.max,
        ),
        "zcr": lambda: librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).T,
        "rmse": lambda: librosa.feature.rms(y=y, hop_length=hop_length).T,
        "spectral_centroid": lambda: librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        ).T,
        "spectral_bandwidth": lambda: librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length
        ).T,
        "spectral_contrast": lambda: librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=hop_length
        ).T,
        "spectral_rolloff": lambda: librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop_length
        ).T,
        "mfccs": lambda: librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=25, hop_length=hop_length, n_fft=frame_length
        ).T,
        "mfccs_d1": lambda: librosa.feature.delta(mfccs).T,
        "mfccs_d2": lambda: librosa.feature.delta(mfccs, order=2).T,
        "envelope": lambda: calculate_envelope(y=y, sr=sr),
    }

    # Compute and repeat each requested feature to match the desired duration
    repeated_features = []
    for feature in feature_list:
        if feature in feature_mapping:
            # Compute the feature and repeat it to match the duration
            feature_array = feature_mapping[feature]()
            repeated_feature = repeat_sound(
                feature_array,
                aimed_duration=aimed_duration,
                sr=sr,
                hop_length=hop_length,
            )
            repeated_features.append(repeated_feature)
        else:
            print(f"Warning: Feature '{feature}' is not supported and will be skipped.")

    # Concatenate the repeated features along axis=1
    return np.concatenate(repeated_features, axis=1) if repeated_features else None


def calculate_envelope(y, sr=None):
    envelope = np.abs(y)
    envelope = librosa.effects.harmonic(envelope)
    envelope /= np.max(envelope)
    return envelope


def repeat_sound(numpy_list_concatenation, aimed_duration=4, sr=22050, hop_length=512):
    # Duración en segundos de cada ventana
    frame_duration = hop_length / sr
    # Duración total del array original
    original_duration = frame_duration * numpy_list_concatenation.shape[0]
    # Número de repeticiones necesarias para alcanzar la duración deseada
    repetitions = int(np.ceil(aimed_duration / original_duration))

    # Repetir las características para alcanzar la duración deseada
    repeated_array = np.tile(numpy_list_concatenation, (repetitions, 1))

    # Recortar al tamaño exacto
    required_frames = int(aimed_duration / frame_duration)
    final_array = repeated_array[:required_frames]

    return final_array


def extract_data_and_folds(
    audio_directory,
    metadata_df,
    feature_list=[
        "mel_spectrogram",
        "zcr",
        "rmse",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
        "spectral_rolloff",
        "mfccs",
    ],
    aimed_duration=4,
):
    """
    Extrae características, etiquetas y el fold de cada archivo de audio.

    Parameters:
    audio_directory (str or Path): Directorio que contiene los archivos de audio.
    metadata_df (DataFrame): DataFrame con la metadata de los audios ('slice_file_name', 'fold', 'classID').

    Returns:
    tuple: (features_list, labels_list, folds_list), donde:
        - features_list: Lista de características extraídas de cada archivo.
        - labels_list: Lista de etiquetas correspondientes.
        - folds_list: Lista de folds a los que pertenece cada archivo.
    """
    audio_directory = Path(audio_directory)
    features_list, labels_list, folds_list = [], [], []

    for file in audio_directory.rglob("*.wav"):
        file_name = file.name

        if file_name not in metadata_df["slice_file_name"].values:
            continue  # Ignorar archivos sin metadata

        # Obtener el fold, la etiqueta y las características
        fold = metadata_df.loc[
            metadata_df["slice_file_name"] == file_name, "fold"
        ].iloc[0]
        label = metadata_df.loc[
            metadata_df["slice_file_name"] == file_name, "classID"
        ].iloc[0]

        # Extraer las características
        features = get_selected_features(
            direction=file, feature_list=feature_list, aimed_duration=4
        )  # Repetir y ajustar duración

        features_list.append(features)
        labels_list.append(label)
        folds_list.append(fold)

    return features_list, labels_list, folds_list


def prepare_datasets(features_list, labels_list, folds_list, train_folds, test_folds):
    """
    Crea conjuntos de datos de entrenamiento y prueba según los folds especificados.

    Parameters:
    features_list (list): Lista de características extraídas.
    labels_list (list): Lista de etiquetas correspondientes.
    folds_list (list): Lista de folds asociados a cada archivo.
    train_folds (list): Lista de números de folds a usar para entrenamiento.
    test_folds (list): Lista de números de folds a usar para prueba.

    Returns:
    tuple: (X_train, X_test, y_train, y_test) donde:
        - X_train, X_test: Listas escaladas de características para entrenamiento y prueba.
        - y_train, y_test: Listas de etiquetas para entrenamiento y prueba.
    """
    # Dividir en conjuntos de entrenamiento y prueba
    train_features, test_features = [], []
    y_train, y_test = [], []

    for features, label, fold in zip(features_list, labels_list, folds_list):
        if fold in train_folds:
            train_features.append(features)
            y_train.append(label)
        elif fold in test_folds:
            test_features.append(features)
            y_test.append(label)

    # Asegurarse de que todas las características tienen la misma forma
    train_features_combined = np.vstack(train_features)
    scaler = StandardScaler()
    scaler.fit(train_features_combined)

    # Escalar características
    X_train = [scaler.transform(features) for features in train_features]
    X_test = [scaler.transform(features) for features in test_features]

    return X_train, X_test, y_train, y_test


def extract_features(metadata, audio_folder, fixed_length=128):
    features = []
    labels = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = None #Initialize to avoid issues in the 'except' block
        try:
            #construct file path
            file_path = os.path.join(audio_folder, f"fold{row['fold']}", row['slice_file_name'])
            #File loading: It reads the audio file using librosa.load and converts the sound into a format your program understands (a waveform and its sample rate)
            y, sr = librosa.load(file_path, sr=22050)
            #adjust n_fft dynamically for short clips
            n_fft = min(2048, len(y))
            #computation of mel-spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=512, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            #padding/truncating
            if S_dB.shape[1]<fixed_length:
                repeat_times = (fixed_length // S_dB.shape[1])+1
                extended = np.tile(S_dB, (1,repeat_times))
                features.append(extended[:, :fixed_length]) 
            else:
                #truncate if too long
                features.append(S_dB[:, :fixed_length])

            labels.append(row['classID'])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return features, labels