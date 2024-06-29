# pipelines/scoring/nodes.py
import os

import joblib
import pandas as pd
import pickle

import wandb


def load_data(raw_data_path: str) -> pd.DataFrame:
    print("load_data")
    df = pd.read_csv(raw_data_path)
    df = df.drop('Unnamed: 0.1', axis=1)
    return df


def predict(model, preprocessed_data: pd.DataFrame, model_folder: str) -> pd.DataFrame:

    # model_path = os.path.join("data/06_models/model.pickle/", model_folder, 'model.pickle')
    # with open(model_path, 'rb') as file:
    #     model = pickle.load(file)

    # Make predictions
    predictions = model.predict(preprocessed_data)
    return pd.DataFrame(predictions, columns=['is_fraud'])


def load_artifacts(model_folder: str):
    wandb.login(key="8a0aeda791abab9d3da3283f4f4129c5d3223aa7")
    wandb.init(project="PJATK_ASI", name="download_run")

    # Download the artifact
    artifact = wandb.use_artifact('pjatk-asi/PJATK_ASI/new_model:' + model_folder, type='model')
    artifact_dir = artifact.download()

    # Load the downloaded artifacts
    ohe_path = artifact_dir + '/ohe.pkl'
    scaler_path = artifact_dir + '/std.pkl'

    ohe = joblib.load(ohe_path)
    std = joblib.load(scaler_path)

    model_path = artifact_dir + '/model.pkl'
    model = joblib.load(model_path)

    # ohe_path = os.path.join("data/06_models/one_hot_encoder.pickle/", model_folder, 'one_hot_encoder.pickle')
    # with open(ohe_path, 'rb') as file:
    #     ohe = pickle.load(file)
    #
    # std_path = os.path.join("data/06_models/std_scaler.pickle/", model_folder, 'std_scaler.pickle')
    # with open(std_path, 'rb') as file:
    #     std = pickle.load(file)

    return ohe, std, model