# pipelines/scoring/nodes.py
import os
import pandas as pd
import pickle


def load_data(raw_data_path: str) -> pd.DataFrame:
    print("load_data")
    return pd.read_csv(raw_data_path)


def predict(preprocessed_data: pd.DataFrame, model_folder: str) -> pd.DataFrame:
    # Load the model from the specified folder
    model_path = os.path.join("data/06_models/model.pickle/", model_folder, 'model.pickle')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    predictions = model.predict(preprocessed_data)
    return pd.DataFrame(predictions, columns=['is_fraud'])


def load_encoders(model_folder: str):
    ohe_path = os.path.join("data/06_models/one_hot_encoder.pickle/", model_folder, 'one_hot_encoder.pickle')
    with open(ohe_path, 'rb') as file:
        ohe = pickle.load(file)

    std_path = os.path.join("data/06_models/std_scaler.pickle/", model_folder, 'std_scaler.pickle')
    with open(std_path, 'rb') as file:
        std = pickle.load(file)

    return ohe, std