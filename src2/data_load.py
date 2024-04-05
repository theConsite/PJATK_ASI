import os
import opendatasets as od
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle


def __download_data():
    path_to_save = '../data/credit_data.csv'
    is_existing = os.path.exists(path_to_save)

    if not is_existing:
        od.download(
            "https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction/download?datasetVersionNumber=1",
            data_dir="../data")
        Path("../data/").mkdir(parents=True, exist_ok=True)
        os.rename("../data/credit-card-fraud-prediction/fraud test.csv", "../data/credit_data.csv")
        os.rmdir("../data/credit-card-fraud-prediction")


def read_data():
    __download_data()
    file_path = '../data/credit_data.csv'
    fraud_df = shuffle(pd.read_csv(file_path))
    return fraud_df


if __name__ == "__main__":
    fraud_df = read_data()
    print(fraud_df.describe())
