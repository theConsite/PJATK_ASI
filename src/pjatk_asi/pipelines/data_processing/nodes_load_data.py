import os
import opendatasets as od
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle

import wandb


def __download_data(path_to_dir : Path, file_name: str):
    wandb.init(project="PJATK_ASI")
    is_existing = os.path.exists(path_to_dir + file_name)

    if not is_existing:
        od.download(
            "https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction/download?datasetVersionNumber=1",
            data_dir=path_to_dir)
        os.rename(path_to_dir + "credit-card-fraud-prediction/fraud test.csv", path_to_dir + file_name)
        os.rmdir(path_to_dir + "credit-card-fraud-prediction")


def get_data(did_wandb_start: bool):
    folder = 'data/01_raw/'
    __download_data(folder, 'credit_data.csv')
    fraud_df = shuffle(pd.read_csv(folder + 'credit_data.csv'))
    return fraud_df
