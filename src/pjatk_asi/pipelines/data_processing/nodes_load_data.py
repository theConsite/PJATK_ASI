import os
import opendatasets as od
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import sqlite3

import wandb

def __load_data(path_to_dir : Path, file_name: str):
    wandb.init(project="PJATK_ASI")
    is_existing = os.path.exists(path_to_dir + file_name)

    if not is_existing:
        od.download(
            "https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction/download?datasetVersionNumber=1",
            data_dir=path_to_dir)
        os.rename(path_to_dir + "credit-card-fraud-prediction/fraud test.csv", path_to_dir + file_name)
        os.rmdir(path_to_dir + "credit-card-fraud-prediction")
        return shuffle(pd.read_csv(path_to_dir + 'credit_data.csv'))
    else:
        conn = sqlite3.connect(path_to_dir + file_name)
        cursor = conn.cursor()
        select_query = 'SELECT * FROM fraud_data'
        cursor.execute(select_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        df = pd.DataFrame(rows, columns=columns).rename(columns={"id": "Unnamed: 0"})
        conn.close()
        return df

def get_data(did_wandb_start: bool):
    folder = 'src/pjatk_asi/'
    fraud_df = __load_data(folder, 'fraud_db.db')
    return fraud_df
