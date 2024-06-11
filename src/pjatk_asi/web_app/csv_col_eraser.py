import pandas as pd
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
file = parent_dir + '\\data\\01_raw\\dataset.csv'

data = pd.read_csv(file)
data = data.drop('is_fraud', axis=1)
data.to_csv(parent_dir+'/data/01_raw/dataset_modified.csv', index=False)