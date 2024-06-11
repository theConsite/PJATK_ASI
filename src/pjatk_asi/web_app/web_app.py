import json
import os.path
import pickle
import subprocess
from io import StringIO
from pathlib import Path

import streamlit as st
import pandas as pd

from kedro.runner import SequentialRunner
from kedro.framework.context import KedroContext
from kedro.framework.hooks.manager import _create_hook_manager
from kedro.config import OmegaConfigLoader
from kedro.framework.project import find_pipelines

def run_kedro_pipeline(model, data):
    # Define the command to run the Kedro pipeline
    command = [
        "python",
        "-W", "default:Kedro is not yet fully compatible",
        "-m", "kedro",
        "run ../../../",
        "--pipeline=sp",
        f"--params=model_folder={model},raw_data_path={data}"
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and error (if any)
    print("Output:\n", result.stdout)
    print("Error:\n", result.stderr)



# Function to flatten the JSON structure
def flatten_metrics(json_data):
    flattened_metrics = {key: value['0'] for key, value in json_data.items()}
    return flattened_metrics


# Path to the directory containing subfolders
models_dir = Path('../../../data/06_models/model.pickle')

# Get a list of subfolder names
subfolders = [subfolder.name for subfolder in models_dir.iterdir() if subfolder.is_dir()]
# Allow the user to choose a subfolder
selected_subfolder = st.sidebar.selectbox('Select model:', subfolders)
selected_subfolder_path = models_dir / selected_subfolder

metrics_dir = os.path.join("../../../data/08_reporting/metrics.json", selected_subfolder)
st.sidebar.markdown("# Model Metrics")
metrics_file = os.path.join(metrics_dir, "metrics.json")
with open(metrics_file, 'r') as file:
    st.sidebar.json(flatten_metrics(json.load(file)))



# Title of the app
st.title("Credit Card Fraud Predictions")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    score_data_path = "score_data.csv"
    data.to_csv(score_data_path)

    if st.button('Predict'):
        run_kedro_pipeline(selected_subfolder, score_data_path)

        data = pd.read_csv(score_data_path)
        score_result = pd.read_csv("../../../data/07_model_output/score_result.csv")

        data['is_fraud'] = score_result['is_fraud']

        def highlight_rows(row):
            color = 'red' if row['is_fraud'] == 1 else 'green'
            return ['background-color: {}'.format(color) for _ in row]

            # Apply the highlight function

        styled_data = data.style.apply(highlight_rows, axis=1)

        # Display the data in Streamlit
        st.title("Score Data with Predictions")
        st.dataframe(styled_data)


        prediction_counts = pd.Series(score_result['is_fraud']).value_counts()
        st.write("Prediction Counts:")
        st.write(prediction_counts)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )
