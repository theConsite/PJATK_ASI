import json
import os.path
import pickle
import subprocess
from io import StringIO
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import wandb

from io import StringIO

from kedro.runner import SequentialRunner
from kedro.framework.context import KedroContext
from kedro.framework.hooks.manager import _create_hook_manager
from kedro.config import OmegaConfigLoader
from kedro.framework.project import find_pipelines


def run_kedro_pipeline(model, data):
    original_cwd = os.getcwd()
    # Define the command to run the Kedro pipeline
    try:
        # Change the current working directory to ../../../
        os.chdir('../../../')
        command = [
            "python",
            "-W", "default:Kedro is not yet fully compatible",
            "-m", "kedro",
            "run",
            "--pipeline=sp",
            f"--params=model_folder={model},raw_data_path={data}"
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output and error (if any)
        print("Output:\n", result.stdout)
        print("Error:\n", result.stderr)
    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)



# Function to flatten the JSON structure
def flatten_metrics(json_data):
    flattened_metrics = {key: value['0'] for key, value in json_data.items()}
    return flattened_metrics


# Path to the directory containing subfolders
#models_dir = Path('../../../data/06_models/model.pickle')

# Get a list of subfolder names
#subfolders = [subfolder.name for subfolder in models_dir.iterdir() if subfolder.is_dir()]
# Allow the user to choose a subfolder
#selected_subfolder = st.sidebar.selectbox('Select model:', subfolders)
#selected_subfolder_path = models_dir / selected_subfolder

#metrics_dir = os.path.join("../../../data/08_reporting/metrics.json", selected_subfolder)
#st.sidebar.markdown("# Model info")
#metrics_file = os.path.join(metrics_dir, "metrics.json")
#with open(metrics_file, 'r') as file:
#    st.sidebar.json(flatten_metrics(json.load(file)))

#hyperparameters_dir = os.path.join("../../../data/08_reporting/selected_hyperparamters.json", selected_subfolder)
#hyperparameters_file = os.path.join(hyperparameters_dir, "selected_hyperparamters.json")
#print(hyperparameters_file)
#with open(hyperparameters_file, 'r') as file:
#    st.sidebar.json(flatten_metrics(json.load(file)))


selected_subfolder = st.sidebar.selectbox('Select model:', ['release', 'latest'])

# Title of the app
st.title("Credit Card Fraud Predictions")

wandb_runnung = False

selected_subfolder_prev = None
@st.cache_resource
def load_artifact_metadata(artifact_name):
    global wandb_runnung
    if wandb_runnung:
        return

    global selected_subfolder_prev
    if selected_subfolder_prev == selected_subfolder:
        return

    selected_subfolde_prev = selected_subfolder

    wandb_runnung = True
    wandb.login(key="8a0aeda791abab9d3da3283f4f4129c5d3223aa7")
    wandb.init(project="PJATK_ASI", name="download_run", reinit=True)

    artifact = wandb.use_artifact(artifact_name, type='model')
    wandb_runnung = False
    return artifact.metadata

if selected_subfolder:
    # wandb.login(key="8a0aeda791abab9d3da3283f4f4129c5d3223aa7")
    # # Initialize W&B run
    # wandb.init(project="PJATK_ASI", name="download_run", reinit=True)

    artifact_metadata = load_artifact_metadata('pjatk-asi/PJATK_ASI/new_model:' + selected_subfolder)

    print(type(artifact_metadata['metrics']))
    print(artifact_metadata['metrics'])
    st.sidebar.markdown("# Model info")
    # Read the metrics string into a DataFrame
    metrics_df = pd.read_csv(StringIO(artifact_metadata['metrics']), delim_whitespace=True)
    # Convert the DataFrame to a dictionary (assuming only one row)
    metrics_dict = metrics_df.iloc[0].to_dict()

    # Display the JSON object using st.json
    st.sidebar.write('Metrics:')
    st.sidebar.json(metrics_dict)

    params_df = pd.read_csv(StringIO(artifact_metadata['parameters']), delim_whitespace=True)
    params_dict = params_df.iloc[0].to_dict()

    # Display the JSON object using st.json
    st.sidebar.write('Hipermarameters:')
    st.sidebar.json(params_dict)


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(data)

    score_data_path = "score_data.csv"
    data.to_csv(score_data_path)

    if st.button('Predict'):
        with st.spinner('Calculating Prediction...'):
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
