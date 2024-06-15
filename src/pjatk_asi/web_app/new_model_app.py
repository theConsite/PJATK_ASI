import datetime
import json
import os
import subprocess

import pandas as pd
import streamlit as st
import plotly.express as px

def run_kedro_pipeline(model_class, hyperparameters_path):
    # Save the current working directory
    original_cwd = os.getcwd()

    try:
        # Change the current working directory to ../../../
        os.chdir('../../../')

        # Define the command to run the Kedro pipeline
        command = [
            "python",
            "-W", "default:Kedro is not yet fully compatible",
            "-m", "kedro",
            "run",
            f"--params=model_type={model_class},hyperparameters={hyperparameters_path}"
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output and error (if any)
        print("Output:\n", result.stdout)
        print("Error:\n", result.stderr)

    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)

def get_newest_subfolder(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    newest_folder = max(subfolders, key=os.path.getmtime)
    return os.path.basename(newest_folder)

# Streamlit page layout
st.title("Model Training")

# Dropdown menu for classifier selection
classifier_option = st.selectbox(
    'Select Classifier',
    ('GradientBoostingClassifier', 'RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier')
)

# Hyperparameter inputs for GradientBoostingClassifier
if classifier_option == 'GradientBoostingClassifier':
    learning_rate = st.text_input('Learning Rate (comma-separated)', value='0.1')
    n_estimators = st.text_input('Number of Estimators (comma-separated)', value='100')
    max_depth = st.text_input('Max Depth (comma-separated)', value='3')
    min_samples_split = st.text_input('Min Samples Split (comma-separated)', value='2')
    min_samples_leaf = st.text_input('Min Samples Leaf (comma-separated)', value='1')

    # Convert inputs to lists
    param_grid = {
        'learning_rate': [float(x) for x in learning_rate.split(',')],
        'n_estimators': [int(x) for x in n_estimators.split(',')],
        'max_depth': [int(x) for x in max_depth.split(',')],
        'min_samples_split': [int(x) for x in min_samples_split.split(',')],
        'min_samples_leaf': [int(x) for x in min_samples_leaf.split(',')]
    }

# Hyperparameter inputs for RandomForestClassifier
elif classifier_option == 'RandomForestClassifier':
    n_estimators = st.text_input('Number of Estimators (comma-separated)', value='100')
    max_depth = st.text_input('Max Depth (comma-separated)', value='None')
    min_samples_split = st.text_input('Min Samples Split (comma-separated)', value='2')
    min_samples_leaf = st.text_input('Min Samples Leaf (comma-separated)', value='1')
    max_features = st.text_input('Max Features (comma-separated)', value='10')

    # Convert inputs to lists
    param_grid = {
        'n_estimators': [int(x) for x in n_estimators.split(',')],
        'max_depth': [None if x == 'None' else int(x) for x in max_depth.split(',')],
        'min_samples_split': [int(x) for x in min_samples_split.split(',')],
        'min_samples_leaf': [int(x) for x in min_samples_leaf.split(',')],
        'max_features': [int(x) for x in max_features.split(',')]
    }

# Hyperparameter inputs for LogisticRegression
elif classifier_option == 'LogisticRegression':
    C = st.text_input('Inverse of Regularization Strength C (comma-separated)', value='1.0')
    max_iter = st.text_input('Maximum Number of Iterations (comma-separated)', value='100')
    solver = st.text_input('Solver (comma-separated)', value='lbfgs')
    penalty = st.text_input('Penalty (comma-separated)', value='l2')

    # Convert inputs to lists
    param_grid = {
        'C': [float(x) for x in C.split(',')],
        'max_iter': [int(x) for x in max_iter.split(',')],
        'solver': solver.split(','),
        'penalty': penalty.split(',')
    }

# Hyperparameter inputs for KNeighborsClassifier
elif classifier_option == 'KNeighborsClassifier':
    n_neighbors = st.text_input('Number of Neighbors (comma-separated)', value='5')
    weights = st.text_input('Weight Function (comma-separated)', value='uniform')
    algorithm = st.text_input('Algorithm (comma-separated)', value='auto')
    leaf_size = st.text_input('Leaf Size (comma-separated)', value='30')

    # Convert inputs to lists
    param_grid = {
        'n_neighbors': [int(x) for x in n_neighbors.split(',')],
        'weights': weights.split(','),
        'algorithm': algorithm.split(','),
        'leaf_size': [int(x) for x in leaf_size.split(',')]
    }

# Train button
if st.button('Train'):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f'{current_datetime}_hyperparameters.json'
    # Save the hyperparameters
    with open(filename, 'w') as file:
        json.dump(param_grid, file, indent=4)

    with st.spinner('Training in progress...'):
        run_kedro_pipeline(classifier_option, os.path.join(os.path.abspath("."), filename))

    # Get the newest subfolder containing the confusion_matrix.png
    report_folder = '../../../data/08_reporting/confusion_matrix.png/'
    newest_folder = get_newest_subfolder(report_folder)
    print(newest_folder)
    confusion_matrix_path = os.path.join(report_folder, newest_folder, 'confusion_matrix.png')
    print(confusion_matrix_path)

    st.write(f"# Model name: {newest_folder}")

    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption='Confusion Matrix')
    else:
        st.error('Confusion matrix not found.')

    metrics_folder = '../../../data/08_reporting/metrics.json/'
    metrics_path = os.path.join(metrics_folder, newest_folder, 'metrics.json')

    # Load and plot metrics
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as file:
            metrics = json.load(file)

        # Flatten the metrics dictionary
        flattened_metrics = {k: list(v.values())[0] for k, v in metrics.items()}

        # Create a DataFrame for plotting
        metrics_df = pd.DataFrame(list(flattened_metrics.items()), columns=['Metric', 'Value'])

        # Create a bar plot
        fig = px.bar(metrics_df, x='Metric', y='Value', title='Model Metrics', labels={'Value': 'Score'})
        st.plotly_chart(fig)

        selected_hyperparameters_folder = '../../../data/08_reporting/selected_hyperparamters.json'
        selected_hyperparametrs = os.path.join(selected_hyperparameters_folder, newest_folder, "selected_hyperparamters.json")
        with open(selected_hyperparametrs, 'r') as file:
            flattened_hyperparameters = {k: list(v.values())[0] for k, v in json.load(file).items()}
            st.json(flattened_hyperparameters)
    else:
        st.error('Metrics file not found.')