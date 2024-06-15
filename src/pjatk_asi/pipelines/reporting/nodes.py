import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


import wandb
def create_metrics(Y_pred, Y_true):
    # Calculate various metrics
    accuracy = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred, average='weighted')
    recall = recall_score(Y_true, Y_pred, average='weighted')
    f1 = f1_score(Y_true, Y_pred, average='weighted')
    #auc = roc_auc_score(Y_true, model.predict_proba(Y_true)[:, 1])

    # Create a dictionary to hold the metrics
    metrics = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
    }

    return pd.DataFrame(metrics)

def save_model_to_WandB(model, metrics, params, ohe, std):
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('ohe.pkl', 'wb') as f:
        pickle.dump(ohe, f)

    with open('std.pkl', 'wb') as f:
        pickle.dump(std, f)

    artifact = wandb.Artifact('new_model', type='model',
                              description='New Model',
                              metadata={"parameters": params, "metrics": metrics})

    # Add the model file to the artifact
    artifact.add_file('model.pkl')
    artifact.add_file('ohe.pkl')
    artifact.add_file('std.pkl')

    # Save the artifact
    wandb.run.log_artifact(artifact)


def create_confusion_matrix(Y_pred, Y_true):
    # Normalize the confusion matrix
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]  # normalize along the true axis

    plt.figure(figsize=(8, 6))
    sn.heatmap(conf_matrix_norm, annot=True, fmt=".2%", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (Normalized)')
    return plt