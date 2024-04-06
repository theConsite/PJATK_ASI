import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px  # noqa:  F401
import plotly.graph_objs as go
import seaborn as sn
from sklearn.metrics import confusion_matrix


# This function uses plotly.express
# def compare_passenger_capacity_exp(preprocessed_shuttles: pd.DataFrame):
#     return (
#         preprocessed_shuttles.groupby(["shuttle_type"])
#         .mean(numeric_only=True)
#         .reset_index()
#     )
#
#
# # This function uses plotly.graph_objects
# def compare_passenger_capacity_go(preprocessed_shuttles: pd.DataFrame):
#
#     data_frame = (
#         preprocessed_shuttles.groupby(["shuttle_type"])
#         .mean(numeric_only=True)
#         .reset_index()
#     )
#     fig = go.Figure(
#         [
#             go.Bar(
#                 x=data_frame["shuttle_type"],
#                 y=data_frame["passenger_capacity"],
#             )
#         ]
#     )
#
#     return fig
#
#
# def create_confusion_matrix(companies: pd.DataFrame):
#     actuals = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1]
#     predicted = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
#     data = {"y_Actual": actuals, "y_Predicted": predicted}
#     df = pd.DataFrame(data, columns=["y_Actual", "y_Predicted"])
#     confusion_matrix = pd.crosstab(
#         df["y_Actual"], df["y_Predicted"], rownames=["Actual"], colnames=["Predicted"]
#     )
#     sn.heatmap(confusion_matrix, annot=True)
#     return plt

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