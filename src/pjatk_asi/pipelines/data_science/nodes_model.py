import json
import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, recall_score
from kedro_datasets.pandas import JSONDataset

import wandb


def _transform_params(json_params):
    # Add 'clf__' prefix to each key in the dictionary
    param_grid = {f'clf__{key}': value for key, value in json_params.items()}
    return param_grid


def select_hiperparameters(X: pd.DataFrame, Y: pd.DataFrame, model_class: str, hyperparameters_path):

    print(hyperparameters_path)
    print(model_class)

    if model_class == "GradientBoostingClassifier":
        model_type = GradientBoostingClassifier
    elif model_class == "RandomForestClassifier":
        model_type = RandomForestClassifier
    elif model_class == "LogisticRegression":
        model_type = LogisticRegression
    elif model_class == "KNeighborsClassifier":
        model_type = KNeighborsClassifier
    else:
        raise ValueError(f"Unsupported model class: {model_class}")

    pipeline = Pipeline([
        ('clf', model_type())
    ])

    # Read JSON from file
    with open(hyperparameters_path, 'r') as file:
        json_params = json.load(file)

    # Transform the parameters
    param_grid = _transform_params(json_params)

    # # Define parameter grid for GridSearchCV
    # param_grid = {
    #     'clf__n_estimators': [50, 100, 150, 200],
    #     'clf__learning_rate': [0.01, 0.1, 0.2],
    #     'clf__max_depth': [3, 4, 5, 6]
    # }
    # Create GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, Y)
    best_params = {k.replace('clf__', ''): v for k, v in grid_search.best_params_.items()}
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        wandb.log({
            'Mean': mean,
            'Stdev': stdev,
            'Params':param
        })
    logger = logging.getLogger(__name__)
    logger.info("Model has best parameters: %s", str(best_params))

    model = _train_final_model(X, Y, model_type, best_params)
    best_params['Model'] = model_class

    # Log the model to WandB
    wandb.log({'model': model})

    return model, pd.DataFrame([best_params])


def _train_final_model(X: pd.DataFrame, Y: pd.DataFrame, model, best_params: Dict[str, str]) -> GradientBoostingClassifier:
    final_model = model(**best_params)
    final_model.fit(X, Y)
    return final_model


def _get_metrics(Y_true: pd.DataFrame, Y_pred: pd.DataFrame):
    return {
        'roc_auc': roc_auc_score(Y_true, Y_pred),
        'accuracy_score': balanced_accuracy_score(Y_true, Y_pred),
        'recall_score': recall_score(Y_true, Y_pred),
    }


def evaluate_model(X: pd.DataFrame, Y: pd.DataFrame, model: GradientBoostingClassifier):
    Y_pred = model.predict(X)
    metrics = _get_metrics(Y, Y_pred)
    wandb.log(metrics)
    logger = logging.getLogger(__name__)
    logger.info("Final model score: %s", str(metrics))

    return Y_pred
