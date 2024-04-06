from kedro.pipeline import Pipeline, node, pipeline

from .nodes_model import select_hiperparameters, evaluate_model
from .nodes_data_preparation import (data_split, train_test_split, balance,
                                     one_hot_encode_fit, one_hot_encode_apply,
                                     normalize_columns_fit, normalize_columns_apply)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=data_split,
                inputs="credit_data_with_removed_columns",
                outputs=["X", "Y"],
                name="data_split_node",
            ),
            node(
                func=train_test_split,
                inputs=["X", "Y"],
                outputs=["X_train", "X_test", "Y_train", "Y_test"],
                name="train_test_split_node",
            ),
            node(
                func=one_hot_encode_fit,
                inputs="X_train",
                outputs=["X_train_ohe", "one_hot_encoder"],
                name="one_hot_encode_train_data_node",
            ),
            node(
                func=normalize_columns_fit,
                inputs="X_train_ohe",
                outputs=["X_train_prepared", "std_scaler"],
                name="normalize_columns_train_data_node",
            ),
            node(
                func=balance,
                inputs=["X_train_prepared", "Y_train"],
                outputs=["X_train_balanced", "Y_train_balanced"],
                name="balance_node",
            ),
            node(
                func=one_hot_encode_apply,
                inputs=["X_test", "one_hot_encoder"],
                outputs="X_test_ohe",
                name="one_hot_encode_test_data_node",
            ),
            node(
                func=normalize_columns_apply,
                inputs=["X_test_ohe", "std_scaler"],
                outputs="X_test_prepared",
                name="normalize_columns_test_data_node",
            ),
            node(
                func=select_hiperparameters,
                inputs=["X_train_balanced", "Y_train_balanced"],
                outputs="model",
                name="select_hiperparameters_node",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test_prepared", "Y_test", "model"],
                outputs="Y_predicted",
                name="evaluate_model_node",
            ),
        ]
    )
