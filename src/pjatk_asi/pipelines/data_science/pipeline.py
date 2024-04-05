from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_hiperparameters, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_hiperparameters,
                inputs=["X_train_balanced", "Y_train_balanced"],
                outputs="model",
                name="select_hiperparameters_node",
            ),
            node(
                func=evaluate_model,
                inputs=["X_test_prepared", "Y_test", "model"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
