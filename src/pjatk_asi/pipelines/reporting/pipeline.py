from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_confusion_matrix, create_metrics, save_model_to_WandB

def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=create_confusion_matrix,
                inputs=["Y_predicted", "Y_test"],
                outputs="confusion_matrix",
            ),
            node(
                func=create_metrics,
                inputs=["Y_predicted", "Y_test"],
                outputs="metrics",
            ),
            node(
                func=save_model_to_WandB,
                inputs=["model", "metrics", "selected_hyperparamters"],
                outputs=None,
            ),
        ]
    )
