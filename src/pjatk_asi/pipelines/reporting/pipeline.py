from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_confusion_matrix

def create_pipeline(**kwargs) -> Pipeline:
    """This is a simple pipeline which generates a pair of plots"""
    return pipeline(
        [
            node(
                func=create_confusion_matrix,
                inputs=["Y_predicted", "Y_test"],
                outputs="confusion_matrix",
            ),
        ]
    )
