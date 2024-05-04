from kedro.pipeline import Pipeline, node, pipeline

from .nodes_caret import run_caret


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_caret,
                inputs="credit_data_with_removed_columns",
                outputs=None,
                name="run_caret_node",
            )
        ]
    )
