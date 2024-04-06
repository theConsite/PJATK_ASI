from kedro.pipeline import Pipeline, node, pipeline

from .nodes_feature_engineering import feature_generation, remove_columns
from .nodes_load_data import get_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_data,
                inputs=None,
                outputs="credit_data",
                name="get_data_node",
            ),
            node(
                func=feature_generation,
                inputs="credit_data",
                outputs="credit_data_with_features",
                name="feature_generation_node",
            ),
            node(
                func=remove_columns,
                inputs="credit_data_with_features",
                outputs="credit_data_with_removed_columns",
                name="remove_columns_node",
            ),
        ]
    )
