# pipelines/scoring/pipeline.py
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import predict, load_data, load_artifacts
from ..data_processing.nodes_feature_engineering import feature_generation, remove_columns
from ..data_science.nodes_data_preparation import one_hot_encode_apply, normalize_columns_apply

def create_pipeline(**kwargs) -> Pipeline:
    from ..data_processing.pipeline import create_pipeline as data_processing_pipeline

    return Pipeline(
        [
            node(
                func=load_data,
                inputs="params:raw_data_path",
                outputs="score_data",
                name="load_raw_data"
            ),
            node(
                func=feature_generation,
                inputs="score_data",
                outputs="score_data_with_features",
                name="feature_generation_node",
            ),
            node(
                func=remove_columns,
                inputs="score_data_with_features",
                outputs="score_data_with_removed_columns",
                name="remove_columns_node",
            ),
            node(
                func=load_artifacts,
                inputs="params:model_folder",
                outputs=["one_hot_encoder_loaded", "std_scaler_loaded", "model"],
                name="load_encoders",
            ),
            node(
                func=one_hot_encode_apply,
                inputs=["score_data_with_removed_columns", "one_hot_encoder_loaded"],
                outputs="score_ohe",
                name="one_hot_encode_test_data_node",
            ),
            node(
                func=normalize_columns_apply,
                inputs=["score_ohe", "std_scaler_loaded"],
                outputs="score_ohe_prepared",
                name="normalize_columns_test_data_node",
            ),
            node(
                func=predict,
                inputs=["model", "score_ohe_prepared", "params:model_folder"],
                outputs="score_result",
                name="make_predictions",
            )
        ]
    )