from kedro.pipeline import Pipeline, node, pipeline

from .nodes import connect_project


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=connect_project,
                inputs=None,
                outputs='wandb_started',
                name="connect_wandb",
            )
        ]
    )
