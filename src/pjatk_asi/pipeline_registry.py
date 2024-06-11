from typing import Dict
from kedro.pipeline import Pipeline
from .pipelines import data_processing as dp
from .pipelines import data_science as mt
from .pipelines import reporting as rep
from .pipelines import scoring as sp
from .pipelines import init_run as ir

def register_pipelines() -> Dict[str, Pipeline]:
    data_processing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    reporting_pipeline = rep.create_pipeline()
    scoring_pipeline = sp.create_pipeline()
    init = ir.create_pipeline()

    return {
        "dp": data_processing_pipeline,
        "mt": model_training_pipeline,
        "rep": reporting_pipeline,
        "sp": scoring_pipeline,
        "__default__": init + data_processing_pipeline + model_training_pipeline + reporting_pipeline
    }