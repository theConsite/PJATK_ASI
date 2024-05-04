import pandas as pd
from pycaret.classification import ClassificationExperiment
import wandb


def run_caret(fraud_df: pd.DataFrame):
    s = ClassificationExperiment()
    s.setup(fraud_df, target = 'is_fraud', profile = True)
    best_model = s.compare_models()
    # s.plot_model(best_model, plot = 'auc')
    # s.plot_model(best_model, plot = 'confusion_matrix')
    predictions = s.predict_model(best_model)
    wandb.log({"predictions": predictions.head().to_json()})
    saved_model_path = s.save_model(best_model, 'best_pipeline')
    wandb.save(saved_model_path)
