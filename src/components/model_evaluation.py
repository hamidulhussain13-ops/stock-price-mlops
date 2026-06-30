import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluation:
    def __init__(self, model_dir="artifacts"):
        self.model_dir = model_dir

    def evaluate(self, models, X_test, y_test):
        report = {}

        for name, model in models.items():
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            report[name] = {
                "mse": mse,
                "mae": mae,
                "r2_score": r2
            }

        return report

    def save_report(self, report):
        report_path = os.path.join(self.model_dir, "evaluation_report.json")

        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Evaluation report saved at {report_path}")