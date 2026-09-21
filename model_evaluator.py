import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class ModelEvaluator:
    def __init__(self, f1_threshold: float = 0.65, experiment_name: str = "Credit Score Pipeline"):
        self.f1_threshold = f1_threshold
        self.experiment_name = experiment_name
        self.artifact_dir = Path(__file__).parent / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        current_dir = Path(__file__).parent.resolve()
        mlflow_uri = f"sqlite:///{current_dir.as_posix()}/mlflow.db"
        mlflow.set_tracking_uri(mlflow_uri)
        
        artifact_loc = Path(current_dir, "mlruns").as_uri()
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name, artifact_location=artifact_loc)
            
        mlflow.set_experiment(self.experiment_name)
        
    def evaluate_and_log(self, models_dict: dict, x_test: pd.DataFrame, y_test: pd.Series):
        best_model_name = None
        best_model = None
        best_f1 = -1.0
        metrics_dict = {}

        print("Evaluasi Model")
        for name, model in models_dict.items():
            preds = model.predict(x_test)
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="macro", zero_division=0)
            rec = recall_score(y_test, preds, average="macro", zero_division=0)
            f1 = f1_score(y_test, preds, average="macro")
            
            metrics_dict[name] = {
                "accuracy": acc, 
                "precision": prec, 
                "recall": rec, 
                "f1_macro": f1
            }
            
            print(f"[{name}] F1 Macro = {f1:.4f} | Accuracy = {acc:.4f} | Recall = {rec:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = name

        print("\nModel Gate Decision")
        if best_f1 >= self.f1_threshold:
            print(f"Model terbaik adalah '{best_model_name}' dengan F1 Macro {best_f1:.4f} >= batas ambang {self.f1_threshold}")
            
            with mlflow.start_run() as run:
                mlflow.log_param("best_model_type", best_model_name)
                mlflow.log_param("threshold", self.f1_threshold)
                
                mlflow.log_metrics(metrics_dict[best_model_name])
                
                model_path = self.artifact_dir / "best_model.joblib"
                joblib.dump(best_model, model_path, compress=3)
                mlflow.sklearn.log_model(best_model, "best_model")
                
                print(f"Model dan metrik tercatat pada MLflow dengan Run ID: {run.info.run_id}")
                print(f"Model disimpan secara lokal di: {model_path}")
        else:
            print(f"Rejected: Performa model terbaik '{best_model_name}' ({best_f1:.4f}) berada di bawah ambang {self.f1_threshold}.")
