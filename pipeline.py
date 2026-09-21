from pathlib import Path
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

class Pipeline:
    """Master Pipeline orchestrator."""
    
    def __init__(self, raw_data_path: str, f1_threshold: float = 0.65):
        self.raw_data_path = Path(raw_data_path)
        self.f1_threshold = f1_threshold
        
        self.preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
        self.trainer = ModelTrainer(random_state=42)
        self.evaluator = ModelEvaluator(f1_threshold=self.f1_threshold)

    def execute(self):        
        print("\nSTEP 1: PREPROCESSING")
        x_train, x_test, y_train, y_test = self.preprocessor.clean_and_split(self.raw_data_path)
        transformer = self.preprocessor.get_transformer(x_train)
        
        print("\nSTEP 2: MODEL TRAINING")
        print("Training Random Forest (Class Weight)")
        rf_model, rf_params, rf_score = self.trainer.train_random_forest(x_train, y_train, transformer)
        
        models_dict = {
            "RandomForest_ClassWeight": rf_model
        }
        
        print("\nSTEP 3: EVALUATION")
        self.evaluator.evaluate_and_log(models_dict, x_test, y_test)
        
        print("\nEksekusi Pipeline Selesai")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_file = base_dir / "data_C.csv"
    
    if not data_file.exists():
        print(f"Error: Tidak bisa menemukan file di {data_file}.")
    else:
        pipeline = Pipeline(raw_data_path=str(data_file), f1_threshold=0.65)
        pipeline.execute()
