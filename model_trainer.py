import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_random_forest(self, x_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Tuple[Pipeline, dict, float]:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1))
        ])
        
        param_grid = {
            'model__n_estimators': [50, 100, 150],
            'model__max_depth': [10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='f1_macro',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        search.fit(x_train, y_train)
        return search.best_estimator_, search.best_params_, search.best_score_

    def train_xgboost(self, x_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> Tuple[Pipeline, dict, float]:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBClassifier(num_class=3, random_state=self.random_state, n_jobs=-1, eval_metric='mlogloss', verbosity=0))
        ])
        
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [4, 6, 8],
            'model__learning_rate': [0.01, 0.05, 0.1],
        }
        
        sample_weights = compute_sample_weight('balanced', y_train)
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
            scoring='f1_macro',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(x_train, y_train, model__sample_weight=sample_weights)
        return search.best_estimator_, search.best_params_, search.best_score_
