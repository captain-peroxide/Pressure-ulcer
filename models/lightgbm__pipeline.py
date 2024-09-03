import lightgbm as lgb
import wandb
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, Dict
import pandas as pd
import os


class LightGBMPipeline:
    def __init__(self, param_grid: Dict, n_splits: int = 5, random_state: int = 42,
                 n_jobs: int = -1, verbosity: int = 2, wandb_project: Optional[str] = None,
                 wandb_entity: Optional[str] = None, wandb_api_key: Optional[str] = None) -> None:
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.model: Optional[GridSearchCV] = None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_initialized = False

        # Initialize WandB with API key
        if self.wandb_project and wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, reinit=True)
            self.wandb_initialized = True

    def load_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(target_column, axis=1)
        y = data[target_column]

        # Convert categorical variables to numeric using Label Encoding
        X = X.apply(self._encode_categorical)

        return X, y

    def _encode_categorical(self, col: pd.Series) -> pd.Series:
        if col.dtype == 'object':
            return LabelEncoder().fit_transform(col)
        return col

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def create_pipeline(self) -> None:
        lgbm = lgb.LGBMClassifier(random_state=self.random_state)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.model = GridSearchCV(lgbm, param_grid=self.param_grid, cv=cv, n_jobs=self.n_jobs, verbose=self.verbosity)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.model is None:
            self.create_pipeline()

        # Log model configuration to WandB if WandB is initialized
        if self.wandb_initialized:
            wandb.config.update({
                'param_grid': self.param_grid,
                'random_state': self.random_state,
                'n_splits': self.n_splits,
                'n_jobs': self.n_jobs,
                'verbosity': self.verbosity
            })

        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics to WandB if WandB is initialized
        if self.wandb_initialized:
            wandb.log({'accuracy': accuracy})

        return accuracy

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        X, y = self.load_data(data, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Train the model
        self.train(X_train, y_train)

        # Evaluate the model
        accuracy = self.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

        return accuracy


# Example usage
if __name__ == "__main__":
    data = pd.read_excel('pressure_ulcer.xlsx')

    # Initialize and run the LightGBM pipeline with WandB tracking
    lgbm_pipeline = LightGBMPipeline(
        param_grid={'num_leaves': [31, 63], 'learning_rate': [0.1, 0.01], 'n_estimators': [100, 200]},
        random_state=42,
        wandb_project='pressure_automl',
        wandb_entity=os.getenv('WANDB_ENTITY'),
        wandb_api_key=os.getenv('WANDB_API')
    )
    accuracy = lgbm_pipeline.run(data, target_column='caretaker score')
