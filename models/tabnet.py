import numpy as np
import pandas as pd
import wandb
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, Tuple

class TabNetPipeline:
    def __init__(self, n_d: int = 8, n_a: int = 8, n_steps: int = 3, gamma: float = 1.3, 
                 lambda_sparse: float = 1e-3, random_state: int = 42, 
                 wandb_project: Optional[str] = None, wandb_entity: Optional[str] = None, 
                 wandb_api_key: Optional[str] = None) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.random_state = random_state
        self.model: Optional[TabNetClassifier] = None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        if self.wandb_project and wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, reinit=True)

    def load_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        data[categorical_columns] = data[categorical_columns].astype(str)
        data = pd.get_dummies(data, columns=categorical_columns)
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def create_model(self) -> None:
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            seed=self.random_state
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.model is None:
            self.create_model()
        if self.wandb_project:
            wandb.config.update({
                'n_d': self.n_d,
                'n_a': self.n_a,
                'n_steps': self.n_steps,
                'gamma': self.gamma,
                'lambda_sparse': self.lambda_sparse,
                'random_state': self.random_state
            })
        self.model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.float32))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        y_pred = self.model.predict(X_test.values.astype(np.float32))
        accuracy = accuracy_score(y_test, y_pred)
        if self.wandb_project:
            wandb.log({'accuracy': accuracy})
        return accuracy

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        X, y = self.load_data(data, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.train(X_train, y_train)
        accuracy = self.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

if __name__ == "__main__":
    data = pd.read_excel('data/pressure ulcer.xlsx')
    tabnet_pipeline = TabNetPipeline(
        n_d=8, 
        n_a=8, 
        n_steps=3, 
        gamma=1.3, 
        lambda_sparse=1e-3, 
        random_state=42,
        wandb_project='pressure_tabnet', 
        wandb_entity=os.getenv('WANDB_ENTITY'),
        wandb_api_key=os.getenv('WANDB_API')
    )
    accuracy = tabnet_pipeline.run(data, target_column='caretaker score')