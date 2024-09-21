import numpy as np
import pandas as pd
import wandb
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from plots import MLPlots
from pathlib import Path
from typing import Optional, Tuple

load_dotenv()
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

        self.plots = MLPlots()    

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
        
        # Ensure X_test is a NumPy array
        X_test_np = X_test.values.astype(np.float32)
        
        y_pred = self.model.predict(X_test_np)
        y_pred_proba = self.model.predict_proba(X_test_np)[:, 1]

        y_test_binary = np.where(y_test == 2, 1, 0)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, average='weighted')
        recall = recall_score(y_test_binary, y_pred, average='weighted')
        f1 = f1_score(y_test_binary, y_pred, average='weighted')
        auc_score = roc_auc_score(y_test_binary, y_pred_proba)
        cm = confusion_matrix(y_test_binary, y_pred)

        # Log evaluation metrics to W&B
        if self.wandb_project:
            wandb.log({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_auc": auc_score,
            })

        # Plot and log confusion matrix
        self.plots.plot_confusion_matrix(y_test_binary, y_pred, labels=[0, 1])
        plt.savefig("confusion_matrix.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

        # Plot and log ROC curve
        self.plots.plot_roc_curve(y_test_binary, y_pred_proba)
        plt.savefig("roc_curve.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"roc_curve": wandb.Image("roc_curve.png")})

        # Plot and log Precision-Recall curve
        self.plots.plot_precision_recall_curve(y_test_binary, y_pred_proba)
        plt.savefig("precision_recall_curve.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"precision_recall_curve": wandb.Image("precision_recall_curve.png")})

        # Plot and log Feature Importance
        self.plots.plot_feature_importance(self.model, X_test.columns)
        plt.savefig("feature_importance.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"feature_importance": wandb.Image("feature_importance.png")})

        # Plot and log Calibration Curve
        self.plots.plot_calibration_curve(y_test_binary, y_pred_proba)
        plt.savefig("calibration_curve.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"calibration_curve": wandb.Image("calibration_curve.png")})

        # Plot and log Cumulative Gain Chart
        self.plots.plot_cumulative_gain(y_test_binary, y_pred_proba)
        plt.savefig("cumulative_gain.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"cumulative_gain": wandb.Image("cumulative_gain.png")})

        # Plot and log Lift Curve
        self.plots.plot_lift_curve(y_test_binary, y_pred_proba)
        plt.savefig("lift_curve.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"lift_curve": wandb.Image("lift_curve.png")})

        return accuracy

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        X, y = self.load_data(data, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.train(X_train, y_train)
        accuracy = self.evaluate(X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

if __name__ == "__main__":
    data = pd.read_csv('data/data.csv')
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