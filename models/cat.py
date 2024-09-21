import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import numpy as np
import pandas as pd
import wandb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from plots import MLPlots
import seaborn as sns

class CatBoostPipeline:
    def __init__(self, iterations: int = 1000, learning_rate: float = 0.05, depth: int = 8, 
                 random_state: int = 42, verbose: bool = True, early_stopping_rounds: Optional[int] = 50,
                 wandb_project: Optional[str] = None, wandb_entity: Optional[str] = None) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        
        # Initialize the CatBoostClassifier with the evaluation metric set to 'Accuracy'
        self.model = CatBoostClassifier(
            iterations=self.iterations, 
            learning_rate=self.learning_rate, 
            depth=self.depth, 
            random_state=self.random_state,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric='Accuracy'  # Ensure accuracy is tracked
        )

        # Initialize W&B if a project name is provided
        if self.wandb_project:
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, config={
                'iterations': self.iterations,
                'learning_rate': self.learning_rate,
                'depth': self.depth,
                'random_state': self.random_state,
                'early_stopping_rounds': self.early_stopping_rounds
            })
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

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        train_pool = Pool(data=X_train, label=y_train)
        self.model.fit(train_pool)

        # Debug: Print available metrics in best_score_
        print("Available metrics:", self.model.best_score_)
        
        # Log train accuracy if available
        if 'Accuracy' in self.model.best_score_['learn']:
            wandb.log({"train_accuracy": self.model.best_score_['learn']['Accuracy']})
        else:
            print("Accuracy metric is not available in best_score_. Available metrics:", self.model.best_score_['learn'])

        # Log feature importance
        feature_importances = self.model.get_feature_importance(train_pool)
        wandb.log({"feature_importance": wandb.Histogram(feature_importances)})

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, X_test_df: pd.DataFrame) -> float:
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

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
        self.plots.plot_confusion_matrix(y_test, y_pred, labels=self.model.classes_)
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
        self.plots.plot_feature_importance(self.model, X_test_df.columns)
        plt.savefig("feature_importance.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"feature_importance": wandb.Image("feature_importance.png")})

        self.plots.plot_calibration_curve(y_test_binary, y_pred_proba)
        plt.savefig("calibration_curve.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"calibration_curve": wandb.Image("calibration_curve.png")})

        self.plots.plot_cumulative_gain(y_test_binary, y_pred_proba)
        plt.savefig("cumulative_gain.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"cumulative_gain": wandb.Image("cumulative_gain.png")})

        self.plots.plot_lift_curve(y_test_binary, y_pred_proba)
        plt.savefig("lift_curve.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"lift_curve": wandb.Image("lift_curve.png")})
        

        return accuracy

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        X, y = self.load_data(data, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        
        
        self.train(X_train_scaled, y_train.values)
        accuracy = self.evaluate(X_test_scaled, y_test.values, X_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

# Example usage
if __name__ == "__main__":
    data_path = 'data/data.csv'
    data = pd.read_csv(data_path)

    catboost_pipeline = CatBoostPipeline(
        iterations=1000, 
        learning_rate=0.001, 
        depth=10, 
        random_state=42, 
        verbose=True,
        early_stopping_rounds=50,
        wandb_project='pressure_cat', 
        wandb_entity=os.getenv('WANDB_ENTITY')
    )

    accuracy = catboost_pipeline.run(data, target_column='caretaker score')
