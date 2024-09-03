import os
import numpy as np
import pandas as pd
import wandb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import matplotlib.pyplot as plt
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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Log evaluation metrics to W&B
        if self.wandb_project:
            wandb.log({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_auc": auc,
            })

        # Plot and log confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()

        return accuracy

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        X, y = self.load_data(data, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.preprocess_data(X_train, X_test)
        
        # Debugging: Print shapes of the data
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        self.train(X_train_scaled, y_train.values)
        accuracy = self.evaluate(X_test_scaled, y_test.values)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

# Example usage
if __name__ == "__main__":
    data_path = 'C:/Users/91932/Downloads/Pressure-ulcer-main1/Pressure-ulcer-main/data_generation/pressure ulcer.xlsx'
    data = pd.read_excel(data_path)

    catboost_pipeline = CatBoostPipeline(
        iterations=1000, 
        learning_rate=0.05, 
        depth=8, 
        random_state=42, 
        verbose=True,
        early_stopping_rounds=50,
        wandb_project='pressure_automl', 
        wandb_entity=os.getenv('WANDB_ENTITY')
    )

    accuracy = catboost_pipeline.run(data, target_column='caretaker score')
