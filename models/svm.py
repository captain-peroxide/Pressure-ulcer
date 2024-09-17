import numpy as np
import pandas as pd
import wandb
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from dotenv import load_dotenv
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
load_dotenv()

class SVMPipeline:
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, random_state: int = 42, 
                 n_jobs: int = -1, wandb_project: Optional[str] = None, 
                 wandb_entity: Optional[str] = None, wandb_api_key: Optional[str] = None) -> None:
        """
        Initialize the SVM pipeline with sklearn and WandB.
        
        Parameters:
        - kernel: Kernel type to be used in the SVM algorithm
        - C: Regularization parameter
        - random_state: Random seed for reproducibility
        - n_jobs: Number of jobs to run in parallel (-1 uses all processors)
        - wandb_project: WandB project name for tracking experiments
        - wandb_entity: WandB entity name for tracking experiments
        - wandb_api_key: WandB API key for logging in programmatically
        """
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: Optional[SVC] = None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # Initialize WandB with API key
        if self.wandb_project and wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, reinit=True)

    def load_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the dataset and split into features and target.
        
        Parameters:
        - data: Pandas DataFrame containing the dataset
        - target_column: Name of the column to be used as the target variable
        
        Returns:
        - Tuple of feature matrix X and target vector y
        """
        # Identify categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        # Convert categorical columns to string type
        data[categorical_columns] = data[categorical_columns].astype(str)
        
        # Apply one-hot encoding
        data = pd.get_dummies(data, columns=categorical_columns)
        
        # Split into features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the dataset into training and testing sets.
        
        Parameters:
        - X: Feature matrix
        - y: Target vector
        - test_size: Proportion of the dataset to include in the test split
        
        Returns:
        - Tuple of training and testing feature matrices and target vectors
        """
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state)

    def create_model(self) -> None:
        """Initialize the SVM model with the provided configuration."""
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            random_state=self.random_state
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the SVM model using the training data.
        
        Parameters:
        - X_train: Training feature matrix
        - y_train: Training target vector
        """
        if self.model is None:
            self.create_model()
        # Log model configuration to WandB
        if self.wandb_project:
            wandb.config.update({
                'kernel': self.kernel,
                'C': self.C,
                'random_state': self.random_state
            })
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Evaluate the trained model on the test data.
        
        Parameters:
        - X_test: Testing feature matrix
        - y_test: Testing target vector
        
        Returns:
        - Accuracy of the model on the test set
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        
        # Log metrics to WandB
        if self.wandb_project:
            wandb.log({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1,
                "test_auc": auc,
            })

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()    
        
        return accuracy

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        """
        Run the entire SVM pipeline from data loading to evaluation.
        
        Parameters:
        - data: Pandas DataFrame containing the dataset
        - target_column: Name of the column to be used as the target variable
        
        Returns:
        - Accuracy of the model on the test set
        """
        # Load and split the data
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
    data = pd.read_csv('data/data.csv')
    kernels =['rbf','linear','poly','sigmoid']
    # Initialize and run the SVM pipeline with WandB tracking
    for ker in kernels:
        svm_pipeline = SVMPipeline(
            kernel=ker, 
            C=1.0, 
            random_state=42,
            wandb_project='pressure_svm', 
            wandb_entity=os.getenv('WANDB_ENTITY'),
            wandb_api_key=os.getenv('WANDB_API') # Your WandB API key here
        )
        accuracy = svm_pipeline.run(data, target_column='caretaker score')
        
    
