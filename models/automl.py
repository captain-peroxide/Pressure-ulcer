import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import numpy as np
import pandas as pd
import wandb
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from dotenv import load_dotenv
from typing import Optional, Tuple
from plots import MLPlots
import matplotlib.pyplot as plt
import seaborn as sns
load_dotenv()

class AutoMLPipeline:
    def __init__(self, generations: int = 10, population_size: int = 100, random_state: int = 42, 
                 n_jobs: int = -1, verbosity: int = 2, wandb_project: Optional[str] = None, 
                 wandb_entity: Optional[str] = None, wandb_api_key: Optional[str] = None) -> None:
        """
        Initialize the AutoML pipeline with TPOT and WandB.
        
        Parameters:
        - generations: Number of generations to run the genetic algorithm
        - population_size: Number of pipelines to retain in each generation
        - random_state: Random seed for reproducibility
        - n_jobs: Number of jobs to run in parallel (-1 uses all processors)
        - verbosity: Level of information TPOT prints during the process
        - wandb_project: WandB project name for tracking experiments
        - wandb_entity: WandB entity name for tracking experiments
        - wandb_api_key: WandB API key for logging in programmatically
        """
        self.generations = generations
        self.population_size = population_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.model: Optional[TPOTClassifier] = None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # Initialize WandB with API key
        if self.wandb_project and wandb_api_key:
            wandb.login(key=wandb_api_key)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity, reinit=True)

        self.plots = MLPlots()    

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

    def create_pipeline(self) -> None:
        """Initialize the TPOTClassifier with the provided configuration."""
        self.model = TPOTClassifier(
            verbosity=self.verbosity,
            generations=self.generations,
            population_size=self.population_size,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the AutoML model using the training data.
        
        Parameters:
        - X_train: Training feature matrix
        - y_train: Training target vector
        """
        if self.model is None:
            self.create_pipeline()
        # Log model configuration to WandB
        if self.wandb_project:
            wandb.config.update({
                'generations': self.generations,
                'population_size': self.population_size,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbosity': self.verbosity
            })
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
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
        '''self.plots.plot_feature_importance(self.model, X_test.columns)
        plt.savefig("feature_importance.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"feature_importance": wandb.Image("feature_importance.png")})'''

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


    def export_pipeline(self, file_name: str = 'best_pipeline.py') -> None:
        """
        Export the best pipeline found by TPOT.
        
        Parameters:
        - file_name: The name of the file to save the pipeline
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        
        self.model.export(file_name)

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        """
        Run the entire AutoML pipeline from data loading to evaluation.
        
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
        accuracy = self.evaluate(X_train, y_train, X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

        # Export the pipeline
        self.export_pipeline()

        return accuracy

# Example usage
if __name__ == "__main__":

    data = pd.read_csv('data\data.csv')

    # Initialize and run the AutoML pipeline with WandB tracking
    automl = AutoMLPipeline(
        generations=5, 
        population_size=20, 
        random_state=42,
        wandb_project='pressure_automl', 
        wandb_entity=os.getenv('WANDB_ENTITY') ,
        wandb_api_key=os.getenv('WANDB_API')  # Your WandB API key here
    )
    accuracy = automl.run(data, target_column='caretaker score')
