import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import numpy as np
import pandas as pd
import wandb
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from plots import MLPlots

class SVMPipeline:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
                 tol=1e-3, cache_size=200, verbose=False, max_iter=-1, random_state=None,
                 wandb_project=None, wandb_entity=None, wandb_api_key: Optional[str] = None) -> None:
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_api_key = wandb_api_key

        # Initialize the SVR model
        self.model = SVR(
            C=self.C, 
            kernel=self.kernel, 
            degree=self.degree, 
            gamma=self.gamma, 
            coef0=self.coef0,
            shrinking=self.shrinking, 
            tol=self.tol, 
            cache_size=self.cache_size,
            verbose=self.verbose, 
            max_iter=self.max_iter
        )

        # Initialize W&B if a project name is provided
        if self.wandb_project:
            wandb.login(key=self.wandb_api_key)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

        # Initialize the MLPlots class
        self.plots = MLPlots()

    def load_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(X, y, test_size=0.2, random_state=self.random_state)

    def preprocess_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        # Identify categorical and numerical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X_train.select_dtypes(include=['number']).columns

        # Create a column transformer with OneHotEncoder for categorical features and StandardScaler for numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )

        # Fit and transform the training data, transform the test data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after transformation
        feature_names = numerical_cols.tolist() + preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()

        return X_train_processed, X_test_processed, feature_names

    def train(self, X_train: np.ndarray, y_train: pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: pd.Series, feature_names: list) -> float:
        y_pred = self.model.predict(X_test)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log evaluation metrics to W&B
        if self.wandb_project:
            wandb.log({
                "test_mae": mae,
                "test_mse": mse,
                "test_r2": r2,
            })

        # Plot and log residuals
        self.plots.plot_residuals(y_test, y_pred)
        plt.savefig("residuals.png")
        plt.close()
        if self.wandb_project:
            wandb.log({"residuals": wandb.Image("residuals.png")})


        return r2

    def run(self, data: pd.DataFrame, target_column: str) -> float:
        X, y = self.load_data(data, target_column)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled, feature_names = self.preprocess_data(X_train, X_test)
        
        # Debugging: Print shapes of the data
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        self.train(X_train_scaled, y_train.values)
        r2 = self.evaluate(X_test_scaled, y_test.values, feature_names)
        print(f"Test R2: {r2:.4f}")
        return r2

# Example usage
if __name__ == "__main__":
    data_path = 'data/final.csv'
    data = pd.read_csv(data_path)

    svm_pipeline = SVMPipeline(
        C=1.0, 
        kernel='rbf', 
        degree=3, 
        gamma='scale', 
        coef0=0.0, 
        shrinking=True,
        tol=1e-3, 
        cache_size=200, 
        verbose=False,
        max_iter=-1, 
        random_state=42,
        wandb_project='pressure_svm', 
        wandb_entity=os.getenv('WANDB_ENTITY'),
        wandb_api_key=os.getenv('WANDB_API')  # Your WandB API key here
    )
    r2 = svm_pipeline.run(data, target_column='Pressure_Ulcer_Probability')