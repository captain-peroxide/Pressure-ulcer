import numpy as np
import pandas as pd
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Optional, Tuple

class AdvancedNeuralNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout_rate: float = 0.5):
        super(AdvancedNeuralNet, self).__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AdvancedNeuralNetPipeline:
    def __init__(self, hidden_dims: list, dropout_rate: float = 0.5,
                 batch_size: int = 32, epochs: int = 100, learning_rate: float = 0.001,
                 wandb_project: Optional[str] = None, wandb_entity: Optional[str] = None,
                 wandb_api_key: Optional[str] = None) -> None:
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_api_key = wandb_api_key

        self.model = None
        self.criterion = nn.MSELoss()
        self.optimizer = None

        if self.wandb_project and wandb_api_key:
            wandb.login(key=self.wandb_api_key)
            wandb.init(project=self.wandb_project, entity=self.wandb_entity)

    def load_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(X, y, test_size=0.2, random_state=42)

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

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        input_dim = X_train.shape[1]
        self.model = AdvancedNeuralNet(input_dim, self.hidden_dims, 1, self.dropout_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(train_loader)}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, feature_names: list) -> float:
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            y_pred_tensor = self.model(X_test_tensor)
            y_pred = y_pred_tensor.numpy().reshape(-1)

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

    neuralnet_pipeline = AdvancedNeuralNetPipeline(
        hidden_dims=[64, 32], 
        dropout_rate=0.5, 
        batch_size=32, 
        epochs=100, 
        learning_rate=0.001,
        wandb_project='pressure_neuralnet', 
        wandb_entity=os.getenv('WANDB_ENTITY'),
        wandb_api_key=os.getenv('WANDB_API')  # Your WandB API key here
    )
    r2 = neuralnet_pipeline.run(data, target_column='Pressure_Ulcer_Probability')