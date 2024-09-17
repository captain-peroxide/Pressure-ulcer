import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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
        # Removed the softmax layer since CrossEntropyLoss includes that
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AdvancedNeuralNetPipeline:
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, 
                 learning_rate: float = 0.001, batch_size: int = 32, epochs: int = 20, 
                 dropout_rate: float = 0.5, random_state: int = 42) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.model = AdvancedNeuralNet(input_dim, hidden_dims, output_dim, dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def load_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        data[categorical_columns] = data[categorical_columns].astype(str)
        data = pd.get_dummies(data, columns=[col for col in categorical_columns if col != target_column])
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
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)  # Removed .data
                all_preds.extend(predicted.numpy())

        accuracy = accuracy_score(y_test, all_preds)
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
        
        self.train(X_train_scaled, y_train.to_numpy())  # Changed to .to_numpy()
        accuracy = self.evaluate(X_test_scaled, y_test.to_numpy())  # Changed to .to_numpy()
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

# Example usage
if __name__ == "__main__":
    data = pd.read_csv('data/data.csv')
    input_dim = data.shape[1] - 1  # Number of features
    hidden_dims = [128, 64, 32]  # List of hidden layer dimensions
    output_dim = len(data['caretaker score'].unique())  # Number of classes

    neural_net_pipeline = AdvancedNeuralNetPipeline(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        output_dim=output_dim, 
        learning_rate=0.001, 
        batch_size=32, 
        epochs=20, 
        dropout_rate=0.5, 
        random_state=42
    )
    accuracy = neural_net_pipeline.run(data, target_column='caretaker score')
