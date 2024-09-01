import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class CatBoostPipeline:
    def __init__(self, iterations: int = 500, learning_rate: float = 0.1, depth: int = 6, 
                 random_state: int = 42, verbose: bool = True):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self.model = CatBoostClassifier(
            iterations=self.iterations, 
            learning_rate=self.learning_rate, 
            depth=self.depth, 
            random_state=self.random_state,
            verbose=self.verbose
        )

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

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
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
    data = pd.read_excel('data/pressure ulcer.xlsx')
    output_dim = len(data['caretaker score'].unique())  # Number of classes

    catboost_pipeline = CatBoostPipeline(
        iterations=500, 
        learning_rate=0.1, 
        depth=6, 
        random_state=42, 
        verbose=True
    )
    accuracy = catboost_pipeline.run(data, target_column='caretaker score')
