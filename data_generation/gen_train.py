from generate import Generate
import warnings
import pandas as pd
import os
import tensorflow as tf
from dotenv import load_dotenv
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  

load_dotenv()
warnings.filterwarnings("ignore")

def initialize_generator(data_path, config_path, gan):
    generator = Generate(data_path, config_path, gan)
    print("Numerical Columns:", generator.numerical_columns)
    print("Categorical Columns:", generator.categorical_columns)
    return generator

def train_gan(generator):
    generator.train_gan()

def generate_synthetic_data(generator, num_samples):
    synthetic_data = generator.generate_data(num_samples)
    print(synthetic_data)
    return synthetic_data

def save_synthetic_data(synthetic_data, save_directory, gan):
    file_name = f"{gan}.xlsx"
    save_path = os.path.join(save_directory, file_name)
    synthetic_data.to_excel(save_path, index=False)
    print(f"Synthetic data saved as {save_path}")

def main():
    data_path = Path(os.getenv('DATA_PATH'))
    config_path = Path(os.getenv('CONFIG_PATH'))
    save_directory = Path(os.getenv('SAVE_PATH'))
    gan = 'CTGAN'
    num_samples = 3000

    generator = initialize_generator(data_path, config_path, gan)
    train_gan(generator)
    synthetic_data = generate_synthetic_data(generator, num_samples)
    save_synthetic_data(synthetic_data, save_directory, gan)

if __name__ == "__main__":
    main()