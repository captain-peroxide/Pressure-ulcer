from generate import Generate
import warnings
warnings.filterwarnings("ignore")


data_path = r'D:\Pressure-ulcer\data\pressure ulcer.xlsx'  
config_path = r'D:\Pressure-ulcer\config\gen_config.yaml'  

generator = Generate(data_path, config_path)

print("Numerical Columns:", generator.numerical_columns)
print("Categorical Columns:", generator.categorical_columns)


generator.train_gan()


num_samples = 100  
synthetic_data = generator.generate_data(num_samples)

print(synthetic_data)