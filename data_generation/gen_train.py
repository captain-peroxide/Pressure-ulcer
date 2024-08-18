from generate import Generate
import warnings
import pandas as pd

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

gan_name = generator.get_gan_name()  
file_name = f"{gan_name}.xlsx"    
synthetic_data.to_excel(file_name, index=False)
# synthetic_data.to_csv(f"{gan_name}.csv", index=False)

print(f"Synthetic data saved as {file_name}")