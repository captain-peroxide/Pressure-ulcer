import yaml
import pandas as pd
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
import warnings
warnings.filterwarnings("ignore")

class Generate:
    def __init__(self, data, config_path):
        self.data = data
        self.config_path = config_path
        self.gan = 'WGAN_GP'
        self.numerical_columns = []
        self.categorical_columns = []
        self.load_config()
        self.segregate_columns()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.model_parameters = config.get('model_config', {})
        self.train_parameters = config.get('train_config', {})

    def segregate_columns(self):
        df = pd.read_excel(self.data)
        self.numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    def train_gan(self):
        data = pd.read_excel(self.data)
        
        for col in self.categorical_columns:
            data[col] = data[col].astype(str)
        
        if self.gan == 'WGAN_GP':
            self.gan = RegularSynthesizer(modelname='wgangp', model_parameters=ModelParameters(batch_size = self.model_parameters['batch_size'], lr = self.model_parameters['learning_rate'], betas = (self.model_parameters['beta_1'], self.model_parameters['beta_2'])))
        else:
            raise ValueError(f"Unsupported GAN type: {self.gan}")
        
        self.gan.fit(data, TrainParameters(epochs = self.train_parameters['epochs']), num_cols=self.numerical_columns, cat_cols=self.categorical_columns)
        
        print(f"{self.gan} model trained successfully.")

    def generate_data(self, num_samples):
        if self.gan is None:
            raise ValueError("The GAN model has not been trained yet.")
        
        synthetic_data = self.gan.sample(num_samples)
        
        return synthetic_data
    
    def get_gan_name(self):
        return self.gan_name