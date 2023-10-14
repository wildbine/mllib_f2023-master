from easydict import EasyDict
from utils.enums import TrainType
import numpy as np

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = 'datasets/linear_regression_dataset_with_inputs_as_vectors.csv'

# cfg.base_functions contains callable functions to transform input features.
# E.g., for polynomial regression: [lambda x: x, lambda x: x**2]
# TODO You should populate this list with suitable functions based on the requirements.

sin_func = lambda x: np.sin(x)
cos_func = lambda x: np.cos(x)
log_func = lambda x: np.log(x)
exp_func = lambda x: np.exp(x)
polynomial_func = lambda x: 5 + x + x**2
def linear_basis(x):
    return x[0] + x[1] + x[2]

def weighted_sum_basis(x):
    return 2 * x[0] + 3 * x[1] - x[2]

def polynomial_basis(x):
    return x[0] + x[1]**2 - x[2]

def product_basis(x):
    return x[0] * x[1] * x[2]

def sine_cosine_basis(x):
    return np.sin(x[0]) + np.cos(x[1]) + np.sin(x[2])

def exponential_basis(x):
        return np.exp(x[0]) - np.exp(x[1]) + np.exp(x[2])

cfg.base_functions = [linear_basis, weighted_sum_basis, polynomial_basis, product_basis, sine_cosine_basis, exponential_basis]

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 100

#cfg.exp_name = ''
cfg.env_path = ''# Путь до файла .env где будет храниться api_token.
cfg.project_name = 'linear-regression'