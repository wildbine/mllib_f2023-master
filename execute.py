# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.

if __name__ == '__main__':
    from datasets.linear_regression_dataset import LinRegDataset
    from models.linear_regression_model import LinearRegression
    from configs.linear_regression_cfg import cfg
    import random
    import numpy as np

    lin_reg_dataset = LinRegDataset(inputs_cols=['x_0', 'x_1', 'x_2'], target_cols='targets')
    reg_coeff = random.uniform(0, 1)
    learning_rate = random.uniform(0, 1)
    """
    либо я обрезаю количество базисных функций здесь, либо обрезаю их на этапе создания матрицы плана
    пожалуй, в методе _plan_matrix я также позабочусь об этом, если вдруг захотят передать все
    базисные функции
    """
    base_functions = random.sample(cfg.base_functions, (np.asarray(lin_reg_dataset.training_inputs)).shape[1])

    lin_regression = LinearRegression(base_functions, learning_rate, reg_coeff, "1st")
    lin_regression.train(np.asarray(lin_reg_dataset.training_inputs),np.asarray(lin_reg_dataset.training_targets))
