# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.

from utils.enums import TrainType

if __name__ == '__main__':
    from datasets.linear_regression_dataset import LinRegDataset
    from models.linear_regression_model import LinearRegression
    from configs.linear_regression_cfg import cfg
    import random
    import numpy as np
    from logginig_example import generate_experiment_name

    """lin_reg_dataset = LinRegDataset(inputs_cols=['x_0', 'x_1', 'x_2'], target_cols='targets')
    reg_coeff = random.uniform(0, 1)
    learning_rate = random.uniform(0, 1)

    base_functions = random.sample(cfg.base_functions, (np.asarray(lin_reg_dataset.training_inputs)).shape[1])

    lin_regression = LinearRegression(base_functions, learning_rate, reg_coeff, "1st")
    lin_regression.train(np.asarray(lin_reg_dataset.training_inputs),np.asarray(lin_reg_dataset.training_targets))
    """
    models = []

    best_valid_mse = 1e1000
    best_number_valid_mse = 0

    num_models = 30
    for j in range(num_models):
        # подбираем гиперпараметры
        learning_rate = np.random.uniform(0.001, 0.01)
        reg_coefficient = np.random.uniform(0.001, 0.01)
        cfg.epoch = np.random.randint(1000, 10000)
        train_types = [TrainType.gradient_descent, TrainType.normal_equation]
        cfg.train_type = np.random.choice(train_types)
        total_percent = 1
        cfg.train_set_percent = np.random.uniform(0.5, 0.8)
        cfg.valid_set_percent = total_percent - cfg.train_set_percent - 0.1

        lin_reg_dataset = LinRegDataset(inputs_cols=['x_0', 'x_1', 'x_2'], target_cols='targets')

        base_functions = cfg.base_functions

        experiment_name, base_function_str = generate_experiment_name(base_functions, reg_coefficient, learning_rate)

        model = LinearRegression(base_functions, learning_rate, reg_coefficient, experiment_name)

        model.train(np.asarray(lin_reg_dataset.training_inputs), np.asarray(lin_reg_dataset.training_targets))

        models.append(model)

        valid_predictions = model.calculate_model_prediction(model._plan_matrix(lin_reg_dataset.valid_inputs))
        valid_targets = np.asarray(lin_reg_dataset.valid_targets)
        mse = np.mean((valid_predictions - valid_targets) ** 2)
        if mse < best_valid_mse - np.finfo(np.float64).eps:
            best_valid_mse = mse
            best_number_valid_mse = j
        model.neptune_logger.save_param('valid', 'mse', mse)
        model.neptune_logger.save_param('valid', 'cost_function',
                                        model.calculate_cost_function(
                                            model._plan_matrix(
                                                lin_reg_dataset.valid_inputs), lin_reg_dataset.valid_targets))
    models[best_number_valid_mse].save('saved_models/' + str(best_valid_mse))
