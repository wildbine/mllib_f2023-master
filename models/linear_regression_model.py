import sys
import numpy as np

import utils.metrics
from configs.linear_regression_cfg import cfg
from logginig_example import generate_experiment_name
from utils.enums import TrainType
from logs.Logger import Logger
import cloudpickle


class LinearRegression():

    def __init__(self, base_functions: list, learning_rate: float, reg_coefficient: float, experiment_name: str):
        self.weights = np.random.randn(len(base_functions) + 1)
        self.base_functions = base_functions
        self.learning_rate = learning_rate
        self.reg_coefficient = reg_coefficient
        self.neptune_logger = Logger(cfg.env_path, cfg.project_name, experiment_name)
        experiment_name, base_function_str = generate_experiment_name(base_functions, reg_coefficient, learning_rate)
        self.neptune_logger.log_hyperparameters(params={
            'base_function': base_function_str,
            'regularisation_coefficient': reg_coefficient,
            'learning_rate': learning_rate
        })

    # Methods related to the Normal EquationD:\PyCharmProjects\pyProjects\mllib_f2023-master\venv\Scripts\activate.bat
    # pip install pandas~=1.3.5

    def _pseudoinverse_matrix(self, matrix: np.ndarray, l2_regularization: bool = False) -> np.ndarray:
        """Compute the pseudoinverse of a matrix using SVD.

        The pseudoinverse (Φ^+) of the design matrix Φ can be computed using the formula:

        Φ^+ = V * Σ^+ * U^T

        Where:
        - U, Σ, and V are the matrices resulting from the SVD of Φ.

        The Σ^+ is computed as:

        Σ'_{i,j} =
        | 1/Σ_{i,j}, if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        and then:
        Σ^+ = Σ'^T

        where:
        - ε is the machine epsilon, which can be obtained in Python using:
            ε = sys.float_info.epsilon
        - N is the number of rows in the design matrix.
        - M is the number of base functions (without φ_0(x_i)=1).

        For regularisation

        Σ'_{i,j} =
        | Σ_{i,j}/(Σ_{i,j}ˆ2 + λ) , if Σ_{i,j} > ε * max(N, M+1) * max(Σ)
        | 0, otherwise

        Note that Σ'_[0,0] = 1/Σ_{i,j}

        TODO: Add regularisation
        """
        U, sigma, Vt = np.linalg.svd(matrix, full_matrices=False)
        epsilon = sys.float_info.epsilon

        max_singular_value = max(sigma)
        threshold = epsilon * max(matrix.shape) * max_singular_value

        sigma_inv = np.zeros_like(sigma)
        for i, singular_value in enumerate(sigma):
            if l2_regularization:
                # L2 регуляризация (Tikhonov regularization)
                regularization_term = self.reg_coefficient
            else:
                # Без регуляризации (0 регуляризация)
                regularization_term = 0.0

            if singular_value > threshold:
                sigma_inv[i] = 1.0 / (singular_value + regularization_term)

        pseudo_inverse = Vt.T @ np.diag(sigma_inv) @ U.T

        return pseudo_inverse

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """Calculate the optimal weights using the normal equation.

            The weights (w) can be computed using the formula:

            w = Φ^+ * t

            Where:
            - Φ^+ is the pseudoinverse of the design matrix and can be defined as:
                Φ^+ = (Φ^T * Φ)^(-1) * Φ^T

            - t is the target vector.

            TODO: Implement this method. Calculate  Φ^+ using _pseudoinverse_matrix function
        """
        self.weights = pseudoinverse_plan_matrix @ targets

    # General methods
    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """Construct the design matrix (Φ) using base functions.

            The structure of the matrix Φ is as follows:

            Φ = [ [ φ_0(x_1), φ_1(x_1), ..., φ_M(x_1) ],
                  [ φ_0(x_2), φ_1(x_2), ..., φ_M(x_2) ],
                  ...
                  [ φ_0(x_N), φ_1(x_N), ..., φ_M(x_N) ] ]

            where:
            - x_i denotes the i-th input vector.
            - φ_j(x_i) represents the j-th base function applied to the i-th input vector.
            - M is the total number of base functions (without φ_0(x_i)=1).
            - N is the total number of input vectors.

            TODO: Implement this method using one loop over the base functions.

        """
        N, D = np.asarray(inputs).shape  # Получаем размеры входных данных
        M = len(self.base_functions) + 1  # Получаем количество базисных функций + 1

        # Создаем пустую матрицу плана Φ с размерами (N, M)
        design_matrix = np.zeros((N, M))

        # Заполняем первый столбец матрицы плана Φ со значениями 1 (φ_0(x_i) = 1)
        design_matrix[:, 0] = 1

        # Заполняем остальные столбцы матрицы плана Φ с использованием базовых функций
        for j, func in enumerate(self.base_functions):
            design_matrix[:, j + 1] = np.apply_along_axis(func, 1, inputs)

        return design_matrix

    def calculate_model_prediction(self, plan_matrix: np.ndarray) -> np.ndarray:
        """Calculate the predictions of the model.

            The prediction (y_pred) can be computed using the formula:

            y_pred = Φ * w^T

            Where:
            - Φ is the design matrix.
            - w^T is the transpose of the weight vector.

            To compute multiplication in Python using numpy, you can use:
            - `numpy.dot(a, b)`
            OR
            - `a @ b`

        TODO: Implement this method without using loop

        """
        y_pred = plan_matrix @ self.weights.T
        return y_pred

    # Methods related to Gradient Descent
    def _calculate_gradient(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the cost function with respect to the weights.

            The gradient of the error with respect to the weights (∆w E) can be computed using the formula:

            ∆w E = (2/N) * Φ^T * (Φ * w - t)

            Where:
            - Φ is the design matrix.
            - w is the weight vector.
            - t is the vector of target values.
            - N is the number of data points.

            This formula represents the partial derivative of the mean squared error with respect to the weights.

            For regularisation
            ∆w E = (2/N) * Φ^T * (Φ * w - t)  + λ * w

            TODO: Implement this method using matrix operations in numpy. a.T - transpose. Do not use loops
            TODO: Add regularisation
            """
        N = len(targets)  # Получаем количество образцов
        M = plan_matrix.shape[1]  # Получаем количество базовых функций (M)
        # Вычисляем ошибку (residuals) как разницу между предсказаниями и целевыми значениями
        errors = plan_matrix @ self.weights.T - targets

        # Вычисляем градиент без регуляризации
        gradient = (2 / N) * plan_matrix.T @ errors

        # Добавляем компоненту регуляризации, если reg_coefficient не равен нулю
        if self.reg_coefficient != 0:
            gradient[1:M + 1] += 2 * self.reg_coefficient * self.weights[1:M + 1]

        return gradient

    def calculate_cost_function(self, plan_matrix, targets) -> float:
        """Calculate the cost function value for the current weights.

        The cost function E(w) represents the mean squared error and is given by:

        E(w) = (1/N) * ∑(t - Φ * w^T)^2

        Where:
        - Φ is the design matrix.
        - w is the weight vector.
        - t is the vector of target values.
        - N is the number of data points.

        For regularisation
        E(w) = (1/N) * ∑(t - Φ * w^T)^2 + λ * w^T * w


        TODO: Implement this method using numpy operations to compute the mean squared error. Do not use loops
        TODO: Add regularisation

        """
        N = len(targets)  # Получаем количество образцов

        # Вычисляем ошибку как разницу между предсказаниями и целевыми значениями
        errors = targets - plan_matrix @ self.weights.T

        # Вычисляем сумму квадратов ошибок (mse) без регуляризации
        mse = (1 / N) * np.sum(errors ** 2)
        mse = np.clip(mse, -1e10, 1e10)

        # Вычисляем компоненту регуляризации
        regularization_term = self.reg_coefficient * self.weights.T @ self.weights

        # Общая функция стоимости с учетом регуляризации
        cost = mse + regularization_term

        return cost

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """Train the model using either the normal equation or gradient descent based on the configuration.
        TODO: Complete the training process.
        """
        train_type = TrainType(cfg.train_type)
        train_type_name = train_type.name
        plan_matrix = self._plan_matrix(inputs)
        if cfg.train_type == TrainType.normal_equation:
            pseudoinverse_plan_matrix = self._pseudoinverse_matrix(plan_matrix, True)
            self._calculate_weights(pseudoinverse_plan_matrix, targets)
            # cost = self.calculate_cost_function(plan_matrix, targets)
            # self.neptune_logger.save_param('normal_equation', 'cost_function', cost)
            cost = self.calculate_cost_function(plan_matrix, targets)
            self.neptune_logger.save_param(str(train_type_name), 'mse',
                                           utils.metrics.MSE(self.__call__(inputs), targets))
            self.neptune_logger.save_param(str(train_type_name), 'cost_function', cost)
        else:
            for e in range(cfg.epoch):
                gradient = self._calculate_gradient(plan_matrix, targets)
                self.weights -= self.learning_rate * gradient
                # Calculate and print the cost function's value
                cost = self.calculate_cost_function(plan_matrix, targets)
                self.neptune_logger.save_param(str(train_type_name), 'mse',
                                               utils.metrics.MSE(self.__call__(inputs), targets))
                self.neptune_logger.save_param(str(train_type_name), 'cost_function', cost)
        self.neptune_logger.log_final_val_mse(utils.metrics.MSE(self.__call__(inputs), targets))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)
        return predictions

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['neptune_logger']
        return state

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return cloudpickle.load(f)
