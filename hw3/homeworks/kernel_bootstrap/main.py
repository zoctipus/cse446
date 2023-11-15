from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    # raise NotImplementedError("Your Code Goes Here")
    poly_ker = (np.multiply.outer(x_i, x_j) + 1)**d
    return poly_ker

@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    
    rbf_ker = np.exp(-gamma * np.subtract.outer(x_i, x_j) ** 2)
    return rbf_ker


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    # raise NotImplementedError("Your Code Goes Here")
    k = kernel_function(x, x, kernel_param)
    a = np.linalg.solve(k + _lambda * np.eye(len(k)), y)
    return a


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    total_mse = 0
    fold_size = len(x) // num_folds
    for i in range(num_folds):
        start, end = i * fold_size, (i + 1) * fold_size

        x_train = np.concatenate([x[:start], x[end:]])
        y_train = np.concatenate([y[:start], y[end:]]) 
        x_test = x[start:end]
        y_test = y[start:end]
        
        a = train(x_train, y_train, kernel_function, kernel_param, _lambda)
        y_predict = a @ kernel_function(x_train, x_test, kernel_param)
        
        total_mse += np.mean((y_test - y_predict) ** 2)
        
    return total_mse / num_folds


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # raise NotImplementedError("Your Code Goes Here")
    
    lambda_values= 10 ** np.linspace(-5, -1, 10)
    gamma = 1/np.median(np.subtract.outer(x, x) ** 2)
    lambda_mse = [(_lambda, gamma, cross_validation(x, y, rbf_kernel, gamma, _lambda, num_folds)) for _lambda in lambda_values]
    return min(lambda_mse, key= lambda item : item[1])[0:2]


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You can use gamma = 1 / median((x_i - x_j)^2) for all unique pairs x_i, x_j in x) for this problem. 
          However, if you would like to search over other possible values of gamma, you are welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """
    # raise NotImplementedError("Your Code Goes Here")
    lambda_values= 10 ** np.linspace(-5, -1, 10)
    print(lambda_values)
    degree_values = np.linspace(5, 25, 21)
    lambda_grid, degree_grid = np.meshgrid(lambda_values, degree_values)
    combos = list(zip(lambda_grid.flatten(), degree_grid.flatten()))
    lambda_mse = [(lambda_value, degree_value, cross_validation(x, y, poly_kernel, degree_value, lambda_value, num_folds)) for lambda_value, degree_value in combos]
    return min(lambda_mse, key=lambda item : item[2])[0:2]

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    poly_lambda, poly_degree = poly_param_search(x_30, y_30, num_folds=10)
    rbf_lambda, rbf_gamma = rbf_param_search(x_30, y_30, num_folds=10)
    
    print ((poly_lambda, poly_degree), (rbf_lambda, rbf_gamma))
    
def plot_data():
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    # rbf_lambda, rbf_gamma = rbf_param_search(x_30, y_30, num_folds=10)
    poly_a = train(x_30, y_30, poly_kernel, 19, 2.782559402207126e-05)
    rbf_a = train(x_30, y_30, rbf_kernel, 11.201924992299844, 1e-05)
    
    f_rbf = lambda x : rbf_a @ rbf_kernel(x_30, x, 11.201924992299844)
    f_poly = lambda x : poly_a @ poly_kernel(x_30, x, 19)
    
    
    # Fine grid for plotting
    fine_grid = np.linspace(0, 1, 100)
    
    # Evaluating functions on the grid
    rbf_predictions = f_rbf(fine_grid)
    poly_predictions = f_poly(fine_grid)
    true_values = f_true(fine_grid)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Original data
    plt.scatter(x_30, y_30, label='Original Data', color='black')

    # True function
    plt.plot(fine_grid, true_values, label='True Function', color='green')

    # RBF predictions
    plt.plot(fine_grid, rbf_predictions, label='RBF Kernel Predictions', color='blue')

    # Polynomial predictions
    # plt.plot(fine_grid, poly_predictions, label='Polynomial Kernel Predictions', color='red')

    plt.title('Kernel Regression Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # plot_data()
    main()
