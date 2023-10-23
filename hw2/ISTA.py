from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
   
    common = y - (np.dot(X, weight) + bias)
    gradient_weight = -2 * X.T @ common
    gradient_bias = -2 * np.sum(common)
    weight_up = weight - eta * gradient_weight
    bias_up = bias - eta * gradient_bias
    weight_up = np.where(weight_up < -2 * eta * _lambda, weight_up + 2 * eta * _lambda, np.where(weight_up > 2 * eta * _lambda, weight_up - 2 * eta * _lambda, 0))
    return weight_up, bias_up
    


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function

    """
    
  
    l1_regularization = _lambda * np.sum(np.abs(weight))
   
    
    loss = np.mean(np.dot((np.dot(X,weight) - y + bias).T, np.dot(X, weight) - y + bias)) + l1_regularization

    return loss
   


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float =  0.000001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None, max_iterations: int = 2000
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (np.ndarray, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None

    
    iteration = 0
    while True:
        # Save a copy of old weights and bias
        old_w = np.copy(start_weight)
        old_b = start_bias
        
        # Perform one step of ISTA
        start_weight, start_bias = step(X, y, start_weight, start_bias, _lambda, eta)

        # Check for convergence
        if old_w is not None and old_b is not None:
            if convergence_criterion(start_weight, old_w, start_bias, old_b, convergence_delta):
                break

        iteration += 1
        if iteration >= max_iterations:
            break

    return start_weight, start_bias


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
   
    max_weight_change=np.max(np.abs(weight-old_w))
    bias_change=np.abs(bias-old_b)

    if (max_weight_change < convergence_delta).all() and (bias_change < convergence_delta).all():
        return True
    else:
        return False
    
def generate_data(n: int, d: int, k: int, noise_std: float) -> Tuple[np.ndarray, np.ndarray]:
   # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate the feature matrix X from a normal distribution with mean 0 and standard deviation 1
    X = np.random.normal(0, 1, (n, d))

    # Standardize the feature matrix X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # Generate true weights according to Equation (2)
    true_weight = np.zeros(d)
    true_weight[:k] = np.arange(1, k + 1) / k

    # Generate noise term (epsilon) from a normal distribution with mean 0 and standard deviation noise_std
    noise = np.random.randn(n) * noise_std

    # Generate target values (y) based on the model
    y = np.dot(X, true_weight) + noise

    return X, y, true_weight
    
    


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # Generate synthetic data
    n, d, k, noise_std = 500, 1000, 100, 1
    X, y, true_weight = generate_data(n, d, k, noise_std)

    # Define parameters for the train function
    convergence_delta = 1e-4
    start_weight = None
    start_bias = None

     # Calculate lambda_max
    lambda_max = 2 * np.max(np.abs(X.T @ (y - np.mean(y))))

    # Initialize lambda and create lists to store non-zero counts, FDR, and TPR values
    _lambda = lambda_max
    non_zero_counts = []
    FDR_values = []
    TPR_values = []

    # Decrease lambda by a constant ratio and solve Lasso problems
    while True:
        learned_weight, learned_bias = train(X, y, _lambda=_lambda, convergence_delta=convergence_delta, start_weight=start_weight, start_bias=start_bias, eta=0.00001)
        
        # Count the number of non-zero weights
        non_zero_count = np.count_nonzero(np.abs(learned_weight) > 1e-6)
        non_zero_counts.append(non_zero_count)

        # Calculate FDR and TPR
        incorrect_nonzeros = np.count_nonzero((learned_weight != 0) & (true_weight == 0))
        FDR = incorrect_nonzeros / non_zero_count if non_zero_count > 0 else 0
        TPR = np.count_nonzero((learned_weight != 0) & (true_weight != 0)) / k
        FDR_values.append(FDR)
        TPR_values.append(TPR)

        # Stop when nearly all features are chosen
        if non_zero_count >= 0.99 * d:
            break

        # Decrease lambda by a constant ratio
        _lambda /= 2

    # Plot the number of non-zeros as a function of lambda
    lambda_values = [lambda_max / (2 ** i) for i in range(len(non_zero_counts))]
    plt.figure(figsize=(10, 5))
    plt.plot(lambda_values, non_zero_counts, marker="o", linestyle="-")
    plt.xlabel("Lambda")
    plt.ylabel("Number of Non-Zero Weights")
    plt.xscale("log")
    plt.title("Number of Non-Zeros vs. Lambda")
    plt.show()

    # Plot FDR vs TPR
    plt.figure(figsize=(10, 5))
    plt.plot(FDR_values, TPR_values, marker="o", linestyle="-")
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.title("FDR vs. TPR")
    plt.show()



if __name__ == "__main__":
    main()


