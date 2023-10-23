from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import logging

from utils import problem

logging.basicConfig(level=logging.DEBUG)
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

    # Compute gradient for weight
    gradient =  2 * (X.T @ (X @ weight + bias - y))
    w_ = weight - eta * gradient
    for i in range(len(w_)):
        if w_[i] >  2 * eta * _lambda :
            w_[i] = w_[i] - 2 * eta * (_lambda)
        elif w_[i] <  -2 * eta * _lambda:
            w_[i] = w_[i] + 2 * eta * (_lambda)
        else:
            w_[i] = 0
    # Compute gradient for bias
    gradient_b = 2 * np.sum(X @ weight + bias - y)
    # Update bias
    b_ = bias - eta * gradient_b

    return w_, b_



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
    # raise NotImplementedError("Your Code Goes Here")
    J = np.sum((y - (X @ weight + bias)) ** 2) + _lambda * np.sum(np.abs(weight))
    return J

@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.000005,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
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
    current_b = np.copy(start_bias)
    current_w = np.copy(start_weight)
    while True:
        old_b = np.copy(current_b)
        old_w = np.copy(current_w)
        current_w, current_b = step(X, y ,weight=current_w, bias=current_b, _lambda = _lambda, eta = eta)
        if convergence_criterion(current_w, old_w, current_b, old_b, convergence_delta):
            break
    return current_w, current_b



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
    # Calculate the maximum absolute change in weights and bias
    max_weight_change = np.max(np.abs(weight - old_w))
    max_bias_change = np.abs(bias - old_b)

    # Check if the maximum changes are below the convergence threshold
    return max_weight_change < convergence_delta


def calc_lambda(X, y):
    y_ = (y - np.average(y)).reshape(-1, 1)
    return np.max(2 * np.abs(np.sum(y_ * X, axis=0)))

@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    k = 100
    d = 1000
    sigma = 1
    n = 500
    #data generation
    X = np.random.randn(n, d)
    w = np.zeros(d,)
    w[0:k] = w[0:k] + np.arange(1, k+1)/k
    y = X @ w + np.random.normal(0, sigma, n)

    #train
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X_ = (X - x_mean) / x_std
    ws = np.empty((0, w.shape[0]))
    bs = np.empty((0, 1))
    _lambda = calc_lambda(X_, y)
    ws_zeros = []
    lams = []
    FDRs = []
    TPRs = []
    zero_percentage = 1
    with open('/media/octipus/ad098048-2ec2-425c-b95b-4940e9cd3d83/446/hw2/homeworks/lasso/progress.txt', 'w') as f:
        f.write("Lambda, Zero_Percentage,FDR, TPR\n")  # Writing the headers

        while zero_percentage > 0.001:
            lams.append(_lambda)
            w_, b_ = train(X_, y, _lambda)
            ws=np.vstack([ws, w_])
            bs=np.vstack([bs, b_])

            FDR_ = np.count_nonzero(w_[k:d]) / (d-k)
            TPR_ = np.count_nonzero(w_[0:k]) / (k)

            zero_percentage = np.count_nonzero(w_ == 0) / w_.shape[0]
            ws_zeros.append(zero_percentage)
            FDRs.append(FDR_)
            TPRs.append(TPR_)

            # Write the current _lambda and zero_percentage to file
            f.write(f"{_lambda}, {zero_percentage}, {FDR_}, {TPR_}\n")

            # Update _lambda for next iteration
            _lambda /= 2


def graph_zero_percentage():
    import csv
    # Initialize empty lists to hold lambda and zero_percentage values
    lams = []
    zero_percentages = []
    # Read the progress.txt file
    with open('/media/octipus/ad098048-2ec2-425c-b95b-4940e9cd3d83/446/hw2/homeworks/lasso/progress.txt', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)  # Skip the header
        for row in csv_reader:
            _lambda, zero_percentage, _, _ = row
            lams.append(float(_lambda))
            zero_percentages.append(float(zero_percentage))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lams, zero_percentages)
    plt.xscale('log')  # Use a log scale for better visibility
    plt.title('Progress Plot')
    plt.xlabel('Lambda')
    plt.ylabel('Zero Percentage')
    # Reverse the x-axis to have the larger values on the left
    plt.xlim(max(lams), min(lams))

    # Save the plot
    plt.savefig('progress_plot.png', format='png', dpi=300)
    plt.show()

def graph_FDR_TPR():
    import csv
    # Initialize empty lists to hold lambda and zero_percentage values
    FDRs = []
    TPRs = []
    # Read the progress.txt file
    with open('/media/octipus/ad098048-2ec2-425c-b95b-4940e9cd3d83/446/hw2/homeworks/lasso/progress.txt', 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)  # Skip the header
        for row in csv_reader:
            _,_, FDR, TPR = row
            FDRs.append(float(FDR))
            TPRs.append(float(TPR))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(FDRs, TPRs)
    plt.title('TPR vs FDR')
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    # Reverse the x-axis to have the larger values on the left

    # Save the plot
    plt.savefig('/media/octipus/ad098048-2ec2-425c-b95b-4940e9cd3d83/446/hw2/homeworks/lasso/TPRvsFDR.png', format='png', dpi=300)
    plt.show()



if __name__ == "__main__":
    # main()
    graph_FDR_TPR()
    # graph_zero_percentage()