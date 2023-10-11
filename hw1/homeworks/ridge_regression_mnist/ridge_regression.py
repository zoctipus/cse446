import numpy as np
import matplotlib.pyplot as plt
from utils import load_dataset, problem

def preprocess(x:np.ndarray):
   x_ = (x - 0.5) * 2
   return x_


@problem.tag("hw1-A")
def train(x: np.ndarray, y: np.ndarray, _lambda: float) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), targets (`y`) and regularization parameter (`_lambda`)
    to train a weight matrix $$\\hat{W}$$.


    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        y (np.ndarray): targets represented as `(n, k)` matrix.
            n is number of observations, k is number of classes.
        _lambda (float): parameter for ridge regularization.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: weight matrix of shape `(d, k)`
            which minimizes Regularized Squared Error on `x` and `y` with hyperparameter `_lambda`.
    """

    x_ = preprocess(x)
    ridge = _lambda * np.eye(x.shape[1])
    # print(ridge.shape)
    weight = np.linalg.solve(x_.T @ x_ + ridge, x_.T @ y)
    # print(weight.shape)
    return weight


@problem.tag("hw1-A")
def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Train function for the Ridge Regression problem.
    Should use observations (`x`), and weight matrix (`w`) to generate predicated class for each observation in x.

    Args:
        x (np.ndarray): observations represented as `(n, d)` matrix.
            n is number of observations, d is number of features.
        w (np.ndarray): weights represented as `(d, k)` matrix.
            d is number of features, k is number of classes.

    Raises:
        NotImplementedError: When problem is not attempted.

    Returns:
        np.ndarray: predictions matrix of shape `(n,)` or `(n, 1)`.
    """
    x_ = preprocess(x)
    _y = x_ @ w
    # print("pre process y prediction:", _y.shape)
    y = np.argmax(_y, axis=1)
    # print("after process y prediction:",y.shape)
    return y



@problem.tag("hw1-A")
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encode a vector `y`.
    One hot encoding takes an array of integers and coverts them into binary format.
    Each number i is converted into a vector of zeros (of size num_classes), with exception of i^th element which is 1.

    Args:
        y (np.ndarray): An array of integers [0, num_classes), of shape (n,)
        num_classes (int): Number of classes in y.

    Returns:
        np.ndarray: Array of shape (n, num_classes).
        One-hot representation of y (see below for example).

    Example:
        ```python
        > one_hot([2, 3, 1, 0], 4)
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        ```
    """
    encoding = np.zeros((len(y),num_classes))
    encoding[np.arange(len(y)), y] = 1

    return encoding


def main():

    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    # Convert to one-hot
    y_train_one_hot = one_hot(y_train.reshape(-1), 10)

    _lambda = 1e-4

    w_hat = train(x_train, y_train_one_hot, _lambda)

    y_train_pred = predict(x_train, w_hat)
    y_test_pred = predict(x_test, w_hat)

    print("Ridge Regression Problem")
    print(
        f"\tTrain Error: {np.average(1 - np.equal(y_train_pred, y_train)) * 100:.6g}%"
    )
    print(f"\tTest Error:  {np.average(1 - np.equal(y_test_pred, y_test)) * 100:.6g}%")

    unequal_indices = np.where(y_test != y_test_pred)[0]
    random_indices = np.random.choice(unequal_indices, size=min(10, len(unequal_indices)), replace=False)
    showIncorrect(x_test[random_indices], y_test_pred[random_indices])



def showIncorrect(x: np.ndarray, labels):
    m, _ = x.shape  # m is the number of images
    sqrt_m = int(np.ceil(np.sqrt(m)))  # This will help in determining the grid size for subplots

    # Create a figure
    plt.figure(figsize=(10, 10))

    # Loop through each image and display it
    for idx in range(m):
        plt.subplot(sqrt_m, sqrt_m, idx + 1)  # Create a subplot for each image
        plt.imshow(x[idx].reshape(28, 28), cmap='gray')  # Display the image
        plt.title(str(labels[idx]), fontsize=16)
        plt.axis('off')  # Hide axes for clarity

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
