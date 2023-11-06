from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

# When choosing your batches / Shuffling your data you should use this RNG variable, and not `np.random.choice` etc.
RNG = np.random.RandomState(seed=446)
Dataset = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def load_2_7_mnist() -> Dataset:
    """
    Loads MNIST data, extracts only examples with 2, 7 as labels, and converts them into -1, 1 labels, respectively.

    Returns:
        Dataset: 2 tuples of numpy arrays, each containing examples and labels.
            First tuple is for training, while second is for testing.
            Shapes as as follows: ((n, d), (n,)), ((m, d), (m,))
    """
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    train_idxs = np.logical_or(y_train == 2, y_train == 7)
    test_idxs = np.logical_or(y_test == 2, y_test == 7)

    y_train_2_7 = y_train[train_idxs]
    y_train_2_7 = np.where(y_train_2_7 == 7, 1, -1)

    y_test_2_7 = y_test[test_idxs]
    y_test_2_7 = np.where(y_test_2_7 == 7, 1, -1)

    return (x_train[train_idxs], y_train_2_7), (x_test[test_idxs], y_test_2_7)


class BinaryLogReg:
    @problem.tag("hw2-A", start_line=4)
    def __init__(self, _lambda: float = 1e-3):
        """Initializes the Binary Log Regression model.
        Args:
            _lambda (float, optional): Ridge Regularization coefficient. Defaults to 1e-3.
        """
        self._lambda: float = _lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        self.bias: float = 0.0
        # raise NotImplementedError("Your Code Goes Here")

    @problem.tag("hw2-A")
    def mu(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate mu in vectorized form, as described in the problem.
        The equation for i^th element of vector mu is given by:

        $$ \mu_i = 1 / (1 + \exp(-y_i (bias + x_i^T weight))) $$

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.

        Returns:
            np.ndarray: An `(n, )` vector containing mu_i for i^th element.
        """
        return 1 / (1 + np.exp(-y * (self.bias + X @ self.weight)))


    @problem.tag("hw2-A")
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate loss J as defined in the problem.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.

        Returns:
            float: Loss given X, y, self.weight, self.bias and self._lambda
        """
        '''
        Octi Edit Begins
        '''
        # raise NotImplementedError("Your Code Goes Here")
        loss = 1/y.shape[0]  * np.sum(np.log(1 + np.exp(-y * (self.bias + X @ self.weight))))  + self._lambda * np.sum(self.weight ** 2)
        return loss
        '''
        Octi Edit Ends
        '''

    @problem.tag("hw2-A")
    def gradient_J_weight(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate gradient of loss J with respect to weight.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.
        Returns:
            np.ndarray: An `(d, )` vector which represents gradient of loss J with respect to self.weight.
        """
        gradient_term = 1/y.shape[0] * np.sum(- y.reshape(-1, 1) * X *(1 - self.mu(X, y)).reshape(-1, 1), axis=0)
        regularization_term = self._lambda * 2 * self.weight
        return gradient_term + regularization_term

    @problem.tag("hw2-A")
    def gradient_J_bias(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate gradient of loss J with respect to bias.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.

        Returns:
            float: A number that represents gradient of loss J with respect to self.bias.
        """
        gradient_term_b = 1/y.shape[0] * np.sum(-y * (1 - self.mu(X, y)))
        return gradient_term_b


    @problem.tag("hw2-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Given X, weight and bias predict values of y.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.

        Returns:
            np.ndarray: An `(n, )` array of either -1s or 1s representing guess for each observation.
        """
        probabilities = sigmoid(X @ self.weight + self.bias)
        predictions = np.where(probabilities >= 0.5, 1, -1)
        return predictions

    @problem.tag("hw2-A")
    def misclassification_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculates misclassification error (the rate at which this model is making incorrect predictions of y).
        Note that `misclassification_error = 1 - accuracy`.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.

        Returns:
            float: percentage of times prediction did not match target, given an observation (i.e. misclassification error).
        """
        y_hat = self.predict(X)
        misclassification_error_rate = np.sum(y_hat != y) / y.shape[0]
        return misclassification_error_rate

    @problem.tag("hw2-A")
    def step(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 1e-4):
        """Single step in training loop.
        It does not return anything but should update self.weight and self.bias with correct values.

        Args:
            X (np.ndarray): observations represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y (np.ndarray): targets represented as `(n, )` vector.
                n is number of observations.
            learning_rate (float, optional): Learning rate of SGD/GD algorithm.
                Defaults to 1e-4.
        """
        w_ = self.gradient_J_weight(X, y)
        b_ = self.gradient_J_bias(X, y)

        self.weight -= learning_rate * w_
        self.bias -= learning_rate * b_


    @problem.tag("hw2-A", start_line=7)
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 100,
    ) -> Dict[str, List[float]]:
        """Train function that given dataset X_train and y_train adjusts weights and biases of this model.
        It also should calculate misclassification error and J loss at the END of each epoch.

        For each epoch please call step function `num_batches` times as defined on top of the starter code.

        NOTE: This function due to complexity and number of possible implementations will not be publicly unit tested.
        However, we might still test it using gradescope, and you will be graded based on the plots that are generated using this function.

        Args:
            X_train (np.ndarray): observations in training set represented as `(n, d)` matrix.
                n is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y_train (np.ndarray): targets in training set represented as `(n, )` vector.
                n is number of observations.
            X_test (np.ndarray): observations in testing set represented as `(m, d)` matrix.
                m is number of observations, d is number of features.
                d = 784 in case of MNIST.
            y_test (np.ndarray): targets in testing set represented as `(m, )` vector.
                m is number of observations.
            learning_rate (float, optional): Learning rate of SGD/GD algorithm. Defaults to 1e-2.
            epochs (int, optional): Number of epochs (loops through the whole data) to train SGD/GD algorithm for.
                Defaults to 30.
            batch_size (int, optional): Number of observation/target pairs to use for a single update.
                Defaults to 100.

        Returns:
            Dict[str, List[float]]: Dictionary containing 4 keys, each pointing to a list/numpy array of length `epochs`:
            {
                "training_losses": [<Loss at the end of each epoch on training set>],
                "training_errors": [<Misclassification error at the end of each epoch on training set>],
                "testing_losses": [<Same as above but for testing set>],
                "testing_errors": [<Same as above but for testing set>],
            }
            Skeleton for this result is provided in the starter code.

        Note:
            - When shuffling batches/randomly choosing batches makes sure you are using RNG variable defined on the top of the file.
        """
        num_batches = int(np.ceil(len(X_train) // batch_size))
        result: Dict[str, List[float]] = {
            "train_losses": [],  # You should append to these lists
            "train_errors": [],
            "test_losses": [],
            "test_errors": [],
        }

        self.weight = np.zeros((X_train.shape[1]))
        self.bias = 0

        for epoch in range(epochs):
            # Shuffle the data
            indices = RNG.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Iterate through mini-batches
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_train))
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                self.step(X_batch, y_batch, learning_rate)
            result["train_losses"].append(self.loss(X_batch, y_batch))
            result["train_errors"].append(self.misclassification_error(X_batch, y_batch))
            result["test_losses"].append(self.loss(X_test, y_test))
            result["test_errors"].append(self.misclassification_error(X_test, y_test))
            print(epoch)
        return result


def sigmoid(x):
        return 1 / (1 + np.exp(-x))


def compare_learning_rate():
    learning_rates = [5e-1, 1e-1, 5e-2]
    (x_train, y_train), (x_test, y_test) = load_2_7_mnist()

    max_loss = float('-inf')
    min_loss = float('inf')
    max_error = float('-inf')
    min_error = float('inf')

    # Create the plots
    fig, axes = plt.subplots(2, len(learning_rates), figsize=(15, 10))

    for i, lr in enumerate(learning_rates):
        model = BinaryLogReg()
        history = model.train(x_train, y_train, x_test, y_test, learning_rate=lr)
        max_loss = max(max_loss, max(history["train_losses"]), max(history["test_losses"]))
        min_loss = min(min_loss, min(history["train_losses"]), min(history["test_losses"]))
        max_error = max(max_error, max(history["train_errors"]), max(history["test_errors"]))
        min_error = min(min_error, min(history["train_errors"]), min(history["test_errors"]))

        # Plot loss
        axes[0, i].plot(history["train_losses"], label="Train")
        axes[0, i].plot(history["test_losses"], label="Test")
        axes[0, i].set_title(f"Loss with Learning Rate {lr}")
        axes[0, i].set_xlabel("Epochs")
        axes[0, i].set_ylabel("Loss")
        axes[0, i].legend()
        axes[0, i].set_ylim([min_loss, max_loss])

        # Plot error
        axes[1, i].plot(history["train_errors"], label="Train")
        axes[1, i].plot(history["test_errors"], label="Test")
        axes[1, i].set_title(f"Error with Learning Rate {lr}")
        axes[1, i].set_xlabel("Epochs")
        axes[1, i].set_ylabel("Misclassification Error")
        axes[1, i].legend()
        axes[1, i].set_ylim([min_error, max_error])

    plt.tight_layout()
    plt.show()

import matplotlib.gridspec as gridspec
def plot_images_and_boundary():
    learning_rate = 5e-1
    (x_train, y_train), (x_test, y_test) = load_2_7_mnist()

    model = BinaryLogReg()
    history = model.train(x_train, y_train, x_test, y_test, learning_rate)

    X, w, b = (x_test, model.weight, model.bias)
    scores = sigmoid(np.dot(X, w) + b)

    # Filter scores less than 0.1
    valid_indices = np.where(scores >= 0.1)
    scores = scores[valid_indices]
    X = X[valid_indices]

    indices = np.linspace(0, len(X) - 1, 100).astype(int)
    sampled_images = X[indices]
    sampled_scores = scores[indices]

    sort_indices = np.argsort(sampled_scores)
    sampled_images = sampled_images[sort_indices]
    sampled_scores = sampled_scores[sort_indices]

    # Create a mapping of score to images, to manage the vertical stacking
    score_image_map = {}
    for img, score in zip(sampled_images, sampled_scores):
        rounded_score = round(score, 2)
        score_image_map.setdefault(rounded_score, []).append(img)

    # Initialize plot with GridSpec
    fig = plt.figure(figsize=(100, 6))
    gs = gridspec.GridSpec(1, len(sampled_scores))

    col_idx = 0
    for score, imgs in score_image_map.items():
        # Only the first image for each score will be shown
        img = imgs[0]
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title(f'{score:.2f}', fontsize=10)
        ax.axis('off')
        col_idx += 1

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    compare_learning_rate()
