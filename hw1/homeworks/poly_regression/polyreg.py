"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.polymean: np.ndarray = None
        self.polystd: np.ndarray = None
        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        # print("\n**** degree ****\n", degree, "\n******X value is:******\n", X)
        if degree != 0:
            result = np.hstack([X ** (i+1) for i in range(degree)])
        else:
            result = None
        return result

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # print("\n**** degree ****\n", self.degree)
        polyfeatures = self.polyfeatures(X, self.degree)
        # print("\n********polyfeature********\n")
        # print(polyfeatures)
        if(self.degree != 0 or polyfeatures is not None):
            self.polymean = np.mean(polyfeatures, axis=0)
            self.polystd = np.std(polyfeatures, axis=0)
            normalized_polyfeatures = (polyfeatures - self.polymean) / self.polystd
            normalized_polyfeatures= np.hstack([np.ones((len(normalized_polyfeatures), 1)), normalized_polyfeatures])
        else:
            normalized_polyfeatures = np.ones((len(X), 1))
        # print("\n********normalized polyfeature********\n")
        # print(normalized_polyfeatures)
        reg_matrix = self.reg_lambda * np.eye(normalized_polyfeatures.shape[1])
        reg_matrix[0, 0] = 0
        self.weight = np.linalg.solve(normalized_polyfeatures.T @ normalized_polyfeatures + reg_matrix,  normalized_polyfeatures.T @ y)
        # print("\n********self.weight ********\n")
        # print(self.weight)


    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        polyfeatures = self.polyfeatures(X, self.degree)
        if polyfeatures is not None:
            normalized_polyfeatures = (polyfeatures - self.polymean) / self.polystd
            normalized_polyfeatures = np.hstack([np.ones((len(normalized_polyfeatures), 1)), normalized_polyfeatures])
        else:
            normalized_polyfeatures = np.ones((len(X), 1))
        return normalized_polyfeatures @ self.weight


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return np.mean((a - b)**2)


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    '''
    The commented code shows how regularization affects the training as the complexity of function increases
    '''
    # n = len(Xtrain)
    # errorTrain = np.zeros(n)
    # errorTest = np.zeros(n)
    # # Fill in errorTrain and errorTest arrays
    # for i in range(degree+1):
    #     pr = PolynomialRegression(degree=i, reg_lambda=reg_lambda)
    #     pr.fit(X=Xtrain, y=Ytrain)
    #     errorTrain[i] = mean_squared_error(Ytrain, pr.predict(Xtrain))
    #     errorTest[i] = mean_squared_error(Ytest, pr.predict(Xtest))
    # print(errorTrain, errorTest)
    # return errorTrain,errorTest

    n = len(Xtrain)
    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    for i in range(2, n):
        Xtrain_ = Xtrain[0:i+1]
        Ytrain_ = Ytrain[0:i+1]
        pr = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        pr.fit(X=Xtrain_, y=Ytrain_)
        errorTrain[i] = mean_squared_error(Ytrain_, pr.predict(Xtrain_))
        errorTest[i] = mean_squared_error(Ytest, pr.predict(Xtest))
    return errorTrain,errorTest