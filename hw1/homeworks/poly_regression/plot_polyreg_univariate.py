import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    # model = PolynomialRegression(degree=d, reg_lambda=0)
    # model.fit(X, y)

    # # output predictions
    # xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    # ypoints = model.predict(xpoints)

    # # plot curve
    # plt.figure()
    # plt.plot(X, y, "rx")
    # plt.title(f"PolyRegression with d = {d}")
    # plt.plot(xpoints, ypoints, "b-")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()
    lambdas = [0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-5, 5e-3, 1e-2, 5e-1, 10]

    n = len(lambdas)
    nrows = int(np.ceil(n / 2.0))
    ncols = 2 if n > 1 else 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5 * nrows))

    for idx, reg_lambda in enumerate(lambdas):
        ax = axes[idx // ncols, idx % ncols] if n > 1 else axes

        model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
        model.fit(X, y)

        # output predictions
        xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
        ypoints = model.predict(xpoints)

        # plot curve
        ax.plot(X, y, "rx")
        ax.set_title(f"reg_lambda = {reg_lambda}")
        ax.plot(xpoints, ypoints, "b-")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

    # for reg_lambda in lambdas:
    #     model = PolynomialRegression(degree=d, reg_lambda=reg_lambda)
    #     model.fit(X, y)

    #     # output predictions
    #     xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    #     ypoints = model.predict(xpoints)

    #     # plot curve
    #     plt.figure()
    #     plt.plot(X, y, "rx")
    #     plt.title(f"PolyRegression with d = {d} and reg_lambda = {reg_lambda}")
    #     plt.plot(xpoints, ypoints, "b-")
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.show()
