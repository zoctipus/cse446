if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)

def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    y_train = df_train["ViolentCrimesPerPop"].values
    X_train = df_train.drop("ViolentCrimesPerPop", axis=1).values
    y_test = df_test["ViolentCrimesPerPop"].values
    X_test = df_test.drop("ViolentCrimesPerPop", axis=1).values
    
    # Define parameters for the train function
    convergence_delta = 1e-4
    start_weight = None
    start_bias = None

    # Calculate lambda_max
    lambda_max = 2 * np.max(np.abs(X_train.T @ (y_train - np.mean(y_train))))
   
    # Initialize lambda and create a list to store non-zero counts
    _lambda = lambda_max
    non_zero_counts = []
    lambda_values = []
    mse_train_values = []
    mse_test_values = []
    L_weight = None

    while _lambda>0.01:
        lambda_values.append(_lambda)
        learned_weight, learned_bias = train(X_train, y_train, _lambda=_lambda, convergence_delta=convergence_delta, start_weight=start_weight, start_bias=start_bias, eta= 0.00001)
        if L_weight is None:
            L_weight = np.expand_dims(learned_weight, axis=1)
          
        else:
            L_weight = np.concatenate((L_weight, np.expand_dims(learned_weight, axis=1)), axis=1)
       

        # Count the number of non-zero weights
        non_zero_count = np.count_nonzero(np.abs(learned_weight) > 1e-6)
        non_zero_counts.append(non_zero_count)

        # Calculate the mean squared error for the training data
        y_train_pred = X_train @ learned_weight + learned_bias
        mse_train = np.mean((y_train - y_train_pred) ** 2)
        mse_train_values.append(mse_train)

        # Calculate the mean squared error for the test data
        y_test_pred = (X_test @ learned_weight) + learned_bias
        mse_test = np.mean((y_test - y_test_pred) ** 2)
        mse_test_values.append(mse_test)
    
    
        

        # Decrease lambda by a constant ratio
        _lambda /= 2

    learned_weight_30, learned_bias_30 = train(X_train, y_train, _lambda=30, convergence_delta=convergence_delta, start_weight=start_weight, start_bias=start_bias, eta= 0.00001)
    A = learned_weight_30
    # print(A)
    i_max = np.argmax(A)
    i_min = np.argmin(A)

    col_name_max = df_train.columns[i_max]
    col_name_min = df_train.columns[i_min]
    
    print("max", col_name_max)
    print("min", col_name_min)     
   

   #Plot 1
    plt.figure(figsize=(10, 5))
    plt.plot(lambda_values, non_zero_counts, marker="o", linestyle="-")
    plt.xlabel("Lambda")
    plt.ylabel("Number of Non-Zero Weights")
    plt.xscale("log")
    plt.title("Number of Non-Zeros vs. Lambda")
    plt.show()

    i1 = df_train.columns.get_loc("agePct12t29") - 1
    i2 = df_train.columns.get_loc("pctWSocSec") - 1
    i3 = df_train.columns.get_loc("pctUrban") - 1
    i4 = df_train.columns.get_loc("agePct65up") - 1
    i5 = df_train.columns.get_loc("householdsize") - 1
    k = len(lambda_values)


    #Plot 2
    plt.figure(figsize=(10,5))
    plt.plot(lambda_values, np.reshape(L_weight[i1, :], (k, )), lambda_values, np.reshape(L_weight[i2, :], (k,)), lambda_values, np.reshape(L_weight[i3, :], (k,)), lambda_values, np.reshape(L_weight[i4, :], (k,)), lambda_values, np.reshape(L_weight[i5, :], (k,)),marker="o")
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient Value')
    plt.title('Regularization Paths')
    plt.legend(["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"])
    plt.show()

    #Plot 3
    plt.figure(figsize=(10, 5))
    plt.plot(lambda_values, mse_train_values, marker="o", linestyle="-", label="Train MSE")
    plt.plot(lambda_values, mse_test_values, marker="o", linestyle="-", label="Test MSE")
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error")
    plt.xscale("log")
    plt.title("Mean Squared Error vs Lambda")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
