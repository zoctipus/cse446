if __name__ == "__main__":
    from ISTA import train  # type: ignore
    from ISTA import calc_lambda
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    train_x = df_train.drop('ViolentCrimesPerPop', axis=1).values  # Replace 'target_column' with the actual target column name
    train_y = df_train['ViolentCrimesPerPop'].values  # Replace 'target_column' with the actual target column name

    d = train_x.shape[1]  # Number of features
    n = train_x.shape[0]  # Number of observations

    # Initialize weights and other parameters
    w = np.zeros(d,)

    X = train_x.to_numpy()
    y = train_y.to_numpy()

    #train
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X_ = (X - x_mean) / x_std
    ws = np.empty((0, w.shape[0]))
    bs = np.empty((0, 1))
    _lambda = calc_lambda(X_, y)
    ws_zeros = []
    lams = []
    zero_percentage = 1
    with open('/media/octipus/ad098048-2ec2-425c-b95b-4940e9cd3d83/446/hw2/homeworks/lasso/crime_data_progress.txt', 'w') as f:
        f.write("Lambda, Zero_Percentage,FDR, TPR\n")  # Writing the headers

        while zero_percentage > 0.001:
            lams.append(_lambda)
            w_, b_ = train(X_, y, _lambda)
            ws=np.vstack([ws, w_])
            bs=np.vstack([bs, b_])

            zero_percentage = np.count_nonzero(w_ == 0) / w_.shape[0]
            ws_zeros.append(zero_percentage)

            # Write the current _lambda and zero_percentage to file
            f.write(f"{_lambda}, {zero_percentage}\n")

            # Update _lambda for next iteration
            _lambda /= 2

if __name__ == "__main__":
    main()
