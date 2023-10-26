if __name__ == "__main__":
    from ISTA import train  # type: ignore
    from ISTA import calc_lambda
    from ISTA import loss
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np
import json

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    feature_indices_to_names = {i: feature for i, feature in enumerate(df_train.columns)}
    with open('homeworks/lasso/feature_indices_to_names.json', 'w') as f:
        json.dump(feature_indices_to_names, f)

    train_x = df_train.drop('ViolentCrimesPerPop', axis=1).values
    train_y = df_train['ViolentCrimesPerPop'].values

    test_x = df_test.drop('ViolentCrimesPerPop', axis=1).values
    test_y = df_test['ViolentCrimesPerPop'].values

    d = train_x.shape[1]  # Number of features
    n = train_x.shape[0]  # Number of observations

    # Initialize weights and other parameters
    w = np.zeros(d,)

    X = train_x
    y = train_y

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X_ = (X - x_mean) / x_std
    test_X_ = (test_x - x_mean) / x_std
    ws = np.empty((0, w.shape[0]))
    bs = np.empty((0, 1))
    _lambda = calc_lambda(X_, y)
    ws_zeros = []
    lams = []
    zero_percentage = 1
    data_to_write = []
    with open('homeworks/lasso/crime_data_progress.json', 'w') as f:

        while _lambda > 0.01:
            lams.append(_lambda)
            # print(_lambda)
            startw = None
            startb = None
            if(len(ws) != 0):
                startw = ws[-1]
                startb = bs[-1]
            w_, b_ = train(X_, y, _lambda, start_weight= startw, start_bias=startb)
            train_loss = loss(X_, y, w_, b_, _lambda)
            test_loss = loss(test_X_, test_y, w_, b_, _lambda)
            # if _lambda < 135:
            #     exit(0)
            ws=np.vstack([ws, w_])
            bs=np.vstack([bs, b_])

            zero_percentage = np.count_nonzero(w_ == 0) / w_.shape[0]
            ws_zeros.append(zero_percentage)

            # Write the current _lambda and zero_percentage to file
            current_data = {'lambda': _lambda,
                            'zero_percentage': zero_percentage,
                            'train_loss': train_loss,
                            'test_loss': test_loss,
                            'w': w_.tolist()}
            data_to_write.append(current_data)

            # Update _lambda for next iteration
            _lambda /= 2
        json.dump(data_to_write, f)

def train_lambda_30():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    feature_indices_to_names = {i: feature for i, feature in enumerate(df_train.columns)}
    with open('homeworks/lasso/feature_indices_to_names.json', 'w') as f:
        json.dump(feature_indices_to_names, f)

    train_x = df_train.drop('ViolentCrimesPerPop', axis=1).values
    train_y = df_train['ViolentCrimesPerPop'].values

    test_x = df_test.drop('ViolentCrimesPerPop', axis=1).values
    test_y = df_test['ViolentCrimesPerPop'].values

    d = train_x.shape[1]  # Number of features
    n = train_x.shape[0]  # Number of observations

    # Initialize weights and other parameters
    w = np.zeros(d,)

    X = train_x
    y = train_y

    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    X_ = (X - x_mean) / x_std
    test_X_ = (test_x - x_mean) / x_std
    ws = np.empty((0, w.shape[0]))
    bs = np.empty((0, 1))
    _lambda = 30
    ws_zeros = []
    lams = []
    zero_percentage = 1
    data_to_write = []
    with open('homeworks/lasso/crime_data_progress_lambda_30.json', 'w') as f:
        lams.append(_lambda)
        # print(_lambda)
        startw = None
        startb = None
        if(len(ws) != 0):
            startw = ws[-1]
            startb = bs[-1]
        w_, b_ = train(X_, y, _lambda, start_weight= startw, start_bias=startb)
        train_loss = loss(X_, y, w_, b_, _lambda)
        test_loss = loss(test_X_, test_y, w_, b_, _lambda)
        # if _lambda < 135:
        #     exit(0)
        ws=np.vstack([ws, w_])
        bs=np.vstack([bs, b_])

        zero_percentage = np.count_nonzero(w_ == 0) / w_.shape[0]
        ws_zeros.append(zero_percentage)

        # Write the current _lambda and zero_percentage to file
        current_data = {'lambda': _lambda,
                        'zero_percentage': zero_percentage,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'w': w_.tolist()}
        data_to_write.append(current_data)

        # Update _lambda for next iteration
        json.dump(data_to_write, f)

def graph_zero_percentage():
    # Initialize empty lists to hold lambda and zero_percentage values
    lams = []
    zero_percentages = []

    # Read the JSON file
    with open('homeworks/lasso/crime_data_progress.json', 'r') as f:
        data_read = json.load(f)

    for entry in data_read:
        lams.append(entry['lambda'])
        zero_percentages.append(entry['zero_percentage'])

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
    plt.savefig('homeworks/lasso/zero_percentage_plot.png', format='png', dpi=300)
    plt.show()


def plot_regularization_paths():
    # Read the feature_indices_to_names.json file to get the mapping from index to feature names
    with open('homeworks/lasso/feature_indices_to_names.json', 'r') as f:
        feature_indices_to_names = json.load(f)

    # Reverse the mapping to go from feature names to indices
    feature_names_to_indices = {v: int(k) for k, v in feature_indices_to_names.items()}

    # Read the crime_data_progress.json file
    with open('homeworks/lasso/crime_data_progress.json', 'r') as f:
        data_read = json.load(f)

    # Initialize arrays to hold lambda values and coefficients
    lams = []
    coefs = {feature: [] for feature in ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']}

    for entry in data_read:
        lams.append(entry['lambda'])
        w = np.array(entry['w'])
        for feature in coefs.keys():
            idx = feature_names_to_indices[feature] - 1
            coefs[feature].append(w[idx])

    # Plotting
    plt.figure(figsize=(10, 6))

    for feature, values in coefs.items():
        plt.plot(lams, values, label=feature)

    plt.xscale('log')
    plt.title('Regularization Paths')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.xlim(max(lams), min(lams))

    # Save the plot
    plt.savefig('homeworks/lasso/regularization_paths.png', format='png', dpi=300)
    plt.show()

def plot_loss():
    # Read the crime_data_progress.json file
    with open('homeworks/lasso/crime_data_progress.json', 'r') as f:
        data_read = json.load(f)

    # Initialize arrays to hold lambda values and coefficients
    lams = []
    train_loss = []
    test_loss = []

    for entry in data_read:
        lams.append(entry['lambda'])
        train_loss.append(entry['train_loss'])
        test_loss.append(entry['test_loss'])
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(lams, train_loss, label="train")
    plt.plot(lams, test_loss, label="loss")
    plt.xscale('log')
    plt.title('Train Loss/Test Loss vs Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(max(lams), min(lams))

    # Save the plot
    plt.savefig('homeworks/lasso/loss.png', format='png', dpi=300)
    plt.show()


def plot_weight():
    # Read the feature_indices_to_names.json file to get the mapping from index to feature names
    with open('homeworks/lasso/feature_indices_to_names.json', 'r') as f:
        feature_indices_to_names = json.load(f)

    # Reverse the mapping to go from feature names to indices
    feature_names_to_indices = {v: int(k) for k, v in feature_indices_to_names.items()}

    # Read the crime_data_progress.json file
    with open('homeworks/lasso/crime_data_progress_lambda_30.json', 'r') as f:
        data_read = json.load(f)

    w = np.array(data_read[-1]['w'])

    # Initialize arrays to hold lambda values and coefficients
    feature_weights = {feature: w[feature_names_to_indices[feature] - 1] for feature in feature_names_to_indices.keys() if feature != "ViolentCrimesPerPop"}

    # Plotting
    plt.figure(figsize=(20, 8))
    plt.bar(feature_weights.keys(), feature_weights.values())
    plt.title('Feature Weights')
    plt.xlabel('Feature Names')
    plt.ylabel('Weights')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot
    plt.savefig('homeworks/lasso/feature_weights_histogram.png', format='png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # main()
    # graph_zero_percentage()
    # plot_regularization_paths()
    # plot_loss()\
    # train_lambda_30()
    plot_weight()