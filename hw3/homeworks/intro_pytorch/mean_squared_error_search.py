if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    # raise NotImplementedError("Your Code Goes Here")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for observation, target in dataloader:
            _predict = model(observation)
            _, predict_ = torch.max(_predict.data, 1)
            _, target_ = torch.max(target, 1)

            total += target.size(0)
            correct += (predict_ == target_).sum().item()
    accuracy = correct / total
    return accuracy


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Notes:
        - Try using learning rate between 1e-5 and 1e-3.
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.
        - When searching over batch_size using powers of 2 (starting at around 32) is typically a good heuristic.
            Make sure it is not too big as you can end up with standard (or almost) gradient descent!

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    # raise NotImplementedError("Your Code Goes Here")
    input_sample, output_sample = dataset_train[0]
    # Determine the shapes
    class LinearModel(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.linear = LinearLayer(input_size, output_size)
        
        def forward(self, inputs):
            x = self.linear(inputs)
            return x

    class OneHiddenLayerModel(nn.Module):
        def __init__(self, input_size, output_size, hidden_size, activation_func):
            super().__init__()
            self.linear0 = LinearLayer(input_size, hidden_size)
            self.activation = activation_func
            self.linear1 = LinearLayer(hidden_size, output_size)
            
        def forward(self, inputs):
            x = self.activation(self.linear0(inputs))
            x = self.linear1(x)
            return x
        
    class TwoHiddenLayerModel(nn.Module):
        def __init__(self, input_size, output_size, hidden_size, activation_func1, activation_func2):
            super().__init__()
            self.linear0 = LinearLayer(input_size, hidden_size)
            self.activation0 = activation_func1
            self.linear1 = LinearLayer(hidden_size, hidden_size)
            self.activation1 = activation_func2
            self.linear2 = LinearLayer(hidden_size, output_size)
            
        def forward(self, inputs):
            x = self.activation0(self.linear0(inputs))
            x = self.activation1(self.linear1(x))
            x = self.linear2(x)
            return x
    
    input_feature_size = input_sample.shape[0]
    output_size = 2
    import itertools
    lr= 10 ** np.linspace(-5, -3, 3)
    batch_size = (2 ** np.linspace(5, 7, 3))
    models = {
        "Linear": LinearModel(input_feature_size, output_size),
        "OneHidden_Sigmoid": OneHiddenLayerModel(input_feature_size, output_size, 2, SigmoidLayer()),
        "OneHidden_ReLU": OneHiddenLayerModel(input_feature_size, output_size, 2, ReLULayer()),
        "TwoHidden_SigmoidReLU": TwoHiddenLayerModel(input_feature_size, output_size, 2, SigmoidLayer(), ReLULayer()),
        "TwoHidden_ReLUSigmoid": TwoHiddenLayerModel(input_feature_size, output_size, 2, ReLULayer(), SigmoidLayer())
    }
    combos = list(itertools.product(lr, batch_size, models.items()))
    result = {}
    count = 0
    for lr, batch_size, [model_name, model] in combos:
        train_loader = DataLoader(dataset_train, batch_size=int(batch_size), shuffle=True)
        val_loader = DataLoader(dataset_val, batch_size=int(batch_size), shuffle=False)
        history = train(train_loader, model, MSELossLayer(), SGDOptimizer(model.parameters(), lr=lr), val_loader)
        model_name_ = f'{model_name}_lr{lr}_batch{int(batch_size)}'
        print(f"{count}: {model_name_}")
        count +=1
        result[model_name_] = {}
        result[model_name_]['train'] = history['train']
        result[model_name_]['val'] = history['val']
        result[model_name_]["model"] = model
        result[model_name_]['lr'] = lr
        result[model_name_]['batch_size'] = int(batch_size)
    return result


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(to_one_hot(y_test))
    )

    result = mse_parameter_search(dataset_train, dataset_val)
    # Plotting Training and Validation Losses
    plot_and_save_graphs(result, "mse_model_losses")

    # Choosing and Reporting the Best Model
    best_model_name = min(result, key=lambda k: min(result[k]['val']))
    print(f"Best Model: {best_model_name}")

    # Plot Best Model Guesses on Test Set
    test_loader = DataLoader(dataset_test, batch_size=32)
    best_model = result[best_model_name]["model"]
    plot_model_guesses(test_loader, best_model, "best model")

    # Report Accuracy on Test Set
    #0.7
    accuracy = accuracy_score(best_model, test_loader)
    print(f"Accuracy on Test Set: {accuracy:.2f}")


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r

def plot_and_save_graphs(results, filename_prefix):
    lr_batch_combinations = set((value['lr'], value['batch_size']) for value in results.values())
    unique_batch_sizes = set(batch_size for _, batch_size in lr_batch_combinations)
    unique_lrs = set(lr for lr, _ in lr_batch_combinations)

    # For each batch size, plot a figure with subplots for each learning rate
    for batch_size in unique_batch_sizes:
        plt.figure(figsize=(15, 6 * len(unique_lrs)))  # Adjust the figure size as needed

        for i, lr in enumerate(unique_lrs, start=1):
            plt.subplot(len(unique_lrs), 1, i)  # Create a subplot for each learning rate

            # Filter and plot models with the current lr and batch_size combination
            for model_name in results:
                if f"_lr{lr}_batch{batch_size}" in model_name:
                    plt.plot(results[model_name]['train'], label=f"{model_name} - Train")
                    plt.plot(results[model_name]['val'], label=f"{model_name} - Val")

            plt.xlabel('Epochs')
            plt.ylabel('MSE Loss')
            plt.title(f'LR: {lr}')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"hw3_written/{filename_prefix}_batch{batch_size}.png")
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    main()
