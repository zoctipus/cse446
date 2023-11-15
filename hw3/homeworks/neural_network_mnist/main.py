# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu, one_hot
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, record_function


from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = math.sqrt(1/d)
        self.W0 = Parameter(torch.empty(d, h).uniform_(-alpha, alpha))
        self.b0 = Parameter(torch.empty(h).uniform_(-alpha, alpha))
        self.W1 = Parameter(torch.empty(h, k).uniform_(-alpha, alpha))
        self.b1 = Parameter(torch.empty(k).uniform_(-alpha, alpha))
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        x = relu(x @ self.W0 + self.b0)
        x = x @ self.W1 + self.b1
        return x


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha = math.sqrt(1/d)
        self.W0 = Parameter(torch.empty(d, h0).uniform_(-alpha, alpha))
        self.b0 = Parameter(torch.empty(h0).uniform_(-alpha, alpha))
        self.W1 = Parameter(torch.empty(h0, h1).uniform_(-alpha, alpha))
        self.b1 = Parameter(torch.empty(h1).uniform_(-alpha, alpha))
        self.W2 = Parameter(torch.empty(h1, k).uniform_(-alpha, alpha))
        self.b2 = Parameter(torch.empty(k).uniform_(-alpha, alpha))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        x = relu(x @ self.W0 + self.b0)
        x = relu(x @ self.W1 + self.b1)
        return x @ self.W2 + self.b2

def accuracy_score(model:Module, dataloader, device: torch.device) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to MSE accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for observation, target in dataloader:
            observation, target = observation.to(device), target.to(device)
            _predict = model(observation)
            _, predict_ = torch.max(_predict.data, 1)
            total += target.size(0)
            correct += (predict_ == target).sum().item()
    accuracy = correct / total
    return accuracy


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader, device) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()
    history = []
    total_loss = 0
    epoch = 0
    accuracy = 0
    model.to(device)
    while  accuracy< 0.99:
        epoch += 1
        total_loss = 0
        for _x, _y in train_loader:
            _x, _y = _x.to(device), _y.to(device)
            optimizer.zero_grad()
            y_pred = model(_x)
            loss = cross_entropy(y_pred, _y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        history.append(avg_train_loss)
        accuracy = accuracy_score(model, train_loader, device)
        print(f"epoch:{epoch} train_loss:{avg_train_loss} accuracy:{accuracy}")
    return history

def eval_test_loss(model:Module, test_loader, device):
    total_val_loss = 0
    model.to(device)
    with torch.no_grad():
        for _x, _y in test_loader:
            _x, _y = _x.to(device), _y.to(device)
            y_pred = model(_x)
            loss = cross_entropy(y_pred, _y)
            total_val_loss += loss.item()
    ave_val_loss = total_val_loss / len(test_loader)
    return ave_val_loss

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

            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title("training loss vs epoch")
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"hw3_written/{filename_prefix}_batch{batch_size}.png")
        plt.close()  # Close the figure to free memory

def count_parameters(model: Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    train_dataset = TensorDataset(x, y)
    test_dataset = TensorDataset(x_test, y_test)
    feature = 28 * 28
    import itertools
    
    result = {}
    lr= 10 ** torch.linspace(-3, -3, 1)
    batch_sizes = 2 ** torch.linspace(8, 9, 1)
    models = {
        # "F1": F1(h=64, d=feature, k=10),
        "F2": F2(h0=32, h1=32, d=feature, k=10)
    }
    combos = list(itertools.product(lr, batch_sizes, models.items()))
    count = 0
    for lr, batch_size, [model_name, model] in combos:
        train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True, num_workers=12)
        test_loader = DataLoader(test_dataset, batch_size=int(batch_size), shuffle=False)
        history = train(model=model, optimizer=Adam(params=model.parameters(), lr=lr), train_loader=train_loader, device=device)
        test_loss = eval_test_loss(model=model, test_loader=test_loader,device=device)
        accuracy = accuracy_score(model, test_loader,device=device)
        model_name_ = f'{model_name}_lr{lr}_batch{int(batch_size)}'
        print(f"{count}: {model_name_} accuracy:{accuracy} train loss:{history[-1]} test_loss:{test_loss} param_size:{count_parameters(model)}")
        count +=1
        result[model_name_] = {}
        result[model_name_]['train'] = history
        result[model_name_]['val_loss'] = test_loss
        result[model_name_]["model"] = model
        result[model_name_]["accuracy"] = accuracy
        result[model_name_]['lr'] = lr
        result[model_name_]['batch_size'] = int(batch_size)
    
        plot_and_save_graphs(result, model_name)

if __name__ == "__main__":
    main()
