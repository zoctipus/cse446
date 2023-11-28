if __name__ == "__main__":
    from k_means import lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    Run Lloyd's Algorithm for k=10, and report 10 centers returned.

    NOTE: This code might take a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), _ = load_dataset("mnist")
    centers = lloyd_algorithm(x_train, num_centers=10, epsilon=10e-3)
    # Reshape centers to 28x28 (assuming MNIST images are 28x28)
    centers_images = centers[0].reshape(-1, 28, 28)

    # Plotting the centers
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(centers_images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Center {i+1}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
