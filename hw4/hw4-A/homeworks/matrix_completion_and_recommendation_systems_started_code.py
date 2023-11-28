import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import torch
data = []
with open('u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)  # num_observations = 100,000
num_users = max(data[:,0])+1  # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1  # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

print(f"Successfully loaded 100K MovieLens dataset with",
      f"{len(train)} training samples and {len(test)} test samples")

# Your code goes here
# Compute estimate
print(np.max(train[:, 1]))
# Calculate average rating for each movie
movie_ratings_sum = np.zeros(num_items)
movie_ratings_count = np.zeros(num_items)

for user, movie, rating in train:
    movie_ratings_sum[movie] += rating
    movie_ratings_count[movie] += 1

# Avoid division by zero
movie_ratings_count[movie_ratings_count == 0] = 1

average_movie_ratings = movie_ratings_sum / movie_ratings_count

# Construct rank-one matrix R_hat
R_hat = np.tile(average_movie_ratings, (num_users, 1))
# Evaluate test error

test_error = 0
for user, movie, actual_rating in test:
    predicted_rating = R_hat[user, movie]
    test_error += (predicted_rating - actual_rating) ** 2

test_error /= len(test)

print("Test Error (MSE) for the estimator R_hat:", test_error)

# Your code goes here
# Create the matrix R twiddle (\widetilde{R}).
R_tilde = np.zeros((num_users, num_items))
for user, movie, rating in train:
    R_tilde[user, movie] = rating
    
  # Your code goes here
def construct_estimator(d, r_twiddle):
    U, s, Vt = svds(r_twiddle, k = d)
    R_hat_d = U @ np.diag(s) @ Vt
    
    return R_hat_d


def get_error(d, r_twiddle, dataset):
    R_hat_d = construct_estimator(d, r_twiddle)
    error = 0
    for user, movie, actual_rating in dataset:
        predicted_rating = R_hat_d[user, movie]
        error += (predicted_rating - actual_rating) ** 2
    return error / len(dataset)

# Your code goes here
# Evaluate train and test error for: d = 1, 2, 5, 10, 20, 50.

d_values = [1, 2, 5, 10, 20, 50]
usv_train_errors = []
usv_test_errors = []
for d in d_values:
    usv_train_error=get_error(d, R_tilde, train)
    usv_test_error=get_error(d, R_tilde, test)
    usv_train_errors.append(usv_train_error)
    usv_test_errors.append(usv_test_error)

# Plot both train and test error as a function of d on the same plot.


plt.figure(figsize=(10, 6))
plt.plot(d_values, usv_train_errors, label='Training Error', marker='o')
plt.plot(d_values, usv_test_errors, label='Test Error', marker='o')
plt.xlabel('d (Number of Singular Values)')
plt.ylabel('Average Squared Error')
plt.title('Training and Test Errors vs. Rank of Approximation')
plt.legend()
plt.grid(True)
plt.show()

# Your code goes here
def closed_form_u(V, U, l, R_tilde):
  print(V.shape, U.shape)
  for i in range(U.shape[0]):
    R_i_tile = R_tilde[i]
    indices = R_i_tile > 0
    V_j = V[indices, :]
    R_i = R_i_tile[indices]
    A = V_j.T @ V_j + l * np.eye(V_j.shape[1])
    b = (V_j.T @ R_i).reshape(-1, 1)
    U[i,:] = np.linalg.solve(A, b).flatten()
  return U

def closed_form_v(V, U, l, R_tilde):
  for j in range(V.shape[0]):
    R_j_tile = R_tilde[:,j]
    indices = R_j_tile > 0
    U_i = U[indices, :]
    R_j = R_j_tile[indices]
    A = U_i.T @ U_i + l * np.eye(U_i.shape[1])
    b = (U_i.T @ R_j).reshape(-1, 1)
    V[j,:] = np.linalg.solve(A, b).flatten()
  return V


def construct_alternating_estimator(
    d, r_twiddle, l=0.0, delta=1e-2, sigma=0.1, U=None, V=None
):
  old_U, old_V = np.zeros((num_users, d)), np.zeros((num_items, d))
  if U is None:
    U = np.random.rand(num_users, d) * sigma
  if V is None:
    V = np.random.rand(num_items, d) * sigma
  while (np.max(np.abs(V - old_V)) > delta and np.max(np.abs(U - old_U)) > delta):
    old_U, old_V = U, V
    U = closed_form_u(V, U, l, r_twiddle)
    V = closed_form_v(V, U, l, r_twiddle)
  return U, V

def calc_uv_error(dataset, U, V):
    user = dataset[:,0]
    item = dataset[:,1]
    score = dataset[:,2]
    # print(U.shape, V.shape)
    pred = np.einsum('ij,ij->i', U[user], V[item])
    mse_error = np.mean((score-pred) ** 2)
    return mse_error
    
from itertools import product
d_vals = [1, 2, 5, 10, 20, 50]
lambdas = [0.5, 1, 5, 10]
sigmas = [0.01, 0.1, 1, 10]

# Prepare the plots
fig, axes = plt.subplots(4, 4, figsize=(20, 20)) # 4x4 grid for 16 combinations
axes = axes.flatten()

# Iterate over all combinations of lambdas and sigmas
for index, (lambda_val, sigma_val) in enumerate(product(lambdas, sigmas)):
    uv_train_errors = []
    uv_test_errors = []

    for d in d_vals:
        U, V = construct_alternating_estimator(d=d, r_twiddle=R_tilde, l=lambda_val, sigma=sigma_val)
        uv_train_errors.append(calc_uv_error(train, U, V))
        uv_test_errors.append(calc_uv_error(test, U, V))

    # Plot the train and test error for each combination
    ax = axes[index]
    ax.plot(d_vals, uv_train_errors, label='Training Error', marker='o')
    ax.plot(d_vals, uv_test_errors, label='Test Error', marker='o')
    ax.set_xlabel('d (Number of Singular Values)')
    ax.set_ylabel('Average Squared Error')
    ax.set_title(f'Train/Test Errors for lambda={lambda_val}, sigma={sigma_val}')
    ax.legend()
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()