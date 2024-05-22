import numpy as np
np.random.seed(0)
from tqdm import tqdm
import os
from IP import *

def save_data(A, c, b, path):
    data = {'A': A, 'c': c, 'b': b}
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.save(path, data)

def generate_packing_IP(m=15, n=30):
    A = np.random.randint(0, 6, (m, n)).astype(float)
    b = np.random.randint(9*n, 10*n+1, (m)).astype(float)
    c = np.random.randint(1, 11, (n)).astype(float)
    return A, c, b

def generate_knapsack_IP(N, K):
    """
    Generates an integer programming formulation for a multi-dimensional knapsack problem.
    
    Parameters:
    - N: int, number of items
    - K: int, number of knapsacks
    
    Mathematically, the formulation is:
    $ \max \sum_{k=1}^{K} \sum_{i=1}^{N} w[i] x_{ik} $
    $ \text{s.t.} $
    $ \sum_{i=1}^{N} w[i] x_{ik} \leq W_k, \forall k \in K$
    $ x_{ik} \in \{0, 1\} $
    """
    w = np.floor(np.random.normal(50, 2, N))
    W = np.array([np.floor(np.sum(w) / (2 * K)) + (k - 1) for k in range(1, K + 1)])

    zero_N = np.zeros(N)
    A_K = [0] * K
    for i in range(K):
        Ai = [zero_N for _ in range(K)]
        Ai[i] = w
        A_K[i] = np.hstack(Ai)
    A_K = np.vstack(A_K)
    A_N = np.hstack([np.eye(N) for _ in range(K)])
    A = np.vstack([A_K, A_N])

    b = np.hstack([W, np.ones(N)])
    c = np.hstack([w for _ in range(K)])
    return A, c, b

if __name__ == "__main__":
    for (N, K) in [(16, 2), (30, 3), (20,1), (30, 1), (50, 1)]:
        knapsack_train_path = f"data/knapsack_{N}_{K}/train"
        knapsack_test_path = f"data/knapsack_{N}_{K}/test"
        for i in tqdm(range(100)):
            A, c, b = generate_knapsack_IP(N=N, K=K)
            save_data(A, c, b, f"{knapsack_train_path}/knapsack_{N}_{K}_train_{i}.npy")
        for i in tqdm(range(100)):
            A, c, b = generate_knapsack_IP(N=N, K=K)
            save_data(A, c, b, f"{knapsack_test_path}/knapsack_{N}_{K}_test_{i}.npy")

    for (m, n) in [(15, 30), (20, 40)]:
        packing_train_path = f"data/packing_{m}_{n}/train"
        packing_test_path = f"data/packing_{m}_{n}/test"
        for i in tqdm(range(100)):
            A, c, b = generate_packing_IP(m=m, n=n)
            save_data(A, c, b, f"{packing_train_path}/packing_{m}_{n}_train_{i}.npy")
        for i in tqdm(range(100)):
            A, c, b = generate_packing_IP(m=m, n=n)
            save_data(A, c, b, f"{packing_test_path}/packing_{m}_{n}_test_{i}.npy")