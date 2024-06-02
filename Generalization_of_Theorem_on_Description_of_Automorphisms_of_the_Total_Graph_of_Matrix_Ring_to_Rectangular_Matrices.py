import numpy as np
from itertools import product

def is_singular(matrix):
    return np.linalg.matrix_rank(matrix) < min(matrix.shape)

def is_injective(func, domain):
    seen = set()
    for x in domain:
        y = tuple(func(x).flatten())
        if y in seen:
            return False
        seen.add(y)
    return True

def is_surjective(func, domain, codomain):
    seen = set()
    for x in domain:
        y = tuple(func(x).flatten())
        seen.add(y)
    return len(seen) == len(codomain)

def sigma1(A):
    spec_A = np.linalg.eigvals(A.T @ A)
    if 0 not in spec_A and not np.array_equal(A, -A.T):
        return np.zeros_like(A)
    elif 0 in spec_A:
        E11 = np.zeros_like(A)
        E11[0, 0] = 1
        return E11
    elif np.array_equal(A, -A.T):
        return -A

def sigma2(A):
    if np.trace(A) == 0:
        return np.zeros_like(A)
    else:
        return A

def sigma3(A):
    return np.diag(np.diag(A))

def sigma4(A):
    return A + np.conjugate(A)

A = np.array([[1, 2], [2, 1]], dtype=np.float64)  # A simple symmetric matrix
B = np.array([[0, 1], [-1, 0]], dtype=np.float64)  # A skew-symmetric matrix
C = np.array([[1, 2], [3, -1]], dtype=np.float64)  # Matrix with trace = 0
D = np.array([[1, 2], [3, 4]], dtype=np.float64)   # Matrix with trace != 0
# Define a small domain and codomain for testing bijection and surjection
domain = [np.array([[i, j], [k, l]], dtype=np.float64) for i, j, k, l in product(range(3), repeat=4)]
codomain = [np.array([[i, j], [k, l]], dtype=np.float64) for i, j, k, l in product(range(3), repeat=4)]



