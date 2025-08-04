#!/usr/bin/env python3
"""
trellis_test.py
Test suite for TCQ code
"""

import numpy as np
import trellis
from numba import njit
from scipy.optimize import minimize_scalar
from typing import Tuple


def M_ary_varying_K_rate_1(M: int, pow_min: int, pow_max: int, n: int, trials: int) -> None:
    """
    Test M_ary PAM while varying K fixed rate 1, saves data to npz
    """
    print(f"Testing M-ary M={M} with n={n}, K {2**pow_min} to {2**pow_max}")
    K = np.array([2**k for k in range(3, 10)])
    avg_distortion = np.zeros(K.size)
    for i, K_arg in enumerate(K):
        print(K_arg)
        for _ in range(trials):
            T = trellis.Trellis(K_arg, n, 0, 0, [M])
            x = np.random.randint(3, size=n)
            avg_distortion[i] += T.encode_vector(x, R=1.0)[2] / trials
    fname = f"data/M_ary_varying_K_rate_1_{M}_{n}.npz"
    np.savez(fname, distortions = avg_distortion, params = np.array([M, pow_min, pow_max, n, trials]))
    return

# Compute binary entropy function for parameter p
@njit
def binary_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Compute inverse binary entropy function for a given rate to find p parameter
def inverse_H(rate):
    # Define a function that returns the absolute difference between H(p) and the desired value
    def func(p):
        return abs(binary_entropy(p) - rate)
    # Use minimize_scalar to find the value of p that minimizes the absolute difference
    result = minimize_scalar(func, bounds=(1e-15, 1-1e-15), method='bounded').x
    if result > 0.5:
        return np.float64(1 - result)
    return np.float64(result)

def check_frac_R_vs_R() -> None:
    """
    Prints out to the terminal with lambda = 0
    """
    dist_R_1 = 0
    dist_frac_R = 0
    phi = inverse_H(0.9)
    for i in range(100):
        T = trellis.Trellis(16, 300, 0, 0, [4])
        x = np.random.randint(4, size=200)
        dist_R_1 += T.encode_vector(x, R=1.0)[2] / 100
        dist_frac_R += T.encode_vector(x, R=0.9, lamb=0.0, phi=phi)[2] / 100
    print(f"Rate 1 Encoder: {dist_R_1}")
    print(f"Fractional Rate, lamb=0: {dist_frac_R}")


def sweep_lamb_fixed_K_Mary(M: int, K: int, R: float, lamb_min: float, lamb_max: float) -> None:
    """
    For fixed K, fixed M, sweeps lambda parameter for target R
    """
    POINTS = 20
    TRIALS = 100
    n = 1000
    phi = inverse_H(R)
    lamb_params = np.linspace(lamb_min, lamb_max, POINTS)
    rates = np.zeros(POINTS)
    distortions = np.zeros(POINTS)
    for i, lamb in enumerate(lamb_params):
        print(i, lamb)
        for _ in range(TRIALS):
            T = trellis.Trellis(K, n, 0, 0, [M])
            x = np.random.randint(M, size=n)
            results = T.encode_vector(x, R=R, lamb=lamb, phi=phi)
            rates[i] += results[1] / TRIALS
            distortions[i] += results[2] / TRIALS
    fname = f"data/sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_{K}_M_{M}.npz"
    np.savez(fname, rates = rates, distortions = distortions, 
             params = np.array([lamb_min, lamb_max, POINTS]))
    return


def gaussian_varying_K_rate_1(pow_min: int, pow_max: int, n: int, trials: int) -> None:
    """
    Test gaussian while varying K fixed rate 1, saves data to npz
    """
    print(f"Testing Gaussian with n={n}, K {2**pow_min} to {2**pow_max}")
    K = np.array([2**k for k in range(3, 10)])
    avg_distortion = np.zeros(K.size)
    for i, K_arg in enumerate(K):
        print(K_arg)
        for _ in range(trials):
            T = trellis.Trellis(K_arg, n, 0, 1, [])
            x = np.random.normal(loc = 0, scale = 1, size=n)
            avg_distortion[i] += T.encode_vector(x, R=1.0)[2] / trials
    fname = f"data/gaussian_varying_K_rate_1_{n}.npz"
    np.savez(fname, distortions = avg_distortion, params = np.array([pow_min, pow_max, n, trials]))
    return
    
def sweep_lamb_fixed_K_gaussian(K: int, R: float, lamb_min: float, lamb_max: float) -> None:
    """
    For fixed K, sweeps lambda parameter for target R
    """
    POINTS = 20
    TRIALS = 100
    n = 1000
    phi = inverse_H(R)
    lamb_params = np.linspace(lamb_min, lamb_max, POINTS)
    print(lamb_params)
    rates = np.zeros(POINTS)
    distortions = np.zeros(POINTS)
    for i, lamb in enumerate(lamb_params):
        print(i, lamb)
        for _ in range(TRIALS):
            T = trellis.Trellis(K, n, 0, 1, [])
            x = np.random.normal(loc=0, scale=1, size=n)
            results = T.encode_vector(x, R=R, lamb=lamb, phi=phi)
            rates[i] += results[1] / TRIALS
            distortions[i] += results[2] / TRIALS
    fname = f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_{K}.npz"
    np.savez(fname, rates = rates, distortions = distortions, 
             params = np.array([lamb_min, lamb_max, POINTS]))
    return

def corr_codebook_R_1(rho: float, K: list) -> None:
    """
    Perform Rate-1 trellis experiment for the correlated codebook setup
    Use K for constraint length and rho for correlation coefficient
    """
    TRIALS = 100
    n = 1000
    avg_dist = np.zeros(len(K))
    for i, K_val in enumerate(K):
        print(K_val, rho)
        for _ in range(TRIALS):
            T = trellis.Trellis(2**K_val, n, 0, 1, ["Corr"])
            x = np.random.normal(loc=0, scale=1, size=n)
            results = T.encode_vector(x, R=1.0, lamb=0.0, phi=0.5, rho=rho)
            avg_dist[i] += results[2] / TRIALS
    fname = f"data/corr_codebook_rho_{str(rho).replace('.', '_')}.npz"
    np.savez(fname, distortions = avg_dist, K_vals = np.array(K))
    return
    
def partitioned_codebook(K: list) -> None:
    """
    Test the positive/negative codebook partition at rate 1
    """
    TRIALS = 100
    n = 1000
    avg_dist = np.zeros(len(K))
    for i, K_val in enumerate(K):
        print(K_val)
        for _ in range(TRIALS):
            T = trellis.Trellis(2**K_val, n, 0, 1, ["PosNeg"])
            x = np.random.normal(loc=0, scale=1, size=n)
            results = T.encode_vector(x, R=1.0, lamb=0.0, phi=0.5)
            avg_dist[i] += results[2] / TRIALS
    fname = f"data/partitioned_codebook_gaussian_n_{n}.npz"
    np.savez(fname, distortions = avg_dist, K_vals = np.array(K))
    return




def main():
    partitioned_codebook([5,6,7,8,9])
    return

if __name__ == "__main__":
    main()
