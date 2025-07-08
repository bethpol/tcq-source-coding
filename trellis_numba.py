#!/usr/bin/env python3
"""
trellis.py
Helper functions for the trellis class optimized with Numba
"""
import numpy as np
from numba import njit
from typing import Tuple

@njit
def build_transition_matrix_0_numba(N: int, state_bits: int) -> np.ndarray:
    """
    Builds the transition matrix using bitwise operations and Numba for speed.

    Args:
        N (int): Number of states, must be a power of 2 (note not constraint length).
        state_bits (int): Number of bits in the state.

    Returns:
        np.ndarray: Transition matrix of shape (K, K) with 1s where transitions exist.
    """
    transition_matrix = np.zeros((N, N), dtype=np.int64)
    state_bits = 0
    temp = N
    while temp > 1:
        temp >>= 1
        state_bits += 1

    for i in range(N):
        next_0 = ((0 << state_bits) | i) >> 1  # prepend 0, drop LSB
        next_1 = ((1 << state_bits) | i) >> 1  # prepend 1, drop LSB
        transition_matrix[i, next_0] = 1
        transition_matrix[i, next_1] = 1

    return transition_matrix

@njit
def fill_p_params_numba(p_params, all_choices) -> None:
    """
    Fills the penalty matrix with the random choices of p params.

    Args:
        p_params (np.ndarray): array to store p params
        all_choices (np.ndarray): Pre-determined positions for phi, (1-phi)

    Returns:
        Nothing
    """
    N, m = all_choices.shape
    for i in range(N):
        for j in range(m):
            choice = all_choices[i, j]
            p_params[2*i + choice, j] = 1

@njit
def distance_each(y_n: np.ndarray, x: int, source_type: int) -> np.ndarray:
    dists = np.empty(y_n.shape[0], dtype=np.float32)

    if source_type == 0:
        for i in range(y_n.shape[0]):
            dists[i] = 0.0 if y_n[i] == x else 1.0
    if source_type == 1:
        for i in range(y_n.shape[0]):
            dists[i] = (y_n[i] - x) ** 2
    return dists


@njit
def encode_R_1_numba(
    x_n: np.ndarray,
    n: int,
    K: int,
    codebook: np.ndarray,
    source_type: int,
    transition: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    TO-DO: Write this
    """
    prev_state = np.full((n + 1, K), -1, dtype=np.int64)
    dist_matrix = np.full((n + 1, K), np.inf, dtype=np.float32)
    dist_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        x = x_n[i - 1]
        reconstructions = np.empty(2 * K, dtype=codebook.dtype)
        for k in range(2 * K):
            reconstructions[k] = codebook[k, i - 1]
        distances = distance_each(reconstructions, x, source_type)  # (2K,)
        for j in range(K):  # destination state
            best_cost = np.inf
            best_prev = -1

            for p in range(K):  # previous state
                if transition[p, j] == 1:
                    # Determine which branch of p leads to j
                    found = 0
                    branch = -1
                    for b in range(K):  # check all possible destinations from p
                        if transition[p, b] == 1:
                            if b == j:
                                branch = found
                            found += 1
                    if branch == -1:
                        continue  # shouldn't happen

                    branch_index = 2 * p + branch
                    cost = dist_matrix[i - 1, p] + distances[branch_index]

                    if cost < best_cost:
                        best_cost = cost
                        best_prev = p

            dist_matrix[i, j] = best_cost
            prev_state[i, j] = best_prev
    # Traceback
    path = np.full(n, -1, dtype=np.int64)
    final_state = 0
    min_cost = dist_matrix[n, 0]
    for j in range(1, K):
        if dist_matrix[n, j] < min_cost:
            min_cost = dist_matrix[n, j]
            final_state = j
    path[n - 1] = final_state

    for i in range(n - 1, 0, -1):
        path[i - 1] = prev_state[i, path[i]]

    final_distortion = dist_matrix[n, final_state] / n
    rate = 1.0

    return path, rate, final_distortion
            
@njit
def compute_rate(phi: float, num_phis: int, n: int) -> float:
    """
    Computes rate with phi and num_phis phi branches
    """
    frac_phis = num_phis / n
    return -frac_phis * np.log2(phi) - (1 - frac_phis) * np.log2(1 - phi)

@njit
def encode_frac_R_numba(
    x_n: np.ndarray,
    phi: float,
    lamb: float,
    n: int,
    K: int,
    codebook: np.ndarray,
    penalties: np.ndarray,
    source_type: int,
    transition: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    TO-DO: Write this
    """
    prev_state = np.full((n + 1, K), -1, dtype=np.int64)
    dist_matrix = np.full((n + 1, K), np.inf, dtype=np.float32)
    cost_matrix = np.full((n + 1, K), np.inf, dtype=np.float32)  # Cost matrix dist + penalty
    phis_matrix = np.zeros((n + 1, K), dtype=np.int64)  # Holds num phis on path
    dist_matrix[0, 0] = 0.0
    cost_matrix[0, 0] = 0.0
    for i in range(1, n + 1):
        x = x_n[i - 1]
        reconstructions = np.empty(2 * K, dtype=codebook.dtype)
        p_params = np.empty(2 * K, dtype=phis_matrix.dtype)
        for k in range(2 * K):
            reconstructions[k] = codebook[k, i - 1]
            p_params[k] = penalties[k, i - 1]
        distances = distance_each(reconstructions, x, source_type)  # (2K,)
        penalty = (
            p_params * -1 * np.log2(phi) +
            (1 - p_params) * -1 * np.log2(1 - phi)
        )
        branch_costs = distances + lamb * penalty
        for j in range(K):  # destination state
            best_cost = np.inf
            best_prev = -1

            for p in range(K):  # previous state
                if transition[p, j] == 1:
                    # Determine which branch of p leads to j
                    found = 0
                    branch = -1
                    for b in range(K):  # check all possible destinations from p
                        if transition[p, b] == 1:
                            if b == j:
                                branch = found
                            found += 1
                    if branch == -1:
                        continue  # shouldn't happen
                    
                    branch_index = 2 * p + branch
                    cost = cost_matrix[i - 1, p] + branch_costs[branch_index]
                    dist = dist_matrix[i - 1, p] + distances[branch_index]
                    phi_count = phis_matrix [i - 1, p] + p_params[branch_index]

                    if cost < best_cost:
                        path_dist = dist
                        best_cost = cost
                        best_prev = p
                        path_phi = phi_count

            cost_matrix[i, j] = best_cost
            dist_matrix[i, j] = path_dist
            prev_state[i, j] = best_prev
            phis_matrix[i, j] = path_phi

    
    path = np.full(n, -1, dtype=np.int64)
    final_state = 0
    min_cost = cost_matrix[n, 0]
    for j in range(1, K):
        if cost_matrix[n, j] < min_cost:
            min_cost = cost_matrix[n, j]
            final_state = j
    path[n - 1] = final_state

    # Traceback
    
    for i in range(n - 1, 0, -1):
        path[i - 1] = prev_state[i + 1, path[i]]

    final_distortion = dist_matrix[n, final_state] / n
    rate = compute_rate(phi, phis_matrix[n, final_state], n)

    return path, rate, final_distortion
            
            
