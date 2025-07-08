#!/usr/bin/env python3
"""
trellis_plot.py
Plotting suite for trellis_test data
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import root_scalar
from scipy.spatial import ConvexHull

def H(p):
    return -1*p*np.log2(p)-(1-p)*np.log2(1-p)

def R_opt_hamming(D, M):
    return np.log2(M) - H(D) - D * np.log2(M-1)

def R_opt_gaussian(D):
    if D <= 1:
        return 0.5 * np.log2(1 / D)
    else: 
        return 0

def D_opt_hamming(R, M, D_bounds=(1e-6, 1 - 1e-6)):
    def f(D):
        return R_opt_hamming(D, M) - R
    
    sol = root_scalar(f, bracket=D_bounds, method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding did not converge.")

def d_r_slope(R):
    return -np.log(2) * 2 **(1-2*R)

def r_d_slope(D):
    return (1 / (2 * D * np.log2(2))) if 0 < D <= 1 else 0.0

def pareto_front(points):
    """
    Returns the nondominated (Distortion, Rate) points forming the Pareto frontier.
    Lower distortion and lower rate are both better.
    """
    # Sort by distortion
    sorted_points = points[np.argsort(points[:, 0])]
    
    pareto = []
    min_rate = np.inf
    for d, r in sorted_points:
        if r < min_rate:
            pareto.append((d, r))
            min_rate = r
    return np.array(pareto)

def M_ary_varying_K_rate_1_rd(M: int) -> None:
    """
    Plot data from M_ary_varying_K_rate_1 with specified M
    Shows difference to optimal rate-distortion point
    File M_ary_varying_K_rate_1_M_1000 must be present for M and 
    """
    if not os.path.exists(f"data/M_ary_varying_K_rate_1_{M}_1000.npz"):
        print("Required data file does not exist.")
        return
    data = np.load(f"data/M_ary_varying_K_rate_1_{M}_1000.npz")
    params = data["params"]
    distortions = data["distortions"]
    D_opt = D_opt_hamming(1, M)
    n = 1000
    K_pows = range(params[1], params[2])
    trials = params[4]
    # A: Linear scale, distortions and D_opt separate curves
    plt.figure()
    plt.plot(K_pows, distortions, "o-", label="Distortions")
    plt.hlines(D_opt, params[1], params[2] - 1, color="tab:purple", label="Optimal Distortion")
    plt.title(f"M-Ary Hamming, M={M}, n={n}, trials={trials}")
    plt.xlabel("K")
    plt.ylabel("Average Distortion")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot/M_ary_varying_K_rate_1_rd_{M}_A.pdf")
    # B: Horizontal scale log, plotting divverence
    plt.figure()
    plt.plot(K_pows, distortions - D_opt, "o-", label="Distortions-D_opt")
    plt.title(f"(M-Ary Hamming Distortion) - D_opt, M={M}, n={n}, trials={trials}")
    plt.xlabel("K")
    plt.ylabel("Average Distortion - D_opt")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(f"plot/M_ary_varying_K_rate_1_rd_{M}_B.pdf")
    return
    
def sweep_lamb_fixed_K_Mary_rd(M: int, R: float) -> None:
    if not os.path.exists(f"data/sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128_M_{M}.npz"):
        print("Required data file does not exist.")
        return
    data = np.load(f"data/sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128_M_{M}.npz")
    rates = data["rates"]
    distortions = data["distortions"]
    d_opt = np.linspace(max(0.01, distortions[0] - 0.1), distortions[-1] + 0.1)
    r_opt = np.array([R_opt_hamming(d, M) for d in d_opt])
    R_closest_idx = np.argmin(np.abs(r_opt - R))
    R_closest = r_opt[R_closest_idx]
    D_closest = d_opt[R_closest_idx]
    plt.figure()
    plt.plot(distortions, rates, "o-", color="tab:blue", label="Target R(D)")
    plt.plot(d_opt, r_opt, color="tab:purple", label="R(D)")
    plt.plot(D_closest, R_closest, "o", color="tab:green", label=f"Opt for {R}=R")
    plt.hlines(R_closest, max(0.01, distortions[0] - 0.1), distortions[-1] + 0.1, 
               color="tab:gray", linestyle="--")
    plt.title(f"$\lambda$ Param Sweep, M={M}, K=128")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot/sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128_M_{M}_rd.pdf")

def sweep_lamb_fixed_K_all_3_pts(M: int) -> None:
    R_points = 0.2, 0.5, 0.8
    for R in R_points:
        if not os.path.exists(f"data/sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128_M_{M}.npz"):
            print("Required data file does not exist.")
            return
    plt.figure()
    d_opt = np.linspace(0.05, 0.9)
    r_opt = np.array([R_opt_hamming(d, M) for d in d_opt])
    colors = ("tab:green", "tab:red", "tab:blue")
    for r, R in enumerate(R_points):
        data = np.load(f"data/sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128_M_{M}.npz")
        rates = data["rates"]
        distortions = data["distortions"]
        R_closest_idx = np.argmin(np.abs(r_opt - R))
        R_closest = r_opt[R_closest_idx]
        D_closest = d_opt[R_closest_idx]
        plt.plot(D_closest, R_closest, "o", color=colors[r], label=f"Opt for {R}=R")
        plt.hlines(R_closest, 0.05, 0.9, color="tab:gray", linestyle="--")
        plt.plot(distortions, rates, "o-", color=colors[r], label="Varying $\lambda$")
        d_opt = np.linspace(distortions[0] - 0.1, distortions[-1])
        r_opt = np.array([R_opt_hamming(d, M) for d in d_opt])
        plt.plot(d_opt, r_opt, color="tab:purple")
    plt.title(f"$\lambda$ Param Sweep, M={M}, K=128")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot/sweep_lamb_fixed_K_all_3_pts_M_{M}_rd.pdf")

def gaussian_varying_K_rate_1_rd() -> None:
    """
    """
    if not os.path.exists("data/gaussian_varying_K_rate_1_1000.npz"):
        print("Required data file does not exist.")
        return
    data = np.load("data/gaussian_varying_K_rate_1_1000.npz")
    params = data["params"]
    distortions = data["distortions"]
    D_opt = 0.25
    n = 1000
    K_pows = range(params[0], params[1])
    trials = params[3]
    # A: Linear scale, distortions and D_opt separate curves
    plt.figure()
    plt.plot(K_pows, distortions, "o-", label="Distortions")
    plt.hlines(D_opt, params[0], params[1] - 1, color="tab:purple", label="Optimal Distortion")
    plt.title(f"Quadratic-Gaussian n={n}, trials={trials}")
    plt.xlabel("K")
    plt.ylabel("Average Distortion")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/gaussian_varying_K_rate_1_rd_A.pdf")
    # B: Horizontal scale log, plotting difference
    plt.figure()
    plt.plot(K_pows, distortions - D_opt, "o-", label="Distortions-D_opt")
    plt.title(f"(Gaussian Distortion) - D_opt, n={n}, trials={trials}")
    plt.xlabel("K")
    plt.ylabel("Average Distortion - D_opt")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("plot/gaussian_varying_K_rate_1_rd_B.pdf")
    return

def gaussian_sweep_lamb_all_3_pts() -> None:
    R_points = 0.05, 0.2, 0.5, 0.8
    for R in R_points:
        if not os.path.exists(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128.npz"):
            print("Required data file does not exist.")
            return
    plt.figure()
    d_opt = np.linspace(0.005, 0.9, 100)
    r_opt = np.array([R_opt_gaussian(d) for d in d_opt])
    colors = ("tab:green", "tab:red", "tab:blue", "tab:pink")
    for r, R in enumerate(R_points):
        data = np.load(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128.npz")
        rates = data["rates"]
        distortions = data["distortions"]
        R_closest_idx = np.argmin(np.abs(r_opt - R))
        R_closest = r_opt[R_closest_idx]
        D_closest = d_opt[R_closest_idx]
        plt.plot(D_closest, R_closest, "o", color=colors[r], label=f"Opt for {R}=R")
        plt.hlines(R_closest, 0.2, 1, color="tab:gray", linestyle="--")
        plt.plot(distortions, rates, "o-", color=colors[r], label="Varying $\lambda$")
        d_opt = np.linspace(distortions[0] - 0.1, distortions[-1])
        r_opt = np.array([R_opt_gaussian(d) for d in d_opt])
        plt.plot(d_opt, r_opt, color="tab:purple")
    plt.title("$\lambda$ Param Sweep, Gaussian-MSE, K=128")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/gaussian_sweep_lamb_all_3_pts_rd.pdf")

def gaussian_sweep_lamb_near_0() -> None:
    R_points = 0.2, 0.05
    for R in R_points:
        if not os.path.exists(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128.npz"):
            print("Required data file does not exist.")
            return
    plt.figure()
    d_opt = np.linspace(0.005, 0.9, 100)
    r_opt = np.array([R_opt_gaussian(d) for d in d_opt])
    colors = ("tab:pink", "tab:blue")
    for r, R in enumerate(R_points):
        data = np.load(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_128.npz")
        rates = data["rates"]
        distortions = data["distortions"]
        if R == 0.2:
            rates = rates[2:]
            distortions = distortions[2:]
        R_closest_idx = np.argmin(np.abs(r_opt - R))
        R_closest = r_opt[R_closest_idx]
        D_closest = d_opt[R_closest_idx]
        plt.plot(D_closest, R_closest, "o", color=colors[r], label=f"Opt for {R}=R")
        plt.hlines(R_closest, 0.7, 1, color="tab:gray", linestyle="--")
        plt.plot(distortions, rates, "o-", color=colors[r], label="Varying $\lambda$")
        d_opt = np.linspace(0.7, distortions[-1])
        r_opt = np.array([R_opt_gaussian(d) for d in d_opt])
        plt.plot(d_opt, r_opt, color="tab:purple")
    plt.title("$\lambda$ Param Sweep, Gaussian-MSE, K=128")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot/gaussian_sweep_lamb_near_0.pdf")

def optimal_lambda_gaussian_grid(K: int) -> None:
    """
    Find lambda that achieves the rate closest to the target rate
    Find the lambda that achieves the lowest distortion for each target rate
    """
    target_rates = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                             0.5, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    achieved_rates = np.zeros((18, 20), dtype=float)
    target_lambdas = np.linspace(0.05, 1.95, 20)
    for i, R in enumerate(target_rates):
        data = np.load(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_{K}.npz")
        print(R)
        print(data["distortions"])
        achieved_rates[i] = data["distortions"]

    # Compute absolute difference
    diff = achieved_rates.copy()  # to avoid modifying original
    for i in range(len(target_rates)):
        for j in range(20):
            diff[i, j] = abs(diff[i, j] - target_rates[i])


    # Plot heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(diff, aspect='auto', cmap='viridis', origin='lower')

    # Label axes
    plt.xlabel('Lambda')
    plt.ylabel('Target Rate')
    plt.title('Closeness of Achieved Rate to Target Rate')
    plt.yticks(ticks=np.arange(len(target_rates)), labels=[f"{r:.3f}" for r in target_rates])
    plt.xticks(ticks=np.arange(len(target_lambdas)), labels=[f"{l:.2f}" for l in target_lambdas])

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Absolute Rate Difference')

    plt.tight_layout()
    plt.savefig('testing_grid.pdf')

def optimal_lambda_gaussian(K: int) -> None:
    """
    Find lambda that achieves the rate closest to the target rate
    Find the lambda that achieves the lowest distortion for each target rate
    """
    plt.figure(figsize=(10, 6))
    plt.grid()
    d_opt = np.linspace(0.25, 0.99, 100)
    r_opt = np.array([R_opt_gaussian(d) for d in d_opt])
    plt.plot(d_opt, r_opt, color="grey", label="RD Curve")
    target_rates = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                             0.5, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00])
    achieved_rates = np.zeros((19, 20), dtype=float)
    achieved_dists = np.zeros((19, 20), dtype=float)
    for i, R in enumerate(target_rates):
        data = np.load(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_{K}.npz")
        print(R)
        achieved_dists[i] = data["distortions"]
        achieved_rates[i] = data["rates"]
        plt.plot(achieved_dists[i], achieved_rates[i], marker='.', label=f"Rate {R}")
    plt.title("Varying $\lambda$ with Target Rates, K=128")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.1)
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.savefig(f"plot/big_gaussian_lambdas_{K}.pdf")
    """
    Now find lambda that achieves closest to optimal distortion and only plot that point
    """
    target_lambdas = np.linspace(0.05, 1.95, 20)
    opt_lambdas = np.zeros(target_rates.size)
    opt_rates = np.zeros(target_rates.size)
    opt_dists = np.zeros(target_rates.size)
    for i, R in enumerate(target_rates):
        best_idx = np.argmin(np.abs(R - achieved_rates[i]))
        opt_lambdas[i] = target_lambdas[best_idx]
        opt_rates[i] = achieved_rates[i, best_idx]
        opt_dists[i] = achieved_dists[i, best_idx]
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.plot(d_opt, r_opt, color="grey", label="RD Curve")
    plt.plot(opt_dists, opt_rates, label="achieved", marker="o")
    plt.title("Rate Point Closest to Each Target")
    plt.xlabel("Distortion")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig(f"plot/big_gaussian_best_lambdas_{K}.pdf")
    """
    Plot lambdas as a function of rate
    """
    plt.figure()
    plt.grid()
    plt.plot(target_rates, opt_lambdas, marker='.', label="Optimal Lambdas")
    plt.title("Lambda for Closest to Target Rate")
    plt.xlabel("Target Rate")
    plt.ylabel("Optimal Lambda from Grid")
    plt.legend()
    plt.savefig(f"plot/big_gaussian_opt_lambdas_{K}.pdf")

def gaussian_pareto(K_states: list) -> None: 
    """
    Plots the lower envelope of all the RD points for each K in the list
    """   
    plt.figure(figsize=(10, 6))
    plt.grid()
    d_opt = np.linspace(0.25, 0.99, 100)
    r_opt = np.array([R_opt_gaussian(d) for d in d_opt])
    plt.plot(d_opt, r_opt, color="grey", label="RD Curve")
    num_points = 19 * 20
    target_rates = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 
                             0.5, 0.55, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00])
    for K_pt in K_states:
        # For each K_pt, collect all the data and find convex envelope, then plot that
        points = np.ndarray((num_points, 2))
        for i, R in enumerate(target_rates):
            data = np.load(f"data/gaussian_sweep_lamb_fixed_R_{str(R).replace('.', '_')}_K_{K_pt}.npz")
            print(K_pt, R)
            achieved_dists = data["distortions"]
            achieved_rates = data["rates"]
            for j in range(20):
                points[i * 20 + j, 0] = achieved_dists[j]
                points[i * 20 + j, 1] = achieved_rates[j]
        pareto = pareto_front(points)
        plt.plot(pareto[:, 0], pareto[:, 1], "o-", linewidth=2, label=f'Pareto Front K={K_pt}')
    plt.title("Pareto Optimal of Test Points")
    plt.legend()
    plt.savefig("debug.pdf")
        



def main():
    gaussian_pareto([128, 512])
    return

if __name__ == "__main__":
    main()