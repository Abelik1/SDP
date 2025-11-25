from sdp_system import LTSDPSystem, dagger, kron, partial_trace  # your core SDP class
from sdp_analysis import LTAnalyzer     # the new module above
import numpy as np
if __name__ == "__main__":
    # --- basic config ---
    SEED = 0
    SOLVER = "SCS"       # or "MOSEK" if you have it
    TOL = 1e-7

    np.random.seed(SEED)

    # --- define a simple 2x2 system ---
    dA = dAp = 2
    H_A  = np.diag([0.0, 1.0])
    H_Ap = np.diag([0.0, 1.5])   # can set = H_A if you want TFD test
    beta = 1.0

    # build the SDP system + analyzer
    system   = LTSDPSystem(H_A, H_Ap, beta, solver=SOLVER, tol=TOL)
    analyzer = LTAnalyzer(system)

    # --- sanity check 1: random state single analysis ---
    def rand_state(d):
        X = np.random.randn(d, d) + 1j*np.random.randn(d, d)
        rho = X @ dagger(X)
        return rho / np.trace(rho)

    rho = rand_state(dA * dAp)
    rep_single = analyzer.analyze_single_state(rho, label="random_state", solver=SOLVER, tol=TOL)

    print("=== Single-state analysis ===")
    print("Label:", rep_single["label"])
    print("Is LT?      ", rep_single["LT_membership"]["is_LT"])
    print("D(rho||γ⊗γ):", rep_single["monotones"]["D_rho_vs_gamma"])
    print("I(A:B):     ", rep_single["monotones"]["I_rho"])
    print("Dist to LT: ", rep_single["distance_to_LT"]["distance"])
    print("Dist to classical LT:", rep_single["distance_to_classical_LT"]["distance"])
    print()

    # --- sanity check 2: convertibility tau -> tau' ---
    tau   = rand_state(dA * dAp)
    tau_p = rand_state(dA * dAp)

    rep_pair = analyzer.analyze_pair(tau, tau_p, label="random_pair", solver=SOLVER, tol=TOL)

    print("=== Convertibility analysis (tau -> tau') ===")
    print("Dims:", rep_pair["dims"])
    print("Global GP feasible?:", rep_pair["feasibility"]["Global_GP"])
    print("Local GP feasible?: ", rep_pair["feasibility"]["Local_GP"])
    print("D_tau  vs γ⊗γ:", rep_pair["monotones"]["D_tau_vs_gamma"])
    print("D_taup vs γ⊗γ:", rep_pair["monotones"]["D_taup_vs_gamma"])
    print("I_tau        :", rep_pair["monotones"]["I_tau"])
    print("I_taup       :", rep_pair["monotones"]["I_taup"])
    print("Dist(tau -> LT):", rep_pair["extra_distances"]["distances_tau"]["to_LT"]["distance"])
    print("Dist(tau'-> LT):", rep_pair["extra_distances"]["distances_taup"]["to_LT"]["distance"])
