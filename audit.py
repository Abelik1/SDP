import numpy as np
from numpy.linalg import norm
from utils import build_system_and_analyzer
from sdp_system import mutual_information

def random_diagT_states(system, n=10, tmax=0.9, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    tries = 0
    while len(out) < n and tries < 5000:
        tries += 1
        tx, ty, tz = rng.uniform(-tmax, tmax, size=3)
        rho = system.lt_diagT_state(tx, ty, tz)  # uses γ⊗γ + (1/4)(tx XX + ty YY + tz ZZ)
        w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        if np.min(w) > -1e-10:  # PSD-ish
            out.append(0.5 * (rho + rho.conj().T))
    return out

def build_adj(system, states, method="single"):
    n = len(states)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i, j] = 1
                continue
            if method == "single":
                ok, _ = system.check_local_gp_feasible(states[i], states[j])
            else:
                ok, _ = system.check_local_gp_feasible_multistart(states[i], states[j], n_random_starts=6, verify=True)
            adj[i, j] = int(ok)
    return adj
def transitivity_violations(adj):
    n = adj.shape[0]
    v = 0
    for i in range(n):
        for j in range(n):
            if adj[i,j] == 0: 
                continue
            for k in range(n):
                if adj[j,k] == 1 and adj[i,k] == 0:
                    v += 1
    return v
if __name__ == "__main__":
    system, analyzer = build_system_and_analyzer(
        dA=2, dAp=2, beta=1.0, solver="SCS", tol=1e-7,
        symmetric=True, eps_eq_global=1e-6, eps_eq_local=1e-6, eps_gibbs=1e-8
    )

    # NOTE: need LTAnalyzer factory method access: easiest is states via analyzer.factory
    # but we can use system.lt_ray_state helpers already embedded via sdp_analysis LTStateFactory.
    # Here: use analyzer.factory.lt_diagT_state
    states = []
    rng = np.random.default_rng(0)
    while len(states) < 15:
        tx, ty, tz = rng.uniform(-0.8, 0.8, size=3)
        rho = analyzer.factory.lt_diagT_state(tx, ty, tz)
        w = np.linalg.eigvalsh(rho)
        if np.min(w) > -1e-10:
            states.append(0.5 * (rho + rho.conj().T))

    # sort by decreasing mutual information (like your ordered heatmap)
    I = [mutual_information(r, (2,2)) for r in states]
    order = np.argsort(-np.array(I))
    states = [states[i] for i in order]
    I = [I[i] for i in order]

    adj_single = build_adj(system, states, method="single")
    adj_multi  = build_adj(system, states, method="multi")

    # "Recovered edges" = edges found by multistart that single missed
    recovered = np.sum((adj_multi == 1) & (adj_single == 0))
    lost      = np.sum((adj_multi == 0) & (adj_single == 1))
    print("Recovered edges (multi found, single missed):", recovered)
    print("Lost edges (single found, multi+verify rejected):", lost)

    # crude fragmentation proxy: average out-degree above diagonal
    n = len(states)
    upper_single = np.sum(np.triu(adj_single, 1))
    upper_multi  = np.sum(np.triu(adj_multi, 1))
    print("Upper-triangular edges (single):", upper_single)
    print("Upper-triangular edges (multi):", upper_multi)
    print("Transitivity violations (single):", transitivity_violations(adj_single))
    print("Transitivity violations (multi):", transitivity_violations(adj_multi))   
