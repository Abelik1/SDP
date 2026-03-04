import numpy as np
from utils import build_system_and_analyzer
from sdp_system import mutual_information

def main(
    num_dirs=1500,
    num_points=15,
    interior_scale=0.5,
    seed=0,
    n_random_starts=6,
):
    system, analyzer = build_system_and_analyzer(
        dA=2, dAp=2, beta=1.0,
        solver="SCS", tol=1e-7,
        symmetric=True,
        eps_eq_global=1e-6,
        eps_eq_local=1e-6,
        eps_gibbs=1e-8,
    )

    rng = np.random.default_rng(seed)

    # EXACTLY as in lt_C_diagT_3d_characterise
    bd = analyzer.diagT_3d_boundary(num_dirs=num_dirs, include_negative=False, tol=1e-12)

    idx = rng.integers(low=0, high=bd["u"].shape[0], size=num_points)
    states = []
    tcoords = []

    for ii in idx:
        u = bd["u"][ii]
        pmax = bd["p_max"][ii]
        p = interior_scale * float(pmax) * float(rng.uniform(0.2, 1.0))
        C0 = system.qubit_C_from_diag_T(u[0], u[1], u[2])
        rho = system.lt_ray_state(C0, p)
        states.append(rho)
        tcoords.append(p * u)

    tcoords = np.array(tcoords)

    # order by MI, as in your ordered heatmaps
    I_vals = [mutual_information(rho, system.dims) for rho in states]
    order = np.argsort(-np.array(I_vals))
    states = [states[k] for k in order]
    tcoords = tcoords[order]  # keep consistent

    N = len(states)
    I_node = np.array([system.monotones(states[i])[1] for i in range(N)], dtype=float)

    mono_tol = 1e-10

    L_single = np.zeros((N, N), dtype=int)
    L_multi  = np.zeros((N, N), dtype=int)
    
    # pairwise tests, with the same monotone prescreen used in experiments
    for i in range(N):
        for j in range(N):
            if i == j:
                L_single[i, j] = 1
                L_multi[i, j] = 1
                continue
            if I_node[i] + mono_tol < I_node[j]:
                continue

            ok1, _ = system.check_local_gp_feasible(
                states[i], states[j],
                solver=system.solver_default, tol=system.tol_default,
                eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs,
                verbose=False, return_details=False
            )
            L_single[i, j] = int(ok1)

            ok2, _ = system.check_local_gp_feasible_multistart(
                states[i], states[j],
                solver=system.solver_default, tol=system.tol_default,
                eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs,
                n_random_starts=n_random_starts, seed=seed,
                verify=True, verbose=False, return_details=False
            )
            L_multi[i, j] = int(ok2)
    lost_pairs = np.argwhere((L_single == 1) & (L_multi == 0) & (~np.eye(N, dtype=bool)))
    print("Lost pairs (i,j):", lost_pairs.tolist())
    for (i,j) in lost_pairs:
        ok, status, det = system.check_local_gp_feasible(
            states[i], states[j],
            solver=system.solver_default, tol=system.tol_default,
            eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs,
            verbose=False, return_details=True
        )
        print("\nPAIR", (i,j), "single_ok=", ok, "status=", status, "residual=", det.get("residual"))

        v = system.verify_local_gp_details(states[i], states[j], det)
        print("verify:", v)

    recovered = int(np.sum((L_multi == 1) & (L_single == 0)))
    lost      = int(np.sum((L_multi == 0) & (L_single == 1)))
    edges_single = int(np.sum(L_single) - N)
    edges_multi  = int(np.sum(L_multi) - N)

    print("N:", N)
    print("Local edges (single):", edges_single)
    print("Local edges (multi) :", edges_multi)
    print("Recovered edges (multi found, single missed):", recovered)
    print("Lost edges (single found, multi rejected):", lost)

if __name__ == "__main__":
    main()
