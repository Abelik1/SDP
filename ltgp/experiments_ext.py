from __future__ import annotations

from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt

from .run_store import save_fig, save_npy


def exp_d3_commuting_sampling(vars_dict: Dict[str, Any], system, analyzer, run_dir: str) -> Dict[str, Any]:
    if system.dims != (3, 3):
        raise ValueError("d3_commuting_sampling requires dA=dAp=3")

    n = int(vars_dict.get("num_samples", 200))
    seed = int(vars_dict.get("seed", 0))
    iters = int(vars_dict.get("sinkhorn_iters", 300))

    states = system.sample_commuting_lt_d3(n=n, seed=seed, iters=iters)
    I = np.array([system.monotones(r)[1] for r in states], dtype=float)
    D = np.array([system.monotones(r)[0] for r in states], dtype=float)

    fig = plt.figure()
    plt.hist(I, bins=30)
    plt.xlabel("Mutual information I(A:B)")
    plt.ylabel("count")
    plt.title("d=3 commuting LT subclass: MI histogram")
    fig_path = save_fig(run_dir, fig, "d3_commuting_MI_hist.png")
    plt.close(fig)

    save_npy(run_dir, I, "I_values.npy")
    save_npy(run_dir, D, "D_values.npy")

    return {
        "summary": (
            f"d=3 commuting LT sampling (n={n}, seed={seed}, sinkhorn_iters={iters})\n"
            f"mean I = {float(I.mean()):.6g}, std I = {float(I.std()):.6g}\n"
            f"mean D = {float(D.mean()):.6g}, std D = {float(D.std()):.6g}\n"
            "Saved: I_values.npy, D_values.npy, d3_commuting_MI_hist.png"
        ),
        "artifacts": {
            "mi_hist": fig_path,
            "I_values": "I_values.npy",
            "D_values": "D_values.npy",
        },
    }


def exp_local_gp_ppt_relax(vars_dict: Dict[str, Any], system, analyzer, run_dir: str) -> Dict[str, Any]:
    d = system.dA * system.dAp
    project_to_lt = bool(vars_dict.get("project_to_lt", True))

    from utils import random_state

    tau = random_state(d)
    tau_p = random_state(d)

    if project_to_lt:
        rep1 = analyzer.analyze_state(tau, label="tau", solver=system.solver_default, tol=system.tol_default, verbose=False)
        rep2 = analyzer.analyze_state(tau_p, label="tau_p", solver=system.solver_default, tol=system.tol_default, verbose=False)
        tau = rep1["distance_to_LT"]["sigma_closest"] or tau
        tau_p = rep2["distance_to_LT"]["sigma_closest"] or tau_p

    eps_map = float(vars_dict.get("eps_eq_local", system.eps_eq_local))
    eps_g = float(vars_dict.get("eps_gibbs", system.eps_gibbs))

    relax = system.check_local_gp_ppt_relaxation(
        tau, tau_p,
        solver=system.solver_default,
        tol=system.tol_default,
        eps_map=eps_map,
        eps_gibbs=eps_g,
        verbose=False,
    )

    try:
        l_ok, l_status = system.check_local_gp_feasible(
            tau, tau_p,
            solver=system.solver_default,
            tol=system.tol_default,
            eps_map=eps_map,
            eps_gibbs=eps_g,
            verbose=False,
            return_details=False,
        )
    except Exception as e:
        l_ok, l_status = False, f"error: {e}"

    summary = (
        "Local GP outer relaxation (PPT on joint Choi)\n\n"
        f"PPT-relax feasible: {relax.feasible} (status={relax.status})\n"
        f"  map_residual={relax.map_residual:.3e}, gibbs_residual={relax.gibbs_residual:.3e}\n\n"
        f"Heuristic local feasible: {bool(l_ok)} (status={l_status})\n\n"
        "Interpretation:\n"
        "- infeasible PPT ⇒ infeasible local GP (necessary condition)\n"
        "- feasible PPT is NOT sufficient (outer relaxation)\n"
    )

    return {"summary": summary, "artifacts": {}}