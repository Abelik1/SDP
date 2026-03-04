from __future__ import annotations

import os
import time
from typing import Any, Dict

import numpy as np

from utils import build_system_and_analyzer, parse_variables_string
from .run_store import make_run_dir, write_json, write_text


def _list_recent_png(since_ts: float) -> list[str]:
    out = []
    if not os.path.isdir("png"):
        return out
    for name in os.listdir("png"):
        if not name.lower().endswith(".png"):
            continue
        path = os.path.join("png", name)
        try:
            if os.path.getmtime(path) >= since_ts:
                out.append(path)
        except Exception:
            pass
    return sorted(out)


def _ensure_system_from_vars(vars_dict: Dict[str, Any], system, analyzer):
    dA = int(vars_dict.get("dA", system.dA))
    dAp = int(vars_dict.get("dAp", system.dAp))
    beta = float(vars_dict.get("beta", system.beta))

    symmetric_flag = bool(vars_dict.get("symmetric", True))
    if symmetric_flag and dA != dAp:
        symmetric_flag = False

    reset_system = bool(vars_dict.get("reset_system", False))
    solver_override = str(vars_dict.get("solver", system.solver_default))

    eps_eq_global = float(vars_dict.get("eps_eq_global", system.eps_eq_global))
    eps_eq_local = float(vars_dict.get("eps_eq_local", system.eps_eq_local))
    eps_gibbs = float(vars_dict.get("eps_gibbs", getattr(system, "eps_gibbs", 1e-8)))

    needs_rebuild = (
        reset_system
        or (system.dA != dA)
        or (system.dAp != dAp)
        or (abs(system.beta - beta) > 1e-15)
        or (str(system.solver_default) != solver_override)
        or (not np.allclose(system.H_Ap, system.H_A) and symmetric_flag)
    )

    if needs_rebuild:
        system, analyzer = build_system_and_analyzer(
            dA=dA,
            dAp=dAp,
            beta=beta,
            solver=solver_override,
            tol=system.tol_default,
            symmetric=symmetric_flag,
            eps_eq_global=eps_eq_global,
            eps_eq_local=eps_eq_local,
            eps_gibbs=eps_gibbs,
        )
    else:
        system.eps_eq_global = eps_eq_global
        system.eps_eq_local = eps_eq_local
        system.eps_gibbs = eps_gibbs
        system.solver_default = solver_override

    seed = int(vars_dict.get("seed", -1))
    if seed >= 0:
        np.random.seed(seed)

    return system, analyzer


def backend_run(config: Dict[str, Any], system, analyzer) -> Dict[str, Any]:
    from .experiments_ext import exp_d3_commuting_sampling, exp_local_gp_ppt_relax

    eq_id = config.get("selected_equation_id")
    eq_name = config.get("selected_equation_name", "")
    vars_str = config.get("variables_str", config.get("variables", ""))

    vars_dict = parse_variables_string(vars_str)

    run_config = dict(config)
    run_config["parsed_variables"] = dict(vars_dict)
    run_dir = make_run_dir(base="results", eq_id=str(eq_id), config=run_config)
    write_json(run_dir, run_config, filename="config.json")

    system, analyzer = _ensure_system_from_vars(vars_dict, system, analyzer)

    started = time.time()

    if eq_id == "d3_commuting_sampling":
        res = exp_d3_commuting_sampling(vars_dict, system, analyzer, run_dir)
    elif eq_id == "local_gp_ppt_relax":
        res = exp_local_gp_ppt_relax(vars_dict, system, analyzer, run_dir)
    else:
        import experiments as legacy
        legacy.backend_run(config, system, analyzer)

        new_png = _list_recent_png(started)
        res = {
            "summary": (
                f"Finished: {eq_id} ({eq_name}).\n"
                f"Run dir: {run_dir}\n\n"
                "Legacy experiment executed. See console logs.\n"
                + (f"New PNGs: {len(new_png)}\n" + "\n".join(new_png) if new_png else "")
            ),
            "artifacts": {"png": new_png},
        }

    out = {
        "eq_id": str(eq_id),
        "eq_name": str(eq_name),
        "run_dir": run_dir,
        **(res or {}),
    }
    write_text(run_dir, out.get("summary", "(no summary)"), filename="summary.txt")
    return out