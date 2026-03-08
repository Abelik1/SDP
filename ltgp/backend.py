from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .experiments_ext import dispatch_experiment
from .run_store import make_run_dir, write_json, write_text
from .system import build_system_and_analyzer, parse_variables_string


def _augment_summary_text(out: Dict[str, Any], run_config: Dict[str, Any]) -> str:
    parts = []
    run_meta = [
        f"eq_id={out.get('eq_id')}",
        f"eq_name={out.get('eq_name')}",
        f"run_dir={out.get('run_dir')}",
    ]
    parsed = run_config.get('parsed_variables', {}) or {}
    if parsed:
        keys = ['dA', 'dAp', 'beta', 'solver', 'eps_eq_global', 'eps_eq_local', 'eps_gibbs', 'num_samples', 'seed', 'local_edge_mode', 'n_random_starts']
        for k in keys:
            if k in parsed:
                run_meta.append(f"{k}={parsed[k]}")
    parts.append("Run metadata\n\n" + "\n".join(run_meta))

    parts.append(str(out.get('summary', '(no summary)')))

    diagnostics = out.get('diagnostics', None)
    if diagnostics:
        diag_lines = [f"{k}={v}" for k, v in diagnostics.items()]
        parts.append("Diagnostics\n\n" + "\n".join(diag_lines))

    warnings = out.get('warnings', None)
    if warnings:
        if isinstance(warnings, (list, tuple)):
            warn_lines = [str(x) for x in warnings]
        else:
            warn_lines = [str(warnings)]
        parts.append("Warnings\n\n" + "\n".join(warn_lines))

    return "\n\n".join(parts).strip() + "\n"


def _ensure_system_from_vars(vars_dict: Dict[str, Any], system, analyzer):
    dA = int(vars_dict.get('dA', system.dA))
    dAp = int(vars_dict.get('dAp', system.dAp))
    beta = float(vars_dict.get('beta', system.beta))
    symmetric_flag = bool(vars_dict.get('symmetric', True))
    if symmetric_flag and dA != dAp:
        symmetric_flag = False
    reset_system = bool(vars_dict.get('reset_system', False))
    solver_override = str(vars_dict.get('solver', system.solver_default))
    eps_eq_global = float(vars_dict.get('eps_eq_global', system.eps_eq_global))
    eps_eq_local = float(vars_dict.get('eps_eq_local', system.eps_eq_local))
    eps_gibbs = float(vars_dict.get('eps_gibbs', getattr(system, 'eps_gibbs', 1e-8)))
    needs_rebuild = reset_system or (system.dA != dA) or (system.dAp != dAp) or (abs(system.beta - beta) > 1e-15) or (str(system.solver_default) != solver_override) or (not np.allclose(system.H_Ap, system.H_A) and symmetric_flag)
    if needs_rebuild:
        system, analyzer = build_system_and_analyzer(dA=dA, dAp=dAp, beta=beta, solver=solver_override, tol=system.tol_default, symmetric=symmetric_flag, eps_eq_global=eps_eq_global, eps_eq_local=eps_eq_local, eps_gibbs=eps_gibbs)
    else:
        system.eps_eq_global = eps_eq_global
        system.eps_eq_local = eps_eq_local
        system.eps_gibbs = eps_gibbs
        system.solver_default = solver_override
    seed = int(vars_dict.get('seed', -1))
    if seed >= 0:
        np.random.seed(seed)
    return system, analyzer


def backend_run(config: Dict[str, Any], system, analyzer) -> Dict[str, Any]:
    eq_id = str(config.get('selected_equation_id'))
    eq_name = str(config.get('selected_equation_name', eq_id))
    vars_str = config.get('variables_str', config.get('variables', ''))
    vars_dict = parse_variables_string(vars_str)
    if config.get('custom_function'):
        vars_dict['custom_payload'] = config.get('custom_function')
    run_config = dict(config)
    run_config['parsed_variables'] = dict(vars_dict)
    run_dir = make_run_dir(base='results', eq_id=eq_id, config=run_config)
    write_json(run_dir, run_config, filename='config.json')
    system, analyzer = _ensure_system_from_vars(vars_dict, system, analyzer)
    res = dispatch_experiment(eq_id, vars_dict, system, analyzer, run_dir)
    out = {'eq_id': eq_id, 'eq_name': eq_name, 'run_dir': run_dir, **(res or {})}
    summary_text = _augment_summary_text(out, run_config)
    write_text(run_dir, summary_text, filename='summary.txt')
    summary_json = {
        'eq_id': eq_id,
        'eq_name': eq_name,
        'run_dir': run_dir,
        'artifacts': out.get('artifacts', {}),
        'summary': out.get('summary', ''),
        'summary_text': summary_text,
        'diagnostics': out.get('diagnostics', {}),
        'warnings': out.get('warnings', []),
        'parsed_variables': run_config.get('parsed_variables', {}),
    }
    write_json(run_dir, summary_json, filename='summary.json')
    return out
