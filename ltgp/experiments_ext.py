from __future__ import annotations

from typing import Any, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .run_store import save_csv_rows, save_fig, save_npy, write_json
from .system import LTAnalyzer, LTGPSystem, embed_state_3d, random_state
import math

def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return x if math.isfinite(x) else None

def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(v)
def count_true(values):
    total = 0
    for v in values:
        if v is None:
            continue
        try:
            total += 1 if bool(v) else 0
        except Exception:
            pass
    return total

def _local_mode(vars_dict: Dict[str, Any]) -> str:
    return str(vars_dict.get('local_edge_mode', 'verified')).strip().lower()


def _summary_lines(title: str, lines: List[str]) -> str:
    return title + "\n\n" + "\n".join(lines)



def _family_graph(system: LTGPSystem, states: list[np.ndarray], p_list: np.ndarray, vars_dict: Dict[str, Any], run_dir: str, prefix: str) -> Dict[str, Any]:
    local_mode = _local_mode(vars_dict)
    n_random_starts = int(vars_dict.get('n_random_starts', 6))
    seed = int(vars_dict.get('seed', 0))
    n = len(states)
    A_g = np.zeros((n, n), dtype=int)
    A_l = np.zeros((n, n), dtype=int)
    A_p = np.zeros((n, n), dtype=int)
    rows = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tau = states[i]
            tau_p = states[j]
            g_ok, g_status, g_det = system.check_global_gp_feasible(
                tau, tau_p,
                solver=system.solver_default,
                tol=system.tol_default,
                eps_map=system.eps_eq_global,
                eps_gibbs=system.eps_gibbs,
                verbose=False,
                return_details=True,
            )
            A_g[i, j] = int(bool(g_ok))

            if local_mode == 'verified':
                l_ok, l_status, l_det = system.check_local_gp_verified(
                    tau, tau_p,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    eps_map=system.eps_eq_local,
                    eps_gibbs=system.eps_gibbs,
                    n_random_starts=n_random_starts,
                    seed=seed + i * n + j,
                    verbose=False,
                    return_details=True,
                )
            elif local_mode in ('multistart', 'multi'):
                l_ok, l_status, l_det = system.check_local_gp_feasible_multistart(
                    tau, tau_p,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    eps_map=system.eps_eq_local,
                    eps_gibbs=system.eps_gibbs,
                    n_random_starts=n_random_starts,
                    seed=seed + i * n + j,
                    verify=False,
                    verbose=False,
                    return_details=True,
                )
            else:
                l_ok, l_status, l_det = system.check_local_gp_feasible(
                    tau, tau_p,
                    solver=system.solver_default,
                    tol=system.tol_default,
                    eps_map=system.eps_eq_local,
                    eps_gibbs=system.eps_gibbs,
                    verbose=False,
                    return_details=True,
                )
            A_l[i, j] = int(bool(l_ok))

            ppt = system.check_local_gp_ppt_relaxation(
                tau, tau_p,
                solver=system.solver_default,
                tol=system.tol_default,
                eps_map=system.eps_eq_local,
                eps_gibbs=system.eps_gibbs,
                verbose=False,
            )
            A_p[i, j] = int(bool(ppt.feasible))

            D_i, I_i, _, _ = system.monotones(tau)
            D_j, I_j, _, _ = system.monotones(tau_p)
            cm_i = system.correlation_metrics(tau)
            cm_j = system.correlation_metrics(tau_p)

            # local details may come back nested under 'best' for multistart
            l_best = (l_det or {}).get('best', l_det or {}) if isinstance(l_det, dict) else {}
            l_best_det = l_best.get('details', l_best) if isinstance(l_best, dict) else {}
            l_verify = None
            if isinstance(l_det, dict):
                l_verify = l_det.get('verification', None)
                if l_verify is None and isinstance(l_best, dict):
                    l_verify = l_best.get('verify', None)
            l_verify = l_verify or {}
            l_core = l_best_det if isinstance(l_best_det, dict) else {}

            row = {
                'source': i,
                'target': j,
                'p_source': float(p_list[i]),
                'p_target': float(p_list[j]),
                'D_source': float(D_i),
                'D_target': float(D_j),
                'I_source': float(I_i),
                'I_target': float(I_j),
                'C1_source': float(cm_i['C_trace_dist']),
                'C1_target': float(cm_j['C_trace_dist']),
                'global_edge': int(A_g[i, j]),
                'local_edge': int(A_l[i, j]),
                'ppt_edge': int(A_p[i, j]),
                'local_status': str(l_status),
                'global_status': str(g_status),
                'ppt_status': str(ppt.status),
                'global_map_residual': _f((g_det or {}).get('map_residual')),
                'global_gibbs_residual': _f((g_det or {}).get('gibbs_residual')),
                'local_residual': _f(l_core.get('residual')),
                'local_threshold': _f(l_core.get('threshold')),
                'local_hint_index': int((l_det or {}).get('hint_index', -1)) if isinstance(l_det, dict) else -1,
                'local_verify_ok': int(bool(l_verify.get('ok', False))) if isinstance(l_verify, dict) else 0,
                'local_verify_map_err': _f(l_verify.get('map_err') if isinstance(l_verify, dict) else None),
                'local_verify_tp_err_A': _f(l_verify.get('tp_err_A') if isinstance(l_verify, dict) else None),
                'local_verify_tp_err_Ap': _f(l_verify.get('tp_err_Ap') if isinstance(l_verify, dict) else None),
                'local_verify_gp_err_A': _f(l_verify.get('gp_err_A') if isinstance(l_verify, dict) else None),
                'local_verify_gp_err_Ap': _f(l_verify.get('gp_err_Ap') if isinstance(l_verify, dict) else None),
                'local_verify_method_gap': _f(((l_core.get('product_application_debug') or {}) if isinstance(l_core, dict) else {}).get('product_method_gap')),
                'ppt_map_residual': _f(getattr(ppt, 'map_residual', None)),
                'ppt_gibbs_residual': _f(getattr(ppt, 'gibbs_residual', None)),
            }
            row['global_monotone_violation'] = int(bool(row['global_edge']) and (row['D_target'] > row['D_source'] + 1e-8 or row['I_target'] > row['I_source'] + 1e-8))
            row['local_without_ppt'] = int(bool(row['local_edge']) and not bool(row['ppt_edge']))
            row['local_unverified'] = int(bool(row['local_edge']) and not bool(row['local_verify_ok']))
            rows.append(row)

    save_npy(run_dir, A_g, f'{prefix}_A_global.npy')
    save_npy(run_dir, A_l, f'{prefix}_A_local.npy')
    save_npy(run_dir, A_p, f'{prefix}_A_ppt.npy')
    save_csv_rows(run_dir, rows, f'{prefix}_edges.csv')
    save_csv_rows(run_dir, [r for r in rows if r['global_monotone_violation']], f'{prefix}_global_monotone_violations.csv')
    save_csv_rows(run_dir, [r for r in rows if r['local_without_ppt']], f'{prefix}_local_without_ppt.csv')
    save_csv_rows(run_dir, [r for r in rows if r['local_unverified']], f'{prefix}_local_unverified.csv')

    for tag, A in [('global', A_g), ('local', A_l), ('ppt', A_p)]:
        fig = plt.figure()
        plt.imshow(A, interpolation='nearest')
        plt.colorbar()
        plt.xlabel('target')
        plt.ylabel('source')
        plt.title(f'{prefix}: {tag} adjacency')
        save_fig(run_dir, fig, f'{prefix}_{tag}_adjacency.png')
        plt.close(fig)

    diagnostics = {
        'global_edges': int(A_g.sum()),
        'local_edges': int(A_l.sum()),
        'ppt_edges': int(A_p.sum()),
        'global_monotone_violations': _count_true(rows, 'global_monotone_violation'),
        'local_without_ppt': _count_true(rows, 'local_without_ppt'),
        'local_unverified': _count_true(rows, 'local_unverified'),
    }
    write_json(run_dir, diagnostics, filename=f'{prefix}_diagnostics.json')
    return {'A_global': A_g, 'A_local': A_l, 'A_ppt': A_p, 'rows': rows, 'diagnostics': diagnostics}


def exp_closest_lt_distance(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    d = system.dA * system.dAp
    rho = random_state(d, seed=int(vars_dict.get('seed', 0)))
    classical = _bool(vars_dict.get('classical', False))
    sigma, dist, status = system.closest_lt_state(rho, classical=classical, solver=system.solver_default, tol=system.tol_default, verbose=False)
    rep = analyzer.analyze_single_state(rho, label='rho', solver=system.solver_default, tol=system.tol_default, verbose=False, classical=True)
    sigma_rep = analyzer.analyze_single_state(sigma, label='sigma', solver=system.solver_default, tol=system.tol_default, verbose=False, classical=True) if sigma is not None else None
    write_json(run_dir, {'rho_report': rep, 'sigma_report': sigma_rep, 'distance': dist, 'status': status}, filename='distance_report.json')
    return {'summary': _summary_lines('Distance to LT', [f'classical={classical}', f'status={status}', f'distance={dist}']), 'artifacts': {'report': 'distance_report.json'}}


def exp_lt_region_geometry(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    n = int(vars_dict.get('num_samples', 40))
    seed = int(vars_dict.get('seed', 0))
    classical = _bool(vars_dict.get('classical', False))
    extremals = analyzer.sample_extremal_lt_states(num_samples=n, classical=classical, solver=system.solver_default, tol=system.tol_default, verbose=False, seed=seed)
    pts = np.array([embed_state_3d(system, item['rho'], rng=np.random.default_rng(seed)) for item in extremals], dtype=float)
    save_npy(run_dir, pts, 'boundary_points.npy')
    rows = []
    for k, item in enumerate(extremals):
        mon = item['monotones']
        rows.append({'idx': k, 'opt_val': float(item['opt_val']), 'I': float(mon['I_rho']), 'D': float(mon['D_rho_vs_gamma'])})
    save_csv_rows(run_dir, rows, 'boundary_points.csv')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18)
    ax.set_title('LT boundary samples')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    save_fig(run_dir, fig, 'lt_boundary_3d.png')
    plt.close(fig)
    return {'summary': _summary_lines('LT boundary geometry', [f'samples={len(extremals)}', f'classical={classical}']), 'artifacts': {'points': 'boundary_points.npy', 'figure': 'lt_boundary_3d.png'}}


def exp_lt_interior_geometry(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    n = int(vars_dict.get('num_samples', 40))
    seed = int(vars_dict.get('seed', 0))
    rng = np.random.default_rng(seed)
    pts = []
    rows = []
    for k in range(n):
        rho = random_state(system.dA * system.dAp, seed=int(rng.integers(0, 10**9)))
        sigma, dist, status = system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)
        if sigma is None:
            continue
        pts.append(embed_state_3d(system, sigma, rng=rng))
        D, I, _, _ = system.monotones(sigma)
        rows.append({'idx': k, 'distance_from_random': float(dist), 'status': str(status), 'D': float(D), 'I': float(I)})
    pts = np.array(pts, dtype=float)
    save_npy(run_dir, pts, 'interior_points.npy')
    save_csv_rows(run_dir, rows, 'interior_points.csv')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=18)
    ax.set_title('Projected LT interior cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    save_fig(run_dir, fig, 'lt_interior_3d.png')
    plt.close(fig)
    return {'summary': _summary_lines('LT interior geometry', [f'samples={len(rows)}']), 'artifacts': {'points': 'interior_points.npy', 'figure': 'lt_interior_3d.png'}}


def exp_lt_geometry_combined(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    exp_lt_region_geometry(vars_dict, system, analyzer, run_dir)
    exp_lt_interior_geometry(vars_dict, system, analyzer, run_dir)
    boundary = np.load(f'{run_dir}/boundary_points.npy')
    interior = np.load(f'{run_dir}/interior_points.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if len(interior):
        ax.scatter(interior[:, 0], interior[:, 1], interior[:, 2], s=10, alpha=0.5, label='interior')
    if len(boundary):
        ax.scatter(boundary[:, 0], boundary[:, 1], boundary[:, 2], s=24, label='boundary')
    ax.legend()
    ax.set_title('LT geometry: boundary + interior')
    save_fig(run_dir, fig, 'lt_geometry_combined.png')
    plt.close(fig)
    return {'summary': _summary_lines('LT geometry combined', [f'boundary points={len(boundary)}', f'interior points={len(interior)}']), 'artifacts': {'figure': 'lt_geometry_combined.png'}}


def exp_tfd_vs_dephased(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    res = analyzer.analyze_tfd_vs_dephased(solver=system.solver_default, tol=system.tol_default, verbose=False)
    write_json(run_dir, res, filename='tfd_vs_dephased.json')
    rt = res['tfd']['monotones']
    rd = res['tfd_dephased']['monotones']
    pair = res['pair']['feasibility']
    return {
        'summary': _summary_lines('TFD vs dephased TFD', [
            f"I(TFD)={rt['I_rho']:.6g}, I(deph)={rd['I_rho']:.6g}",
            f"D(TFD)={rt['D_rho_vs_gamma']:.6g}, D(deph)={rd['D_rho_vs_gamma']:.6g}",
            f"Global GP: {pair['Global_GP']['feasible']} | {pair['Global_GP']['status']}",
            f"Local GP ({pair['Local_GP']['mode']}): {pair['Local_GP']['feasible']} | {pair['Local_GP']['status']}",
        ]),
        'artifacts': {'report': 'tfd_vs_dephased.json'},
    }


def exp_mix_with_gamma(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    try:
        rho0 = analyzer.factory.tfd_state()
        label = 'TFD'
    except Exception:
        rho0 = random_state(system.dA * system.dAp, seed=int(vars_dict.get('seed', 0)))
        label = 'random'
    lam_grid = np.linspace(0.0, 1.0, int(vars_dict.get('num_samples', 25)))
    reports = analyzer.scan_mixture_with_gamma(rho0, lam_grid, solver=system.solver_default, tol=system.tol_default, verbose=False)
    rows = [{'lambda': float(rep['lambda']), 'D': float(rep['monotones']['D_rho_vs_gamma']), 'I': float(rep['monotones']['I_rho'])} for rep in reports]
    save_csv_rows(run_dir, rows, 'mix_with_gamma.csv')
    fig = plt.figure()
    plt.plot(lam_grid, [r['D'] for r in rows], marker='o', label='D')
    plt.plot(lam_grid, [r['I'] for r in rows], marker='s', label='I')
    plt.xlabel('lambda')
    plt.legend()
    plt.title(f'mix_with_gamma ({label})')
    save_fig(run_dir, fig, 'mix_with_gamma.png')
    plt.close(fig)
    return {'summary': _summary_lines('Thermalisation path', [f'source={label}', f'samples={len(rows)}']), 'artifacts': {'csv': 'mix_with_gamma.csv', 'figure': 'mix_with_gamma.png'}}


def _exp_family_common(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str, family_kind: str) -> Dict[str, Any]:
    num_points = int(vars_dict.get('num_points', vars_dict.get('num_samples', 17)))
    include_negative = _bool(vars_dict.get('include_negative', False))
    if family_kind == 'ray':
        label = str(vars_dict.get('label', 'XX'))
        fam = analyzer.scan_lt_ray_family_pauli(label=label, num_points=num_points, include_negative=include_negative)
        prefix = f'ray_{label}'
        title = f'LT Pauli ray {label}'
    else:
        tx = float(vars_dict.get('tx', 1.0))
        ty = float(vars_dict.get('ty', 0.0))
        tz = float(vars_dict.get('tz', 0.0))
        fam = analyzer.scan_lt_diagT_family(t0=(tx, ty, tz), num_points=num_points, include_negative=include_negative)
        prefix = f'diagT_{tx:g}_{ty:g}_{tz:g}'.replace('-', 'm').replace('.', 'p')
        title = f'LT diagonal-T ray ({tx:g},{ty:g},{tz:g})'
    p_list = np.array(fam['p_list'], dtype=float)
    obs = analyzer.compute_family_observables(fam['states'])
    rows = []
    for i, p in enumerate(p_list):
        row = {'idx': i, 'p': float(p), 'D': float(obs['D'][i]), 'I': float(obs['I'][i]), 'C1': float(obs['C_trace_dist'][i]), 'CF': float(obs['C_fro'][i]), 'ppt_min_eig': float(obs['ppt_min_eig'][i])}
        svals = obs['T_svals'][i]
        if svals is not None:
            row.update({'s1': float(svals[0]), 's2': float(svals[1]), 's3': float(svals[2])})
        rows.append(row)
    save_csv_rows(run_dir, rows, f'{prefix}_family.csv')
    fig = plt.figure()
    plt.plot(p_list, obs['I'], marker='o')
    plt.xlabel('p')
    plt.ylabel('I(A:B)')
    plt.title(f'{title} — mutual information')
    save_fig(run_dir, fig, f'{prefix}_I.png')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(p_list, obs['D'], marker='o', label='D')
    plt.plot(p_list, obs['C_trace_dist'], marker='s', label='0.5||C||1')
    plt.plot(p_list, obs['C_fro'], marker='^', label='||C||F')
    plt.xlabel('p')
    plt.legend()
    plt.title(f'{title} — monotones / norms')
    save_fig(run_dir, fig, f'{prefix}_monotones.png')
    plt.close(fig)
    graph = _family_graph(system, fam['states'], p_list, vars_dict, run_dir, prefix=prefix)
    monotone_violations = [
        r for r in graph['rows']
        if r['global_edge'] and (r['D_target'] > r['D_source'] + 1e-8 or r['I_target'] > r['I_source'] + 1e-8)
    ]
    save_csv_rows(run_dir, monotone_violations, f'{prefix}_monotone_violations.csv')
    return {
        'summary': _summary_lines(title, [
            f"p_bounds={fam['p_bounds']}",
            f'samples={len(p_list)}',
            f"global edges={int(graph['A_global'].sum())}",
            f"local edges={int(graph['A_local'].sum())}",
            f"ppt edges={int(graph['A_ppt'].sum())}",
            f"global monotone violations={int(graph['diagnostics']['global_monotone_violations'])}",
            f"local edges without PPT={int(graph['diagnostics']['local_without_ppt'])}",
            f"local edges failing explicit verification={int(graph['diagnostics']['local_unverified'])}",
            f'monotone violations among global edges={len(monotone_violations)}',
        ]),
        'artifacts': {'csv': f'{prefix}_family.csv', 'figure': f'{prefix}_monotones.png'},
    }


def exp_lt_family_ray_validation(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    return _exp_family_common(vars_dict, system, analyzer, run_dir, family_kind='ray')


def exp_lt_family_diagT_validation(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    return _exp_family_common(vars_dict, system, analyzer, run_dir, family_kind='diagT')


def exp_d3_commuting_sampling(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    if system.dims != (3, 3):
        raise ValueError('d3_commuting_sampling requires dA=dAp=3')
    n = int(vars_dict.get('num_samples', 200))
    seed = int(vars_dict.get('seed', 0))
    iters = int(vars_dict.get('sinkhorn_iters', 300))
    states = system.sample_commuting_lt_d3(n=n, seed=seed, iters=iters)
    I = np.array([system.monotones(r)[1] for r in states], dtype=float)
    D = np.array([system.monotones(r)[0] for r in states], dtype=float)
    ppt_min = np.array([system.ppt_status(r)['min_pt_eig'] for r in states], dtype=float)
    fig = plt.figure()
    plt.hist(I, bins=30)
    plt.xlabel('Mutual information I(A:B)')
    plt.ylabel('count')
    plt.title('d=3 commuting LT subclass: MI histogram')
    save_fig(run_dir, fig, 'd3_commuting_MI_hist.png')
    plt.close(fig)
    save_npy(run_dir, I, 'I_values.npy')
    save_npy(run_dir, D, 'D_values.npy')
    save_npy(run_dir, ppt_min, 'ppt_min_eigs.npy')
    return {'summary': _summary_lines('d=3 commuting LT sampling', [f'n={n}, seed={seed}, sinkhorn_iters={iters}', f'mean I={float(I.mean()):.6g}, std I={float(I.std()):.6g}', f'mean D={float(D.mean()):.6g}, std D={float(D.std()):.6g}', f'mean min PT eig={float(ppt_min.mean()):.6g}']), 'artifacts': {'mi_hist': 'd3_commuting_MI_hist.png', 'I_values': 'I_values.npy', 'D_values': 'D_values.npy'}}


def exp_random_pair_gp_lgp(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    d = system.dA * system.dAp
    seed = int(vars_dict.get('seed', 0))
    tau = random_state(d, seed=seed)
    tau_p = random_state(d, seed=seed + 1)
    if _bool(vars_dict.get('project_to_lt', False)):
        tau = system.closest_lt_state(tau, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
        tau_p = system.closest_lt_state(tau_p, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
    rep = analyzer.analyze_pair(tau, tau_p, label='random_pair', solver=system.solver_default, tol=system.tol_default, verbose=False, local_mode=_local_mode(vars_dict), n_random_starts=int(vars_dict.get('n_random_starts', 6)), seed=seed)
    write_json(run_dir, rep, filename='random_pair_report.json')
    return {'summary': _summary_lines('Random pair GP vs local GP', [f"Global GP: {rep['feasibility']['Global_GP']['feasible']} | {rep['feasibility']['Global_GP']['status']}", f"Local GP ({rep['feasibility']['Local_GP']['mode']}): {rep['feasibility']['Local_GP']['feasible']} | {rep['feasibility']['Local_GP']['status']}", f"PPT outer: {rep['feasibility']['Local_PPT_outer']}" ]), 'artifacts': {'report': 'random_pair_report.json'}}


def exp_local_gp_ppt_relax(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    d = system.dA * system.dAp
    seed = int(vars_dict.get('seed', 0))
    tau = random_state(d, seed=seed)
    tau_p = random_state(d, seed=seed + 1)
    if _bool(vars_dict.get('project_to_lt', True)):
        tau = analyzer.analyze_single_state(tau, label='tau', solver=system.solver_default, tol=system.tol_default, verbose=False)['distance_to_LT']['sigma_closest']
        tau_p = analyzer.analyze_single_state(tau_p, label='tau_p', solver=system.solver_default, tol=system.tol_default, verbose=False)['distance_to_LT']['sigma_closest']
    relax = system.check_local_gp_ppt_relaxation(tau, tau_p, solver=system.solver_default, tol=system.tol_default, eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs, verbose=False)
    l_ok, l_status, _ = system.check_local_gp_verified(tau, tau_p, solver=system.solver_default, tol=system.tol_default, eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs, n_random_starts=int(vars_dict.get('n_random_starts', 6)), seed=seed, verbose=False, return_details=True)
    summary = _summary_lines('Local GP outer relaxation (PPT on joint Choi)', [f'PPT-relax feasible: {relax.feasible} (status={relax.status})', f'map_residual={relax.map_residual:.3e}, gibbs_residual={relax.gibbs_residual:.3e}', f'verified local feasible: {bool(l_ok)} (status={l_status})', 'Interpretation: infeasible PPT ⇒ infeasible local GP; feasible PPT is only an outer bound.'])
    return {'summary': summary, 'artifacts': {}}


def exp_extract_global_channel(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    d = system.dA * system.dAp
    seed = int(vars_dict.get('seed', 0))
    tau = random_state(d, seed=seed)
    tau_p = random_state(d, seed=seed + 1)
    if _bool(vars_dict.get('project_to_lt', True)):
        tau = system.closest_lt_state(tau, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
        tau_p = system.closest_lt_state(tau_p, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
    details = system.find_global_gp_channel(tau, tau_p, solver=system.solver_default, tol=system.tol_default, eps_gibbs=system.eps_gibbs, verbose=False)
    J = details.get('J')
    if J is not None:
        save_npy(run_dir, J, 'global_channel_choi.npy')
        diag = system.choi_diagnostics(J, d_in=d, d_out=d, gamma_in=np.kron(system.gammaA, system.gammaAp), gamma_out=np.kron(system.gammaA, system.gammaAp))
    else:
        diag = None
    write_json(run_dir, {'details': details, 'diagnostics': diag}, filename='global_channel_report.json')
    return {'summary': _summary_lines('Extract global GP channel', [f"status={details.get('status')}", f"map_residual={details.get('map_residual')}", f"gibbs_residual={details.get('gibbs_residual')}" ]), 'artifacts': {'choi': 'global_channel_choi.npy' if J is not None else None, 'report': 'global_channel_report.json'}}


def exp_lt_convertibility_graph(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    n = int(vars_dict.get('num_samples', 8))
    seed = int(vars_dict.get('seed', 0))
    family = str(vars_dict.get('ensemble', 'mixed')).lower()
    rng = np.random.default_rng(seed)
    states = []
    labels = []
    if family in ('diagt', 'diag_t') and system.dims == (2, 2):
        t0 = (float(vars_dict.get('tx', 1.0)), float(vars_dict.get('ty', 0.0)), float(vars_dict.get('tz', 0.0)))
        fam = analyzer.scan_lt_diagT_family(t0=t0, num_points=n, include_negative=_bool(vars_dict.get('include_negative', False)))
        states = fam['states']
        labels = [f'p={p:.3g}' for p in fam['p_list']]
        p_list = np.array(fam['p_list'], dtype=float)
    else:
        if system.dims == (2, 2):
            try:
                states.append(analyzer.factory.tfd_state())
                labels.append('TFD')
                states.append(system.dephase_global_in_energy_basis(states[0]))
                labels.append('TFD_dephased')
            except Exception:
                pass
        while len(states) < n:
            rho = random_state(system.dA * system.dAp, seed=int(rng.integers(0, 10**9)))
            sigma = system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
            states.append(sigma)
            labels.append(f'ext_{len(states)-1}')
        p_list = np.arange(len(states), dtype=float)
    graph = _family_graph(system, states, p_list, vars_dict, run_dir, prefix='convertibility')
    save_csv_rows(run_dir, [{'idx': i, 'label': labels[i]} for i in range(len(labels))], 'vertices.csv')
    return {'summary': _summary_lines('Convertibility graph', [f'vertices={len(states)}', f"global edges={int(graph['A_global'].sum())}", f"local edges={int(graph['A_local'].sum())}", f"ppt edges={int(graph['A_ppt'].sum())}", f"local edges without PPT={int(graph['diagnostics']['local_without_ppt'])}", f"local edges failing explicit verification={int(graph['diagnostics']['local_unverified'])}", f'local mode={_local_mode(vars_dict)}']), 'artifacts': {'global_adj': 'convertibility_global_adjacency.png', 'local_adj': 'convertibility_local_adjacency.png', 'ppt_adj': 'convertibility_ppt_adjacency.png', 'edge_csv': 'convertibility_edges.csv', 'diagnostics': 'convertibility_diagnostics.json'}}


def exp_sanity_checks(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    rows = []
    d = system.dA * system.dAp
    seed = int(vars_dict.get('seed', 0))
    examples = [('gamma', np.kron(system.gammaA, system.gammaAp), np.kron(system.gammaA, system.gammaAp))]
    if system.dims == (2, 2):
        try:
            tfd = analyzer.factory.tfd_state()
            examples.append(('TFD->dephased', tfd, system.dephase_global_in_energy_basis(tfd)))
        except Exception:
            pass
    rho = random_state(d, seed=seed)
    sigma = system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
    examples.append(('random->LTproj', rho, sigma))
    for label, tau, tau_p in examples:
        rep = analyzer.analyze_pair(tau, tau_p, label=label, solver=system.solver_default, tol=system.tol_default, verbose=False, local_mode=_local_mode(vars_dict), n_random_starts=int(vars_dict.get('n_random_starts', 6)), seed=seed)
        rows.append({'label': label, 'LT_tau': rep['LT_tau'], 'LT_taup': rep['LT_taup'], 'D_tau': rep['monotones']['D_tau_vs_gamma'], 'D_taup': rep['monotones']['D_taup_vs_gamma'], 'I_tau': rep['monotones']['I_tau'], 'I_taup': rep['monotones']['I_taup'], 'global_feasible': rep['feasibility']['Global_GP']['feasible'], 'local_feasible': rep['feasibility']['Local_GP']['feasible'], 'ppt_outer': None if rep['feasibility']['Local_PPT_outer'] is None else rep['feasibility']['Local_PPT_outer']['feasible']})
    save_csv_rows(run_dir, rows, 'sanity_checks.csv')
    return {'summary': _summary_lines('Sanity checks', [f'rows={len(rows)}', 'Saved sanity_checks.csv']), 'artifacts': {'csv': 'sanity_checks.csv'}}


def exp_separable_vs_entangled_lt(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    n = int(vars_dict.get('num_samples', 50))
    seed = int(vars_dict.get('seed', 0))
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        rho = random_state(system.dA * system.dAp, seed=int(rng.integers(0, 10**9)))
        sigma = system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
        ppt = system.ppt_status(sigma)
        D, I, _, _ = system.monotones(sigma)
        rows.append({'idx': k, 'D': float(D), 'I': float(I), 'is_ppt': int(bool(ppt['is_ppt'])), 'min_pt_eig': float(ppt['min_pt_eig']), 'separable_if_low_dim': ppt['separable_if_low_dim']})
    save_csv_rows(run_dir, rows, 'lt_ppt_classification.csv')
    fig = plt.figure()
    plt.scatter([r['D'] for r in rows], [r['I'] for r in rows], c=[r['is_ppt'] for r in rows])
    plt.xlabel('D')
    plt.ylabel('I')
    plt.title('LT samples: PPT classification')
    save_fig(run_dir, fig, 'lt_ppt_classification.png')
    plt.close(fig)
    ppt_fraction = float(np.mean([r['is_ppt'] for r in rows])) if rows else 0.0
    return {'summary': _summary_lines('PPT / separability in LT', [f'samples={n}', f'PPT fraction={ppt_fraction:.3f}']), 'artifacts': {'csv': 'lt_ppt_classification.csv', 'figure': 'lt_ppt_classification.png'}}



def exp_local_gp_closure_test(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    n = int(vars_dict.get('num_samples', 20))
    seed = int(vars_dict.get('seed', 0))
    rng = np.random.default_rng(seed)
    rows = []
    weird = []
    for k in range(n):
        rho = random_state(system.dA * system.dAp, seed=int(rng.integers(0, 10**9)))
        sigma = system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0]
        J_A, stA, diagA = system.find_random_local_gp_channel(which='A', solver=system.solver_default, tol=system.tol_default, eps_gibbs=system.eps_gibbs, seed=int(rng.integers(0, 10**9)))
        J_B, stB, diagB = system.find_random_local_gp_channel(which='Ap', solver=system.solver_default, tol=system.tol_default, eps_gibbs=system.eps_gibbs, seed=int(rng.integers(0, 10**9)))
        
        # Debug: check if the random local channels actually preserve the marginals as they should
        # Phi_gamma_A = system.choi_apply_numpy(J_A, system.gammaA, d_in=system.dA, d_out=system.dA)
        # errA = np.linalg.norm(Phi_gamma_A - system.gammaA, 'fro')
        # Phi_gamma_B = system.choi_apply_numpy(J_B, system.gammaAp, d_in=system.dAp, d_out=system.dAp)
        # errB = np.linalg.norm(Phi_gamma_B - system.gammaAp, 'fro')
        # print("Err A and B for random local channels:")
        # print(errA)
        # print(errB)
        # Ks = system.kraus_from_choi(J_A, d_in=system.dA, d_out=system.dA)
        # gammaA_k = sum(K @ system.gammaA @ K.conj().T for K in Ks)
        # errA_k = np.linalg.norm(gammaA_k - system.gammaA, 'fro')
        # print("Gamma and Err A from Kraus reconstruction:")
        # print(gammaA_k,"\n", errA_k)
        # out1 = system.choi_apply_numpy(J, gamma, d_in=2, d_out=2)
        # out2 = sum(K @ gamma @ K.conj().T for K in system.kraus_from_choi(J, 2, 2))
        out1 = system.choi_apply_numpy(J, gamma, d_in=2, d_out=2)
        Ks = system.kraus_from_choi(J, 2, 2)
        out2 = sum(K @ gamma @ K.conj().T for K in Ks)
        gap = np.linalg.norm(out1 - out2, 'fro')
        assert gap < 1e-8
        # print(np.linalg.norm(out1 - out2, 'fro'))
        
        if J_A is None or J_B is None:
            rows.append({'idx': k, 'constructed': 0, 'status_A': str(stA), 'status_B': str(stB)})
            weird.append({'idx': k, 'reason': 'missing_random_channel', 'status_A': str(stA), 'status_B': str(stB)})
            continue
        out = system.apply_local_product_channel(sigma, J_A, J_B)
        lt_ok, _, _, margA, margB = system.lt_membership(out, tol=1e-6)
        debug = system.local_channel_application_debug(sigma, J_A=J_A, J_Ap=J_B)
        out_report = debug.get('product_out_report', {})
        row = {
            'idx': k,
            'constructed': 1,
            'lt_ok': int(bool(lt_ok)),
            'status_A': str(stA),
            'status_B': str(stB),
            'errA': float(np.linalg.norm(margA - system.gammaA, 'fro')),
            'errB': float(np.linalg.norm(margB - system.gammaAp, 'fro')),
            'trace_err': float(abs(np.trace(out) - 1.0)),
            'min_eig_out': _f(out_report.get('min_eig')),
            'D_minus_I_out': _f(out_report.get('D_minus_I')),
            'A_method_gap': _f(debug.get('A_method_gap')),
            'Ap_method_gap': _f(debug.get('Ap_method_gap')),
            'product_method_gap': _f(debug.get('product_method_gap')),
            'A_tp_err': _f((debug.get('A_diag') or {}).get('tp_fro_err')),
            'Ap_tp_err': _f((debug.get('Ap_diag') or {}).get('tp_fro_err')),
            'A_gp_err': _f((debug.get('A_diag') or {}).get('gp_fro_err')),
            'Ap_gp_err': _f((debug.get('Ap_diag') or {}).get('gp_fro_err')),
            'A_min_eig_J': _f((debug.get('A_diag') or {}).get('min_eig_J')),
            'Ap_min_eig_J': _f((debug.get('Ap_diag') or {}).get('min_eig_J')),
        }
        rows.append(row)
        if (not lt_ok) or row['product_method_gap'] > 1e-7 or row['A_method_gap'] > 1e-7 or row['Ap_method_gap'] > 1e-7:
            weird.append(dict(row))
    save_csv_rows(run_dir, rows, 'local_gp_closure.csv')
    save_csv_rows(run_dir, weird, 'local_gp_closure_weird.csv')
    diagnostics = {
        'tests_requested': n,
        'tests_completed': len(rows),
        'constructed_channels': int(sum(1 for r in rows if int(r.get('constructed', 1)) == 1)),
        'pass_count': int(sum(1 for r in rows if int(r.get('lt_ok', 0)) == 1)),
        'pass_fraction': float(np.mean([r.get('lt_ok', 0) for r in rows if int(r.get('constructed', 1)) == 1])) if any(int(r.get('constructed', 1)) == 1 for r in rows) else 0.0,
        'max_A_method_gap': max([abs(r.get('A_method_gap', 0.0)) for r in rows if 'A_method_gap' in r] or [0.0]),
        'max_Ap_method_gap': max([abs(r.get('Ap_method_gap', 0.0)) for r in rows if 'Ap_method_gap' in r] or [0.0]),
        'max_product_method_gap': max([abs(r.get('product_method_gap', 0.0)) for r in rows if 'product_method_gap' in r] or [0.0]),
        'max_marginal_err_A': max([abs(r.get('errA', 0.0)) for r in rows if 'errA' in r] or [0.0]),
        'max_marginal_err_Ap': max([abs(r.get('errB', 0.0)) for r in rows if 'errB' in r] or [0.0]),
        'weird_count': len(weird),
    }
    write_json(run_dir, diagnostics, filename='local_gp_closure_diagnostics.json')
    return {'summary': _summary_lines('LT closure under random local GP channels', [f'tests={len(rows)}', f"constructed channels={diagnostics['constructed_channels']}", f"pass fraction={diagnostics['pass_fraction']:.3f}", f"max product-method gap={diagnostics['max_product_method_gap']:.3e}", f"max marginal error A={diagnostics['max_marginal_err_A']:.3e}", f"max marginal error A'={diagnostics['max_marginal_err_Ap']:.3e}", f"weird rows saved={len(weird)}"]), 'artifacts': {'csv': 'local_gp_closure.csv', 'weird_csv': 'local_gp_closure_weird.csv', 'diagnostics': 'local_gp_closure_diagnostics.json'}, 'diagnostics': diagnostics}



def exp_verified_local_edge_audit(vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    n = int(vars_dict.get('num_samples', 6))
    seed = int(vars_dict.get('seed', 0))
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        rho = random_state(system.dA * system.dAp, seed=int(rng.integers(0, 10**9)))
        states.append(system.closest_lt_state(rho, classical=False, solver=system.solver_default, tol=system.tol_default, verbose=False)[0])
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            h_ok, h_status, h_det = system.check_local_gp_feasible(states[i], states[j], solver=system.solver_default, tol=system.tol_default, eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs, verbose=False, return_details=True)
            v_ok, v_status, v_det = system.check_local_gp_verified(states[i], states[j], solver=system.solver_default, tol=system.tol_default, eps_map=system.eps_eq_local, eps_gibbs=system.eps_gibbs, n_random_starts=int(vars_dict.get('n_random_starts', 6)), seed=seed + i * n + j, verbose=False, return_details=True)
            v_best = (v_det or {}).get('best', v_det or {}) if isinstance(v_det, dict) else {}
            v_verify = (v_best.get('verify') if isinstance(v_best, dict) else None) or (v_det.get('verification') if isinstance(v_det, dict) else None) or {}
            rows.append({
                'source': i,
                'target': j,
                'heuristic_ok': int(bool(h_ok)),
                'verified_ok': int(bool(v_ok)),
                'heuristic_status': str(h_status),
                'verified_status': str(v_status),
                'heuristic_residual': _f((h_det or {}).get('residual')),
                'verified_best_residual': _f(((v_best.get('details') or {}) if isinstance(v_best, dict) else {}).get('residual')),
                'verified_check_ok': int(bool(v_verify.get('ok', False))) if isinstance(v_verify, dict) else 0,
                'verified_map_err': _f(v_verify.get('map_err') if isinstance(v_verify, dict) else None),
                'verified_tp_err_A': _f(v_verify.get('tp_err_A') if isinstance(v_verify, dict) else None),
                'verified_tp_err_Ap': _f(v_verify.get('tp_err_Ap') if isinstance(v_verify, dict) else None),
            })
    save_csv_rows(run_dir, rows, 'verified_local_edge_audit.csv')
    suspicious = [r for r in rows if r['verified_ok'] and not r['verified_check_ok']]
    save_csv_rows(run_dir, suspicious, 'verified_local_edge_audit_suspicious.csv')
    fp = sum(1 for r in rows if r['heuristic_ok'] and not r['verified_ok'])
    return {'summary': _summary_lines('Verified local edge audit', [f'pairs={len(rows)}', f'heuristic false positives={fp}', f'verified edges failing explicit check={len(suspicious)}']), 'artifacts': {'csv': 'verified_local_edge_audit.csv', 'suspicious_csv': 'verified_local_edge_audit_suspicious.csv'}}


def dispatch_experiment(eq_id: str, vars_dict: Dict[str, Any], system: LTGPSystem, analyzer: LTAnalyzer, run_dir: str) -> Dict[str, Any]:
    table = {
        'closest_lt_distance': exp_closest_lt_distance,
        'lt_region_geometry': exp_lt_region_geometry,
        'lt_interior_geometry': exp_lt_interior_geometry,
        'lt_geometry_combined': exp_lt_geometry_combined,
        'tfd_vs_dephased': exp_tfd_vs_dephased,
        'mix_with_gamma': exp_mix_with_gamma,
        'lt_family_ray_validation': exp_lt_family_ray_validation,
        'lt_family_diagT_validation': exp_lt_family_diagT_validation,
        'd3_commuting_sampling': exp_d3_commuting_sampling,
        'random_pair_gp_lgp': exp_random_pair_gp_lgp,
        'lt_convertibility_graph': exp_lt_convertibility_graph,
        'local_gp_ppt_relax': exp_local_gp_ppt_relax,
        'extract_global_channel': exp_extract_global_channel,
        'sanity_checks': exp_sanity_checks,
        'separable_vs_entangled_lt': exp_separable_vs_entangled_lt,
        'local_gp_closure_test': exp_local_gp_closure_test,
        'verified_local_edge_audit': exp_verified_local_edge_audit,
    }
    if eq_id == 'custom':
        payload = vars_dict.get('custom_payload', vars_dict)
        return {'summary': 'Custom payload received. No custom executor configured.', 'artifacts': {'payload': payload}}
    if eq_id not in table:
        raise KeyError(f'Unknown experiment id: {eq_id}')
    return table[eq_id](vars_dict, system, analyzer, run_dir)
