from __future__ import annotations

import json
import traceback
from pathlib import Path

from ltgp.backend import backend_run
from ltgp.registry import get_catalog
from ltgp.system import build_system_and_analyzer


SOLVER = "SCS"
BETA = 1.0
LOCAL_EDGE_MODE = "verified"   # keep this
N_RANDOM_STARTS = 8
BASE_NUM_SAMPLES = 40
BASE_NUM_POINTS = 17
SEED0 = 12345


EXPERIMENT_CONFIGS = {
    # "closest_lt_distance": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 1,
    #     "classical": False,
    # },
    # "lt_region_geometry": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 2,
    #     "num_samples": 150,
    #     "classical": False,
    # },
    # "lt_interior_geometry": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 3,
    #     "num_samples": 200,
    # },
    # "lt_geometry_combined": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 4,
    #     "num_samples": 200,
    #     "classical": False,
    # },
    # "tfd_vs_dephased": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 5,
    # },
    # "mix_with_gamma": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 6,
    #     "num_samples": 31,
    # },
    "lt_family_ray_validation": { #long
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 7,
        "num_points": 21,
        "include_negative": True,
        "label": "XX",
    },
    "lt_family_diagT_validation": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 8,
        "num_points": 21,
        "include_negative": True,
        "tx": 1.0,
        "ty": 0.0,
        "tz": 0.0,
    },
    # "d3_commuting_sampling": {
    #     "dA": 3, "dAp": 3,
    #     "seed": SEED0 + 9,
    #     "num_samples": 250,
    #     "sinkhorn_iters": 300,
    # },
    # "random_pair_gp_lgp": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 10,
    #     "project_to_lt": True,
    # },
    # "lt_convertibility_graph": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 11,
    #     "num_samples": 8,
    #     "ensemble": "mixed",
    #     "include_negative": True,
    # },
    # "local_gp_ppt_relax": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 12,
    #     "project_to_lt": True,
    # },
    # "extract_global_channel": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 13,
    #     "project_to_lt": True,
    # },
    # "sanity_checks": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 14,
    # },
    # "separable_vs_entangled_lt": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 15,
    #     "num_samples": 80,
    # },
    "local_gp_closure_test": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 16,
        "num_samples": 200,
    },
    # "verified_local_edge_audit": {
    #     "dA": 2, "dAp": 2,
    #     "seed": SEED0 + 17,
    #     "num_samples": 8,
    # },
}


def vars_dict_to_string(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def flatten_catalog_ids() -> list[str]:
    ids = []
    for _, specs in get_catalog().items():
        for spec in specs:
            if spec.eq_id == "custom":
                continue
            ids.append(spec.eq_id)
    return ids


def build_base_vars(dA: int, dAp: int) -> dict:
    return {
        "dA": dA,
        "dAp": dAp,
        "beta": BETA,
        "solver": SOLVER,
        "symmetric": True,
        "reset_system": True,
        "local_edge_mode": LOCAL_EDGE_MODE,
        "n_random_starts": N_RANDOM_STARTS,
        "eps_eq_global": 1e-6,
        "eps_eq_local": 1e-6,
        "eps_gibbs": 1e-8,
        "num_samples": BASE_NUM_SAMPLES,
        "num_points": BASE_NUM_POINTS,
    }


def main() -> None:
    out_root = Path("batch_logs")
    out_root.mkdir(exist_ok=True)

    experiment_ids = flatten_catalog_ids()

    completed = []
    failed = []

    system_cache = {}

    print("=" * 80)
    print("Running all registered experiments")
    print(f"Solver: {SOLVER}")
    print(f"Local mode: {LOCAL_EDGE_MODE}")
    print("=" * 80)

    for i, eq_id in enumerate(experiment_ids, start=1):
        if eq_id not in EXPERIMENT_CONFIGS:
            failed.append({
                "eq_id": eq_id,
                "error": "Missing config in EXPERIMENT_CONFIGS"
            })
            print(f"[{i}/{len(experiment_ids)}] SKIP {eq_id}: missing config")
            continue

        cfg = dict(EXPERIMENT_CONFIGS[eq_id])
        dA = int(cfg["dA"])
        dAp = int(cfg["dAp"])

        base_vars = build_base_vars(dA, dAp)
        base_vars.update(cfg)

        key = (dA, dAp)
        if key not in system_cache:
            system_cache[key] = build_system_and_analyzer(
                dA=dA,
                dAp=dAp,
                beta=BETA,
                solver=SOLVER,
                symmetric=True,
                eps_eq_global=1e-6,
                eps_eq_local=1e-6,
                eps_gibbs=1e-8,
            )
        system, analyzer = system_cache[key]

        config = {
            "selected_equation_id": eq_id,
            "selected_equation_name": eq_id,
            "variables_str": vars_dict_to_string(base_vars),
        }

        print(f"[{i}/{len(experiment_ids)}] START {eq_id} (dA=dAp={dA})")

        try:
            result = backend_run(config, system, analyzer)
            completed.append({
                "eq_id": eq_id,
                "dA": dA,
                "dAp": dAp,
                "run_dir": result.get("run_dir"),
                "summary": result.get("summary", ""),
            })
            print(f"[{i}/{len(experiment_ids)}] DONE  {eq_id}")
            print(f"    run_dir: {result.get('run_dir')}")
        except Exception as e:
            tb = traceback.format_exc()
            failed.append({
                "eq_id": eq_id,
                "dA": dA,
                "dAp": dAp,
                "error": str(e),
                "traceback": tb,
            })
            print(f"[{i}/{len(experiment_ids)}] FAIL  {eq_id}")
            print(f"    {e}")

    summary = {
        "solver": SOLVER,
        "local_edge_mode": LOCAL_EDGE_MODE,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed": completed,
        "failed": failed,
    }

    summary_path = out_root / "run_all_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("Batch complete")
    print(f"Completed: {len(completed)}")
    print(f"Failed:    {len(failed)}")
    print(f"Summary:   {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()