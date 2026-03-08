from __future__ import annotations
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

MAX_WORKERS = 3



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
    "closest_lt_distance": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 1,
        "classical": False,
    },
    "lt_region_geometry": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 2,
        "num_samples": 150,
        "classical": False,
    },
    "lt_interior_geometry": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 3,
        "num_samples": 200,
    },
    "lt_geometry_combined": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 4,
        "num_samples": 200,
        "classical": False,
    },
    "tfd_vs_dephased": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 5,
    },
    "mix_with_gamma": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 6,
        "num_samples": 31,
    },
    "lt_family_ray_validation": {
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
    "d3_commuting_sampling": {
        "dA": 3, "dAp": 3,
        "seed": SEED0 + 9,
        "num_samples": 250,
        "sinkhorn_iters": 300,
    },
    "random_pair_gp_lgp": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 10,
        "project_to_lt": True,
    },
    "lt_convertibility_graph": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 11,
        "num_samples": 8,
        "ensemble": "mixed",
        "include_negative": True,
    },
    "local_gp_ppt_relax": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 12,
        "project_to_lt": True,
    },
    "extract_global_channel": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 13,
        "project_to_lt": True,
    },
    "sanity_checks": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 14,
    },
    "separable_vs_entangled_lt": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 15,
        "num_samples": 80,
    },
    "local_gp_closure_test": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 16,
        "num_samples": 30,
    },
    "verified_local_edge_audit": {
        "dA": 2, "dAp": 2,
        "seed": SEED0 + 17,
        "num_samples": 8,
    },
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


def run_one_experiment(eq_id: str, cfg: dict) -> dict:
    dA = int(cfg["dA"])
    dAp = int(cfg["dAp"])

    base_vars = build_base_vars(dA, dAp)
    base_vars.update(cfg)

    system, analyzer = build_system_and_analyzer(
        dA=dA,
        dAp=dAp,
        beta=BETA,
        solver=SOLVER,
        symmetric=True,
        eps_eq_global=1e-6,
        eps_eq_local=1e-6,
        eps_gibbs=1e-8,
    )

    config = {
        "selected_equation_id": eq_id,
        "selected_equation_name": eq_id,
        "variables_str": vars_dict_to_string(base_vars),
    }

    try:
        result = backend_run(config, system, analyzer)
        return {
            "ok": True,
            "eq_id": eq_id,
            "dA": dA,
            "dAp": dAp,
            "run_dir": result.get("run_dir"),
            "summary": result.get("summary", ""),
        }
    except Exception as e:
        return {
            "ok": False,
            "eq_id": eq_id,
            "dA": dA,
            "dAp": dAp,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    out_root = Path("batch_logs")
    out_root.mkdir(exist_ok=True)

    experiment_ids = flatten_catalog_ids()

    jobs = []
    completed = []
    failed = []

    print("=" * 80)
    print("Running all registered experiments")
    print(f"Solver: {SOLVER}")
    print(f"Local mode: {LOCAL_EDGE_MODE}")
    print(f"Workers: {MAX_WORKERS}")
    print("=" * 80)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {}

        for eq_id in experiment_ids:
            if eq_id not in EXPERIMENT_CONFIGS:
                failed.append({
                    "eq_id": eq_id,
                    "error": "Missing config in EXPERIMENT_CONFIGS"
                })
                continue

            cfg = dict(EXPERIMENT_CONFIGS[eq_id])
            fut = ex.submit(run_one_experiment, eq_id, cfg)
            future_map[fut] = eq_id

        for i, fut in enumerate(as_completed(future_map), start=1):
            eq_id = future_map[fut]
            try:
                out = fut.result()
            except Exception as e:
                failed.append({
                    "eq_id": eq_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"[{i}/{len(future_map)}] FAIL  {eq_id}")
                print(f"    {e}")
                continue

            if out["ok"]:
                completed.append({
                    "eq_id": out["eq_id"],
                    "dA": out["dA"],
                    "dAp": out["dAp"],
                    "run_dir": out["run_dir"],
                    "summary": out["summary"],
                })
                print(f"[{i}/{len(future_map)}] DONE  {eq_id}")
                print(f"    run_dir: {out['run_dir']}")
            else:
                failed.append(out)
                print(f"[{i}/{len(future_map)}] FAIL  {eq_id}")
                print(f"    {out['error']}")

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