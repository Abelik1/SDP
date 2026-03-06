# Experiment Guide

## Core policy

Use `local_edge_mode=verified` for any figure or table you plan to put in the report. The `single` local solver is heuristic only.

## Recommended capstone figures

1. `lt_geometry_combined`
2. `lt_family_diagT_validation`
3. `lt_convertibility_graph`
4. `local_gp_ppt_relax`
5. `tfd_vs_dephased`
6. `separable_vs_entangled_lt`

## Important variables

- `local_edge_mode=verified|multistart|single`
- `n_random_starts=<int>`
- `project_to_lt=true|false`
- `sinkhorn_iters=<int>`
- qubit families: `label=XX` or `tx=1,ty=0,tz=1`

## Output contract

Every run writes a deterministic run folder under `results/` with configuration, summary, and artifacts.
