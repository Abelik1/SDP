# LT/GP SDP Capstone Toolkit

This refactor makes `ltgp/` the real source of truth while keeping `main.py`, `experiments.py`, and the markdown files at the repository root.

## Architecture

```text
.
├── main.py
├── experiments.py
├── README.md
├── Experiments.md
├── test_capstone.py
└── ltgp/
    ├── __init__.py
    ├── backend.py
    ├── experiments_ext.py
    ├── registry.py
    ├── run_store.py
    ├── system.py
    └── ui.py
```

## What changed

- `ltgp/system.py` now embeds the core SDP machinery and the analyzer/state-factory layer.
- `classical=True` is now implemented as *energy-basis diagonal*, not accidental computational-basis diagonal.
- report-facing local convertibility defaults to **verified** local GP, not the raw two-step heuristic.
- local channel application diagnostics no longer renormalize outputs.
- qutrit commuting LT sampling, PPT local outer relaxation, PPT/separability diagnostics, structured run folders, and report-friendly experiment outputs are integrated.

## Running

GUI:

```bash
python main.py
```

Tests:

```bash
pytest -q test_capstone.py
```

## Report-quality experiments

Recommended first runs:

```text
lt_geometry_combined
lt_family_diagT_validation   with local_edge_mode=verified
lt_convertibility_graph      with local_edge_mode=verified
local_gp_ppt_relax
tfd_vs_dephased
separable_vs_entangled_lt
```

Each run writes:

- `results/<timestamp>_<eq_id>_<hash>/config.json`
- `results/<...>/summary.json`
- `results/<...>/summary.txt`
- figures / CSV / NPY artifacts
