# Experiment Reference Guide (LT/GP SDP Toolkit)

This document describes each experiment available in the GUI (grouped A–E), what it does, which parameters matter, how to run it, and how to interpret outputs.

---

## Common concepts and outputs

### Core objects
- **System**: local Hamiltonians `H_A`, `H_A'`, inverse temperature `β`, local Gibbs states `γ_A`, `γ_A'`.
- **LT constraint**: `Tr_{A'}(ρ)=γ_A` and `Tr_A(ρ)=γ_A'`, plus `ρ ⪰ 0`, `Tr ρ=1`.
- **Global GP**: CPTP map `G` with `G(γ⊗γ)=γ⊗γ` and `G(τ)=τ'` (Choi SDP).
- **Local GP**: product map `G_A ⊗ G_A'` with each Gibbs-fixing. Nonconvex; implemented as heuristic + verification.

### Standard parameters (available via GUI controls)
- `dA`, `dAp`: local dimensions.
- `beta`: inverse temperature β.
- `solver`: solver hint (`SCS` safest).
- `eps_eq_global`: mapping tolerance for global GP SDPs (Frobenius norm residual).
- `eps_eq_local`: mapping tolerance for local GP routines.
- `eps_gibbs`: Gibbs-fixing tolerance.
- `tol`: solver tolerance (where exposed; typically set inside system defaults).
- `num_samples`: number of generated states/samples for sweeps.
- `seed`: RNG seed (deterministic if ≥ 0).
- `classical`: if an experiment supports “diagonal only” restriction.
- `symmetric`: attempt `H_A = H_A'` when dimensions match.
- `reset_system`: force rebuild of the system when running an experiment.
- `extra_vars`: experiment-specific knobs in `k=v` format (comma-separated).

### Common result quantities (monotones/diagnostics)
- `D(ρ || γ⊗γ)`: relative entropy to global Gibbs.
- `I(A:A')`: mutual information.
- `C = ρ - γ⊗γ`: correlation operator.
- “Operator-Schmidt / singular values of C”: via reshaping to `(dA^2, dAp^2)` and taking SVD.
- Local/Global feasibility status: `optimal` or solver-specific.
- Residuals:
  - `map_residual`: `||G(τ) - τ'||` (Frobenius), used for feasibility.
  - `gibbs_residual`: `||G(γ⊗γ) - γ⊗γ||` (Frobenius).

### Important policy for local edges (report-quality)
Local GP feasibility is **nonconvex**. The “single” two-step heuristic can output false positives.
For any adjacency/heatmap/report figure, use:
- `local_edge_mode=verified` (multistart + explicit verification) as the default.

If you must run the heuristic:
- interpret any “local edge” as *candidate only* unless verification passes.

### Outputs
- **New refactor experiments** write into:
  - `results/<timestamp>_<exp_id>_<hash>/config.json`
  - `results/<...>/summary.txt`
  - and additional artifacts (PNGs, NPYs).
- **Legacy experiments** mostly write to:
  - `png/*.png`

---

## Parameter cheat sheet (experiment-specific keys)

Use these keys in the GUI `Extra vars` line when needed:

- `local_edge_mode`: `verified` (recommended), `multistart`, or `single`
- `n_random_starts`: integer (default ~6), used in verified/multistart local GP
- `sinkhorn_iters`: integer (default ~300), used for qutrit commuting sampling
- `project_to_lt`: boolean (default true in PPT-relax demo), project random states onto LT before testing convertibility
- Any additional knobs in legacy experiments may exist; consult the experiment’s console log or source if you see “custom vars”.

---

# A) LT Geometry

## A1. `closest_lt_distance`
### What it does
Computes the closest LT state to a given input state ρ via trace-norm projection:

    minimize  1/2 ||ρ - σ||_1
    subject to  σ ⪰ 0, Tr σ = 1,
                Tr_{A'} σ = γ_A, Tr_A σ = γ_A'

Optionally supports a “classical/diagonal” restriction for certain dimensions.

### Required parameters
- `dA`, `dAp`, `beta` (defines γ)
- Solver settings: `solver`, `tol` (default in system)

### Typical usage
- Use this to quantify “distance to LT” for arbitrary states.
- Use σ (the closest LT point) as an LT proxy for convertibility experiments.

### Meaning of results
- Small distance ≈ already LT or nearly LT.
- Returned σ is a “best LT approximation” in trace distance (operationally meaningful).

---

## A2. `lt_region_geometry`
### What it does
Approximates the boundary of the LT set by sampling support-function extremals:

    maximize  Tr(K ρ)
    subject to ρ ∈ LT

for random Hermitian “probe” operators K. The set of maximizers samples exposed faces of the LT convex body.

### Required parameters
- `num_samples`: number of probes (larger → better boundary sampling)
- `seed`: reproducibility
- `dA`, `dAp`, `beta`, `solver`

### Typical usage
- Generate boundary point clouds for geometry figures.
- Compare boundary projections between different β or Hamiltonians.

### Meaning of results
- Extremal points represent “most aligned” LT states in the direction of K.
- A dense set of directions approximates the convex boundary.

---

## A3. `lt_interior_geometry`
### What it does
Samples interior points by:
- generating random density matrices, then
- projecting onto LT (trace-norm SDP) to obtain feasible LT points.

### Required parameters
- `num_samples`, `seed`
- solver parameters

### Typical usage
- Produce interior scatter plots of LT geometry.
- Overlay with boundary points (`lt_geometry_combined`).

### Meaning of results
- These are not uniform samples in LT, but good for qualitative “filled-in” pictures.
- Projection introduces bias: points concentrate near regions “closest” to random states.

---

## A4. `lt_geometry_combined`
### What it does
Produces the “final figure” geometry plot by overlaying:
- boundary samples (support function),
- interior samples (projection),
- optionally special lines (e.g., classical LT line for qubits).

### Required parameters
- Inherits parameters of `lt_region_geometry` and `lt_interior_geometry`:
  - `num_samples`, `seed`, `dA`, `dAp`, `beta`, `solver`

### Typical usage
- Use this figure in the report to show LT geometry.

### Meaning of results
- Boundary points indicate feasible extremes.
- Interior points show the convex bulk.
- Any highlighted structured families reveal how families carve the convex body.

---

# B) State Families

## B1. `tfd_vs_dephased`
### What it does
Constructs a TFD-like LT pure state (when spectra match appropriately) and compares it to its dephased (energy-diagonal) version:
- compute monotones (`D`, `I`, etc.)
- test convertibility properties (depending on legacy implementation)

### Required parameters
- `dA=dAp`
- `beta`
- solver settings

### Typical usage
- Demonstrates “coherent LT correlations” vs “classical/commuting LT correlations”.
- Useful for narrative about LT structure tiers (TFD-like vs dephased).

### Meaning of results
- Dephasing typically reduces certain correlation signatures and may affect convertibility.
- Compare `I` and `D` to quantify correlation vs athermality relative to γ⊗γ.

---

## B2. `mix_with_gamma`
### What it does
Studies a thermalization path:

    ρ_λ = (1-λ) ρ + λ (γ⊗γ)

Tracks monotones and/or feasibility as λ increases.

### Required parameters
- `beta`, `dA`, `dAp`
- legacy experiment may have its own λ-grid settings (check its log/plots)

### Typical usage
- Show monotones decay and approach to thermal equilibrium.
- Use to test monotonicity of `D(·||γ⊗γ)` under GP-like processes.

### Meaning of results
- As λ→1, state approaches γ⊗γ:
  - `D` should go to 0
  - correlations typically diminish

---

## B3. `lt_family_ray_validation` (qubits)
### What it does
Constructs an LT ray:

    ρ(p) = γ⊗γ + p C0

where `C0` is a Pauli-direction correlation operator. It:
- computes the analytic PSD interval for p (whitening eigenvalues)
- samples points along the ray
- tests local GP feasibility and monotones along p

### Required parameters
- `dA=dAp=2`
- `beta`
- ray choices are typically internal (direction i,j), unless exposed via legacy extra vars

### Useful extra vars
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Typical usage
- Demonstrate “resource concentration” or convertibility changes along structured direction.
- Identify boundary hits at p_min/p_max.

### Meaning of results
- p outside the computed interval ⇒ state becomes non-PSD.
- If local feasibility breaks near the boundary, interpret as “harder to realize locally”.
- Monotones vs p provide geometric/operational signatures.

---

## B4. `lt_family_diagT_validation` (qubits)
### What it does
A diagonal correlation-tensor family:

    ρ(p) = γ⊗γ + (p/4) (t_x XX + t_y YY + t_z ZZ)

or a related diagonal-T parameterization. Validates feasibility and convertibility/monotones.

### Required parameters
- `dA=dAp=2`
- `beta`
- grid/ray settings are legacy-defined

### Useful extra vars
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Meaning of results
- Diagonal-T slices are useful because positivity constraints can often be understood more directly.
- Local vs global convertibility differences typically show up clearly in these families.

---

## B5. `lt_C_diagT_plane_characterise` (qubits)
### What it does
Characterizes feasibility and/or convertibility on a 2D plane in diagonal-T parameter space.
Produces heatmaps or boundary plots.

### Required parameters
- `dA=dAp=2`
- `beta`
- legacy plane/grid density settings

### Critical extra vars (for correctness)
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Meaning of results
- Heatmap cells typically represent:
  - feasible LT states in that parameter region
  - convertibility edges counted/visualized across the set
- Any “unique mapping” claims must be verified edges only (see §Common concepts).

---

## B6. `lt_C_diagT_3d_characterise` (qubits)
### What it does
Extends diag-T characterization to 3D region sampling.
Often used to build a 3D feasible-region point cloud and test convertibility relationships on sampled points.

### Required parameters
- `dA=dAp=2`
- `beta`
- `num_samples` often controls how many points in the region are sampled

### Critical extra vars (for correctness)
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Meaning of results
- Reveals how feasible LT region behaves in full diagonal-T space.
- Shows where local convertibility differs from global convertibility.

---

## B7. `d3_commuting_sampling` (qutrits, NEW)
### What it does
Samples energy-diagonal (commuting) LT states for `dA=dAp=3` by sampling a nonnegative `3×3` matrix P with:

- row sums = γ_A in energy basis
- column sums = γ_A' in energy basis

This is the **transport polytope** (doubly-γ-stochastic constraints). Sampling is performed via **Sinkhorn scaling**.

From P, builds a diagonal density matrix in the energy product basis and maps back to the original basis.

### Required parameters
- `dA=dAp=3`
- `beta`
- `num_samples`
- `seed`

### Useful extra vars
- `sinkhorn_iters=<int>` (default ~300)

### Meaning of results
- These states are LT by construction and commuting with local Hamiltonians (in energy basis).
- Distributions of `I` and `D` quantify “classical” LT correlations in d=3.
- Use as a baseline class compared to noncommuting qutrit LT states (future work).

---

# C) Convertibility

## C1. `random_pair_gp_lgp`
### What it does
Samples a random pair `(τ, τ')` (often projected to LT depending on legacy design) and tests:
- global GP feasibility
- local GP feasibility (heuristic and/or verified)

### Required parameters
- `dA`, `dAp`, `beta`
- `eps_eq_global`, `eps_eq_local`, `eps_gibbs`

### Critical extra vars
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Meaning of results
- If global feasible and verified-local infeasible: evidence of “global advantage”.
- If verified-local feasible: indicates plausible product implementation.

---

## C2. `lt_convertibility_graph`
### What it does
Builds an ensemble of LT states (from a generator in the legacy pipeline) and computes pairwise convertibility:
- adjacency matrix for global GP edges
- adjacency matrix for local GP edges
- optionally derived plots/graphs

### Required parameters
- `num_samples`: number of states in the set (O(N^2) pair tests)
- solver settings and eps tolerances

### Critical extra vars
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Meaning of results
- Directed edge `i→j` means `ρ_i` can be converted to `ρ_j` by the chosen operation class.
- Compare:
  - global adjacency vs verified-local adjacency vs relaxed-local adjacency (if enabled)
- Use mutual information ordering as a visual heuristic; do not assume it fully orders convertibility.

---

## C3. `extract_global_channel`
### What it does
Solves for an explicit global GP channel (Choi matrix `J`) that approximately maps `τ→τ'` while fixing `γ⊗γ`.

Outputs:
- feasibility status
- residual errors
- extracted `J` (depending on legacy save path)

### Required parameters
- `dA`, `dAp`, `beta`
- `eps_eq_global`, `eps_gibbs`

### Meaning of results
- If feasible with small residuals: there exists a global GP channel meeting constraints.
- Use extracted channel to validate mapping and to build “constructive examples”.

---

## C4. `extract_local_channels`
### What it does
Runs the two-step local solver to find:
- `J_A` for subsystem A
- intermediate ω
- `J_A'` for subsystem A'

Optionally performs verification.

### Required parameters
- `eps_eq_local`, `eps_gibbs`
- solver settings

### Critical extra vars
- `local_edge_mode=verified`
- `n_random_starts=<int>`

### Meaning of results
- Treat unverified `J_A, J_A'` as candidates only.
- Verified success implies a genuine product map mapping `τ→τ'` to within tolerance.

---

## C5. `local_gp_ppt_relax` (NEW)
### What it does
Implements a **convex outer relaxation** for local GP feasibility:
- solve a global Choi SDP for mapping τ→τ' and fixing γ⊗γ
- add **PPT constraint** on the Choi matrix across the bipartition corresponding to Bob vs Alice degrees of freedom in the Choi space

Interpretation:
- If PPT-relaxation is infeasible ⇒ local product feasibility is ruled out (necessary condition).
- If PPT-relaxation is feasible ⇒ local feasibility is not guaranteed (outer relaxation).

### Required parameters
- `dA`, `dAp`, `beta`
- `eps_eq_local` (used as mapping tolerance for the relaxation)
- `eps_gibbs`

### Useful extra vars
- `project_to_lt=true|false` (default true in demo; makes τ,τ' LT-like)
- `seed=<int>`

### Meaning of results
- Use PPT-relaxation to cheaply prune impossible local edges before running expensive verified multistart.
- Use in plots as an “outer bound layer” (superset of local edges).

---

# D) Monotones & Invariants

## D1. `sanity_checks`
### What it does
Produces a compact diagnostic report/table for selected states and mappings:
- LT marginal errors
- PSD/trace checks
- GP mapping residuals
- monotone changes (ΔD, ΔI, etc.)

### Required parameters
- depends on which example states are used in legacy logic
- solver settings

### Meaning of results
- Use as a “methods validation” section in the report.
- Also useful to detect solver regressions after refactors.

---

# E) Utilities & Diagnostics

## E1. `local_gp_closure_test`
### What it does
Tests that LT is closed under local GP channels numerically by:
- generating LT states
- applying random local GP channels
- re-checking LT constraints on output

### Required parameters
- `num_samples`, `seed`
- `dA`, `dAp`, `beta`

### Meaning of results
- Should produce near-zero LT marginal errors (within numerical tolerance).
- If errors are systematic/non-small: likely a bug in channel application or normalization.

---

## E2. `custom`
### What it does
Passes JSON directly to the backend (advanced use). Intended for rapid prototyping without adding a full experiment spec.

### Required parameters
- `Mode=Custom` in GUI
- Valid JSON structure (backend-defined)

### Meaning of results
- Depends on what custom backend handler does.
- Use for one-off sweeps or debugging.

---

# Practical recipes

## Recipe 1: Report-quality convertibility heatmaps (qubits)
1. Set `dA=dAp=2`, choose `β`.
2. Run a diagT plane or 3D characterization experiment.
3. In extra vars set:
   - `local_edge_mode=verified, n_random_starts=6`
4. Export PNGs and record `results/<run>/config.json` in the report appendix.

Interpretation:
- Use verified local edges only.
- Optionally overlay PPT-relax edges as an outer bound (expect more edges than verified local).

## Recipe 2: Baseline qutrit commuting LT dataset
1. Set `dA=dAp=3`, choose `β`.
2. Run `d3_commuting_sampling` with:
   - `num_samples=500`, `seed=0`, `sinkhorn_iters=400`
3. Use saved `I_values.npy` and `D_values.npy` for plots in the report.

Interpretation:
- These are energy-diagonal LT states with “classical” correlations.
- Compare against any future noncommuting qutrit LT constructions.

## Recipe 3: Geometry figure (boundary + interior)
1. Run `lt_region_geometry` with `num_samples=200` (boundary).
2. Run `lt_interior_geometry` with `num_samples=500` (interior).
3. Run `lt_geometry_combined` to produce final overlay plot.

Interpretation:
- The combined plot is suitable for “LT set geometry” section of the report.

---

# Notes on numerical tolerances

- If you see solver instability:
  - increase `eps_eq_*` by 10× and re-check qualitative conclusions
  - prefer SCS for complex SDPs
  - use verified local edges with a slightly looser `eps_eq_local` if needed
- Always report:
  - solver used
  - tolerances
  - seed
  - number of samples
  - whether edges were verified