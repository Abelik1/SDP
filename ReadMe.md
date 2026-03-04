# LT/GP SDP Capstone Toolkit

Research tool for **locally-thermal (LT) bipartite states** and **Gibbs-preserving operations** (global GP and local GP). Built around **CVXPY** SDPs and a **PyQt5 GUI**.

This repo started as a single-script prototype; it is now being refactored into a coherent package while preserving legacy experiments.

---

## 0) System + Definitions (capstone-level)

### Physical setup

We study a bipartite system \(A \otimes A'\) with local Hamiltonians

\[
H_A,\; H_{A'}
\]

and inverse temperature \(\beta\). The local Gibbs (thermal) states are

\[
\gamma_X = \frac{e^{-\beta H_X}}{Z_X}.
\]

The reference global Gibbs state is the product

\[
\gamma_{AA'} = \gamma_A \otimes \gamma_{A'}.
\]

### Locally-thermal (LT) states

A bipartite state \(\rho_{AA'}\) is **locally-thermal** iff

\[
\operatorname{Tr}_{A'}(\rho) = \gamma_A, \qquad
\operatorname{Tr}_{A}(\rho) = \gamma_{A'}.
\]

This defines a convex spectrahedron: PSD + affine marginal constraints.

We call states **LNT** (locally non-thermal) when one or both marginals differ from the corresponding Gibbs state.

### Gibbs-preserving operations (GP)

A CPTP map \(\mathcal{G}\) is **global Gibbs-preserving** if

\[
\mathcal{G}(\gamma_{AA'}) = \gamma_{AA'}.
\]

**Global GP convertibility** (decision problem):

Given \(\tau,\tau'\), decide whether there exists a CPTP map \(\mathcal{G}\) such that

\[
\mathcal{G}(\gamma_{AA'})=\gamma_{AA'}, \qquad
\mathcal{G}(\tau)=\tau'.
\]

This is solved via a Choi SDP.

### Local GP (LGP)

Local GP convertibility asks for a product channel

\[
\mathcal{G} = \mathcal{G}_A \otimes \mathcal{G}_{A'}
\]

with each factor CPTP and Gibbs-fixing locally:

\[
\mathcal{G}_A(\gamma_A)=\gamma_A, \qquad
\mathcal{G}_{A'}(\gamma_{A'})=\gamma_{A'}.
\]

Exact feasibility is nonconvex due to the product constraint.

**Important methodological note:** this repo contains a **heuristic** local solver (two-step with an intermediate \(\omega\)) which can produce **false positives** unless followed by explicit verification (see §6).

---

## 1) What’s implemented

### 1.1 LT generation + parameterizations

- **General LT set**
  - Enforced by PSD + partial-trace constraints in SDPs.

- **d = 2 (qubits)**
  - Pauli/traceless basis correlation representation

        ρ = γ⊗γ + Σ_ij C_ij B_i ⊗ B_j

    with \(C \in \mathbb{R}^{3\times3}\).

  - Ray families

        ρ(p) = γ⊗γ + p C₀

    with analytic PSD interval via whitening eigenvalues.

  - Diagonal-T families used heavily in legacy experiments.

- **d = 3 (qutrits)**

  - Gell-Mann/traceless basis correlation representation

        C ∈ ℝ^(8×8)

  - **Commuting / energy-diagonal LT subclass**

    Sampled as a transport-polytope distribution with row/column sums matching \(\gamma\) (Sinkhorn scaling).

---

### 1.2 Geometry probing

**Closest-LT projection**

\[
\min_{\sigma \in LT} \frac12 \|\rho - \sigma\|_1
\]

implemented via trace-norm SDP.

**Support function / boundary sampling**

\[
\max_{\rho \in LT} \mathrm{Tr}(K\rho)
\]

used to approximate the boundary and build LT-region figures.

---

### 1.3 Convertibility tests

- **Global GP convertibility**

  Choi SDP with

  - CPTP constraints
  - Gibbs-fixing constraint
  - mapping constraint.

- **Local GP (heuristic)**

  Two-step solver with intermediate state \(\omega\).

  **Must be verified** (see §6).

- **Local GP outer relaxation (convex)**

  Uses **PPT constraint on a joint Choi** as a necessary condition.

  Interpretation:

  - infeasible PPT ⇒ infeasible local GP  
  - feasible PPT ⇏ feasible local GP

---

### 1.4 Monotones / diagnostics

Computed for analysis and plots:

- Relative entropy to the product Gibbs

  \[
  D(\rho\Vert\gamma\otimes\gamma)
  \]

- Mutual information \(I(A:A')\)

- Operator-Schmidt singular values of

  \[
  C = \rho - \gamma\otimes\gamma
  \]

  where \(C\) is reshaped into \((d_A^2, d_{A'}^2)\).

- Various correlation metrics used in legacy experiments.

---

### 1.5 Reproducibility + run artifacts

New runs create folders under

    ./results/

containing

    config.json
    summary.txt
    <other artifacts>

Legacy experiments still write plots to

    ./png/

The wrapper records which PNGs were produced during each run.

---

## 2) New refactor structure

Current transition state (legacy preserved; new package added):

    .
    ├── main.py                # entry point (new GUI)
    ├── utils.py               # common utilities + system/analyzer builder
    ├── sdp_system.py          # core SDPs + channel utilities
    ├── sdp_analysis.py        # analyzer + state factories
    ├── experiments.py         # legacy experiment dispatcher
    ├── ltgp/                  # NEW package (refactor layer)
    │   ├── __init__.py
    │   ├── system.py
    │   ├── registry.py
    │   ├── ui.py
    │   ├── backend.py
    │   ├── experiments_ext.py
    │   └── favorites.json
    ├── audit.py
    ├── audit_diagT3D.py
    ├── results/
    └── png/

### Meaning of `ltgp`

`ltgp` stands for **Locally-Thermal + Gibbs-Preserving**.  
It is a namespace package containing the refactored code.

---

## 3) Installation

Recommended environment

- Python **3.10+**
- Packages
  - cvxpy
  - numpy
  - scipy
  - matplotlib
  - PyQt5

Solvers:

- **SCS** (default; robust for complex SDPs)
- **MOSEK** (optional but faster)
- CVXOPT may work depending on SDP form.

Example installation:

    python -m venv venv

Windows:

    venv\Scripts\activate

Mac / Linux:

    source venv/bin/activate

Install dependencies:

    pip install numpy scipy matplotlib pyqt5 cvxpy scs

---

## 4) Running the GUI

Run:

    python main.py

GUI features:

- Experiment groups A–E (no giant scrolling list)
- Search/filter by id, title, or tags
- Favorites pinning
- Control panel for
  - dimension \(d\)
  - β
  - solver
  - tolerances
  - sampling parameters
- Results panel
  - summary text
  - run directory
- Export last run summary as JSON.

---

## 5) Experiments (taxonomy)

### A) LT Geometry

- lt_region_geometry
- lt_interior_geometry
- lt_geometry_combined
- closest_lt_distance

### B) State Families

- tfd_vs_dephased
- mix_with_gamma
- lt_family_ray_validation
- lt_family_diagT_validation
- lt_C_diagT_plane_characterise
- lt_C_diagT_3d_characterise
- d3_commuting_sampling

### C) Convertibility

- random_pair_gp_lgp
- lt_convertibility_graph
- extract_global_channel
- extract_local_channels
- local_gp_ppt_relax

### D) Monotones & Invariants

- sanity_checks

### E) Utilities & Diagnostics

- local_gp_closure_test
- custom

---

## 6) Critical correctness: local edges, audits, and verification

### Why the audit mattered

The two-step local solver can return

    single_ok = True

while the resulting product channel **fails to map**

\[
\tau \rightarrow \tau'
\]

This is a **methodological issue**, not a physics result.

Your audit scripts detect these failures:

- audit_diagT3D.py  
  finds “lost edges” where heuristic accepted but verification rejects.

- audit.py  
  performs broader checks (recovered/lost edges, transitivity violations).

---

### What the refactor does

The verification machinery already exists in the core:

    verify_local_gp_details
    _no_norm

The refactor ensures that plots and adjacency graphs **should default to verified edges**.

If legacy adjacency loops are still used:

- ensure adjacency builders call a **verified local-edge routine**
- or implement

    local_edge_mode = "verified"

---

### Recommended edge policy for report figures

1. **Global GP edges**

2. **Verified local edges**

   - multistart solver
   - explicit verification

3. **PPT-relax edges** (optional)

   - used as an outer bound.

---

## 7) Output files and where to find results

### New experiments

Located in

    ltgp/experiments_ext.py

Output to

    results/<timestamp>_<eq_id>_<hash>/

with

    config.json
    summary.txt

and any generated artifacts.

---

### Legacy experiments

Still write plots to

    png/*.png

The backend wrapper logs which PNG files were generated.

---

## 8) Migration notes

### Keep (required)

- main.py
- utils.py
- sdp_system.py
- sdp_analysis.py
- experiments.py
- ltgp/

### Safe to remove

- sdp_gui.py

### Recommended to keep

- audit.py
- audit_diagT3D.py

These act as regression tests.

---

## 9) Developer notes / extending the tool

### Adding a new experiment

1. Register metadata in

        ltgp/registry.py

2. Implement experiment in

        ltgp/experiments_ext.py

3. Add dispatch case in

        ltgp/backend.py

---

### Long sweeps / checkpointing

Current state:

- run folders are created
- configs are saved

Planned:

- resumable sweeps
- per-pair or per-grid checkpointing
- safe interrupt handling.

---

### Unit tests (planned)

Categories:

- LT marginal constraints
- PSD / trace normalization
- Basis orthogonality (Pauli, Gell-Mann)
- CPTP constraints on Choi matrices
- Gibbs-fixing conditions
- Reproducibility via seeded RNG.

---

## 10) Known limitations

- Exact **local GP feasibility is nonconvex**.

- The current “local solver” is **heuristic** and must be verified.

- **PPT relaxation** is only an **outer relaxation**  
  (necessary but not sufficient).

- Current \(d=3\) implementation focuses on

  - commuting subclass sampling
  - Gell-Mann correlation representation
  - LT-set and GP convertibility SDPs.

A general **dimension-agnostic basis module** could be added later.

---

## 11) Citation / theory references

This codebase supports the capstone narrative on

- geometry of locally thermal sets
- monotonicity of relative entropy to Gibbs
- differences between global GP and local GP convertibility
- structured LT families (TFD-like, dephased/commuting, diagonal slices).

The **capstone report** should contain the literature citations; the README documents **software behavior and implementation details**.

---

### Possible future improvements

- Integrate audit scripts directly into the GUI as a **Diagnostics** experiment group.
- Add a `tests/` folder with **pytest** so `audit_diagT3D` becomes an automated regression test.