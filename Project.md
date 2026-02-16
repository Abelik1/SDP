# Capstone Project Summary  
## Hierarchy of Locally Thermal Bipartite States under Global and Local Gibbs-Preserving Operations

---

# 0. System Setup

We consider a symmetric bipartite quantum system:

\[
\mathcal{H}_A \otimes \mathcal{H}_B
\]

with identical Hamiltonians:

\[
H_A = H_B
\]

Thermal (Gibbs) state:

\[
\gamma = \frac{e^{-\beta H}}{Z}
\]

Global equilibrium reference state:

\[
\gamma_{AB} = \gamma \otimes \gamma
\]

Allowed operations:

- **Global Gibbs-preserving (GP):**
  \[
  \Phi(\gamma_{AB}) = \gamma_{AB}
  \]
  with \(\Phi\) completely positive and trace-preserving (CPTP).

- **Local Gibbs-preserving (Local GP):**
  \[
  \Phi = \Phi_A \otimes \Phi_B,
  \quad
  \Phi_X(\gamma) = \gamma
  \]

Monotone used:

\[
D(\rho \Vert \gamma \otimes \gamma)
\]

On the locally thermal (LT) set:

\[
D(\rho \Vert \gamma \otimes \gamma) = I(A:B)
\]

---

# Objective 1  
## Identify and Characterise the Set of Locally Thermal States

### Definition

A bipartite state is **locally thermal (LT)** if:

\[
\mathrm{Tr}_A(\rho) = \gamma,
\quad
\mathrm{Tr}_B(\rho) = \gamma
\]

### Structural Form

Every LT state can be written as:

\[
\rho = \gamma \otimes \gamma + C
\]

with:

\[
\mathrm{Tr}_A C = 0,
\quad
\mathrm{Tr}_B C = 0,
\quad
\rho \succeq 0
\]

Thus LT is:

- An affine subspace (marginal constraints),
- Intersected with the positive semidefinite (PSD) cone,
- Convex.

---

## Qubit Case Representation

For qubits:

\[
C = \frac{1}{4}\sum_{i,j\in\{x,y,z\}} T_{ij}\,\sigma_i\otimes\sigma_j
\]

So LT states are parameterised by a real \(3 \times 3\) correlation matrix \(T\).

Dimension count (qubits):

- Full state space: 15 real parameters  
- Marginal constraints fix 6 parameters  
- Remaining LT degrees of freedom: 9 (entries of \(T\))

---

## 2D and 3D Geometric Characterisation of LT

We restrict to the diagonal correlation slice:

\[
C = \frac{1}{4}\left(
t_x\,\sigma_x\otimes\sigma_x
+
t_y\,\sigma_y\otimes\sigma_y
+
t_z\,\sigma_z\otimes\sigma_z
\right)
\]

LT holds automatically (Pauli operators are traceless), so feasibility reduces to:

\[
\gamma \otimes \gamma + C \succeq 0
\]

---

### 3D Picture (Diagonal Slice)

The allowed coefficients:

\[
(t_x,t_y,t_z) \in \mathcal{R}_\beta \subset \mathbb{R}^3
\]

form a convex region.

- At \(\beta = 0\):  
  \[
  1 \pm t_x \pm t_y \pm t_z \ge 0
  \]
  giving a **tetrahedron** (Bell-diagonal polytope).

- At \(\beta > 0\):  
  The tetrahedron smoothly deforms into a convex temperature-dependent body (numerically mapped via ray eigenvalue bounds).

Thus:

> The set of locally thermal but globally athermal qubit states corresponds exactly to all non-zero points inside this convex 3D region.

---

### 2D Picture (Plane Slices)

Example: \(t_y = 0\)

\[
C = \frac{1}{4}(t_x XX + t_z ZZ)
\]

- At \(\beta=0\): region becomes a diamond  
  \[
  |t_x| + |t_z| \le 1
  \]

- At \(\beta>0\): diamond smoothly deforms.

Boundary computed using analytic ray eigenvalue bounds.

Convexity verified numerically (100% midpoint PSD test pass-rate).

---

## Ordered Convertibility Analysis (New)

States in the 2D/3D slice are now:

1. Sampled from interior of feasible region.
2. Sorted by decreasing mutual information \(I(A:B)\).
3. Adjacency matrices constructed in sorted order.

Result:

- **Global GP adjacency becomes upper-triangular**, confirming monotonicity of \(I\).
- **Local GP remains sparse**, revealing strong incomparability.

This confirms:

> Global GP hierarchy is scalar-monotone governed,  
> Local GP hierarchy is structurally constrained.

---

## Singular-Value Contraction Test

For diag-\(T\) slice:

\[
s(T) = \mathrm{sorted}(|t_x|,|t_y|,|t_z|)
\]

Observation:

- Along single rays → contraction holds.
- In full 2D slice → ~60% violations.
- In small 3D sample → inconclusive (few local edges).

Conclusion:

> Componentwise contraction of diag-\(T\) coordinates is not a full characterisation of Local GP in multi-dimensional slices.

---

# Objective 2  
## Construct the Hierarchy under Global Gibbs-Preserving Operations

Definition:

\[
\rho \succ \sigma
\quad \text{if} \quad
\exists \Phi \text{ (GP)} \text{ such that } \Phi(\rho) = \sigma
\]

Monotonicity:

\[
D(\rho \Vert \gamma \otimes \gamma)
\ge
D(\Phi(\rho) \Vert \gamma \otimes \gamma)
\]

On LT:

\[
I(A:B) \text{ is monotone}
\]

---

## Computational Results (Global GP)

- Pairwise feasibility via SDP.
- Ordered adjacency matrix becomes upper-triangular.
- Near-total preorder observed.
- Explicit GP channel constructed.
- Monotonicity numerically verified.

Conclusion:

> Global GP hierarchy is largely governed by scalar thermodynamic monotones.

---

# Objective 3  
## Hierarchy under Local Gibbs-Preserving Operations

Restrict:

\[
\Phi = \Phi_A \otimes \Phi_B
\]

Bloch representation:

\[
r \mapsto M r + t,
\quad
t = (I-M) r_\gamma
\]

Correlation transformation:

\[
T \mapsto T' = M_A T M_B^T
\]

---

## Computational Results (Local GP)

- Pairwise Local GP feasibility via SDP.
- Ordered adjacency shows extreme sparsity.
- ~96% incomparability persists.
- Scalar monotones insufficient.

---

## Closure Test

Random local GP channels constructed via Choi SDP:

- CP + TP + \(\Phi(\gamma)=\gamma\)
- Applied to random LT states.

Result:

- LT preserved with marginal error ~\(10^{-8}\).
- Confirms numerical closure.

---

## Extracted Explicit Channels

For feasible ray transformations:

- Full Choi matrices extracted.
- Intermediate state \(\omega\) verified.
- Reconstruction error ~\(10^{-8}\).

This provides explicit witness maps for local convertibility.

---

# Current Structural Understanding

1. LT is a convex affine 9D subset (qubits).
2. Diagonal slice gives clear 2D and 3D geometric pictures.
3. Global GP induces near-total preorder aligned with \(I\).
4. Local GP induces highly fragmented partial order.
5. Tensor-level constraints govern local convertibility.

---

# Next Steps

1. Increase 3D sampling for stronger statistics.
2. Analyse full singular value spectrum of general \(T\).
3. Investigate basis dependence of contraction.
4. Extend beyond diagonal slice.
5. Move toward analytic characterisation of full 9D LT region.

---

# Changelog Old to New

[UNCHANGED — Original changelog retained below exactly as written]

What I added (so you get “real results” today)
Objective 1 (characterise LT via γ⊗γ + C): 2D + 3D “C-space” boundary + convertibility structure

Two new experiments that directly characterise which coefficients are allowed in C (within a qubit-diagonal correlation slice) and then immediately compare Global GP vs Local GP structure on interior samples.

New experiments

lt_C_diagT_plane_characterise

Computes the boundary of feasible C in a chosen 2D plane of diag correlation coordinates (e.g. tx–tz with ty=0) using your ray PSD bounds method (whitened eigenvalue bounds).

Samples interior points, builds Global GP vs Local GP adjacency, and checks whether componentwise singular-value contraction predicts Local GP edges.

lt_C_diagT_3d_characterise

Computes an approximate 3D boundary surface (sampling directions on the sphere, taking p_max along each direction).

Samples interior points and does the same convertibility + singular-value heuristic check.

These are exactly “today-results” friendly: you’ll get figures + adjacency heatmaps + a quantitative “does s-value contraction fail?” rate.

Objective 2 (LT closure under local GPOs): random local GP channel generation + closure test

New experiment:

local_gp_closure_test

Generates LT states by random → LT projection (your closest_lt_state SDP).

Generates random local GP channels on A and A' by solving an SDP over Choi matrices:

CP + TP + Φ(γ)=γ (within eps_gibbs)

random linear objective to avoid always getting the same map

Applies them and checks LT marginal errors numerically.

This gives you a clean “closure is numerically verified” result with actual channels.

Supporting system additions:

find_random_local_gp_channel(which="A"/"Ap")

apply_local_channel_A, apply_local_channel_Ap

choi_diagnostics(...) (CP/TP/GP errors)

Objective 3 (explicit SDPs + extracting witnesses): extract actual local channels

New experiment:

extract_local_channels

Picks a simple LT ray (label=XX etc, p_src → p_tgt)

Runs your two-step local GP feasibility SDP with return_details=True

Saves:

J_A, J_Ap, intermediate ω

diagnostics (TP error, GP error, min eig of Choi)

recomputed step errors using numpy application of Choi

Output file: local_gp_channels.npy

This is presentation-grade: you can say “here is the explicit map my SDP found”.
