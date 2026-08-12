# Capstone Experiments Overview

## System
**Symmetric bipartite system:** A ⊗ A′  
**Identical Hamiltonians:** H_A = H_A′  
**Inverse temperature:** β  
**Thermal (Gibbs) state on X:**  
γ_X = exp(-β H_X) / Z_X

**Global reference Gibbs state:**  
γ_AA′ = γ_A ⊗ γ_A′

---

## Quantities used throughout

### (Athermality) Relative entropy to the product Gibbs state
D(ρ || γ_A ⊗ γ_A′)

Meaning: how “non-thermal” the joint state ρ is, when the baseline is the uncorrelated thermal state.

### Mutual information (total correlation)
I(A:A′)_ρ = S(ρ_A) + S(ρ_A′) - S(ρ_AA′)

Meaning: how much correlation (classical + quantum) exists between A and A′.

### Locally Thermal (LT) definition
ρ is **LT** iff:
Tr_{A′}(ρ) = γ_A  
Tr_{A}(ρ)  = γ_A′

Meaning: each subsystem looks exactly thermal by itself, but the joint state can still contain correlations.

### Key identity on the LT set
If ρ is LT, then:
D(ρ || γ⊗γ) = I(A:A′)_ρ

Meaning: inside the LT set, “athermality relative to γ⊗γ” is *exactly the same number* as “total correlation”.

Important: this does **not** mean the LT set is 1D in state space (it isn’t).  
It means LT collapses to a 1D curve **only when you plot using the two numbers (D, I)**.

---

------------------------------------------------------------
EXPERIMENT 1
TFD → Dephased TFD (Global GP vs Local GP)
------------------------------------------------------------

Initial state: Thermo-Field Double (TFD)

TFD ket:
|TFD> = Σ_i sqrt(g_i) |E_i>_A ⊗ |E_i>_A′

TFD density matrix:
τ_TFD = |TFD><TFD|

Dephased TFD (energy-basis dephasing):
τ_deph = Σ_i g_i |E_i E_i><E_i E_i|

Test:
Is there a Gibbs-preserving map (GP map) such that τ_TFD → τ_deph?

We test both:
- Global GP (one joint operation on AA′ that preserves γ_AA′)
- Local GP (separate operations on A and A′ that each preserve γ_A and γ_A′)

Results:
- Global GP: feasible
- Local GP: infeasible

Meaning:
Global GP can do a *joint energy measurement* on AA′ and then re-prepare the dephased state (a “measure-and-prepare” channel).
Local GP cannot coordinate a joint dephasing because each side acts independently.

Conclusion:
Local Gibbs-preserving operations produce a strict partial order inside LT.

------------------------------------------------------------
EXPERIMENT 2
Random τ → τ′ (Global GP vs Local GP)
------------------------------------------------------------

Generate two independent random bipartite states (Ginibre sampling).

Test:
Does there exist a GP map (global or local) such that Φ(τ) ≈ τ′ ?

Results:
- Global GP sometimes feasible.
- Local GP almost never feasible.

Meaning:
Random states don’t “line up” in the specific way required by Gibbs-preserving constraints.
This serves as a baseline / negative control: feasibility is nontrivial and structure-dependent.

------------------------------------------------------------
EXPERIMENT 3
Thermalisation path: (1-λ)ρ + λ(γ⊗γ)
------------------------------------------------------------

Define a straight-line mixing path:
ρ(λ) = (1-λ) ρ + λ (γ⊗γ),  with λ in [0,1].

Track:
- D(ρ(λ) || γ⊗γ)
- I(A:A′)
- Distance to LT

Observations:
- D decreases with λ.
- I decreases with λ.
- If the starting ρ is LT, distance to LT stays 0 for all λ (LT is convex).

Meaning:
Mixing an LT state with γ⊗γ is a valid “correlation erasure” trajectory that stays inside LT.

------------------------------------------------------------
EXPERIMENT 4
Distance to LT (trace norm SDP)
------------------------------------------------------------

For a random state ρ, solve:
min over σ in LT  of  (1/2) * ||ρ - σ||_1

Also compute distance to “classical LT” (a stricter subset).

Meaning of the optimisation:
- We search for the closest locally-thermal state σ to a given ρ.
- This is a principled way to measure “how far” ρ is from satisfying the LT constraints.

Observations:
- Random states are typically not LT.
- Distance to classical LT is usually larger than distance to LT.

Conclusion:
classical LT is strictly contained in LT.
Quantum-only structure (beyond classical correlation) is a big part of the gap.

------------------------------------------------------------
EXPERIMENT 5
LT Geometry — Extremal Boundary (support function SDP)
------------------------------------------------------------

Goal:
Find “boundary” LT states in a systematic way (not by guessing families).

Method:
Pick a Hermitian matrix K (a random direction in state space), then solve the optimisation:

Maximise:  Tr(K ρ)
Subject to:
- ρ >= 0   (valid state)
- Tr(ρ) = 1
- Tr_A(ρ)  = γ
- Tr_A′(ρ) = γ

This is a standard “support function” idea:
- For a convex set, maximising a linear functional Tr(Kρ) returns a boundary (often extremal) point.

What math is happening:
- You are solving a semidefinite program (SDP).
- The LT constraints are linear equations on the entries of ρ.
- The constraint ρ >= 0 forces physical validity.
- The optimiser pushes ρ as far as possible in the direction K while staying inside LT.

What you plot:
You map each boundary state to a pair of numbers:
- D(ρ || γ⊗γ)
- I(A:A′)

Result / meaning:
You observe all points lie on the line I = D.

This does NOT mean the boundary is trivial.
It means (D, I) is too low-dimensional to show LT geometry:
- Many different LT states (with different coherence / structure) can have the same (D, I).
- The true “shape” is in the full matrix space (high-dimensional), not in that 2-number projection.

Why we do this:
This is your cleanest “geometry” experiment because it does not rely on special families.
It samples extremal/boundary LT states in many directions.

------------------------------------------------------------
EXPERIMENT 6
LT Interior — Random → LT projection
------------------------------------------------------------

Goal:
Sample typical (interior) LT states and compare them to boundary states.

Method:
Take a random state ρ_rand and compute its closest LT state via:

σ* = argmin over σ in LT of (1/2) * ||ρ_rand - σ||_1

This is a projection (in trace distance) onto the LT set.

What math is happening:
- Another SDP: same LT constraints + PSD + trace-1.
- Objective forces σ to be the closest LT state to a generic random ρ_rand.
- This tends to produce interior points (not extreme boundary points), because “closest feasible point” usually lands inside unless forced to the edge.

What you compute/plot:
For each σ* you compute:
- D(σ* || γ⊗γ)
- I(A:A′)  (should equal D because σ* is LT)
- distance to classical LT

Meaning of results (typical pattern):
- Most projected LT states have low correlation (low D/I).
- Many are close to classical LT, meaning “mostly classical correlation structure”.
- Strongly “quantum-structured” LT states (far from classical LT) tend to be less common and often appear nearer the boundary.

Why we do this:
Boundary sampling (Exp 5) tells you what is possible.
Projection sampling (Exp 6) tells you what is typical.

------------------------------------------------------------
EXPERIMENT 7
LT Geometry Combined (Boundary + Interior + classical LT)
------------------------------------------------------------

Goal:
Compare “possible extreme LT states” vs “typical LT states” in one picture.

Method:
Overlay:
- boundary/extremal LT samples from Exp 5
- interior LT samples from Exp 6
- classical LT reference set/curve (if plotted)

Key clarification:
In the (D, I) plane, LT always lies on I = D, so the overlay looks “1D”.

Meaning:
This experiment is mainly a diagnostic:
- It confirms the identity I = D numerically.
- It shows that (D, I) coordinates do not resolve LT structure.

What would show real geometry:
Add extra axes/plots that do not collapse on LT, e.g.
- distance to classical LT
- a coherence measure (e.g., how large off-diagonal blocks are in the energy basis)
- spectral features of ρ
- correlation tensor structure (in qubits)

------------------------------------------------------------
EXPERIMENT 8
LT Convertibility Graph (Global GP vs Local GP)
------------------------------------------------------------

Goal:
Turn “convertibility” into a graph problem and compare global vs local constraints.

Method:
Take a finite set of LT states {ρ_i}. For each ordered pair (i -> j) solve an SDP:

Does there exist a GP map Φ such that:
Φ(ρ_i) = ρ_j
and Φ preserves Gibbs (global or local version)?

Output:
Adjacency matrix A where:
A[i,j] = 1 if conversion is feasible, else 0.

Results (typical):
- Global GP: graph is close to a total preorder (many edges), largely governed by D (and on LT, D = I).
- Local GP: graph is sparse, many pairs incomparable (~high incomparability fraction).

What math is happening:
- “Existence of a GP map” becomes feasibility of linear constraints on the Choi matrix (the matrix representation of the channel) plus PSD constraints.
- Global GP allows joint correlations to be manipulated.
- Local GP restricts the channel to product/local structure, which removes many degrees of freedom.

Meaning:
Scalar monotones like D (or I) are not enough to predict local convertibility.
Two LT states can have the same D/I but still be incomparable under local GP.

This is your central “structure result”:
Locality creates genuine operational constraints not visible in basic thermodynamic numbers.

------------------------------------------------------------
EXPERIMENT 9
Extract explicit global GP channel (Choi matrix)
------------------------------------------------------------

Extract the Choi matrix J_Φ for a known feasible mapping (TFD -> dephased).

Verify:
- J_Φ >= 0
- trace-preserving constraints
- Gibbs preservation: Φ(γ⊗γ) = γ⊗γ
- correct action on the input: Φ(τ_TFD) ≈ τ_deph

Meaning:
You move from “existence” to an explicit channel description.

------------------------------------------------------------
EXPERIMENT 10
Numerical robustness checks
------------------------------------------------------------

Verify errors are within tolerance:
- Gibbs preservation residuals
- mapping residuals
- monotone non-increase (where expected)

Meaning:
Confirms observed structure is not a solver artefact.

------------------------------------------------------------
OVERALL STRUCTURE OF RESULTS
------------------------------------------------------------

1. LT set is defined by linear marginal constraints + PSD.
2. On LT, D(ρ||γ⊗γ) equals mutual information I(A:A′).
3. Global GP convertibility is mostly governed by this scalar monotone.
4. Local GP convertibility is far more restricted and fragmented.
5. You can extract an explicit global GP channel for key transformations.

Core thesis:
Local Gibbs-preserving operations impose strictly stronger constraints than global monotones,
creating true operational incomparability even within the locally thermal set.
## NEW EXPERIMENTS (added)

------------------------------------------------------------
EXPERIMENT 11
LT Family — Ray family  ρ(p) = γ⊗γ + p C0  (Local GP monotones)  [qubits only]
------------------------------------------------------------

Goal:
Build a clean 1-parameter family of locally-thermal (LT) states and test what “should decrease” along feasible local GP conversions.

Initial state / family definition:
We define a fixed correlation-direction matrix C0 (chosen by a label like XX, YY, ZZ, XY, XZ, YZ) and scan

ρ(p) = γ⊗γ + p * C0

For the “ray family (XX)” case:
C0 = (1/4) * (X ⊗ X)

Why this automatically stays LT:
LT means both marginals are thermal:
Tr_A'(ρ) = γ   and   Tr_A(ρ) = γ

This holds for the whole ray because C0 is chosen so that:
Tr_A'(C0) = 0   and   Tr_A(C0) = 0
So adding p*C0 never changes either marginal.

What maths is happening (positivity / PSD interval):
Not every p gives a valid density matrix. We must enforce:
ρ(p) >= 0   (PSD = physically valid)

The code computes an analytic allowed interval:
p in [p_min, p_max]

Conceptually:
- You “whiten” the direction by the Gibbs state:
  C_tilde = (γ^(-1/2) ⊗ γ^(-1/2)) * C0 * (γ^(-1/2) ⊗ γ^(-1/2))
- Then validity reduces to:
  I + p*C_tilde >= 0
- So p_max is determined by the most negative eigenvalue of C_tilde, and p_min by the most positive one.

You then shrink slightly away from the boundary (p_shrink) because near the edge numerical solvers get unstable.

What you compute along the ray (what the plots mean):
For each p you compute correlation diagnostics:

1) Correlation size (two norms on C = ρ - γ⊗γ = p*C0)
- 0.5*||C||_1  (trace norm; “sum of singular values / eigenvalue magnitudes”)
- ||C||_F      (Frobenius norm; “sqrt(sum of squares of entries)”)
Expected result:
Both scale linearly with |p| because C = p*C0.

2) Mutual information I(A:A')
I(A:A') = S(ρ_A) + S(ρ_A') - S(ρ)
Meaning: “total correlation”
Expected result:
I starts at 0 when p=0 and grows nonlinearly (often ~ p^2 for small p).

3) Singular values of correlation tensor T (qubits only)
T is the 3x3 matrix of Pauli-Pauli correlations (up to conventions).
You plot its singular values s1 >= s2 >= s3.
Meaning:
- These are “axis-independent correlation strengths”.
Expected result for a pure XX direction:
Only one singular value is nonzero (rank-1 correlation), and it grows linearly with p.

What conversion test is done:
You take multiple points {ρ(p_i)} and test feasibility of conversions between them under LOCAL GP:
- local GP means operations act locally and preserve γ on each subsystem.
Feasibility is checked via an SDP per ordered pair.

Why we do this:
This gives you a controlled testbed where:
- You know exactly what family you’re exploring
- You can compare “simple correlation measures” vs actual SDP convertibility

Meaning of results:
If local GP can convert from p_high -> p_low but not the reverse, that supports the idea that
“correlation strength along this direction behaves like a resource under local GP”.

Files produced:
- png/ray_<label>_C_norms_vs_p.png
- png/ray_<label>_I_vs_p.png
- png/ray_<label>_T_svals_vs_p.png


------------------------------------------------------------
EXPERIMENT 12
LT Family — Diagonal-T ray (tx,ty,tz)  (Local GP monotones)  [qubits only]
------------------------------------------------------------

Goal:
Same idea as Experiment 11, but using a cleaner and more interpretable 3-parameter direction family.

Initial state / family definition:
Choose a direction using diagonal Pauli correlations:

C0 = (1/4) * ( tx*(X⊗X) + ty*(Y⊗Y) + tz*(Z⊗Z) )

Then scan:
ρ(p) = γ⊗γ + p * C0

Important special case:
diagT t0=(1,0,0) is the same as the XX ray direction.

Why it stays LT:
Same reason as Exp 11: partial traces of these Pauli⊗Pauli terms vanish, so marginals stay γ.

What maths is happening:
Same PSD interval computation:
ρ(p) >= 0 gives a valid p-range, then shrink to avoid numerical edge issues.

What the plots mean (same as Exp 11):
- correlation norms vs p: should be linear in p
- mutual information vs p: grows from 0, typically nonlinear
- singular values of T vs p:
  For diagonal-T directions, T is diagonal (in an appropriate convention) so singular values are basically |tx|,|ty|,|tz| scaled by p (and sorted).
  If only one of tx,ty,tz is nonzero, you again see only one nonzero singular value.

Why we do this:
Diagonal-T is a standard “slice” of two-qubit correlation space.
It’s easier to interpret, and is a good bridge between:
- simple analytic families
- and your convertibility/monotone tests

Files produced:
- png/diagT_<tx>_<ty>_<tz>_C_norms_vs_p.png
- png/diagT_<tx>_<ty>_<tz>_I_vs_p.png
- png/diagT_<tx>_<ty>_<tz>_T_svals_vs_p.png


------------------------------------------------------------
EXPERIMENT 13
LT Structured-Family Hierarchy (custom JSON)  (Adjacency + predictor validation)
------------------------------------------------------------

Goal:
Turn “convertibility along a structured LT family” into a graph, and test whether a cheap predictor matches the SDP results.

What family you can choose:
You specify a family via JSON in the “Custom” experiment mode, e.g.

Ray example:
{"experiment":"lt_structured_family_hierarchy",
 "family":"ray", "label":"XX",
 "num_p":21, "pair_mode":"decreasing",
 "include_negative":false, "p_shrink":0.98,
 "mono_tol":1e-8}

Diagonal-T example:
{"experiment":"lt_structured_family_hierarchy",
 "family":"diagT", "tx":1, "ty":0, "tz":1,
 "num_p":21, "pair_mode":"all",
 "include_negative":true, "p_shrink":0.98,
 "mono_tol":1e-8}

What maths is happening:
Step 1: Construct the LT family ρ(p) = γ⊗γ + p*C0
Step 2: Compute analytic PSD interval for p and sample points inside it
Step 3: For each ordered pair (i -> j), solve an SDP:
“Does there exist a GP map taking ρ_i to ρ_j?”
and do this for BOTH:
- global GP
- local GP

This produces adjacency matrices:
A_global[i,j] = 1 if feasible under global GP else 0
A_local[i,j]  = 1 if feasible under local GP else 0

Monotone checks (why we log violations):
For every feasible edge i -> j, the code checks that certain quantities did not increase
(within tolerance), and logs violations if they occur.
This is a consistency check: if an alleged monotone increases, either:
- the monotone isn’t actually a monotone for that operation set, or
- numerical tolerance / solver issues need attention.

Local-feasibility predictor (what the TP/FP/TN/FN means):
In the qubit case you compute the singular values of the correlation tensor T:
s(ρ) = (s1,s2,s3)

A simple “predictor” is:
Predict i -> j is feasible (under local GP) if s(ρ_j) <= s(ρ_i) componentwise
(i.e., each singular value does not increase).

Then compare predictor vs SDP ground truth:
- TP (true positive): predictor says feasible, SDP says feasible
- FP (false positive): predictor says feasible, SDP says infeasible
- TN (true negative): predictor says infeasible, SDP says infeasible
- FN (false negative): predictor says infeasible, SDP says feasible

Meaning:
This tells you whether singular-value contraction is a good cheap proxy for local GP convertibility
for that structured family.

How to interpret “specificity = nan”:
Specificity = TN / (TN + FP).
If you only test “decreasing” pairs, you might have no negatives at all (TN+FP=0),
so specificity is undefined and prints as nan. That is expected, not a bug.

Why we do this:
This experiment directly supports your capstone “hierarchy” goal:
- it produces a convertibility graph (global vs local)
- it tests whether a simple analytic/cheap criterion can predict local convertibility
- it generates clean numerical artifacts: adjacency matrices + violation logs

Files produced:
- png/<tag>_T_svals_vs_p.png
- png/<tag>_adj_global.npy
- png/<tag>_adj_local.npy
- png/<tag>_monotone_violations.json


{"experiment":"lt_structured_family_hierarchy","family":"ray","label":"XX","num_p":21,"pair_mode":"decreasing","mono_tol":1e-8,"p_shrink":0.98}